"""Causal, state-updating Astrocyte-Hebbian attention."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .model import spike_fn


class CausalAstrocyteHebbianAttention(nn.Module):
    """Causal linear attention with a recurrent decayed Hebbian state.

    Supports both sequence processing ('recurrent' and 'parallel' implementations)
    and streaming token-by-token processing via explicit state passing.
    """

    def __init__(self, d_model=128, num_heads=4, v_levels=1, alpha=10.0, gradient_mode="exact"):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if v_levels != 1:
            raise ValueError("the causal prototype currently supports v_levels=1 only")
        if gradient_mode not in ("surrogate", "exact", "reciprocal"):
            raise ValueError("gradient_mode must be 'exact', 'surrogate', or 'reciprocal'")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.alpha = float(alpha)
        self.gradient_mode = gradient_mode
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.decay_logit = nn.Parameter(torch.empty(d_model).uniform_(5.0, 8.0))
        self.eps = 1e-6

    def forward(
        self,
        x: torch.Tensor,
        implementation: str = "recurrent",
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape (batch, sequence, {self.d_model})")
        if x.shape[1] == 0:
            raise ValueError("sequence length must be positive")
        if implementation not in ("recurrent", "parallel"):
            raise ValueError("implementation must be 'recurrent' or 'parallel'")

        batch_size, sequence_length, _ = x.shape
        query = spike_fn(self.q_proj(x), self.alpha, mode=self.gradient_mode).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = spike_fn(self.k_proj(x), self.alpha, mode=self.gradient_mode).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = spike_fn((torch.tanh(self.v_proj(x)) + 1.0) / 2.0 - 0.5, self.alpha, mode=self.gradient_mode).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        decay = torch.sigmoid(self.decay_logit).view(
            1, self.num_heads, self.head_dim
        ).to(x.dtype)

        if implementation == "parallel" and state is None:
            log_decay = torch.log(decay.clamp(min=1e-8, max=1.0 - 1e-8))
            max_neg_exponent = ((sequence_length - 1) * (-log_decay)).max().item()
            if max_neg_exponent <= 80.0:
                positions = torch.arange(
                    sequence_length, device=x.device, dtype=x.dtype
                ).view(1, 1, sequence_length, 1)

                log_decay_unrolled = log_decay.unsqueeze(-2) * positions
                powers_pos = torch.exp(log_decay_unrolled)
                powers_neg = torch.exp(-log_decay_unrolled)

                pair = key.unsqueeze(-1) * value.unsqueeze(-2)
                trace = (
                    torch.cumsum(pair * powers_neg.unsqueeze(-1), dim=2)
                    * powers_pos.unsqueeze(-1)
                )
                mass = (
                    torch.cumsum(key * powers_neg.squeeze(-1), dim=2)
                    * powers_pos
                )
                numerator = torch.matmul(query.unsqueeze(-2), trace).squeeze(-2)
                denominator = (
                    (query * mass).sum(dim=-1, keepdim=True).clamp_min(self.eps)
                )
                output = numerator / denominator
                output = output.transpose(1, 2).contiguous().view(
                    batch_size, sequence_length, self.d_model
                )
                output = self.o_proj(output)
                if return_state:
                    final_trace = trace[:, :, -1]
                    final_mass = mass[:, :, -1]
                    return output, (final_trace, final_mass)
                return output

        if state is None:
            trace = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype,
            )
            mass = torch.zeros(
                batch_size, self.num_heads, self.head_dim,
                device=x.device, dtype=x.dtype,
            )
        else:
            trace, mass = state

        outputs = []
        for position in range(sequence_length):
            key_position = key[:, :, position].unsqueeze(-1)
            value_position = value[:, :, position].unsqueeze(-2)
            trace = trace * decay.unsqueeze(-1) + key_position * value_position
            mass = mass * decay + key[:, :, position]
            numerator = torch.matmul(query[:, :, position].unsqueeze(-2), trace).squeeze(-2)
            denominator = (
                query[:, :, position] * mass
            ).sum(dim=-1, keepdim=True).clamp_min(self.eps)
            outputs.append(numerator / denominator)

        output = torch.stack(outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        output = self.o_proj(output)

        if return_state or state is not None:
            return output, (trace, mass)
        return output