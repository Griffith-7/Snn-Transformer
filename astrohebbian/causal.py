"""Causal, state-updating Astrocyte-Hebbian attention."""

import torch
import torch.nn as nn

from .model import spike_fn


class CausalAstrocyteHebbianAttention(nn.Module):
    """Causal linear attention with a recurrent decayed Hebbian state.

    Unlike the full-sequence attention module, each output only reads the
    prefix ending at its current token. The state update is explicit so the
    module can later be adapted to streaming execution.
    """

    def __init__(self, d_model=128, num_heads=4, v_levels=1):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if v_levels != 1:
            raise ValueError("the causal prototype currently supports v_levels=1 only")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.decay_logit = nn.Parameter(torch.empty(d_model).uniform_(5.0, 8.0))
        self.eps = 1e-6

    def forward(self, x, implementation="recurrent"):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape (batch, sequence, {self.d_model})")
        if x.shape[1] == 0:
            raise ValueError("sequence length must be positive")
        if implementation not in ("recurrent", "parallel"):
            raise ValueError("implementation must be 'recurrent' or 'parallel'")
        batch_size, sequence_length, _ = x.shape
        query = spike_fn(self.q_proj(x)).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = spike_fn(self.k_proj(x)).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = spike_fn((torch.tanh(self.v_proj(x)) + 1.0) / 2.0 - 0.5).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        decay = torch.sigmoid(self.decay_logit).view(
            1, self.num_heads, self.head_dim
        ).to(x.dtype)
        if implementation == "parallel":
            positions = torch.arange(
                sequence_length, device=x.device, dtype=x.dtype
            ).view(1, 1, sequence_length, 1)
            inverse_decay = decay.reciprocal()
            powers = inverse_decay.unsqueeze(-2).pow(positions)
            pair = key.unsqueeze(-1) * value.unsqueeze(-2)
            trace = (
                torch.cumsum(pair * powers.unsqueeze(-1), dim=2)
                * decay.unsqueeze(-2).pow(positions).unsqueeze(-1)
            )
            mass = (
                torch.cumsum(key * powers.squeeze(-1), dim=2)
                * decay.unsqueeze(-2).pow(positions)
            )
            numerator = torch.matmul(query.unsqueeze(-2), trace).squeeze(-2)
            denominator = (
                (query * mass).sum(dim=-1, keepdim=True).clamp_min(self.eps)
            )
            output = numerator / denominator
            output = output.transpose(1, 2).contiguous().view(
                batch_size, sequence_length, self.d_model
            )
            return self.o_proj(output)

        trace = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=x.device, dtype=x.dtype,
        )
        mass = torch.zeros(
            batch_size, self.num_heads, self.head_dim,
            device=x.device, dtype=x.dtype,
        )
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
        return self.o_proj(output)