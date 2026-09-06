"""Astrocyte-Hebbian spiking linear Transformer components.

This package is intentionally standalone. It depends on PyTorch only and does
not import or depend on Exact-SNN.
"""

import torch
import torch.nn as nn


class SurrogateHeaviside(torch.autograd.Function):
    """Binary threshold in the forward pass with a fast-sigmoid backward pass."""

    @staticmethod
    def forward(ctx, input, alpha=10.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        sigmoid = torch.sigmoid(alpha * input)
        return grad_output * sigmoid * (1 - sigmoid) * alpha, None


class ReciprocalSurrogateSpike(torch.autograd.Function):
    """Binary threshold in the forward pass with a bounded reciprocal surrogate gradient.

    Backward pass evaluates g'(x) = 1 / (|x| + eps) for |x| <= 1.0, modeling inverse potential
    sensitivity around the firing threshold without requiring full TTFS membrane trajectory tracking.
    """

    @staticmethod
    def forward(ctx, input, eps=1e-3):
        ctx.save_for_backward(input)
        ctx.eps = eps
        return (input > 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        eps = ctx.eps
        reciprocal_grad = 1.0 / (torch.abs(input) + eps)
        active_mask = (torch.abs(input) <= 1.0).to(input.dtype)
        return grad_output * reciprocal_grad * active_mask, None


# Backward-compatible alias
ExactIFTSpike = ReciprocalSurrogateSpike


def spike_fn(x, alpha=10.0, mode="surrogate"):
    if mode in ("exact", "reciprocal"):
        return ReciprocalSurrogateSpike.apply(x)
    return SurrogateHeaviside.apply(x, alpha)


class AstrocyteHebbianAttention(nn.Module):
    """Multi-head spiking linear attention with learnable channel decay."""

    def __init__(
        self,
        d_model=128,
        num_heads=4,
        v_levels=1,
        alpha=10.0,
        learnable_thresholds=False,
        gradient_mode="surrogate",
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if v_levels <= 0:
            raise ValueError("v_levels must be positive")
        if gradient_mode not in ("surrogate", "exact"):
            raise ValueError("gradient_mode must be 'surrogate' or 'exact'")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.value_levels = int(v_levels)
        self.alpha = float(alpha)
        self.gradient_mode = gradient_mode
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        init_thresholds = (
            torch.arange(1, self.value_levels + 1, dtype=torch.float32)
            / (self.value_levels + 1)
        )
        if learnable_thresholds:
            self.thresholds = nn.Parameter(init_thresholds)
        else:
            self.register_buffer("thresholds", init_thresholds)

        self.decay_logit = nn.Parameter(torch.empty(d_model).uniform_(5.0, 8.0))
        self.eps = 1e-6

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model)")
        batch_size, sequence_length, d_model = x.shape
        if sequence_length == 0:
            raise ValueError("sequence length must be positive")
        if d_model != self.d_model:
            raise ValueError(f"last dimension must be {self.d_model}")

        query = spike_fn(self.q_proj(x), self.alpha, mode=self.gradient_mode)
        key = spike_fn(self.k_proj(x), self.alpha, mode=self.gradient_mode)
        value_unit = (torch.tanh(self.v_proj(x)) + 1.0) / 2.0
        if self.value_levels == 1:
            value = spike_fn(value_unit - self.thresholds[0], self.alpha, mode=self.gradient_mode)
        else:
            value_spikes = spike_fn(
                value_unit.unsqueeze(-1) - self.thresholds.view(1, 1, 1, -1),
                self.alpha,
                mode=self.gradient_mode,
            )
            value = value_spikes.mean(dim=-1)

        positions = torch.arange(
            sequence_length - 1,
            -1,
            -1,
            device=x.device,
            dtype=x.dtype,
        )
        decay = torch.sigmoid(self.decay_logit).to(x.dtype)
        weights = decay.unsqueeze(0).pow(positions.unsqueeze(1))
        weighted_key = key * weights.unsqueeze(0)

        query = query.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        weighted_key = weighted_key.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key_value_trace = torch.matmul(weighted_key.transpose(-2, -1), value)
        key_mass = weighted_key.sum(dim=-2, keepdim=True).transpose(-2, -1)
        numerator = torch.matmul(query, key_value_trace)
        denominator = torch.matmul(query, key_mass) + self.eps
        output = numerator / denominator

        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)
        return self.o_proj(output)


class SpikingFFN(nn.Module):
    """Feed-forward network with a binary hidden activation."""

    def __init__(self, d_model=128, expansion=4, gradient_mode="exact"):
        super().__init__()
        if d_model <= 0 or expansion <= 0:
            raise ValueError("d_model and expansion must be positive")
        if gradient_mode not in ("surrogate", "exact", "reciprocal"):
            raise ValueError("gradient_mode must be 'exact', 'surrogate', or 'reciprocal'")
        self.gradient_mode = gradient_mode
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x):
        return self.fc2(spike_fn(self.fc1(x), mode=self.gradient_mode))


class AstrocyteHebbianBlock(nn.Module):
    """Pre-norm Transformer block using Astrocyte-Hebbian attention."""

    def __init__(self, d_model=128, num_heads=4, expansion=4, v_levels=1, gradient_mode="exact"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = AstrocyteHebbianAttention(
            d_model, num_heads, v_levels, gradient_mode=gradient_mode
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SpikingFFN(d_model, expansion, gradient_mode=gradient_mode)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class AstrocyteHebbianClassifier(nn.Module):
    """Sequence classifier built from standalone Astrocyte-Hebbian blocks."""

    def __init__(
        self,
        input_dim=1,
        d_model=128,
        seq_len=784,
        num_classes=10,
        num_layers=1,
        num_heads=4,
        v_levels=1,
        gradient_mode="exact",
    ):
        super().__init__()
        if seq_len <= 0 or num_classes <= 0 or num_layers <= 0:
            raise ValueError("seq_len, num_classes, and num_layers must be positive")
        if input_dim <= 0 or d_model <= 0 or num_heads <= 0:
            raise ValueError("input_dim, d_model, and num_heads must be positive")
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                AstrocyteHebbianBlock(
                    d_model, num_heads, v_levels=v_levels, gradient_mode=gradient_mode
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.embedding.in_features:
            raise ValueError("x must have shape (batch, sequence, input_dim)")
        if x.shape[1] > self.pos_encoder.shape[1]:
            raise ValueError("sequence length exceeds configured seq_len")
        x = self.embedding(x) + self.pos_encoder[:, : x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        return self.classifier(x[:, -1, :])

