"""Astrocyte-Hebbian spiking linear Transformer components with Exact-SNN gradients.

This package integrates the exact Implicit Function Theorem (IFT) gradient from
Exact-SNN (Griffith-7/Exact-Snn).  ``ExactLinearSpike`` fuses a linear
projection and binary threshold into a single ``torch.autograd.Function`` whose
backward pass differentiates the spike-time map of an exponential
integrate-and-fire membrane:

    ``u(t) = (Wx + b) * (1 - exp(-t / tau))``

Setting ``u(t*) = theta`` and applying the IFT gives

    ``dt*/d(Wx+b) = tau / (Wx+b - theta)``

so the gradient flows through the membrane dynamics rather than through a
surrogate approximation.
"""

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Surrogate gradient (original fast-sigmoid, kept for gradient_mode="surrogate")
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Exact IFT spike (standalone, for places that still use spike_fn)
# ---------------------------------------------------------------------------

class ExactSpike(torch.autograd.Function):
    """Binary threshold with an exact IFT gradient derived from membrane dynamics.

    Forward: ``s = (x > theta)``.

    Backward (IFT): models an exponential integrate-and-fire membrane
    ``u(t) = x * (1 - exp(-t / tau))`` and differentiates its spike-time map.
    The spike time ``t*`` satisfies ``u(t*) = theta``, giving

        ``dt*/dx = -tau * theta / (x * (x - theta))``   for x > theta,

    i.e. only neurons that fire (``x > theta``) receive a gradient, and the
    magnitude decays as the membrane moves away from threshold.  A floor on the
    denominator keeps the gradient finite for neurons just above threshold.
    """

    IFT_GRAD_FLOOR = 1e-2

    @staticmethod
    def forward(ctx, input, tau=1.0, theta=0.5):
        ctx.save_for_backward(input)
        ctx.tau = float(tau)
        ctx.theta = theta.detach() if torch.is_tensor(theta) else float(theta)
        ctx.theta_needs_grad = bool(torch.is_tensor(theta) and theta.requires_grad)
        return (input > ctx.theta).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tau = ctx.tau
        theta = ctx.theta
        firing = (input > theta).to(input.dtype)
        denom = (input * (input - theta)).abs().clamp_min(ExactSpike.IFT_GRAD_FLOOR)
        flux = tau * theta / denom
        grad_input = grad_output * flux * firing
        if ctx.theta_needs_grad:
            # dt*/d(theta) = tau / (pre - theta); the output step decreases when
            # the threshold rises, so the threshold adjoint is negated.
            mem_denom = (input - theta).abs().clamp_min(ExactSpike.IFT_GRAD_FLOOR)
            theta_flux = grad_output * tau / mem_denom * firing
            theta_shape = theta.shape
            lead = theta_flux.ndim - len(theta_shape)
            reduce_dims = tuple(range(lead)) + tuple(
                d for d, s in enumerate(theta_shape, start=lead)
                if s == 1 and theta_flux.shape[d] != 1
            )
            grad_theta = -theta_flux.sum(dim=reduce_dims).reshape(theta_shape)
        else:
            grad_theta = None
        return grad_input, None, grad_theta


ExactIFTSpike = ExactSpike


def spike_fn(x, alpha=10.0, mode="surrogate", tau=1.0, theta=0.0):
    if mode in ("exact", "reciprocal"):
        return ExactSpike.apply(x, tau, theta)
    return SurrogateHeaviside.apply(x, alpha)


# ---------------------------------------------------------------------------
# Fused Exact-SNN linear + spike layer
# ---------------------------------------------------------------------------

class _ExactLinearSpikeFn(torch.autograd.Function):
    """Fused linear projection + binary spike with exact IFT backward.

    Forward:
        ``pre = Wx + b``, ``s = step(pre - theta)``

    Backward (IFT):
        Differentiates the spike-time map of an exponential IF membrane.
        ``dt*/d(Wx+b) = -tau * theta / (pre * (pre - theta))`` for neurons
        that fire.  Silent neurons receive zero gradient (consistent with
        Exact-SNN).
    """

    @staticmethod
    def forward(ctx, input, weight, bias, tau, theta):
        pre = torch.nn.functional.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, bias, pre)
        ctx.tau = float(tau)
        ctx.theta = float(theta)
        return (pre > float(theta)).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, pre = ctx.saved_tensors
        tau = ctx.tau
        theta = ctx.theta

        firing = (pre > theta).to(pre.dtype)
        denom = (pre * (pre - theta)).abs().clamp_min(ExactSpike.IFT_GRAD_FLOOR)
        adjoint = grad_output * tau * theta / denom * firing

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.functional.linear(adjoint, weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = adjoint.transpose(-2, -1) @ input
            if grad_weight.ndim > 2:
                grad_weight = grad_weight.sum(dim=0)
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = adjoint.sum(dim=list(range(adjoint.ndim - 1)))

        return grad_input, grad_weight, grad_bias, None, None


class ExactLinearSpike(nn.Module):
    """Linear layer followed by a binary spike, trained with exact IFT gradients.

    Drop-in replacement for ``nn.Linear`` + ``spike_fn``.  Fuses both into a
    single ``torch.autograd.Function`` so that the backward pass differentiates
    the spike-time map of the underlying membrane model rather than using a
    surrogate gradient.

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        tau: membrane time constant (controls gradient magnitude).
        theta: firing threshold.
        bias: if ``True``, adds a bias term.
    """

    def __init__(self, in_features, out_features, tau=1.0, theta=0.5, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = float(tau)
        self.theta = float(theta)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _ExactLinearSpikeFn.apply(x, self.weight, self.bias, self.tau, self.theta)


# ---------------------------------------------------------------------------
# Astrocyte-Hebbian Attention
# ---------------------------------------------------------------------------

class AstrocyteHebbianAttention(nn.Module):
    """Multi-head spiking linear attention with learnable channel decay.

    In exact mode, Q/K/V projections use ``ExactLinearSpike`` (fused linear +
    spike with IFT backward).  In surrogate mode, falls back to ``nn.Linear``
    + ``spike_fn``.
    """

    def __init__(
        self,
        d_model=128,
        num_heads=4,
        v_levels=1,
        alpha=10.0,
        learnable_thresholds=False,
        gradient_mode="exact",
        tau=1.0,
        theta=0.5,
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
        self.tau = float(tau)
        self.theta = float(theta)
        self.gradient_mode = gradient_mode

        if gradient_mode == "exact":
            self.q_proj = ExactLinearSpike(d_model, d_model, tau=tau, theta=theta)
            self.k_proj = ExactLinearSpike(d_model, d_model, tau=tau, theta=theta)
            self.v_proj = nn.Linear(d_model, d_model)
            self.o_proj = nn.Linear(d_model, d_model)
        else:
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

        if self.gradient_mode == "exact":
            query = self.q_proj(x)
            key = self.k_proj(x)
            value_unit = (torch.tanh(self.v_proj(x)) + 1.0) / 2.0
            if self.value_levels == 1:
                value = ExactSpike.apply(value_unit, self.tau, self.thresholds[0])
            else:
                value_spikes = ExactSpike.apply(
                    value_unit.unsqueeze(-1), self.tau,
                    self.thresholds.view(1, 1, 1, -1),
                )
                value = value_spikes.mean(dim=-1)
        else:
            query = spike_fn(
                self.q_proj(x), self.alpha, mode=self.gradient_mode,
            )
            key = spike_fn(
                self.k_proj(x), self.alpha, mode=self.gradient_mode,
            )
            value_unit = (torch.tanh(self.v_proj(x)) + 1.0) / 2.0
            if self.value_levels == 1:
                value = spike_fn(
                    value_unit - self.thresholds[0], self.alpha,
                    mode=self.gradient_mode,
                )
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


# ---------------------------------------------------------------------------
# Spiking FFN
# ---------------------------------------------------------------------------

class SpikingFFN(nn.Module):
    """Feed-forward network with a binary hidden activation.

    In exact mode, the hidden layer uses ``ExactLinearSpike``.
    """

    def __init__(self, d_model=128, expansion=4, gradient_mode="exact", tau=1.0, theta=0.5):
        super().__init__()
        if d_model <= 0 or expansion <= 0:
            raise ValueError("d_model and expansion must be positive")
        if gradient_mode not in ("surrogate", "exact", "reciprocal"):
            raise ValueError("gradient_mode must be 'exact', 'surrogate', or 'reciprocal'")
        self.gradient_mode = gradient_mode
        if gradient_mode == "exact":
            self.fc1 = ExactLinearSpike(d_model, d_model * expansion, tau=tau, theta=theta)
        else:
            self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x):
        if self.gradient_mode == "exact":
            return self.fc2(self.fc1(x))
        return self.fc2(spike_fn(self.fc1(x), mode=self.gradient_mode))


# ---------------------------------------------------------------------------
# Block & Classifier
# ---------------------------------------------------------------------------

class AstrocyteHebbianBlock(nn.Module):
    """Pre-norm Transformer block using Astrocyte-Hebbian attention."""

    def __init__(
        self, d_model=128, num_heads=4, expansion=4, v_levels=1,
        gradient_mode="exact", tau=1.0, theta=0.5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = AstrocyteHebbianAttention(
            d_model, num_heads, v_levels, gradient_mode=gradient_mode,
            tau=tau, theta=theta,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SpikingFFN(d_model, expansion, gradient_mode=gradient_mode, tau=tau, theta=theta)

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
        tau=1.0,
        theta=0.5,
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
                    d_model, num_heads, v_levels=v_levels,
                    gradient_mode=gradient_mode, tau=tau, theta=theta,
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
