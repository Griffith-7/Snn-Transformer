
import pytest
import torch

from astrohebbian import (
    AstrocyteHebbianClassifier,
    CausalAstrocyteLanguageModel,
    ExactLinearSpike,
    ExactSpike,
)
from astrohebbian.model import spike_fn


def test_spike_fn_modes():
    x = torch.randn(4, 8, requires_grad=True)

    y_surr = spike_fn(x, mode="surrogate")
    assert y_surr.shape == (4, 8)
    loss_surr = y_surr.sum()
    loss_surr.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    x_exact = torch.randn(4, 8, requires_grad=True)
    y_exact = spike_fn(x_exact, mode="exact")
    assert y_exact.shape == (4, 8)
    loss_exact = y_exact.sum()
    loss_exact.backward()
    assert x_exact.grad is not None
    assert not torch.isnan(x_exact.grad).any()


def test_exact_ift_gradient_matches_formula():
    tau, theta = 2.0, 0.5
    x = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], requires_grad=True)
    y = ExactSpike.apply(x, tau, theta)
    y.sum().backward()

    firing = (x.detach() > theta).float()
    denom = (x.detach() * (x.detach() - theta)).abs().clamp_min(ExactSpike.IFT_GRAD_FLOOR)
    expected_grad = tau * theta / denom * firing
    assert torch.allclose(x.grad, expected_grad, atol=1e-6)


def test_exact_ift_gradient_below_threshold_is_zero():
    tau, theta = 1.0, 0.5
    x = torch.tensor([-2.0, -0.1, 0.0, 0.49], requires_grad=True)
    y = ExactSpike.apply(x, tau, theta)
    y.sum().backward()
    assert (x.grad == 0).all()


def test_exact_ift_gradient_above_threshold_is_positive():
    tau, theta = 1.0, 0.5
    x = torch.tensor([0.501, 1.0, 2.0, 10.0], requires_grad=True)
    y = ExactSpike.apply(x, tau, theta)
    y.sum().backward()
    assert (x.grad > 0).all()


def test_exact_ift_gradient_decreases_far_from_threshold():
    tau, theta = 1.0, 0.5
    x = torch.tensor([1.0, 2.0, 5.0, 10.0], requires_grad=True)
    y = ExactSpike.apply(x, tau, theta)
    y.sum().backward()
    assert x.grad[0] > x.grad[1] > x.grad[2] > x.grad[3]


def test_exact_ift_gradient_finite_near_threshold():
    tau, theta = 1.0, 0.5
    x = torch.tensor([0.501, 0.52, 0.6, 0.9], requires_grad=True)
    y = ExactSpike.apply(x, tau, theta)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# ExactLinearSpike tests (fused linear + IFT spike)
# ---------------------------------------------------------------------------

class TestExactLinearSpike:
    def test_output_shape(self):
        layer = ExactLinearSpike(32, 64)
        x = torch.randn(4, 16, 32)
        out = layer(x)
        assert out.shape == (4, 16, 64)

    def test_output_is_binary(self):
        layer = ExactLinearSpike(32, 64)
        x = torch.randn(4, 16, 32)
        out = layer(x)
        assert set(out.unique().tolist()) <= {0.0, 1.0}

    def test_backward_produces_finite_grads(self):
        layer = ExactLinearSpike(32, 64)
        x = torch.randn(4, 16, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.bias is not None
        assert layer.bias.grad is not None
        assert torch.isfinite(layer.bias.grad).all()

    def test_gradient_matches_ift_formula(self):
        tau, theta = 1.0, 0.5
        layer = ExactLinearSpike(3, 2, tau=tau, theta=theta, bias=False)
        x = torch.randn(1, 3, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        pre = x.detach() @ layer.weight.detach().t()
        firing = (pre > theta).float()
        denom = (pre * (pre - theta)).abs().clamp_min(ExactSpike.IFT_GRAD_FLOOR)
        adjoint = tau * theta / denom * firing

        expected_w_grad = adjoint.t() @ x.detach()
        expected_x_grad = adjoint @ layer.weight.detach()
        assert torch.allclose(layer.weight.grad, expected_w_grad, atol=1e-5)
        assert torch.allclose(x.grad, expected_x_grad, atol=1e-5)

    def test_no_bias(self):
        layer = ExactLinearSpike(16, 32, bias=False)
        assert layer.bias is None
        x = torch.randn(2, 8, 16)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None

    def test_matches_nn_linear_output_shape(self):
        exact = ExactLinearSpike(32, 64)
        dense = torch.nn.Linear(32, 64)
        x = torch.randn(2, 8, 32)
        assert exact(x).shape == dense(x).shape

    def test_training_step_reduces_loss(self):
        torch.manual_seed(42)
        layer = ExactLinearSpike(16, 4)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
        x = torch.randn(8, 3, 16)
        target = torch.randint(0, 4, (8, 3))

        losses = []
        for _ in range(10):
            opt.zero_grad()
            out = layer(x)
            loss = torch.nn.functional.cross_entropy(
                out.reshape(-1, 4), target.reshape(-1)
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gradient_mode", ["surrogate", "exact"])
def test_classifier_gradient_modes(gradient_mode):
    model = AstrocyteHebbianClassifier(
        input_dim=1,
        d_model=32,
        seq_len=64,
        num_classes=10,
        num_layers=1,
        num_heads=2,
        gradient_mode=gradient_mode,
    )
    x = torch.randn(2, 64, 1, requires_grad=True)
    logits = model(x)
    assert logits.shape == (2, 10)

    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.parametrize("gradient_mode", ["surrogate", "exact"])
def test_causal_lm_gradient_modes(gradient_mode):
    model = CausalAstrocyteLanguageModel(
        vocab_size=256,
        d_model=32,
        seq_len=64,
        num_heads=2,
        num_layers=1,
        gradient_mode=gradient_mode,
    )
    token_ids = torch.randint(0, 256, (2, 32))
    logits = model(token_ids)
    assert logits.shape == (2, 32, 256)

    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_exact_mode_trains():
    torch.manual_seed(0)
    model = AstrocyteHebbianClassifier(
        input_dim=1, d_model=32, seq_len=32, num_classes=3,
        num_layers=1, num_heads=2, gradient_mode="exact",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 32, 1)
    y = torch.randint(0, 3, (4,))

    initial_logits = model(x).detach().clone()
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    final_logits = model(x).detach()
    assert not torch.allclose(initial_logits, final_logits, atol=1e-6)


def test_exact_mode_all_params_have_grad():
    model = AstrocyteHebbianClassifier(
        input_dim=1, d_model=32, seq_len=32, num_classes=3,
        num_layers=1, num_heads=2, gradient_mode="exact",
    )
    x = torch.randn(2, 32, 1)
    logits = model(x)
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
