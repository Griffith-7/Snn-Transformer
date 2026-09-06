import pytest
import torch
from astrohebbian import (
    AstrocyteHebbianClassifier,
    CausalAstrocyteLanguageModel,
)
from astrohebbian.model import spike_fn


def test_spike_fn_modes():
    x = torch.randn(4, 8, requires_grad=True)
    
    # Surrogate mode
    y_surr = spike_fn(x, mode="surrogate")
    assert y_surr.shape == (4, 8)
    loss_surr = y_surr.sum()
    loss_surr.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    
    # Exact IFT mode
    x_exact = torch.randn(4, 8, requires_grad=True)
    y_exact = spike_fn(x_exact, mode="exact")
    assert y_exact.shape == (4, 8)
    loss_exact = y_exact.sum()
    loss_exact.backward()
    assert x_exact.grad is not None
    assert not torch.isnan(x_exact.grad).any()


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
    
    # Verify parameters received valid non-NaN gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
