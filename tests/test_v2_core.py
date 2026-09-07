"""
Tests for AstroHebbian v2 core enhancements.
"""
import torch
import torch.nn as nn

from astrohebbian import (
    AstrocyteHebbianAttention,
    CausalAstrocyteHebbianAttention,
    CausalAstrocyteLanguageModel,
    spike_fn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestV2NumericalStability:
    def test_long_sequence_parallel_causal_no_nans(self):
        """Parallel causal attention must remain finite for long sequences (N=1024, 2048)."""
        attn = CausalAstrocyteHebbianAttention(d_model=64, num_heads=2).to(DEVICE)
        x = torch.randn(1, 1024, 64, device=DEVICE)
        out_parallel = attn(x, implementation="parallel")
        assert torch.isfinite(out_parallel).all(), "Parallel causal output contains non-finite values for N=1024"

        x_long = torch.randn(1, 2048, 64, device=DEVICE)
        out_long = attn(x_long, implementation="parallel")
        assert torch.isfinite(out_long).all(), "Parallel causal output contains non-finite values for N=2048"


class TestV2StreamingStatefulAPI:
    def test_token_by_token_streaming_matches_full_recurrent(self):
        """Streaming state pass-through step-by-step must match full sequence forward pass."""
        torch.manual_seed(42)
        attn = CausalAstrocyteHebbianAttention(d_model=32, num_heads=2).to(DEVICE).eval()
        x = torch.randn(1, 16, 32, device=DEVICE)

        full_out = attn(x, implementation="recurrent")

        streaming_outputs = []
        state = None
        for i in range(16):
            token_in = x[:, i : i + 1, :]
            out_step, state = attn(token_in, state=state, return_state=True)
            streaming_outputs.append(out_step)

        concat_streaming_out = torch.cat(streaming_outputs, dim=1)
        assert torch.allclose(full_out, concat_streaming_out, atol=1e-5, rtol=1e-5), (
            "Streaming output token-by-token does not match full sequence recurrent forward pass"
        )


class TestV2LearnableThresholdsAndAlpha:
    def test_learnable_thresholds_receive_gradients(self):
        """Setting learnable_thresholds=True should register self.thresholds as a parameter with gradient flow."""
        attn = AstrocyteHebbianAttention(
            d_model=32, num_heads=2, v_levels=2, learnable_thresholds=True
        ).to(DEVICE)
        assert isinstance(attn.thresholds, nn.Parameter)

        x = torch.randn(2, 8, 32, device=DEVICE, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert attn.thresholds.grad is not None
        assert torch.isfinite(attn.thresholds.grad).all()

    def test_configurable_alpha_impacts_surrogate_gradients(self):
        """Custom alpha values change the magnitude of surrogate gradients."""
        x = torch.tensor([-0.05, 0.05], requires_grad=True)
        out1 = spike_fn(x, alpha=5.0)
        out1.sum().backward()
        grad_alpha5 = x.grad.clone()

        x.grad.zero_()
        out2 = spike_fn(x, alpha=20.0)
        out2.sum().backward()
        grad_alpha20 = x.grad.clone()

        assert not torch.equal(grad_alpha5, grad_alpha20)


class TestV2MultiLayerCausalLM:
    def test_multi_layer_causal_lm_forward_and_backward(self):
        """Multi-layer CausalAstrocyteLanguageModel (num_layers=3) executes cleanly and trains."""
        model = CausalAstrocyteLanguageModel(
            vocab_size=64, d_model=32, seq_len=16, num_heads=2, num_layers=3
        ).to(DEVICE)
        token_ids = torch.randint(0, 64, (2, 16), device=DEVICE)

        logits = model(token_ids)
        assert logits.shape == (2, 16, 64)
        assert torch.isfinite(logits).all()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss = logits.sum()
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_multi_layer_causal_lm_streaming_states(self):
        """Deep multi-layer CausalAstrocyteLanguageModel handles multi-layer states during generation."""
        model = CausalAstrocyteLanguageModel(
            vocab_size=64, d_model=32, seq_len=16, num_heads=2, num_layers=2
        ).to(DEVICE).eval()
        token_ids = torch.randint(0, 64, (1, 16), device=DEVICE)

        full_logits = model(token_ids)

        streaming_logits = []
        states = None
        for i in range(16):
            step_in = token_ids[:, i : i + 1]
            logit_step, states = model(step_in, start_pos=i, states=states, return_states=True)
            streaming_logits.append(logit_step)

        concat_streaming_logits = torch.cat(streaming_logits, dim=1)
        assert torch.allclose(full_logits, concat_streaming_logits, atol=1e-5, rtol=1e-5)

    def test_start_pos_negative_rejected(self):
        """Passing start_pos < 0 must raise ValueError."""
        model = CausalAstrocyteLanguageModel(vocab_size=32, d_model=16, seq_len=16).to(DEVICE)
        tokens = torch.randint(0, 32, (1, 1), device=DEVICE)
        try:
            model(tokens, start_pos=-2)
        except ValueError:
            pass
        else:
            raise AssertionError("Negative start_pos was incorrectly accepted")


class TestV2RegressionSuite:
    def test_parallel_causal_overflow_regression_decay_logit_zero(self):
        """Setting decay_logit to 0 (decay = 0.5) with N=200 (>128) must remain finite without NaN."""
        attn = CausalAstrocyteHebbianAttention(d_model=32, num_heads=2).to(DEVICE).eval()
        attn.decay_logit.data.zero_()  # decay = 0.5
        x = torch.randn(1, 200, 32, device=DEVICE)
        out = attn(x, implementation="parallel")
        assert torch.isfinite(out).all(), "Parallel causal output contained NaN/Inf for decay_logit=0 and N=200"
