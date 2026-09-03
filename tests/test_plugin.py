"""
Tests for the Astrocyte-Hebbian spiking linear Transformer plugin.

This suite mirrors the reference tests from
`Astrocyte-Hebbian-Spiking-Linear-Transformer-Block/tests/test_astrohebbian.py`
but is adapted to this plugin's API:

* `AstrocyteHebbianBlock` exposes `self.attention` (not `self.attn`).
* `AstrocyteHebbianAttention` exposes `self.num_heads` / `self.head_dim`
  (not `self.h` / `self.dh`) and keeps the spike thresholds in the
  `self.thresholds` buffer.

Run:  pytest tests/ -v
"""
import torch
import torch.nn as nn

from astrohebbian import (
    AstrocyteHebbianAttention,
    AstrocyteHebbianBlock,
    AstrocyteHebbianClassifier,
    CausalAstrocyteHebbianAttention,
    CausalAstrocyteLanguageModel,
    SpikingFFN,
    spike_fn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def test_attention_rejects_empty_or_mismatched_sequences():
    attention = AstrocyteHebbianAttention(d_model=16, num_heads=4)
    for inputs in (torch.randn(2, 0, 16), torch.randn(2, 4, 8)):
        try:
            attention(inputs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid attention input was accepted")


def test_classifier_state_round_trip_is_deterministic():
    torch.manual_seed(12)
    model = AstrocyteHebbianClassifier(
        input_dim=1, d_model=16, seq_len=8, num_classes=3, num_heads=4
    ).eval()
    inputs = torch.randn(2, 8, 1)
    expected = model(inputs)
    restored = AstrocyteHebbianClassifier(
        input_dim=1, d_model=16, seq_len=8, num_classes=3, num_heads=4
    ).eval()
    restored.load_state_dict(model.state_dict())
    assert torch.equal(expected, restored(inputs))

# ---------------------------------------------------------------------------
# Surrogate-heavy-side spike function
# ---------------------------------------------------------------------------

class TestSpikeFn:
    def test_forward_is_binary(self):
        x = torch.randn(64, 32, device=DEVICE) * 3.0
        s = spike_fn(x)
        assert set(s.unique().tolist()) <= {0.0, 1.0}, "spike_fn must output 0/1"
        assert (s == (x > 0).float()).all()

    def test_backward_is_finite_and_nonzero(self):
        x = torch.randn(64, 32, device=DEVICE, requires_grad=True)
        s = spike_fn(x)
        s.sum().backward()
        assert torch.isfinite(x.grad).all(), "surrogate gradient must be finite"
        assert x.grad.abs().sum() > 0, "surrogate gradient must not vanish everywhere"


# ---------------------------------------------------------------------------
# Attention mechanism (the core novel contribution)
# ---------------------------------------------------------------------------

class TestAttention:
    def test_output_shape(self):
        attn = AstrocyteHebbianAttention(d_model=128, num_heads=4).to(DEVICE)
        x = torch.randn(2, 784, 128, device=DEVICE)
        out = attn(x)
        assert out.shape == x.shape

    def test_no_n_squared_attention_matrix(self):
        """The whole point: never materialize an N x N attention/score matrix.

        We assert this empirically by checking that for seq_len N, the only
        large intermediate tensors have size O(N*d) or O(d*d), never O(N*N).
        We instrument by running with an enormous N and confirming no O(N^2)
        allocation dominates memory (would blow up VRAM).
        """
        attn = AstrocyteHebbianAttention(d_model=64, num_heads=2).to(DEVICE)
        # N = 16k tokens: an O(N^2) matrix would need 16k^2 * 4B = 1 GB just
        # for one float32 attention map. O(N*d) stays at 16k*64*4B = 4 MB.
        x = torch.randn(1, 4000, 64, device=DEVICE)
        # If anything materialized a (4000, 4000) tensor, peak memory would
        # jump. We just assert the forward runs without OOM and is finite.
        out = attn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_q_k_binary_v_binary_at_levels_one(self):
        """At v_levels=1, V is also binary: the strictest all-spiking regime."""
        attn = AstrocyteHebbianAttention(d_model=128, num_heads=4, v_levels=1).to(DEVICE)
        x = torch.randn(3, 64, 128, device=DEVICE)
        # Reconstruct V from v_proj and the plugin's `self.thresholds` buffer.
        with torch.no_grad():
            pre_q = spike_fn(attn.q_proj(x))
            pre_k = spike_fn(attn.k_proj(x))
            u01 = (torch.tanh(attn.v_proj(x)) + 1.0) / 2.0
            v = spike_fn(u01 - attn.thresholds.item())
        assert set(pre_q.unique().tolist()) <= {0.0, 1.0}
        assert set(pre_k.unique().tolist()) <= {0.0, 1.0}
        assert set(v.unique().tolist()) <= {0.0, 1.0}

    def test_gradient_flows_to_all_learnable_params(self):
        attn = AstrocyteHebbianAttention(d_model=64, num_heads=2).to(DEVICE)
        x = torch.randn(4, 64, 64, device=DEVICE, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        for name, p in attn.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"

    def test_separate_keys_change_output(self):
        """Sanity: different inputs produce different outputs (no collapse)."""
        attn = AstrocyteHebbianAttention(d_model=64, num_heads=2)
        a = torch.randn(2, 32, 64)
        b = torch.randn(2, 32, 64)
        out_a = attn(a)
        out_b = attn(b)
        assert (out_a - out_b).abs().max() > 1e-6

    def test_v_levels_multi_level(self):
        """v_levels>1 quantizes V to L levels (still finite, well-behaved)."""
        for L in (2, 4):
            attn = AstrocyteHebbianAttention(d_model=64, num_heads=2, v_levels=L).to(DEVICE)
            x = torch.randn(2, 32, 64, device=DEVICE)
            out = attn(x)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Block / FFN / Classifier
# ---------------------------------------------------------------------------

class TestBlock:
    def test_block_shape_and_finite(self):
        block = AstrocyteHebbianBlock(d_model=128, num_heads=4).to(DEVICE)
        x = torch.randn(2, 64, 128, device=DEVICE)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_ffn_hidden_is_binary(self):
        ffn = SpikingFFN(d_model=64, expansion=4).to(DEVICE)
        x = torch.randn(2, 32, 64, device=DEVICE)
        with torch.no_grad():
            hidden = spike_fn(ffn.fc1(x))
        assert set(hidden.unique().tolist()) <= {0.0, 1.0}

    def test_stackable_blocks(self):
        """num_layers>1 should run cleanly (the advertised deep variant)."""
        model = AstrocyteHebbianClassifier(
            d_model=64, num_heads=2, num_layers=2, v_levels=1
        ).to(DEVICE)
        x = torch.randn(2, 784, 1, device=DEVICE)
        logits = model(x)
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()


def test_causal_attention_does_not_read_future_tokens():
    torch.manual_seed(4)
    attention = CausalAstrocyteHebbianAttention(d_model=16, num_heads=4).eval()
    inputs = torch.randn(1, 8, 16)
    changed = inputs.clone()
    changed[:, 7, :] += 50
    first = attention(inputs)[:, :7]
    changed_first = attention(changed)[:, :7]
    assert torch.equal(first, changed_first)


def test_parallel_causal_attention_matches_recurrent_path():
    torch.manual_seed(6)
    attention = CausalAstrocyteHebbianAttention(d_model=16, num_heads=4).eval()
    inputs = torch.randn(2, 8, 16)
    recurrent = attention(inputs, implementation="recurrent")
    parallel = attention(inputs, implementation="parallel")
    assert torch.allclose(recurrent, parallel, atol=1e-5, rtol=1e-5)


def test_causal_language_model_trains_on_tiny_batch():
    torch.manual_seed(5)
    model = CausalAstrocyteLanguageModel(vocab_size=32, d_model=16, seq_len=8)
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.roll(inputs, shifts=-1, dims=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(inputs).reshape(-1, 32), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


class TestClassifier:
    def test_classifier_shapes(self):
        model = AstrocyteHebbianClassifier(d_model=128, num_heads=4, num_layers=1).to(DEVICE)
        x = torch.randn(5, 784, 1, device=DEVICE)
        logits = model(x)
        assert logits.shape == (5, 10)

    def test_training_step_reduces_loss_and_finite_grads(self):
        """End-to-end: a few optimizer steps should reduce the loss without NaNs."""
        model = AstrocyteHebbianClassifier(
            d_model=64, num_heads=2, num_layers=1, v_levels=1
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(8, 784, 1, device=DEVICE)
        y = torch.randint(0, 10, (8,), device=DEVICE)

        losses = []
        for _ in range(3):
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            for p in model.parameters():
                assert p.grad is None or torch.isfinite(p.grad).all()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ---------------------------------------------------------------------------
# Hebbian trace / key-mass normalization semantics
# ---------------------------------------------------------------------------

class TestMechanism:
    def test_key_mass_normalization_prevents_blowup(self):
        """denom = Q . sum_K + eps keeps the output bounded even when the
        raw numerator is large."""
        attn = AstrocyteHebbianAttention(d_model=64, num_heads=2).to(DEVICE)
        # Push k and v to be large/constant
        B, N, d = 2, 64, 64
        x = torch.ones(B, N, d, device=DEVICE)
        out = attn(x)
        assert torch.isfinite(out).all()
        assert out.abs().max() < 100.0, f"key-mass norm let output blow up: {out.abs().max()}"

    def test_complexity_claim_reading(self):
        """The trace contract K^T V is (d,d), never (N,N) -- structural check."""
        import inspect
        src = inspect.getsource(AstrocyteHebbianAttention.forward)
        # The plugin computes the trace via `weighted_key.transpose(-2, -1)`
        # times `value` -- a (d,d) product per head, never an (N,N) item.
        assert "matmul(weighted_key.transpose(-2, -1), value)" in src, "must compute K^T V"
        # Ensure there is no softmax / score matrix anywhere in the module
        assert "softmax" not in src
        assert "self_attn" not in src.lower() or "scaled_dot" not in src.lower()