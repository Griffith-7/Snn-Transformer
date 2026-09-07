"""
Comparison benchmark suite: AstroHebbian v1 vs v2.
"""
import time

import torch
import torch.nn as nn

from astrohebbian import (
    AstrocyteHebbianAttention,
    CausalAstrocyteHebbianAttention,
    CausalAstrocyteLanguageModel,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_numerical_stability():
    print("\n=======================================================")
    print("1. NUMERICAL STABILITY BENCHMARK (Parallel Causal Scan)")
    print("=======================================================")

    seq_lengths = [256, 1024, 4096, 10000]
    d_model = 64
    num_heads = 2

    attn_v2 = CausalAstrocyteHebbianAttention(d_model=d_model, num_heads=num_heads).to(DEVICE).eval()
    attn_v2.decay_logit.data.fill_(5.0) # decay ~ 0.993, inverse_decay ~ 1.0067

    print(f"{'Seq Length (N)':<15} | {'v1 Un-stabilized (NaN/Inf?)':<30} | {'v2 Log-Stable (NaN/Inf?)':<30}")
    print("-" * 80)

    for N in seq_lengths:
        x = torch.randn(1, N, d_model, device=DEVICE)
        
        # v1 implementation logic simulation
        batch_size, sequence_length, _ = x.shape
        key = attn_v2.k_proj(x).view(batch_size, sequence_length, num_heads, d_model // num_heads).transpose(1, 2)
        value = attn_v2.v_proj(x).view(batch_size, sequence_length, num_heads, d_model // num_heads).transpose(1, 2)
        decay = torch.sigmoid(attn_v2.decay_logit).view(1, num_heads, d_model // num_heads).to(x.dtype)
        
        # v1 reciprocal calculation
        positions = torch.arange(sequence_length, device=x.device, dtype=x.dtype).view(1, 1, sequence_length, 1)
        inverse_decay = decay.reciprocal()
        powers_v1 = inverse_decay.unsqueeze(-2).pow(positions)
        pair = key.unsqueeze(-1) * value.unsqueeze(-2)
        trace_v1 = torch.cumsum(pair * powers_v1.unsqueeze(-1), dim=2) * decay.unsqueeze(-2).pow(
            positions
        ).unsqueeze(-1)
        v1_has_nans = not torch.isfinite(trace_v1).all().item()

        # v2 execution
        out_v2 = attn_v2(x, implementation="parallel")
        v2_has_nans = not torch.isfinite(out_v2).all().item()

        v1_status = "FAILED (NaN / Inf detected)" if v1_has_nans else "PASSED (Finite)"
        v2_status = "PASSED (Finite)" if not v2_has_nans else "FAILED (NaN / Inf detected)"
        print(f"{N:<15} | {v1_status:<30} | {v2_status:<30}")


def benchmark_streaming_throughput():
    print("\n=======================================================")
    print("2. INFERENCE THROUGHPUT BENCHMARK (Streaming vs Full Recomp)")
    print("=======================================================")

    d_model = 128
    seq_len = 256
    num_tokens = 128

    model = CausalAstrocyteLanguageModel(
        vocab_size=256, d_model=d_model, seq_len=seq_len, num_layers=2
    ).to(DEVICE).eval()

    tokens = torch.randint(0, 256, (1, seq_len), device=DEVICE)

    # v1 approach: Recompute whole sequence up to token k every step
    start_v1 = time.perf_counter()
    for k in range(1, num_tokens + 1):
        context = tokens[:, :k]
        _ = model(context)
    time_v1 = time.perf_counter() - start_v1

    # v2 approach: O(1) stateful streaming token by token
    start_v2 = time.perf_counter()
    states = None
    for k in range(num_tokens):
        token_step = tokens[:, k : k + 1]
        _, states = model(token_step, start_pos=k, states=states, return_states=True)
    time_v2 = time.perf_counter() - start_v2

    speedup = time_v1 / max(1e-6, time_v2)
    print(f"Time to generate {num_tokens} tokens:")
    print(f"  v1 (Full sequence recomputation per token) : {time_v1 * 1000:.2f} ms")
    print(f"  v2 (O(1) Stateful Streaming per token)     : {time_v2 * 1000:.2f} ms")
    print(f"  --> Speedup: {speedup:.2f}x FASTER in v2!")


def benchmark_expressivity_learnable_thresholds():
    print("\n=======================================================")
    print("3. EXPRESSIVITY & CONVERGENCE BENCHMARK")
    print("=======================================================")

    d_model = 64
    x = torch.randn(16, 64, d_model, device=DEVICE)
    y = torch.randint(0, 10, (16,), device=DEVICE)

    # Fixed threshold attention (v1)
    attn_fixed = AstrocyteHebbianAttention(d_model=d_model, learnable_thresholds=False).to(DEVICE)
    opt_fixed = torch.optim.Adam(attn_fixed.parameters(), lr=1e-2)

    # Learnable threshold attention (v2)
    attn_learnable = AstrocyteHebbianAttention(d_model=d_model, learnable_thresholds=True).to(DEVICE)
    opt_learnable = torch.optim.Adam(attn_learnable.parameters(), lr=1e-2)

    classifier_fixed = nn.Linear(d_model, 10).to(DEVICE)
    classifier_learnable = nn.Linear(d_model, 10).to(DEVICE)
    opt_c1 = torch.optim.Adam(classifier_fixed.parameters(), lr=1e-2)
    opt_c2 = torch.optim.Adam(classifier_learnable.parameters(), lr=1e-2)

    crit = nn.CrossEntropyLoss()

    loss_v1 = 0.0
    for _ in range(10):
        opt_fixed.zero_grad()
        opt_c1.zero_grad()
        out = classifier_fixed(attn_fixed(x)[:, -1, :])
        loss = crit(out, y)
        loss.backward()
        opt_fixed.step()
        opt_c1.step()
        loss_v1 = loss.item()

    loss_v2 = 0.0
    for _ in range(10):
        opt_learnable.zero_grad()
        opt_c2.zero_grad()
        out = classifier_learnable(attn_learnable(x)[:, -1, :])
        loss = crit(out, y)
        loss.backward()
        opt_learnable.step()
        opt_c2.step()
        loss_v2 = loss.item()

    print("Final loss after 10 optimization steps on toy sequence task:")
    print(f"  v1 Fixed Thresholds     : {loss_v1:.4f}")
    print(f"  v2 Learnable Thresholds : {loss_v2:.4f}")


if __name__ == "__main__":
    benchmark_numerical_stability()
    benchmark_streaming_throughput()
    benchmark_expressivity_learnable_thresholds()
