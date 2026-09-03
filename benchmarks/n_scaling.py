"""Measure sequence-length scaling for SNN and dense attention."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from astrohebbian import AstrocyteHebbianAttention


class DenseAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(self, inputs):
        batch_size, sequence_length, d_model = inputs.shape
        query = self.q_proj(inputs).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(inputs).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(inputs).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)
        return self.o_proj(output)


def measure(model, inputs, device, mode, warmup, steps):
    model.eval()
    for _ in range(warmup):
        with torch.set_grad_enabled(mode == "training"):
            output = model(inputs)
            if mode == "training":
                output.sum().backward()
                model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.perf_counter()

    for _ in range(steps):
        with torch.set_grad_enabled(mode == "training"):
            output = model(inputs)
            if mode == "training":
                output.sum().backward()
                model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 1000 / steps
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        elapsed = (time.perf_counter() - start_time) / steps
        peak_memory = 0.0
    return {"seconds": elapsed, "peak_vram_mb": peak_memory}


def measure_one(sequence_length, args, device):
    inputs = torch.randn(args.batch_size, sequence_length, args.d_model, device=device)
    row = {"sequence_length": sequence_length, "models": {}}
    model_classes = {
        "astrohebbian": AstrocyteHebbianAttention,
        "dense": DenseAttention,
    }
    for name, model_class in model_classes.items():
        model = model_class(args.d_model, args.num_heads).to(device)
        try:
            row["models"][name] = measure(
                model, inputs, device, args.mode, args.warmup, args.steps
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row["models"][name] = {"status": "oom"}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def plot_results(results, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in (("astrohebbian", "Astrocyte-Hebbian"), ("dense", "Dense attention")):
        rows = [row for row in results if "seconds" in row["models"].get(name, {})]
        axes[0].plot(
            [row["sequence_length"] for row in rows],
            [row["models"][name]["peak_vram_mb"] for row in rows],
            marker="o", label=label,
        )
        axes[1].plot(
            [row["sequence_length"] for row in rows],
            [row["models"][name]["seconds"] for row in rows],
            marker="o", label=label,
        )
    axes[0].set(xlabel="Sequence length", ylabel="Peak VRAM (MB)", title="Memory scaling")
    axes[1].set(xlabel="Sequence length", ylabel="Seconds per step", title="Time scaling")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[784, 1568, 3136])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=("inference", "training"), default="inference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/n_scaling.json"))
    parser.add_argument("--plot", type=Path, default=Path("results/n_scaling.png"))
    args = parser.parse_args(argv)
    if min(args.sequence_lengths) <= 0:
        raise ValueError("sequence lengths must be positive")
    if min(args.d_model, args.num_heads, args.batch_size, args.warmup, args.steps) <= 0:
        raise ValueError("dimensions and step counts must be positive")
    if args.d_model % args.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    results = [
        measure_one(sequence_length, args, device)
        for sequence_length in args.sequence_lengths
    ]
    payload = {"config": vars(args) | {"device": str(device)}, "results": results}
    payload["config"]["output"] = str(args.output)
    payload["config"]["plot"] = str(args.plot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        plot_results(results, args.plot)
    except ImportError:
        print("matplotlib is unavailable; JSON was written but no plot was generated")
    print(f"Wrote scaling results to {args.output}")
    print(f"Wrote scaling plot to {args.plot}")
    return payload


if __name__ == "__main__":
    main()