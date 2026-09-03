"""Profile one Astrocyte-Hebbian attention training step."""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from astrohebbian import AstrocyteHebbianAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=784)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if min(args.batch_size, args.sequence_length, args.d_model, args.warmup, args.steps) <= 0:
        raise ValueError("profile dimensions and step counts must be positive")

    device = torch.device(args.device)
    model = AstrocyteHebbianAttention(args.d_model, num_heads=4).to(device)
    inputs = torch.randn(args.batch_size, args.sequence_length, args.d_model, device=device)

    def run_step():
        model.zero_grad(set_to_none=True)
        model(inputs).sum().backward()

    for _ in range(args.warmup):
        run_step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(args.steps):
            run_step()
        end_event.record()
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000 / args.steps
    else:
        import time

        start_time = time.perf_counter()
        for _ in range(args.steps):
            run_step()
        elapsed = (time.perf_counter() - start_time) / args.steps

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with profile(
        activities=[ProfilerActivity.CPU]
        + ([ProfilerActivity.CUDA] if device.type == "cuda" else []),
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(1):
            with record_function("astrohebbian_attention_step"):
                run_step()
            profiler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    print(f"device={device} mean_step_seconds={elapsed:.6f} peak_vram_mb={peak_memory:.2f}")
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=12))


if __name__ == "__main__":
    main()
