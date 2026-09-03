"""Reproducible psMNIST benchmark for Astrocyte-Hebbian attention."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

sys.path.insert(0, str(ROOT))
from astrohebbian.model import AstrocyteHebbianClassifier  # noqa: E402


class TransformerClassifier(nn.Module):
    def __init__(self, d_in, d_model, seq_len, num_classes, num_heads):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj_in(x) + self.pos_encoder[:, : x.shape[1], :]
        return self.classifier(self.transformer(x)[:, -1, :])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, test_loader, device, seq_len):
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data = data.view(data.size(0), seq_len, 1).to(device)
        target = target.to(device)
        correct += model(data).argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
    return 100.0 * correct / total


def train_model(model, name, train_loader, test_loader, device, epochs, seq_len):
    print(f"\n--- Training {name} ---", flush=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine(step, steps_per_epoch, steps_per_epoch * epochs),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for data, target in train_loader:
            data = data.view(data.size(0), seq_len, 1).to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += data.size(0)
        print(
            f"Epoch {epoch + 1}/{epochs} | train acc: {100.0 * correct / total:.2f}%"
            f" | lr: {scheduler.get_last_lr()[0]:.6f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    test_accuracy = evaluate(model, test_loader, device, seq_len)
    result = {
        "test_accuracy": test_accuracy,
        "train_time_seconds": elapsed,
        "peak_vram_mb": peak_memory,
    }
    print(
        f"[{name}] Time: {elapsed:.2f}s | VRAM: {peak_memory:.2f} MB"
        f" | TEST acc: {test_accuracy:.2f}%",
        flush=True,
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--v-levels", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Write results to JSON")
    parser.add_argument(
        "--save-checkpoints", action="store_true", help="Save model state dictionaries"
    )
    return parser


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.test_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    set_seed(args.seed)
    device = select_device(args.device)
    seq_len = 784
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)
    if args.train_size is not None:
        train_dataset = Subset(train_dataset, range(min(args.train_size, len(train_dataset))))
    if args.test_size is not None:
        test_dataset = Subset(test_dataset, range(min(args.test_size, len(test_dataset))))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, pin_memory=device.type == "cuda"
    )
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    print(f"BENCHMARK | seed: {args.seed} | device: {device}", flush=True)
    print(
        f"psMNIST (N={seq_len}) | train: {len(train_dataset)} | test: {len(test_dataset)}"
        f" | {args.epochs} epochs | warmup+cosine | clip 1.0",
        flush=True,
    )
    results = {
        "config": {
            "seed": args.seed, "device": str(device), "epochs": args.epochs,
            "batch_size": args.batch_size, "test_batch_size": args.test_batch_size,
            "d_model": args.d_model, "num_heads": args.num_heads,
            "num_layers": args.num_layers, "v_levels": args.v_levels,
            "train_size": len(train_dataset), "test_size": len(test_dataset),
        },
        "models": {},
    }

    snn = AstrocyteHebbianClassifier(
        d_model=args.d_model, num_heads=args.num_heads,
        num_layers=args.num_layers, seq_len=seq_len, v_levels=args.v_levels
    )
    results["models"]["AstroHebbian Pure SNN"] = train_model(
        snn, "AstroHebbian Pure SNN", train_loader, test_loader, device, args.epochs, seq_len
    )
    if args.save_checkpoints:
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save(snn.state_dict(), CKPT_DIR / "astrohebbian_pure_snn.pt")
    del snn
    if device.type == "cuda":
        torch.cuda.empty_cache()

    transformer = TransformerClassifier(1, args.d_model, seq_len, 10, args.num_heads)
    results["models"]["Transformer (dense)"] = train_model(
        transformer, "Transformer baseline", train_loader, test_loader, device, args.epochs, seq_len
    )
    if args.save_checkpoints:
        torch.save(transformer.state_dict(), CKPT_DIR / "transformer_baseline.pt")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote results to {args.output}", flush=True)
    return results


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
