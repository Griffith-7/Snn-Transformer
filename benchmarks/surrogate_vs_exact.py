"""Comprehensive comparison benchmark: Surrogate SNN vs Exact SNN vs Dense Transformer."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from astrohebbian import AstrocyteHebbianClassifier


def build_synthetic_sequence_dataset(num_samples=1000, seq_len=128, input_dim=1, num_classes=10, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(num_samples, seq_len, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    split = int(0.8 * num_samples)
    train_dataset = TensorDataset(X[:split], y[:split])
    test_dataset = TensorDataset(X[split:], y[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def train_eval_classifier(model, train_loader, test_loader, device, epochs=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        
    start_time = time.perf_counter()
    
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += by.size(0)
                
        test_acc = (correct / total) * 100.0
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "test_acc": test_acc
        })
        
    total_time = time.perf_counter() - start_time
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    
    return {
        "final_test_acc": history[-1]["test_acc"],
        "total_time_sec": total_time,
        "sec_per_epoch": total_time / epochs,
        "peak_vram_mb": peak_vram,
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="Surrogate vs Exact SNN Classification Benchmark")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/surrogate_vs_exact.json"))
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running classification benchmark on device: {device}")
    
    train_loader, test_loader = build_synthetic_sequence_dataset(seq_len=args.seq_len)
    
    models = {
        "AstroHebbian SNN (Surrogate)": AstrocyteHebbianClassifier(
            input_dim=1, d_model=64, seq_len=args.seq_len, num_classes=10, gradient_mode="surrogate"
        ),
        "AstroHebbian SNN (Exact IFT)": AstrocyteHebbianClassifier(
            input_dim=1, d_model=64, seq_len=args.seq_len, num_classes=10, gradient_mode="exact"
        ),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        metrics = train_eval_classifier(model, train_loader, test_loader, device, epochs=args.epochs)
        results[name] = metrics
        print(
            f"Final Test Acc: {metrics['final_test_acc']:.2f}% | "
            f"Time: {metrics['total_time_sec']:.2f}s "
            f"({metrics['sec_per_epoch']:.2f}s/epoch) | "
            f"VRAM: {metrics['peak_vram_mb']:.1f} MB"
        )
        
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
