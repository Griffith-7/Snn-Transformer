"""Train a small causal language model on a streamed JSONL text sample."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from astrohebbian import CausalAstrocyteLanguageModel


class DenseCausalLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, num_heads):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device), diagonal=1
        ).bool()
        hidden = self.embedding(token_ids) + self.pos_encoder[:, :sequence_length, :]
        return self.lm_head(self.transformer(hidden, mask=mask))


def read_text_sample(path, max_bytes):
    chunks = []
    total_bytes = 0
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if not isinstance(text, str):
                continue
            chunks.append(text)
            total_bytes += len(text.encode("utf-8"))
            if total_bytes >= max_bytes:
                break
    text = "\n".join(chunks)
    return text.encode("utf-8")[:max_bytes]


def build_datasets(text_bytes, sequence_length, batch_size, seed):
    tokens = torch.tensor(list(text_bytes), dtype=torch.long)
    usable = (len(tokens) - 1) // sequence_length * sequence_length
    inputs = tokens[:usable].view(-1, sequence_length)
    targets = tokens[1 : usable + 1].view(-1, sequence_length)
    split = max(1, int(inputs.shape[0] * 0.9))
    split = min(split, inputs.shape[0] - 1)
    train_dataset = TensorDataset(inputs[:split], targets[:split])
    validation_dataset = TensorDataset(inputs[split:], targets[split:])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": float(torch.exp(torch.tensor(mean_loss)))}


def train_model(model, train_loader, validation_loader, device, steps):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iterator = iter(train_loader)
    losses = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    validation = evaluate_model(model, validation_loader, device)
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "seconds": time.perf_counter() - start,
        "peak_vram_mb": peak,
        "validation_loss": validation["loss"],
        "validation_perplexity": validation["perplexity"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--attention-implementation", choices=("recurrent", "parallel"), default="parallel"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/lm_prototype.json"))
    args = parser.parse_args(argv)
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if min(args.max_bytes, args.sequence_length, args.batch_size, args.steps, args.d_model) <= 0:
        raise ValueError("sizes and step counts must be positive")
    if args.d_model % args.num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    text_bytes = read_text_sample(args.data, args.max_bytes)
    train_loader, validation_loader = build_datasets(
        text_bytes, args.sequence_length, args.batch_size, args.seed
    )
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("the selected text sample is too small for train and validation")
    print(
        f"sample_bytes={len(text_bytes)} train_batches={len(train_loader)}"
        f" validation_batches={len(validation_loader)} device={device}"
    )
    results = {
        "config": {
            "data": str(args.data), "sample_bytes": len(text_bytes),
            "sequence_length": args.sequence_length, "batch_size": args.batch_size,
            "steps": args.steps, "d_model": args.d_model, "num_heads": args.num_heads,
            "attention_implementation": args.attention_implementation,
            "seed": args.seed, "device": str(device),
        },
        "models": {},
    }
    models = {
        "AstroHebbian SNN (Surrogate)": CausalAstrocyteLanguageModel(
            vocab_size=256, d_model=args.d_model,
            seq_len=args.sequence_length, num_heads=args.num_heads,
            attention_implementation=args.attention_implementation,
            gradient_mode="surrogate",
        ),
        "AstroHebbian SNN (Exact IFT)": CausalAstrocyteLanguageModel(
            vocab_size=256, d_model=args.d_model,
            seq_len=args.sequence_length, num_heads=args.num_heads,
            attention_implementation=args.attention_implementation,
            gradient_mode="exact",
        ),
        "Dense causal Transformer": DenseCausalLanguageModel(
            256, args.d_model, args.sequence_length, args.num_heads
        ),
    }
    for name, model in models.items():
        results["models"][name] = train_model(
            model, train_loader, validation_loader, device, args.steps
        )
        print(f"[{name}] {results['models'][name]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote LM results to {args.output}")
    return results


if __name__ == "__main__":
    main()
