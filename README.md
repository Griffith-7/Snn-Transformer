# Astrocyte-Hebbian Spiking Transformer Plugin

A standalone PyTorch plugin containing Astrocyte-Hebbian spiking linear-attention components. This project is intentionally independent from Exact-SNN; no Exact-SNN code, imports, or dependencies are included.

## Scope

This is a focused model plugin, not a general SNN framework. It provides:

- `AstrocyteHebbianAttention`: multi-head linear attention using binary Q/K/V activations.
- `AstrocyteHebbianBlock`: pre-norm Transformer-style block with a spiking FFN.
- `AstrocyteHebbianClassifier`: ready-to-train sequence classifier.
- `spike_fn`: binary Heaviside forward pass with surrogate gradients.

The implementation avoids an `N x N` attention matrix by computing the `K^T V` trace. It still uses ordinary dense PyTorch tensors for projections, normalization, residual paths, and training.

## What this is (and is not)

- **Binary activation spikes**: the inter-layer signals (Q, K, V, and the FFN hidden activation) are binary Heaviside spikes (0/1).
- **Surrogate gradients**: training uses a fast-sigmoid surrogate gradient through the spike threshold; it is not an exact spike-time gradient library.
- **Dense PyTorch execution**: forward/backward run on ordinary dense GPU tensors. This is a CPU/GPU software package, not an event-driven neuromorphic-hardware implementation, and reported times/memory are wall-clock/FLOP measurements, not hardware energy.
- **Full-sequence psMNIST mode**: the core `AstrocyteHebbianClassifier` (and the frozen baseline below) is full-sequence attention over N=784 pixels.
- **Separate causal LM experimental mode**: a distinct, experimental causal path (`CausalAstrocyteLanguageModel`) is provided for small language-model proof-of-concept work only.

This is a **focused model plugin, not a complete SNN framework**.

## Install

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e .[test]
pytest -q
```

For the optional psMNIST benchmark:

```bash
pip install -e .[benchmark]
python astrohebbian/benchmark.py
```

For a controlled three-seed summary:

```bash
python benchmarks/multi_seed.py --seeds 1 2 3 --output results/multi_seed.json
```

For the small causal language-model proof of concept:

```bash
python benchmarks/lm_prototype.py \
    --data /path/to/pretraining_code.jsonl \
    --max-bytes 10000000 --steps 50 --output results/lm_prototype.json
```

The causal LM is a separate experimental path. The full-sequence psMNIST model
and its baseline remain unchanged.

## Example

```python
import torch
from astrohebbian import AstrocyteHebbianClassifier

model = AstrocyteHebbianClassifier(
    input_dim=1,
    d_model=128,
    seq_len=784,
    num_heads=4,
    v_levels=1,
)

pixels = torch.randn(8, 784, 1)
logits = model(pixels)
print(logits.shape)  # torch.Size([8, 10])
```

## Results

Frozen three-seed psMNIST baseline (N=784, 60k train / 10k test, 6 epochs, RTX 3050, batch 64):

| Model | 3-seed mean test acc | Peak VRAM |
| --- | ---: | ---: |
| AstroHebbian Pure SNN | **86.82% ± 2.48%** | ~1004 MB |
| Transformer (dense O(N²)) | 77.80% ± 3.06% | ~1565 MB |

The SNN beats the dense baseline by **+9.03 pts accuracy** at **~36% lower peak VRAM**.
Mean runtime is not a headline figure: seed 2 was a large hardware/runtime outlier,
so only accuracy and memory are claimed as reliable. Full per-seed data:
`results/multi_seed.json`; the N-scaling memory crossover is in `results/n_scaling.png`.

## Project status

This is the clean standalone starting point for production hardening. The current attention is full-sequence rather than causal or streaming. Results are tracked in `docs/benchmark_baseline.md`.
