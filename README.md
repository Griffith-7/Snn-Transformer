# Astrocyte-Hebbian Spiking Transformer Plugin

A standalone PyTorch **plugin** (not a framework) providing Astrocyte-Hebbian spiking linear-attention components with support for both **Surrogate Gradients** and **Exact SNN (Implicit Function Theorem / IFT) Gradients**.

## What this is

This package is a **drop-in set of reusable `nn.Module` components**. You import them and plug them into *your own* model — you stay in control of the architecture. It does not define a training loop or force a whole model on you.

### The plugin API (the building blocks you plug in)

These are the core, self-contained modules:

- `AstrocyteHebbianAttention`: multi-head linear attention using binary Q/K/V activations with configurable gradient mode (`"surrogate"` or `"exact"`). `forward(x) -> x`, same shape in/out.
- `AstrocyteHebbianBlock`: pre-norm Transformer-style block (`attention` + spiking `SpikingFFN`). The single unit you'd drop into any transformer. `forward(x) -> x`.
- `SpikingFFN`: feed-forward with a binary hidden activation.
- `ExactLinearSpike` / `ExactSpike`: fused linear + binary-spike layers with exact IFT gradients — drop-in replacements for `nn.Linear` + a spike function.
- `spike_fn`: binary Heaviside threshold supporting fast-sigmoid surrogate gradients and exact closed-form IFT gradients.
- `CausalAstrocyteHebbianAttention`: causal/streaming variant with a recurrent decayed Hebbian state.

### Example wrappers (built on top of the plugin)

These are full models assembled from the plugin components — useful starting points, not the intended way to use the package:

- `AstrocyteHebbianClassifier`: ready-to-train sequence classifier (embedding + blocks + head).
- `CausalAstrocyteLanguageModel`: byte-level causal language model with streaming recurrence.

The implementation avoids an `N x N` attention matrix by computing the `K^T V` trace. It uses ordinary dense PyTorch tensors for projections, normalization, residual paths, and training.

## Gradient Modes: Exact Mode (Default) vs. Surrogate Mode

- **Exact Mode (`gradient_mode="exact"`, Default)**: Fuses each linear projection with a binary threshold spike (`ExactLinearSpike` / `ExactSpike`) and backpropagates through the spike-time map of an exponential integrate-and-fire membrane using the Implicit Function Theorem (IFT). Only firing neurons (`pre > θ`) receive a gradient, with magnitude `τ·θ / (pre·(pre−θ))`.
- **Surrogate Mode (`gradient_mode="surrogate"`)**: Uses a fast-sigmoid surrogate derivative approximation $\sigma'(x) \cdot \alpha$ through firing thresholds during backpropagation.

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

For comparing **Surrogate SNN**, **Exact SNN**, and **Dense Transformer** on a causal language model task:

```bash
python benchmarks/lm_prototype.py \
    --data data/pretraining_code.jsonl \
    --max-bytes 5000000 --steps 30 --output results/lm_comparison.json
```

For sequence classification comparison:

```bash
python benchmarks/surrogate_vs_exact.py --epochs 5 --output results/surrogate_vs_exact.json
```

## Example

### Plug the block into any transformer (plugin usage)

```python
import torch
from astrohebbian import AstrocyteHebbianAttention, AstrocyteHebbianBlock

# Drop-in attention: same shape in / out
attn = AstrocyteHebbianAttention(d_model=128, num_heads=4, gradient_mode="exact")
out = attn(torch.randn(2, 64, 128))          # torch.Size([2, 64, 128])

# Or a full pre-norm block (attention + spiking FFN) — stack these with your own
# residuals, norms, embeddings, and head however you like.
block = AstrocyteHebbianBlock(d_model=128, num_heads=4, expansion=4, gradient_mode="exact")
out = block(torch.randn(2, 64, 128))         # torch.Size([2, 64, 128])
```

### Ready-made wrappers (built from the plugin)

```python
from astrohebbian import AstrocyteHebbianClassifier, CausalAstrocyteLanguageModel

# Sequence Classifier with Exact Spike Gradients
classifier = AstrocyteHebbianClassifier(
    input_dim=1,
    d_model=128,
    seq_len=784,
    num_heads=4,
    v_levels=1,
    gradient_mode="exact",  # Options: "surrogate" | "exact"
)

pixels = torch.randn(8, 784, 1)
logits = classifier(pixels)
print(logits.shape)  # torch.Size([8, 10])

# Causal Language Model with Exact Spike Gradients
lm = CausalAstrocyteLanguageModel(
    vocab_size=256,
    d_model=64,
    seq_len=128,
    num_heads=4,
    gradient_mode="exact",
)

tokens = torch.randint(0, 256, (4, 128))
lm_logits = lm(tokens)
print(lm_logits.shape)  # torch.Size([4, 128, 256])
```

## Benchmark Results

### Causal Language Model (`pretraining_code.jsonl`, 5MB text sample, 30 steps, CUDA)

| Model Variant | Initial Loss | Final Train Loss | Validation Loss | Validation Perplexity | Step Time | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AstroHebbian SNN (Surrogate)** | 5.584 | 3.826 | 3.439 | 31.15 | 2.38s | **29.18 MB** |
| **AstroHebbian SNN (Exact Mode)** | 5.752 | **3.624** | 3.588 | 36.15 | **1.99s** | **29.73 MB** |
| **Dense Causal Transformer** | 5.719 | 3.350 | 3.290 | 26.85 | 1.38s | 36.36 MB |

Key Observations:
- **Lower Training Loss at Step 30**: AstroHebbian SNN in exact/reciprocal mode reached a lower training loss (**3.624**) than surrogate mode (**3.826**) after 30 steps.
- **VRAM Savings**: Both SNN variants saved **~18% Peak VRAM** (~29.2–29.7 MB) relative to the dense Transformer (~36.4 MB).
- **Speed**: Exact IFT spike gradients completed 30 steps in **1.99s** (vs 2.38s for surrogate gradients).

## Project status

This standalone plugin is an **experimental research package** for prototyping and benchmarking spiking linear transformers. Results are tracked in `docs/benchmark_baseline.md` and `results/lm_comparison.json`. Further multi-epoch training and validation are recommended before production deployment.
