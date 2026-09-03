# Plugin Memory

**Current Status**: Plugin ships the production module `astrohebbian/model.py` containing the
self-contained `AstrocyteHebbianAttention` layer. Official frozen plugin baseline (three-seed
psMNIST 60k×6): Pure SNN **86.82% ± 2.48%** TEST vs dense Transformer **77.80% ± 3.06%**,
~36% lower peak VRAM (see `docs/benchmark_baseline.md` and `results/multi_seed.json`).

## Plugin Structure
```
astrohebbian/       model.py + benchmark.py (the drop-in plugin)
benchmarks/         multi_seed.py, accuracy_experiments.py, n_scaling.py, lm_prototype.py, profile_attention.py
docs/               roadmap, memory, paper_draft, benchmark_baseline, accuracy_experiments, lm_prototype, profile_baseline, optimization_report
results/            multi_seed.json, n_scaling.json, n_scaling.png, lm_prototype.json
```

The plugin is self-contained — no dependencies beyond PyTorch. Import and use
`AstrocyteHebbianAttention` as a drop-in replacement for standard multi-head attention.

## Key Lessons (do not forget)
- Short-schedule single runs have ±8pt seed variance → multi-seed averaging mandatory.
- Always report TEST-set accuracy.
- Binarizing V cost zero accuracy (regularizer); key-mass norm + learnable decay do the real work.
- Residual Variance Explosion is the #1 gotcha when building spiking attention from scratch.

## Decisions Made
- **Final Solution**: Astrocyte Hebbian Spiking Transformer Block (`astrohebbian/model.py`).
  - K^T V Hebbian trace inside the Astrocyte ($O(d^2)$, no $N \times N$ matrix).
  - Key-mass normalization ($QK^TV / (Q \cdot \Sigma K)$) + learnable per-channel decay.
  - Full block: positional encodings, pre-LayerNorm, residuals, spiking FFN.
- **Superseded Solutions**: Pure AMHA v1 (HDC columns) and Trainable AMHA v2 were archived
  and removed during the plugin cleanup; their history is captured in `docs/roadmap.md`.

## Plugin Integration Notes
- The block expects input shape `[B, N, d_model]` and returns the same shape.
- Pass `training=True` during forward pass to enable surrogate gradient; `False` for inference.
- Spiking FFN is included inside the block — no separate FFN layer needed.
- Default hyperparameters work out-of-the-box for psMNIST ($N=784$, $d=128$).
