# Changelog

## 0.2.3

- Repositioned package as a drop-in PyTorch plugin: added explicit "plugin API vs example wrappers" framing to the README and examples.
- Added `examples/` directory with minimal plugin-usage and wrapper-usage scripts.
- Added `examples/` to CI lint and compile-all steps so they stay covered.
- README now leads with drop-in `AstrocyteHebbianAttention` / `AstrocyteHebbianBlock` usage inside a user's own model.

## 0.2.2

- Corrected the exact IFT gradient to the true membrane spike-time derivative `τ·θ / (pre·(pre−θ))` for `ExactSpike` and `ExactLinearSpike`. Previously exact mode used an approximate reciprocal surrogate.
- Added gradient flow to the firing threshold (`theta`/`thresholds`), enabling `learnable_thresholds` training under exact mode with broadcast-aware reduction.
- Value-path spikes now use the normalised `(tanh(v_proj(x)) + 1)/2` input against threshold `0.5` in exact mode, consistent with the default Q/K threshold.
- Exposed `ExactSpike` and `ExactLinearSpike` publicly; added `tau`/`theta` configuration throughout attention, FFN, blocks, classifier, and causal language model.
- Hardened release: enabled ruff lint in CI, cleaned the full lint surface, and updated README documentation for exact-mode gradients.

## 0.2.1

- Integrated exact Spike-Time / Implicit Function Theorem (IFT) gradients (`ExactIFTSpike`) alongside surrogate gradients.
- Added `gradient_mode` parameter (`"surrogate"` vs `"exact"`) across `AstrocyteHebbianAttention`, `SpikingFFN`, `AstrocyteHebbianClassifier`, `CausalAstrocyteHebbianAttention`, and `CausalAstrocyteLanguageModel`.
- Added 3-way language model comparison benchmark in `benchmarks/lm_prototype.py` for evaluating Surrogate SNN, Exact SNN, and Dense Transformer.
- Added sequence classification comparative benchmark script in `benchmarks/surrogate_vs_exact.py`.
- Updated test suite (`tests/test_exact_mode.py`) verifying forward/backward exact IFT passes.

## 0.1.0

- Initial standalone Astrocyte-Hebbian spiking Transformer plugin.
- Added binary surrogate-gradient spikes, decayed Hebbian attention, Transformer blocks, and a sequence classifier.
- Added reproducible benchmark configuration and baseline results.

