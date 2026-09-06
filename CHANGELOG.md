# Changelog

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

