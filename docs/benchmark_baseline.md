# Benchmark Baseline

This record freezes the fresh benchmarks for the standalone plugin. The
three-seed table is the official accuracy baseline for future changes.

## Fresh Run

| Model | Test accuracy | Train time | Peak VRAM | Relative speed |
| --- | ---: | ---: | ---: | ---: |
| AstroHebbian SNN (plugin) | 85.62% | 384 s | 1003 MB | 3.0x |
| Transformer (dense) | 78.55% | 1156 s | 1565 MB | 1.0x |

This "relative speed" column is a single-fast-run observation and is superseded by
the controlled three-seed run below, which does not support a reliable mean speedup.

Protocol: RTX 3050, pixel-sequential MNIST with sequence length 784, 60,000
training examples, 10,000 test examples, 6 epochs. The SNN used the default
`d_model=128`, `num_heads=4`, `num_layers=1`, and `v_levels=1` configuration.

## Controlled Three-Seed Run

| Model | Mean test accuracy | Accuracy std | Mean peak VRAM | Runtime observations |
| --- | ---: | ---: | ---: | --- |
| AstroHebbian SNN (plugin) | 86.82% | 2.48% | 1003.76 MB | 398 s, 3666 s, 416 s |
| Transformer (dense) | 77.80% | 3.06% | 1564.70 MB | 1174 s, 1185 s, 1187 s |

The controlled run used seeds `1`, `2`, and `3` with the same RTX 3050,
pixel-sequential MNIST protocol, 6 epochs, batch size 64, `d_model=128`,
`num_heads=4`, `num_layers=1`, and `v_levels=1`. Full per-seed data is stored
in `results/multi_seed.json`.

The SNN accuracy advantage over the dense baseline is `9.03` percentage points
on the three-seed mean. Peak VRAM is consistently about 36% lower. SNN runtime
is not yet a stable claim because seed 2 was a large hardware/runtime outlier;
repeat timing under controlled system load before publishing a mean speedup.

## Interpretation

The plugin reproduces the original project behavior within the observed run
variance. The SNN avoids an `N x N` attention matrix, but the implementation
still uses dense PyTorch tensor operations and is not a hardware energy
measurement.