# Resource Profile Baseline

These measurements are an isolated model-step profile on the RTX 3050. They
are useful for regression checks, but are not a replacement for the end-to-end
psMNIST benchmark.

## Frozen Configuration

- Batch size: 4
- Sequence length: 784
- Model width: 128
- Attention heads: 4
- Forward plus backward steps: 5 measured steps after 2 warm-up steps

## Measurement

- Mean forward/backward time: approximately `5.37 ms`
- Peak allocated VRAM: approximately `79.6 MB`

The single-level value path avoids extracting the threshold with `.item()`,
which keeps the computation in tensor form and removes a graph-break source for
execution optimizers. This is an implementation optimization only; it does not
change the threshold, spike values, or attention equation.

## Attention Scaling Snapshot

With batch size 1 and width 128, the measured peak allocations were:

| Sequence length | Peak allocated VRAM |
| ---: | ---: |
| 784 | 22.72 MB |
| 1568 | 28.86 MB |
| 3136 | 41.92 MB |

Short GPU timings are noisy and should be repeated with CUDA events before
making speed claims. The memory trend is consistent with sequence-linear
storage plus the fixed-width trace rather than an `N x N` attention map.

The repeatable profiler command is:

```bash
python benchmarks/profile_attention.py --device cuda --batch-size 4 --sequence-length 784 --d-model 128
```

## N-Scaling Crossover

Inference measurements on the RTX 3050 with batch size 1, width 128, four
heads, three warm-up steps, and ten measured steps:

| Sequence length | Astrocyte-Hebbian VRAM | Dense attention VRAM | Astrocyte-Hebbian time | Dense attention time |
| ---: | ---: | ---: | ---: | ---: |
| 784 | 12.62 MB | 29.81 MB | 0.883 ms | 0.787 ms |
| 1568 | 16.85 MB | 89.74 MB | 0.812 ms | 2.045 ms |
| 3136 | 25.30 MB | 320.03 MB | 0.941 ms | 8.110 ms |

The plot and machine-readable measurements are stored in
`results/n_scaling.png` and `results/n_scaling.json`. This is an inference
scaling result, not a full training benchmark. It shows the expected quadratic
dense-attention memory growth against substantially slower growth for the
Astrocyte-Hebbian implementation.