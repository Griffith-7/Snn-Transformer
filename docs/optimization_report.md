# Optimization Report

The core attention equation and model architecture remain unchanged. These
experiments compare execution options against the eager FP32 implementation.

## Baseline Profile

RTX 3050, attention only, batch size 4, sequence length 784, width 128:

- Synchronized CUDA-event timing: `3.36 ms` per forward/backward step.
- Peak allocation: `43.03 MB` in the profiling run.
- Dominant operators: batched matrix multiplication, elementwise multiply,
  projection matrix multiplication, and tensor copies.

## Mixed Precision

Training-step comparison with batch size 4 and ten measured steps:

| Mode | Mean step time | Peak allocation | Finite |
| --- | ---: | ---: | :---: |
| Eager FP32 | 3.94 ms | 46.60 MB | yes |
| CUDA autocast FP16 | 5.94 ms | 38.25 MB | yes |

Autocast reduced memory by about 18%, but was slower on this hardware and
workload. It remains an optional future memory mode, not the default.

## Float32 Matmul Precision and Batch Size

Measured training-step timings in seconds:

| Matmul precision | Batch 1 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: |
| `highest` | 0.004283 | 0.003921 | 0.004192 |
| `high` | 0.003773 | 0.004119 | 0.003868 |

The `high` setting was slightly faster for batch sizes 1 and 8 but slightly
slower for batch size 4. It can alter floating-point results, so it is not
made a package-wide default without an end-to-end accuracy check.

## Kept Optimization

The single-level value threshold now stays as a tensor (`thresholds[0]`) rather
than using `.item()`. This preserves output behavior, avoids a host scalar
extraction, and improves compatibility with graph optimizers.

## Decision

Keep eager FP32 as the default. The evidence currently supports targeting
batched matrix multiplications and allocation/copy overhead in future work;
architecture and accuracy changes remain separate experiments.