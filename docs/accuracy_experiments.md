# Accuracy Experiments

The official frozen accuracy baseline is the three-seed full psMNIST result in
`docs/benchmark_baseline.md`. The experiment runner keeps later changes
separate from that baseline and saves a partial JSON result after every seed.

Available controlled configurations:

| Name | Attention blocks | Value levels |
| --- | ---: | ---: |
| `baseline` | 1 | 1 |
| `two_layers` | 2 | 1 |
| `multi_level_values` | 1 | 2 |

Run a full experiment set with:

```bash
python benchmarks/accuracy_experiments.py \
  --experiments baseline two_layers multi_level_values \
  --seeds 1 2 3 \
  --output results/accuracy_experiments.json
```

For a quick wiring smoke test, use small subsets and one epoch. Smoke results
must not be compared with the official full-dataset accuracy baseline.

Every experiment must be evaluated by test accuracy, standard deviation,
training time, peak VRAM, and finite outputs. A configuration is retained only
when its improvement is reproducible and does not regress the baseline's
accuracy or numerical stability.