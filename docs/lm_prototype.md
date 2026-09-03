# Causal Language-Model Prototype

The plugin now includes a separate causal path. It does not replace the frozen
full-sequence psMNIST model.

## Data and Protocol

- Source: `pretraining_code.jsonl`
- Reader: line-by-line JSONL streaming
- Sample: 10,000,000 bytes
- Encoding: raw UTF-8 bytes, vocabulary size 256
- Sequence length: 64
- Batch size: 8
- Training steps: 50
- Width: 64
- Heads: 4
- Device: RTX 3050 CUDA

## Proof-of-Concept Result

| Model | Initial loss | Final loss | Loss decreased | Time | Peak VRAM |
| --- | ---: | ---: | :---: | ---: | ---: |
| Astrocyte-Hebbian causal | 5.5419 | 3.0002 | yes | 40.49 s | 23.68 MB |
| Dense causal Transformer | 5.7559 | 3.0293 | yes | 2.86 s | 35.81 MB |

The result verifies that the causal Astrocyte-Hebbian path can learn a next-byte
prediction objective on a small real code sample while using less memory. On the
held-out split, Astrocyte-Hebbian reached loss `3.2096` and perplexity `24.77`,
while the dense baseline reached loss `3.1419` and perplexity `23.15`.

This remains a proof of concept rather than a quality benchmark: only 50 steps
were run and the byte-level vocabulary is intentionally simple.

The causal attention currently updates its trace in a Python loop over sequence
positions when `attention_implementation="recurrent"`. The optional parallel
prefix implementation is mathematically equivalent for this workload and is
used by default in the LM runner. On the same CUDA microbenchmark, recurrent
mode took `72.03 ms` per training step and parallel mode took `3.99 ms`; peak
memory was `20.43 MB` and `27.52 MB`, respectively. Parallel mode is therefore
the better short-sequence GPU choice, while recurrent mode remains the
streaming-friendly choice.