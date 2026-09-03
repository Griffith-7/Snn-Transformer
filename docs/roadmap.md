# Spiking Attention Plugin — Roadmap

## Vision
Provide a drop-in Spiking Linear Attention plugin that replaces dense $O(N^2)$ self-attention with a purely spiking, $O(d^2)$ Hebbian mechanism—making it possible to swap in sparse, event-driven attention for any Transformer encoder with zero code changes beyond an import.

## Plugin Status
The plugin ships a single, self-contained PyTorch module (`AstrocyteHebbianAttention`) and a benchmark script. It is designed to be imported into any existing SNN or Transformer pipeline as a replacement layer.

## Sub-Problems & Progress

### [x] Sub-Problem 1: Mathematical Formulation
- Define how Query, Key, and Value spikes are projected into Hyperdimensional space.
- Formulate the binding operation (Coincidence Detection).
- Formulate the global context aggregation (Superposition via Neuromodulation).
- Formulate the routing operation (Unbinding).

### [x] Sub-Problem 2: Neuron Dynamics
- Select and configure the spiking neuron models (LIF with surrogate gradients).
- Map the mathematical operations (binding, superposition) to biological neuron mechanisms.

### [x] Sub-Problem 3: Network Architecture
- Design the topological structure of the AMHA module.
- Define input encoding (Fixed Random Projection), the sequence of layers (1-to-1 vertical wiring), and output decoding.

### [x] Sub-Problem 4: Simulation and Validation
- Implement the HDSA module as a self-contained plugin in PyTorch.
- Design unit tests to prove routing works without building an $N \times N$ dense matrix.
- Benchmark against standard attention on synthetic tasks.

## Tasks Done
- [x] Initial research on the Locality Deficit.
- [x] Identification of HDC and Binding by Synchrony.
- [x] Rapid Prototyping and Math Formulation.
- [x] Selection of Astrocyte-Modulated HDC.
- [x] Formalize Differential Equations for LIF and Astrocyte models.
- [x] Finalize Network Topology (Fixed HDC Encoder + 1-to-1 AMHA Core).
- [x] Implement PyTorch Simulation and verify $O(N \cdot D)$ time complexity.
- [x] Phase 1 Complete: Astrocyte Hebbian Spiking Transformer Block (36.51% psMNIST prototype).
- [x] Archive superseded solutions to `archive/`.

## Phase 2: Closing the Accuracy Gap (Complete)
Goal: Beat the Transformer baseline (52.45%) on psMNIST while keeping $O(d^2)$ complexity.

- [x] **P2.1 Capacity Recovery**: Analog V pathway (spike only Q/K); restore information lost to binarization.
- [x] **P2.2 Multi-Head Hebbian Traces**: Split $d=128$ into 4 heads of independent $32 \times 32$ traces.
- [x] **P2.3 Proper Normalization**: Replace `1/N` homeostatic scaling with key-mass normalization ($QK^TV$)/(Q·ΣK).
- [x] **P2.4 Temporal Selectivity**: Learnable per-channel decay $\lambda \in [0.993, 0.9997]$ in the trace (delta-rule / RWKV-style forgetting).
- [x] **P2.5 Training Hygiene**: LR warmup + cosine decay, gradient clipping.
- [ ] **P2.6 Evidence**: Full 60k dataset, more epochs, N-scaling crossover benchmark ($N \in \{784, 1568, 3136\}$).

### Phase 2 Results (psMNIST, 15k samples, 3 epochs)
| Model | Acc | Time | VRAM |
| :--- | :--- | :--- | :--- |
| Transformer (same run) | 45.28% | 122.70s | 1564 MB |
| **AstroHebbian v2** | **38.51%** | **39.74s** | 954 MB |
| AstroHebbian v1 (archived) | 36.51% | 37.08s | 815 MB |

### Phase 2 FULL Results — MILESTONE: BEAT THE TRANSFORMER ✅
(psMNIST, FULL 60k samples, 6 epochs, warmup+cosine LR, grad clip. Single-run
history; accuracy only — runtime figures are superseded by the 3-seed baseline above.)
| Model | Final Acc | Time | Peak VRAM |
| :--- | :--- | :--- | :--- |
| Transformer (dense O(N²)) | 82.18% | 1047s (~17.4 min) | 1565 MB |
| **AstroHebbian v2 (1 block)** | **82.72%** ✅ | 335s | **954 MB (-39%)** |
| **AstroHebbian v2-deep (2 blocks)** | **85.81%** ✅✅ | 626s | 1474 MB |

## Phase 3: Fully-Spiking Variant (Complete)
- [x] **v3 Pure Binary**: Q, K, V, and FFN hidden states all communicated as binary spikes.
- [x] **Finding**: Going 100% spiking costs nothing — the pure binary model slightly *outperforms* the hybrid.
- [x] Multi-level spike coding replaces the analog V pathway entirely.

### FROZEN RESULTS (psMNIST, full 60k train, 3-seed test avg; see `benchmark_baseline.md`)
| Model | Test Acc (3-seed mean) | Peak VRAM |
| :--- | :--- | :--- |
| **AstroHebbian (pure spiking)** | **86.82% ± 2.48%** | 1003.76 MB |
| Transformer (dense O(N²)) | 77.80% ± 3.06% | 1564.70 MB |

## Remaining Future Work
- [x] N-scaling crossover benchmark ($N \in \{784, 1568, 3136\}$) — headline VRAM-explosion figure.
- [x] Multi-seed averages for final table; longer schedules toward 90%+.
- [x] Language-modeling benchmark beyond psMNIST classification.
- [ ] Synaptic-op energy accounting / neuromorphic deployment story.
- [x] Package as a pip-installable module with proper `setup.py` / `pyproject.toml`.
- [x] Multi-seed psMNIST averages and N-scaling crossover recorded under `results/` and `docs/`.
- [x] Causal LM proof-of-concept (`language_model.py`, `benchmarks/lm_prototype.py`).
