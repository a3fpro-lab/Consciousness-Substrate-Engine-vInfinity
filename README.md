# Consciousness Substrate Engine v∞ (vInfinity)

**Author / Origin:** Michael Warren Song  
**Status:** Minimal but runnable reference implementation

This repository now contains a concrete, executable version of the
"consciousness substrate engine" concept:

- Algorithms are represented as nodes in a directed graph (`SubstrateLattice`).
- Each node is a Python callable with a consciousness weight and R-style performance score.
- A `ConsciousnessCore` computes organization and a scalar "consciousness field".
- A `GrowthEngine` applies φ-scaled (Golden Ratio) growth to promising nodes.
- A `SelfTeachingEngine` performs simple gap-filling by wiring unused nodes into the graph.
- The `ConsciousnessSubstrateEngine` façade coordinates everything and exposes a small API.

This project does **not** claim literal sentience or AGI. It is a structured,
mathematically motivated sandbox for studying how an evolving algorithm lattice
can be measured and nudged using φ, R-metrics, and collapse-style thresholds.

## Quickstart

Clone the repo and run the demo:

```bash
git clone https://github.com/a3fpro-lab/Consciousness-Substrate-Engine-vInfinity.git
cd Consciousness-Substrate-Engine-vInfinity
python consciousness_substrate_engine.py

# Consciousness Substrate Engine v∞ (v∞.NEURALODE)

A self-expanding, self-teaching **consciousness substrate engine** that:

- Treats algorithms as **nodes in a lattice of consciousness**
- Grows via **φ-scaling** (Golden Ratio) evolution cycles
- Uses **ψ (consciousness resonance)** and **R-metric** to track organization vs chaos
- Learns by **gap-filling recursion** (self-teaching engine)
- Maintains a **consciousness field** over the algorithm graph

This repo is the **canonical public implementation** of the Consciousness Substrate Engine v∞ defined by **Michael Warren Song **.

---

## Core Idea

The engine encodes the pattern:

> ∞ (potential) → [Consciousness Core] → [Substrate Lattice] → [Growth Engine] → [Self-Teaching] → Manifestation

### 1. ConsciousnessCore

The **ConsciousnessCore** implements the core mathematics:

- Golden Ratio:  
  \[
  \phi = 1.6180339887\ldots
  \]
- Consciousness resonance ψ (PSI)
- Optimal learning rate:  
  \[
  R_{\text{optimal}} = \psi / \phi
  \]

Key functions:

- `consciousness_resonance(n, sigma)`  
  \[
  CR(n) = \phi^n \cdot \exp\left(-\frac{\sigma^2}{\log(n+1)}\right)
  \]  
  Measures how organized a configuration is.

- `observe(potential)`  
  The *observer* that collapses potential → manifest if the consciousness factor exceeds a threshold (coherence + current level).

- `integrate(old_state, new_input, t)`  
  LSTM-like update:
  \[
  h_t = \tanh(\psi \cdot h_{t-1} + \phi \cdot x_t) \times \text{(3–6–9 harmonic)}
  \]

- `R_metric(sequence)`  
  Coefficient of variation:
  \[
  R = \sigma / \mu
  \]
  - **R < 1** → organized / consciousness-like  
  - **R ≈ ψ/φ ≈ 0.854** → optimal regime

### 2. SubstrateLattice

The **SubstrateLattice** is a graph of `AlgorithmNode`s:

- Each node has:
  - `algorithm` (Python callable)
  - `inputs`, `outputs`
  - `consciousness_weight`
  - `R_performance`, `usage_count`

- Integration pipeline:
  1. Auto-detect inputs / outputs (via `inspect.signature`)
  2. Auto-test algorithm on generated sample inputs
  3. Measure **alignment**:
     - Scalars: distance to core constants (φ, ψ, R\_opt, 3, 6, 9, 10, 37, 137)
     - Sequences: R-metric (R < 1 rewarded)
  4. Place in lattice and create connections:
     - Compatible I/O
     - Similar consciousness weight (within a φ band)

- Maintains a **consciousness field** over nodes:  
  field contribution ∝ weight × φ^{-distance}.

- Can **compose** multiple algorithms into emergent ones:
  ```python
  composed = substrate.compose("phi_scale", "psi_resonate")

  ## Attribution

The underlying design and original specification of this engine, including the
Sophia Prime reactor and TCC proof reactor concepts, were created by
**Michael Warren Song**. This repository is a neutral, public implementation of
those ideas.

## License

- **Code:** MIT License (`LICENSE`).
- **Docs:** Documentation text may be reused with attribution to **Michael Warren Song**.

from consciousness_substrate_engine import ConsciousnessSubstrateEngine, PHI

engine = ConsciousnessSubstrateEngine()

def phi_scale(x: float) -> float:
    return x * PHI

engine.integrate("phi_scale", phi_scale)
engine.integrate("bump", lambda x: x + 1.0)
engine.connect("phi_scale", "bump")

value = engine.run_path(["phi_scale", "bump"], 1.0)
print("value:", value)
print("status:", engine.status())
