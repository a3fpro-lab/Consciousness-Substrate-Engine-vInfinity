# Benchmarks and Results — Sophia Prime & Proof Reactor

This document aggregates the key **numerical results and benchmarks** from the
design notes:

- Sophia Prime prime-cluster reactor
- Reaction–diffusion “Colossus” simulation
- TCC + Coq proof reactor (“96% world’s best by 10%”)

All underlying concepts are attributed to **Michael Warren Song**.

---

## 1. Sophia Prime Clusters

Sophia Prime clusters come from a fractional reaction–diffusion system with a
φ-based closure term.

Core pieces:

- Activator–inhibitor system (Gafiychuk-style).
- Fractional order **α** controlling regime (especially α ≈ 1.8).
- Closure term of the form **φ^{-α}**, coupling distant parts of the field.
- Center activator peaks are sampled and mapped to integer candidates.
- Candidates are filtered to primes → **Sophia clusters** such as:
  - 2–3–5
  - 11–13–17
  - 19–23–29

### 1.1 Cluster Size and Gap Formulas

From the derivation:

- **Cluster size**:
  \[
  \text{size} \approx \mathrm{round}(\phi^\alpha)
  \]

- **Cluster gap** (from critical wave number
  \(k_c = \sqrt{(f_u + \phi^{-α}) / D}\)):
  \[
  \boxed{\Delta p = \phi^{-α-1} \log p}
  \]

These formulas define the expected prime cluster structure.

### 1.2 Colossus Simulation (α = 1.8)

The “Colossus” GPU simulation described in the notes ran many reactor
iterations across α values and reported, for α = 1.8:

- Cluster size:
  - Observed: \(4 \pm 1\)
  - Theory: \(\mathrm{round}(\phi^{1.8}) = 4\)
  - Error: ~0%

- Internal gap:
  - Observed: \(0.208 \pm 0.005\)
  - Theory: ≈ 0.208 from \(\phi^{-α-1} \log p\)
  - Error: ~0%

- Prime coverage up to 100:
  - 14 out of 15 primes ≤ 100 recovered (only 37 missing).
  - Coverage: ≈ **93.33%**.

These numbers are encoded conceptually in `sophia_prime_reactor.py`, which
implements a simplified SophiaPrimeReactor with:

- Fractional dynamics approximated numerically.
- φ^{-α}-style closure.
- Peak detection at the center of the field.
- Prime filtering and cluster extraction.

---

## 2. TCC + Coq Proof Reactor

The second pillar of the “96% world’s best by 10%” document is the **TCC +
Coq proof reactor** concept:

- Start from an automated prover (AlphaProof-style).
- Add a **closure condition** (`TCC`) that encodes a structural constraint
  (often involving φ^{-α} or related inequalities).
- Export the proof structure as **Coq code**, embedding the closure condition
  directly into the proof skeleton.

### 2.1 Reported Benchmark

The design notes give a benchmark table:

| System                   | Score | Boost  |
|--------------------------|------:|------:|
| AlphaProof baseline      | 85.7  | —     |
| TCC + AlphaProof + Coq   | 95.3  | +9.6% |

Interpretation:

- The TCC + Coq layer yields a **+9.6% relative improvement** over the
  baseline AlphaProof-style solver on the same benchmark set.
- The phrase “Here I reach the 96% world’s best by 10%” refers to:
  - a boosted score in the mid-90s range, and
  - a ~10% relative gain compared to the baseline system.

### 2.2 Reactor Implementation (in this Repo)

The file `tcc_proof_reactor.py` contains a conceptual proof reactor:

- **`TCCTheorem`**  
  Holds:
  - `name`
  - informal `sketch`
  - exact `closure_condition`
  - `tags`
  - `baseline_score` and `tcc_score`
  - last generated `coq_stub`

- **`TCCProofReactor`**  
  Provides:
  - `register_theorem(...)`
  - `generate_coq_stub(name)` → Coq skeleton with closure embedded
  - `score_with_alpha_proof(name, ...)` → returns a heuristic baseline vs TCC score
  - `summary()` → aggregates mean scores and mean lift

Example usage:

```python
from tcc_proof_reactor import TCCProofReactor

reactor = TCCProofReactor()

thm = reactor.register_theorem(
    name="SophiaPrime_cluster_gap",
    sketch=(
        "Bound internal gaps in a Sophia Prime cluster via the "
        "Δp = φ^{-α-1} log p structure."
    ),
    closure_condition="phi**(-alpha - 1) * log(p) < C  for α > 1.8",
    tags=["number_theory", "sophia_prime", "tcc"],
)

reactor.score_with_alpha_proof(thm.name)
coq_stub = reactor.generate_coq_stub(thm.name)
print(coq_stub)
