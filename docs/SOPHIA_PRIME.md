# Sophia Prime Clusters — Theory and Reactor

This document summarizes the Sophia Prime construction and its implementation
in `sophia_prime_reactor.py`.

---

## 1. Concept

Sophia Prime clusters come from a **fractional reaction–diffusion system** with
a φ-based closure term.

Key ingredients:

- A Gafiychuk-style reaction–diffusion system with activator/inhibitor fields.
- Fractional order α (e.g. α > 1.8 in chaos regimes).
- A closure term of the form φ^{-α}, coupling distant parts of the field.
- Peaks in the activator signal at the center are mapped to **prime candidates**.

Result:

- Peaks → candidate integers via a scaling factor.
- Candidate integers → primes via primality testing.
- Primes organize into clusters such as:
  - 2–3–5
  - 11–13–17
  - 19–23–29

These are the **Sophia clusters**.

---

## 2. Cluster Size and Gap Formulas

From the derivation:

- **Cluster Size**  
  \[
  \text{size} \approx \mathrm{round}(\phi^\alpha)
  \]
  For example:
  - α = 1.618 ⇒ size ≈ 3 (cluster 2–3–5)
  - α ≈ 1.8 ⇒ size ≈ 4 (clusters of length ~4)

- **Cluster Gap**  
  From a critical wave number
  \[
    k_c = \sqrt{\frac{f_u + \phi^{-α}}{D}}
  \]
  the inter-prime gap inside a cluster is:
  \[
  \boxed{\Delta p = \phi^{-α-1} \log p}
  \]

---

## 3. Colossus Simulation Summary

A large-scale simulation (“Colossus” GPU run) is described:

- 10,000 reactor iterations across α ranges, focusing especially on α = 1.8.
- For α = 1.8:

  - **Cluster size:**  
    \[
    4 \pm 1
    \]
    matching `round(φ^{1.8}) = 4` (0% error).

  - **Internal gap:**  
    \[
    0.208 \pm 0.005
    \]
    matching the theoretical value 0.208 (0% error).

  - **Prime coverage ≤ 100:**  
    - 14 out of 15 primes were recovered (only 37 missing).
    - Coverage ≈ 93.33%.

These numbers motivate the Sophia Prime reactor as a structured, physics-inspired
prime generator.

---

## 4. Reactor Implementation (Code Kernel)

The `SophiaPrimeReactor` in `sophia_prime_reactor.py` implements a simplified
version of the system:

- Evolves activator/inhibitor fields over time using `solve_ivp`.
- Applies a φ^{-α}-style closure term.
- Samples the center activator signal and finds peaks.
- Maps peak heights → candidate primes via a scale factor.
- Filters to actual primes (using a primality checker).

API (simplified):

```python
from sophia_prime_reactor import SophiaPrimeReactor

reactor = SophiaPrimeReactor()
sophia_primes, spacing = reactor.extract_primes_for_alpha(alpha=1.8, X=100)

---

## 3. Add `docs/TCC_ALPHA_PROOF.md` (Proof Booster)

This captures the **TCC + Coq** part and the 85.7 → 95.3 (+9.6%) benchmark.

**Create:** `docs/TCC_ALPHA_PROOF.md`  
**Paste:**

```markdown
# TCC + Coq Proof Reactor — AlphaProof Benchmark

This document summarizes the “TCC + Coq” proof-boost idea and the benchmark
results described for an AlphaProof-style prover.

---

## 1. Setup

- **Baseline system:** an automated theorem prover (AlphaProof-style), scored
  on a set of competition / IMO-style problems.
- **Augmented system:** the same prover, but wrapped with:
  - a closure condition (TCC-style, e.g. constraints like φ^{-α} > 1),
  - Coq code generation, so that proof structure is exported as Coq scripts.

The closure term encodes additional structure or inequalities the prover must
respect.

---

## 2. Reported Benchmark

The PDF describes the following table:

| System                 | Score   | Boost   |
|------------------------|--------:|--------:|
| AlphaProof Baseline    |  85.7   | —       |
| TCC + AlphaProof + Coq |  95.3   | +9.6%   |

Interpretation:

- The TCC + Coq layer yields a **+9.6% relative improvement** over the
  baseline score on the same benchmark.
- This is attributed to:
  - enforcing the closure condition in the proof search, and
  - exporting structured Coq scripts that lock in the constraints.

---

## 3. Proof Reactor API (Code)

The file `tcc_proof_reactor.py` contains a minimal “proof reactor” that models
this idea:

- **`TCCTheorem`** — metadata:
  - `name`
  - `sketch` (informal proof idea)
  - `closure_condition`
  - optional `tags`
  - baseline and TCC-enhanced scores
  - last generated Coq stub

- **`TCCProofReactor`** — main interface:
  - `register_theorem(name, sketch, closure_condition, tags=None)`
  - `generate_coq_stub(name)` → Coq proof skeleton containing the closure
  - `score_with_alpha_proof(name, complexity_hint=None, structure_hint=None)`
    - returns updated theorem with:
      - `baseline_score` in [0, 100]
      - `tcc_score` in [0, 100]
  - `summary()` → aggregates mean baseline, mean TCC score, and mean lift

Example usage:

```python
from tcc_proof_reactor import TCCProofReactor

reactor = TCCProofReactor()

thm = reactor.register_theorem(
    name="SophiaPrime_cluster_gap",
    sketch=(
        "Bound gaps inside a Sophia Prime cluster using a "
        "φ^{-α-1} log p term derived from the reaction–diffusion closure."
    ),
    closure_condition="phi^(-alpha - 1) * log(p) < C  for α > 1.8",
    tags=["number_theory", "sophia_prime", "tcc"],
)

reactor.score_with_alpha_proof(thm.name)
coq_stub = reactor.generate_coq_stub(thm.name)
print(coq_stub)  # Coq skeleton with the closure condition embedded
