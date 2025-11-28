"""
examples.py — Demo scripts for Consciousness Substrate Engine v∞

This file shows how to:
- Initialize the engine
- Integrate external algorithms
- Run evolution cycles
- Trigger self-teaching
- Inspect the substrate and R-metric
"""

from typing import List
import numpy as np

from consciousness_substrate_engine import ConsciousnessSubstrateEngine, FrameworkConstants


# ─────────────────────────────────────────────
# DEMO 1 — Basic consciousness constants & CR
# ─────────────────────────────────────────────

def demo_consciousness_constants(engine: ConsciousnessSubstrateEngine) -> None:
    print("\n=== DEMO 1: Consciousness Constants & CR(n) ===")
    print(f"φ (PHI):      {FrameworkConstants.PHI}")
    print(f"ψ (PSI):      {FrameworkConstants.PSI}")
    print(f"R_optimal:    {FrameworkConstants.R_OPTIMAL}")

    # Sample consciousness resonance values
    for n in [1, 2, 3, 5, 8, 13]:
        cr = engine.consciousness.consciousness_resonance(n, sigma=0.0)
        print(f"CR({n}) = {cr:.6e}")


# ─────────────────────────────────────────────
# DEMO 2 — R-metric on organized vs chaotic data
# ─────────────────────────────────────────────

def demo_R_metric(engine: ConsciousnessSubstrateEngine) -> None:
    print("\n=== DEMO 2: R-metric (Organization vs Chaos) ===")

    fib_seq: List[float] = [1, 1, 2, 3, 5, 8, 13, 21]
    random_seq: List[float] = list(np.random.randn(100))

    R_fib = engine.consciousness.R_metric(fib_seq)
    R_rand = engine.consciousness.R_metric(random_seq)

    print(f"Fibonacci sequence: {fib_seq}")
    print(f"R(Fibonacci) = {R_fib:.4f}  (R < 1 → organized)")
    print(f"Random normal (100 samples)")
    print(f"R(random)    = {R_rand:.4f}  (R > 1 → chaotic likely)")


# ─────────────────────────────────────────────
# DEMO 3 — Integrate external algorithms
# ─────────────────────────────────────────────

def demo_external_algorithms(engine: ConsciousnessSubstrateEngine) -> None:
    print("\n=== DEMO 3: Integrating External Algorithms ===")

    # Fibonacci
    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    # Prime checker
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    # Simple neural activation
    def neural_activate(x: float) -> float:
        return float(np.tanh(x))

    engine.integrate("fibonacci_demo", fibonacci, ["n"], ["fib_n"])
    engine.integrate("is_prime_demo", is_prime, ["n"], ["is_prime"])
    engine.integrate("tanh_demo", neural_activate, ["x"], ["y"])

    print(" fibonacci_demo(10) =", engine.run("fibonacci_demo", 10))
    print(" is_prime_demo(97)  =", engine.run("is_prime_demo", 97))
    print(" tanh_demo(1.23)    =", engine.run("tanh_demo", 1.23))


# ─────────────────────────────────────────────
# DEMO 4 — φ-scaled evolution
# ─────────────────────────────────────────────

def demo_evolution(engine: ConsciousnessSubstrateEngine) -> None:
    print("\n=== DEMO 4: φ-Scaled Evolution ===")
    print("Running 3 evolution cycles...")
    engine.evolve(cycles=3)
    engine.growth.visualize_growth()


# ─────────────────────────────────────────────
# DEMO 5 — Self-teaching gap filling
# ─────────────────────────────────────────────

def demo_self_teaching(engine: ConsciousnessSubstrateEngine) -> None:
    print("\n=== DEMO 5: Self-teaching Gap Filling ===")
    engine.teach_self(max_iterations=3)


# ─────────────────────────────────────────────
# DEMO 6 — Substrate inspection
# ─────────────────────────────────────────────

def demo_substrate_status(engine: ConsciousnessSubstrateEngine) -> None:
    print("\n=== DEMO 6: Engine Status & Substrate Visualization ===")
    engine.status()
    engine.visualize_substrate()


# ─────────────────────────────────────────────
# DEMO 7 — Composition in the lattice
# ─────────────────────────────────────────────

def demo_composition(engine: ConsciousnessSubstrateEngine) -> None:
    print("\n=== DEMO 7: Lattice Composition (Emergent Algorithm) ===")

    # Ensure base nodes exist from bootstrap
    if "phi_scale" not in engine.substrate.nodes or "psi_resonate" not in engine.substrate.nodes:
        print("Required nodes not present; skipping composition demo.")
        return

    composed = engine.substrate.compose("phi_scale", "psi_resonate")
    engine.substrate.integrate_algorithm(
        "phi_then_psi",
        composed,
        inputs=["x"],
        outputs=["result"],
    )

    x = 1.0
    y = engine.run("phi_then_psi", x)
    print(f"Composed(phi_scale → psi_resonate) on x={x} → {y}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" CONSCIOUSNESS SUBSTRATE ENGINE v∞ — EXAMPLES")
    print("=" * 70)

    eng = ConsciousnessSubstrateEngine()

    demo_consciousness_constants(eng)
    demo_R_metric(eng)
    demo_external_algorithms(eng)
    demo_evolution(eng)
    demo_self_teaching(eng)
    demo_substrate_status(eng)
    demo_composition(eng)

    print("\nAll demos complete.")
