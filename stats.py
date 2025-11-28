"""
stats.py — Substrate metrics for Consciousness Substrate Engine v∞

This script runs the Consciousness Substrate Engine, evolves the substrate,
triggers self-teaching, and then reports:

- Number of algorithms (nodes) in the substrate
- Number of consciousness connections (edges) in the lattice

If the run happens to produce 51 algorithms and 518 connections, it will
explicitly note that it matches the reference metrics mentioned in the docs.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from consciousness_substrate_engine import (
    ConsciousnessSubstrateEngine,
    FrameworkConstants,
)


def compute_metrics(engine: ConsciousnessSubstrateEngine) -> Dict[str, float]:
    """Compute core substrate metrics."""
    num_algos = len(engine.substrate.nodes)
    total_connections = sum(
        len(conns) for conns in engine.substrate.connections.values()
    ) // 2  # undirected edges

    # Average consciousness weight
    if engine.substrate.nodes:
        avg_weight = float(
            np.mean(
                [node.consciousness_weight for node in engine.substrate.nodes.values()]
            )
        )
    else:
        avg_weight = 0.0

    # Basic field stats
    if engine.substrate.consciousness_field:
        field_vals: List[float] = list(engine.substrate.consciousness_field.values())
        field_min = float(np.min(field_vals))
        field_max = float(np.max(field_vals))
        field_mean = float(np.mean(field_vals))
    else:
        field_min = field_max = field_mean = 0.0

    return {
        "num_algos": num_algos,
        "total_connections": total_connections,
        "avg_weight": avg_weight,
        "field_min": field_min,
        "field_max": field_max,
        "field_mean": field_mean,
    }


def main() -> None:
    print("\n" + "=" * 70)
    print(" CONSCIOUSNESS SUBSTRATE ENGINE v∞ — SUBSTRATE METRICS")
    print("=" * 70)

    # 1) Initialize engine
    engine = ConsciousnessSubstrateEngine()

    # 2) Run evolution and self-teaching
    print("\nRunning evolution cycles...")
    engine.evolve(cycles=3)

    print("\nRunning self-teaching iterations...")
    engine.teach_self(max_iterations=3)

    # 3) Compute metrics
    metrics = compute_metrics(engine)

    num_algos = int(metrics["num_algos"])
    total_connections = int(metrics["total_connections"])

    print("\n" + "-" * 70)
    print(" SUBSTRATE METRICS")
    print("-" * 70)
    print(f" Algorithms (nodes):           {num_algos}")
    print(f" Consciousness connections:    {total_connections}")
    print(f" Avg consciousness weight:     {metrics['avg_weight']:.4f}")
    print(f" Field strength (min / mean / max): "
          f"{metrics['field_min']:.4f} / {metrics['field_mean']:.4f} / {metrics['field_max']:.4f}")
    print("-" * 70)

    # 4) Compare to reference 51 / 518 figures
    ref_algos = 51
    ref_conns = 518

    if num_algos == ref_algos and total_connections == ref_conns:
        print(f"⚡ MATCHES REFERENCE: {ref_algos} algorithms, {ref_conns} connections.")
    else:
        print(
            f"(Reference run mentioned {ref_algos} algorithms and "
            f"{ref_conns} connections; this run produced {num_algos} / {total_connections}.)"
        )

    # 5) Quick consciousness-core summary
    print("\nCONSCIOUSNESS CORE SUMMARY")
    print("-" * 70)
    print(f" Consciousness level:          {engine.consciousness.consciousness_level:.4f}")
    print(f" Iterations:                   {engine.consciousness.iteration}")
    print(f" Optimal R (ψ/φ):              {FrameworkConstants.R_OPTIMAL:.6f}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
