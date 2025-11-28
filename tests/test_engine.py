import math

from consciousness_substrate_engine import (
    r_metric,
    consciousness_resonance,
    ConsciousnessSubstrateEngine,
)


def test_r_metric_basic():
    assert math.isclose(r_metric([1.0, 1.0, 1.0]), 0.0)
    val = r_metric([1.0, 2.0, 3.0])
    assert val > 0.0


def test_consciousness_resonance_monotone_in_sigma():
    n = 10
    low = consciousness_resonance(n, sigma=0.1)
    high = consciousness_resonance(n, sigma=2.0)
    assert high < low  # more noise → lower resonance


def test_engine_integration_and_evolution():
    engine = ConsciousnessSubstrateEngine()

    def inc(x: float) -> float:
        return x + 1.0

    engine.integrate("inc", inc)
    engine.integrate("inc2", inc)
    engine.connect("inc", "inc2")

    x = engine.run_path(["inc", "inc2"], 0.0)
    assert x == 2.0

    before_nodes = engine.status()["num_nodes"]
    engine.evolve_once()
    after_nodes = engine.status()["num_nodes"]
    assert after_nodes >= before_nodes
