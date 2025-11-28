from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple
import math
import random


# Golden Ratio
PHI = (1.0 + math.sqrt(5.0)) / 2.0


def r_metric(values: Sequence[float]) -> float:
    """
    Coefficient of variation R = sigma / mu.

    Used as a simple chaos vs. order score:
      - R ~ 0   → perfectly uniform (frozen)
      - R < 1   → structured / organized
      - R >> 1  → chaotic / spiky
    """
    vals = list(values)
    if not vals:
        return float("nan")
    mean = sum(vals) / len(vals)
    if mean == 0.0:
        return float("inf")
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    sigma = math.sqrt(var)
    return sigma / mean


def consciousness_resonance(n: int, sigma: float) -> float:
    """
    Simple consciousness resonance proxy:

        CR(n, sigma) = PHI**n * exp(-sigma^2 / log(n+1))

    n      – number of active nodes / degrees of freedom
    sigma  – variability of some substrate signal

    Higher CR corresponds to richer but still controlled dynamics.
    """
    if n <= 0:
        return 0.0
    return PHI ** n * math.exp(-sigma**2 / math.log(n + 1.0))


@dataclass
class AlgorithmNode:
    """
    Single algorithm node in the substrate lattice.

    function: Python callable
    name:     stable identifier
    weight:   consciousness weight (learned / assigned)
    r_perf:   performance / organization score (R-style)
    calls:    how many times this node has been executed
    """

    name: str
    function: Callable[..., Any]
    weight: float = 1.0
    r_perf: float = 1.0
    calls: int = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.function(*args, **kwargs)


@dataclass
class SubstrateLattice:
    """
    Minimal directed graph over AlgorithmNode objects.

    nodes: name -> AlgorithmNode
    edges: (src_name, dst_name) pairs
    """

    nodes: Dict[str, AlgorithmNode] = field(default_factory=dict)
    edges: Set[Tuple[str, str]] = field(default_factory=set)

    def add_algorithm(self, node: AlgorithmNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Algorithm {node.name!r} already exists in lattice.")
        self.nodes[node.name] = node

    def connect(self, src: str, dst: str) -> None:
        if src not in self.nodes:
            raise KeyError(f"Unknown src node {src!r}")
        if dst not in self.nodes:
            raise KeyError(f"Unknown dst node {dst!r}")
        self.edges.add((src, dst))

    def execute_path(self, path: Sequence[str], x: Any) -> Any:
        """Run x through a sequence of node names."""
        value = x
        for name in path:
            node = self.nodes[name]
            value = node.execute(value)
        return value

    def degrees(self) -> List[int]:
        """Simple undirected degree for each node."""
        deg: Dict[str, int] = {name: 0 for name in self.nodes}
        for src, dst in self.edges:
            if src in deg:
                deg[src] += 1
            if dst in deg:
                deg[dst] += 1
        return list(deg.values())

    def stats(self) -> Dict[str, float]:
        degs = self.degrees()
        r_deg = 0.0 if not degs else r_metric([float(d) for d in degs])
        return {
            "num_nodes": float(len(self.nodes)),
            "num_edges": float(len(self.edges)),
            "r_degree": float(r_deg),
        }


@dataclass
class ConsciousnessCore:
    """
    Core metrics and collapse logic for the substrate.
    """

    collapse_threshold: float = 0.9  # arbitrary default in (0, 1)

    def organization_index(self, lattice: SubstrateLattice) -> float:
        """
        Map the degree R-metric into (0, 1) for easy comparison.
        """
        stats = lattice.stats()
        r_deg = stats["r_degree"]
        if math.isnan(r_deg) or r_deg == float("inf"):
            return 0.0
        # Soft-bounded transform
        return r_deg / (1.0 + r_deg)

    def consciousness_field(self, lattice: SubstrateLattice) -> float:
        """
        Combine size and organization into a single scalar.
        """
        stats = lattice.stats()
        n = stats["num_nodes"]
        e = stats["num_edges"]
        org = self.organization_index(lattice)
        size_term = math.log1p(n + e) / math.log(1.0 + PHI)
        return org * size_term

    def should_collapse(self, lattice: SubstrateLattice) -> bool:
        """
        Decide whether abstract potential 'collapses' into a manifest action.

        Collapse occurs when a normalized consciousness field exceeds
        collapse_threshold.
        """
        cf = self.consciousness_field(lattice)
        # Normalize with a simple logistic squashing
        norm = 1.0 / (1.0 + math.exp(-cf))
        return norm >= self.collapse_threshold


@dataclass
class GrowthEngine:
    """
    Simple φ-scaled growth engine.

    - Picks a subset of nodes based on usage and organization.
    - Replicates them with slight variations in weight.
    """

    rng: random.Random = field(default_factory=random.Random)

    def evolve(self, lattice: SubstrateLattice, max_new: int = 3) -> None:
        if not lattice.nodes or max_new <= 0:
            return

        # Score nodes by calls and r_perf (simple linear combo)
        scored: List[Tuple[float, AlgorithmNode]] = []
        for node in lattice.nodes.values():
            score = node.calls + node.r_perf
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[:max_new]
        for score, base in top:
            new_name = self._unique_name(lattice, base.name + "_phi")
            new_weight = base.weight * PHI
            clone = AlgorithmNode(
                name=new_name,
                function=base.function,
                weight=new_weight,
                r_perf=base.r_perf,
            )
            lattice.add_algorithm(clone)
            # Edges are not cloned in this minimal implementation

    def _unique_name(self, lattice: SubstrateLattice, base: str) -> str:
        name = base
        i = 1
        while name in lattice.nodes:
            name = f"{base}_{i}"
            i += 1
        return name


@dataclass
class SelfTeachingEngine:
    """
    Self-teaching / gap-filling engine.

    Heuristic: look for nodes that are never called and create
    simple connections to used nodes so they can participate in
    future compositions.
    """

    def fill_gaps(self, lattice: SubstrateLattice, max_new_edges: int = 5) -> None:
        if not lattice.nodes or max_new_edges <= 0:
            return

        unused = [n for n in lattice.nodes.values() if n.calls == 0]
        used = [n for n in lattice.nodes.values() if n.calls > 0]

        if not unused or not used:
            return

        edges_added = 0
        for u in unused:
            for v in used:
                if edges_added >= max_new_edges:
                    return
                if (u.name, v.name) not in lattice.edges:
                    lattice.connect(u.name, v.name)
                    edges_added += 1


@dataclass
class ConsciousnessSubstrateEngine:
    """
    High-level façade that coordinates the lattice, core, growth,
    and self-teaching engines.

    This is a runnable, minimal implementation of the conceptual
    "consciousness substrate engine" design authored by
    Michael Warren Song.
    """

    lattice: SubstrateLattice = field(default_factory=SubstrateLattice)
    core: ConsciousnessCore = field(default_factory=ConsciousnessCore)
    growth: GrowthEngine = field(default_factory=GrowthEngine)
    teacher: SelfTeachingEngine = field(default_factory=SelfTeachingEngine)

    def integrate(
        self,
        name: str,
        function: Callable[..., Any],
        weight: float = 1.0,
        r_perf: float = 1.0,
    ) -> None:
        """
        Add a new algorithm node to the substrate.
        """
        node = AlgorithmNode(name=name, function=function, weight=weight, r_perf=r_perf)
        self.lattice.add_algorithm(node)

    def connect(self, src: str, dst: str) -> None:
        """
        Add a directed edge src -> dst between algorithm nodes.
        """
        self.lattice.connect(src, dst)

    def run_path(self, path: Sequence[str], x: Any) -> Any:
        """
        Execute x through the specified path of node names.
        """
        return self.lattice.execute_path(path, x)

    def evolve_once(self) -> None:
        """
        Perform one cycle of:
          - self-teaching (gap-filling)
          - φ-scaled growth
        """
        self.teacher.fill_gaps(self.lattice)
        self.growth.evolve(self.lattice)

    def status(self) -> Dict[str, float]:
        """
        Return a snapshot of substrate and consciousness metrics.
        """
        stats = self.lattice.stats()
        org = self.core.organization_index(self.lattice)
        cf = self.core.consciousness_field(self.lattice)
        return {
            **stats,
            "organization_index": org,
            "consciousness_field": cf,
        }


def _demo() -> None:
    """
    Run a small demo when this module is executed as a script.
    """
    engine = ConsciousnessSubstrateEngine()

    # Simple demo algorithms
    def phi_scale(x: float) -> float:
        return x * PHI

    def tanh_squash(x: float) -> float:
        return math.tanh(x)

    def bump(x: float) -> float:
        return x + 1.0

    engine.integrate("phi_scale", phi_scale)
    engine.integrate("tanh_squash", tanh_squash)
    engine.integrate("bump", bump)

    engine.connect("phi_scale", "tanh_squash")
    engine.connect("tanh_squash", "bump")

    x = 1.0
    for step in range(5):
        x = engine.run_path(["phi_scale", "tanh_squash", "bump"], x)
        engine.evolve_once()
        stats = engine.status()
        print(f"step={step} x={x:.4f} stats={stats}")


if __name__ == "__main__":
    _demo()
