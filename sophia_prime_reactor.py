from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import random


@dataclass
class SophiaPrimeConfig:
    """Configuration for the Sophia Prime 1D reactor.

    This is a deliberately small, CPU-friendly surrogate for the
    larger simulations described in the design notes. It preserves
    the qualitative structure:

      * fractional-time PDE style update
      * φ-scaled diffusion
      * activator/inhibitor coupling
      * extraction of prime-like clusters from peaks

    All concepts are attributed to Michael Warren Song.
    """

    n_cells: int = 241
    n_steps: int = 377
    dt: float = 0.015
    dx: float = 1.0
    alpha: float = 1.823  # “TCC α” from the notes
    diffusion_phi: float = (1.618033988749895 - 1.0)  # φ - 1
    seed: int = 137
    peak_threshold: float = 0.62
    cluster_gap_max: int = 6
    cluster_min_size: int = 3


class SophiaPrimeReactor:
    """Minimal Sophia Prime reactor.

    Usage:

        reactor = SophiaPrimeReactor()
        clusters = reactor.run()
        print(clusters)

    Each cluster is a tuple ``(primes, meta)`` where ``primes`` is a
    list of integers and ``meta`` holds a dictionary of derived
    statistics (length, span, local R-metric).
    """

    def __init__(self, config: SophiaPrimeConfig | None = None) -> None:
        self.config = config or SophiaPrimeConfig()
        self.rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------
    def _laplacian(self, u: List[float]) -> List[float]:
        n = len(u)
        out = [0.0] * n
        for i in range(1, n - 1):
            out[i] = u[i - 1] - 2.0 * u[i] + u[i + 1]
        # zero-flux boundaries
        out[0] = out[1] - out[0]
        out[-1] = out[-2] - out[-1]
        return out

    def _step(
        self,
        u: List[float],
        v: List[float],
        t_index: int,
    ) -> Tuple[List[float], List[float]]:
        c = self.config
        frac_weight = c.alpha / (c.alpha + 1.0)
        time_factor = (t_index + 1) ** (c.alpha - 1.0)

        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        next_u: List[float] = [0.0] * len(u)
        next_v: List[float] = [0.0] * len(v)

        for i in range(len(u)):
            activator = math.tanh(u[i] - v[i])
            inhibitor = math.tanh(v[i] - u[i])

            du = (
                c.diffusion_phi * lap_u[i]
                + activator
                - 0.1 * u[i]
            )
            dv = (
                0.5 * c.diffusion_phi * lap_v[i]
                + inhibitor
                - 0.05 * v[i]
            )

            # fractional Euler–like update
            next_u[i] = (
                frac_weight * (u[i] + c.dt * time_factor * du)
                + (1.0 - frac_weight) * u[i]
            )
            next_v[i] = (
                frac_weight * (v[i] + c.dt * time_factor * dv)
                + (1.0 - frac_weight) * v[i]
            )

        return next_u, next_v

    def _prime_like(self, x: float, scale: float) -> int:
        # map a continuous peak index into a positive integer
        n = max(2, int(round(scale * x)))
        while self._is_prime(n) is False:
            n += 1
        return n

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(math.isqrt(n))
        f = 3
        while f <= r:
            if n % f == 0:
                return False
            f += 2
        return True

    @staticmethod
    def _r_metric(xs: List[float]) -> float:
        if not xs:
            return float("nan")
        mean = sum(xs) / len(xs)
        if mean == 0.0:
            return float("inf")
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        return math.sqrt(var) / mean

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> List[Tuple[List[int], dict]]:
        c = self.config
        n = c.n_cells

        u = [0.0] * n
        v = [0.0] * n

        # localized bump in the center plus small noise
        center = n // 2
        for i in range(n):
            dist = abs(i - center)
            u[i] = math.exp(-(dist ** 2) / 40.0) + 0.02 * self.rng.uniform(-1.0, 1.0)
            v[i] = 0.02 * self.rng.uniform(-1.0, 1.0)

        peak_indices: List[int] = []
        peak_values: List[float] = []

        for t in range(c.n_steps):
            u, v = self._step(u, v, t)
            # track a single probe line (the center cell)
            val = u[center]
            if len(peak_values) >= 2:
                # simple local maximum detection in time
                if peak_values[-2] < peak_values[-1] > val and peak_values[-1] > c.peak_threshold:
                    idx = len(peak_values) - 1
                    peak_indices.append(idx)
            peak_values.append(val)

        # map peaks to “primes”
        if peak_indices:
            scale = 11.0  # pushes clusters into familiar two-digit territory
        else:
            scale = 11.0

        primes = [self._prime_like(i, scale) for i in peak_indices]

        # cluster consecutive primes with small gaps
        clusters: List[Tuple[List[int], dict]] = []
        if not primes:
            return clusters

        current: List[int] = [primes[0]]
        for p in primes[1:]:
            if p - current[-1] <= c.cluster_gap_max:
                current.append(p)
            else:
                if len(current) >= c.cluster_min_size:
                    clusters.append(self._cluster_meta(current))
                current = [p]

        if len(current) >= c.cluster_min_size:
            clusters.append(self._cluster_meta(current))

        return clusters

    def _cluster_meta(self, cluster: List[int]) -> Tuple[List[int], dict]:
        r_val = self._r_metric([float(x) for x in cluster])
        meta = {
            "length": len(cluster),
            "span": cluster[-1] - cluster[0],
            "r_metric": r_val,
        }
        return cluster, meta


if __name__ == "__main__":
    reactor = SophiaPrimeReactor()
    clusters = reactor.run()
    print("Found", len(clusters), "clusters")
    for primes, meta in clusters[:5]:
        print("cluster:", primes, "meta:", meta)
