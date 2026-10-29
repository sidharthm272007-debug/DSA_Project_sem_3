"""A specific bridge or flyover collapses or is closed outright -- not a debris-clearable hazard (a collapsed
structure needs reconstruction, not a repair crew), and the only hazard type that targets named edges directly
rather than a geographic radius.
"""
from __future__ import annotations

import numpy as np

from sim.hazards.base import Hazard, HazardContext, IncidentSpec


class StructureFailure(Hazard):
    kind = "structure_failure"
    title = "Bridge / flyover failure or closure"

    def __init__(self, structure: dict | None = None, edges: list[int] | None = None, mode: str = "collapse",
                 start_min: float = 0.0, duration_min: float | None = None, name: str = ""):
        super().__init__(name or f"{mode.title()}: {structure['name'] if structure else 'road'}", start_min)
        self.structure, self.edges_in, self.mode, self.duration = structure, edges, mode, duration_min

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        es = self.edges_in if self.edges_in is not None else self.structure["edges"]
        self.edges = np.array(sorted(set(int(e) for e in es) | {int(r) for e in es if (r := ctx.city.g.rev[e]) >= 0}), dtype=np.int64)
        self.mask = np.zeros(self.E, dtype=bool)
        self.mask[self.edges] = True
        g = ctx.city.g
        self.lat = float(np.mean(g.lat[g.edge_tails[self.edges]]))
        self.lon = float(np.mean(g.lon[g.edge_tails[self.edges]]))
        self.node = ctx.city.snap(self.lat, self.lon)[0]
        self._emitted = False
        self.n_nodes = np.unique(np.concatenate([g.edge_tails[self.edges], g.indices[self.edges]]))

    def active(self) -> bool:
        if self.t < self.start_min:
            return False
        return self.duration is None or self.t <= self.start_min + self.duration

    def factor(self) -> np.ndarray | None:
        if not self.active():
            return None
        f = np.ones(self.E, dtype=np.float32)
        f[self.mask] = self.ctx.A["hazards"]["structure_failure"]["partial_capacity_factor"] if self.mode == "partial" else 0.0
        return f

    def repairable(self) -> np.ndarray | None:
        return None                      # a collapsed or closed structure is not cleared by a debris crew

    def exposure(self) -> np.ndarray | None:
        if not self.active() or self.mode != "collapse":
            return None
        e = np.zeros(self.N)
        d = np.hypot(self.city.xy[:, 0] - self.city.xy[self.node, 0], self.city.xy[:, 1] - self.city.xy[self.node, 1])
        e[d < 300] = 0.7
        return e

    def incidents(self, t0: float, t1: float) -> list[IncidentSpec]:
        if self._emitted or self.mode != "collapse" or not (t0 <= self.start_min < t1 + 1e-9):
            return []
        self._emitted = True
        rng = self._rng(13)
        n = 10 if (self.structure or {}).get("kind") == "flyover" else 6
        return [IncidentSpec(str(rng.choice(["medical", "rescue"], p=[0.5, 0.5])), self.node, int(rng.choice([2, 3], p=[0.6, 0.4])),
                             self.start_min + float(rng.uniform(0, 15)), self.name) for _ in range(n)]

    def footprint(self) -> list[dict]:
        if not self.active():
            return []
        return [{"type": "circle", "kind": "structure", "level": 3, "lat": self.lat, "lon": self.lon, "radius_m": 160.0}]

    def location(self) -> tuple[float, float]:
        return self.lat, self.lon
