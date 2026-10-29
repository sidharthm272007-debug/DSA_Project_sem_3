"""Severe storm / cyclone: city-wide, not localised. Every road independently rolls whether a tree falls on it
(weighted by road type and storm severity), with a random onset time spread across the storm's build-up so
failures stagger in rather than all happening at once. Debris clears faster than quake/fire damage
(``repair_mult`` below 1), reflecting that a fallen tree is quicker work than a collapsed structure.
"""
from __future__ import annotations

import numpy as np

from sim.hazards.base import Hazard, HazardContext
from sim.rand import cell_uniform


class Storm(Hazard):
    kind = "storm"
    title = "Severe storm / cyclone"
    repair_mult = 0.6

    def __init__(self, severity: int = 3, start_min: float = 0.0, duration_min: float = 360.0, name: str = ""):
        super().__init__(name or f"Storm (severity {severity})", start_min)
        self.sev, self.duration = int(min(5, max(1, severity))), duration_min

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        S = ctx.A["hazards"]["storm"]
        c = ctx.city
        w = np.where(c.is_residential, S["tree_fall_weights"]["residential"], S["tree_fall_weights"]["arterial"])
        p_km = S["tree_fall_p_per_km_by_severity"][self.sev - 1] * w
        p_edge = 1.0 - (1.0 - np.clip(p_km, 0, 0.9)) ** np.maximum(c.edge_km, 1e-3)
        cx, cy = self._cells()
        u = cell_uniform(ctx.seed * 17 + 5, cx, cy)
        self.fallen = u < p_edge
        self.onset = self.start_min + self._rng(3).uniform(0, 0.6 * self.duration, self.E)
        self.sf = S["speed_factor_by_severity"][self.sev - 1]

    def active(self) -> bool:
        return self.start_min <= self.t <= self.start_min + self.duration + 60

    def _ramp(self) -> float:
        dt = self.t - self.start_min
        if dt < 0:
            return 0.0
        if dt < 30:
            return dt / 30.0
        if dt > self.duration:
            return max(0.0, 1.0 - (dt - self.duration) / 60.0)
        return 1.0

    def factor(self) -> np.ndarray | None:
        if not self.active():
            return None
        f = np.full(self.E, 1.0 - (1.0 - self.sf) * self._ramp(), dtype=np.float32)
        f[self.fallen & (self.t >= self.onset)] = 0.0
        return f

    def repairable(self) -> np.ndarray | None:
        return (self.fallen & (self.t >= self.onset)) if self.active() else None

    def exposure(self) -> np.ndarray | None:
        return np.full(self.N, 0.15 * self._ramp()) if self.active() else None

    def footprint(self) -> list[dict]:
        return []
