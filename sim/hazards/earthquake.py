"""Earthquake: shaking damage from a point epicenter, with aftershocks and secondary fires.

Each simulated shock (mainshock + any aftershocks strong enough to matter) independently rolls, road by road,
whether that stretch collapses -- probability rises with Gaussian-decayed shaking intensity, the road's own
vulnerability (bridges and flyovers fail more easily than ordinary residential streets), and the road's length
(a longer stretch has more chances to fail). Hospitals near the epicenter can lose power/structural integrity
outright. A strong-enough quake also spawns a handful of ``Fire`` hazards (see fire.py) a few minutes later,
weighted towards the most heavily shaken, most populated areas.
"""
from __future__ import annotations

import math

import numpy as np

from sim.hazards.base import Hazard, HazardContext, IncidentSpec, WorldView
from sim.hazards.fire import Fire
from sim.rand import cell_uniform


class Earthquake(Hazard):
    kind = "earthquake"
    title = "Earthquake"

    def __init__(self, lat: float, lon: float, magnitude: float = 6.0, start_min: float = 0.0,
                 secondary_fires: bool = True, name: str = ""):
        super().__init__(name or f"M{magnitude:.1f} earthquake", start_min)
        self.lat, self.lon, self.M, self.secondary_fires = lat, lon, float(magnitude), secondary_fires

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        c, q = ctx.city, ctx.A["hazards"]["earthquake"]
        ex, ey = c.proj.to_xy(self.lat, self.lon)
        self.exy = (float(ex), float(ey))
        self.r_edge = np.hypot(c.mid_xy[:, 0] - self.exy[0], c.mid_xy[:, 1] - self.exy[1])
        self.r_node = np.hypot(c.xy[:, 0] - self.exy[0], c.xy[:, 1] - self.exy[1])
        self.felt_m = 1000.0 * q["felt_radius_km_at_m6"] * 10 ** (q["radius_growth_per_magnitude"] * (self.M - 6))
        self.sigma = self.felt_m / 2.0
        self.intensity_edge = np.exp(-(self.r_edge / self.sigma) ** 2)
        self.intensity_node = np.exp(-(self.r_node / self.sigma) ** 2)
        vuln = np.ones(self.E)
        vuln[c.is_residential] = q["vulnerability"]["residential"]
        vuln[c.is_bridge] = q["vulnerability"]["bridge"]
        vuln[c.is_flyover] = q["vulnerability"]["flyover"]
        self.vuln = vuln
        self.cx, self.cy = self._cells()
        self.shocks: list[tuple[float, np.ndarray]] = []
        self._new_shock(self.start_min, self.M, 0)
        for k, (dt, dM) in enumerate(q["aftershocks"], start=1):
            if self.M + dM >= 4.5:
                self._new_shock(self.start_min + dt, self.M + dM, k)
        self.blocked = np.zeros(self.E, dtype=bool)
        rng = self._rng(7)
        # hospitals damaged by strong shaking
        self._out: dict[int, float] = {}
        for f in c.hospitals(3):
            p = q["hospital_outage_p_at_m6"] * 10 ** (0.5 * (self.M - 6)) * math.exp(-(self._dist_xy(c.xy[f.node]) / self.sigma) ** 2)
            if rng.random() < min(0.9, p):
                self._out[f.id] = 0.0
        self._children: list[Hazard] = []
        if self.secondary_fires and self.M >= 5.5:
            n = int(q["secondary_fires"] * min(1.5, 10 ** (0.5 * (self.M - 6))))
            w = c.pop * self.intensity_node
            if w.sum() > 0:
                for i, node in enumerate(rng.choice(self.N, size=max(1, n), replace=False, p=w / w.sum())):
                    la, lo = float(c.g.lat[node]), float(c.g.lon[node])
                    self._children.append(Fire(la, lo, severity=int(rng.integers(2, 4)),
                                               start_min=self.start_min + float(rng.uniform(*q["fire_delay_min"])), name=f"Fire after quake #{i + 1}"))

    def _dist_xy(self, p) -> float:
        return float(math.hypot(p[0] - self.exy[0], p[1] - self.exy[1]))

    def _new_shock(self, t0: float, mag: float, k: int) -> None:
        q = self.ctx.A["hazards"]["earthquake"]
        p_km = min(0.95, q["p_per_km_at_m6"] * 10 ** (q["growth_per_magnitude"] * (mag - 6))) * self.intensity_edge * self.vuln
        p_edge = 1.0 - (1.0 - np.clip(p_km, 0, 0.95)) ** np.maximum(self.city.edge_km, 1e-3)
        u = cell_uniform(self.ctx.seed * 131 + 977 * (k + 1), self.cx, self.cy)
        self.shocks.append((t0, u < p_edge))

    def spawn(self) -> list[Hazard]:
        if self._spawned:
            return []
        self._spawned = True
        return self._children

    def step(self, t_min: float, world: WorldView) -> None:
        super().step(t_min, world)
        blocked = np.zeros(self.E, dtype=bool)
        for t0, mask in self.shocks:
            if t_min >= t0:
                blocked |= mask
        self.blocked = blocked

    def factor(self) -> np.ndarray | None:
        if not self.active():
            return None
        recover = max(0.0, 1.0 - (self.t - self.start_min) / 360.0)
        f = (1.0 - 0.4 * self.intensity_edge * recover).astype(np.float32)
        f[self.blocked] = 0.0
        return f

    def repairable(self) -> np.ndarray | None:
        return self.blocked if self.active() else None

    def exposure(self) -> np.ndarray | None:
        if not self.active():
            return None
        rec = max(0.15, 1.0 - (self.t - self.start_min) / 720.0)
        return self.intensity_node * rec

    def outages(self) -> dict[int, float]:
        return dict(self._out) if self.active() else {}

    def incidents(self, t0: float, t1: float) -> list[IncidentSpec]:
        return []

    def footprint(self) -> list[dict]:
        if not self.active():
            return []
        lat, lon = self.lat, self.lon
        return [{"type": "circle", "lat": lat, "lon": lon, "radius_m": self.sigma * m, "kind": "quake", "level": lvl}
                for m, lvl in ((0.6, 3), (1.1, 2), (1.7, 1))]

    def location(self) -> tuple[float, float]:
        return self.lat, self.lon

    def info(self) -> str:
        return f"{self.name}: felt to ~{self.felt_m / 1000:.0f} km"
