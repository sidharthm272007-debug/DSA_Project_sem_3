"""Gas leak / chemical release: a wind-driven teardrop plume, not a simple circle.

Unlike the other area hazards, the footprint is asymmetric: short upwind of the source, long downwind,
narrow across -- modelled as separate "effective distance" scalings ahead of and behind the source along the
wind axis. Plume length grows while the leak is active and shrinks once it stops.
"""
from __future__ import annotations

import math

import numpy as np

from sim.hazards.base import AreaHazard, HazardContext, ellipse_polygon


class GasLeak(AreaHazard):
    kind = "plume"
    title = "Gas leak / chemical release"
    band_factor = 0.3

    def __init__(self, lat: float, lon: float, severity: int = 3, start_min: float = 0.0, name: str = ""):
        super().__init__(lat, lon, start_min, name or f"Gas leak (severity {severity})")
        self.sev = int(min(5, max(1, severity)))

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        self.P = ctx.A["hazards"]["plume"]
        self.theta_w = math.radians(90.0 - ctx.wind_toward_deg)
        self._frame(self.theta_w, 1.0)
        # teardrop: long downwind, short upwind, narrow across
        dx, dy = self.city.mid_xy[:, 0] - self.cxy[0], self.city.mid_xy[:, 1] - self.cxy[1]
        self._pe = (dx * math.cos(self.theta_w) + dy * math.sin(self.theta_w), -dx * math.sin(self.theta_w) + dy * math.cos(self.theta_w))
        nx, ny = self.city.xy[:, 0] - self.cxy[0], self.city.xy[:, 1] - self.cxy[1]
        self._pn = (nx * math.cos(self.theta_w) + ny * math.sin(self.theta_w), -nx * math.sin(self.theta_w) + ny * math.cos(self.theta_w))
        self.L = 0.0

    def extents(self, t: float) -> tuple[float, float]:
        P, s = self.P, self.sev - 1
        dt = t - self.start_min
        if dt < 0:
            return 0.0, 0.0
        lmax = 1000.0 * P["length_km_by_severity"][s]
        leak_end = P["leak_duration_min_by_severity"][s]
        L = min(lmax, 1000.0 * P["growth_km_per_h"] * dt / 60.0)
        if dt > leak_end:
            L = min(lmax, 1000.0 * P["growth_km_per_h"] * leak_end / 60.0) * max(0.0, 1.0 - (dt - leak_end) / 60.0)
        self.L = L
        if L < 40:
            return 0.0, 0.0
        self._recompute(L)
        return 1.0, 1.5

    def _recompute(self, L: float) -> None:
        asp = self.P["aspect"]
        for arrs, name in ((self._pe, "d_e"), (self._pn, "d_n")):
            x, y = arrs
            semi = np.where(x >= 0, L, 0.15 * L)
            setattr(self, name, np.hypot(x / semi, y / (asp * L)))

    def footprint(self) -> list[dict]:
        if self.core <= 0:
            return []
        L, asp = self.L, self.P["aspect"]
        c = self.city
        cx = self.cxy[0] + 0.425 * L * math.cos(self.theta_w)
        cy = self.cxy[1] + 0.425 * L * math.sin(self.theta_w)
        return [{"type": "polygon", "kind": "plume", "level": 1, "coords": ellipse_polygon(c, cx, cy, 0.6 * L, 1.5 * asp * L * 0.55, self.theta_w)},
                {"type": "polygon", "kind": "plume", "level": 3, "coords": ellipse_polygon(c, cx, cy, 0.575 * L, asp * L * 0.55, self.theta_w)}]

    def info(self) -> str:
        return f"{self.name}: plume {self.L / 1000:.1f} km downwind"
