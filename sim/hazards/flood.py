"""Flood: rises along a named river/lake (distance-to-water drives the footprint) or, for a flash flood, around
a point. Zone-based with a hydrograph-shaped rise/hold/recede over time -- there is no terrain/elevation model.

Bridges and flyovers stay open below a severity threshold (an elevated deck doesn't flood like a surface
street); above it, they close like any other road.
"""
from __future__ import annotations

import numpy as np

from sim.hazards.base import AreaHazard, Hazard, HazardContext


class FloodZone(AreaHazard):
    """Flooding around a river/lake (or a point). Zone-based: extent from a hydrograph; no terrain model."""

    kind = "flood_zone"
    title = "Flood"
    band_factor = 0.35

    def __init__(self, water: str | None, severity: int = 3, start_min: float = 0.0, lat: float | None = None,
                 lon: float | None = None, name: str = ""):
        super().__init__(lat if lat is not None else 0.0, lon if lon is not None else 0.0, start_min, name or f"Flood (severity {severity})")
        self.water, self.sev, self.point = water, int(min(5, max(1, severity))), lat is not None

    def reset(self, ctx: HazardContext) -> None:
        Hazard.reset(self, ctx)
        self.F = ctx.A["hazards"]["flood_zone"]
        c = ctx.city
        if self.point:
            x0, y0 = c.proj.to_xy(self.lat, self.lon)
            self.cxy = (float(x0), float(y0))
            self.node = c.snap(self.lat, self.lon)[0]
            self.d_e = np.hypot(c.mid_xy[:, 0] - self.cxy[0], c.mid_xy[:, 1] - self.cxy[1])
            self.d_n = np.hypot(c.xy[:, 0] - self.cxy[0], c.xy[:, 1] - self.cxy[1])
        else:
            self.d_e = c.dist_to_water(self.water)
            self.d_n = self._node_dist()
            mid = int(np.argmin(self.d_e))
            self.lat, self.lon = float(c.g.lat[c.g.edge_tails[mid]]), float(c.g.lon[c.g.edge_tails[mid]])
            x0, y0 = c.proj.to_xy(self.lat, self.lon)
            self.cxy = (float(x0), float(y0))
            self.node = c.snap(self.lat, self.lon)[0]
        self.ext = 0.0

    def _node_dist(self) -> np.ndarray:
        c = self.city
        out = np.full(self.N, np.inf)
        # nodes are only needed for exposure: approximate with the distance of their nearest incident edge
        e_d = self.d_e
        np.minimum.at(out, c.g.edge_tails, e_d)
        return out

    def extents(self, t: float) -> tuple[float, float]:
        F = self.F
        peak = F["extent_m_by_severity"][self.sev - 1]
        dt = t - self.start_min
        if dt < 0:
            return 0.0, 0.0
        hold = 90.0
        if dt <= F["rise_min"]:
            e = peak * dt / F["rise_min"]
        elif dt <= F["rise_min"] + hold:
            e = peak
        else:
            e = peak * max(0.0, 1.0 - (dt - F["rise_min"] - hold) / F["recede_min"])
        self.ext = e
        if e < 25:
            return 0.0, 0.0
        return 0.6 * e, e

    def _mask_core(self) -> np.ndarray:
        m = self.d_e <= self.core
        if self.sev < self.F["bridge_fail_severity"]:
            m &= ~(self.city.is_bridge | self.city.is_flyover)         # elevated decks stay open
        return m

    def factor(self) -> np.ndarray | None:
        if self.core <= 0:
            return None
        f = np.ones(self.E, dtype=np.float32)
        band = (self.d_e > self.core) & (self.d_e <= self.band)
        band &= ~(self.city.is_flyover)
        f[band] = self.band_factor
        f[self._mask_core()] = 0.0
        return f

    def footprint(self) -> list[dict]:
        if self.core <= 0:
            return []
        c = self.city
        out = []
        if self.point:
            return [{"type": "circle", "kind": "flood", "level": 1, "lat": self.lat, "lon": self.lon, "radius_m": self.band},
                    {"type": "circle", "kind": "flood", "level": 3, "lat": self.lat, "lon": self.lon, "radius_m": self.core}]
        keys = [self.water] if self.water else list(c.water)
        for k in keys:
            pts = c.water[k]
            step = max(1, len(pts) // 14)
            for la, lo in pts[::step]:
                out.append({"type": "circle", "kind": "flood", "level": 1, "lat": float(la), "lon": float(lo), "radius_m": self.band})
        return out

    def info(self) -> str:
        return f"{self.name}: water reaches ~{self.ext:.0f} m from {self.water or 'the source'}"
