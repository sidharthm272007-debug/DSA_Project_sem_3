"""Custom event: the catch-all for anything not covered by a named hazard type -- a building collapse, a
crowd surge, a protest, a cordon. A circle or ellipse with a chosen closed/slowed effect and an optional
one-time casualty count, left fully parameterised rather than physics-modelled.
"""
from __future__ import annotations

import math

import numpy as np

from sim.hazards.base import AreaHazard, HazardContext, IncidentSpec, ellipse_polygon


class CustomEvent(AreaHazard):
    """Anything else: building collapse, crowd surge, protest, cordon... A circle/ellipse with a chosen effect."""

    kind = "custom"
    title = "Custom event"

    def __init__(self, lat: float, lon: float, radius_m: float = 300.0, closed: bool = True, slow_factor: float = 0.3,
                 start_min: float = 0.0, duration_min: float = 240.0, growth_m_per_min: float = 0.0, casualties: int = 0,
                 aspect: float = 1.0, angle_deg: float = 0.0, name: str = "Custom event"):
        super().__init__(lat, lon, start_min, name)
        self.r, self.closed, self.slow, self.dur, self.grow = radius_m, closed, slow_factor, duration_min, growth_m_per_min
        self.casualties, self.aspect, self.angle = casualties, max(1.0, aspect), math.radians(angle_deg)
        self.band_factor = slow_factor
        self._emitted = False

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        self._frame(self.angle, self.aspect)
        self._emitted = False

    def extents(self, t: float) -> tuple[float, float]:
        dt = t - self.start_min
        if dt < 0 or dt > self.dur:
            return 0.0, 0.0
        r = min(self.r, 60.0 + self.grow * dt) if self.grow > 0 else self.r
        if self.closed:
            return r, r * 1.5
        return 1e-6, r                        # slowed everywhere inside, nothing closed

    def incidents(self, t0: float, t1: float) -> list[IncidentSpec]:
        if self._emitted or self.casualties <= 0 or not (t0 <= self.start_min < t1 + 1e-9):
            return []
        self._emitted = True
        rng = self._rng(11)
        mix = self.ctx.A["incidents"]["mix"]["custom"]
        kinds, p = list(mix), np.array(list(mix.values()), dtype=float)
        near = np.flatnonzero(self.d_n <= max(self.r, 100.0))
        if len(near) == 0:
            near = np.array([self.node])
        out = []
        for i in range(self.casualties):
            out.append(IncidentSpec(str(rng.choice(kinds, p=p / p.sum())), int(rng.choice(near)), int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])),
                                    self.start_min + float(rng.uniform(0, 20)), self.name))
        return out

    def footprint(self) -> list[dict]:
        if self.core <= 0:
            return []
        r = self.band
        return [{"type": "polygon", "kind": "custom", "level": 1, "coords": ellipse_polygon(self.city, *self.cxy, r * self.aspect, r, self.angle)},
                {"type": "polygon", "kind": "custom", "level": 3, "coords": ellipse_polygon(self.city, *self.cxy, self.core * self.aspect, self.core, self.angle)}]
