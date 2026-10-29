"""Urban fire: an ``AreaHazard`` whose radius grows while unchecked and shrinks once contained.

Growth is linear up to a severity-capped maximum radius; the fire is marked contained once enough engines
have reached its ignition incident (checked via ``WorldView.engines_on_scene``, answered by the running
simulation -- see ``sim/engine_incidents.py``), after which it burns down over about an hour. The ignition
itself is emitted once as a single fire incident at the hazard's start time.
"""
from __future__ import annotations

import math

from sim.hazards.base import AreaHazard, HazardContext, IncidentSpec, WorldView, ellipse_polygon


class Fire(AreaHazard):
    kind = "fire"
    title = "Urban fire"
    band_factor = 0.45

    def __init__(self, lat: float, lon: float, severity: int = 3, start_min: float = 0.0, name: str = ""):
        super().__init__(lat, lon, start_min, name or f"Fire (severity {severity})")
        self.sev = int(min(5, max(1, severity)))
        self.contained_at: float | None = None
        self._emitted = False

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        self.F = ctx.A["hazards"]["fire"]
        toward = math.radians(90.0 - ctx.wind_toward_deg)          # compass bearing -> math angle
        elong = 1.0 + min(0.5, 0.03 * ctx.wind_kmh)
        self._frame(toward, elong)
        self.contained_at = None
        self._emitted = False

    def extents(self, t: float) -> tuple[float, float]:
        F, s = self.F, self.sev - 1
        dt = t - self.start_min
        if dt < 0:
            return 0.0, 0.0
        rmax = F["max_radius_m_by_severity"][s]
        r = min(rmax, F["start_radius_m"] + F["growth_m_per_min_by_severity"][s] * dt)
        end = self.contained_at if self.contained_at is not None else self.start_min + F["burnout_min_by_severity"][s]
        if t > end:                                   # dying down over ~60 minutes
            r_end = min(rmax, F["start_radius_m"] + F["growth_m_per_min_by_severity"][s] * (end - self.start_min))
            r = r_end * max(0.0, 1.0 - (t - end) / 60.0)
        if r < 15:
            return 0.0, 0.0
        return r, r + F["smoke_band_m"]

    def step(self, t_min: float, world: WorldView) -> None:
        if self.contained_at is None and t_min >= self.start_min and world.engines_on_scene(self.name) >= self.F["contained_after_engines"]:
            self.contained_at = t_min
        super().step(t_min, world)

    def incidents(self, t0: float, t1: float) -> list[IncidentSpec]:
        if not self._emitted and t0 <= self.start_min < t1 + 1e-9:
            self._emitted = True
            return [IncidentSpec("fire", self.node, min(3, max(1, (self.sev + 1) // 2)), self.start_min, self.name,
                                 needs=min(3, max(1, (self.sev + 1) // 2)))]
        return []

    def footprint(self) -> list[dict]:
        if self.core <= 0:
            return []
        c = self.city
        a_core, b_core = self.core * self.elong, self.core
        a_band, b_band = self.band * self.elong, self.band
        return [{"type": "polygon", "kind": "fire", "level": 1, "coords": ellipse_polygon(c, *self.cxy, a_band, b_band, self.theta)},
                {"type": "polygon", "kind": "fire", "level": 3, "coords": ellipse_polygon(c, *self.cxy, a_core, b_core, self.theta)}]

    def info(self) -> str:
        return f"{self.name}: radius {self.core:.0f} m" + (" (contained)" if self.contained_at is not None else "")
