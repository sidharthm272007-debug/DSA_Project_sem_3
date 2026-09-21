"""Hazard events. Every disaster is a footprint x time profile x effect on roads and facilities, so one engine
handles earthquakes, fires, gas leaks, floods, structure failures, accidents, storms and custom events.

A hazard exposes, at the current simulation time:
    factor()        per-road speed factor, 0 = closed (float32, shape (E,))
    repairable()    which closed roads are DEBRIS a crew can clear (bool, shape (E,))
    exposure()      per-node exposure 0..1 (drives incident generation and "people affected")
    outages()       {facility id: capacity 0..1} (hospitals damaged by the event)
    footprint()     map primitives (circles / polygons)
    incidents()     hazard-specific incidents to create (e.g. a fire ignition, casualties at a collapse)
    spawn()         child hazards (fires after an earthquake)
All coefficients come from assumptions.yaml and are labelled placeholders; results are a rehearsal, not a forecast.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from sim.city import CityData
from sim.rand import cell_uniform


@dataclass
class IncidentSpec:
    kind: str                  # medical | fire | rescue
    node: int
    severity: int              # 1..3
    t_min: float
    tag: str = ""              # links an incident back to its hazard
    needs: int = 1             # units required (fire engines)


@dataclass
class HazardContext:
    city: CityData
    A: dict
    seed: int
    wind_toward_deg: float = 90.0     # direction the wind blows TOWARDS (degrees clockwise from north)
    wind_kmh: float = 12.0


class WorldView:
    """What hazards may ask the running simulation (coupling response -> hazard)."""

    def engines_on_scene(self, tag: str) -> int:        # noqa: D401
        return 0


def circle_polygon(city: CityData, lat: float, lon: float, r_m: float, n: int = 40) -> list[list[float]]:
    x0, y0 = city.proj.to_xy(lat, lon)
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    lat_, lon_ = city.proj.to_latlon(float(x0) + r_m * np.cos(a), float(y0) + r_m * np.sin(a))
    return [[float(o), float(l)] for l, o in zip(lat_, lon_)]


def ellipse_polygon(city: CityData, cx: float, cy: float, a_m: float, b_m: float, theta: float, n: int = 48) -> list[list[float]]:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ex, ey = a_m * np.cos(t), b_m * np.sin(t)
    x = cx + ex * math.cos(theta) - ey * math.sin(theta)
    y = cy + ex * math.sin(theta) + ey * math.cos(theta)
    lat_, lon_ = city.proj.to_latlon(x, y)
    return [[float(o), float(l)] for l, o in zip(lat_, lon_)]


class Hazard:
    kind = "hazard"
    title = "Hazard"
    repair_mult = 1.0            # debris clearing time multiplier (trees clear faster than collapsed buildings)

    def __init__(self, name: str = "", start_min: float = 0.0):
        self.name = name or self.title
        self.start_min = float(start_min)
        self.t = 0.0
        self._spawned = False

    # lifecycle ----------------------------------------------------------------------------------
    def reset(self, ctx: HazardContext) -> None:
        self.ctx = ctx
        self.city = ctx.city
        self.E = ctx.city.g.n_edges
        self.N = ctx.city.g.n_nodes

    def step(self, t_min: float, world: WorldView) -> None:
        self.t = float(t_min)

    def active(self) -> bool:
        return self.t >= self.start_min

    # state ---------------------------------------------------------------------------------------
    def factor(self) -> np.ndarray | None:
        return None

    def repairable(self) -> np.ndarray | None:
        return None

    def exposure(self) -> np.ndarray | None:
        return None

    def outages(self) -> dict[int, float]:
        return {}

    def footprint(self) -> list[dict]:
        return []

    def incidents(self, t0: float, t1: float) -> list[IncidentSpec]:
        return []

    def spawn(self) -> list["Hazard"]:
        return []

    def info(self) -> str:
        return self.name

    def location(self) -> tuple[float, float] | None:
        return None

    # helpers -------------------------------------------------------------------------------------
    def _rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.ctx.seed * 1_000_003 + salt + (zlib.crc32(self.name.encode()) & 0xFFFF))

    def _cells(self, cell_m: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
        m = self.city.mid_xy
        return np.floor(m[:, 0] / cell_m).astype(np.int64), np.floor(m[:, 1] / cell_m).astype(np.int64)


# ======================================================================================================
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


# ======================================================================================================
class AreaHazard(Hazard):
    """Closed core + slowed band around a (possibly wind-stretched) area that grows and shrinks over time."""

    band_factor = 0.4
    repairable_core = False

    def __init__(self, lat: float, lon: float, start_min: float = 0.0, name: str = ""):
        super().__init__(name, start_min)
        self.lat, self.lon = lat, lon
        self.core = 0.0
        self.band = 0.0
        self._f_cache: np.ndarray | None = None

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        c = ctx.city
        x0, y0 = c.proj.to_xy(self.lat, self.lon)
        self.cxy = (float(x0), float(y0))
        self.node = c.snap(self.lat, self.lon)[0]
        self._frame()

    def _frame(self, theta: float = 0.0, elong: float = 1.0) -> None:
        c = self.city
        # theta: direction of the long axis, radians counter-clockwise from +x (east)
        dx, dy = c.mid_xy[:, 0] - self.cxy[0], c.mid_xy[:, 1] - self.cxy[1]
        self.d_e = np.hypot((dx * math.cos(theta) + dy * math.sin(theta)) / elong, -dx * math.sin(theta) + dy * math.cos(theta))
        nx, ny = c.xy[:, 0] - self.cxy[0], c.xy[:, 1] - self.cxy[1]
        self.d_n = np.hypot((nx * math.cos(theta) + ny * math.sin(theta)) / elong, -nx * math.sin(theta) + ny * math.cos(theta))
        self.theta, self.elong = theta, elong

    def extents(self, t_min: float) -> tuple[float, float]:
        """(core, band) in the hazard's own units; core 0 => nothing happening."""
        raise NotImplementedError

    def step(self, t_min: float, world: WorldView) -> None:
        super().step(t_min, world)
        self.core, self.band = self.extents(t_min)
        self._f_cache = None

    def _mask_core(self) -> np.ndarray:
        return self.d_e <= self.core

    def factor(self) -> np.ndarray | None:
        if self.core <= 0:
            return None
        if self._f_cache is None:
            f = np.ones(self.E, dtype=np.float32)
            f[(self.d_e > self.core) & (self.d_e <= self.band)] = self.band_factor
            f[self._mask_core()] = 0.0
            self._f_cache = f
        return self._f_cache

    def repairable(self) -> np.ndarray | None:
        return self._mask_core() if (self.repairable_core and self.core > 0) else None

    def exposure(self) -> np.ndarray | None:
        if self.core <= 0:
            return None
        e = np.zeros(self.N)
        e[self.d_n <= self.band] = 0.5
        e[self.d_n <= self.core] = 1.0
        return e

    def location(self) -> tuple[float, float]:
        return self.lat, self.lon


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
        keys = [self.water] if self.water else list(c.water)
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


# ======================================================================================================
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


class Accident(Hazard):
    kind = "accident"
    title = "Major road accident"

    def __init__(self, lat: float, lon: float, severity: int = 3, start_min: float = 0.0, duration_min: float = 120.0, name: str = ""):
        super().__init__(name or f"Road accident (severity {severity})", start_min)
        self.lat, self.lon, self.sev, self.duration = lat, lon, int(min(5, max(1, severity))), duration_min

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        c, g = ctx.city, ctx.city.g
        A = ctx.A["hazards"]["accident"]
        node = c.snap(self.lat, self.lon)[0]
        self.node = node
        k = A["blocked_edges_by_severity"][self.sev - 1]
        # block the arcs leaving/entering the impact node along the busiest road, then the next ones along it
        cand = [e for e in list(g.out_edges(node))]
        cand += list(np.flatnonzero(g.indices == node))
        seen: list[int] = []
        frontier = [int(e) for e in cand]
        while frontier and len(seen) < 2 * k:
            e = frontier.pop(0)
            if e in seen:
                continue
            seen.append(e)
            r = int(g.rev[e])
            if r >= 0 and r not in seen:
                seen.append(r)
            head = int(g.indices[e])
            nxt = [int(x) for x in g.out_edges(head) if int(g.indices[x]) != int(g.edge_tails[e])]
            same = [x for x in nxt if g.way_id[x] == g.way_id[e]]
            frontier += same[:1]
        self.blocked = np.array(seen[: max(2, 2 * k)], dtype=np.int64)
        self.mask = np.zeros(self.E, dtype=bool)
        self.mask[self.blocked] = True
        # queue spillback upstream: reverse Dijkstra by length from the blocked edges' tails
        self.qf = np.ones(self.E, dtype=np.float32)
        L, qmin = A["queue_length_m"], A["queue_min_factor"]
        rg, emap = g.reverse()
        import heapq
        dist = {}
        heap = [(0.0, int(g.edge_tails[e])) for e in self.blocked]
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if u in dist or d > L:
                continue
            dist[u] = d
            for j in range(rg.indptr[u], rg.indptr[u + 1]):
                v = int(rg.indices[j])
                if v not in dist:
                    heapq.heappush(heap, (d + float(rg.length_m[j]), v))
                    e = int(emap[j])                        # original arc v -> u carries the queue
                    self.qf[e] = min(self.qf[e], qmin + (1 - qmin) * min(1.0, d / L))
        self._emitted = False

    def active(self) -> bool:
        return self.start_min <= self.t <= self.start_min + self.duration

    def factor(self) -> np.ndarray | None:
        if not self.active():
            return None
        f = self.qf.copy()
        f[self.mask] = 0.0
        return f

    def repairable(self) -> np.ndarray | None:
        return None

    def incidents(self, t0: float, t1: float) -> list[IncidentSpec]:
        if self._emitted or not (t0 <= self.start_min < t1 + 1e-9):
            return []
        self._emitted = True
        n = 1 + self.sev // 2
        return [IncidentSpec("medical", self.node, min(3, 1 + self.sev // 2), self.start_min + i, self.name) for i in range(n)]

    def footprint(self) -> list[dict]:
        if not self.active():
            return []
        return [{"type": "circle", "kind": "accident", "level": 3, "lat": self.lat, "lon": self.lon, "radius_m": 90.0}]

    def location(self) -> tuple[float, float]:
        return self.lat, self.lon


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


HAZARD_CLASSES = {c.kind: c for c in (Earthquake, Fire, GasLeak, FloodZone, CustomEvent, StructureFailure, Accident, Storm)}
