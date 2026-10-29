"""Shared hazard machinery: the ``Hazard`` contract every hazard type implements, the ``AreaHazard`` base for
anything that is "a growing/shrinking closed core + slowed band around a point" (fire, gas leak, flood, custom
event all build on this), and the small support types (``IncidentSpec``, ``HazardContext``, ``WorldView``) plus
the circle/ellipse footprint helpers used for the map.

A hazard exposes, at the current simulation time:
    factor()        per-road speed factor, 0 = closed (float32, shape (E,))
    repairable()    which closed roads are DEBRIS a crew can clear (bool, shape (E,))
    exposure()      per-node exposure 0..1 (drives incident generation and "people affected")
    outages()       {facility id: capacity 0..1} (hospitals damaged by the event)
    footprint()     map primitives (circles / polygons)
    incidents()     hazard-specific incidents to create (e.g. a fire ignition, casualties at a collapse)
    spawn()         child hazards (fires after an earthquake)
All coefficients come from assumptions.yaml and are labelled placeholders; results are a rehearsal, not a
forecast.

Each concrete hazard type lives in its own file next to this one (earthquake.py, fire.py, gas_leak.py,
flood.py, custom_event.py, structure_failure.py, accident.py, storm.py) -- that is the file to open for an
in-depth look at how a specific disaster is modelled. ``sim/hazards/__init__.py`` re-exports everything so
the rest of the codebase can keep writing ``from sim.hazards import Fire`` etc.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from sim.city import CityData


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
