"""A snapshot of the world at one moment: which roads are closed or slowed, who is exposed, which hospitals are out.

Used by route advice. Three ways to get one: a normal day, hazards stepped to a chosen time (no emergency response, no
debris clearing: the raw damage), or any moment replayed from a Disaster-lab run (which includes reopened roads).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.city import CityData
from sim.hazards import HazardContext, WorldView


@dataclass
class WorldState:
    label: str
    F: np.ndarray                     # per-edge capacity factor: 0 closed, 1 normal
    expo: np.ndarray                  # per-node exposure to the hazard, 0..1
    bg: float                         # traffic x weather speed factor
    closed_fac: set[int] = field(default_factory=set)
    prims: list[dict] = field(default_factory=list)
    t: float | None = None            # minutes since the event began (None: normal day)

    @property
    def closed_edges(self) -> np.ndarray:
        return np.flatnonzero(self.F == 0)


def normal_state(city: CityData, bg: float, label: str = "Normal day") -> WorldState:
    return WorldState(label, np.ones(city.g.n_edges, dtype=np.float32), np.zeros(city.g.n_nodes), bg)


def hazard_state(city: CityData, A: dict, hazards: list, bg: float, t: float, seed: int = 1, wind: tuple[float, float] = (90.0, 12.0),
                 label: str = "") -> WorldState:
    """Step ``hazards`` from minute 0 to minute ``t`` (every 10 min, like the simulator) and combine their effects. No response, no repairs."""
    ctx = HazardContext(city, A, seed, wind[0], wind[1])
    world = WorldView()
    hs = list(hazards)
    for h in hs:
        h.reset(ctx)
    steps = sorted(set(list(np.arange(0.0, float(t), 10.0)) + [float(t)]))
    for tt in steps:
        for h in hs:
            h.step(float(tt), world)
        for h in list(hs):
            for kid in h.spawn():
                kid.reset(ctx)
                kid.step(float(tt), world)
                hs.append(kid)
    E, N = city.g.n_edges, city.g.n_nodes
    F, expo, out = np.ones(E, dtype=np.float32), np.zeros(N), {}
    for h in hs:
        f = h.factor()
        if f is not None:
            F *= f
        e = h.exposure()
        if e is not None:
            expo = np.maximum(expo, e)
        for fid, cap in h.outages().items():
            out[fid] = min(cap, out.get(fid, 1.0))
    return WorldState(label or "Scenario", F, expo, bg, {fid for fid, cap in out.items() if cap <= 0}, [p for h in hs for p in h.footprint()], float(t))


def state_from_frame(city: CityData, res, fi: int) -> WorldState:
    """Replay moment ``fi`` of a Disaster-lab run (includes roads the crews had already reopened)."""
    fr = res.frames[fi]
    F = np.ones(city.g.n_edges, dtype=np.float32)
    F[fr["F_idx"]] = fr["F_val"].astype(np.float32)
    expo = np.zeros(city.g.n_nodes)
    expo[fr["expo_idx"]] = fr["expo_val"].astype(np.float64)
    return WorldState(res.scenario_name, F, expo, float(res.bg), set(fr["closed_fac"]), list(fr["prims"]), float(fr["t"]))
