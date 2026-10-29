"""The disaster simulator -- orchestrator.

One minute at a time: hazards evolve, incidents appear where people are exposed, units are dispatched over the
current (damaged) road network, re-route when roads close mid-trip, crews clear debris in a connectivity-first
order, and every few minutes we recompute who can still reach a hospital. The fire hazard reacts to whether
engines actually arrive.

This file only holds the scenario/result data shapes and the top-level ``run()`` loop. Each distinct action
the simulator performs lives in its own file, mixed into the ``Simulation`` class below -- go there directly
for an in-depth look at how a specific piece works:

    sim/engine_state.py      hazard -> road/network state update, metrics + replay-frame recording
    sim/engine_incidents.py  turning hazards into 911-style calls, call expiry
    sim/engine_dispatch.py   matching a waiting call to the nearest idle unit
    sim/engine_movement.py   per-minute movement, live re-routing, arrival/stage transitions
    sim/engine_repair.py     repair-crew job assignment (uses the plan from sim/repair.py)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from sim.city import CityData
from sim.engine_dispatch import DispatchMixin
from sim.engine_incidents import IncidentMixin
from sim.engine_movement import Leg, MovementMixin
from sim.engine_repair import RepairMixin
from sim.engine_state import StateMixin
from sim.hazards import Hazard, HazardContext, IncidentSpec, WorldView
from sim.metrics import catchments, compute_access, summarize
from sim.repair import RepairTarget
from sim.response import Incident, Resources, Unit, build_units
from sim.router import Router

__all__ = ["Context", "Scenario", "Leg", "SimResult", "Simulation", "compute_baseline"]


@dataclass
class Context:
    time_of_day: str = "off_peak"       # night | off_peak | peak
    weather: str = "clear"              # clear | light_rain | heavy_rain | storm
    wind_toward_deg: float = 90.0
    wind_kmh: float = 12.0
    start_hour: float = 9.0

    def background(self, A: dict) -> float:
        return A["context"]["time_of_day_speed"][self.time_of_day] * A["context"]["weather_speed"][self.weather]


@dataclass
class Scenario:
    name: str
    hazards: list[Hazard]
    duration_h: float = 12.0
    seed: int = 1
    context: Context = field(default_factory=Context)
    resources: Resources = field(default_factory=Resources)
    incident_scale: float = 1.0
    state_step_min: int = 10


@dataclass
class SimResult:
    scenario_name: str
    times: list[float]
    metrics: list[dict]
    frames: list[dict]
    incidents: list[Incident]
    units: list[Unit]
    legs: list[Leg]
    repair_log: list[dict]
    baseline: dict
    peak: dict
    summary: dict
    runtime_s: float
    hazard_names: list[str]
    hazard_points: list[dict]
    bg: float = 1.0                      # traffic x weather speed factor of the scenario


class Simulation(StateMixin, IncidentMixin, DispatchMixin, MovementMixin, RepairMixin, WorldView):
    """The running simulation. Each base class contributes one action (see the module docstring above); this
    class itself only owns setup (``__init__``), the top-level per-minute loop (``run``), and wrapping the
    results up (``_finish``)."""

    def __init__(self, city: CityData, scenario: Scenario, A: dict, router: Router | None = None, baseline: dict | None = None):
        self.city, self.sc, self.A = city, scenario, A
        self.router = router or Router(city.g)
        seen: dict[str, int] = {}
        for h in scenario.hazards:                       # unique names: incidents are linked to hazards by name
            n = seen.get(h.name, 0)
            seen[h.name] = n + 1
            if n:
                h.name = f"{h.name} #{n + 1}"
        self.hazards: list[Hazard] = list(scenario.hazards)
        self.baseline = baseline
        self.tag_inc: dict[str, Incident] = {}

    # ------------------------------------------------------------------------------------------------
    def run(self, progress=None) -> SimResult:
        t_wall = time.perf_counter()
        city, sc, A, g = self.city, self.sc, self.A, self.city.g
        E = g.n_edges
        self.rng = np.random.default_rng(sc.seed)
        ctx = HazardContext(city, A, sc.seed, sc.context.wind_toward_deg, sc.context.wind_kmh)
        for h in self.hazards:
            h.reset(ctx)
        self.units = build_units(city, sc.resources, A)
        self.incidents: list[Incident] = []
        self.future: list[tuple[float, int, IncidentSpec | tuple]] = []
        self.legs: list[Leg] = []
        self.repair_log: list[dict] = []
        self.repaired = np.zeros(E, dtype=bool)
        self.pair_first = (g.rev < 0) | (np.arange(E) < g.rev)
        self.bg = sc.context.background(A)
        self.frames: list[dict] = []
        self.metrics: list[dict] = []
        self.times: list[float] = []
        self._plan: list[RepairTarget] = []
        self._plan_t = -1e9
        self._plan_n = -1
        self._repairs_since_plan = 0
        self._claimed: set[int] = set()
        self.fire_nodes = [f.node for f in city.of_kind("fire_station")] + [u.node for u in self.units if u.kind == "fire" and u.label == "added"]
        self.T = int(round(sc.duration_h * 60))
        step = sc.state_step_min
        if self.baseline is None:
            self.baseline = compute_baseline(city, self.router, A, self.bg, self.fire_nodes)
        for t in range(0, self.T + 1):
            if t % step == 0:
                self._state_update(t)
                self._plan_incidents(t, t + step)
                if progress:
                    progress(t / max(self.T, 1), f"t = {t // 60}h{t % 60:02d}m")
            self._release_future(t)
            self._dispatch(t)
            self._move(t)
            self._crews(t)
            self._expire(t)
        for inc in self.incidents:                      # anything still open at the end
            if inc.state in ("pending", "responding"):
                inc.state = "unserved"
        self._close_legs(self.T)
        return self._finish(time.perf_counter() - t_wall)

    # ---- results ---------------------------------------------------------------------------------
    def _finish(self, runtime: float) -> SimResult:
        from sim.report import summarize_run
        hp = []
        for h in self.hazards:
            loc = h.location()
            if loc:
                hp.append({"name": h.name, "kind": h.kind, "lat": loc[0], "lon": loc[1], "start_min": h.start_min})
        res = SimResult(self.sc.name, self.times, self.metrics, self.frames, self.incidents, self.units, self.legs, self.repair_log,
                        self.baseline, {}, {}, runtime, [h.name for h in self.hazards], hp, self.bg)
        res.summary, res.peak = summarize_run(self.city, self.A, res)
        return res


def compute_baseline(city: CityData, router: Router, A: dict, background: float, fire_nodes: list[int]) -> dict:
    """Access with no disaster (same traffic/weather background)."""
    costs = router.costs(None, background)
    any_nodes = {f.node for f in city.hospitals(2)}
    t1_nodes = {f.node for f in city.hospitals(1)}
    d_any, d_t1, d_fire, lab = compute_access(city, router, costs, any_nodes, t1_nodes, fire_nodes)
    s = summarize(city, A, d_any, d_t1, d_fire)
    return {"summary": s, "catch": catchments(city, d_any, lab, A["impact"]["slow_minutes"] * 60), "d_any": d_any, "d_t1": d_t1, "d_fire": d_fire}
