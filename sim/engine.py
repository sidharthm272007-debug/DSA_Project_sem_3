"""The disaster simulator.

One minute at a time: hazards evolve, incidents appear where people are exposed, units are dispatched over the current
(damaged) road network, re-route when roads close mid-trip, crews clear debris in a connectivity-first order, and every
few minutes we recompute who can still reach a hospital. The fire hazard reacts to whether engines actually arrive.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

from sim.city import CityData
from sim.hazards import Hazard, HazardContext, IncidentSpec, WorldView
from sim.metrics import catchments, compute_access, summarize
from sim.repair import RepairTarget, plan_repairs
from sim.response import Incident, Resources, Unit, build_units
from sim.router import Costs, Router


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
class Leg:
    unit: int
    kind: str
    leg: str
    incident: int | None
    edges: list[tuple[int, float, float]]


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


class Simulation(WorldView):
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

    # WorldView ---------------------------------------------------------------------------------------
    def engines_on_scene(self, tag: str) -> int:
        inc = self.tag_inc.get(tag)
        return inc.arrived if inc is not None and inc.kind == "fire" else 0

    # ------------------------------------------------------------------------------------------------
    def run(self, progress=None) -> SimResult:
        t_wall = time.perf_counter()
        city, sc, A, g = self.city, self.sc, self.A, self.city.g
        E, N = g.n_edges, g.n_nodes
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
        R = A["response"]

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

    # ---- state -----------------------------------------------------------------------------------
    def _state_update(self, t: int) -> None:
        city, g, A = self.city, self.city.g, self.A
        E, N = g.n_edges, g.n_nodes
        for h in self.hazards:
            h.step(float(t), self)
        for h in list(self.hazards):
            for kid in h.spawn():
                kid.reset(HazardContext(city, A, self.sc.seed, self.sc.context.wind_toward_deg, self.sc.context.wind_kmh))
                kid.step(float(t), self)
                self.hazards.append(kid)
        F = np.ones(E, dtype=np.float32)
        rep = np.zeros(E, dtype=bool)
        nonrep = np.zeros(E, dtype=bool)
        mult = np.ones(E)
        expo = np.zeros(N)
        dom = np.full(N, -1, dtype=np.int64)
        out: dict[int, float] = {}
        for i, h in enumerate(self.hazards):
            f = h.factor()
            if f is not None:
                F *= f
                r = h.repairable()
                if r is not None:
                    rep |= r
                    mult[r] = np.minimum(mult[r], h.repair_mult)
                    nonrep |= (f == 0) & ~r
                else:
                    nonrep |= f == 0
            e = h.exposure()
            if e is not None:
                better = e > expo
                expo[better] = e[better]
                dom[better] = i
            for fid, cap in h.outages().items():
                out[fid] = min(cap, out.get(fid, 1.0))
        reopen = self.repaired & rep & ~nonrep
        F[reopen] = 1.0
        self.F = F
        self.rep_mask = rep & ~self.repaired
        self.rep_mult = mult
        self.expo, self.dom = expo, dom
        self.costs = self.router.costs(F, self.bg)
        self.closed_fac = {fid for fid, cap in out.items() if cap <= 0}
        hosp = [f for f in city.hospitals(2) if f.id not in self.closed_fac]
        self.open_any = {f.node for f in hosp}
        self.open_t1 = {f.node for f in hosp if f.tier == 1}
        self._record(t)

    def _refresh_costs(self) -> None:
        F = self.F.copy()
        rep = self.rep_mask | self.repaired
        F[self.repaired & rep] = np.where(self.F[self.repaired & rep] == 0, 1.0, self.F[self.repaired & rep])
        self.costs = self.router.costs(F, self.bg)
        self.F = F

    def _record(self, t: int) -> None:
        city, A, g = self.city, self.A, self.city.g
        acc = compute_access(city, self.router, self.costs, self.open_any, self.open_t1, self.fire_nodes)
        d_any, d_t1, d_fire, lab = acc
        s = summarize(city, A, d_any, d_t1, d_fire)
        cat = catchments(city, d_any, lab, A["impact"]["slow_minutes"] * 60)
        base_cat = self.baseline["catch"]
        ratio = {f.id: (cat.get(f.node, 0.0) / base_cat[f.node]) if base_cat.get(f.node, 0) > 0 else None for f in city.hospitals(2) if f.tier == 1}
        F = self.F
        pf = self.pair_first
        blocked = np.flatnonzero((F == 0) & pf)
        slowed = np.flatnonzero((F > 0) & (F < 0.85) & pf)
        bad = ~np.isfinite(d_any) | (d_any > A["impact"]["severe_minutes"] * 60)
        cells = np.bincount(city.cell_id[bad], weights=city.pop[bad], minlength=city.n_cells)
        nz = np.flatnonzero(cells > 50)
        prims = [p for h in self.hazards for p in h.footprint()]
        open_inc = sum(1 for i in self.incidents if i.state in ("pending", "responding", "onscene", "transport"))
        served = sum(1 for i in self.incidents if i.state == "done")
        row = {"t": t, **s,
               "open_road_share": float(g.length_m[F > 0].sum() / g.length_m.sum()),
               "blocked_km": float(g.length_m[(F == 0) & pf].sum() / 1000.0),
               "exposed": float(city.pop[self.expo >= 0.5].sum()),
               "incidents_open": open_inc, "incidents_total": len(self.incidents), "served": served,
               "hospitals_closed": len([f for f in city.hospitals(2) if f.id in self.closed_fac]),
               "repaired_km": float(g.length_m[self.repaired & pf].sum() / 1000.0)}
        self.metrics.append(row)
        self.times.append(float(t))
        self.frames.append({"t": t, "blocked": blocked.astype(np.int32), "slowed": slowed.astype(np.int32), "prims": prims,
                            "hospital_ratio": ratio, "closed_fac": sorted(self.closed_fac),
                            "cut_cells": (nz.astype(np.int32), cells[nz].astype(np.float32)),
                            "hazard_info": [h.info() for h in self.hazards if h.active()],
                            # compact copies of the full state, so any moment can be replayed for route advice
                            "F_idx": np.flatnonzero(F < 1).astype(np.int32), "F_val": F[F < 1].astype(np.float16),
                            "expo_idx": np.flatnonzero(self.expo > 0).astype(np.int32), "expo_val": self.expo[self.expo > 0].astype(np.float16)})

    # ---- incidents -------------------------------------------------------------------------------
    def _plan_incidents(self, t0: int, t1: int) -> None:
        A, city = self.A, self.city
        I = A["incidents"]
        for h in self.hazards:
            for spec in h.incidents(float(t0), float(t1)):
                heapq.heappush(self.future, (spec.t_min, len(self.future), spec))
        dt_h = (t1 - t0) / 60.0
        lam = city.pop / 1000.0 * (I["baseline_per_1000_per_hour"] + I["hazard_boost_per_1000_per_hour"] * self.expo) * self.sc.incident_scale
        counts = self.rng.poisson(lam * dt_h)
        for node in np.flatnonzero(counts):
            di = int(self.dom[node])
            mixkey = self.hazards[di].kind if (di >= 0 and self.expo[node] > 0.05) else "baseline"
            mix = I["mix"].get(mixkey, I["mix"]["baseline"])
            kinds, p = list(mix), np.array(list(mix.values()), dtype=float)
            for _ in range(int(counts[node])):
                kind = str(self.rng.choice(kinds, p=p / p.sum()))
                sev = int(self.rng.choice([1, 2, 3], p=I["medical_severity_probs"] if kind != "fire" else [0.55, 0.35, 0.10]))
                spec = IncidentSpec(kind, int(node), sev, t0 + float(self.rng.uniform(0, t1 - t0)), mixkey, needs=(1 if sev == 1 or kind != "fire" else 2))
                heapq.heappush(self.future, (spec.t_min, len(self.future), spec))

    def _release_future(self, t: int) -> None:
        cap = self.A["incidents"]["max_incidents"]
        while self.future and self.future[0][0] <= t + 1e-9:
            _, _, spec = heapq.heappop(self.future)
            if len(self.incidents) >= cap and spec.tag in ("baseline", "earthquake", "storm", "fire", "plume", "flood_zone", "custom", "structure_failure", "accident") and not self.tag_inc_link(spec):
                continue
            inc = Incident(len(self.incidents), spec.kind, spec.node, spec.severity, float(spec.t_min), spec.tag, spec.needs)
            self.incidents.append(inc)
            if spec.tag and self.tag_inc_link(spec):
                self.tag_inc[spec.tag] = inc

    def tag_inc_link(self, spec: IncidentSpec) -> bool:
        return any(h.name == spec.tag for h in self.hazards)

    def _expire(self, t: int) -> None:
        w = self.A["response"]["max_wait_min"]
        for inc in self.incidents:
            if inc.state == "pending" and t - inc.t > w:
                inc.state = "unserved"

    # ---- dispatch --------------------------------------------------------------------------------
    def _dispatch(self, t: int) -> None:
        R = self.A["response"]
        pend = [i for i in self.incidents if i.state in ("pending", "responding") and len(i.units) < i.needs and i.retry_at <= t]
        if not pend:
            return
        pend.sort(key=lambda i: (-i.severity, i.t))
        elig = {"medical": ["ambulance"], "fire": ["fire"], "rescue": ["rescue", "fire"]}
        for inc in pend:
            while len(inc.units) < inc.needs:
                unit = self._assign(inc, elig[inc.kind], t)
                if unit is None:
                    inc.retry_at = t + (R["retry_min"] if inc.state == "pending" else 3)
                    break

    def _assign(self, inc: Incident, kinds: list[str], t: int) -> Unit | None:
        for kind in kinds:
            idle = [u for u in self.units if u.kind == kind and u.state == "idle"]
            if not idle:
                continue
            by_node: dict[int, list[Unit]] = {}
            for u in idle:
                by_node.setdefault(u.node, []).append(u)
            node, sec, path = self.router.nearest_target(self.costs, inc.node, by_node)
            if node is None:
                continue
            u = by_node[node][0]
            u.state, u.path, u.pi, u.left_s, u.incident, u.leg, u.target = "enroute", path, 0, None, inc.id, "to_scene", inc.node
            u.trip = []
            inc.units.append(u.id)
            inc.state = "responding"
            if inc.t_assign is None:
                inc.t_assign = float(t)
            return u
        return None

    # ---- movement --------------------------------------------------------------------------------
    def _move(self, t: int) -> None:
        idx = self.router.indices
        for u in self.units:
            if u.state in ("enroute", "transport", "returning"):
                status = self._advance(u, t)
                if status == "blocked":
                    self._replan(u, t)
                elif status == "arrived":
                    self._arrive(u, t)
            elif u.state in ("onscene", "handover", "working") and t >= u.busy_until:
                self._finish_stage(u, t)

    def _advance(self, u: Unit, t: int) -> str:
        rem = 60.0
        cost, idx = self.costs.fwd, self.router.indices
        t0 = t * 60.0
        while rem > 0 and u.pi < len(u.path):
            e = u.path[u.pi]
            if u.left_s is None:
                c = cost[e]
                if c == math.inf:
                    u.blocked_wait_s += rem
                    return "blocked"
                u.left_s = c
                u.edge_in_s = t0 + (60.0 - rem)
            if u.left_s <= rem:
                rem -= u.left_s
                u.trip.append((e, u.edge_in_s, t0 + (60.0 - rem)))          # actual finish time (an edge can span several minutes)
                u.node = idx[e]
                u.pi += 1
                u.left_s = None
            else:
                u.left_s -= rem
                rem = 0
        u.busy_min += 1
        return "arrived" if u.pi >= len(u.path) else "moving"

    def _dest(self, u: Unit) -> int:
        if u.leg == "return":
            return u.base
        if u.leg == "to_repair":
            return int(u.target.tail)          # type: ignore[union-attr]
        return int(u.target)                   # type: ignore[arg-type]

    def _replan(self, u: Unit, t: int) -> None:
        """A road ahead closed. Find another way now; if there is none, wait and look again in a few minutes."""
        if t < u.replan_at:
            return
        path, sec = self.router.path(self.costs, u.node, self._dest(u))
        if path or u.node == self._dest(u):
            self._close_leg(u, t)
            u.path, u.pi, u.left_s, u.blocked_wait_s = path, 0, None, 0.0
            return
        if u.leg == "transport":                           # hospital became unreachable: try another one
            node, sec, path = self.router.nearest_forward(self.costs, u.node, self.open_any)
            if node is not None:
                self._close_leg(u, t)
                u.target, u.path, u.pi, u.left_s = node, path, 0, None
                u.blocked_wait_s = 0.0
                return
        u.replan_at = t + 3

    def _arrive(self, u: Unit, t: int) -> None:
        R = self.A["response"]
        self._close_leg(u, t + 1)
        if u.kind == "crew":
            if u.leg == "to_repair":
                tg: RepairTarget = u.target  # type: ignore[assignment]
                u.state, u.busy_until = "working", t + tg.clear_min
            else:
                u.state, u.leg = "idle", ""
            return
        inc = self.incidents[u.incident] if u.incident is not None else None
        if u.leg == "to_scene" and inc is not None:
            inc.arrived += 1
            if inc.t_arrive is None:
                inc.t_arrive = float(t + 1)
            u.state = "onscene"
            u.busy_until = 10 ** 9                                   # released when the incident is served
            if inc.kind == "fire":
                if inc.arrived >= inc.needs:
                    inc.state, inc.service_end = "onscene", t + R["service_min"]["fire"][inc.severity - 1]
                    for uid in inc.units:
                        self.units[uid].busy_until = inc.service_end
            elif inc.kind == "medical":
                u.busy_until = t + R["service_min"]["medical"][inc.severity - 1]
                inc.state = "onscene"
            else:
                u.busy_until = t + R["service_min"]["rescue"][inc.severity - 1]
                inc.state = "onscene"
        elif u.leg == "transport" and inc is not None:
            u.state, u.busy_until = "handover", t + R["hospital_handover_min"]
        elif u.leg == "return":
            u.state, u.leg, u.incident = "idle", "", None

    def _finish_stage(self, u: Unit, t: int) -> None:
        inc = self.incidents[u.incident] if u.incident is not None else None
        if u.state == "working":
            tg: RepairTarget = u.target  # type: ignore[assignment]
            for e in tg.edges:
                self.repaired[e] = True
            self._claimed.discard(tg.edges[0])
            self._repairs_since_plan += 1
            self.repair_log.append({"t": float(t), "edges": tg.edges, "name": tg.name or self.city.edge_name(tg.edges[0]), "people": tg.people,
                                    "clear_min": tg.clear_min, "connects": tg.connects})
            self._refresh_costs()
            u.state, u.leg, u.target = "idle", "", None
            return
        if inc is None:
            u.state = "idle"
            return
        if u.state == "onscene" and inc.kind == "medical":
            self._start_transport(u, inc, t)
        elif u.state == "onscene":                                   # fire / rescue served
            if inc.state != "done":
                inc.state, inc.t_done = "done", float(t)
            for uid in inc.units:
                v = self.units[uid]
                if v.state == "onscene":
                    self._go_home(v, t)
        elif u.state == "handover":
            inc.state, inc.t_done, inc.t_delivered = "done", float(t), float(t)
            self._go_home(u, t)

    def _start_transport(self, u: Unit, inc: Incident, t: int) -> None:
        targets = self.open_any
        node = None
        if inc.severity == 3 and self.open_t1:
            node, sec, path = self.router.nearest_forward(self.costs, u.node, self.open_t1, limit=45 * 60.0)
        if node is None:
            node, sec, path = self.router.nearest_forward(self.costs, u.node, targets, limit=120 * 60.0)
        if node is None:
            inc.state, inc.t_done = "stranded", float(t)
            self._go_home(u, t)
            return
        inc.state, inc.hospital = "transport", node
        u.state, u.leg, u.target, u.path, u.pi, u.left_s, u.trip = "transport", "transport", node, path, 0, None, []

    def _go_home(self, u: Unit, t: int) -> None:
        path, sec = self.router.path(self.costs, u.node, u.base) if u.node != u.base else ([], 0.0)
        if u.node != u.base and not path:
            u.state, u.leg, u.incident = "idle", "", None
            return
        u.state, u.leg, u.target, u.path, u.pi, u.left_s, u.trip = "returning", "return", None, path, 0, None, []
        if not path:
            u.state, u.leg, u.incident = "idle", "", None

    # ---- repair crews ----------------------------------------------------------------------------
    def _crews(self, t: int) -> None:
        crews = [u for u in self.units if u.kind == "crew"]
        idle = [u for u in crews if u.state == "idle" and u.replan_at <= t]
        if not idle or t % 2:
            return
        n_now = int((self.rep_mask & (self.F == 0)).sum())
        if n_now == 0:
            return
        stale = t - self._plan_t >= 30 or self._repairs_since_plan >= 5 or n_now != self._plan_n
        if stale and (not self._plan or t - self._plan_t >= 10):
            self._plan = plan_repairs(self.city, self.F, self.rep_mask, self.rep_mult, self.A)
            self._plan_t, self._plan_n, self._repairs_since_plan = t, n_now, 0
        open_jobs = [tg for tg in self._plan if tg.edges[0] not in self._claimed and not self.repaired[tg.edges[0]]]
        if not open_jobs:
            return
        for u in idle:
            dist, path_to = self.router.reach_from(self.costs, u.node, limit=3 * 3600.0)     # one search per crew, not one per job
            for tg in open_jobs:
                if tg.edges[0] in self._claimed or not np.isfinite(dist[tg.tail]):
                    continue
                path = path_to(tg.tail) if u.node != tg.tail else []
                self._claimed.add(tg.edges[0])
                u.state, u.leg, u.target, u.path, u.pi, u.left_s, u.trip = "enroute", "to_repair", tg, path, 0, None, []
                if not path:
                    self._arrive(u, t)
                break
            else:
                u.replan_at = t + 10                        # nothing reachable right now: look again later

    def _close_leg(self, u: Unit, t: int) -> None:
        if u.trip:
            self.legs.append(Leg(u.id, u.kind, u.leg, u.incident, list(u.trip)))
        u.trip = []

    def _close_legs(self, t: int) -> None:
        for u in self.units:
            self._close_leg(u, t)

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
