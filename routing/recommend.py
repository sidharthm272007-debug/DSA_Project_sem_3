"""Which road should I take right now?

For one trip in one scenario this finds several genuinely different routes, measures each on time and on risk,
computes the actual Pareto-optimal trade-off between them, and recommends one point on it for the traveller's
stated attitude to risk. It is an optimiser with an explanation, not a black box: every number in the reasons is
measured from the route itself.

  * Candidate routes: Yen's algorithm (routing/kshortest.py) finds the K genuinely-different shortest routes by pure
    time, plus one further route minimising a heavily risk-penalised cost, to make sure the safest extreme of the
    trade-off is represented even when it needs a big detour Yen's-by-time alone would not rank highly. Reported
    times are always the true travel times on the damaged network.
  * The Pareto front (routing/pareto.py) is then extracted from that pool: the routes that are not beaten on both
    time AND risk by some other candidate. Minimising a single blended number like "time + v*risk" directly, without
    first finding the front, can skip a genuinely best-available route sitting on a concave stretch of the trade-off
    curve -- a provable limitation of that approach, not a tuning problem (see routing/pareto.py).
  * Recommendation: of the routes ON the front, pick the one minimising  time + v x risk-km,  where v = minutes the
    traveller would add to avoid one km of full-risk road (cautious / balanced / fastest, from assumptions.yaml).
  * Local knowledge: roads the traveller knows are blocked are removed; roads they know are unsafe get risk 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from routing.kshortest import k_shortest_paths
from routing.pareto import pareto_front
from routing.risk import edge_risk
from sim.city import CityData
from sim.router import Router
from sim.state import WorldState


def fmt_min(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "no route"
    m = seconds / 60.0
    return f"{m:.0f} min" if m >= 9.5 else f"{m:.1f} min"


@dataclass
class RouteOption:
    key: str
    edges: list[int]
    time_s: float
    dist_m: float
    risk_km: float                     # sum of length_km x risk over the route
    hazard_km: float                   # km on segments whose risk is at or above the "high" threshold
    max_risk: float
    objective: float = 0.0             # minutes + v x risk-km (the quantity the recommendation minimises, among Pareto-optimal routes)
    tags: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    coords: list[list[float]] = field(default_factory=list)


@dataclass
class RouteAdvice:
    options: list[RouteOption]
    recommended: RouteOption | None
    usual: RouteOption | None          # the same trip on a normal day
    usual_cut_at: list[str]            # named roads on the usual route that are closed in this scenario
    reasons: list[str]
    tolerance: str
    v: float
    walk_first: dict | None = None     # if the start is cut off: {"walk_m", "node", "where", "advice"}
    unreachable: bool = False


def _measure(city: CityData, A: dict, edges: list[int], time_arr: np.ndarray, risk: np.ndarray, key: str) -> RouteOption:
    g = city.g
    e = np.asarray(edges, dtype=np.int64)
    length_km = g.length_m[e] / 1000.0
    high = float(A["routing"]["risk"]["high"])
    r = risk[e]
    coords: list[list[float]] = []
    for x in edges:
        pts = np.round(city.edge_polyline(int(x)).astype(np.float64), 5).tolist()
        coords.extend(pts if not coords else pts[1:])
    return RouteOption(key, edges, float(time_arr[e].sum()), float(g.length_m[e].sum()), float((length_km * r).sum()),
                       float(length_km[r >= high].sum()), float(r.max()) if len(r) else 0.0, coords=coords)


def route_steps(city: CityData, A: dict, opt: RouteOption, time_arr: np.ndarray, risk: np.ndarray) -> list[dict]:
    """Merge consecutive stretches of the same named road into readable steps."""
    g = city.g
    high = float(A["routing"]["risk"]["high"])
    steps: list[dict] = []
    for e in opt.edges:
        name = city.way_names.get(int(g.way_id[e])) or ("flyover" if city.is_flyover[e] else "unnamed road")
        if steps and steps[-1]["road"] == name:
            s = steps[-1]
        else:
            s = {"road": name, "m": 0.0, "s": 0.0, "rmax": 0.0, "deck": False}
            steps.append(s)
        s["m"] += float(g.length_m[e])
        s["s"] += float(time_arr[e])
        s["rmax"] = max(s["rmax"], float(risk[e]))
        s["deck"] = s["deck"] or bool(city.on_structure[g.edge_tails[e]] and city.on_structure[g.indices[e]])
    for s in steps:
        s["risky"] = s["rmax"] >= high
    return steps


def advise(city: CityData, router: Router, A: dict, state: WorldState, origin: int, dest: int, tolerance: str = "balanced",
           blocked_edges: np.ndarray | None = None, avoid_edges: np.ndarray | None = None, _allow_walk: bool = True) -> RouteAdvice:
    R = A["routing"]
    v = float(R["tolerance_min_per_risk_km"][tolerance])
    g = city.g
    F = state.F.copy()
    if blocked_edges is not None and len(blocked_edges):
        be = np.asarray(blocked_edges, dtype=np.int64)
        F[be] = 0.0
        rv = g.rev[be]
        F[rv[rv >= 0]] = 0.0
    st = replace(state, F=F)
    arr = router.costs(F, state.bg).arr
    risk = edge_risk(city, A, st, avoid_edges)

    # the same trip on a normal day, for comparison
    arr_n = router.costs(None, state.bg).arr
    usual, cut_at = None, []
    if origin != dest:
        un = k_shortest_paths(router, arr_n, origin, dest, 1)
        if un:
            usual = _measure(city, A, list(un[0].edges), arr_n, risk, "usual")
            usual.steps = route_steps(city, A, usual, arr_n, risk)
            names = []
            for e in un[0].edges:
                if arr[e] == np.inf:
                    nm = city.way_names.get(int(g.way_id[e])) or "an unnamed road"
                    if nm not in names:
                        names.append(nm)
            cut_at = names

    if origin == dest:
        return RouteAdvice([], None, None, [], ["Start and destination are the same junction."], tolerance, v, unreachable=True)

    # Candidates: the K genuinely-different shortest routes by pure time (Yen's algorithm), plus one further route
    # minimising a heavily risk-penalised cost, so the safe extreme of the trade-off is represented even if it
    # needs a detour long enough that it would never rank in the top K by time alone.
    found: dict[tuple, RouteOption] = {}
    for c in k_shortest_paths(router, arr, origin, dest, int(R["k_shortest"])):
        if not np.isfinite(arr[np.asarray(c.edges)]).all():
            continue
        if c.edges not in found:
            found[c.edges] = _measure(city, A, list(c.edges), arr, risk, "route")
    safety_cost = arr * (1.0 + float(R["safety_search_penalty"]) * risk.astype(np.float64))
    safest_search = k_shortest_paths(router, safety_cost, origin, dest, 1)
    if safest_search and safest_search[0].edges not in found and np.isfinite(arr[np.asarray(safest_search[0].edges)]).all():
        found[safest_search[0].edges] = _measure(city, A, list(safest_search[0].edges), arr, risk, "route")

    if not found:
        adv = RouteAdvice([], None, usual, cut_at, ["No open road route exists between these two points in this scenario."], tolerance, v, unreachable=True)
        if _allow_walk:
            _suggest_walk(city, router, A, state, arr, origin, dest, adv, tolerance, blocked_edges, avoid_edges)
        return adv

    # The Pareto-optimal front of that pool (routing/pareto.py): every route shown is one no other candidate beats
    # on both time and risk at once, so every one of them is the right answer for SOME genuine attitude to risk.
    opts = pareto_front(list(found.values()))
    for o in opts:
        o.objective = o.time_s / 60.0 + v * o.risk_km
    fastest, safest = opts[0], opts[-1]                 # pareto_front returns time-ascending, so risk-descending-then-min at the end
    rec = min(opts, key=lambda o: o.objective)
    fastest.tags.append("Fastest")
    safest.tags.append("Safest")
    rec.tags.insert(0, "Recommended")
    for o in opts:
        o.steps = route_steps(city, A, o, arr, risk)
    opts.sort(key=lambda o: (o is not rec, o.time_s))
    return RouteAdvice(opts, rec, usual, cut_at, _reasons(rec, fastest, safest, usual, cut_at, len(opts), A), tolerance, v)


def _reasons(rec: RouteOption, fast: RouteOption, safe: RouteOption, usual: RouteOption | None, cut_at: list[str], n: int, A: dict) -> list[str]:
    out = [f"{fmt_min(rec.time_s)} for {rec.dist_m / 1000:.1f} km."]
    if usual is not None:
        d = (rec.time_s - usual.time_s) / 60.0
        out[0] = (f"{fmt_min(rec.time_s)} for {rec.dist_m / 1000:.1f} km: "
                  + (f"{abs(d):.0f} min {'longer' if d > 0 else 'shorter'} than the same trip on a normal day ({fmt_min(usual.time_s)})." if abs(d) >= 0.5 else f"about the same as a normal day ({fmt_min(usual.time_s)})."))
    if cut_at:
        out.append("The usual route is cut at " + ", ".join(cut_at[:4]) + ".")
    if rec is fast:
        out.append("It is also the fastest open route." if n > 1 else "It is the only sensible open route.")
    else:
        extra = (rec.time_s - fast.time_s) / 60.0
        out.append(f"It is {extra:.1f} min slower than the fastest open route, but avoids {fast.risk_km - rec.risk_km:.1f} risk-km ({max(fast.hazard_km - rec.hazard_km, 0):.1f} km of high-risk road).")
    if rec.hazard_km > 0.05:
        out.append(f"{rec.hazard_km:.1f} km of it is on high-risk road (inside the hazard, slowed, near a closure, or a bridge/flyover at risk)" + (" and no open route has less." if rec is safe else "."))
    else:
        out.append("It stays clear of high-risk road.")
    return out


def _suggest_walk(city, router, A, state, arr, origin, dest, adv: RouteAdvice, tolerance, blocked, avoid) -> None:
    """The start is cut off from the destination: find the nearest junction that still has an open route, within walking distance."""
    from sim.router import Costs
    d, _ = router.reverse_multi(Costs(arr, arr[router.emap]), [dest])
    ok = np.flatnonzero(np.isfinite(d) & city.giant)
    if not len(ok):
        return
    dist, j = cKDTree(city.xy[ok]).query(city.xy[origin])
    if dist > float(A["routing"]["max_walk_m"]):
        adv.reasons.append(f"No junction with an open route to the destination lies within {A['routing']['max_walk_m']:.0f} m of the start.")
        return
    node = int(ok[j])
    inner = advise(city, router, A, state, node, dest, tolerance, blocked, avoid, _allow_walk=False)
    adv.walk_first = {"walk_m": float(dist), "node": node, "where": city.describe_node(node), "advice": inner}
    adv.reasons.append(f"The start is cut off. The nearest junction with an open route is about {dist:.0f} m away ({city.describe_node(node)}).")


def hospital_options(city: CityData, router: Router, A: dict, state: WorldState, origin: int, tier: int | None = None, top: int = 4,
                     blocked_edges: np.ndarray | None = None) -> dict:
    """Open hospitals ranked by time from ``origin`` on the damaged network, plus which nearby ones are out of service."""
    F = state.F.copy()
    if blocked_edges is not None and len(blocked_edges):
        be = np.asarray(blocked_edges, dtype=np.int64)
        F[be] = 0.0
        rv = city.g.rev[be]
        F[rv[rv >= 0]] = 0.0
    arr = router.costs(F, state.bg).arr
    dist = dijkstra(router._matrix(arr, router._fplan), directed=True, indices=int(origin))
    rows = []
    for f in city.hospitals(2):
        if tier is not None and f.tier > tier:
            continue
        air = float(np.hypot(*(city.xy[f.node] - city.xy[origin])))
        if f.id in state.closed_fac:
            status = "out of service"
        elif not np.isfinite(dist[f.node]):
            status = "no open route"
        else:
            status = "open"
        rows.append({"fac": f, "air_m": air, "time_s": float(dist[f.node]), "status": status})
    open_rows = sorted([r for r in rows if r["status"] == "open"], key=lambda r: r["time_s"])[:top]
    return {"open": open_rows, "nearest_by_air": sorted(rows, key=lambda r: r["air_m"])[:top]}
