"""Which roads do people depend on to reach a hospital?

For every home, take its fastest route to the nearest open hospital. The load on a road is the number of people whose
fastest route uses it. Comparing the load in a scenario with the load on a normal day shows the lifelines (roads that
many depend on), the detour corridors (roads whose load surges because something else failed) and the roads that lost
their load (cut). Exact: one compiled search plus one pass over the shortest-path forest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sim.city import CityData
from sim.router import Router
from sim.state import WorldState


@dataclass
class Dependence:
    flow: np.ndarray            # per edge: people whose fastest route to a hospital uses it
    reach: float                # people who can reach any open hospital at all
    cut: float                  # people who cannot


def hospital_dependence(city: CityData, router: Router, F: np.ndarray, bg: float, hospital_nodes: list[int]) -> Dependence:
    g, N = city.g, city.g.n_nodes
    costs = router.costs(F, bg)
    flow = np.zeros(g.n_edges, dtype=np.float64)
    if not hospital_nodes:
        return Dependence(flow, 0.0, float(city.pop.sum()))
    mat = router._matrix(costs.rev_arr, router._rplan)
    d, pred, _ = dijkstra(mat, directed=True, indices=np.array(hospital_nodes, dtype=np.int64), min_only=True, return_predecessors=True)
    fin = np.isfinite(d)
    sub = np.where(fin, city.pop, 0.0)
    order = np.flatnonzero(fin)
    order = order[np.argsort(-d[order], kind="stable")]              # farthest first: every child before its parent
    carried = np.zeros(N)
    par = pred
    for x in order:
        p = par[x]
        carried[x] = sub[x]
        if p >= 0:
            sub[p] += sub[x]
    xs = order[par[order] >= 0]
    key = g.edge_tails.astype(np.int64) * N + g.indices
    eo = np.lexsort((costs.arr, key))                                # parallel roads: the cheaper one carries the flow
    ks = key[eo]
    first = np.r_[True, ks[1:] != ks[:-1]]
    ukey, uedge = ks[first], eo[first]
    q = xs.astype(np.int64) * N + par[xs]
    flow[uedge[np.searchsorted(ukey, q)]] = carried[xs]
    return Dependence(flow, float(city.pop[fin].sum()), float(city.pop[~fin].sum()))


def open_hospital_nodes(city: CityData, state: WorldState, tier: int | None = None) -> list[int]:
    return sorted({f.node for f in city.hospitals(2) if f.id not in state.closed_fac and (tier is None or f.tier <= tier)})


def lifelines(city: CityData, router: Router, state: WorldState, tier: int | None = None) -> dict:
    """Load per road now versus a normal day, summarised by named road."""
    g = city.g
    now = hospital_dependence(city, router, state.F, state.bg, open_hospital_nodes(city, state, tier))
    norm = hospital_dependence(city, router, np.ones(g.n_edges, dtype=np.float32), state.bg, open_hospital_nodes(city, normal_like(state), tier))
    change = now.flow - norm.flow
    ways = g.way_id
    rows = {}
    for e in np.flatnonzero((now.flow > 0) | (norm.flow > 0)):
        name = city.way_names.get(int(ways[e]))
        if not name:
            continue
        r = rows.setdefault(name, {"road": name, "now": 0.0, "normal": 0.0, "rise": 0.0, "fall": 0.0, "edge_now": -1})
        if now.flow[e] > r["now"]:
            r["now"], r["edge_now"] = float(now.flow[e]), int(e)
        r["normal"] = max(r["normal"], float(norm.flow[e]))
        r["rise"] = max(r["rise"], float(change[e]))
        r["fall"] = min(r["fall"], float(change[e]))
    table = list(rows.values())
    # every hospital's last stretch carries its whole catchment by construction; leave those out of the rankings (they stay on the map)
    hx = city.xy[[f.node for f in city.hospitals(2)]]
    near_h = set()
    if len(hx):
        from scipy.spatial import cKDTree
        mid = 0.5 * (city.xy[g.edge_tails] + city.xy[g.indices])
        d, _ = cKDTree(hx).query(mid)
        near_h = set(np.flatnonzero(d <= 300.0).tolist())
    if near_h:
        keep = {}
        for e in np.flatnonzero((now.flow > 0) | (norm.flow > 0)):
            if e in near_h:
                continue
            name = city.way_names.get(int(ways[e]))
            if not name:
                continue
            r = keep.setdefault(name, {"road": name, "now": 0.0, "normal": 0.0, "rise": 0.0, "fall": 0.0, "edge_now": -1})
            if now.flow[e] > r["now"]:
                r["now"], r["edge_now"] = float(now.flow[e]), int(e)
            r["normal"] = max(r["normal"], float(norm.flow[e]))
            r["rise"] = max(r["rise"], float(change[e]))
            r["fall"] = min(r["fall"], float(change[e]))
        table = list(keep.values())
    return {"now": now, "normal": norm, "change": change, "table": table,
            "top_load": sorted(table, key=lambda r: -r["now"])[:12],
            "top_rise": sorted([r for r in table if r["rise"] > 0], key=lambda r: -r["rise"])[:12],
            "top_fall": sorted([r for r in table if r["fall"] < 0], key=lambda r: r["fall"])[:12]}


def normal_like(state: WorldState) -> WorldState:
    return WorldState("normal", np.ones_like(state.F), np.zeros_like(state.expo), state.bg, set())
