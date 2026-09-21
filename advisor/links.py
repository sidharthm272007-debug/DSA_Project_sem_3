"""Where would a new road or bridge actually help?

A new link is worth considering where two places are close as the crow flies but far apart by road at peak hour (a river,
a railway, a missing connection between two arterials). We look for such pairs among junctions on real arterial and
collector roads, rank them cheaply by (time saved) x (people living near the ends), keep a spread-out shortlist, and only
then score the shortlist properly against the whole scenario portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from advisor.evaluator import Evaluator
from store.csr_graph import ROAD_CLASS_NAMES


@dataclass
class LinkIdea:
    u: int
    v: int
    length_m: float
    tau_s: float                      # free-flow drive time of the new road
    crosses_water: bool
    today_min: float                  # drive time between the ends today, at peak hour
    with_min: float                   # ... with the new road, at peak hour
    proxy: float
    cost_crore: float = 0.0
    effect: dict | None = None
    where_u: str = ""
    where_v: str = ""
    kind: str = "link"


def link_geometry(ev: Evaluator, u: int, v: int) -> tuple[float, float, bool]:
    """(length_m, free-flow seconds, crosses water) for a straight-ish new road between two nodes."""
    ex, c = ev.ev, ev.city
    straight = float(np.hypot(*(c.xy[u] - c.xy[v])))
    length = straight * float(ex["link_detour_factor"])
    tau = length / (float(ex["link_speed_kmh"]) / 3.6)
    ts = np.linspace(0.1, 0.9, 9)[:, None]
    along = c.xy[u] + (c.xy[v] - c.xy[u]) * ts                        # sample points on the straight segment
    crosses = bool((c.water_distance_xy(along) < 60.0).any())         # passes within 60 m of a mapped waterline => priced as a bridge
    return length, tau, crosses


def link_cost_crore(A: dict, length_m: float, crosses_water: bool) -> float:
    cc = A["advisor"]["cost_crore"]
    return float(length_m / 1000.0 * (cc["bridge_per_km"] if crosses_water else cc["new_link_per_km"]))


def connectors(ev: Evaluator, cell_m: float = 300.0) -> np.ndarray:
    c, g = ev.city, ev.city.g
    ok = np.isin(g.road_class, [ROAD_CLASS_NAMES.index(k) for k in ("trunk", "primary", "secondary", "tertiary") if k in ROAD_CLASS_NAMES])
    on = np.zeros(g.n_nodes, dtype=bool)
    on[g.edge_tails[ok]] = True
    nodes = np.flatnonzero(on & c.giant & ~c.on_structure)       # a new road cannot join a flyover deck or a tunnel mid-span
    cells = np.floor(c.xy[nodes] / cell_m).astype(np.int64)
    _, first = np.unique(cells[:, 0] * 1_000_003 + cells[:, 1], return_index=True)
    return nodes[np.sort(first)]


def generate_links(ev: Evaluator, n_shortlist: int = 60, seed_cell_m: float = 700.0, min_len: float = 250.0, max_len: float = 2500.0,
                   min_ratio: float = 2.0, min_saved_s: float = 240.0, mid_sep_m: float = 1500.0, progress=None) -> list[LinkIdea]:
    c, A = ev.city, ev.A
    conn = connectors(ev)
    tree = cKDTree(c.xy[conn])
    pop_tree = cKDTree(c.xy)
    pop1k = np.array([c.pop[idx].sum() for idx in pop_tree.query_ball_point(c.xy[conn], r=1000.0)])
    pos = {int(n): i for i, n in enumerate(conn)}
    fwd = ev.router._matrix(ev.ev_arr, ev.router._fplan)
    bg_peak = A["context"]["time_of_day_speed"]["peak"]
    lim = 25 * 60.0
    cells = np.floor(c.xy[conn] / seed_cell_m).astype(np.int64)
    _, first = np.unique(cells[:, 0] * 1_000_003 + cells[:, 1], return_index=True)
    seeds = conn[np.sort(first)]
    raw = []
    B = 40
    for b0 in range(0, len(seeds), B):
        batch = seeds[b0:b0 + B]
        D = dijkstra(fwd, directed=True, indices=batch.astype(np.int64), limit=lim)
        for row, u in zip(D, batch):
            ids = np.array([i for i in tree.query_ball_point(c.xy[u], r=max_len) if conn[i] != u], dtype=np.int64)
            if not len(ids):
                continue
            vs = conn[ids]
            dist = np.hypot(*(c.xy[vs] - c.xy[u]).T)
            keep = dist >= min_len
            vs, dist, ids = vs[keep], dist[keep], ids[keep]
            length = dist * float(ev.ev["link_detour_factor"])
            tau_peak = length / (float(ev.ev["link_speed_kmh"]) / 3.6) / bg_peak
            t_net = np.minimum(row[vs], lim)                          # beyond the search limit counts as the limit (conservative)
            saved = t_net - tau_peak
            ok = (saved >= min_saved_s) & (t_net / np.maximum(tau_peak, 1.0) >= min_ratio)
            for v, s_, tn, tp, ln in zip(vs[ok], saved[ok], t_net[ok], tau_peak[ok], length[ok]):
                raw.append((float(s_ * (pop1k[pos[int(u)]] + pop1k[pos[int(v)]])), int(u), int(v), float(tn), float(tp), float(ln)))
        if progress:
            progress(min(1.0, (b0 + B) / max(len(seeds), 1))* 0.5)
    raw.sort(reverse=True)
    picked: list[tuple] = []
    mids: list[np.ndarray] = []
    for r in raw:
        _, u, v, tn, tp, ln = r
        mid = 0.5 * (c.xy[u] + c.xy[v])
        if any(np.hypot(*(mid - m)) < mid_sep_m for m in mids):            # one idea per neighbourhood, so the shortlist covers the whole city
            continue
        if any((np.hypot(*(c.xy[u] - c.xy[u2])) < 600 and np.hypot(*(c.xy[v] - c.xy[v2])) < 600) or
               (np.hypot(*(c.xy[u] - c.xy[v2])) < 600 and np.hypot(*(c.xy[v] - c.xy[u2])) < 600) for _, u2, v2, *_ in picked):
            continue
        picked.append(r)
        mids.append(mid)
        if len(picked) >= n_shortlist:
            break
    out = []
    for proxy, u, v, tn, tp, ln in picked:
        length, tau, cw = link_geometry(ev, u, v)
        out.append(LinkIdea(u, v, length, tau, cw, tn / 60.0, tp / 60.0, proxy, link_cost_crore(A, length, cw),
                            where_u=c.describe_node(u), where_v=c.describe_node(v)))
    return out


def score_links(ev: Evaluator, ideas: list[LinkIdea], progress=None) -> list[LinkIdea]:
    for i, idea in enumerate(ideas):
        idea.effect = ev.eval_link(idea.u, idea.v, idea.tau_s)
        if progress:
            progress(0.5 + 0.5 * (i + 1) / len(ideas))
    return ideas
