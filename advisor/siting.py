"""Where should the next hospital / fire station / ambulance post go?

Greedy facility location across the whole scenario portfolio (the classic greedy heuristic for the p-median / covering
problem): pick the site that saves the most person-time to the service, update the picture, repeat. Because the
objective is averaged over disasters as well as a normal day, a site that only helps on a good day loses to one that
also helps when roads are cut.

Candidate sites are real road junctions, thinned so they are not all in one block, and pre-ranked by how many people
they would be close to who are currently poorly served.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from advisor.evaluator import IDEA_SERVICES, Evaluator
from store.csr_graph import ROAD_CLASS_NAMES


@dataclass
class SitePick:
    rank: int
    node: int
    lat: float
    lon: float
    where: str
    effect: dict                    # aggregated marginal effect given the earlier picks
    primary: str


PRIMARY = {"hospital": "hosp", "hospital_major": "t1", "fire_station": "fire", "ambulance_post": "amb"}


def candidate_sites(ev: Evaluator, idea: str, n: int, min_sep_m: float = 500.0) -> list[int]:
    c, g = ev.city, ev.city.g
    z = PRIMARY[idea]
    ok_class = np.isin(g.road_class, [ROAD_CLASS_NAMES.index(k) for k in ("trunk", "primary", "secondary", "tertiary") if k in ROAD_CLASS_NAMES])
    on_road = np.zeros(g.n_nodes, dtype=bool)
    on_road[g.edge_tails[ok_class]] = True
    on_road &= c.giant & ~c.on_structure           # never on a flyover / bridge deck / tunnel
    nodes = np.flatnonzero(on_road)
    # how often (weighted over the portfolio) is each node poorly served? people x that
    unserved = np.zeros(g.n_nodes)
    for st in ev.snaps:
        unserved += st.snap.weight * (st.d[z] > ev.sla[z])
    need = c.pop * unserved
    tree = cKDTree(c.xy)
    score = np.empty(len(nodes))
    for s0 in range(0, len(nodes), 600):                       # in chunks: all 25k neighbour lists at once are ~1 GB of Python integers
        near = tree.query_ball_point(c.xy[nodes[s0:s0 + 600]], r=2000.0)
        score[s0:s0 + 600] = [need[idx].sum() for idx in near]
    order = np.argsort(-score)
    chosen: list[int] = []
    for i in order:
        if score[i] <= 0 or len(chosen) >= n:
            break
        if all(np.hypot(*(c.xy[nodes[i]] - c.xy[j])) >= min_sep_m for j in chosen):
            chosen.append(int(nodes[i]))
    return chosen


def greedy_sites(ev: Evaluator, idea: str, candidates: list[int], k: int, progress=None) -> list[SitePick]:
    """Pick up to ``k`` sites one at a time; each pick's effect is its marginal gain given the earlier picks."""
    z = PRIMARY[idea]
    services = IDEA_SERVICES[idea]
    pop, cap = ev.pop, ev.cap
    # for every (candidate, snapshot): the nodes it would improve and by how much (relative to today's picture).
    # Stored compactly (32-bit indices, 32-bit seconds): only the ranking uses these; each pick's reported effect is recomputed exactly below.
    reach: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for ci, node in enumerate(candidates):
        for si, st in enumerate(ev.snaps):
            R = ev._reach(st, node)
            idx = np.flatnonzero(R < st.d[z])
            reach[(ci, si)] = (idx.astype(np.int32), R[idx].astype(np.float32))
        if progress:
            progress(ci / max(len(candidates), 1))
    cur = [{zz: st.d[zz].copy() for zz in services} for st in ev.snaps]
    w = np.array([st.snap.weight for st in ev.snaps])
    picks: list[SitePick] = []
    available = set(range(len(candidates)))
    for rank in range(1, k + 1):
        best, best_gain = None, 0.0
        for ci in sorted(available):
            gain = 0.0
            for si in range(len(ev.snaps)):
                idx, Rv = reach[(ci, si)]
                if not len(idx) or w[si] == 0:
                    continue
                d0 = np.minimum(cur[si][z][idx], cap)
                gain += w[si] * float((pop[idx] * np.maximum(d0 - np.minimum(Rv.astype(np.float64), cap), 0.0)).sum())
            if gain > best_gain:
                best, best_gain = ci, gain
        if best is None or best_gain <= 0:
            break
        rows = []
        for si, st in enumerate(ev.snaps):
            R = ev._reach(st, candidates[best])
            new = {zz: np.minimum(cur[si][zz], R) for zz in services}
            rows.append(ev._diff(st, cur[si], new, services))
            cur[si] = new
        node = candidates[best]
        lat, lon = float(ev.city.g.lat[node]), float(ev.city.g.lon[node])
        picks.append(SitePick(rank, node, lat, lon, ev.city.describe_node(node), ev.aggregate(rows), z))
        available.discard(best)
    return picks
