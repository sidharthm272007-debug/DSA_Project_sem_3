"""Debris-clearing plan: reconnect cut-off people first.

Blocked roads are ranked by (people reconnected + hospital value) / clearing time. Only roads that join two
disconnected parts of the network can reconnect anyone, so we build the *component graph* (nodes = connected pieces of
the usable network, edges = blocked roads) and take its minimum spanning forest with Kruskal + Union-Find: a road that
would only duplicate an existing connection is ranked far lower.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RepairTarget:
    edges: list[int]              # arcs to reopen (both directions of one physical road)
    tail: int
    clear_min: float
    score: float
    people: float                 # people (plus hospital value) reconnected
    connects: bool
    name: str = ""
    in_tree: bool = False


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def components(city, usable: np.ndarray) -> np.ndarray:
    """Connected components of the network using only usable arcs (treated as undirected). Labels 0..K-1."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    g = city.g
    e = np.flatnonzero(usable)
    m = csr_matrix((np.ones(len(e), dtype=np.int8), (g.edge_tails[e], g.indices[e])), shape=(g.n_nodes, g.n_nodes))
    return connected_components(m, directed=False)[1]


def plan_repairs(city, factor: np.ndarray, rep_mask: np.ndarray, repair_mult: np.ndarray, A: dict, limit: int = 300) -> list[RepairTarget]:
    g, R = city.g, A["repair"]
    blocked = rep_mask & (factor <= 0)
    if not blocked.any():
        return []
    lab = components(city, factor > 0)
    k = int(lab.max()) + 1
    value = np.bincount(lab, weights=city.pop, minlength=k)
    hw = A["repair"].get("hospital_value_people", {1: 30000, 2: 3000})
    for f in city.hospitals(2):
        value[lab[f.node]] += hw.get(f.tier, hw.get(str(f.tier), 0))
    rows = []
    seen = set()
    for e in np.flatnonzero(blocked):
        r = int(g.rev[e])
        key = (min(int(e), r) if r >= 0 else int(e))
        if key in seen:
            continue
        seen.add(key)
        arcs = [int(e)] + ([r] if r >= 0 and blocked[r] else [])
        length = float(g.length_m[e])
        clear = (R["base_min"] + R["min_per_100m"] * length / 100.0) * float(repair_mult[e])
        a, b = int(lab[g.edge_tails[e]]), int(lab[g.indices[e]])
        rows.append((clear, arcs, a, b, length))
    rows.sort(key=lambda x: x[0])
    tree = DSU(k)
    targets: list[RepairTarget] = []
    for clear, arcs, a, b, length in rows:
        e = arcs[0]
        connects = a != b
        in_tree = connects and tree.union(a, b)
        if connects:
            people = float(min(value[a], value[b]))
            score = people / clear * (1.0 if in_tree else 0.15)
        else:
            people = 0.0
            score = 1e-3 * (length * (3.0 if g.is_arterial[e] else 1.0)) / clear
        targets.append(RepairTarget(arcs, int(g.edge_tails[e]), clear, score, people, connects, in_tree=in_tree))
    targets.sort(key=lambda t: (-t.score, t.edges[0]))
    targets = targets[:limit]
    for t in targets[:60]:
        t.name = city.edge_name(t.edges[0])
    return targets
