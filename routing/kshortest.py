"""Yen's algorithm: the K shortest LOOPLESS paths between two junctions, in strictly increasing order of cost.

Why this instead of the old approach (Dijkstra on ``time * (1 + penalty * risk)`` swept over a handful of penalty
values): sweeping a penalty only ever finds paths that are optimal for SOME linear combination of time and risk.
That misses routes on a concave stretch of the true trade-off curve -- a well-known, provable limitation of linear
scalarisation in multi-objective optimisation, not a tuning problem. Yen's algorithm instead enumerates paths by
pure rank (1st shortest, 2nd shortest, ...) under a single objective; running it under two different objectives
(pure time, and a heavily risk-penalised time) gives a genuinely richer, more defensible candidate pool: every
route in it is actually a best-or-near-best path under some real criterion, not just "whatever a sweep produced".
``routing/recommend.py`` then extracts the Pareto-optimal front from that pool.

Algorithm (Yen 1971): keep a list A of confirmed shortest paths (shortest, 2nd-shortest, ...) and a min-heap B of
candidate "deviations". To extend A by one more path, take the most recently confirmed path P and, for every node i
along it (the "spur node"), compute the shortest path from that spur node to the destination with (a) every edge
that any already-confirmed-or-candidate path also takes out of that exact same spur node removed, so the deviation
is never identical to something already found, and (b) every node earlier on P's root removed, so the result can
never loop back into its own prefix. Splicing the root (src -> spur) onto that deviation gives one new candidate;
push it onto B. After trying every spur node on P, pop the cheapest item off B and append it to A. Repeat.

On speed: an earlier version of this file restricted each spur search to a geographic box around the trip, to avoid
repeating hundreds of searches across the full 72,000-junction network. It was wrong. Twice. The first version
could return "no path" for a spur node whose only route left the box -- an easy bug, caught immediately. The second,
subtler version could return a real, valid-LOOKING path confined entirely inside the box while a cheaper one used
excluded territory, with no local signal (not even the path touching the box's edge) to say so -- caught only by
cross-checking every result against ``networkx.shortest_simple_paths``, an independent implementation of the same
algorithm, in tests/test_kshortest.py. A geographically-bounded search can be done correctly (A* with an admissible
straight-line heuristic naturally and provably limits exploration to the relevant area, with no hand-tuned box to
get wrong), but that needs its own priority-queue search loop and its own round of exactly this kind of verification,
which is real, separate work, not a quick patch. Until that exists, every search here runs on the full network: K is
kept modest (routing/recommend.py uses 4) and the user sees a short wait, honestly, rather than a wrong answer fast.

Verified against ``networkx.shortest_simple_paths`` on random graphs and on the toy city, and against brute-force
path enumeration for small graphs, in tests/test_kshortest.py.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from sim.router import Router


@dataclass
class Candidate:
    nodes: tuple[int, ...]
    edges: tuple[int, ...]
    cost: float


@dataclass(order=True)
class _Heap:
    cost: float
    tie: int
    cand: Candidate = field(compare=False)


def _point_path(router: Router, cost: np.ndarray, src: int, dst: int) -> Candidate | None:
    """One scipy Dijkstra, restricted to ``cost`` (inf = unusable), reconstructed as a loopless node/edge path."""
    mat = router._matrix(cost, router._fplan)
    d, pred = sp_dijkstra(mat, directed=True, indices=int(src), return_predecessors=True)
    if not np.isfinite(d[dst]):
        return None
    nodes = [int(dst)]
    while nodes[-1] != src:
        nodes.append(int(pred[nodes[-1]]))
    nodes.reverse()
    g = router.g
    edges = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        lo, hi = g.indptr[u], g.indptr[u + 1]
        cand = np.flatnonzero((g.indices[lo:hi] == v) & np.isfinite(cost[lo:hi])) + lo
        if not len(cand):
            return None
        edges.append(int(cand[np.argmin(cost[cand])]))
    return Candidate(tuple(nodes), tuple(edges), float(d[dst]))


def k_shortest_paths(router: Router, cost: np.ndarray, src: int, dst: int, K: int, xy: np.ndarray | None = None) -> list[Candidate]:
    """Yen's algorithm. Returns up to ``K`` loopless paths from ``src`` to ``dst``, strictly increasing in cost,
    each one genuinely different from every other (no two share the same edge set -- Yen's deviates from a
    different spur node each time, so it can never just re-find the same route). ``xy`` is accepted but currently
    unused (see the module docstring: a geographic speed-up was tried, found unsound, and removed); it stays in the
    signature so callers do not need to change when a correct one lands.
    """
    if src == dst:
        return []
    first = _point_path(router, cost, src, dst)
    if first is None:
        return []
    A: list[Candidate] = [first]
    B: list[_Heap] = []
    seen_edges: set[tuple[int, ...]] = {first.edges}
    tie = 0
    while len(A) < K:
        prev = A[-1]
        for i in range(len(prev.nodes) - 1):
            spur_node = prev.nodes[i]
            root_nodes = prev.nodes[:i + 1]
            root_cost = float(cost[np.asarray(prev.edges[:i], dtype=np.int64)].sum()) if i else 0.0

            c = cost.copy()
            for p in A:
                if p.nodes[:i + 1] == root_nodes and len(p.edges) > i:
                    c[p.edges[i]] = np.inf                          # never re-take the same next step out of this spur
            for n in root_nodes[:-1]:                                # never loop back through the already-used root
                lo, hi = router.g.indptr[n], router.g.indptr[n + 1]
                c[lo:hi] = np.inf
                rlo, rhi = router.rindptr[n], router.rindptr[n + 1]
                c[router.emap[rlo:rhi]] = np.inf

            spur = _point_path(router, c, spur_node, dst)
            if spur is None:
                continue
            nodes = root_nodes[:-1] + spur.nodes
            edges = prev.edges[:i] + spur.edges
            if edges in seen_edges:
                continue
            total = root_cost + spur.cost
            heapq.heappush(B, _Heap(total, tie, Candidate(nodes, edges, total)))
            tie += 1
        if not B:
            break
        nxt = heapq.heappop(B).cand
        while nxt.edges in seen_edges:
            if not B:
                nxt = None
                break
            nxt = heapq.heappop(B).cand
        if nxt is None:
            break
        seen_edges.add(nxt.edges)
        A.append(nxt)
    return A
