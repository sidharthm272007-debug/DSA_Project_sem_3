"""Dynamic-cost routing. The disaster changes road speeds every few minutes, so costs are rebuilt per state.

    costs = router.costs(edge_factor)                        # factor 0 => closed, 1 => normal speed
    dist, label = router.reverse_multi(costs, hospital_nodes)   # seconds to the nearest hospital, for every node (compiled, ~15 ms)
    node, t, path = router.nearest_target(costs, incident, idle_unit_nodes)   # early-exit heap Dijkstra for point queries

Bulk "distance to nearest facility" maps use scipy's compiled Dijkstra on a de-duplicated graph (parallel roads collapse
to the cheaper one, closed roads are dropped). Point queries with early exit use a small heap Dijkstra, which is faster
when the target is close. ``tests/test_router.py`` checks both against each other on random graphs.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from store.csr_graph import CSRGraph

INF = math.inf


@dataclass
class Costs:
    arr: np.ndarray                 # seconds per edge, original edge order (inf = closed)
    rev_arr: np.ndarray             # same values in reverse-graph order
    _fwd: list | None = field(default=None, repr=False)
    _rev: list | None = field(default=None, repr=False)

    @property
    def fwd(self) -> list[float]:   # python lists are only built if a heap search asks for them
        if self._fwd is None:
            self._fwd = self.arr.tolist()
        return self._fwd

    @property
    def rev(self) -> list[float]:
        if self._rev is None:
            self._rev = self.rev_arr.tolist()
        return self._rev


def _dedup_plan(rows: np.ndarray, cols: np.ndarray, n: int):
    """Static grouping of parallel arcs (same tail and head): returns (order, group_starts, urows, ucols)."""
    key = rows.astype(np.int64) * n + cols
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return order, starts, (ks[starts] // n).astype(np.int64), (ks[starts] % n).astype(np.int64)


class Router:
    def __init__(self, g: CSRGraph):
        self.g = g
        self.n = g.n_nodes
        self.indptr = g.indptr.tolist()
        self.indices = g.indices.tolist()
        self.tails = g.edge_tails.tolist()
        rg, emap = g.reverse()
        self.rindptr = rg.indptr.tolist()
        self.rindices = rg.indices.tolist()
        self.emap = emap                      # reverse edge j -> original edge id
        self.emap_l = emap.tolist()
        self.tt = g.tt_s
        n = g.n_nodes
        self._rplan = _dedup_plan(np.repeat(np.arange(n), np.diff(rg.indptr)), rg.indices, n)
        self._fplan = _dedup_plan(np.repeat(np.arange(n), np.diff(g.indptr)), g.indices, n)

    def costs(self, factor: np.ndarray | None = None, background: float = 1.0) -> Costs:
        if factor is None:
            arr = self.tt / max(background, 1e-3)
        else:
            f = np.asarray(factor, dtype=np.float64) * background
            arr = np.where(f > 1e-3, self.tt / np.maximum(f, 1e-3), np.inf)
        return Costs(arr, arr[self.emap])

    def _matrix(self, cost: np.ndarray, plan) -> csr_matrix:
        order, starts, urows, ucols = plan
        m = np.minimum.reduceat(cost[order], starts)          # parallel roads: keep the cheaper one
        ok = np.isfinite(m)                                    # closed roads leave the graph
        indptr = np.concatenate([[0], np.cumsum(np.bincount(urows[ok], minlength=self.n))])
        return csr_matrix((np.maximum(m[ok], 1e-6), ucols[ok], indptr), shape=(self.n, self.n))

    # ---- core loop ------------------------------------------------------------------------------------
    def _run(self, indptr, indices, cost, eids, sources, targets=None, limit=INF, labels=False):
        n = self.n
        dist = [INF] * n
        par = [-1] * n
        lab = [-1] * n if labels else None
        heap = []
        for s, d0 in sources:
            if d0 < dist[s]:
                dist[s] = d0
                if labels:
                    lab[s] = s
                heap.append((d0, s))
        heapq.heapify(heap)
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, u = pop(heap)
            if d > dist[u]:
                continue
            if d > limit:
                break
            if targets is not None and u in targets:
                return dist, par, lab, u
            lu = lab[u] if labels else 0
            for j in range(indptr[u], indptr[u + 1]):
                c = cost[j]
                if c == INF:
                    continue
                v = indices[j]
                nd = d + c
                if nd < dist[v]:
                    dist[v] = nd
                    par[v] = eids[j] if eids is not None else j
                    if labels:
                        lab[v] = lu
                    push(heap, (nd, v))
        return dist, par, lab, -1

    # ---- public searches ------------------------------------------------------------------------------
    def reverse_multi(self, costs: Costs, sources, limit: float = INF):
        """dist[v] = travel time from v to the nearest source (inf if none is reachable); label[v] = that source's node id, -1 if none."""
        items = list(sources.items()) if isinstance(sources, dict) else [(s, 0.0) for s in sources]
        if not items:
            return np.full(self.n, np.inf), np.full(self.n, -1, dtype=np.int64)
        if all(d0 == 0.0 for _, d0 in items):
            idx = np.array([s for s, _ in items], dtype=np.int64)
            d, _, lab = dijkstra(self._matrix(costs.rev_arr, self._rplan), directed=True, indices=idx, min_only=True,
                                 return_predecessors=True, limit=limit)
            lab = np.where(np.isfinite(d), lab, -1)
            return d, lab.astype(np.int64)
        dist, _, lab, _ = self._run(self.rindptr, self.rindices, costs.rev, None, items, None, limit, True)
        return np.array(dist), np.array(lab, dtype=np.int64)

    def reach_from(self, costs: Costs, src: int, limit: float = INF):
        """One compiled search from ``src``: returns (dist, path_to) where path_to(node) gives the edge path or None."""
        mat = self._matrix(costs.arr, self._fplan)
        dist, pred = dijkstra(mat, directed=True, indices=int(src), return_predecessors=True, limit=limit)
        arr = costs.arr
        indptr, indices = self.g.indptr, self.g.indices

        def path_to(node: int):
            if not np.isfinite(dist[node]):
                return None
            nodes = [int(node)]
            while nodes[-1] != src:
                nodes.append(int(pred[nodes[-1]]))
            nodes.reverse()
            out = []
            for u, v in zip(nodes[:-1], nodes[1:]):            # cheapest arc u->v (parallel roads)
                lo, hi = indptr[u], indptr[u + 1]
                cand = np.flatnonzero(indices[lo:hi] == v) + lo
                out.append(int(cand[np.argmin(arr[cand])]))
            return out
        return dist, path_to

    def components(self, usable: np.ndarray) -> np.ndarray:
        """Connected components (undirected) of the network using only usable arcs. Labels 0..K-1."""
        g = self.g
        e = np.flatnonzero(usable)
        m = csr_matrix((np.ones(len(e), dtype=np.int8), (g.edge_tails[e], g.indices[e])), shape=(self.n, self.n))
        return connected_components(m, directed=False)[1]

    def forward_from(self, costs: Costs, source: int, limit: float = INF) -> list[float]:
        dist, _, _, _ = self._run(self.indptr, self.indices, costs.fwd, None, [(source, 0.0)], None, limit)
        return dist

    def nearest_target(self, costs: Costs, start: int, targets, limit: float = INF):
        """Nearest member of ``targets`` that can REACH ``start`` (a unit travelling to an incident).
        Returns (target_node, seconds, edge path from target to start) or (None, inf, [])."""
        tset = targets if isinstance(targets, (set, frozenset, dict)) else set(targets)
        if not tset:
            return None, INF, []
        dist, par, _, hit = self._run(self.rindptr, self.rindices, costs.rev, self.emap_l, [(start, 0.0)], tset, limit)
        if hit < 0:
            return None, INF, []
        # the reverse search stores, for each node, the ORIGINAL edge that leaves it towards ``start``
        path, v = [], hit
        while v != start:
            e = par[v]
            path.append(e)
            v = self.indices[e]
        return hit, dist[hit], path

    def nearest_forward(self, costs: Costs, start: int, targets, limit: float = INF):
        """Nearest member of ``targets`` reachable FROM ``start`` (an ambulance driving to a hospital)."""
        tset = targets if isinstance(targets, (set, frozenset, dict)) else set(targets)
        if not tset:
            return None, INF, []
        dist, par, _, hit = self._run(self.indptr, self.indices, costs.fwd, None, [(start, 0.0)], tset, limit)
        if hit < 0:
            return None, INF, []
        path, v = [], hit
        while v != start:
            e = par[v]
            path.append(e)
            v = self.tails[e]
        path.reverse()
        return hit, dist[hit], path

    def path(self, costs: Costs, src: int, dst: int, limit: float = INF):
        node, t, path = self.nearest_forward(costs, src, {dst}, limit) if src != dst else (dst, 0.0, [])
        return (path, t) if node is not None else ([], INF)
