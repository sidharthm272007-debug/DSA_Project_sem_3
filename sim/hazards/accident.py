"""Major road accident: blocks a handful of arcs around the impact point along the busiest road through it,
then models queue spillback upstream with a reverse Dijkstra by distance -- traffic backs up along the
approach roads, slowing (not stopping) them in proportion to how close to the blockage they are.
"""
from __future__ import annotations

import heapq

import numpy as np

from sim.hazards.base import Hazard, HazardContext, IncidentSpec


class Accident(Hazard):
    kind = "accident"
    title = "Major road accident"

    def __init__(self, lat: float, lon: float, severity: int = 3, start_min: float = 0.0, duration_min: float = 120.0, name: str = ""):
        super().__init__(name or f"Road accident (severity {severity})", start_min)
        self.lat, self.lon, self.sev, self.duration = lat, lon, int(min(5, max(1, severity))), duration_min

    def reset(self, ctx: HazardContext) -> None:
        super().reset(ctx)
        c, g = ctx.city, ctx.city.g
        A = ctx.A["hazards"]["accident"]
        node = c.snap(self.lat, self.lon)[0]
        self.node = node
        k = A["blocked_edges_by_severity"][self.sev - 1]
        # block the arcs leaving/entering the impact node along the busiest road, then the next ones along it
        cand = [e for e in list(g.out_edges(node))]
        cand += list(np.flatnonzero(g.indices == node))
        seen: list[int] = []
        frontier = [int(e) for e in cand]
        while frontier and len(seen) < 2 * k:
            e = frontier.pop(0)
            if e in seen:
                continue
            seen.append(e)
            r = int(g.rev[e])
            if r >= 0 and r not in seen:
                seen.append(r)
            head = int(g.indices[e])
            nxt = [int(x) for x in g.out_edges(head) if int(g.indices[x]) != int(g.edge_tails[e])]
            same = [x for x in nxt if g.way_id[x] == g.way_id[e]]
            frontier += same[:1]
        self.blocked = np.array(seen[: max(2, 2 * k)], dtype=np.int64)
        self.mask = np.zeros(self.E, dtype=bool)
        self.mask[self.blocked] = True
        # queue spillback upstream: reverse Dijkstra by length from the blocked edges' tails
        self.qf = np.ones(self.E, dtype=np.float32)
        L, qmin = A["queue_length_m"], A["queue_min_factor"]
        rg, emap = g.reverse()
        dist = {}
        heap = [(0.0, int(g.edge_tails[e])) for e in self.blocked]
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if u in dist or d > L:
                continue
            dist[u] = d
            for j in range(rg.indptr[u], rg.indptr[u + 1]):
                v = int(rg.indices[j])
                if v not in dist:
                    heapq.heappush(heap, (d + float(rg.length_m[j]), v))
                    e = int(emap[j])                        # original arc v -> u carries the queue
                    self.qf[e] = min(self.qf[e], qmin + (1 - qmin) * min(1.0, d / L))
        self._emitted = False

    def active(self) -> bool:
        return self.start_min <= self.t <= self.start_min + self.duration

    def factor(self) -> np.ndarray | None:
        if not self.active():
            return None
        f = self.qf.copy()
        f[self.mask] = 0.0
        return f

    def repairable(self) -> np.ndarray | None:
        return None

    def incidents(self, t0: float, t1: float) -> list[IncidentSpec]:
        if self._emitted or not (t0 <= self.start_min < t1 + 1e-9):
            return []
        self._emitted = True
        n = 1 + self.sev // 2
        return [IncidentSpec("medical", self.node, min(3, 1 + self.sev // 2), self.start_min + i, self.name) for i in range(n)]

    def footprint(self) -> list[dict]:
        if not self.active():
            return []
        return [{"type": "circle", "kind": "accident", "level": 3, "lat": self.lat, "lon": self.lon, "radius_m": 90.0}]

    def location(self) -> tuple[float, float]:
        return self.lat, self.lon
