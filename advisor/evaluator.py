"""Scores infrastructure ideas against the scenario portfolio, fast and exactly.

For every snapshot we keep, for each *service* (any hospital, major hospital, fire station, ambulance base), the travel
time from every node to the nearest open source on the damaged road network. An idea then changes those fields:

* a new facility at node c        ->  d'(x) = min( d(x), t(x -> c) )
* a new two-way link u <-> v (tau) ->  d'(x) = min( d(x), t(x->u) + tau + d(v), t(x->v) + tau + d(u) )
  (a shortest path uses a single new link at most once, so this is exact, not an approximation; tests/test_evaluator.py
  checks it against rebuilding the whole graph)
* removing existing roads / a facility -> recompute the field without them

"Everyday accessibility" uses the same trick on a peak-hour network with junction delays, measured as the average
drive time from every home to a set of main destinations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from advisor.portfolio import Snapshot
from sim.city import CityData
from sim.response import Resources, build_units
from sim.router import Router
from traffic.network import physical_degree

SERVICES = ("hosp", "t1", "fire", "amb")
SERVICE_LABEL = {"hosp": "any hospital within 30 min", "t1": "major hospital within 60 min", "fire": "fire engine within 8 min", "amb": "ambulance within 10 min"}
IDEA_SERVICES = {"hospital": ("hosp",), "hospital_major": ("hosp", "t1"), "fire_station": ("fire",), "ambulance_post": ("amb",)}


@dataclass
class SnapState:
    snap: Snapshot
    costs: object
    mat: object                       # reverse CSR matrix (distance TO targets)
    src: dict[str, list[int]]
    d: dict[str, np.ndarray]
    closed_tree: cKDTree | None = None


def _multi(mat, sources: list[int], limit: float = np.inf) -> np.ndarray:
    n = mat.shape[0]
    if not sources:
        return np.full(n, np.inf)
    return dijkstra(mat, directed=True, indices=np.asarray(sources, dtype=np.int64), min_only=True, limit=limit)


class Evaluator:
    def __init__(self, city: CityData, A: dict, portfolio: list[Snapshot], router: Router | None = None):
        self.city, self.A = city, A
        self.router = router or Router(city.g)
        self.pop, self.tot = city.pop, max(city.total_pop, 1.0)
        imp, adv = A["impact"], A["advisor"]
        self.sla = {"hosp": imp["slow_minutes"] * 60.0, "t1": imp["severe_minutes"] * 60.0, "fire": adv["sla_min"]["fire"] * 60.0, "amb": adv["sla_min"]["ambulance"] * 60.0}
        self.cap = imp["cap_minutes"] * 60.0
        self.common = float(adv["common_cause_m"])
        self.ev = adv["everyday"]
        self.portfolio = portfolio
        base_units = build_units(city, Resources(), A)
        self.amb_nodes = sorted({u.node for u in base_units if u.kind == "ambulance"})
        self.hosp_nodes = {f.id: f.node for f in city.hospitals(2)}
        self.snaps = [self._prepare(s) for s in portfolio]
        self._prepare_everyday()

    # ------------------------------------------------------------------------------------------------
    def sources(self, closed_fac: frozenset[int]) -> dict[str, list[int]]:
        c = self.city
        closed_nodes = {c.by_id[f].node for f in closed_fac}
        open_h = [f for f in c.hospitals(2) if f.id not in closed_fac]
        return {"hosp": sorted({f.node for f in open_h}), "t1": sorted({f.node for f in open_h if f.tier == 1}),
                "fire": sorted({f.node for f in c.of_kind("fire_station")}), "amb": [n for n in self.amb_nodes if n not in closed_nodes]}

    def _prepare(self, s: Snapshot) -> SnapState:
        costs = self.router.costs(s.F, s.bg)
        mat = self.router._matrix(costs.rev_arr, self.router._rplan)
        src = self.sources(s.closed_fac)
        return SnapState(s, costs, mat, src, {z: _multi(mat, src[z]) for z in SERVICES})

    def _closed_tree(self, st: SnapState) -> cKDTree | None:
        if st.closed_tree is None and len(st.snap.closed_edges):
            st.closed_tree = cKDTree(self.city.mid_xy[st.snap.closed_edges])
        return st.closed_tree

    def link_is_hit(self, st: SnapState, u: int, v: int) -> bool:
        """A new link is assumed damaged when it lies within the common-cause distance of roads the hazard closed."""
        tree = self._closed_tree(st)
        if tree is None:
            return False
        xy = self.city.xy
        pts = np.array([xy[u], xy[v], 0.5 * (xy[u] + xy[v])])
        return bool((tree.query(pts)[0] <= self.common).any())

    # ---- everyday accessibility ---------------------------------------------------------------------
    def _everyday_tt(self) -> np.ndarray:
        g = self.city.g
        delay = float(self.A["traffic"]["junction_delay_s"])
        junction = physical_degree(g) >= 3
        return g.tt_s + delay * np.where(g.is_arterial, 1.0, 0.5) * junction[g.indices]

    def _prepare_everyday(self) -> None:
        c, A = self.city, self.A
        g = c.g
        bg = A["context"]["time_of_day_speed"]["peak"] * A["context"]["weather_speed"]["clear"]
        self.ev_arr = self._everyday_tt() / bg
        self.ev_rev = self.ev_arr[self.router.emap]
        self.ev_mat = self.router._matrix(self.ev_rev, self.router._rplan)
        self.ev_cap = self.ev["cap_minutes"] * 60.0
        self.hubs = self.pick_hubs()
        self.H = self._hub_fields(self.ev_mat)                   # (K, N) seconds from x to hub k
        self.A0 = np.minimum(self.H, self.ev_cap).mean(axis=0)

    def pick_hubs(self) -> list[int]:
        """Main destinations: the most populated 500 m cells (kept >= 2.5 km apart) plus stations and major hospitals."""
        c = self.city
        order = np.argsort(-c.cell_pop)
        chosen: list[int] = []
        xy_c = []
        for cell in order:
            if len(chosen) >= self.ev["activity_centres"] or c.cell_pop[cell] <= 0:
                break
            members = np.flatnonzero(c.cell_id == cell)
            members = members[c.giant[members]]
            if not len(members):
                continue
            m = members[np.argmax(c.pop[members])]
            if all(np.hypot(*(c.xy[m] - p)) >= self.ev["centre_separation_m"] for p in xy_c):
                chosen.append(int(m)); xy_c.append(c.xy[m])
        extra = [l["node"] for l in c.landmarks if l["kind"] in ("station", "bus_station") and c.giant[l["node"]]]
        extra += [f.node for f in c.hospitals(1)]
        return sorted(set(chosen) | set(int(x) for x in extra))

    def _hub_fields(self, mat) -> np.ndarray:
        return np.asarray(dijkstra(mat, directed=True, indices=np.asarray(self.hubs, dtype=np.int64), limit=self.ev_cap * 1.5), dtype=np.float32)

    # ---- measures ----------------------------------------------------------------------------------------
    def measure(self, d: dict[str, np.ndarray]) -> dict[str, float]:
        out = {}
        for z, dz in d.items():
            out[f"{z}_within"] = float(self.pop[dz <= self.sla[z]].sum())
            out[f"{z}_mean"] = float((self.pop * np.minimum(dz, self.cap)).sum() / self.tot / 60.0)
        return out

    def everyday_delta(self, A_new: np.ndarray) -> dict[str, float]:
        gain = self.A0 - A_new
        w = float(self.ev["trips_per_person_per_day"])
        return {"hub_minutes": float((self.pop * gain).sum() / self.tot / 60.0),
                "hub_people_2min": float(self.pop[gain >= 120.0].sum()),
                "hub_hours_per_day": float((self.pop * gain).sum() / 3600.0 * w)}

    # ---- ideas -------------------------------------------------------------------------------------------
    def _blank(self, st: SnapState) -> dict[str, float]:
        return {"name": st.snap.name, "kind": st.snap.kind, "weight": st.snap.weight}

    def _reach(self, st: SnapState, node: int) -> np.ndarray:
        return dijkstra(st.mat, directed=True, indices=int(node), limit=self.cap)

    def eval_facility(self, idea: str, node: int) -> dict:
        """A new hospital / fire station / ambulance post at ``node``."""
        services = IDEA_SERVICES[idea]
        rows = []
        for st in self.snaps:
            R = self._reach(st, node)
            new = {z: np.minimum(st.d[z], R) if z in services else st.d[z] for z in SERVICES}
            rows.append(self._diff(st, st.d, new, services))
        return self.aggregate(rows)

    def link_fields(self, st: SnapState, u: int, v: int, tau_s: float) -> dict[str, np.ndarray]:
        """``tau_s`` is the free-flow drive time; the new road is slowed by this snapshot's traffic/weather like every other road."""
        tau = tau_s / st.snap.bg
        Ru, Rv = self._reach(st, u), self._reach(st, v)
        return {z: np.minimum(st.d[z], np.minimum(Ru + tau + st.d[z][v], Rv + tau + st.d[z][u])) for z in SERVICES}

    def eval_link(self, u: int, v: int, tau_s: float) -> dict:
        """A new two-way road u <-> v whose free-flow drive time is tau_s seconds."""
        rows = []
        for st in self.snaps:
            if self.link_is_hit(st, u, v):
                row = self._diff(st, st.d, st.d, SERVICES)
                row["hit"] = 1.0
                rows.append(row)
                continue
            rows.append(self._diff(st, st.d, self.link_fields(st, u, v, tau_s), SERVICES))
        agg = self.aggregate(rows)
        agg["everyday"] = self.everyday_link(u, v, tau_s / self.A["context"]["time_of_day_speed"]["peak"])
        return agg

    def everyday_link(self, u: int, v: int, tau_ev_s: float) -> dict:
        Ru = dijkstra(self.ev_mat, directed=True, indices=int(u), limit=self.ev_cap * 1.5)
        Rv = dijkstra(self.ev_mat, directed=True, indices=int(v), limit=self.ev_cap * 1.5)
        via_u = Ru[None, :] + (tau_ev_s + self.H[:, v])[:, None]
        via_v = Rv[None, :] + (tau_ev_s + self.H[:, u])[:, None]
        Hn = np.minimum(self.H, np.minimum(via_u, via_v))
        return self.everyday_delta(np.minimum(Hn, self.ev_cap).mean(axis=0))

    def eval_remove_edges(self, edges: np.ndarray, snaps: list[int] | None = None) -> dict:
        """What the city loses if these roads (both directions) were not there. Positive numbers = harm."""
        edges = np.asarray(edges, dtype=np.int64)
        g = self.city.g
        both = np.unique(np.concatenate([edges, g.rev[edges][g.rev[edges] >= 0]]))
        idx = range(len(self.snaps)) if snaps is None else snaps
        rows = []
        for i in idx:
            st = self.snaps[i]
            F = np.ones(g.n_edges, dtype=np.float32) if st.snap.F is None else st.snap.F.copy()
            F[both] = 0
            costs = self.router.costs(F, st.snap.bg)
            mat = self.router._matrix(costs.rev_arr, self.router._rplan)
            new = {z: _multi(mat, st.src[z]) for z in SERVICES}
            rows.append(self._diff(st, st.d, new, SERVICES))
        agg = self.aggregate(rows)
        ev_rev = self.ev_rev.copy()
        ev_rev[np.isin(self.router.emap, both)] = np.inf
        Hn = self._hub_fields(self.router._matrix(ev_rev, self.router._rplan))
        agg["everyday"] = self.everyday_delta(np.minimum(Hn, self.ev_cap).mean(axis=0))
        return agg

    def eval_remove_facility(self, idea_service: str, fac_id: int, snaps: list[int] | None = None) -> dict:
        """What the city loses if this existing hospital were not there (positive = harm)."""
        node = self.hosp_nodes[fac_id]
        services = IDEA_SERVICES[idea_service]
        idx = range(len(self.snaps)) if snaps is None else snaps
        rows = []
        for i in idx:
            st = self.snaps[i]
            new = dict(st.d)
            for z in services:
                new[z] = _multi(st.mat, [n for n in st.src[z] if n != node])
            rows.append(self._diff(st, st.d, new, services))
        return self.aggregate(rows)

    # ---- bookkeeping ---------------------------------------------------------------------------------------
    def _diff(self, st: SnapState, old: dict, new: dict, services) -> dict:
        row = self._blank(st)
        m0 = self.measure({z: old[z] for z in services})
        m1 = self.measure({z: new[z] for z in services})
        for k in m0:
            row[k] = m1[k] - m0[k]                     # improvement: within-people go up, mean-minutes go down
        return row

    def aggregate(self, rows: list[dict]) -> dict:
        keys = [k for k in rows[0] if k not in ("name", "kind", "weight", "hit")]
        W = np.array([r["weight"] for r in rows])
        W = W / W.sum() if W.sum() > 0 else W
        out = {"per_snapshot": rows, "weighted": {k: float(sum(w * r[k] for w, r in zip(W, rows))) for k in keys}}
        out["normal"] = {k: rows[0][k] for k in keys} if rows[0]["kind"] == "baseline" else {}
        dis = [(w, r) for w, r in zip(W, rows) if r["kind"] != "baseline"]
        tw = sum(w for w, _ in dis)
        out["disaster_avg"] = {k: float(sum(w * r[k] for w, r in dis) / tw) if tw > 0 else 0.0 for k in keys}
        by_kind: dict[str, dict] = {}
        for kind in sorted({r["kind"] for r in rows}):
            rk = [r for r in rows if r["kind"] == kind]
            by_kind[kind] = {k: float(np.mean([r[k] for r in rk])) for k in keys}
        out["by_kind"] = by_kind
        out["n_hit"] = int(sum(r.get("hit", 0) for r in rows))
        return out
