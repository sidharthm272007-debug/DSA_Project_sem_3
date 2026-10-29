"""Action: hazard / network state update.

Every ``state_step_min`` minutes the simulation asks each active hazard how much of the road network it is
currently affecting, folds that into one per-road speed factor and one per-node exposure field, re-costs the
graph for routing, and records a metrics/frame snapshot for the timeline charts and the map replay.

If a professor asks "how does a hazard actually change the road network minute by minute?" or "how is the
timeline chart data produced?" -- this is the file. ``Simulation._state_update`` is called once per state step
from ``sim.engine.Simulation.run``; everything else here is a helper it calls.
"""
from __future__ import annotations

import numpy as np

from sim.hazards import HazardContext
from sim.metrics import catchments, compute_access, summarize


class StateMixin:
    """Mixed into ``sim.engine.Simulation``. Owns hazard aggregation, re-costing, and metrics recording."""

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
        """Called after a repair crew clears a road: re-cost the graph without waiting for the next state step."""
        F = self.F.copy()
        rep = self.rep_mask | self.repaired
        F[self.repaired & rep] = np.where(self.F[self.repaired & rep] == 0, 1.0, self.F[self.repaired & rep])
        self.costs = self.router.costs(F, self.bg)
        self.F = F

    def _record(self, t: int) -> None:
        """Snapshot one row of metrics (for the timeline) and one map frame (for the replay) at minute ``t``."""
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

