"""Action: repair-crew coordination.

Decides what idle debris-clearance crews should clear next. Rather than re-planning from scratch on every
call, it keeps a standing repair plan (see ``sim.repair.plan_repairs`` -- a connectivity-first, Kruskal/
Union-Find-style ordering) and only recomputes it when enough has changed (new damage, a burst of repairs, or
the plan going stale). Each idle crew then does one routed search from its own position and claims the
nearest still-open job on the plan that it can actually reach, so two crews never race for the same road.

If a professor asks "how do you decide which road gets repaired first?" or "why isn't the repair plan
recomputed every single minute?" -- this is the file. The actual ranking algorithm lives in ``sim/repair.py``;
this file only decides when to ask for a new plan and how crews pick jobs off it.
"""
from __future__ import annotations

import numpy as np

from sim.repair import plan_repairs


class RepairMixin:
    """Mixed into ``sim.engine.Simulation``. Owns repair-plan refresh and crew-to-job assignment."""

    # ---- repair crews ----------------------------------------------------------------------------
    def _crews(self, t: int) -> None:
        crews = [u for u in self.units if u.kind == "crew"]
        idle = [u for u in crews if u.state == "idle" and u.replan_at <= t]
        if not idle or t % 2:
            return
        n_now = int((self.rep_mask & (self.F == 0)).sum())
        if n_now == 0:
            return
        stale = t - self._plan_t >= 30 or self._repairs_since_plan >= 5 or n_now != self._plan_n
        if stale and (not self._plan or t - self._plan_t >= 10):
            self._plan = plan_repairs(self.city, self.F, self.rep_mask, self.rep_mult, self.A)
            self._plan_t, self._plan_n, self._repairs_since_plan = t, n_now, 0
        open_jobs = [tg for tg in self._plan if tg.edges[0] not in self._claimed and not self.repaired[tg.edges[0]]]
        if not open_jobs:
            return
        for u in idle:
            dist, path_to = self.router.reach_from(self.costs, u.node, limit=3 * 3600.0)     # one search per crew, not one per job
            for tg in open_jobs:
                if tg.edges[0] in self._claimed or not np.isfinite(dist[tg.tail]):
                    continue
                path = path_to(tg.tail) if u.node != tg.tail else []
                self._claimed.add(tg.edges[0])
                u.state, u.leg, u.target, u.path, u.pi, u.left_s, u.trip = "enroute", "to_repair", tg, path, 0, None, []
                if not path:
                    self._arrive(u, t)
                break
            else:
                u.replan_at = t + 10                        # nothing reachable right now: look again later
