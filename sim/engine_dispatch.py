"""Action: unit dispatch / assignment.

Matches waiting incidents to idle units: most-severe-and-oldest call first, nearest eligible idle unit by
actual routed travel time (not straight-line distance) on the current, possibly-damaged road network. A call
that cannot be matched right away is retried a few minutes later rather than held open forever.

If a professor asks "how does the simulator decide which ambulance/engine answers which call?" -- this is the
file. The actual routing search is ``sim.router.Router.nearest_target``; this file only decides eligibility and
ordering.
"""
from __future__ import annotations

from sim.response import Incident, Unit


class DispatchMixin:
    """Mixed into ``sim.engine.Simulation``. Owns the pending-call -> idle-unit matching."""

    # ---- dispatch --------------------------------------------------------------------------------
    def _dispatch(self, t: int) -> None:
        R = self.A["response"]
        pend = [i for i in self.incidents if i.state in ("pending", "responding") and len(i.units) < i.needs and i.retry_at <= t]
        if not pend:
            return
        pend.sort(key=lambda i: (-i.severity, i.t))                  # most severe, then oldest, goes first
        elig = {"medical": ["ambulance"], "fire": ["fire"], "rescue": ["rescue", "fire"]}
        for inc in pend:
            while len(inc.units) < inc.needs:
                unit = self._assign(inc, elig[inc.kind], t)
                if unit is None:
                    inc.retry_at = t + (R["retry_min"] if inc.state == "pending" else 3)
                    break

    def _assign(self, inc: Incident, kinds: list[str], t: int) -> Unit | None:
        """Try each eligible unit kind in priority order; within a kind, route-distance to every idle unit's
        node and take the nearest. Returns the assigned unit, or None if nothing of any eligible kind is free
        or reachable."""
        for kind in kinds:
            idle = [u for u in self.units if u.kind == kind and u.state == "idle"]
            if not idle:
                continue
            by_node: dict[int, list[Unit]] = {}
            for u in idle:
                by_node.setdefault(u.node, []).append(u)
            node, sec, path = self.router.nearest_target(self.costs, inc.node, by_node)
            if node is None:
                continue
            u = by_node[node][0]
            u.state, u.path, u.pi, u.left_s, u.incident, u.leg, u.target = "enroute", path, 0, None, inc.id, "to_scene", inc.node
            u.trip = []
            inc.units.append(u.id)
            inc.state = "responding"
            if inc.t_assign is None:
                inc.t_assign = float(t)
            return u
        return None
