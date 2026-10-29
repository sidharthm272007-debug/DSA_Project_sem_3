"""Action: unit movement, live re-routing, and stage transitions.

Advances every moving unit along its current path one simulated minute at a time, edge by edge, tracking the
exact second it entered and left each edge (so a replayed trip looks continuous even though the simulation
only steps in whole minutes). If the next edge on a unit's path has just closed, it re-plans from where it
actually is -- not from scratch -- and if nothing reaches the destination at all, it waits and tries again a
few minutes later. Also handles what happens at each arrival: reaching a scene, handing a patient to a
hospital (re-routing to another one if the first becomes unreachable), finishing on-scene service, and sending
a unit back to base.

If a professor asks "what happens when an ambulance's route closes while it's driving?" or "how do you avoid
teleporting units -- how does partial progress along a road carry over one minute to the next?" -- this is
the file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sim.repair import RepairTarget
from sim.response import Incident, Unit


@dataclass
class Leg:
    """One unit's uninterrupted trip (e.g. "ambulance #4, enroute to scene") -- the unit of replay for the map."""
    unit: int
    kind: str
    leg: str
    incident: int | None
    edges: list[tuple[int, float, float]]


class MovementMixin:
    """Mixed into ``sim.engine.Simulation``. Owns per-minute movement, re-routing, and arrival handling."""

    # ---- movement --------------------------------------------------------------------------------
    def _move(self, t: int) -> None:
        for u in self.units:
            if u.state in ("enroute", "transport", "returning"):
                status = self._advance(u, t)
                if status == "blocked":
                    self._replan(u, t)
                elif status == "arrived":
                    self._arrive(u, t)
            elif u.state in ("onscene", "handover", "working") and t >= u.busy_until:
                self._finish_stage(u, t)

    def _advance(self, u: Unit, t: int) -> str:
        """Spend this minute's 60 seconds moving ``u`` along its path, edge by edge, carrying over whatever
        time is left on a partially-travelled edge into the next minute. Returns "blocked" if the next edge
        has zero capacity right now, "arrived" if the path is exhausted, else "moving"."""
        rem = 60.0
        cost, idx = self.costs.fwd, self.router.indices
        t0 = t * 60.0
        while rem > 0 and u.pi < len(u.path):
            e = u.path[u.pi]
            if u.left_s is None:
                c = cost[e]
                if c == math.inf:
                    u.blocked_wait_s += rem
                    return "blocked"
                u.left_s = c
                u.edge_in_s = t0 + (60.0 - rem)
            if u.left_s <= rem:
                rem -= u.left_s
                u.trip.append((e, u.edge_in_s, t0 + (60.0 - rem)))          # actual finish time (an edge can span several minutes)
                u.node = idx[e]
                u.pi += 1
                u.left_s = None
            else:
                u.left_s -= rem
                rem = 0
        u.busy_min += 1
        return "arrived" if u.pi >= len(u.path) else "moving"

    def _dest(self, u: Unit) -> int:
        if u.leg == "return":
            return u.base
        if u.leg == "to_repair":
            return int(u.target.tail)          # type: ignore[union-attr]
        return int(u.target)                   # type: ignore[arg-type]

    def _replan(self, u: Unit, t: int) -> None:
        """A road ahead closed. Find another way now; if there is none, wait and look again in a few minutes."""
        if t < u.replan_at:
            return
        path, sec = self.router.path(self.costs, u.node, self._dest(u))
        if path or u.node == self._dest(u):
            self._close_leg(u, t)
            u.path, u.pi, u.left_s, u.blocked_wait_s = path, 0, None, 0.0
            return
        if u.leg == "transport":                           # hospital became unreachable: try another one
            node, sec, path = self.router.nearest_forward(self.costs, u.node, self.open_any)
            if node is not None:
                self._close_leg(u, t)
                u.target, u.path, u.pi, u.left_s = node, path, 0, None
                u.blocked_wait_s = 0.0
                return
        u.replan_at = t + 3

    def _arrive(self, u: Unit, t: int) -> None:
        R = self.A["response"]
        self._close_leg(u, t + 1)
        if u.kind == "crew":
            if u.leg == "to_repair":
                tg: RepairTarget = u.target  # type: ignore[assignment]
                u.state, u.busy_until = "working", t + tg.clear_min
            else:
                u.state, u.leg = "idle", ""
            return
        inc = self.incidents[u.incident] if u.incident is not None else None
        if u.leg == "to_scene" and inc is not None:
            inc.arrived += 1
            if inc.t_arrive is None:
                inc.t_arrive = float(t + 1)
            u.state = "onscene"
            u.busy_until = 10 ** 9                                   # released when the incident is served
            if inc.kind == "fire":
                if inc.arrived >= inc.needs:
                    inc.state, inc.service_end = "onscene", t + R["service_min"]["fire"][inc.severity - 1]
                    for uid in inc.units:
                        self.units[uid].busy_until = inc.service_end
            elif inc.kind == "medical":
                u.busy_until = t + R["service_min"]["medical"][inc.severity - 1]
                inc.state = "onscene"
            else:
                u.busy_until = t + R["service_min"]["rescue"][inc.severity - 1]
                inc.state = "onscene"
        elif u.leg == "transport" and inc is not None:
            u.state, u.busy_until = "handover", t + R["hospital_handover_min"]
        elif u.leg == "return":
            u.state, u.leg, u.incident = "idle", "", None

    def _finish_stage(self, u: Unit, t: int) -> None:
        """A unit's dwell time (on-scene service, hospital handover, or debris clearance) is up -- move it to
        the next stage: transport to hospital, close out a served incident and send everyone home, or (for a
        repair crew) mark the road clear and refresh the whole network's costs."""
        inc = self.incidents[u.incident] if u.incident is not None else None
        if u.state == "working":
            tg: RepairTarget = u.target  # type: ignore[assignment]
            for e in tg.edges:
                self.repaired[e] = True
            self._claimed.discard(tg.edges[0])
            self._repairs_since_plan += 1
            self.repair_log.append({"t": float(t), "edges": tg.edges, "name": tg.name or self.city.edge_name(tg.edges[0]), "people": tg.people,
                                    "clear_min": tg.clear_min, "connects": tg.connects})
            self._refresh_costs()
            u.state, u.leg, u.target = "idle", "", None
            return
        if inc is None:
            u.state = "idle"
            return
        if u.state == "onscene" and inc.kind == "medical":
            self._start_transport(u, inc, t)
        elif u.state == "onscene":                                   # fire / rescue served
            if inc.state != "done":
                inc.state, inc.t_done = "done", float(t)
            for uid in inc.units:
                v = self.units[uid]
                if v.state == "onscene":
                    self._go_home(v, t)
        elif u.state == "handover":
            inc.state, inc.t_done, inc.t_delivered = "done", float(t), float(t)
            self._go_home(u, t)

    def _start_transport(self, u: Unit, inc: Incident, t: int) -> None:
        """Route a treated patient to the nearest hospital that can take them: a tier-1 (trauma) hospital first
        for the most severe cases, falling back to any open hospital, and marking the patient stranded if none
        is reachable at all."""
        targets = self.open_any
        node = None
        if inc.severity == 3 and self.open_t1:
            node, sec, path = self.router.nearest_forward(self.costs, u.node, self.open_t1, limit=45 * 60.0)
        if node is None:
            node, sec, path = self.router.nearest_forward(self.costs, u.node, targets, limit=120 * 60.0)
        if node is None:
            inc.state, inc.t_done = "stranded", float(t)
            self._go_home(u, t)
            return
        inc.state, inc.hospital = "transport", node
        u.state, u.leg, u.target, u.path, u.pi, u.left_s, u.trip = "transport", "transport", node, path, 0, None, []

    def _go_home(self, u: Unit, t: int) -> None:
        path, sec = self.router.path(self.costs, u.node, u.base) if u.node != u.base else ([], 0.0)
        if u.node != u.base and not path:
            u.state, u.leg, u.incident = "idle", "", None
            return
        u.state, u.leg, u.target, u.path, u.pi, u.left_s, u.trip = "returning", "return", None, path, 0, None, []
        if not path:
            u.state, u.leg, u.incident = "idle", "", None

    def _close_leg(self, u: Unit, t: int) -> None:
        """Flush ``u``'s in-progress trip into the completed-legs list (used for the route replay), ready for
        the next leg to start a fresh trip."""
        if u.trip:
            self.legs.append(Leg(u.id, u.kind, u.leg, u.incident, list(u.trip)))
        u.trip = []

    def _close_legs(self, t: int) -> None:
        for u in self.units:
            self._close_leg(u, t)
