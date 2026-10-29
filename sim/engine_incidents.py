"""Action: incident generation and expiry.

Converts hazards into 911-style calls: a Poisson process scaled by population and hazard exposure generates
background and hazard-driven incidents every state step, plus each hazard can emit its own scripted incidents
(a fire's ignition call, casualties at a collapse). Calls wait in a time-ordered heap (``self.future``) until
their minute arrives, then become live ``Incident`` objects that dispatch (see ``sim/engine_dispatch.py``) can
pick up; a call that waits too long for a unit expires as "unserved".

If a professor asks "where do the emergency calls come from?" or "what happens if nobody answers a call in
time?" -- this is the file.
"""
from __future__ import annotations

import heapq

import numpy as np

from sim.hazards import IncidentSpec
from sim.response import Incident


class IncidentMixin:
    """Mixed into ``sim.engine.Simulation``. Owns incident generation, release, linking, and expiry."""

    # WorldView ---------------------------------------------------------------------------------------
    def engines_on_scene(self, tag: str) -> int:
        """Answers a hazard's question "how many fire engines have reached the incident tagged `tag`?" --
        this is how a Fire hazard knows when it has been contained (see sim/hazards/fire.py)."""
        inc = self.tag_inc.get(tag)
        return inc.arrived if inc is not None and inc.kind == "fire" else 0

    # ---- incidents -------------------------------------------------------------------------------
    def _plan_incidents(self, t0: int, t1: int) -> None:
        """Queue every incident due in [t0, t1): hazard-scripted ones, plus a Poisson draw of background +
        hazard-boosted medical/fire/rescue calls per populated node."""
        A, city = self.A, self.city
        I = A["incidents"]
        for h in self.hazards:
            for spec in h.incidents(float(t0), float(t1)):
                heapq.heappush(self.future, (spec.t_min, len(self.future), spec))
        dt_h = (t1 - t0) / 60.0
        lam = city.pop / 1000.0 * (I["baseline_per_1000_per_hour"] + I["hazard_boost_per_1000_per_hour"] * self.expo) * self.sc.incident_scale
        counts = self.rng.poisson(lam * dt_h)
        for node in np.flatnonzero(counts):
            di = int(self.dom[node])
            mixkey = self.hazards[di].kind if (di >= 0 and self.expo[node] > 0.05) else "baseline"
            mix = I["mix"].get(mixkey, I["mix"]["baseline"])
            kinds, p = list(mix), np.array(list(mix.values()), dtype=float)
            for _ in range(int(counts[node])):
                kind = str(self.rng.choice(kinds, p=p / p.sum()))
                sev = int(self.rng.choice([1, 2, 3], p=I["medical_severity_probs"] if kind != "fire" else [0.55, 0.35, 0.10]))
                spec = IncidentSpec(kind, int(node), sev, t0 + float(self.rng.uniform(0, t1 - t0)), mixkey, needs=(1 if sev == 1 or kind != "fire" else 2))
                heapq.heappush(self.future, (spec.t_min, len(self.future), spec))

    def _release_future(self, t: int) -> None:
        """Pop every queued incident whose minute has arrived and turn it into a live ``Incident``."""
        cap = self.A["incidents"]["max_incidents"]
        while self.future and self.future[0][0] <= t + 1e-9:
            _, _, spec = heapq.heappop(self.future)
            if len(self.incidents) >= cap and spec.tag in ("baseline", "earthquake", "storm", "fire", "plume", "flood_zone", "custom", "structure_failure", "accident") and not self.tag_inc_link(spec):
                continue
            inc = Incident(len(self.incidents), spec.kind, spec.node, spec.severity, float(spec.t_min), spec.tag, spec.needs)
            self.incidents.append(inc)
            if spec.tag and self.tag_inc_link(spec):
                self.tag_inc[spec.tag] = inc

    def tag_inc_link(self, spec: IncidentSpec) -> bool:
        """True when ``spec``'s tag names one of the running hazards -- i.e. this incident is the hazard's own
        scripted one (a fire's ignition), which gets tracked in ``self.tag_inc`` so ``engines_on_scene`` can find it."""
        return any(h.name == spec.tag for h in self.hazards)

    def _expire(self, t: int) -> None:
        """A call still waiting past the max response wait is written off as unserved."""
        w = self.A["response"]["max_wait_min"]
        for inc in self.incidents:
            if inc.state == "pending" and t - inc.t > w:
                inc.state = "unserved"
