"""The advisor: one object that answers the four planning questions.

  1. Where should the next hospital / fire station / ambulance post go?          -> recommend_sites
  2. Where would a new road or bridge help most?                                  -> recommend_links
  3. I have an idea at this spot (or between these two points): is it worth it?   -> assess_facility / assess_link
  4. Is what we already have actually useful?                                     -> assess_structure / assess_existing_facility

All of them are scored against the same scenario portfolio, so answers are comparable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from advisor.evaluator import IDEA_SERVICES, SERVICES, Evaluator
from advisor.links import LinkIdea, generate_links, link_cost_crore, link_geometry, score_links
from advisor.portfolio import Snapshot, assign_weights, build_portfolio
from advisor.siting import PRIMARY, SitePick, candidate_sites, greedy_sites
from advisor.verdict import Verdict, care_people, judge_build, judge_existing
from sim.city import CityData
from sim.router import Router

IDEA_LABEL = {"hospital": "General hospital", "hospital_major": "Major hospital", "fire_station": "Fire station", "ambulance_post": "Ambulance post"}
IDEA_COST_KEY = {"hospital": "hospital", "hospital_major": "hospital_major", "fire_station": "fire_station", "ambulance_post": "ambulance_post"}


@dataclass
class SiteRec:
    pick: SitePick
    cost_crore: float
    verdict: Verdict


@dataclass
class LinkRec:
    idea: LinkIdea
    verdict: Verdict


@dataclass
class Assessment:
    title: str
    verdict: Verdict
    effect: dict
    cost_crore: float = 0.0
    extra: dict | None = None


class Advisor:
    def __init__(self, city: CityData, A: dict, weights: dict | None = None, seed: int | None = None,
                 router: Router | None = None, portfolio: list[Snapshot] | None = None):
        self.city, self.A = city, A
        self.router = router or Router(city.g)
        self.portfolio = portfolio or build_portfolio(city, A, weights=weights, seed=seed)
        self.ev = Evaluator(city, A, self.portfolio, self.router)
        self._cache: dict = {}
        n_fire = len(city.of_kind("fire_station"))
        self.n_fire = n_fire
        self.fire_ok = n_fire >= int(A["advisor"]["min_fire_stations"])
        # the services whose numbers drive verdicts for roads/bridges/existing structures
        self.services_all = ("hosp", "amb") + (("fire",) if self.fire_ok else ())

    def data_notes(self, fire_idea: bool = False, link: bool = False) -> list[str]:
        """Plain statements about the data behind a result. They are appended to every verdict."""
        notes = []
        if not self.fire_ok or fire_idea:
            notes.append(f"Only {self.n_fire} fire station(s) are mapped in this data, far fewer than the city really has. "
                         + ("Fire-service gains are shown but left out of the headline numbers. " if not self.fire_ok else "")
                         + "Add the missing stations to data/coimbatore/facilities_extra.csv to make fire results decision-grade.")
        if link:
            notes.append("Roads are screened on OpenStreetMap roads only. Service, campus and informal roads are not in the network, and railways are barriers, so check on the ground whether a connection "
                         "already exists or a rail crossing is the real need before believing a suggestion.")
        notes.append("Ambulance bases are assumed (3 per major hospital, the rest spread over other hospitals); the real 108 posts are not mapped.")
        approx = [f.name for f in self.city.facilities if f.approx and f.kind in ("hospital", "fire_station")]
        if approx:
            notes.append("Approximate positions (not in OpenStreetMap): " + ", ".join(approx) + ".")
        notes.append(self.city.meta.get("population_source", "Population is a proxy, not a census grid.").replace("OSM", "OpenStreetMap"))
        return notes

    def _finish(self, v: Verdict, fire_idea: bool = False, link: bool = False) -> Verdict:
        v.caveats.extend(self.data_notes(fire_idea, link))
        return v

    def fork(self) -> "Advisor":
        """A cheap copy with its own scenario weights and its own result cache, so two users on one server cannot disturb each other.
        Everything heavy (distance fields, matrices, road graph) is shared read-only."""
        new = copy.copy(self)
        new.portfolio = [replace(s) for s in self.portfolio]
        ev = copy.copy(self.ev)
        ev.portfolio = new.portfolio
        ev.snaps = [replace(st, snap=sn) for st, sn in zip(self.ev.snaps, new.portfolio)]
        new.ev = ev
        new._cache = {}
        new._wkey = "unset"
        return new

    # ---- importance weights ------------------------------------------------------------------------
    def reweight(self, weights: dict | None) -> None:
        """Change how much each kind of scenario matters (kind -> relative importance). Cached answers are dropped."""
        if weights is not None and sum(max(float(v), 0.0) for v in weights.values()) <= 0:
            raise ValueError("At least one kind of situation must have a positive weight.")
        key = None if weights is None else tuple(sorted((k, round(float(v), 6)) for k, v in weights.items()))
        if getattr(self, "_wkey", "unset") == key:
            return
        self._wkey = key
        assign_weights(self.portfolio, weights, self.A)
        self._cache.clear()

    def weights_now(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.portfolio:
            out[s.kind] = out.get(s.kind, 0.0) + s.weight
        return out

    @property
    def n_scen(self) -> int:
        return len(self.portfolio)

    def cost_of(self, idea: str) -> float:
        return float(self.A["advisor"]["cost_crore"][IDEA_COST_KEY[idea]])

    # ---- 1. where to build a facility ---------------------------------------------------------------
    def need_map(self, idea: str) -> dict:
        """Per 500 m cell: how many people are, on weighted average across the scenarios, outside the standard for this service."""
        z, ev, c = PRIMARY[idea], self.ev, self.city
        unserved = np.zeros(c.g.n_nodes)
        for st in ev.snaps:
            unserved += st.snap.weight * (st.d[z] > ev.sla[z])
        people = np.bincount(c.cell_id, weights=c.pop * unserved, minlength=c.n_cells)
        keep = np.flatnonzero(people > 20)
        return {"lat": c.cell_lat[keep], "lon": c.cell_lon[keep], "people": people[keep]}

    def recommend_sites(self, idea: str, k: int = 5, n_candidates: int | None = None, progress=None) -> list[SiteRec]:
        key = ("sites", idea, k, n_candidates, tuple(round(s.weight, 6) for s in self.portfolio))
        if key in self._cache:
            return self._cache[key]
        n = n_candidates or int(self.A["advisor"]["candidate_counts"]["facility"])
        cands = candidate_sites(self.ev, idea, n)
        picks = greedy_sites(self.ev, idea, cands, k, progress=progress)
        cost = self.cost_of(idea)
        z = (PRIMARY[idea],)
        peers = [care_people(p.effect["weighted"], z) / cost for p in picks] if cost > 0 else None
        recs = [SiteRec(p, cost, self._finish(judge_build(self.A, p.effect, z, cost, primary=z[0], peer_per_crore=peers,
                                                          what=f"Site #{p.rank} ({p.where})", n_scen=self.n_scen), idea == "fire_station")) for p in picks]
        self._cache[key] = recs
        return recs

    def assess_facility(self, idea: str, lat: float, lon: float) -> Assessment:
        node, snap_m = self.city.snap(lat, lon)
        eff = self.ev.eval_facility(idea, node)
        z = (PRIMARY[idea],)
        cost = self.cost_of(idea)
        where = self.city.describe_node(node)
        v = self._finish(judge_build(self.A, eff, z, cost, primary=z[0], what=f"A {IDEA_LABEL[idea].lower()} at {where}", n_scen=self.n_scen), idea == "fire_station")
        if snap_m > 400:
            v.caveats.insert(0, f"The spot you picked is {snap_m:.0f} m from the nearest mapped road; it was moved to the closest junction.")
        return Assessment(f"{IDEA_LABEL[idea]} at {where}", v, eff, cost, {"node": node, "lat": float(self.city.g.lat[node]), "lon": float(self.city.g.lon[node]), "snap_m": snap_m})

    def better_nearby(self, idea: str, node: int, n: int = 8) -> dict:
        """Is there a clearly better spot close by? Tests the most-needed junctions within the configured radius."""
        adv = self.A["advisor"]
        z = (PRIMARY[idea],)
        here = care_people(self.ev.eval_facility(idea, node)["weighted"], z)
        c = self.city
        cands = [x for x in candidate_sites(self.ev, idea, 60, min_sep_m=400.0) if np.hypot(*(c.xy[x] - c.xy[node])) <= adv["better_site_radius_m"] and x != node][:n]
        best, best_val = None, here
        for x in cands:
            val = care_people(self.ev.eval_facility(idea, x)["weighted"], z)
            if val > best_val:
                best, best_val = x, val
        ok = best is not None and best_val >= adv["better_site_ratio"] * max(here, 1.0) and best_val - here >= adv["noise_floor_people"]
        return {"here": here, "better": ok, "node": best, "value": best_val if ok else None,
                "lat": float(c.g.lat[best]) if ok else None, "lon": float(c.g.lon[best]) if ok else None, "where": c.describe_node(best) if ok else None}

    def gain_map(self, idea: str, node: int, snapshot: int = 0) -> dict:
        """Where the minutes are saved (per 500 m cell, population-weighted) in one chosen scenario."""
        z = PRIMARY[idea]
        st, c = self.ev.snaps[snapshot], self.city
        R = self.ev._reach(st, node)
        new = np.minimum(st.d[z], R)
        return self._cell_gain(np.minimum(st.d[z], self.ev.cap) - np.minimum(new, self.ev.cap))

    def _cell_gain(self, saved_s: np.ndarray) -> dict:
        c = self.city
        w = c.pop * (saved_s > 0)
        ppl = np.bincount(c.cell_id, weights=w, minlength=c.n_cells)
        mins = np.bincount(c.cell_id, weights=c.pop * np.maximum(saved_s, 0), minlength=c.n_cells) / np.maximum(np.bincount(c.cell_id, weights=c.pop, minlength=c.n_cells), 1e-9) / 60.0
        keep = np.flatnonzero(ppl > 20)
        return {"lat": c.cell_lat[keep], "lon": c.cell_lon[keep], "people": ppl[keep], "minutes": mins[keep]}

    # ---- 2. where a road / bridge would help ---------------------------------------------------------
    def recommend_links(self, n: int = 8, shortlist: int | None = None, progress=None) -> list[LinkRec]:
        key = ("links", n, shortlist, tuple(round(s.weight, 6) for s in self.portfolio))
        if key in self._cache:
            return self._cache[key]
        m = shortlist or int(self.A["advisor"]["candidate_counts"]["link"])
        ideas = score_links(self.ev, generate_links(self.ev, n_shortlist=m, progress=progress), progress=progress)
        scored = []
        for i in ideas:
            care = care_people(i.effect["weighted"], self.services_all)
            scored.append((care + self._everyday_as_people(i.effect["everyday"]), i))
        scored.sort(key=lambda x: -x[0])
        ideas = [i for _, i in scored[:n]]
        peers = [care_people(i.effect["weighted"], self.services_all) / max(i.cost_crore, 1e-9) for i in ideas]
        recs = [LinkRec(i, self._finish(judge_build(self.A, i.effect, self.services_all, i.cost_crore, everyday=i.effect["everyday"], peer_per_crore=peers,
                                                    what=f"A new {'bridge' if i.crosses_water else 'road'} between {i.where_u} and {i.where_v}", n_scen=self.n_scen), link=True)) for i in ideas]
        self._cache[key] = recs
        return recs

    def _everyday_as_people(self, eve: dict) -> float:
        """Rank key only: express the everyday saving on the same scale as the emergency gains (minutes saved x noise floor / threshold)."""
        us = self.A["advisor"]["usefulness"]
        return float(eve.get("hub_minutes", 0.0)) / max(us["everyday_minutes"], 1e-9) * float(self.A["advisor"]["noise_floor_people"])

    def assess_link(self, a: tuple[float, float], b: tuple[float, float]) -> Assessment:
        u, _ = self.city.snap(*a)
        v, _ = self.city.snap(*b)
        if u == v:
            raise ValueError("The two ends snap to the same junction; pick points further apart.")
        length, tau, cw = link_geometry(self.ev, u, v)
        cost = link_cost_crore(self.A, length, cw)
        eff = self.ev.eval_link(u, v, tau)
        wu, wv = self.city.describe_node(u), self.city.describe_node(v)
        verdict = self._finish(judge_build(self.A, eff, self.services_all, cost, everyday=eff["everyday"],
                                           what=f"A new {'bridge' if cw else 'road'} ({length / 1000:.1f} km) between {wu} and {wv}", n_scen=self.n_scen), link=True)
        return Assessment(f"{'Bridge' if cw else 'Road'}: {wu} to {wv}", verdict, eff, cost,
                          {"u": u, "v": v, "length_m": length, "tau_s": tau, "crosses_water": cw,
                           "a": (float(self.city.g.lat[u]), float(self.city.g.lon[u])), "b": (float(self.city.g.lat[v]), float(self.city.g.lon[v]))})

    # ---- 3/4. is what exists useful? -----------------------------------------------------------------
    def _snaps_without(self, edges: np.ndarray) -> list[int]:
        """Scenarios in which these roads are still standing (they are the ones where removing them changes anything)."""
        g = self.city.g
        both = np.unique(np.concatenate([edges, g.rev[edges][g.rev[edges] >= 0]]))
        keep = []
        for i, st in enumerate(self.ev.snaps):
            closed = np.isin(both, st.snap.closed_edges).mean() if len(st.snap.closed_edges) else 0.0
            if closed < 0.5:
                keep.append(i)
        return keep

    def assess_structure(self, i: int) -> Assessment:
        s = self.city.structures[i]
        edges = np.array(s["edges"], dtype=np.int64)
        idx = self._snaps_without(edges)
        eff = self.ev.eval_remove_edges(edges, snaps=idx)
        v = self._finish(judge_existing(self.A, eff, self.services_all, s["name"], everyday=eff["everyday"], kind_word="it"))
        cost = 0.0
        note = None
        cc = self.A["advisor"]["cost_crore"]
        if self.city.is_flyover[edges].any():
            cost = s["length_m"] / 1000.0 * cc["flyover_per_km"]
            note = (f"For scale: at the Avinashi Road elevated corridor's price (Rs {cc['flyover_per_km']} crore per km, press-reported) a flyover this long costs about Rs {cost:,.0f} crore; "
                    f"that is Rs {cost / max(care_people(eff['weighted'], self.services_all, remove=True), 1e-9):.3f} crore per person kept inside the standards.")
            v.reasons.append(note)
        return Assessment(s["name"], v, eff, cost, {"edges": edges, "length_m": s["length_m"], "excluded_scenarios": len(self.ev.snaps) - len(idx)})

    def assess_existing_facility(self, fac_id: int) -> Assessment:
        f = self.city.by_id[fac_id]
        idea = "hospital_major" if f.tier == 1 else "hospital"
        services = ("hosp", "t1") if f.tier == 1 else ("hosp",)
        idx = [i for i, st in enumerate(self.ev.snaps) if fac_id not in st.snap.closed_fac]
        eff = self.ev.eval_remove_facility(idea, fac_id, snaps=idx)
        v = self._finish(judge_existing(self.A, eff, services, f.name, kind_word="it"))
        if f.approx:
            v.caveats.append("This facility's position is approximate (not in OpenStreetMap; placed from a locality centre or a published description).")
        return Assessment(f.name, v, eff, 0.0, {"fac": f, "excluded_scenarios": len(self.ev.snaps) - len(idx)})

    def rank_structures(self, progress=None, min_length_m: float = 400.0) -> list[Assessment]:
        key = ("structs", min_length_m, tuple(round(s.weight, 6) for s in self.portfolio))
        if key in self._cache:
            return self._cache[key]
        idxs = [i for i, s in enumerate(self.city.structures) if s["length_m"] >= min_length_m]
        out = []
        for n, i in enumerate(idxs):
            out.append(self.assess_structure(i))
            if progress:
                progress((n + 1) / len(idxs))
        out.sort(key=lambda a: -a.verdict.numbers["loss"])
        self._cache[key] = out
        return out

    def rank_hospitals(self, tier: int = 1, progress=None) -> list[Assessment]:
        key = ("hosp", tier, tuple(round(s.weight, 6) for s in self.portfolio))
        if key in self._cache:
            return self._cache[key]
        fs = self.city.of_kind("hospital", None)
        fs = [f for f in fs if f.tier == tier]
        out = []
        for n, f in enumerate(fs):
            out.append(self.assess_existing_facility(f.id))
            if progress:
                progress((n + 1) / len(fs))
        out.sort(key=lambda a: -a.verdict.numbers["loss"])
        self._cache[key] = out
        return out
