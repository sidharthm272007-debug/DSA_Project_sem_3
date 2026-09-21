"""Independent end-to-end checks on the REAL Coimbatore data (not the toy city).

Run:  python -m scripts.verify
Prints PASS / WARN / FAIL per check. Nothing here is mocked: each check recomputes a headline number a different way and compares.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402  (test oracle only)
import numpy as np  # noqa: E402

from advisor.evaluator import SERVICES, Evaluator  # noqa: E402
from advisor.portfolio import build_portfolio  # noqa: E402
from advisor.service import Advisor  # noqa: E402
from advisor.siting import candidate_sites  # noqa: E402
from assumptions import load_assumptions  # noqa: E402
from sim.city import load_city  # noqa: E402
from sim.engine import Simulation  # noqa: E402
from ui.scenarios import make_scenario, presets  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main() -> int:
    A = load_assumptions()
    city = load_city()
    g = city.g
    t0 = time.time()

    # ---- data is real and plausible -------------------------------------------------------------
    check("graph is city-scale", g.n_nodes > 50_000 and g.n_edges > 120_000, f"{g.n_nodes:,} nodes, {g.n_edges:,} arcs")
    check("coordinates inside the Coimbatore area", 10.7 < g.lat.min() and g.lat.max() < 11.3 and 76.6 < g.lon.min() and g.lon.max() < 77.3)
    check("population equals the stated city estimate (giant component)", abs(city.total_pop - A["population"]["total"]) / A["population"]["total"] < 0.02, f"{city.total_pop:,.0f}")
    check("named water bodies exist and are long", len(city.water) >= 5 and max(len(v) for v in city.water.values()) > 50, f"{len(city.water)} bodies")
    check("named flyovers exist", len(city.structures) >= 20 and any("Naidu" in s["name"] for s in city.structures), f"{len(city.structures)} structures")
    n_fire = len(city.of_kind("fire_station"))
    check("fire stations mapped >= advisor threshold", n_fire >= A["advisor"]["min_fire_stations"], f"{n_fire} mapped vs {A['advisor']['min_fire_stations']} needed; fire results are excluded from headline advisor numbers", warn=True)
    check("no facility is exactly at a hand-typed spot without an 'approx' flag",
          all(f.approx for f in city.facilities if f.source == "manual"), f"{sum(f.approx for f in city.facilities)} flagged approximate")

    # ---- simulator: reproducible, and physically sensible -------------------------------------------
    p = presets(city)["M6.0 earthquake under the city centre"]
    mk = lambda seed: make_scenario(city, "q", p["specs"], p["ctx"], 6, seed)
    a, b = Simulation(city, mk(5), A).run(), Simulation(city, mk(5), A).run()
    c = Simulation(city, mk(6), A).run()
    same = [m["gt30_any"] for m in a.metrics] == [m["gt30_any"] for m in b.metrics] and len(a.incidents) == len(b.incidents)
    check("same seed reproduces the same run", same)
    check("different seed gives a different run", [i.node for i in a.incidents] != [i.node for i in c.incidents])
    quiet = Simulation(city, make_scenario(city, "none", [], p["ctx"], 3, 1), A).run()
    check("no hazard -> no lost access", quiet.summary["never_hit"] and max(m["cut_any"] for m in quiet.metrics) < A["advisor"]["noise_floor_people"])
    check("quake closes roads and kills hospitals", max(m["blocked_km"] for m in a.metrics) > 1 and max(m["hospitals_closed"] for m in a.metrics) > 0)

    # ---- advisor: fast formulas equal brute force on the real network ------------------------------------
    adv = Advisor(city, A)
    ev = adv.ev
    st = ev.snaps[3]
    g_nx = nx.DiGraph()
    g_nx.add_nodes_from(range(g.n_nodes))
    arr = st.costs.arr
    for e in range(g.n_edges):
        if np.isfinite(arr[e]):
            u, v = int(g.edge_tails[e]), int(g.indices[e])
            if not g_nx.has_edge(v, u) or g_nx[v][u]["weight"] > arr[e]:
                g_nx.add_edge(v, u, weight=float(arr[e]))
    ref = np.full(g.n_nodes, np.inf)
    for k, val in nx.multi_source_dijkstra_path_length(g_nx, set(st.src["hosp"])).items():
        ref[k] = val
    fin = np.isfinite(ref)
    check("hospital-access field equals networkx on the damaged real network", (fin == np.isfinite(st.d["hosp"])).all() and np.allclose(ref[fin], st.d["hosp"][fin]), st.snap.name)

    cands = candidate_sites(ev, "hospital", 25)
    scores = {cnode: -ev.eval_facility("hospital", cnode)["weighted"]["hosp_mean"] for cnode in cands}
    from advisor.siting import greedy_sites
    pick = greedy_sites(ev, "hospital", cands, 1)[0]
    check("first greedy hospital site = brute-force best of the same candidates", abs(scores[pick.node] - max(scores.values())) < 1e-9, adv.city.describe_node(pick.node))

    links = adv.recommend_links(n=1, shortlist=6)
    if links:
        idea = links[0].idea
        # rebuild the whole graph with the new road and recompute the hospital field on one scenario
        s0 = ev.snaps[0]
        Ru, Rv = ev._reach(s0, idea.u), ev._reach(s0, idea.v)
        formula = np.minimum(s0.d["hosp"], np.minimum(Ru + idea.tau_s / s0.snap.bg + s0.d["hosp"][idea.v], Rv + idea.tau_s / s0.snap.bg + s0.d["hosp"][idea.u]))
        fin_ok = np.isfinite(formula)
        arr0 = s0.costs.arr
        G2 = nx.DiGraph()
        G2.add_nodes_from(range(g.n_nodes))
        for e in range(g.n_edges):
            if np.isfinite(arr0[e]):
                u, v = int(g.edge_tails[e]), int(g.indices[e])
                if not G2.has_edge(v, u) or G2[v][u]["weight"] > arr0[e]:
                    G2.add_edge(v, u, weight=float(arr0[e]))
        tau = idea.tau_s / s0.snap.bg
        for x, y in ((idea.u, idea.v), (idea.v, idea.u)):
            if not G2.has_edge(x, y) or G2[x][y]["weight"] > tau:
                G2.add_edge(x, y, weight=tau)
        truth = np.full(g.n_nodes, np.inf)
        for k, val in nx.multi_source_dijkstra_path_length(G2, set(s0.src["hosp"])).items():
            truth[k] = val
        check("new-road shortcut formula equals rebuilding the real graph with the road", (np.isfinite(truth) == fin_ok).all() and np.allclose(truth[fin_ok], formula[fin_ok]), f"{idea.where_u} <-> {idea.where_v}")

    # ---- route advice and lifelines on the real damaged network ---------------------------------------------------
    from routing.lifelines import hospital_dependence, open_hospital_nodes
    from routing.recommend import advise
    from sim.state import hazard_state
    from ui.scenarios import make_hazard
    rp = presets(city)["M6.0 earthquake under the city centre"]
    qs = hazard_state(city, A, [make_hazard(city, sp) for sp in rp["specs"]], 0.85, 90.0)
    arr_q = adv.router.costs(qs.F, qs.bg).arr
    G_q = nx.DiGraph()
    G_q.add_nodes_from(range(g.n_nodes))
    for e in range(g.n_edges):
        if np.isfinite(arr_q[e]):
            u, v = int(g.edge_tails[e]), int(g.indices[e])
            if not G_q.has_edge(u, v) or G_q[u][v]["weight"] > arr_q[e]:
                G_q.add_edge(u, v, weight=float(arr_q[e]))
    rng = np.random.default_rng(4)
    giant = np.flatnonzero(city.giant)
    exact, avoids_closed, cautious_ok, n_done = True, True, True, 0
    for _ in range(6):
        a_, b_ = (int(x) for x in rng.choice(giant, 2, replace=False))
        try:
            ref_t = nx.shortest_path_length(G_q, a_, b_, weight="weight")
        except nx.NetworkXNoPath:
            continue
        fast = advise(city, adv.router, A, qs, a_, b_, "fastest")
        care = advise(city, adv.router, A, qs, a_, b_, "cautious")
        if not fast.options:
            exact = False
            continue
        n_done += 1
        exact &= abs(min(o.time_s for o in fast.options) - ref_t) < 1e-6 * max(ref_t, 1)
        avoids_closed &= all((qs.F[o.edges] > 0).all() for o in fast.options + care.options)
        cautious_ok &= care.recommended.risk_km <= fast.recommended.risk_km + 1e-9
    check("fastest route equals networkx shortest path on the earthquake-damaged real network", exact and n_done >= 3, f"{n_done} random trips")
    check("no recommended or alternative route uses a closed road", avoids_closed)
    check("a cautious traveller is never sent on a riskier route than a hurried one", cautious_ok)
    dep = hospital_dependence(city, adv.router, qs.F, qs.bg, open_hospital_nodes(city, qs))
    hn = open_hospital_nodes(city, qs)
    inflow = sum(dep.flow[g.indices == h].sum() for h in hn)
    check("lifelines conserve people: everyone who can reach a hospital flows into one", abs(dep.reach + dep.cut - city.pop.sum()) < 1e-3 and abs(inflow - (dep.reach - city.pop[hn].sum())) < 1e-3,
          f"reach {dep.reach:,.0f}, cut {dep.cut:,.0f}, into hospitals {inflow:,.0f}")

    # ---- results are stable, not an accident of the sampled scenarios ---------------------------------------
    adv2 = Advisor(city, A, seed=A["advisor"]["portfolio"]["seed"] + 11, router=adv.router)
    r1 = [r.pick.node for r in adv.recommend_sites("hospital", 3)]
    r2 = [r.pick.node for r in adv2.recommend_sites("hospital", 3)]
    xy = city.xy
    near = sum(any(np.hypot(*(xy[a_] - xy[b_])) < 3000 for b_ in r2) for a_ in r1)
    check("top-3 hospital sites are stable under a different scenario sample (within 3 km)", near >= 2, f"{near} of 3 recur; other seed: {[adv2.city.describe_node(n) for n in r2]}", warn=True)

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\n{len(RESULTS) - len(fails) - len(warns)} PASS, {len(warns)} WARN, {len(fails)} FAIL  ({time.time() - t0:.0f} s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
