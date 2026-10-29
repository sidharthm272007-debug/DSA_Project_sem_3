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

from assumptions import load_assumptions  # noqa: E402
from sim.city import load_city  # noqa: E402
from sim.engine import Simulation  # noqa: E402
from sim.router import Router  # noqa: E402
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
    router = Router(g)
    t0 = time.time()

    # ---- data is real and plausible -------------------------------------------------------------
    check("graph is city-scale", g.n_nodes > 50_000 and g.n_edges > 120_000, f"{g.n_nodes:,} nodes, {g.n_edges:,} arcs")
    check("coordinates inside the Coimbatore area", 10.7 < g.lat.min() and g.lat.max() < 11.3 and 76.6 < g.lon.min() and g.lon.max() < 77.3)
    check("population equals the stated city estimate (giant component)", abs(city.total_pop - A["population"]["total"]) / A["population"]["total"] < 0.02, f"{city.total_pop:,.0f}")
    check("named water bodies exist and are long", len(city.water) >= 5 and max(len(v) for v in city.water.values()) > 50, f"{len(city.water)} bodies")
    check("named flyovers exist", len(city.structures) >= 20 and any("Naidu" in s["name"] for s in city.structures), f"{len(city.structures)} structures")
    n_fire = len(city.of_kind("fire_station"))
    check("fire stations mapped >= standard threshold", n_fire >= A["standards"]["min_fire_stations"],
          f"{n_fire} mapped vs {A['standards']['min_fire_stations']} needed; fire-response numbers run worse than the real city until more are mapped", warn=True)
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
    check("no hazard -> no lost access", quiet.summary["never_hit"] and max(m["cut_any"] for m in quiet.metrics) < A["standards"]["noise_floor_people"])
    check("quake closes roads and kills hospitals", max(m["blocked_km"] for m in a.metrics) > 1 and max(m["hospitals_closed"] for m in a.metrics) > 0)

    # ---- route advice and lifelines on the real damaged network ---------------------------------------------------
    from routing.lifelines import hospital_dependence, open_hospital_nodes
    from routing.recommend import advise
    from sim.state import hazard_state
    from ui.scenarios import make_hazard
    rp = presets(city)["M6.0 earthquake under the city centre"]
    qs = hazard_state(city, A, [make_hazard(city, sp) for sp in rp["specs"]], 0.85, 90.0)
    arr_q = router.costs(qs.F, qs.bg).arr

    # independent oracle: the damaged network's own shortest-path field, built with a different library entirely
    G_q = nx.DiGraph()
    G_q.add_nodes_from(range(g.n_nodes))
    for e in range(g.n_edges):
        if np.isfinite(arr_q[e]):
            u, v = int(g.edge_tails[e]), int(g.indices[e])
            if not G_q.has_edge(u, v) or G_q[u][v]["weight"] > arr_q[e]:
                G_q.add_edge(u, v, weight=float(arr_q[e]))
    rev = nx.DiGraph()
    rev.add_nodes_from(range(g.n_nodes))
    for e in range(g.n_edges):
        if np.isfinite(arr_q[e]):
            u, v = int(g.edge_tails[e]), int(g.indices[e])
            if not rev.has_edge(v, u) or rev[v][u]["weight"] > arr_q[e]:
                rev.add_edge(v, u, weight=float(arr_q[e]))
    hosp_nodes = {f.node for f in city.hospitals(2) if f.id not in qs.closed_fac}
    ref = np.full(g.n_nodes, np.inf)
    for k, val in nx.multi_source_dijkstra_path_length(rev, hosp_nodes).items():
        ref[k] = val
    from scipy.sparse.csgraph import dijkstra
    costs_q = router.costs(qs.F, qs.bg)
    got = dijkstra(router._matrix(costs_q.rev_arr, router._rplan), directed=True, indices=np.array(list(hosp_nodes), dtype=np.int64), min_only=True)
    fin = np.isfinite(ref)
    check("hospital-access field equals networkx on the earthquake-damaged real network", (fin == np.isfinite(got)).all() and np.allclose(ref[fin], got[fin]), f"{len(hosp_nodes)} open hospitals")

    rng = np.random.default_rng(4)
    giant = np.flatnonzero(city.giant)
    exact, avoids_closed, cautious_ok, n_done = True, True, True, 0
    for _ in range(6):
        a_, b_ = (int(x) for x in rng.choice(giant, 2, replace=False))
        try:
            ref_t = nx.shortest_path_length(G_q, a_, b_, weight="weight")
        except nx.NetworkXNoPath:
            continue
        fast = advise(city, router, A, qs, a_, b_, "fastest")
        care = advise(city, router, A, qs, a_, b_, "cautious")
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
    dep = hospital_dependence(city, router, qs.F, qs.bg, open_hospital_nodes(city, qs))
    hn = open_hospital_nodes(city, qs)
    inflow = sum(dep.flow[g.indices == h].sum() for h in hn)
    check("lifelines conserve people: everyone who can reach a hospital flows into one", abs(dep.reach + dep.cut - city.pop.sum()) < 1e-3 and abs(inflow - (dep.reach - city.pop[hn].sum())) < 1e-3,
          f"reach {dep.reach:,.0f}, cut {dep.cut:,.0f}, into hospitals {inflow:,.0f}")

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\n{len(RESULTS) - len(fails) - len(warns)} PASS, {len(warns)} WARN, {len(fails)} FAIL  ({time.time() - t0:.0f} s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
