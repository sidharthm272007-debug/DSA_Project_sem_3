"""Route advice and lifelines, checked against networkx and against invariants that must hold on any network."""
import networkx as nx
import numpy as np
import pytest

from routing.lifelines import hospital_dependence, lifelines, open_hospital_nodes
from routing.recommend import advise, fmt_min, hospital_options
from routing.risk import edge_risk
from sim.hazards import Earthquake
from sim.router import Router
from sim.state import WorldState, hazard_state, normal_state

BG = 0.85


@pytest.fixture(scope="module")
def router(toy):
    return Router(toy.g)


@pytest.fixture(scope="module")
def quake(toy, A):
    g = toy.g
    mid = 10 * 21 + 10
    return hazard_state(toy, A, [Earthquake(float(g.lat[mid]), float(g.lon[mid]), magnitude=6.8)], BG, 30.0, seed=2)


def nx_graph(g, cost):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n_nodes))
    for e in range(g.n_edges):
        if np.isfinite(cost[e]):
            u, v = int(g.edge_tails[e]), int(g.indices[e])
            if not G.has_edge(u, v) or G[u][v]["weight"] > cost[e]:
                G.add_edge(u, v, weight=float(cost[e]))
    return G


def test_state_has_real_damage(quake):
    assert len(quake.closed_edges) > 0 and quake.expo.max() > 0.5


def test_fastest_route_equals_networkx_and_every_option_is_genuinely_pareto_optimal(toy, A, router, quake):
    g = toy.g
    src, dst = 1 * 21 + 1, 19 * 21 + 19
    adv = advise(toy, router, A, quake, src, dst, "balanced")
    assert adv.options and not adv.unreachable
    arr = router.costs(quake.F, quake.bg).arr
    G = nx_graph(g, arr)
    best = nx.shortest_path_length(G, src, dst, weight="weight")
    assert min(o.time_s for o in adv.options) == pytest.approx(best, rel=1e-9)
    # every option returned must be genuinely non-dominated: no OTHER option beats it on both time and risk at once
    # (brute-force O(n^2) check against the independent definition in routing/pareto.py)
    for o in adv.options:
        for p in adv.options:
            if p is o:
                continue
            assert not (p.time_s <= o.time_s and p.risk_km <= o.risk_km and (p.time_s < o.time_s or p.risk_km < o.risk_km)), (o, p)
    # and the recommended one really does minimise the stated objective among exactly those options
    assert adv.recommended.objective == pytest.approx(min(o.objective for o in adv.options))


def test_no_route_ever_uses_a_closed_road(toy, A, router, quake):
    for src, dst in ((0, 440), (21 * 3 + 3, 21 * 17 + 15), (10 * 21 + 2, 10 * 21 + 18)):
        adv = advise(toy, router, A, quake, src, dst, "cautious")
        for o in adv.options:
            assert (quake.F[o.edges] > 0).all()


def test_recommendation_minimises_the_stated_objective_and_more_caution_never_raises_risk(toy, A, router, quake):
    src, dst = 1 * 21 + 1, 19 * 21 + 19
    got = {}
    for tol in ("fastest", "balanced", "cautious"):
        v = A["routing"]["tolerance_min_per_risk_km"][tol]
        adv = advise(toy, router, A, quake, src, dst, tol)
        for o in adv.options:                                            # recompute the objective independently from the route's own numbers
            assert o.objective == pytest.approx(o.time_s / 60.0 + v * o.risk_km)
        assert adv.recommended.objective == pytest.approx(min(o.time_s / 60.0 + v * o.risk_km for o in adv.options))
        assert "Recommended" in adv.recommended.tags
        got[tol] = adv.recommended
    assert got["cautious"].risk_km <= got["balanced"].risk_km + 1e-9 <= got["fastest"].risk_km + 2e-9
    assert got["cautious"].time_s >= got["balanced"].time_s - 1e-6 >= got["fastest"].time_s - 2e-6


def test_caution_really_changes_the_recommendation_somewhere(toy, A, router, quake):
    """The preference must matter: for some trips a cautious traveller is sent a different way than a hurried one."""
    rng = np.random.default_rng(1)
    differs = 0
    for _ in range(25):
        a, b = (int(x) for x in rng.choice(toy.g.n_nodes, 2, replace=False))
        hurry = advise(toy, router, A, quake, a, b, "fastest")
        care = advise(toy, router, A, quake, a, b, "cautious")
        if hurry.recommended and care.recommended and hurry.recommended.edges != care.recommended.edges:
            differs += 1
            assert care.recommended.risk_km < hurry.recommended.risk_km and care.recommended.time_s > hurry.recommended.time_s
    assert differs >= 1


def test_roads_the_traveller_knows_are_blocked_are_avoided(toy, A, router):
    st = normal_state(toy, BG)
    src, dst = 2 * 21 + 2, 18 * 21 + 18
    first = advise(toy, router, A, st, src, dst).recommended
    mid = first.edges[len(first.edges) // 2]
    again = advise(toy, router, A, st, src, dst, blocked_edges=np.array([mid])).recommended
    assert mid not in again.edges and toy.g.rev[mid] not in again.edges
    assert again.time_s >= first.time_s - 1e-9


def test_avoided_roads_count_as_maximum_risk(toy, A):
    st = normal_state(toy, BG)
    r = edge_risk(toy, A, st, avoid_edges=np.array([5, 9]))
    assert r[5] == 1.0 and r[9] == 1.0 and r[100] == 0.0


def test_usual_route_names_the_road_that_is_now_cut(toy, A, router):
    g = toy.g
    src, dst = 10 * 21 + 0, 10 * 21 + 20
    usual = advise(toy, router, A, normal_state(toy, BG), src, dst).usual
    e = usual.edges[len(usual.edges) // 2]
    F = np.ones(g.n_edges, dtype=np.float32)
    F[e] = 0
    if g.rev[e] >= 0:
        F[g.rev[e]] = 0
    adv = advise(toy, router, A, WorldState("cut", F, np.zeros(g.n_nodes), BG), src, dst)
    assert adv.usual_cut_at and adv.recommended is not None and e not in adv.recommended.edges


def test_cut_off_start_gets_a_walk_suggestion(toy, A, router):
    g = toy.g
    node = 10 * 21 + 10
    F = np.ones(g.n_edges, dtype=np.float32)
    F[(g.edge_tails == node) | (g.indices == node)] = 0                 # every road at this junction is closed
    st = WorldState("island", F, np.zeros(g.n_nodes), BG)
    adv = advise(toy, router, A, st, node, 19 * 21 + 19)
    assert adv.unreachable and adv.walk_first
    assert 0 < adv.walk_first["walk_m"] <= A["routing"]["max_walk_m"]
    assert adv.walk_first["advice"].recommended is not None


def test_same_start_and_end_is_reported_not_crashed(toy, A, router):
    adv = advise(toy, router, A, normal_state(toy, BG), 5, 5)
    assert adv.unreachable and adv.recommended is None


def test_hospital_options_skip_closed_hospitals_and_sort_by_time(toy, A, router):
    st = normal_state(toy, BG)
    hs = toy.hospitals(2)
    st.closed_fac = {hs[0].id}
    r = hospital_options(toy, router, A, st, 10 * 21 + 10, top=5)
    assert all(x["fac"].id != hs[0].id for x in r["open"])
    times = [x["time_s"] for x in r["open"]]
    assert times == sorted(times)
    assert any(x["status"] == "out of service" for x in r["nearest_by_air"]) or hs[0].id not in [x["fac"].id for x in r["nearest_by_air"]]


def test_fmt_min():
    assert fmt_min(600) == "10 min" and fmt_min(90) == "1.5 min" and fmt_min(float("inf")) == "no route"


# ---------------------------------------------------------------------------------------------- lifelines
def test_lifelines_conserve_people_and_match_a_networkx_forest(toy, A, router):
    g = toy.g
    hosp = [f.node for f in toy.hospitals(2)][:1]
    rng = np.random.default_rng(3)
    # unique shortest paths: multiply each road's speed factor by a tiny distinct number
    F = (1.0 - 1e-3 * rng.random(g.n_edges)).astype(np.float32)
    dep = hospital_dependence(toy, router, F, BG, hosp)
    assert dep.reach + dep.cut == pytest.approx(toy.pop.sum())
    costs = router.costs(F, BG)
    G = nx_graph(g, costs.arr).reverse(copy=True)
    dist, paths = nx.single_source_dijkstra(G, hosp[0], weight="weight")
    load = np.zeros(g.n_edges)
    for x, path in paths.items():
        if x == hosp[0]:
            continue
        p = path[::-1]                                              # x -> ... -> hospital in the original direction
        for a, b in zip(p[:-1], p[1:]):
            lo, hi = g.indptr[a], g.indptr[a + 1]
            cand = np.flatnonzero(g.indices[lo:hi] == b) + lo
            load[cand[np.argmin(costs.arr[cand])]] += toy.pop[x]
    assert np.allclose(load, dep.flow, atol=1e-6)
    # everyone who can reach the hospital ends up flowing into it
    into = dep.flow[g.indices == hosp[0]].sum()
    assert into == pytest.approx(dep.reach - toy.pop[hosp[0]])


def test_closing_a_road_removes_its_load_and_creates_a_surge_elsewhere(toy, A, router):
    g = toy.g
    st = normal_state(toy, BG)
    base = lifelines(toy, router, st)
    e = int(np.argmax(base["now"].flow))
    F = np.ones(g.n_edges, dtype=np.float32)
    F[e] = 0
    if g.rev[e] >= 0:
        F[g.rev[e]] = 0
    hit = lifelines(toy, router, WorldState("cut", F, np.zeros(g.n_nodes), BG))
    assert hit["now"].flow[e] == 0 and hit["change"][e] < 0
    assert (hit["change"] > 0).any()                                # the load moved somewhere


def test_closed_hospitals_take_no_load_and_normal_day_has_no_change(toy, A, router):
    st = normal_state(toy, BG)
    same = lifelines(toy, router, st)
    assert np.allclose(same["change"], 0)
    hs = toy.hospitals(2)
    st2 = normal_state(toy, BG)
    st2.closed_fac = {f.id for f in hs[1:]}
    assert open_hospital_nodes(toy, st2) == [hs[0].node]
