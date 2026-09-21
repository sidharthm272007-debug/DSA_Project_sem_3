"""The evaluator's shortcuts must equal brute force: rebuild the graph / recompute from scratch and compare."""
import networkx as nx
import numpy as np
import pytest

from advisor.evaluator import SERVICES, Evaluator, _multi
from advisor.portfolio import build_portfolio
from store.csr_graph import FLAG_ADDED
from sim.router import Router


@pytest.fixture(scope="module")
def ev(toy, A):
    return Evaluator(toy, A, build_portfolio(toy, A, counts={"earthquake": 2, "fire": 2, "flood": 1, "structure": 1, "storm": 1}))


def nx_field(g, costs_arr, sources):
    """Independent oracle: travel time from every node to the nearest source, with networkx on the reversed graph."""
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n_nodes))
    for e in range(g.n_edges):
        c = costs_arr[e]
        if np.isfinite(c):
            u, v = int(g.edge_tails[e]), int(g.indices[e])
            if not G.has_edge(v, u) or G[v][u]["weight"] > c:
                G.add_edge(v, u, weight=float(c))
    d = np.full(g.n_nodes, np.inf)
    if sources:
        for k, val in nx.multi_source_dijkstra_path_length(G, set(sources)).items():
            d[k] = val
    return d


def test_portfolio_is_deterministic_and_weights_sum_to_one(toy, A):
    a = build_portfolio(toy, A)
    b = build_portfolio(toy, A)
    assert [s.name for s in a] == [s.name for s in b]
    assert all(np.array_equal(x.F, y.F) for x, y in zip(a, b) if x.F is not None)
    assert sum(s.weight for s in a) == pytest.approx(1.0)
    assert a[0].kind == "baseline" and a[0].F is None and a[0].weight == pytest.approx(0.25)
    assert any(len(s.closed_edges) for s in a[1:])                 # the disasters really close roads


def test_weights_follow_the_users_priorities(toy, A):
    s = build_portfolio(toy, A, weights={"baseline": 0.0, "earthquake": 1.0, "fire": 0.0, "flood": 0.0, "structure": 0.0, "storm": 0.0})
    assert sum(x.weight for x in s if x.kind == "earthquake") == pytest.approx(1.0)
    assert all(x.weight == 0 for x in s if x.kind != "earthquake")


def test_snapshot_fields_match_networkx(ev, toy):
    for st in ev.snaps:
        for z in SERVICES:
            ref = nx_field(toy.g, st.costs.arr, st.src[z])
            assert (np.isfinite(ref) == np.isfinite(st.d[z])).all()
            fin = np.isfinite(ref)
            assert np.allclose(ref[fin], st.d[z][fin])


def test_new_facility_equals_adding_a_source(ev):
    node = 3 * 21 + 17
    for idea, services in (("fire_station", ("fire",)), ("hospital_major", ("hosp", "t1"))):
        got = ev.eval_facility(idea, node)
        for st, row in zip(ev.snaps, got["per_snapshot"]):
            new = {z: (_multi(st.mat, st.src[z] + [node]) if z in services else st.d[z]) for z in SERVICES}
            want = ev._diff(st, st.d, new, services)
            for k in want:
                if k not in ("name", "kind", "weight"):
                    assert row[k] == pytest.approx(want[k])


def test_new_link_equals_rebuilding_the_graph(ev, toy):
    """The 'single new edge' identity is exact: compare every snapshot and service with a rebuilt graph."""
    g = toy.g
    u, v, tau = 2 * 21 + 3, 16 * 21 + 4, 240.0
    tails, heads = [u, v], [v, u]
    g2, old_to_new, new_ids = g.with_extra_edges(tails, heads, tt_s=[tau, tau], length_m=[1000, 1000], road_class=[3, 3], lanes_dir=[1, 1],
                                                 flags=[FLAG_ADDED, FLAG_ADDED], new_rev=[1, 0])
    R2 = Router(g2)
    for st in ev.snaps:
        F = np.ones(g.n_edges, dtype=np.float32) if st.snap.F is None else st.snap.F
        F2 = g2.tt_s * 0 + 1
        F2 = np.ones(g2.n_edges, dtype=np.float32)
        F2[old_to_new] = F
        costs2 = R2.costs(F2, st.snap.bg)
        fast = ev.link_fields(st, u, v, tau)
        for z in SERVICES:
            brute, _ = R2.reverse_multi(costs2, st.src[z])
            fin = np.isfinite(brute)
            assert (np.isfinite(fast[z]) == fin).all(), (st.snap.name, z)
            assert np.allclose(fast[z][fin], brute[fin]), (st.snap.name, z)


def test_everyday_link_equals_rebuilding_the_graph(ev, toy):
    g = toy.g
    u, v, tau_peak = 1 * 21 + 2, 17 * 21 + 3, 300.0
    fast = ev.everyday_link(u, v, tau_peak)
    g2, old_to_new, new_ids = g.with_extra_edges([u, v], [v, u], tt_s=[1, 1], length_m=[1, 1], road_class=[3, 3], lanes_dir=[1, 1],
                                                 flags=[FLAG_ADDED, FLAG_ADDED], new_rev=[1, 0])
    arr = np.full(g2.n_edges, np.inf)
    arr[old_to_new] = ev.ev_arr
    arr[new_ids] = tau_peak
    G_hubs = [nx_field(g2, arr, [h]) for h in ev.hubs]
    A_new = np.minimum(np.array(G_hubs), ev.ev_cap).mean(axis=0)
    want = ev.everyday_delta(A_new)
    for k in want:
        assert fast[k] == pytest.approx(want[k], rel=1e-4, abs=1e-6)
    assert fast["hub_minutes"] > 0                                       # a long shortcut across the grid must help someone


def test_removal_equals_networkx(ev, toy):
    edges = np.array(toy.structures[0]["edges"])
    got = ev.eval_remove_edges(edges, snaps=[0, 1])
    g = toy.g
    both = np.unique(np.concatenate([edges, g.rev[edges]]))
    for i, row in zip([0, 1], got["per_snapshot"]):
        st = ev.snaps[i]
        arr = st.costs.arr.copy()
        arr[both] = np.inf
        new = {z: nx_field(g, arr, st.src[z]) for z in SERVICES}
        want = ev._diff(st, st.d, new, SERVICES)
        for k in want:
            if k not in ("name", "kind", "weight"):
                assert row[k] == pytest.approx(want[k], abs=1e-6)


def test_removal_is_never_beneficial_and_removing_a_hospital_hurts(ev, toy):
    rem = ev.eval_remove_edges(np.array(toy.structures[0]["edges"]), snaps=[0])
    assert rem["normal"]["hosp_within"] <= 1e-9 and rem["normal"]["hosp_mean"] >= -1e-9
    hosp = ev.eval_remove_facility("hospital_major", 0, snaps=[0])          # the central major hospital
    assert hosp["normal"]["t1_mean"] > 0


def test_new_link_hit_by_disaster_gives_no_benefit_there(ev, toy):
    quake_i = next(i for i, s in enumerate(ev.snaps) if s.snap.kind == "earthquake" and len(s.snap.closed_edges))
    st = ev.snaps[quake_i]
    e = int(st.snap.closed_edges[0])
    u, v = int(toy.g.edge_tails[e]), int(toy.g.indices[e])
    assert ev.link_is_hit(st, u, v)
    res = ev.eval_link(u, v, 60.0)
    row = res["per_snapshot"][quake_i]
    assert row.get("hit") == 1.0 and all(row[k] == 0 for k in row if k.endswith(("_within", "_mean")))


def test_aggregation_math(ev):
    res = ev.eval_facility("hospital", 5 * 21 + 5)
    rows = res["per_snapshot"]
    assert res["weighted"]["hosp_within"] == pytest.approx(sum(r["weight"] * r["hosp_within"] for r in rows) / sum(r["weight"] for r in rows))
    dis = [r for r in rows if r["kind"] != "baseline"]
    assert res["disaster_avg"]["hosp_within"] == pytest.approx(sum(r["weight"] * r["hosp_within"] for r in dis) / sum(r["weight"] for r in dis))
    assert res["normal"]["hosp_within"] == rows[0]["hosp_within"]
