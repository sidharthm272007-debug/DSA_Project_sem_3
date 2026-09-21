"""The compiled bulk search and the heap point-search must agree, including with parallel roads and closed roads."""
import numpy as np
import pytest

from sim.router import Router


def closed_factor(g, share, seed):
    F = np.ones(g.n_edges, dtype=np.float32)
    F[np.random.default_rng(seed).random(g.n_edges) < share] = 0
    return F


@pytest.mark.parametrize("share,seed", [(0.0, 0), (0.1, 1), (0.35, 2)])
def test_bulk_matches_heap_search_on_toy(toy, share, seed):
    g = toy.g
    R = Router(g)
    costs = R.costs(closed_factor(g, share, seed), 0.85)
    srcs = [f.node for f in toy.hospitals(2)]
    d, lab = R.reverse_multi(costs, srcs)
    ref, ref_lab, _, _ = R._run(R.rindptr, R.rindices, costs.rev, None, [(s, 0.0) for s in srcs], None, float("inf"), True)
    ref = np.array(ref)
    assert (np.isfinite(d) == np.isfinite(ref)).all()
    fin = np.isfinite(ref)
    assert np.allclose(d[fin], ref[fin])
    assert set(lab[fin].tolist()) <= set(srcs)            # every label is a real source
    assert (lab[~fin] == -1).all()


def test_parallel_roads_use_the_cheaper_one():
    from store.csr_graph import CSRGraph
    # two roads 0->1 (100 s and 40 s) and one 1->2 (10 s): the compiled search must not add the parallel costs together
    g = CSRGraph.from_edges(3, [0, 0, 1], [1, 1, 2], [100.0, 40.0, 10.0])
    R = Router(g)
    d, _ = R.reverse_multi(R.costs(None), {2: 0.0})
    assert d[0] == pytest.approx(50.0)
    dist, path_to = R.reach_from(R.costs(None), 0)
    assert dist[2] == pytest.approx(50.0)
    assert [float(g.tt_s[e]) for e in path_to(2)] == [40.0, 10.0]      # the cheaper parallel arc is on the path


def test_reach_from_paths_are_valid_and_cost_matches(toy):
    g = toy.g
    R = Router(g)
    costs = R.costs(closed_factor(g, 0.1, 5), 0.85)
    dist, path_to = R.reach_from(costs, 0)
    rng = np.random.default_rng(0)
    for v in rng.integers(0, g.n_nodes, 25):
        p = path_to(int(v))
        if not np.isfinite(dist[v]):
            assert p is None
            continue
        assert g.edge_tails[p[0]] == 0 if p else v == 0
        assert all(g.indices[a] == g.edge_tails[b] for a, b in zip(p[:-1], p[1:]))     # contiguous
        assert (g.indices[p[-1]] == v) if p else True
        assert sum(costs.arr[e] for e in p) == pytest.approx(dist[v])
        assert all(np.isfinite(costs.arr[e]) for e in p)                                # never through a closed road


def test_point_search_agrees_with_bulk(toy):
    g = toy.g
    R = Router(g)
    costs = R.costs(closed_factor(g, 0.08, 3), 0.85)
    dist, _ = R.reach_from(costs, 5)
    for v in (17, 100, 250, 400):
        path, t = R.path(costs, 5, v)
        if np.isfinite(dist[v]):
            assert t == pytest.approx(dist[v])
        else:
            assert path == [] and t == float("inf")


def test_components_split_when_roads_close(toy):
    g = toy.g
    R = Router(g)
    assert R.components(np.ones(g.n_edges, dtype=bool)).max() == 0
    usable = np.ones(g.n_edges, dtype=bool)
    col = 10                                                         # cut the whole grid along one column of x-edges
    for e in range(g.n_edges):
        a, b = int(g.edge_tails[e]), int(g.indices[e])
        if {a % 21, b % 21} == {col, col + 1}:
            usable[e] = False
    lab = R.components(usable)
    assert lab.max() == 1 and lab[0] != lab[20]


def test_real_city_bulk_matches_heap_search():
    from sim.city import load_city
    city = load_city()
    R = Router(city.g)
    costs = R.costs(closed_factor(city.g, 0.06, 9), 0.85)
    srcs = [f.node for f in city.hospitals(1)]
    d, _ = R.reverse_multi(costs, srcs)
    ref, _, _, _ = R._run(R.rindptr, R.rindices, costs.rev, None, [(s, 0.0) for s in srcs], None, float("inf"), True)
    ref = np.array(ref)
    fin = np.isfinite(ref)
    assert (np.isfinite(d) == fin).all() and np.allclose(d[fin], ref[fin])
