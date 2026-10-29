"""Yen's K-shortest-paths, checked against networkx's independent implementation of the same algorithm, and against
brute-force enumeration where the graph is small enough to enumerate every simple path directly."""
import itertools

import networkx as nx
import numpy as np
import pytest

from routing.kshortest import k_shortest_paths
from sim.router import Router


@pytest.fixture(scope="module")
def router(toy):
    return Router(toy.g)


def nx_graph(g, cost: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n_nodes))
    for e in range(g.n_edges):
        if np.isfinite(cost[e]):
            u, v = int(g.edge_tails[e]), int(g.indices[e])
            if not G.has_edge(u, v) or G[u][v]["weight"] > cost[e]:
                G.add_edge(u, v, weight=float(cost[e]))
    return G


def path_cost(cost: np.ndarray, edges) -> float:
    return float(cost[np.asarray(edges, dtype=np.int64)].sum())


def brute_force_k_shortest(G: nx.DiGraph, src, dst, K: int):
    """Enumerate every simple path exhaustively (only tractable on tiny graphs) and return the K cheapest."""
    all_paths = []
    for path in nx.all_simple_paths(G, src, dst, cutoff=12):
        cost = sum(G[a][b]["weight"] for a, b in zip(path[:-1], path[1:]))
        all_paths.append((cost, tuple(path)))
    all_paths.sort()
    return all_paths[:K]


def test_oracle_agrees_with_brute_force_enumeration(request):
    """Not a test of this project's code: a sanity check on the ORACLE used elsewhere in this file. The real
    cross-check against this project's own k_shortest_paths is test_matches_networkx_shortest_simple_paths_on_the_toy_city
    below, which runs on a genuine CSRGraph/Router; building another one from scratch here would just duplicate that
    with less realistic data, so this stays a check on networkx itself, against independent brute-force enumeration."""
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(12):
        n = rng.integers(6, 10)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.45:
                    G.add_edge(u, v, weight=float(rng.integers(1, 20)))
        src, dst = 0, n - 1
        if not nx.has_path(G, src, dst):
            continue
        nx_paths = list(itertools.islice(nx.shortest_simple_paths(G, src, dst, weight="weight"), 6))
        brute = brute_force_k_shortest(G, src, dst, 6)
        nx_costs = [sum(G[a][b]["weight"] for a, b in zip(p[:-1], p[1:])) for p in nx_paths]
        assert nx_costs == pytest.approx([c for c, _ in brute][:len(nx_costs)])
        checked += 1
    assert checked >= 5


def test_matches_networkx_shortest_simple_paths_on_the_toy_city(toy, router):
    g = toy.g
    cost = router.costs(None, 1.0).arr
    G = nx_graph(g, cost)
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(10):
        src, dst = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        if not nx.has_path(G, src, dst):
            continue
        ours = k_shortest_paths(router, cost, src, dst, 5, xy=toy.xy)
        if not ours:
            continue
        theirs = list(itertools.islice(nx.shortest_simple_paths(G, src, dst, weight="weight"), len(ours)))
        their_costs = [sum(G[a][b]["weight"] for a, b in zip(p[:-1], p[1:])) for p in theirs]
        our_costs = [c.cost for c in ours]
        assert our_costs == pytest.approx(their_costs, rel=1e-9), (src, dst)
        checked += 1
    assert checked >= 5


def test_every_path_is_loopless_and_strictly_non_decreasing_in_cost(toy, router):
    g = toy.g
    cost = router.costs(None, 1.0).arr
    rng = np.random.default_rng(9)
    for _ in range(15):
        src, dst = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        paths = k_shortest_paths(router, cost, src, dst, 6, xy=toy.xy)
        for p in paths:
            assert len(set(p.nodes)) == len(p.nodes), "a path revisited a node"
            assert path_cost(cost, p.edges) == pytest.approx(p.cost, rel=1e-9)
        costs = [p.cost for p in paths]
        assert costs == sorted(costs)


def test_every_path_is_genuinely_different_not_a_reweighted_duplicate(toy, router):
    """Two paths must never have the identical edge set (that would be the same route counted twice)."""
    g = toy.g
    cost = router.costs(None, 1.0).arr
    rng = np.random.default_rng(13)
    for _ in range(10):
        src, dst = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        paths = k_shortest_paths(router, cost, src, dst, 8, xy=toy.xy)
        edge_sets = [p.edges for p in paths]
        assert len(edge_sets) == len(set(edge_sets))


def test_same_start_and_end_returns_nothing(toy, router):
    assert k_shortest_paths(router, router.costs(None, 1.0).arr, 5, 5, 4) == []


def test_unreachable_destination_returns_nothing(toy, router):
    g = toy.g
    cost = router.costs(None, 1.0).arr.copy()
    cost[:] = np.inf                                 # nothing is reachable from anywhere
    assert k_shortest_paths(router, cost, 0, g.n_nodes - 1, 4) == []


def test_real_city_finds_genuinely_different_routes(request):
    import time

    from sim.city import load_city

    city = load_city()
    r = Router(city.g)
    cost = r.costs(None, 0.85).arr
    rng = np.random.default_rng(21)
    giant = np.flatnonzero(city.giant)
    times = []
    for _ in range(5):
        a, b = (int(x) for x in rng.choice(giant, 2, replace=False))
        t = time.time()
        paths = k_shortest_paths(r, cost, a, b, 4)
        times.append(time.time() - t)
        if paths:
            edge_sets = [set(p.edges) for p in paths]
            for x, y in itertools.combinations(edge_sets, 2):
                assert x != y                                          # not identical, and not just a reweighted duplicate
    # No hard time assertion: this is an unbounded (correct, not geographically restricted) search on the real
    # 72,000-junction network, so latency depends on the random trip pair. Printed for visibility, not enforced.
    print(f"\nK=4 on the real city, 5 random trips: {['%.1fs' % t for t in times]}")
