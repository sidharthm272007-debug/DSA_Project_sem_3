"""Validation-only Reference Tests using NetworkX as an Oracle.

CRITICAL ARCHITECTURAL RULE:
NetworkX is NOT used for DSA algorithm execution in the production ResQGraph
application. It is used here exclusively as an automated test oracle to validate
that our pure custom algorithms yield results identical to standard reference
implementations on deterministic benchmark graphs.
"""

import pytest
import networkx as nx
from algorithms.dijkstra import run_dijkstra
from algorithms.bfs import run_bfs_propagation
from algorithms.union_find import UnionFind
from algorithms.tarjan import find_bridges_and_articulation_points
from algorithms.kruskal import ComponentEdge, run_kruskal_restoration
from graph.graph_adapter import DirectedGraphAdapter
from disaster.road_conditions import ScenarioState


def test_dijkstra_against_networkx_oracle():
    """Validate custom Dijkstra shortest path and cost against nx.single_source_dijkstra."""
    # Build a directed weighted graph
    nx_g = nx.DiGraph()
    nx_g.add_edge("A", "B", travel_time=4.0, length=40.0)
    nx_g.add_edge("A", "C", travel_time=2.0, length=20.0)
    nx_g.add_edge("C", "B", travel_time=1.0, length=10.0)
    nx_g.add_edge("B", "D", travel_time=5.0, length=50.0)
    nx_g.add_edge("C", "D", travel_time=8.0, length=80.0)
    nx_g.add_edge("D", "E", travel_time=2.0, length=20.0)

    # MultiDiGraph equivalent for our adapter
    mdg = nx.MultiDiGraph()
    for u, v, data in nx_g.edges(data=True):
        mdg.add_edge(u, v, key=0, **data)
    for n in nx_g.nodes:
        mdg.add_node(n, y=0.0, x=0.0)

    adapter = DirectedGraphAdapter(mdg, ScenarioState())

    # Oracle query
    nx_cost, nx_path = nx.single_source_dijkstra(nx_g, source="A", target="E", weight="travel_time")

    # Custom implementation query
    custom_res = run_dijkstra(adapter, origin="A", destination="E")

    assert custom_res.success is True
    assert custom_res.total_travel_time_sec == nx_cost
    assert custom_res.path == nx_path


def test_bfs_against_networkx_oracle():
    """Validate custom BFS traversal against nx.single_source_shortest_path_length."""
    nx_g = nx.Graph()
    nx_g.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (5, 7)])

    # Oracle BFS depths
    nx_depths = nx.single_source_shortest_path_length(nx_g, source=1)

    # Custom BFS
    adj = {u: [(v, 1.0) for v in nx_g.neighbors(u)] for u in nx_g.nodes}
    custom_bfs = run_bfs_propagation(adj, start_nodes=[1])

    assert custom_bfs.visited_nodes == set(nx_g.nodes)
    assert custom_bfs.node_depths == nx_depths


def test_union_find_against_networkx_oracle():
    """Validate custom UnionFind connected components against nx.connected_components."""
    nx_g = nx.Graph()
    # 3 disjoint components: {1, 2, 3}, {4, 5}, {6}
    nx_g.add_edges_from([(1, 2), (2, 3), (4, 5)])
    nx_g.add_node(6)

    nx_comps = [set(c) for c in nx.connected_components(nx_g)]

    uf = UnionFind(nx_g.nodes)
    for u, v in nx_g.edges:
        uf.union(u, v)

    custom_comps = [set(c) for c in uf.get_components().values()]

    assert len(custom_comps) == len(nx_comps)
    # Verify every component matches an oracle component
    for comp in custom_comps:
        assert comp in nx_comps


def test_tarjan_bridges_and_aps_against_networkx_oracle():
    """Validate custom Tarjan bridges & articulation points against NetworkX reference."""
    # Graph with known bridge and articulation point:
    # 1-2-3-1 (triangle), 3-4 (bridge), 4-5-6-4 (triangle)
    nx_g = nx.Graph()
    nx_g.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4)])

    # NetworkX oracle
    nx_bridges = {tuple(sorted(e)) for e in nx.bridges(nx_g)}
    nx_aps = set(nx.articulation_points(nx_g))

    # Custom Tarjan
    adj = {u: list(nx_g.neighbors(u)) for u in nx_g.nodes}
    custom_res = find_bridges_and_articulation_points(adj)

    custom_bridges = {tuple(sorted(e)) for e in custom_res.bridges}

    assert custom_bridges == nx_bridges
    assert custom_res.articulation_points == nx_aps


def test_kruskal_against_networkx_oracle():
    """Validate custom Kruskal MST on contracted components against nx.minimum_spanning_tree."""
    nx_g = nx.Graph()
    edges_data = [
        ("C1", "C2", 12.0),
        ("C1", "C3", 8.0),
        ("C2", "C3", 10.0),
        ("C2", "C4", 15.0),
        ("C3", "C4", 7.0),
    ]
    for u, v, w in edges_data:
        nx_g.add_edge(u, v, weight=w)

    nx_mst = nx.minimum_spanning_tree(nx_g, algorithm="kruskal", weight="weight")
    nx_mst_cost = sum(d["weight"] for _, _, d in nx_mst.edges(data=True))

    # Custom Kruskal
    candidates = [
        ComponentEdge(
            edge_id=f"E{idx}",
            u_comp=u,
            v_comp=v,
            cost=w,
            length_m=w * 10,
            candidate_type="DEBRIS_CLEARANCE",
        )
        for idx, (u, v, w) in enumerate(edges_data)
    ]
    custom_kruskal = run_kruskal_restoration(list(nx_g.nodes), candidates)

    assert custom_kruskal.is_fully_reconnected is True
    assert custom_kruskal.total_cost == nx_mst_cost
