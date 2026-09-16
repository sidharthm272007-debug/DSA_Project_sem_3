"""Unit tests for Tarjan's bridge and articulation point algorithms."""

import pytest
from algorithms.tarjan import find_bridges_and_articulation_points


def test_tarjan_barbell_graph():
    # Triangle A: (1, 2, 3), Bridge: (3, 4), Triangle B: (4, 5, 6)
    adj = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2, 4],
        4: [3, 5, 6],
        5: [4, 6],
        6: [4, 5],
    }
    res = find_bridges_and_articulation_points(adj)

    # Exactly one bridge: between 3 and 4
    assert len(res.bridges) == 1
    u, v = res.bridges[0]
    assert {u, v} == {3, 4}

    # Articulation points must be 3 and 4
    assert res.articulation_points == {3, 4}


def test_tarjan_cycle_graph():
    # Simple 4-cycle: 1 - 2 - 3 - 4 - 1 (no bridges, no articulation points)
    adj = {
        1: [2, 4],
        2: [1, 3],
        3: [2, 4],
        4: [1, 3],
    }
    res = find_bridges_and_articulation_points(adj)
    assert len(res.bridges) == 0
    assert len(res.articulation_points) == 0


def test_tarjan_star_graph():
    # Star graph: Center node 0 connected to 1, 2, 3
    adj = {
        0: [1, 2, 3],
        1: [0],
        2: [0],
        3: [0],
    }
    res = find_bridges_and_articulation_points(adj)
    # Center node 0 is the articulation point
    assert res.articulation_points == {0}
    # All 3 edges are bridges
    assert len(res.bridges) == 3


def test_tarjan_parallel_edges_not_a_bridge():
    # Nodes 1 and 2 connected by TWO parallel physical edges
    adj = {
        1: [2],
        2: [1],
    }
    # Multiplicity between 1 and 2 is 2
    multiplicity = {("1", "2"): 2}
    res = find_bridges_and_articulation_points(adj, edge_multiplicity=multiplicity)
    # Since there are parallel paths, neither is a single point of failure
    assert len(res.bridges) == 0
