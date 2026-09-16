"""Unit tests for custom Breadth-First Search (BFS) disaster propagation."""

import pytest
from algorithms.bfs import run_bfs_propagation


def test_bfs_linear_graph():
    # 1 -> 2 -> 3 -> 4
    adj = {
        1: [(2, 10.0)],
        2: [(3, 20.0)],
        3: [(4, 30.0)],
        4: [],
    }
    res = run_bfs_propagation(adj, start_nodes=[1])
    assert res.visited_nodes == {1, 2, 3, 4}
    assert res.node_depths[1] == 0
    assert res.node_depths[2] == 1
    assert res.node_depths[3] == 2
    assert res.node_depths[4] == 3
    assert res.node_distances_m[4] == 60.0


def test_bfs_max_depth_cutoff():
    adj = {
        1: [(2, 10.0)],
        2: [(3, 10.0)],
        3: [(4, 10.0)],
        4: [],
    }
    res = run_bfs_propagation(adj, start_nodes=[1], max_depth=1)
    assert res.visited_nodes == {1, 2}
    assert 3 not in res.visited_nodes
    assert 4 not in res.visited_nodes


def test_bfs_empty_and_disconnected():
    # Empty
    res_empty = run_bfs_propagation({}, start_nodes=[])
    assert len(res_empty.visited_nodes) == 0

    # Disconnected node
    adj = {
        1: [(2, 5.0)],
        2: [],
        3: [(4, 5.0)],
        4: [],
    }
    res = run_bfs_propagation(adj, start_nodes=[1])
    assert res.visited_nodes == {1, 2}
    assert 3 not in res.visited_nodes


def test_bfs_cycles():
    # Triangle cycle 1-2-3-1
    adj = {
        1: [(2, 10.0), (3, 10.0)],
        2: [(3, 10.0)],
        3: [(1, 10.0)],
    }
    res = run_bfs_propagation(adj, start_nodes=[1])
    assert res.visited_nodes == {1, 2, 3}
    assert res.node_depths[2] == 1
    assert res.node_depths[3] == 1
