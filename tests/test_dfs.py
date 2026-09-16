"""Unit tests for custom Depth-First Search (DFS) reachability analysis."""

import pytest
from algorithms.dfs import run_dfs_reachability


def test_dfs_reachability_all_connected():
    # 1 -> 2 -> 3
    adj = {
        1: [2],
        2: [3],
        3: [],
    }
    all_nodes = [1, 2, 3]
    res = run_dfs_reachability(adj, all_nodes, origin_node=1)
    assert res.reachable_nodes == {1, 2, 3}
    assert res.unreachable_nodes == set()
    assert res.reachability_percentage == 100.0
    assert res.is_fully_connected is True


def test_dfs_reachability_partial_disconnected():
    # 1 -> 2; 3 -> 4 (disconnected from 1)
    adj = {
        1: [2],
        2: [],
        3: [4],
        4: [],
    }
    all_nodes = [1, 2, 3, 4]
    res = run_dfs_reachability(adj, all_nodes, origin_node=1)
    assert res.reachable_nodes == {1, 2}
    assert res.unreachable_nodes == {3, 4}
    assert res.reachability_percentage == 50.0
    assert res.is_fully_connected is False


def test_dfs_directed_one_way():
    # 1 -> 2, but no edge 2 -> 1
    adj = {
        1: [2],
        2: [],
    }
    all_nodes = [1, 2]
    # Reachable from 1: 1, 2
    res1 = run_dfs_reachability(adj, all_nodes, origin_node=1)
    assert res1.reachable_nodes == {1, 2}

    # Reachable from 2: only 2
    res2 = run_dfs_reachability(adj, all_nodes, origin_node=2)
    assert res2.reachable_nodes == {2}
    assert res2.unreachable_nodes == {1}


def test_dfs_empty_and_single_node():
    res = run_dfs_reachability({}, [], origin_node=1)
    assert len(res.reachable_nodes) == 0

    res_single = run_dfs_reachability({1: []}, [1], origin_node=1)
    assert res_single.reachable_nodes == {1}
    assert res_single.reachability_percentage == 100.0
