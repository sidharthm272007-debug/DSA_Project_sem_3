"""Unit tests for Disjoint Set Union (Union-Find) with path compression and rank."""

import pytest
from algorithms.union_find import UnionFind


def test_union_find_basic_merges():
    uf = UnionFind(range(5))
    assert uf.component_count() == 5

    # Union (0, 1)
    assert uf.union(0, 1) is True
    assert uf.component_count() == 4
    assert uf.connected(0, 1) is True
    assert uf.connected(0, 2) is False

    # Redundant union should return False and not reduce count
    assert uf.union(0, 1) is False
    assert uf.component_count() == 4

    # Union (2, 3)
    uf.union(2, 3)
    assert uf.component_count() == 3

    # Union (1, 3) should merge {0, 1} and {2, 3} into {0, 1, 2, 3}
    uf.union(1, 3)
    assert uf.component_count() == 2
    assert uf.connected(0, 3) is True
    assert uf.connected(1, 2) is True
    assert uf.connected(0, 4) is False


def test_union_find_path_compression():
    # Chain: 0 - 1 - 2 - 3 - 4
    uf = UnionFind(range(5))
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(3, 4)

    # Calling find on 4 compresses the path to the root
    root = uf.find(4)
    for i in range(5):
        assert uf.find(i) == root

    comps = uf.get_components()
    assert len(comps) == 1
    assert set(comps[root]) == {0, 1, 2, 3, 4}


def test_union_find_dynamic_addition():
    uf = UnionFind()
    uf.add("A")
    uf.add("B")
    assert uf.component_count() == 2
    uf.union("A", "B")
    assert uf.connected("A", "B") is True
    assert uf.component_count() == 1
