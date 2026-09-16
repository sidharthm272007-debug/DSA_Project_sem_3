"""Unit tests for isolated-region detection and facility tracking."""

import pytest
import networkx as nx
from analysis.isolation import detect_isolated_regions
from graph.graph_adapter import UndirectedGraphAdapter
from disaster.road_conditions import ScenarioState, RoadCondition


def test_isolated_region_detection():
    # Graph with 2 disconnected clusters:
    # Cluster A: nodes 1, 2, 3
    # Cluster B: nodes 4, 5
    # Link (3, 4) is blocked
    g = nx.MultiDiGraph()
    for i in range(1, 6):
        g.add_node(i, y=0.0, x=float(i) * 0.01)

    g.add_edge(1, 2, key=0)
    g.add_edge(2, 3, key=0)
    g.add_edge(3, 4, key=0)
    g.add_edge(4, 5, key=0)

    state = ScenarioState()
    # Block the connection between cluster A and cluster B
    state.set_condition(3, 4, 0, RoadCondition.BLOCKED)

    adapter = UndirectedGraphAdapter(g, state)

    facilities = [
        {"name": "Hospital Central", "type": "hospital", "node": 1, "lat": 0.0, "lon": 0.01},
        {"name": "East Clinic", "type": "clinic", "node": 5, "lat": 0.0, "lon": 0.05},
    ]

    report = detect_isolated_regions(
        undirected_adapter=adapter,
        reference_origin_node=1,
        facilities=facilities,
    )

    assert report.total_components == 2
    assert report.has_isolated_regions is True
    assert report.isolated_regions_count == 1
    assert set(report.main_component_nodes) == {1, 2, 3}

    isolated = report.isolated_regions[0]
    assert set(isolated.nodes) == {4, 5}
    # East Clinic is trapped in the isolated region!
    assert len(isolated.trapped_facilities) == 1
    assert isolated.trapped_facilities[0]["name"] == "East Clinic"
