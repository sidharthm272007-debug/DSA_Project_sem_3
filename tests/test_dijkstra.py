"""Unit tests for Dijkstra's algorithm with lazy deletion."""

import pytest
import networkx as nx
from algorithms.dijkstra import run_dijkstra
from graph.graph_adapter import DirectedGraphAdapter
from disaster.road_conditions import ScenarioState, RoadCondition


def make_test_multigraph():
    """Create a 4-node directed multigraph:
       (1) --[time=10, len=100]--> (2) --[time=10, len=100]--> (4)
        |                                                        ^
        +----[time=25, len=250]---> (3) ---[time=5, len=50]-----+
    """
    g = nx.MultiDiGraph()
    g.add_node(1, y=0.0, x=0.0)
    g.add_node(2, y=0.0, x=1.0)
    g.add_node(3, y=1.0, x=0.0)
    g.add_node(4, y=1.0, x=1.0)

    g.add_edge(1, 2, key=0, travel_time=10.0, length=100.0)
    g.add_edge(2, 4, key=0, travel_time=10.0, length=100.0)
    g.add_edge(1, 3, key=0, travel_time=25.0, length=250.0)
    g.add_edge(3, 4, key=0, travel_time=5.0, length=50.0)
    return g


def test_dijkstra_shortest_path():
    g = make_test_multigraph()
    state = ScenarioState()
    adapter = DirectedGraphAdapter(g, state)

    res = run_dijkstra(adapter, origin=1, destination=4)
    assert res.success is True
    # Path via 2: 10 + 10 = 20 vs Path via 3: 25 + 5 = 30
    assert res.path == [1, 2, 4]
    assert res.total_travel_time_sec == 20.0
    assert res.total_distance_m == 200.0


def test_dijkstra_blocked_road_detour():
    g = make_test_multigraph()
    state = ScenarioState()
    # Block edge (1 -> 2)
    state.set_condition(1, 2, 0, RoadCondition.BLOCKED)
    adapter = DirectedGraphAdapter(g, state)

    res = run_dijkstra(adapter, origin=1, destination=4)
    assert res.success is True
    # Now forced to take detour via 3: 25 + 5 = 30
    assert res.path == [1, 3, 4]
    assert res.total_travel_time_sec == 30.0


def test_dijkstra_slowed_road_penalties():
    g = make_test_multigraph()
    state = ScenarioState()
    # Slow down edge (1 -> 2): time becomes 10 * 2.0 = 20.0, total via 2 becomes 30.0
    # Make it SEVERELY_AFFECTED: 10 * 4.0 = 40.0, total via 2 becomes 50.0
    state.set_condition(1, 2, 0, RoadCondition.SEVERELY_AFFECTED)
    adapter = DirectedGraphAdapter(g, state)

    res = run_dijkstra(adapter, origin=1, destination=4)
    assert res.success is True
    # Detour via 3 is 30.0, which is faster than 50.0!
    assert res.path == [1, 3, 4]
    assert res.total_travel_time_sec == 30.0


def test_dijkstra_no_path_failure():
    g = make_test_multigraph()
    state = ScenarioState()
    # Block both exits from node 1
    state.set_condition(1, 2, 0, RoadCondition.BLOCKED)
    state.set_condition(1, 3, 0, RoadCondition.BLOCKED)
    adapter = DirectedGraphAdapter(g, state)

    res = run_dijkstra(adapter, origin=1, destination=4)
    assert res.success is False
    assert "No route is currently available" in res.failure_reason


def test_dijkstra_same_origin_and_destination():
    g = make_test_multigraph()
    state = ScenarioState()
    adapter = DirectedGraphAdapter(g, state)

    res = run_dijkstra(adapter, origin=1, destination=1)
    assert res.success is True
    assert res.path == [1]
    assert res.total_travel_time_sec == 0.0
