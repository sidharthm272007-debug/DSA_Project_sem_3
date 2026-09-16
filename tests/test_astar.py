"""Unit tests for A* search with admissible heuristic."""

import pytest
import networkx as nx
from algorithms.astar import run_astar
from algorithms.dijkstra import run_dijkstra
from graph.graph_adapter import DirectedGraphAdapter
from disaster.road_conditions import ScenarioState, RoadCondition
from graph.graph_utils import haversine_distance, get_node_coords


def make_grid_network():
    """Create a 3x3 grid directed network with geographic coordinates."""
    g = nx.MultiDiGraph()
    # Coordinates in decimal degrees around lat 37.7, lon -122.4
    nodes = {}
    idx = 0
    for r in range(3):
        for c in range(3):
            lat = 37.7000 + (r * 0.01)
            lon = -122.4000 + (c * 0.01)
            g.add_node(idx, y=lat, x=lon)
            nodes[(r, c)] = idx
            idx += 1

    # Connect horizontal and vertical grid edges
    for r in range(3):
        for c in range(3):
            u = nodes[(r, c)]
            u_lat, u_lon = get_node_coords(g, u)
            if c + 1 < 3:
                v = nodes[(r, c + 1)]
                v_lat, v_lon = get_node_coords(g, v)
                dist = haversine_distance(u_lat, u_lon, v_lat, v_lon)
                # 50 km/h = 13.88 m/s
                t = dist / 13.88
                g.add_edge(u, v, key=0, travel_time=t, length=dist, speed_kph=50.0)
                g.add_edge(v, u, key=0, travel_time=t, length=dist, speed_kph=50.0)
            if r + 1 < 3:
                v = nodes[(r + 1, c)]
                v_lat, v_lon = get_node_coords(g, v)
                dist = haversine_distance(u_lat, u_lon, v_lat, v_lon)
                t = dist / 13.88
                g.add_edge(u, v, key=0, travel_time=t, length=dist, speed_kph=50.0)
                g.add_edge(v, u, key=0, travel_time=t, length=dist, speed_kph=50.0)

    g.graph["v_max_mps"] = 20.0  # Safe upper bound (> 13.88 m/s)
    g.graph["v_max_kph"] = 72.0
    return g


def test_astar_admissibility_and_optimality():
    g = make_grid_network()
    state = ScenarioState()
    adapter = DirectedGraphAdapter(g, state)

    origin = 0  # bottom-left (0, 0)
    goal = 8    # top-right (2, 2)

    dijkstra_res = run_dijkstra(adapter, origin, goal)
    astar_res = run_astar(adapter, origin, goal)

    assert dijkstra_res.success is True
    assert astar_res.success is True

    # Critical requirement: A* with admissible heuristic MUST find the exact same optimal cost
    assert pytest.approx(astar_res.total_travel_time_sec, rel=1e-4) == dijkstra_res.total_travel_time_sec
    assert pytest.approx(astar_res.total_distance_m, rel=1e-4) == dijkstra_res.total_distance_m

    # A* should typically explore <= nodes than Dijkstra
    assert astar_res.nodes_explored <= dijkstra_res.nodes_explored


def test_astar_heuristic_never_overestimates():
    g = make_grid_network()
    state = ScenarioState()
    adapter = DirectedGraphAdapter(g, state)
    goal = 8
    v_max_mps = g.graph["v_max_mps"]

    goal_lat, goal_lon = adapter.get_node_coords(goal)

    # For every node, h(u) <= true_cost_to_goal(u)
    for u in g.nodes:
        u_lat, u_lon = adapter.get_node_coords(u)
        h_val = haversine_distance(u_lat, u_lon, goal_lat, goal_lon) / v_max_mps

        true_res = run_dijkstra(adapter, u, goal)
        if true_res.success:
            assert h_val <= true_res.total_travel_time_sec + 1e-6
