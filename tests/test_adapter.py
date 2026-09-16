"""Integration tests verifying shared ScenarioState between Directed and Undirected adapters."""

import pytest
import networkx as nx
from graph.graph_adapter import DirectedGraphAdapter, UndirectedGraphAdapter
from disaster.road_conditions import ScenarioState, RoadCondition


def test_shared_scenario_state_synchronization():
    """Verify that changing a road condition in the shared ScenarioState is immediately

    and identically observed by both the directed routing adapter and the undirected
    structural connectivity adapter without maintaining separate copies.
    """
    g = nx.MultiDiGraph()
    g.add_node(1, y=0.0, x=0.0)
    g.add_node(2, y=0.0, x=1.0)
    g.add_node(3, y=0.0, x=2.0)

    g.add_edge(1, 2, key=0, travel_time=10.0, length=100.0)
    g.add_edge(2, 3, key=0, travel_time=15.0, length=150.0)

    # Single shared ScenarioState instance
    shared_state = ScenarioState()

    # Both adapters reference the EXACT SAME shared_state
    directed_adapter = DirectedGraphAdapter(g, shared_state)
    undirected_adapter = UndirectedGraphAdapter(g, shared_state)

    # 1. Initially all edges are NORMAL
    # In directed adapter: 1 has outgoing edge to 2
    out_1 = directed_adapter.get_outgoing_edges(1)
    assert len(out_1) == 1
    assert out_1[0][0] == 2
    assert out_1[0][4] == RoadCondition.NORMAL

    # In undirected adapter: 2 is a physical neighbor of 1
    assert 2 in undirected_adapter.get_neighbors(1)

    # 2. Modify condition in the shared state: block edge (1, 2, 0)
    shared_state.set_condition(1, 2, 0, RoadCondition.BLOCKED)

    # 3. Directed adapter must immediately observe the blockage (outgoing edge disappears)
    out_1_after = directed_adapter.get_outgoing_edges(1)
    assert len(out_1_after) == 0  # BLOCKED edge filtered out of routing!

    # 4. Undirected adapter must also immediately observe the blockage (neighbor relationship severed)
    nbrs_1_after = undirected_adapter.get_neighbors(1)
    assert 2 not in nbrs_1_after

    # 5. Now change condition to SLOWED in the shared state
    shared_state.set_condition(1, 2, 0, RoadCondition.SLOWED)

    # Directed adapter sees it re-open with 2.0x travel-time penalty
    out_1_slowed = directed_adapter.get_outgoing_edges(1)
    assert len(out_1_slowed) == 1
    target, key, eff_time, length, cond = out_1_slowed[0]
    assert target == 2
    assert cond == RoadCondition.SLOWED
    assert eff_time == 20.0  # 10.0 * 2.0 = 20.0

    # Undirected adapter sees it physically reconnected
    assert 2 in undirected_adapter.get_neighbors(1)
