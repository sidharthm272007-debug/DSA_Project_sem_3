"""Unit tests for restoration candidate generation and candidate type separation."""

import pytest
import networkx as nx
from restoration.candidates import generate_restoration_candidates
from analysis.isolation import detect_isolated_regions
from graph.graph_adapter import UndirectedGraphAdapter
from disaster.road_conditions import ScenarioState, RoadCondition


def test_candidate_generation_debris_and_hypothetical():
    g = nx.MultiDiGraph()
    # Main component nodes 1, 2
    g.add_node(1, y=37.7749, x=-122.4194)
    g.add_node(2, y=37.7750, x=-122.4190)
    g.add_edge(1, 2, key=0, length=50.0, name="Main St")

    # Isolated region nodes 3, 4
    g.add_node(3, y=37.7755, x=-122.4180)
    g.add_node(4, y=37.7760, x=-122.4175)
    g.add_edge(3, 4, key=0, length=60.0, name="Isolated Ave")

    # Connecting road 2 -> 3 is BLOCKED
    g.add_edge(2, 3, key=0, length=80.0, name="Bridge Way")

    state = ScenarioState()
    state.set_condition(2, 3, 0, RoadCondition.BLOCKED)

    adapter = UndirectedGraphAdapter(g, state)
    iso_report = detect_isolated_regions(adapter, reference_origin_node=1)

    candidates = generate_restoration_candidates(
        base_graph=g,
        scenario_state=state,
        isolation_report=iso_report,
        disaster_severity=3,
    )

    assert len(candidates) >= 1

    # Check for Debris Clearance candidate
    clearance_cands = [c for c in candidates if c.candidate_type == "DEBRIS_CLEARANCE"]
    assert len(clearance_cands) == 1
    c_cand = clearance_cands[0]
    assert c_cand.u_node == 2
    assert c_cand.v_node == 3
    assert c_cand.osm_edge_key == 0
    assert c_cand.metadata["road_name"] == "Bridge Way"
    assert c_cand.metadata["is_hypothetical"] is False
    assert c_cand.length_m == 80.0
    assert c_cand.cost > 0.0
