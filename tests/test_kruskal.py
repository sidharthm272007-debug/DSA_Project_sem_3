"""Unit tests for Kruskal's MST algorithm on contracted component graphs."""

import pytest
from algorithms.kruskal import ComponentEdge, run_kruskal_restoration


def test_kruskal_component_reconnection():
    # 3 components: Main, Region_1, Region_2
    components = ["Main", "Region_1", "Region_2"]

    # 3 candidate edges forming a triangle between components
    candidates = [
        ComponentEdge(
            edge_id="E1",
            u_comp="Main",
            v_comp="Region_1",
            cost=100.0,
            length_m=50.0,
            candidate_type="DEBRIS_CLEARANCE",
        ),
        ComponentEdge(
            edge_id="E2",
            u_comp="Region_1",
            v_comp="Region_2",
            cost=150.0,
            length_m=75.0,
            candidate_type="HYPOTHETICAL_LINK",
        ),
        ComponentEdge(
            edge_id="E3",
            u_comp="Main",
            v_comp="Region_2",
            cost=300.0,
            length_m=120.0,
            candidate_type="HYPOTHETICAL_LINK",
        ),
    ]

    res = run_kruskal_restoration(components, candidates, target_main_component="Main")

    assert res.is_fully_reconnected is True
    assert len(res.selected_edges) == 2
    # Kruskal selects E1 (100) and E2 (150)
    assert {e.edge_id for e in res.selected_edges} == {"E1", "E2"}
    assert res.total_cost == 250.0
    assert res.total_distance_m == 125.0


def test_kruskal_single_component_no_action():
    res = run_kruskal_restoration(["Main"], [], target_main_component="Main")
    assert res.is_fully_reconnected is True
    assert len(res.selected_edges) == 0
    assert res.total_cost == 0.0


def test_kruskal_unreconnected_island():
    # 3 components, but candidates only connect Main and Region_1; Region_2 has no candidate links
    components = ["Main", "Region_1", "Region_2"]
    candidates = [
        ComponentEdge(
            edge_id="E1",
            u_comp="Main",
            v_comp="Region_1",
            cost=100.0,
            length_m=50.0,
            candidate_type="DEBRIS_CLEARANCE",
        )
    ]
    res = run_kruskal_restoration(components, candidates, target_main_component="Main")
    assert res.is_fully_reconnected is False
    assert res.final_components_count == 2
