"""Restoration Optimization via Kruskal's Algorithm on Contracted Component Graph.

Applies Kruskal's MST algorithm over component super-nodes to select the minimum-cost
set of restoration connections (debris clearances and emergency bypasses) that re-establishes
network-wide connectivity.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
import networkx as nx
from algorithms.kruskal import ComponentEdge, KruskalResult, run_kruskal_restoration
from analysis.isolation import IsolationAnalysisReport
from restoration.candidates import generate_restoration_candidates
from disaster.road_conditions import ScenarioState
from config import SYNTHETIC_RESTORATION_CONFIG


@dataclass
class RestorationPlan:
    """Comprehensive restoration strategy recommendation."""
    selected_connections: List[ComponentEdge] = field(default_factory=list)
    total_synthetic_cost: float = 0.0
    total_distance_m: float = 0.0
    debris_clearance_count: int = 0
    hypothetical_links_count: int = 0
    components_reconnected_count: int = 0
    unreconnected_regions: List[str] = field(default_factory=list)
    is_fully_reconnected: bool = False
    kruskal_result: Optional[KruskalResult] = None
    disclaimer: str = SYNTHETIC_RESTORATION_CONFIG["disclaimer"]

    @property
    def total_connections_count(self) -> int:
        return len(self.selected_connections)


def build_restoration_plan(
    base_graph: nx.MultiDiGraph,
    scenario_state: ScenarioState,
    isolation_report: IsolationAnalysisReport,
    disaster_severity: int = 3,
) -> RestorationPlan:
    """Compute optimal restoration plan reconnecting isolated regions to the main network.
    
    Args:
        base_graph: Authoritative OSM MultiDiGraph.
        scenario_state: Shared scenario state.
        isolation_report: Output of detect_isolated_regions.
        disaster_severity: Scenario severity for cost scaling.
        
    Returns:
        RestorationPlan with Kruskal's MST selected connections and synthetic metrics.
    """
    plan = RestorationPlan()

    if not isolation_report.has_isolated_regions:
        plan.is_fully_reconnected = True
        return plan

    # 1. Generate all candidate restoration edges
    candidates = generate_restoration_candidates(
        base_graph=base_graph,
        scenario_state=scenario_state,
        isolation_report=isolation_report,
        disaster_severity=disaster_severity,
    )

    if not candidates:
        plan.unreconnected_regions = [r.region_id for r in isolation_report.isolated_regions]
        plan.is_fully_reconnected = False
        return plan

    # 2. Super-nodes: all component IDs (Main_Component + Region_1, Region_2, ...)
    all_comp_ids = list(isolation_report.all_components_map.keys())

    # 3. Execute Kruskal's MST on the contracted component graph
    kruskal_res = run_kruskal_restoration(
        component_ids=all_comp_ids,
        candidate_edges=candidates,
        target_main_component=isolation_report.main_component_id,
    )

    plan.selected_connections = kruskal_res.selected_edges
    plan.total_synthetic_cost = kruskal_res.total_cost
    plan.total_distance_m = kruskal_res.total_distance_m
    plan.kruskal_result = kruskal_res

    # Count candidate breakdown
    for edge in kruskal_res.selected_edges:
        if edge.candidate_type == "DEBRIS_CLEARANCE":
            plan.debris_clearance_count += 1
        else:
            plan.hypothetical_links_count += 1

    plan.components_reconnected_count = len(kruskal_res.reconnected_component_ids)
    plan.is_fully_reconnected = kruskal_res.is_fully_reconnected

    # Check which regions remain unreconnected
    all_isolated_ids = {r.region_id for r in isolation_report.isolated_regions}
    plan.unreconnected_regions = list(all_isolated_ids - kruskal_res.reconnected_component_ids)

    return plan
