"""Restoration Candidate Generation for Isolated Network Pockets.

Discovers and evaluates possible reconnection options between isolated regions
and the accessible main network.

CRITICAL SEPARATION OF CANDIDATE TYPES:
  1. DEBRIS CLEARANCE: Reopening existing physical OSM roads that are currently
     BLOCKED by the disaster. References genuine OSM (u, v, key) attributes.
  2. HYPOTHETICAL EMERGENCY LINK: Conceptual connections across geographic gaps
     (e.g. temporary pontoon bridge or emergency bypass). Explicitly labeled as
     SIMULATED/CONCEPTUAL and NEVER presented as an existing physical road.

DISCLAIMER:
  All restoration costs, obstacle penalties, and rankings are SYNTHETIC simulation
  heuristics for comparative graph optimization. Not civil engineering estimates.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import networkx as nx
from algorithms.kruskal import ComponentEdge
from analysis.isolation import IsolationAnalysisReport
from disaster.road_conditions import ScenarioState, RoadCondition
from graph.graph_utils import haversine_distance, get_node_coords
from config import SYNTHETIC_RESTORATION_CONFIG


def generate_restoration_candidates(
    base_graph: nx.MultiDiGraph,
    scenario_state: ScenarioState,
    isolation_report: IsolationAnalysisReport,
    disaster_severity: int = 3,
    max_gap_distance_m: float = 2500.0,
) -> List[ComponentEdge]:
    """Generate both existing-road debris clearance and hypothetical emergency candidates.
    
    Args:
        base_graph: Authoritative OSM MultiDiGraph.
        scenario_state: Shared scenario state containing blocked edges.
        isolation_report: Output from detect_isolated_regions.
        disaster_severity: Scenario severity (1 to 5) for cost penalty scaling.
        max_gap_distance_m: Maximum geographic search distance for hypothetical bypasses.
        
    Returns:
        List of ComponentEdge objects ready for contracted component-level Kruskal MST.
    """
    candidates: List[ComponentEdge] = []
    cfg = SYNTHETIC_RESTORATION_CONFIG
    base_debris_rate = cfg["base_debris_clearance_per_meter"]
    base_hypo_rate = cfg["base_hypothetical_link_per_meter"]
    sev_mult = cfg["severity_penalty_multiplier"].get(disaster_severity, 1.5)

    # Build fast node -> component_id lookup map
    node_to_comp: Dict[Any, str] = {}
    for comp_id, nodes in isolation_report.all_components_map.items():
        for n in nodes:
            node_to_comp[n] = comp_id

    # -------------------------------------------------------------
    # TYPE 1: Existing-Road Debris Clearance Candidates
    # -------------------------------------------------------------
    # Examine all currently BLOCKED edges in the scenario state
    blocked_edges = scenario_state.get_blocked_edges()
    cand_idx = 1

    for u, v, k in blocked_edges:
        comp_u = node_to_comp.get(u)
        comp_v = node_to_comp.get(v)

        # Candidate must span across two different components (e.g. Region_1 and Main_Component)
        if comp_u is not None and comp_v is not None and comp_u != comp_v:
            # Fetch real OSM edge attributes
            edge_data = base_graph.get_edge_data(u, v, k, default={})
            length = edge_data.get("length", 50.0)
            road_name = edge_data.get("name", "Existing Street")
            if isinstance(road_name, list) and road_name:
                road_name = str(road_name[0])

            # Synthetic cost heuristic: length * base_debris_rate * severity_penalty
            cost = round(length * base_debris_rate * sev_mult, 1)

            candidates.append(
                ComponentEdge(
                    edge_id=f"CLEAR-{cand_idx}",
                    u_comp=comp_u,
                    v_comp=comp_v,
                    cost=cost,
                    length_m=round(length, 1),
                    candidate_type="DEBRIS_CLEARANCE",
                    u_node=u,
                    v_node=v,
                    osm_edge_key=k,
                    metadata={
                        "road_name": road_name,
                        "is_hypothetical": False,
                        "description": f"Clear debris on existing {road_name} ({length:.0f}m)",
                        "disclaimer": cfg["disclaimer"],
                    },
                )
            )
            cand_idx += 1

    # -------------------------------------------------------------
    # TYPE 2: Hypothetical Emergency Links (Conceptual Bypasses)
    # -------------------------------------------------------------
    # For each isolated region, discover closest boundary node pair to other components
    # (especially Main_Component) where no direct blocked road exists
    main_nodes = isolation_report.main_component_nodes
    main_id = isolation_report.main_component_id

    # Subsample main nodes if main component is very large for candidate generation performance
    sampled_main_nodes = main_nodes
    if len(main_nodes) > 200:
        step = max(1, len(main_nodes) // 100)
        sampled_main_nodes = main_nodes[::step]

    hypo_idx = 1
    for region in isolation_report.isolated_regions:
        r_id = region.region_id
        # Check if debris clearance already connects r_id to main_id
        already_has_clearance = any(
            (c.u_comp == r_id and c.v_comp == main_id) or (c.u_comp == main_id and c.v_comp == r_id)
            for c in candidates
        )

        # Search for minimum geographic gap between region nodes and main nodes
        best_pair = None
        min_dist = float("inf")

        # Check up to 50 sample nodes in the isolated region
        sample_r_nodes = region.nodes[:50]
        for u in sample_r_nodes:
            u_lat, u_lon = get_node_coords(base_graph, u)
            for v in sampled_main_nodes:
                v_lat, v_lon = get_node_coords(base_graph, v)
                dist = haversine_distance(u_lat, u_lon, v_lat, v_lon)
                if dist < min_dist and dist <= max_gap_distance_m:
                    min_dist = dist
                    best_pair = (u, v)

        if best_pair is not None and min_dist > 5.0:
            u_best, v_best = best_pair
            # Synthetic cost heuristic for hypothetical construction/pontoon link
            cost = round(min_dist * base_hypo_rate * sev_mult, 1)

            candidates.append(
                ComponentEdge(
                    edge_id=f"HYPO-{hypo_idx}",
                    u_comp=r_id,
                    v_comp=main_id,
                    cost=cost,
                    length_m=round(min_dist, 1),
                    candidate_type="HYPOTHETICAL_LINK",
                    u_node=u_best,
                    v_node=v_best,
                    osm_edge_key=None,
                    metadata={
                        "road_name": "[SIMULATED CONCEPTUAL LINK - NO EXISTING ROAD]",
                        "is_hypothetical": True,
                        "description": (
                            f"Conceptual emergency bypass/bridge ({min_dist:.0f}m gap) "
                            f"between {r_id} and {main_id}."
                        ),
                        "disclaimer": cfg["disclaimer"],
                    },
                )
            )
            hypo_idx += 1

    return candidates
