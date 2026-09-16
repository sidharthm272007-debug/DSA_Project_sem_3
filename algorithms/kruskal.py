"""Kruskal's Minimum Spanning Tree (MST) Algorithm for Component Restoration.

Purpose:
  Answers the operational optimization question:
  "What is the minimum-cost set of restoration connections required to reconnect
  isolated network pockets back to the accessible main network?"

Architectural Design (Component-Level Contraction):
  Rather than running Kruskal naively across thousands of individual street intersections,
  the post-disaster network is contracted into a component-level graph where:
    - Each connected component (including the Main Component and each Isolated Region)
      forms a single super-node C_i.
    - Candidate restoration connections (debris clearances or emergency bypasses)
      form weighted undirected edges between component pairs (C_i, C_j).
    - Kruskal's algorithm runs on this contracted component graph to select the
      minimal-cost spanning forest/tree.

Disclaimer:
  SIMULATION ESTIMATE ONLY. Costs are synthetic optimization weights and do not
  represent real civil engineering, governmental, or financial budgets.

Complexity:
  - Time Complexity: O(E log E), dominated by sorting the candidate restoration edges.
  - Space Complexity: O(V_comp + E_cand) using the custom UnionFind data structure.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from algorithms.union_find import UnionFind


@dataclass
class ComponentEdge:
    """Represents a restoration candidate connection between two contracted components."""
    edge_id: str
    u_comp: Any  # Super-node ID of component 1
    v_comp: Any  # Super-node ID of component 2
    cost: float  # Synthetic restoration cost
    length_m: float  # Physical distance in meters
    candidate_type: str  # 'DEBRIS_CLEARANCE' or 'HYPOTHETICAL_LINK'
    # Reference to original network endpoints or OSM road data
    u_node: Any = None
    v_node: Any = None
    osm_edge_key: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KruskalResult:
    """Encapsulates the output of Kruskal's MST component restoration."""
    selected_edges: List[ComponentEdge] = field(default_factory=list)
    total_cost: float = 0.0
    total_distance_m: float = 0.0
    initial_components_count: int = 0
    final_components_count: int = 0
    is_fully_reconnected: bool = False
    reconnected_component_ids: Set[Any] = field(default_factory=set)


def run_kruskal_restoration(
    component_ids: List[Any],
    candidate_edges: List[ComponentEdge],
    target_main_component: Optional[Any] = None,
) -> KruskalResult:
    """Execute Kruskal's algorithm on the contracted component graph.
    
    Args:
        component_ids: List of all component super-node IDs (main + isolated).
        candidate_edges: List of available restoration candidate connections.
        target_main_component: Optional ID of the primary emergency response component.
        
    Returns:
        KruskalResult with minimum-cost selected edges and connection metrics.
    """
    result = KruskalResult(initial_components_count=len(component_ids))

    if not component_ids:
        return result

    if len(component_ids) == 1:
        result.final_components_count = 1
        result.is_fully_reconnected = True
        return result

    # 1. Sort candidate edges in ascending order of synthetic cost (greedy choice)
    sorted_candidates = sorted(candidate_edges, key=lambda e: e.cost)

    # 2. Initialize custom Union-Find on component super-nodes
    uf = UnionFind(component_ids)

    selected: List[ComponentEdge] = []
    total_cost = 0.0
    total_dist = 0.0
    reconnected_comps = set()

    for edge in sorted_candidates:
        # Check if endpoints belong to different component sets
        if uf.union(edge.u_comp, edge.v_comp):
            selected.append(edge)
            total_cost += edge.cost
            total_dist += edge.length_m
            reconnected_comps.add(edge.u_comp)
            reconnected_comps.add(edge.v_comp)

            # Early termination: if all components merged into 1
            if uf.component_count() == 1:
                break

    result.selected_edges = selected
    result.total_cost = total_cost
    result.total_distance_m = total_dist
    result.final_components_count = uf.component_count()
    result.reconnected_component_ids = reconnected_comps

    # If target_main_component is specified, check if all components are in same set as main
    if target_main_component is not None:
        main_root = uf.find(target_main_component)
        all_connected_to_main = all(
            uf.find(c) == main_root for c in component_ids
        )
        result.is_fully_reconnected = all_connected_to_main
    else:
        result.is_fully_reconnected = (uf.component_count() == 1)

    return result
