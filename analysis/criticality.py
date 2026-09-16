"""Critical Infrastructure & Structural Vulnerability Analysis.

Uses Tarjan's low-link algorithm to identify structural single points of failure
(critical bridges and articulation-point intersections).

IMPORTANT METHODOLOGICAL NOTE:
Phase 1 does NOT model real traffic flow or congestion because no real traffic
sensor dataset is present. All vulnerability assessments are derived strictly
from structural topological connectivity, component severance impacts, and
facility accessibility metrics.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from algorithms.tarjan import find_bridges_and_articulation_points, CriticalityResult
from graph.graph_adapter import UndirectedGraphAdapter
from graph.graph_utils import get_node_coords, haversine_distance


@dataclass
class BridgeDetail:
    """Detailed structural assessment of a critical bridge road."""
    u: Any
    v: Any
    length_m: float
    u_coords: Tuple[float, float]
    v_coords: Tuple[float, float]
    road_name: str = "Unnamed Road"
    severance_impact_description: str = ""


@dataclass
class ArticulationPointDetail:
    """Detailed structural assessment of a critical junction intersection."""
    node_id: Any
    coords: Tuple[float, float]
    incident_edges_count: int
    severance_impact_description: str = ""


@dataclass
class CriticalityAnalysisReport:
    """Consolidated structural resilience report."""
    bridges: List[BridgeDetail] = field(default_factory=list)
    articulation_points: List[ArticulationPointDetail] = field(default_factory=list)
    total_nodes_analyzed: int = 0
    total_edges_analyzed: int = 0
    methodology_note: str = (
        "Based on structural topological connectivity and emergency access metrics. "
        "No real-world traffic sensor flow is assumed in Phase 1."
    )

    @property
    def bridge_count(self) -> int:
        return len(self.bridges)

    @property
    def articulation_point_count(self) -> int:
        return len(self.articulation_points)


def analyze_criticality(
    undirected_adapter: UndirectedGraphAdapter,
) -> CriticalityAnalysisReport:
    """Run Tarjan's algorithms and generate structural criticality assessments.
    
    Args:
        undirected_adapter: UndirectedGraphAdapter with current scenario conditions.
        
    Returns:
        CriticalityAnalysisReport with bridge and articulation point breakdowns.
    """
    base_graph = undirected_adapter.base_graph

    # Build undirected adjacency and calculate edge multiplicity
    adjacency: Dict[Any, List[Any]] = {}
    edge_multiplicity: Dict[Tuple[Any, Any], int] = {}

    for u in undirected_adapter.nodes:
        nbrs = list(undirected_adapter.get_neighbors(u))
        adjacency[u] = nbrs
        for v in nbrs:
            canonical = (min(str(u), str(v)), max(str(u), str(v)))
            # Count active OSM edges between u and v
            active_osm = undirected_adapter.get_active_osm_edges_between(u, v)
            edge_multiplicity[canonical] = len(active_osm)

    tarjan_res: CriticalityResult = find_bridges_and_articulation_points(
        undirected_adjacency=adjacency,
        edge_multiplicity=edge_multiplicity,
    )

    # Enrich bridge details
    bridge_details: List[BridgeDetail] = []
    for u, v in tarjan_res.bridges:
        u_lat, u_lon = undirected_adapter.get_node_coords(u)
        v_lat, v_lon = undirected_adapter.get_node_coords(v)
        dist = haversine_distance(u_lat, u_lon, v_lat, v_lon)

        # Lookup name from base graph
        road_name = "Unnamed Road"
        if base_graph.has_edge(u, v):
            data = next(iter(base_graph[u][v].values()))
            name = data.get("name")
            if name:
                road_name = name if isinstance(name, str) else str(name[0])
        elif base_graph.has_edge(v, u):
            data = next(iter(base_graph[v][u].values()))
            name = data.get("name")
            if name:
                road_name = name if isinstance(name, str) else str(name[0])

        impact = (
            f"Structural Bridge: Loss of this {dist:.0f}m link ({road_name}) strictly increases "
            "the number of disconnected components in the network."
        )

        bridge_details.append(
            BridgeDetail(
                u=u,
                v=v,
                length_m=dist,
                u_coords=(u_lat, u_lon),
                v_coords=(v_lat, v_lon),
                road_name=road_name,
                severance_impact_description=impact,
            )
        )

    # Enrich articulation point details
    ap_details: List[ArticulationPointDetail] = []
    for node_id in tarjan_res.articulation_points:
        coords = undirected_adapter.get_node_coords(node_id)
        nbr_count = len(adjacency.get(node_id, []))
        impact = (
            f"Critical Intersection: Removal of junction {node_id} (degree {nbr_count}) "
            "severs connectivity between adjacent neighborhoods."
        )

        ap_details.append(
            ArticulationPointDetail(
                node_id=node_id,
                coords=coords,
                incident_edges_count=nbr_count,
                severance_impact_description=impact,
            )
        )

    return CriticalityAnalysisReport(
        bridges=bridge_details,
        articulation_points=ap_details,
        total_nodes_analyzed=tarjan_res.total_nodes_analyzed,
        total_edges_analyzed=tarjan_res.total_edges_analyzed,
    )
