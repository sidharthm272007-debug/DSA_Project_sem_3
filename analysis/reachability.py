"""Reachability analysis using Depth-First Search (DFS).

Audits post-disaster accessibility from the emergency response origin to all
network intersections and critical facilities (hospitals, fire stations, etc.).
"""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from algorithms.dfs import run_dfs_reachability, DFSReachabilityResult
from graph.graph_adapter import DirectedGraphAdapter


@dataclass
class FacilityReachabilityAudit:
    """Status of critical infrastructure reachability from origin."""
    facility_name: str
    facility_type: str
    nearest_node: Any
    is_reachable: bool
    lat: float
    lon: float


@dataclass
class ReachabilityAnalysisReport:
    """Comprehensive report summarizing post-disaster reachability."""
    origin_node: Any
    total_nodes: int
    reachable_nodes_count: int
    unreachable_nodes_count: int
    reachability_percentage: float
    dfs_result: DFSReachabilityResult
    facility_audits: List[FacilityReachabilityAudit] = field(default_factory=list)

    @property
    def reachable_facilities_count(self) -> int:
        return sum(1 for f in self.facility_audits if f.is_reachable)

    @property
    def unreachable_facilities_count(self) -> int:
        return sum(1 for f in self.facility_audits if not f.is_reachable)


def analyze_reachability(
    directed_adapter: DirectedGraphAdapter,
    origin_node: Any,
    facilities: Optional[List[Dict[str, Any]]] = None,
) -> ReachabilityAnalysisReport:
    """Perform reachability analysis from an emergency origin over directed non-blocked roads.
    
    Args:
        directed_adapter: DirectedGraphAdapter reflecting current disaster scenario.
        origin_node: Node ID of emergency response center.
        facilities: Optional list of facility dicts with keys ('name', 'type', 'node', 'lat', 'lon').
        
    Returns:
        ReachabilityAnalysisReport with statistics and facility accessibility breakdown.
    """
    all_nodes = directed_adapter.nodes

    # Build outgoing adjacency list for DFS
    adjacency: Dict[Any, List[Any]] = {}
    for u in all_nodes:
        out_edges = directed_adapter.get_outgoing_edges(u)
        adjacency[u] = [v for v, _, _, _, _ in out_edges]

    dfs_res = run_dfs_reachability(
        directed_adjacency=adjacency,
        all_nodes=all_nodes,
        origin_node=origin_node,
    )

    facility_audits: List[FacilityReachabilityAudit] = []
    if facilities:
        for fac in facilities:
            n_node = fac.get("node")
            is_reach = n_node in dfs_res.reachable_nodes if n_node is not None else False
            facility_audits.append(
                FacilityReachabilityAudit(
                    facility_name=fac.get("name", "Unknown Facility"),
                    facility_type=fac.get("type", "General"),
                    nearest_node=n_node,
                    is_reachable=is_reach,
                    lat=fac.get("lat", 0.0),
                    lon=fac.get("lon", 0.0),
                )
            )

    return ReachabilityAnalysisReport(
        origin_node=origin_node,
        total_nodes=len(all_nodes),
        reachable_nodes_count=len(dfs_res.reachable_nodes),
        unreachable_nodes_count=len(dfs_res.unreachable_nodes),
        reachability_percentage=dfs_res.reachability_percentage,
        dfs_result=dfs_res,
        facility_audits=facility_audits,
    )
