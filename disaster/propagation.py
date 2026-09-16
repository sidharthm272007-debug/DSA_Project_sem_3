"""Topological Disaster Propagation using Breadth-First Search (BFS).

Simulates network-constrained disaster spreading (such as urban fire spread or
cascading infrastructure failure) where damage follows connected physical street segments.
"""

from typing import Dict, List, Tuple, Any, Set
import networkx as nx
from algorithms.bfs import run_bfs_propagation
from disaster.road_conditions import RoadCondition


def compute_bfs_propagation_conditions(
    graph: nx.MultiDiGraph,
    epicenter_nodes: List[Any],
    max_hops: int = 4,
    severity: int = 3,
) -> Dict[Tuple[Any, Any, int], RoadCondition]:
    """Simulate disaster propagation through the network using BFS.
    
    Damage decays with hop distance from the epicenter:
      - Hops <= 1 (or 2 for severity 5): BLOCKED
      - Intermediate hops: SEVERELY_AFFECTED or SLOWED
      - Peripheral hops: SLOWED
      
    Args:
        graph: Base OSM network.
        epicenter_nodes: Seed nodes at the epicenter.
        max_hops: Maximum BFS exploration depth.
        severity: 1 (minor) to 5 (extreme).
        
    Returns:
        Dict mapping (u, v, key) -> RoadCondition.
    """
    # Build undirected adjacency for propagation (damage spreads along roads regardless of one-way)
    adjacency: Dict[Any, List[Tuple[Any, float]]] = {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        length = data.get("length", 10.0)
        adjacency.setdefault(u, []).append((v, length))
        adjacency.setdefault(v, []).append((u, length))

    bfs_res = run_bfs_propagation(
        adjacency=adjacency,
        start_nodes=epicenter_nodes,
        max_depth=max_hops,
    )

    edge_conditions: Dict[Tuple[Any, Any, int], RoadCondition] = {}

    # Define thresholds based on severity
    blocked_hop_cutoff = 1 if severity <= 3 else 2
    severely_hop_cutoff = blocked_hop_cutoff + (1 if severity >= 3 else 0)

    for u, v, k in graph.edges(keys=True):
        u_depth = bfs_res.node_depths.get(u)
        v_depth = bfs_res.node_depths.get(v)

        # If neither node was reached by BFS, road is untouched
        if u_depth is None and v_depth is None:
            continue

        min_depth = min(
            u_depth if u_depth is not None else 999,
            v_depth if v_depth is not None else 999,
        )

        if min_depth <= blocked_hop_cutoff:
            cond = RoadCondition.BLOCKED
        elif min_depth <= severely_hop_cutoff:
            cond = RoadCondition.SEVERELY_AFFECTED
        elif min_depth <= max_hops:
            cond = RoadCondition.SLOWED
        else:
            cond = RoadCondition.NORMAL

        if cond != RoadCondition.NORMAL:
            edge_conditions[(u, v, k)] = cond

    return edge_conditions
