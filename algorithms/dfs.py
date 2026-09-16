"""Depth-First Search (DFS) for Post-Disaster Reachability Analysis.

Purpose:
  Answers the operational question:
  "Which locations remain reachable by emergency vehicles from the dispatch center
  after disaster damage?"

Algorithm Overview:
  Traverses the non-blocked directed road network using an iterative LIFO stack.
  Deeply probes branch connectivity to identify the boundary of the accessible
  network.

Complexity:
  - Time Complexity: O(V + E), where V is reachable road intersections and E is
    traversable directed road segments.
  - Space Complexity: O(V) to maintain the explicit traversal stack and visited set.

Educational Notes:
  Implemented using an explicit stack rather than Python recursion to eliminate
  RecursionError risks on deep paths in large urban road networks.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DFSReachabilityResult:
    """Encapsulates post-disaster reachability analysis from an origin."""
    origin_node: Any
    total_network_nodes: int
    reachable_nodes: Set[Any] = field(default_factory=set)
    unreachable_nodes: Set[Any] = field(default_factory=set)
    traversal_order: List[Any] = field(default_factory=list)
    discovery_tree: Dict[Any, Any] = field(default_factory=dict)  # child -> parent

    @property
    def reachability_percentage(self) -> float:
        if self.total_network_nodes == 0:
            return 0.0
        return (len(self.reachable_nodes) / self.total_network_nodes) * 100.0

    @property
    def is_fully_connected(self) -> bool:
        return len(self.unreachable_nodes) == 0


def run_dfs_reachability(
    directed_adjacency: Dict[Any, List[Any]],
    all_nodes: List[Any],
    origin_node: Any,
) -> DFSReachabilityResult:
    """Execute iterative DFS to discover all locations reachable from origin_node.
    
    Args:
        directed_adjacency: Dict mapping node -> list of active outgoing neighbor nodes.
        all_nodes: Complete list of node IDs in the network.
        origin_node: Starting intersection/dispatch point.
        
    Returns:
        DFSReachabilityResult containing reachable and unreachable node sets.
    """
    total_nodes_count = len(all_nodes)
    result = DFSReachabilityResult(
        origin_node=origin_node,
        total_network_nodes=total_nodes_count,
    )

    if origin_node not in directed_adjacency and origin_node not in all_nodes:
        result.unreachable_nodes = set(all_nodes)
        return result

    # Iterative stack traversal
    stack: List[Any] = [origin_node]
    visited: Set[Any] = {origin_node}
    result.traversal_order.append(origin_node)

    while stack:
        u = stack.pop()
        neighbors = directed_adjacency.get(u, [])
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                result.discovery_tree[v] = u
                result.traversal_order.append(v)
                stack.append(v)

    result.reachable_nodes = visited
    all_nodes_set = set(all_nodes)
    result.unreachable_nodes = all_nodes_set - visited

    return result
