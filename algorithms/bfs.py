"""Breadth-First Search (BFS) for Disaster Propagation Simulation.

Purpose:
  Answers the operational question:
  "How far has the simulated disaster spread through the network from the epicenter?"

Algorithm Overview:
  Traverses the network outward in concentric rings (levels) using a First-In,
  First-Out (FIFO) queue. Models progressive physical contagion along connected road
  infrastructure.

Complexity:
  - Time Complexity: O(V + E), where V is the number of road nodes and E is the number
    of connecting road edges visited.
  - Space Complexity: O(V) to maintain the queue, visited set, and distance maps.

Educational Notes:
  Pure Python implementation using collections.deque without NetworkX shortcuts.
"""

from collections import deque
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BFSResult:
    """Encapsulates the output of a BFS disaster propagation traversal."""
    start_nodes: List[Any]
    visited_nodes: Set[Any] = field(default_factory=set)
    node_depths: Dict[Any, int] = field(default_factory=dict)  # Hop distance from seed
    node_distances_m: Dict[Any, float] = field(default_factory=dict)  # Geographic distance
    traversal_order: List[Any] = field(default_factory=list)
    edges_traversed: List[Tuple[Any, Any]] = field(default_factory=list)

    @property
    def max_depth_reached(self) -> int:
        return max(self.node_depths.values()) if self.node_depths else 0

    @property
    def total_nodes_affected(self) -> int:
        return len(self.visited_nodes)


def run_bfs_propagation(
    adjacency: Dict[Any, List[Tuple[Any, float]]],
    start_nodes: List[Any],
    max_depth: Optional[int] = None,
    max_distance_m: Optional[float] = None,
) -> BFSResult:
    """Execute BFS traversal over an adjacency representation.
    
    Args:
        adjacency: Dict mapping node -> list of (neighbor_node, edge_length_m).
        start_nodes: Initial nodes where disaster initiates (epicenter).
        max_depth: Optional maximum number of graph hops to propagate.
        max_distance_m: Optional maximum cumulative road distance in meters.
        
    Returns:
        BFSResult containing affected nodes, propagation levels, and metrics.
    """
    result = BFSResult(start_nodes=list(start_nodes))

    if not start_nodes:
        return result

    queue: deque = deque()

    # Initialize start nodes
    for seed in start_nodes:
        if seed in adjacency or any(seed in nbrs for nbrs in adjacency.values()):
            result.visited_nodes.add(seed)
            result.node_depths[seed] = 0
            result.node_distances_m[seed] = 0.0
            result.traversal_order.append(seed)
            queue.append(seed)

    while queue:
        u = queue.popleft()
        current_depth = result.node_depths[u]
        current_dist = result.node_distances_m[u]

        # Stop expanding if max_depth reached
        if max_depth is not None and current_depth >= max_depth:
            continue

        neighbors = adjacency.get(u, [])
        for v, edge_length in neighbors:
            new_dist = current_dist + edge_length

            # Check distance threshold
            if max_distance_m is not None and new_dist > max_distance_m:
                continue

            if v not in result.visited_nodes:
                result.visited_nodes.add(v)
                result.node_depths[v] = current_depth + 1
                result.node_distances_m[v] = new_dist
                result.traversal_order.append(v)
                result.edges_traversed.append((u, v))
                queue.append(v)

    return result
