"""Dijkstra's Algorithm with Lazy Deletion for Emergency Routing.

Purpose:
  Finds the optimal, lowest-travel-time emergency rescue route through the
  directed road network while avoiding blocked roads and accounting for
  disaster slowdown penalties.

Data Structure & Pattern:
  Uses Python's standard `heapq` as a binary min-heap. Because `heapq` does not
  provide a direct O(log V) decrease-key operation, this implementation employs
  the industry-standard LAZY DELETION pattern:
    - Whenever a shorter distance to an adjacent node v is found, the improved
      (new_cost, v) is pushed onto the heap.
    - When an entry (cost, u) is popped, it is checked against the best recorded
      distance: if cost > dist[u], it is recognized as stale and skipped.
    - No in-place heap mutation is performed.

Complexity:
  - Time Complexity: O((V + E) log V) on average. Each edge relaxation may push an
    entry onto the heap, resulting in at most E heap insertions.
  - Space Complexity: O(V + E) worst case, accounting for stale heap entries and
    the predecessor map.

Academic Requirements:
  Pure custom implementation without NetworkX path-finding shortcuts.
"""

import heapq
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DijkstraResult:
    """Encapsulates the output of a Dijkstra emergency routing query."""
    origin: Any
    destination: Any
    success: bool
    path: List[Any] = field(default_factory=list)  # Ordered sequence of node IDs
    edge_keys: List[int] = field(default_factory=list)  # OSM edge keys corresponding to path
    total_travel_time_sec: float = 0.0
    total_distance_m: float = 0.0
    nodes_explored: int = 0  # Number of nodes popped from priority queue
    edges_examined: int = 0
    failure_reason: Optional[str] = None


def run_dijkstra(
    directed_adapter,
    origin: Any,
    destination: Any,
) -> DijkstraResult:
    """Execute Dijkstra's algorithm to find minimum travel-time path from origin to destination.
    
    Args:
        directed_adapter: DirectedGraphAdapter instance.
        origin: Start node ID.
        destination: Goal node ID.
        
    Returns:
        DijkstraResult containing the optimal path, travel time, and metrics.
    """
    result = DijkstraResult(origin=origin, destination=destination, success=False)

    if origin not in directed_adapter.base_graph:
        result.failure_reason = f"Origin node '{origin}' is not present in the road network."
        return result

    if destination not in directed_adapter.base_graph:
        result.failure_reason = f"Destination node '{destination}' is not present in the road network."
        return result

    if origin == destination:
        result.success = True
        result.path = [origin]
        result.total_travel_time_sec = 0.0
        result.total_distance_m = 0.0
        return result

    # Priority queue stores tuples: (accumulated_travel_time, node_id)
    pq: List[Tuple[float, Any]] = [(0.0, origin)]

    # Best-known travel times and distances
    dist: Dict[Any, float] = {origin: 0.0}
    dist_m: Dict[Any, float] = {origin: 0.0}

    # Predecessors for path reconstruction: v -> (parent_u, edge_key, edge_time, edge_length)
    pred: Dict[Any, Tuple[Any, int, float, float]] = {}

    nodes_explored = 0
    edges_examined = 0

    while pq:
        current_time, u = heapq.heappop(pq)
        nodes_explored += 1

        # LAZY DELETION CHECK:
        # If this heap entry is stale (we already found a strictly better path to u), skip it.
        if current_time > dist[u]:
            continue

        # Early exit if target destination is reached with optimal cost
        if u == destination:
            break

        # Explore active outgoing edges from u (directed_adapter filters out BLOCKED roads)
        for v, key, edge_time, edge_length, _ in directed_adapter.get_outgoing_edges(u):
            edges_examined += 1
            new_time = current_time + edge_time

            if v not in dist or new_time < dist[v]:
                dist[v] = new_time
                dist_m[v] = dist_m.get(u, 0.0) + edge_length
                pred[v] = (u, key, edge_time, edge_length)
                # Push improved distance (may create duplicate stale entries for v in heap)
                heapq.heappush(pq, (new_time, v))

    result.nodes_explored = nodes_explored
    result.edges_examined = edges_examined

    # Path reconstruction
    if destination not in pred:
        result.failure_reason = (
            "No route is currently available between the selected origin and destination "
            "because all modeled connecting paths are blocked or disconnected."
        )
        return result

    # Trace back from destination to origin
    path_nodes = []
    path_keys = []
    curr = destination
    total_time = 0.0
    total_len = 0.0

    while curr != origin:
        path_nodes.append(curr)
        parent, key, e_time, e_len = pred[curr]
        path_keys.append(key)
        total_time += e_time
        total_len += e_len
        curr = parent

    path_nodes.append(origin)
    path_nodes.reverse()
    path_keys.reverse()

    result.success = True
    result.path = path_nodes
    result.edge_keys = path_keys
    result.total_travel_time_sec = total_time
    result.total_distance_m = total_len

    return result
