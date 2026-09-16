"""A* Search Algorithm with Admissible Travel-Time Heuristic.

Purpose:
  Finds the optimal, lowest-travel-time emergency rescue route through the directed
  road network using goal-directed heuristic search to reduce explored search space.

Heuristic Design & Proof of Admissibility:
  Let u be the current intersection and goal be the target destination.
  Let d_geo(u, goal) be the Haversine great-circle distance (meters).
  Let v_max be the maximum possible vehicle speed anywhere in the network (m/s).
  
  Heuristic Definition:
    h(u) = d_geo(u, goal) / v_max
    
  Proof of Admissibility:
    1. Geodesic lower bound: The actual road network distance d_road(u, goal) satisfies
       d_road(u, goal) >= d_geo(u, goal) because roads cannot be shorter than the straight-line
       geodesic between two geographic points on Earth.
    2. Speed upper bound: By construction, every traversable edge in the network has an effective
       speed v_edge <= v_max.
    3. Therefore:
       t*(u, goal) = Sum_{edges} (length_edge / v_edge)
                   >= Sum_{edges} (length_edge / v_max)
                   = d_road(u, goal) / v_max
                   >= d_geo(u, goal) / v_max = h(u).
    4. Since h(u) <= t*(u, goal) for all u, h(u) never overestimates the true minimum travel
       time. Hence, h is strictly ADMISSIBLE and CONSISTENT (monotonic).

Complexity:
  - Time Complexity: O((V + E) log V) worst case; often significantly faster in practice due to
    goal-directed pruning.
  - Space Complexity: O(V + E) with binary heap and lazy deletion.
"""

import heapq
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from graph.graph_utils import haversine_distance, get_node_coords
from config import DEFAULT_MAX_SPEED_KPH


@dataclass
class AStarResult:
    """Encapsulates the output of an A* emergency routing query."""
    origin: Any
    destination: Any
    success: bool
    path: List[Any] = field(default_factory=list)
    edge_keys: List[int] = field(default_factory=list)
    total_travel_time_sec: float = 0.0
    total_distance_m: float = 0.0
    nodes_explored: int = 0
    edges_examined: int = 0
    failure_reason: Optional[str] = None


def run_astar(
    directed_adapter,
    origin: Any,
    destination: Any,
    v_max_mps: Optional[float] = None,
) -> AStarResult:
    """Execute A* algorithm with admissible travel-time heuristic.
    
    Args:
        directed_adapter: DirectedGraphAdapter instance.
        origin: Start node ID.
        destination: Goal node ID.
        v_max_mps: Maximum speed bound in meters/second. If None, derived from graph.
        
    Returns:
        AStarResult containing the optimal path, travel time, and search metrics.
    """
    result = AStarResult(origin=origin, destination=destination, success=False)

    base_graph = directed_adapter.base_graph
    if origin not in base_graph:
        result.failure_reason = f"Origin node '{origin}' is not present in the road network."
        return result

    if destination not in base_graph:
        result.failure_reason = f"Destination node '{destination}' is not present in the road network."
        return result

    if origin == destination:
        result.success = True
        result.path = [origin]
        result.total_travel_time_sec = 0.0
        result.total_distance_m = 0.0
        return result

    # Determine guaranteed speed upper bound (v_max in m/s)
    if v_max_mps is None:
        v_max_mps = base_graph.graph.get(
            "v_max_mps", (DEFAULT_MAX_SPEED_KPH * 1000.0 / 3600.0)
        )

    # Pre-fetch goal coordinates for fast heuristic evaluation
    goal_lat, goal_lon = directed_adapter.get_node_coords(destination)

    def heuristic(u: Any) -> float:
        """Admissible travel-time heuristic lower bound in seconds."""
        u_lat, u_lon = directed_adapter.get_node_coords(u)
        dist_m = haversine_distance(u_lat, u_lon, goal_lat, goal_lon)
        return dist_m / v_max_mps

    # Priority queue stores: (f_score, g_score, node_id)
    # where f_score = g_score + h(node)
    h_start = heuristic(origin)
    pq: List[Tuple[float, float, Any]] = [(h_start, 0.0, origin)]

    # g_score: exact travel time from origin to node
    g_score: Dict[Any, float] = {origin: 0.0}
    dist_m: Dict[Any, float] = {origin: 0.0}

    # Predecessors for path reconstruction: v -> (parent_u, edge_key, edge_time, edge_len)
    pred: Dict[Any, Tuple[Any, int, float, float]] = {}

    nodes_explored = 0
    edges_examined = 0

    while pq:
        f_val, current_g, u = heapq.heappop(pq)
        nodes_explored += 1

        # LAZY DELETION CHECK:
        # If we have already reached u with a strictly lower g_score, this entry is stale.
        if current_g > g_score[u]:
            continue

        # Goal reached with guaranteed optimal cost (due to admissible heuristic)
        if u == destination:
            break

        for v, key, edge_time, edge_length, _ in directed_adapter.get_outgoing_edges(u):
            edges_examined += 1
            tentative_g = current_g + edge_time

            if v not in g_score or tentative_g < g_score[v]:
                g_score[v] = tentative_g
                dist_m[v] = dist_m.get(u, 0.0) + edge_length
                pred[v] = (u, key, edge_time, edge_length)

                f_score = tentative_g + heuristic(v)
                heapq.heappush(pq, (f_score, tentative_g, v))

    result.nodes_explored = nodes_explored
    result.edges_examined = edges_examined

    # Path reconstruction
    if destination not in pred:
        result.failure_reason = (
            "No route is currently available between the selected origin and destination "
            "because all modeled connecting paths are blocked or disconnected."
        )
        return result

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
