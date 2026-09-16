"""Graph Adapter Layer for ResQGraph.

Provides clean algorithm-friendly adjacency representations while preserving
the underlying OSMnx MultiDiGraph structures, coordinates, and keys:
  - DirectedGraphAdapter: Directed routing graph for Dijkstra and A*.
  - UndirectedGraphAdapter: Physical structural connectivity projection for
    Union-Find and Tarjan's bridge/articulation-point algorithms.

CRITICAL ARCHITECTURE RULE:
Both adapters reference the SAME underlying base graph and the SAME shared
ScenarioState instance (u, v, key) -> RoadCondition.
"""

from typing import Dict, List, Tuple, Any, Set, Optional
import networkx as nx
from disaster.road_conditions import ScenarioState, RoadCondition


class DirectedGraphAdapter:
    """Directed routing graph adapter for Dijkstra and A* algorithms.
    
    Respects OSM one-ways, parallel edge keys, and dynamic road conditions.
    Filters out BLOCKED edges and applies travel-time multipliers for SLOWED roads.
    """

    def __init__(self, base_graph: nx.MultiDiGraph, scenario_state: ScenarioState):
        self.base_graph = base_graph
        self.scenario_state = scenario_state

    @property
    def nodes(self) -> List[Any]:
        """List of all node IDs in the network."""
        return list(self.base_graph.nodes)

    def node_count(self) -> int:
        return len(self.base_graph.nodes)

    def get_node_coords(self, node_id: Any) -> Tuple[float, float]:
        """Get (lat, lon) for node."""
        data = self.base_graph.nodes[node_id]
        return float(data.get("y", data.get("lat", 0.0))), float(data.get("x", data.get("lon", 0.0)))

    def get_outgoing_edges(self, u: Any) -> List[Tuple[Any, int, float, float, RoadCondition]]:
        """Retrieve all active outgoing edges from node u.
        
        Returns:
            List of tuples: (target_node_v, edge_key, dynamic_travel_time_sec, length_m, condition)
            Only edges with condition.is_traversable (i.e. not BLOCKED) are returned.
        """
        results = []
        if u not in self.base_graph:
            return results

        for v, edge_dict in self.base_graph[u].items():
            for key, data in edge_dict.items():
                cond = self.scenario_state.get_condition(u, v, key)
                if not cond.is_traversable:
                    continue  # Skip blocked roads

                base_time = data.get("travel_time", 1.0)
                length = data.get("length", 1.0)
                effective_time = base_time * cond.travel_time_multiplier

                results.append((v, key, effective_time, length, cond))

        return results

    def get_edge_data(self, u: Any, v: Any, key: int = 0) -> Dict[str, Any]:
        """Retrieve raw edge data dict with current condition."""
        data = dict(self.base_graph.get_edge_data(u, v, key, default={}))
        data["condition"] = self.scenario_state.get_condition(u, v, key)
        return data


class UndirectedGraphAdapter:
    """Undirected physical connectivity projection of the road network.
    
    Used by Union-Find (connected components), Tarjan's bridges, and Tarjan's
    articulation points. Represents the physical road grid regardless of legal
    traffic direction.
    
    Shares the exact same ScenarioState instance as DirectedGraphAdapter.
    """

    def __init__(self, base_graph: nx.MultiDiGraph, scenario_state: ScenarioState):
        self.base_graph = base_graph
        self.scenario_state = scenario_state

    @property
    def nodes(self) -> List[Any]:
        return list(self.base_graph.nodes)

    def node_count(self) -> int:
        return len(self.base_graph.nodes)

    def get_node_coords(self, node_id: Any) -> Tuple[float, float]:
        data = self.base_graph.nodes[node_id]
        return float(data.get("y", data.get("lat", 0.0))), float(data.get("x", data.get("lon", 0.0)))

    def get_neighbors(self, u: Any) -> Set[Any]:
        """Get all physical neighbors of node u connected by at least one non-blocked edge.
        
        Considers edges in both forward (u -> v) and reverse (v -> u) directions.
        """
        neighbors = set()
        if u not in self.base_graph:
            return neighbors

        # Outgoing edges from u
        for v, edge_dict in self.base_graph[u].items():
            for key in edge_dict:
                cond = self.scenario_state.get_condition(u, v, key)
                if cond.is_traversable:
                    neighbors.add(v)
                    break

        # Incoming edges to u (physical road connects them even if one-way)
        if hasattr(self.base_graph, "_pred") and u in self.base_graph._pred:
            for pred, edge_dict in self.base_graph._pred[u].items():
                for key in edge_dict:
                    cond = self.scenario_state.get_condition(pred, u, key)
                    if cond.is_traversable:
                        neighbors.add(pred)
                        break

        return neighbors

    def get_active_osm_edges_between(self, u: Any, v: Any) -> List[Tuple[Any, Any, int]]:
        """Return all non-blocked directed OSM edges existing between u and v in either direction.
        
        Useful for determining if an undirected edge {u, v} has parallel structural paths.
        """
        active_edges = []
        # Check u -> v
        if self.base_graph.has_edge(u, v):
            for key in self.base_graph[u][v]:
                cond = self.scenario_state.get_condition(u, v, key)
                if cond.is_traversable:
                    active_edges.append((u, v, key))

        # Check v -> u
        if self.base_graph.has_edge(v, u):
            for key in self.base_graph[v][u]:
                cond = self.scenario_state.get_condition(v, u, key)
                if cond.is_traversable:
                    active_edges.append((v, u, key))

        return active_edges

    def get_undirected_edge_list(self) -> List[Tuple[Any, Any]]:
        """Get unique unordered node pairs {u, v} connected by at least one non-blocked edge."""
        seen = set()
        edges = []
        for u in self.base_graph.nodes:
            for v in self.get_neighbors(u):
                pair = (min(str(u), str(v)), max(str(u), str(v)))
                if pair not in seen:
                    seen.add(pair)
                    edges.append((u, v))
        return edges
