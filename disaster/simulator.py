"""Disaster Impact Engine for ResQGraph.

Translates a DisasterScenario into dynamic edge condition changes (ScenarioState).
Maintains strict separation between geographic proximity impact and BFS topological
propagation, resolving hybrid scenarios via a deterministic severity hierarchy:
  NORMAL < SLOWED < SEVERELY_AFFECTED < BLOCKED.
"""

from typing import Dict, Tuple, Any, List, Set
import networkx as nx
from disaster.scenario import DisasterScenario
from disaster.road_conditions import RoadCondition, ScenarioState
from disaster.propagation import compute_bfs_propagation_conditions
from graph.graph_utils import haversine_distance, get_node_coords, find_nearest_node

# Deterministic hierarchy for condition composition
SEVERITY_ORDER = {
    RoadCondition.NORMAL: 0,
    RoadCondition.SLOWED: 1,
    RoadCondition.SEVERELY_AFFECTED: 2,
    RoadCondition.BLOCKED: 3,
}


def _max_condition(c1: RoadCondition, c2: RoadCondition) -> RoadCondition:
    """Return the more severe condition between c1 and c2."""
    return c1 if SEVERITY_ORDER[c1] >= SEVERITY_ORDER[c2] else c2


class DisasterImpactEngine:
    """Engine responsible for computing damage states across the road network."""

    @staticmethod
    def compute_geographic_impact(
        graph: nx.MultiDiGraph,
        scenario: DisasterScenario,
    ) -> Dict[Tuple[Any, Any, int], RoadCondition]:
        """Compute damage based purely on geographic distance from the disaster epicenter.
        
        Zones:
          - Inner zone (dist <= 0.40 * radius): BLOCKED
          - Middle zone (0.40 * radius < dist <= 0.75 * radius): SEVERELY_AFFECTED
          - Outer zone (0.75 * radius < dist <= radius): SLOWED
        """
        edge_conditions: Dict[Tuple[Any, Any, int], RoadCondition] = {}
        epi_lat, epi_lon = scenario.epicenter
        radius = scenario.radius_m

        if radius <= 0:
            return edge_conditions

        # Severity scaling: higher severity widens the inner blocked zone
        # severity 1: inner = 0.20 * r; severity 5: inner = 0.60 * r
        inner_ratio = 0.10 + (scenario.severity * 0.10)
        mid_ratio = min(inner_ratio + 0.35, 0.90)

        inner_threshold = radius * inner_ratio
        mid_threshold = radius * mid_ratio

        for u, v, k in graph.edges(keys=True):
            u_lat, u_lon = get_node_coords(graph, u)
            v_lat, v_lon = get_node_coords(graph, v)

            # Measure distance from epicenter to both endpoints and the midpoint
            d_u = haversine_distance(epi_lat, epi_lon, u_lat, u_lon)
            d_v = haversine_distance(epi_lat, epi_lon, v_lat, v_lon)
            d_mid = haversine_distance(
                epi_lat, epi_lon, (u_lat + v_lat) / 2.0, (u_lon + v_lon) / 2.0
            )
            min_dist = min(d_u, d_v, d_mid)

            if min_dist > radius:
                continue

            if min_dist <= inner_threshold:
                edge_conditions[(u, v, k)] = RoadCondition.BLOCKED
            elif min_dist <= mid_threshold:
                edge_conditions[(u, v, k)] = RoadCondition.SEVERELY_AFFECTED
            else:
                edge_conditions[(u, v, k)] = RoadCondition.SLOWED

        return edge_conditions

    @classmethod
    def apply_scenario(
        cls,
        graph: nx.MultiDiGraph,
        scenario: DisasterScenario,
    ) -> ScenarioState:
        """Evaluate disaster scenario and return a shared ScenarioState instance.
        
        Determines road conditions via the selected mode:
          - 'geographic': Proximity decay from epicenter.
          - 'bfs_topological': Network cascade along road links from epicenter.
          - 'hybrid': Deterministic combination (max severity between geographic and BFS).
        
        Manual blocked edges are applied as strict top-priority overrides.
        """
        combined_conditions: Dict[Tuple[Any, Any, int], RoadCondition] = {}

        if scenario.propagation_mode == "geographic":
            combined_conditions = cls.compute_geographic_impact(graph, scenario)

        elif scenario.propagation_mode == "bfs_topological":
            epi_node = find_nearest_node(graph, scenario.epicenter_lat, scenario.epicenter_lon)
            if epi_node is not None:
                max_hops = 2 + scenario.severity
                combined_conditions = compute_bfs_propagation_conditions(
                    graph=graph,
                    epicenter_nodes=[epi_node],
                    max_hops=max_hops,
                    severity=scenario.severity,
                )

        elif scenario.propagation_mode == "hybrid":
            geo_conds = cls.compute_geographic_impact(graph, scenario)

            epi_node = find_nearest_node(graph, scenario.epicenter_lat, scenario.epicenter_lon)
            bfs_conds = {}
            if epi_node is not None:
                max_hops = 2 + scenario.severity
                bfs_conds = compute_bfs_propagation_conditions(
                    graph=graph,
                    epicenter_nodes=[epi_node],
                    max_hops=max_hops,
                    severity=scenario.severity,
                )

            # Deterministic combination: condition = max(geo_cond, bfs_cond)
            all_edges = set(geo_conds.keys()) | set(bfs_conds.keys())
            for edge in all_edges:
                c1 = geo_conds.get(edge, RoadCondition.NORMAL)
                c2 = bfs_conds.get(edge, RoadCondition.NORMAL)
                combined_conditions[edge] = _max_condition(c1, c2)

        # Apply manual blocked overrides with highest priority
        for edge in scenario.manual_blocked_edges:
            combined_conditions[edge] = RoadCondition.BLOCKED

        return ScenarioState(combined_conditions)
