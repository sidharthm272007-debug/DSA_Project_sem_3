"""Graph builder and attribute enrichment for ResQGraph.

Processes raw OSM MultiDiGraphs, extracts realistic road speeds, calculates
unimpeded travel times, and determines the network-wide maximum speed bound
v_max to guarantee A* heuristic admissibility.
"""

import re
from typing import Dict, Any, Union, List
import networkx as nx
from config import DEFAULT_HIGHWAY_SPEEDS_KPH, DEFAULT_MAX_SPEED_KPH
from graph.graph_utils import haversine_distance, get_node_coords


def parse_speed_kph(maxspeed: Any, highway_type: Any) -> float:
    """Parse speed limit from OSM maxspeed or highway category.
    
    Handles formats such as:
      - 50
      - '50'
      - '50 mph'
      - ['40', '50']
      - 'signals', 'none', 'walk'
    """
    # 1. Try parsing explicit maxspeed
    if maxspeed is not None:
        if isinstance(maxspeed, list) and len(maxspeed) > 0:
            maxspeed = maxspeed[0]

        if isinstance(maxspeed, (int, float)):
            if maxspeed > 0:
                return float(maxspeed)

        if isinstance(maxspeed, str):
            maxspeed_str = maxspeed.strip().lower()
            # Match numeric value and optional 'mph'
            match = re.search(r"(\d+(?:\.\d+)?)", maxspeed_str)
            if match:
                val = float(match.group(1))
                if "mph" in maxspeed_str:
                    return val * 1.60934
                return val

    # 2. Fallback to highway type lookup
    h_type = highway_type
    if isinstance(h_type, list) and len(h_type) > 0:
        h_type = h_type[0]
    if isinstance(h_type, str):
        h_type = h_type.lower()
        if h_type in DEFAULT_HIGHWAY_SPEEDS_KPH:
            return DEFAULT_HIGHWAY_SPEEDS_KPH[h_type]

    # 3. Default urban residential fallback
    return DEFAULT_HIGHWAY_SPEEDS_KPH.get("residential", 30.0)


def enrich_graph_attributes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Enrich nodes and edges with validated coordinates, speeds, and travel times.
    
    Computes and stores:
      - edge['speed_kph']: Effective road speed in km/h.
      - edge['speed_mps']: Speed in meters per second.
      - edge['length']: Length in meters (computed via haversine if missing).
      - edge['travel_time']: Base unimpeded traversal time in seconds.
      - graph.graph['v_max_kph']: Maximum edge speed in the network.
      - graph.graph['v_max_mps']: Maximum edge speed in m/s for A* admissibility.
      
    Returns:
        Enriched nx.MultiDiGraph.
    """
    max_observed_speed = DEFAULT_MAX_SPEED_KPH

    # Process all edges
    for u, v, k, data in graph.edges(keys=True, data=True):
        # 1. Ensure edge length
        length = data.get("length")
        if length is None or length <= 0:
            try:
                u_lat, u_lon = get_node_coords(graph, u)
                v_lat, v_lon = get_node_coords(graph, v)
                length = haversine_distance(u_lat, u_lon, v_lat, v_lon)
                if length == 0.0:
                    length = 1.0  # minimal 1 meter to avoid 0-division
            except Exception:
                length = 1.0
        data["length"] = float(length)

        # 2. Extract and assign speed
        maxspeed_val = data.get("maxspeed")
        highway_val = data.get("highway")
        speed_kph = parse_speed_kph(maxspeed_val, highway_val)
        speed_mps = max(speed_kph * 1000.0 / 3600.0, 0.5)  # min 0.5 m/s (~1.8 km/h)

        data["speed_kph"] = speed_kph
        data["speed_mps"] = speed_mps

        if speed_kph > max_observed_speed:
            max_observed_speed = speed_kph

        # 3. Base travel time (seconds)
        data["travel_time"] = data["length"] / speed_mps

    # Guarantee A* admissibility: v_max >= every effective edge speed in graph
    v_max_kph = max_observed_speed
    v_max_mps = v_max_kph * 1000.0 / 3600.0

    graph.graph["v_max_kph"] = v_max_kph
    graph.graph["v_max_mps"] = v_max_mps

    return graph
