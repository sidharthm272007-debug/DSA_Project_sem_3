"""Geographic and graph utility functions for ResQGraph."""

import math
from typing import Tuple, List, Optional, Any, Dict
import networkx as nx


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two geographic points in meters.
    
    Uses the Haversine formula on a spherical Earth (WGS84 mean radius R = 6,371,000 m).
    
    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.
        
    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Mean radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def get_node_coords(graph: nx.Graph, node_id: Any) -> Tuple[float, float]:
    """Retrieve (lat, lon) coordinates for a given graph node.
    
    Supports both OSMnx node conventions (y=lat, x=lon) and fallback attributes (lat, lon).
    
    Args:
        graph: NetworkX graph.
        node_id: Identifier of the node.
        
    Returns:
        Tuple of (lat, lon).
    """
    node_data = graph.nodes[node_id]
    lat = node_data.get("y", node_data.get("lat", 0.0))
    lon = node_data.get("x", node_data.get("lon", 0.0))
    return float(lat), float(lon)


def find_nearest_node(graph: nx.Graph, lat: float, lon: float) -> Optional[Any]:
    """Find the graph node geographically closest to the specified (lat, lon) coordinates.
    
    Args:
        graph: NetworkX graph with node coordinates (y/lat, x/lon).
        lat: Target latitude.
        lon: Target longitude.
        
    Returns:
        Node ID of the closest node, or None if graph is empty.
    """
    if len(graph.nodes) == 0:
        return None

    best_node = None
    min_dist = float("inf")

    for node_id, data in graph.nodes(data=True):
        n_lat = data.get("y", data.get("lat"))
        n_lon = data.get("x", data.get("lon"))
        if n_lat is None or n_lon is None:
            continue
        dist = haversine_distance(lat, lon, float(n_lat), float(n_lon))
        if dist < min_dist:
            min_dist = dist
            best_node = node_id

    return best_node


def get_edge_geometry(graph: nx.MultiDiGraph, u: Any, v: Any, key: int = 0) -> List[Tuple[float, float]]:
    """Extract ordered polyline coordinates [(lat, lon), ...] for a road edge.
    
    If the edge has a Shapely LineString geometry, extracts all intermediate vertices.
    Otherwise, returns a straight line between node u and node v.
    
    Args:
        graph: NetworkX MultiDiGraph.
        u: Source node ID.
        v: Target node ID.
        key: Edge multigraph key.
        
    Returns:
        List of (latitude, longitude) coordinate tuples.
    """
    edge_data = graph.get_edge_data(u, v, key, default={})
    geom = edge_data.get("geometry")

    if geom is not None and hasattr(geom, "coords"):
        # Shapely LineString coords are (x, y) = (lon, lat)
        return [(float(lat), float(lon)) for lon, lat in geom.coords]

    # Fallback to straight segment between nodes
    u_lat, u_lon = get_node_coords(graph, u)
    v_lat, v_lon = get_node_coords(graph, v)
    return [(u_lat, u_lon), (v_lat, v_lon)]


def get_graph_bounds(graph: nx.Graph) -> Tuple[float, float, float, float]:
    """Calculate the geographic bounding box of the graph: (min_lat, min_lon, max_lat, max_lon).
    
    Args:
        graph: NetworkX graph.
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon).
    """
    lats = []
    lons = []
    for _, data in graph.nodes(data=True):
        lat = data.get("y", data.get("lat"))
        lon = data.get("x", data.get("lon"))
        if lat is not None and lon is not None:
            lats.append(float(lat))
            lons.append(float(lon))

    if not lats:
        return 0.0, 0.0, 0.0, 0.0

    return min(lats), min(lons), max(lats), max(lons)


def get_graph_centroid(graph: nx.Graph) -> Tuple[float, float]:
    """Calculate the mean geographic center (lat, lon) of all nodes in the graph."""
    min_lat, min_lon, max_lat, max_lon = get_graph_bounds(graph)
    return (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0
