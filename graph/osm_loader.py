"""OpenStreetMap Road Network Ingestion & Local GraphML Caching.

Uses OSMnx 2.x API to acquire real-world road networks and caches them locally
as GraphML to prevent redundant network calls and respect Overpass API limits.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import networkx as nx
import osmnx as ox

logger = logging.getLogger(__name__)

# Ensure data/osm directory exists
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "osm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class OSMDownloadError(Exception):
    """Raised when OSMnx fails to retrieve or parse road network data."""
    pass


class EmptyGraphError(Exception):
    """Raised when the acquired road network has no nodes or edges."""
    pass


def _sanitize_filename(name: str) -> str:
    """Sanitize string to create safe filename for caching."""
    clean = re.sub(r"[^\w\-_.]", "_", name.strip())
    return clean[:80].lower()


def get_cached_graph_path(identifier: str) -> Path:
    """Compute local path for a cached GraphML file."""
    safe_name = _sanitize_filename(identifier)
    return CACHE_DIR / f"{safe_name}.graphml"


def load_from_cache(identifier: str) -> Optional[nx.MultiDiGraph]:
    """Load a graph from local GraphML cache if available."""
    cache_path = get_cached_graph_path(identifier)
    if cache_path.exists():
        try:
            logger.info("Loading OSM graph from cache: %s", cache_path)
            # OSMnx 2.x API: ox.io.load_graphml
            graph = ox.io.load_graphml(cache_path)
            if len(graph.nodes) == 0:
                return None
            return graph
        except Exception as e:
            logger.warning("Failed to load cached graph %s: %s. Re-downloading.", cache_path, e)
            return None
    return None


def save_to_cache(graph: nx.MultiDiGraph, identifier: str) -> bool:
    """Save graph to local GraphML cache."""
    cache_path = get_cached_graph_path(identifier)
    try:
        # OSMnx 2.x API: ox.io.save_graphml
        ox.io.save_graphml(graph, cache_path)
        logger.info("Saved OSM graph to cache: %s", cache_path)
        return True
    except Exception as e:
        logger.warning("Failed to cache graph to %s: %s", cache_path, e)
        return False


def load_osm_by_place(place_name: str, network_type: str = "drive", force_refresh: bool = False) -> nx.MultiDiGraph:
    """Load road network for a named place (city, district, or municipality).
    
    Args:
        place_name: e.g., 'Monaco', 'Oxford, UK'.
        network_type: 'drive', 'all', 'walk', etc.
        force_refresh: If True, ignores cached GraphML and re-queries Overpass.
        
    Returns:
        nx.MultiDiGraph representing the raw geographic road network.
        
    Raises:
        OSMDownloadError: If download fails or place name is invalid.
        EmptyGraphError: If returned graph has zero nodes.
    """
    cache_id = f"place_{place_name}_{network_type}"
    if not force_refresh:
        cached = load_from_cache(cache_id)
        if cached is not None:
            return cached

    try:
        # OSMnx 2.x API: ox.graph.graph_from_place
        logger.info("Downloading road network for place: '%s' (network_type: %s)", place_name, network_type)
        graph = ox.graph.graph_from_place(place_name, network_type=network_type, simplify=True)
    except Exception as e:
        raise OSMDownloadError(f"Could not load road network for '{place_name}': {str(e)}") from e

    if len(graph.nodes) == 0:
        raise EmptyGraphError(f"Retrieved road network for '{place_name}' is empty (0 nodes).")

    save_to_cache(graph, cache_id)
    return graph


def load_osm_by_point(
    center_point: Tuple[float, float],
    dist: int = 1000,
    network_type: str = "drive",
    identifier: Optional[str] = None,
    force_refresh: bool = False,
) -> nx.MultiDiGraph:
    """Load road network within a distance radius around a (lat, lon) coordinate.
    
    Args:
        center_point: (latitude, longitude) tuple.
        dist: Radius in meters.
        network_type: 'drive', 'all', etc.
        identifier: Optional cache key identifier.
        force_refresh: If True, bypasses cache.
        
    Returns:
        nx.MultiDiGraph.
    """
    lat, lon = center_point
    cache_id = identifier or f"point_{lat:.4f}_{lon:.4f}_{dist}_{network_type}"

    if not force_refresh:
        cached = load_from_cache(cache_id)
        if cached is not None:
            return cached

    try:
        # OSMnx 2.x API: ox.graph.graph_from_point
        logger.info("Downloading road network around point (%f, %f) dist=%dm", lat, lon, dist)
        graph = ox.graph.graph_from_point(center_point, dist=dist, network_type=network_type, simplify=True)
    except Exception as e:
        raise OSMDownloadError(f"Could not load road network around ({lat:.4f}, {lon:.4f}): {str(e)}") from e

    if len(graph.nodes) == 0:
        raise EmptyGraphError(f"Retrieved road network around ({lat:.4f}, {lon:.4f}) is empty.")

    save_to_cache(graph, cache_id)
    return graph


def load_osm_by_bbox(
    bbox: Tuple[float, float, float, float],
    network_type: str = "drive",
    identifier: Optional[str] = None,
    force_refresh: bool = False,
) -> nx.MultiDiGraph:
    """Load road network within a geographic bounding box.
    
    In OSMnx 2.x, bbox is (left, bottom, right, top) = (min_lon, min_lat, max_lon, max_lat).
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        network_type: 'drive', etc.
        identifier: Optional cache key identifier.
        force_refresh: If True, bypasses cache.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    cache_id = identifier or f"bbox_{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}_{network_type}"

    if not force_refresh:
        cached = load_from_cache(cache_id)
        if cached is not None:
            return cached

    try:
        # OSMnx 2.x API: ox.graph.graph_from_bbox
        logger.info("Downloading road network for bbox: %s", bbox)
        graph = ox.graph.graph_from_bbox(bbox, network_type=network_type, simplify=True)
    except Exception as e:
        raise OSMDownloadError(f"Could not load road network for bbox {bbox}: {str(e)}") from e

    if len(graph.nodes) == 0:
        raise EmptyGraphError("Retrieved road network for bounding box is empty.")

    save_to_cache(graph, cache_id)
    return graph
