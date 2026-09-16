"""Critical Facility Acquisition & OSM POI Mapping.

Retrieves hospitals, fire stations, police stations, clinics, schools, and shelters
using OSMnx 2.x features API and maps each facility to its nearest network road node.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import osmnx as ox
from graph.graph_utils import find_nearest_node

logger = logging.getLogger(__name__)

FACILITY_TAGS = {
    "amenity": [
        "hospital",
        "clinic",
        "fire_station",
        "police",
        "shelter",
        "school",
    ]
}

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "osm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_facilities_for_graph(
    graph: nx.MultiDiGraph,
    place_name: Optional[str] = None,
    point: Optional[Tuple[float, float]] = None,
    dist: int = 1200,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Retrieve emergency and public facilities within the road network area.
    
    Args:
        graph: Base OSM road network.
        place_name: Optional place name to query.
        point: Optional (lat, lon) center point.
        dist: Search radius in meters if point is specified.
        
    Returns:
        Tuple of (facility_list, user_message).
        user_message explains if data is incomplete or unavailable.
    """
    facilities: List[Dict[str, Any]] = []
    status_message: Optional[str] = None

    try:
        gdf = None
        if place_name:
            logger.info("Querying OSM facilities for '%s'...", place_name)
            # OSMnx 2.x API: ox.features.features_from_place
            gdf = ox.features.features_from_place(place_name, tags=FACILITY_TAGS)
        elif point:
            logger.info("Querying OSM facilities around point (%f, %f)...", point[0], point[1])
            # OSMnx 2.x API: ox.features.features_from_point
            gdf = ox.features.features_from_point(point, tags=FACILITY_TAGS, dist=dist)

        if gdf is not None and not gdf.empty:
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue

                # Compute centroid coordinate
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x

                # Map to closest road intersection in the graph
                nearest_node = find_nearest_node(graph, lat, lon)
                if nearest_node is None:
                    continue

                amenity_val = row.get("amenity", "facility")
                name_val = row.get("name")
                if not name_val or str(name_val).strip() == "" or str(name_val).lower() == "nan":
                    name_val = f"{str(amenity_val).replace('_', ' ').title()}"

                facilities.append(
                    {
                        "id": f"fac_{len(facilities) + 1}",
                        "name": str(name_val),
                        "type": str(amenity_val).lower(),
                        "lat": float(lat),
                        "lon": float(lon),
                        "node": nearest_node,
                    }
                )

    except Exception as e:
        logger.warning("Could not query OSM facilities via Overpass: %s", e)
        status_message = (
            "Facility query could not reach the Overpass API or timed out. "
            "Proceeding without automated OSM facility overlays."
        )

    if not facilities and status_message is None:
        status_message = (
            "No hospitals or emergency facilities were found in the selected OSM area. "
            "Note: This reflects OpenStreetMap tagging availability in this sector, "
            "not proof that no facilities exist in reality."
        )

    return facilities, status_message
