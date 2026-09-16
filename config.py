"""ResQGraph Configuration & Constants.

Defines global simulation assumptions, road speed models, disaster severities,
synthetic restoration costs, and visualization styling.

IMPORTANT:
All restoration costs, repair times, and obstacle penalties are SYNTHETIC
simulation heuristics designed for comparative graph optimization. They do NOT
represent real-world civil engineering, financial, or governmental estimates.
"""

from typing import Dict, Any

# Fallback road speeds (in km/h) based on standard OpenStreetMap 'highway' tags.
# Used when OSM data lacks explicit 'maxspeed' attributes.
DEFAULT_HIGHWAY_SPEEDS_KPH: Dict[str, float] = {
    "motorway": 100.0,
    "motorway_link": 60.0,
    "trunk": 80.0,
    "trunk_link": 50.0,
    "primary": 60.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 35.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
    "pedestrian": 10.0,
    "track": 15.0,
    "escape": 30.0,
    "road": 30.0,
}

# Conservative default upper bound for road speeds in km/h.
# For A* heuristic admissibility, v_max is dynamically computed as:
#   v_max = max(DEFAULT_MAX_SPEED_KPH, max(effective_speeds_in_graph))
# This guarantees that straight_line_distance / v_max <= true_shortest_travel_time.
DEFAULT_MAX_SPEED_KPH: float = 120.0

# Road condition travel-time penalty multipliers.
# Documented as simulation assumptions:
#   NORMAL: 1.0x (unimpeded traffic flow)
#   SLOWED: 2.0x (minor debris, high water, rubble - travel time doubled)
#   SEVERELY_AFFECTED: 4.0x (heavy mud, structural fissures - travel time quadrupled)
#   BLOCKED: float('inf') (impassable - edge excluded from routing)
CONDITION_TIME_MULTIPLIERS: Dict[str, float] = {
    "NORMAL": 1.0,
    "SLOWED": 2.0,
    "SEVERELY_AFFECTED": 4.0,
    "BLOCKED": float("inf"),
}

# Synthetic restoration costing parameters (Simulation Heuristics).
# Not real-world construction costs; used solely for MST graph optimization.
SYNTHETIC_RESTORATION_CONFIG: Dict[str, Any] = {
    "base_debris_clearance_per_meter": 2.5,  # Units / meter
    "base_hypothetical_link_per_meter": 8.0,  # Units / meter for conceptual bridges/bypasses
    "severity_penalty_multiplier": {
        1: 1.0,
        2: 1.5,
        3: 2.2,
        4: 3.2,
        5: 5.0,
    },
    "obstacle_terrain_factors": {
        "standard": 1.0,
        "flood_water": 2.5,
        "landslide_rock": 3.0,
        "fire_hazard": 2.0,
        "structural_collapse": 4.0,
    },
    "disclaimer": (
        "SIMULATION ESTIMATE ONLY: All restoration costs, distances, obstacle penalties, "
        "and rankings are synthetic simulation heuristics for educational and graph-optimization "
        "demonstration. They do not represent real civil engineering, governmental, or financial estimates."
    ),
}

# Preset study areas for fast demonstration and offline/quick evaluation
PRESET_STUDY_AREAS: Dict[str, Dict[str, Any]] = {
    "Monaco (Full Principality)": {
        "place_query": "Monaco",
        "network_type": "drive",
        "center_lat": 43.7384,
        "center_lon": 7.4246,
        "default_zoom": 14,
        "sample_origin_label": "Port Hercules (Emergency Dispatch)",
        "sample_destination_label": "Princess Grace Hospital",
    },
    "Downtown San Francisco (Financial & Embarcadero)": {
        "point": (37.7937, -122.3965),
        "dist": 1100,  # meters radius
        "network_type": "drive",
        "center_lat": 37.7937,
        "center_lon": -122.3965,
        "default_zoom": 15,
        "sample_origin_label": "Market & 1st St (Fire Station 13)",
        "sample_destination_label": "Pier 14 Emergency Pier",
    },
    "Central Oxford, UK": {
        "point": (51.7520, -1.2577),
        "dist": 1200,
        "network_type": "drive",
        "center_lat": 51.7520,
        "center_lon": -1.2577,
        "default_zoom": 15,
        "sample_origin_label": "Oxford Town Hall",
        "sample_destination_label": "Radcliffe Infirmary Quarter",
    },
    "Vatican City & Borgo, Rome": {
        "point": (41.9029, 12.4534),
        "dist": 900,
        "network_type": "drive",
        "center_lat": 41.9029,
        "center_lon": 12.4534,
        "default_zoom": 15,
        "sample_origin_label": "St. Peter's Square Access",
        "sample_destination_label": "Santo Spirito Hospital Corridor",
    },
}

# Visualization safeguard limits
MAX_EDGES_TO_RENDER_FULL_DETAIL: int = 5000
MAX_NODES_TO_RENDER_INTERACTIVE: int = 2500

# Visual colors for map layers and dashboard
PALETTE = {
    "normal_road": "#9E9E9E",
    "slowed_road": "#FFA000",
    "severely_affected_road": "#E65100",
    "blocked_road": "#D32F2F",
    "emergency_route": "#0288D1",
    "critical_bridge": "#FF6F00",
    "articulation_point": "#FBC02D",
    "hospital": "#D81B60",
    "fire_station": "#E53935",
    "police": "#1E88E5",
    "shelter": "#43A047",
    "isolated_region": "#8E24AA",
    "restoration_candidate_clearance": "#00897B",
    "restoration_candidate_hypothetical": "#AB47BC",
    "selected_mst": "#2E7D32",
    "disaster_zone": "#FF5252",
}
