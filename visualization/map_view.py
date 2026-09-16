"""Folium Interactive Disaster Response Command Map.

Builds a layered geographic command-center map with semantic visual hierarchy:
  - Base road network (muted gray) with large-graph rendering safeguards
  - Disaster impact epicenter and zone
  - Blocked roads (bold red) & slowed roads (amber)
  - Emergency vehicle routes (cyan/blue)
  - Isolated regions (color-coded)
  - Critical bridges (dashed orange) & articulation points (yellow markers)
  - Critical facilities (hospitals, fire stations, police)
  - Restoration candidate options & selected MST connections
  - Persistent semantic legend and OpenStreetMap attribution
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import folium
from folium import plugins
import networkx as nx
from disaster.scenario import DisasterScenario
from disaster.road_conditions import ScenarioState, RoadCondition
from analysis.isolation import IsolationAnalysisReport
from analysis.criticality import CriticalityAnalysisReport
from restoration.restoration import RestorationPlan
from graph.graph_utils import get_edge_geometry, get_node_coords, get_graph_centroid
from config import PALETTE, MAX_EDGES_TO_RENDER_FULL_DETAIL


def create_disaster_command_map(
    base_graph: nx.MultiDiGraph,
    scenario: DisasterScenario,
    scenario_state: ScenarioState,
    isolation_report: Optional[IsolationAnalysisReport] = None,
    criticality_report: Optional[CriticalityAnalysisReport] = None,
    restoration_plan: Optional[RestorationPlan] = None,
    emergency_route: Optional[List[Any]] = None,
    facilities: Optional[List[Dict[str, Any]]] = None,
    origin_node: Optional[Any] = None,
    destination_node: Optional[Any] = None,
    zoom_start: int = 14,
) -> folium.Map:
    """Construct the comprehensive interactive Folium command-center map.
    
    Args:
        base_graph: Authoritative OSM MultiDiGraph.
        scenario: Active DisasterScenario.
        scenario_state: Shared edge conditions.
        isolation_report: Connected component and isolation analysis.
        criticality_report: Tarjan bridges and articulation points.
        restoration_plan: Kruskal MST restoration recommendations.
        emergency_route: Ordered list of node IDs for current vehicle route.
        facilities: Emergency POIs (hospitals, fire stations).
        origin_node: Dispatch start node ID.
        destination_node: Incident destination node ID.
        zoom_start: Initial map zoom level.
        
    Returns:
        Configured folium.Map instance.
    """
    # 1. Determine Map Center
    if scenario.epicenter_lat != 0.0 and scenario.epicenter_lon != 0.0:
        center_lat, center_lon = scenario.epicenter_lat, scenario.epicenter_lon
    else:
        center_lat, center_lon = get_graph_centroid(base_graph)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -------------------------------------------------------------
    # LAYER 1: Base Road Network (Large Graph Safeguard)
    # -------------------------------------------------------------
    fg_base = folium.FeatureGroup(name="Base Road Network (Muted Gray)", show=True)
    all_edges = list(base_graph.edges(keys=True, data=True))
    total_edges = len(all_edges)

    # Safeguard: if graph has massive edge count, subsample base visual layer to prevent browser freeze
    step = 1
    if total_edges > MAX_EDGES_TO_RENDER_FULL_DETAIL:
        step = max(2, total_edges // MAX_EDGES_TO_RENDER_FULL_DETAIL)

    for i in range(0, total_edges, step):
        u, v, k, data = all_edges[i]
        cond = scenario_state.get_condition(u, v, k)
        # Only render normal roads in base layer (damaged roads rendered in dedicated layers)
        if cond == RoadCondition.NORMAL:
            geom = get_edge_geometry(base_graph, u, v, k)
            folium.PolyLine(
                locations=geom,
                color=PALETTE["normal_road"],
                weight=1.5,
                opacity=0.45,
            ).add_to(fg_base)

    fg_base.add_to(m)

    # -------------------------------------------------------------
    # LAYER 2: Disaster Impact Zone
    # -------------------------------------------------------------
    fg_disaster = folium.FeatureGroup(name="Disaster Impact Zone", show=True)
    if scenario.epicenter_lat != 0.0 and scenario.epicenter_lon != 0.0:
        # Epicenter pin
        folium.Marker(
            location=[scenario.epicenter_lat, scenario.epicenter_lon],
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
            tooltip=f"Epicenter: {scenario.disaster_type} (Severity {scenario.severity}/5)",
            popup=(
                f"<b>{scenario.disaster_type} Epicenter</b><br>"
                f"Severity: {scenario.severity}/5<br>"
                f"Impact Radius: {scenario.radius_m:.0f}m<br>"
                f"Mode: {scenario.propagation_mode.title()}"
            ),
        ).add_to(fg_disaster)

        # Impact perimeter circle
        folium.Circle(
            location=[scenario.epicenter_lat, scenario.epicenter_lon],
            radius=scenario.radius_m,
            color=PALETTE["disaster_zone"],
            fill=True,
            fill_color=PALETTE["disaster_zone"],
            fill_opacity=0.18,
            weight=2,
            dash_array="5, 5",
            tooltip=f"Impact Radius: {scenario.radius_m:.0f}m",
        ).add_to(fg_disaster)

    fg_disaster.add_to(m)

    # -------------------------------------------------------------
    # LAYER 3: Damaged Roads (Blocked & Slowed)
    # -------------------------------------------------------------
    fg_blocked = folium.FeatureGroup(name="Blocked Roads (Impassable)", show=True)
    fg_slowed = folium.FeatureGroup(name="Slowed / Damaged Roads", show=True)

    for u, v, k in base_graph.edges(keys=True):
        cond = scenario_state.get_condition(u, v, k)
        if cond == RoadCondition.NORMAL:
            continue

        geom = get_edge_geometry(base_graph, u, v, k)
        edge_data = base_graph.get_edge_data(u, v, k, default={})
        name = edge_data.get("name", "Road")
        if isinstance(name, list) and name:
            name = name[0]

        if cond == RoadCondition.BLOCKED:
            folium.PolyLine(
                locations=geom,
                color=PALETTE["blocked_road"],
                weight=4.5,
                opacity=0.90,
                tooltip=f"BLOCKED: {name}",
                popup=f"<b>BLOCKED ROAD</b><br>{name}<br>Status: Impassable to emergency vehicles",
            ).add_to(fg_blocked)
        elif cond == RoadCondition.SEVERELY_AFFECTED:
            folium.PolyLine(
                locations=geom,
                color=PALETTE["severely_affected_road"],
                weight=3.5,
                opacity=0.85,
                dash_array="6, 4",
                tooltip=f"Severely Affected (4x Delay): {name}",
            ).add_to(fg_slowed)
        elif cond == RoadCondition.SLOWED:
            folium.PolyLine(
                locations=geom,
                color=PALETTE["slowed_road"],
                weight=3.0,
                opacity=0.80,
                dash_array="4, 4",
                tooltip=f"Slowed (2x Delay): {name}",
            ).add_to(fg_slowed)

    fg_blocked.add_to(m)
    fg_slowed.add_to(m)

    # -------------------------------------------------------------
    # LAYER 4: Isolated Regions
    # -------------------------------------------------------------
    fg_isolated = folium.FeatureGroup(name="Isolated Network Regions", show=True)
    if isolation_report and isolation_report.has_isolated_regions:
        region_colors = ["#8E24AA", "#00897B", "#3949AB", "#D81B60", "#F4511E"]
        for idx, reg in enumerate(isolation_report.isolated_regions):
            color = region_colors[idx % len(region_colors)]
            # Draw marker at centroid
            folium.CircleMarker(
                location=[reg.centroid_lat, reg.centroid_lon],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=f"Isolated {reg.region_id} ({reg.node_count} intersections)",
                popup=(
                    f"<b>{reg.region_id} (Isolated Pocket)</b><br>"
                    f"Size: {reg.node_count} nodes<br>"
                    f"Trapped Facilities: {len(reg.trapped_facilities)}<br>"
                    f"Status: Physically cut off from Main Access Network"
                ),
            ).add_to(fg_isolated)

            # Highlight road segments belonging entirely to this isolated region
            region_node_set = set(reg.nodes)
            for u in reg.nodes:
                if u in base_graph:
                    for v, edge_dict in base_graph[u].items():
                        if v in region_node_set:
                            for k in edge_dict:
                                geom = get_edge_geometry(base_graph, u, v, k)
                                folium.PolyLine(
                                    locations=geom,
                                    color=color,
                                    weight=2.5,
                                    opacity=0.75,
                                ).add_to(fg_isolated)

    fg_isolated.add_to(m)

    # -------------------------------------------------------------
    # LAYER 5: Critical Bridges & Articulation Points
    # -------------------------------------------------------------
    fg_critical = folium.FeatureGroup(name="Critical Bottlenecks (Bridges & Junctions)", show=False)
    if criticality_report:
        # Tarjan Bridges
        for b in criticality_report.bridges:
            folium.PolyLine(
                locations=[b.u_coords, b.v_coords],
                color=PALETTE["critical_bridge"],
                weight=5.0,
                opacity=0.95,
                dash_array="8, 5",
                tooltip=f"Tarjan Bridge: {b.road_name}",
                popup=f"<b>Structural Single Point of Failure</b><br>{b.severance_impact_description}",
            ).add_to(fg_critical)

        # Tarjan Articulation Points
        for ap in criticality_report.articulation_points:
            folium.CircleMarker(
                location=[ap.coords[0], ap.coords[1]],
                radius=6,
                color=PALETTE["articulation_point"],
                fill=True,
                fill_color="#FFEE58",
                fill_opacity=0.9,
                weight=2,
                tooltip=f"Critical Junction (Degree {ap.incident_edges_count})",
                popup=f"<b>Articulation Point</b><br>{ap.severance_impact_description}",
            ).add_to(fg_critical)

    fg_critical.add_to(m)

    # -------------------------------------------------------------
    # LAYER 6: Critical Facilities
    # -------------------------------------------------------------
    fg_facilities = folium.FeatureGroup(name="Critical Facilities (Hospitals, Fire, Police)", show=True)
    if facilities:
        for fac in facilities:
            fac_type = fac.get("type", "").lower()
            icon_name = "building"
            icon_color = "blue"

            if "hospital" in fac_type or "clinic" in fac_type:
                icon_name = "plus"
                icon_color = "red"
            elif "fire" in fac_type:
                icon_name = "fire-extinguisher"
                icon_color = "orange"
            elif "police" in fac_type:
                icon_name = "shield"
                icon_color = "darkblue"
            elif "shelter" in fac_type:
                icon_name = "home"
                icon_color = "green"

            folium.Marker(
                location=[fac["lat"], fac["lon"]],
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa"),
                tooltip=f"{fac['name']} ({fac_type.title()})",
                popup=f"<b>{fac['name']}</b><br>Category: {fac_type.title()}<br>Nearest Node: {fac.get('node')}",
            ).add_to(fg_facilities)

    fg_facilities.add_to(m)

    # -------------------------------------------------------------
    # LAYER 7: Emergency Rescue Route
    # -------------------------------------------------------------
    fg_route = folium.FeatureGroup(name="Emergency Vehicle Route", show=True)
    if emergency_route and len(emergency_route) >= 2:
        route_coords = []
        for i in range(len(emergency_route) - 1):
            u = emergency_route[i]
            v = emergency_route[i + 1]
            geom = get_edge_geometry(base_graph, u, v, 0)
            route_coords.extend(geom)

        folium.PolyLine(
            locations=route_coords,
            color=PALETTE["emergency_route"],
            weight=6.0,
            opacity=0.95,
            tooltip="Active Emergency Route",
            popup="<b>Emergency Response Route</b><br>Optimized for minimum travel time",
        ).add_to(fg_route)

    # Origin & Destination markers
    if origin_node is not None and origin_node in base_graph:
        o_lat, o_lon = get_node_coords(base_graph, origin_node)
        folium.Marker(
            location=[o_lat, o_lon],
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
            tooltip="Emergency Dispatch Origin",
            popup="<b>Emergency Dispatch Center</b>",
        ).add_to(fg_route)

    if destination_node is not None and destination_node in base_graph:
        d_lat, d_lon = get_node_coords(base_graph, destination_node)
        folium.Marker(
            location=[d_lat, d_lon],
            icon=folium.Icon(color="darkred", icon="flag-checkered", prefix="fa"),
            tooltip="Incident Destination",
            popup="<b>Emergency Incident Target</b>",
        ).add_to(fg_route)

    fg_route.add_to(m)

    # -------------------------------------------------------------
    # LAYER 8: Restoration Plan (MST Reconnections)
    # -------------------------------------------------------------
    fg_restoration = folium.FeatureGroup(name="Restoration Strategy (Kruskal MST)", show=True)
    if restoration_plan and restoration_plan.selected_connections:
        for edge in restoration_plan.selected_connections:
            u_lat, u_lon = get_node_coords(base_graph, edge.u_node)
            v_lat, v_lon = get_node_coords(base_graph, edge.v_node)

            if edge.candidate_type == "DEBRIS_CLEARANCE":
                # Existing physical road
                geom = get_edge_geometry(
                    base_graph, edge.u_node, edge.v_node, edge.osm_edge_key or 0
                )
                folium.PolyLine(
                    locations=geom,
                    color=PALETTE["selected_mst"],
                    weight=5.5,
                    opacity=0.95,
                    tooltip=f"MST Restoration: {edge.edge_id} (Debris Clearance)",
                    popup=(
                        f"<b>Recommended Restoration: Debris Clearance</b><br>"
                        f"Road: {edge.metadata.get('road_name')}<br>"
                        f"Reconnecting: {edge.u_comp} ↔ {edge.v_comp}<br>"
                        f"Length: {edge.length_m:.0f}m<br>"
                        f"Synthetic Cost: {edge.cost} units<br>"
                        f"<small><i>{edge.metadata.get('disclaimer')}</i></small>"
                    ),
                ).add_to(fg_restoration)
            else:
                # User correction #4: Clearly labeled as conceptual/hypothetical
                folium.PolyLine(
                    locations=[(u_lat, u_lon), (v_lat, v_lon)],
                    color=PALETTE["restoration_candidate_hypothetical"],
                    weight=4.0,
                    opacity=0.90,
                    dash_array="6, 6",
                    tooltip=f"MST Restoration: {edge.edge_id} (Simulated Emergency Bypass)",
                    popup=(
                        f"<b>SIMULATED CONCEPTUAL LINK (NO EXISTING ROAD)</b><br>"
                        f"Type: Temporary Emergency Bypass / Bridge<br>"
                        f"Connecting: {edge.u_comp} ↔ {edge.v_comp}<br>"
                        f"Span: {edge.length_m:.0f}m<br>"
                        f"Synthetic Cost: {edge.cost} units<br>"
                        f"<small><i>{edge.metadata.get('disclaimer')}</i></small>"
                    ),
                ).add_to(fg_restoration)

    fg_restoration.add_to(m)

    # -------------------------------------------------------------
    # Layer Control & Persistent Legend
    # -------------------------------------------------------------
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Add HTML Legend with explicit text colors (prevents dark-mode inheritance)
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 25px; 
        left: 25px; 
        width: 260px;
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 1px solid #B0BEC5;
        border-radius: 8px;
        padding: 12px 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 11px;
        line-height: 1.5;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    ">
        <div style="font-weight: 700; font-size: 12px; color: #0D47A1 !important; margin-bottom: 8px; border-bottom: 1px solid #E0E0E0; padding-bottom: 4px;">
            🚨 ResQGraph Operations Legend
        </div>
        <div style="display: flex; flex-direction: column; gap: 4px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:18px; height:4px; background-color:{PALETTE['normal_road']}; border-radius: 2px;"></span>
                <span style="color: #212121 !important; font-weight: 500;">Normal Road</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:18px; height:4px; background-color:{PALETTE['blocked_road']}; border-radius: 2px;"></span>
                <span style="color: #B71C1C !important; font-weight: 600;">Blocked Road (Impassable)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:18px; height:4px; background-color:{PALETTE['slowed_road']}; border-top: 2px dashed {PALETTE['slowed_road']};"></span>
                <span style="color: #E65100 !important; font-weight: 500;">Slowed Road (2x-4x delay)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:18px; height:5px; background-color:{PALETTE['emergency_route']}; border-radius: 2px;"></span>
                <span style="color: #01579B !important; font-weight: 600;">Emergency Vehicle Route</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{PALETTE['isolated_region']};"></span>
                <span style="color: #4A148C !important; font-weight: 500;">Isolated Region Pocket</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:18px; height:4px; background-color:{PALETTE['critical_bridge']}; border-top: 2px dashed #D84315;"></span>
                <span style="color: #D84315 !important; font-weight: 600;">Tarjan Bridge (Single Point)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background-color:{PALETTE['articulation_point']}; border: 1.5px solid #F57F17;"></span>
                <span style="color: #F57F17 !important; font-weight: 600;">Critical Articulation Junction</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:18px; height:5px; background-color:{PALETTE['selected_mst']}; border-radius: 2px;"></span>
                <span style="color: #1B5E20 !important; font-weight: 600;">MST Debris Clearance</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:18px; height:4px; background-color:{PALETTE['restoration_candidate_hypothetical']}; border-top: 2px dashed #6A1B9A;"></span>
                <span style="color: #6A1B9A !important; font-weight: 500;">Simulated Emergency Link</span>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 9px; color: #616161 !important; border-top: 1px solid #ECEFF1; padding-top: 4px;">
            Map Data © OpenStreetMap contributors
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
