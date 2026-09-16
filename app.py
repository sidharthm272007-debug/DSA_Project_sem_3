"""ResQGraph — OpenStreetMap Disaster Response & Road Network Resilience Simulator (Phase 1).

Main Streamlit Command Center Application.
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple, List
import streamlit as st
from streamlit_folium import st_folium
import networkx as nx
import pandas as pd

# Configure page settings
st.set_page_config(
    page_title="ResQGraph | Disaster Resilience Simulator",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import PRESET_STUDY_AREAS, PALETTE, SYNTHETIC_RESTORATION_CONFIG
from graph.osm_loader import (
    load_osm_by_place,
    load_osm_by_point,
    OSMDownloadError,
    EmptyGraphError,
)
from graph.graph_builder import enrich_graph_attributes
from graph.graph_adapter import DirectedGraphAdapter, UndirectedGraphAdapter
from graph.graph_utils import (
    get_graph_bounds,
    get_graph_centroid,
    find_nearest_node,
    get_node_coords,
    haversine_distance,
)
from disaster.scenario import DisasterScenario
from disaster.simulator import DisasterImpactEngine
from disaster.road_conditions import ScenarioState, RoadCondition
from facilities.osm_facilities import load_facilities_for_graph
from algorithms.dijkstra import run_dijkstra, DijkstraResult
from algorithms.astar import run_astar, AStarResult
from analysis.reachability import analyze_reachability, ReachabilityAnalysisReport
from analysis.isolation import detect_isolated_regions, IsolationAnalysisReport
from analysis.criticality import analyze_criticality, CriticalityAnalysisReport
from restoration.restoration import build_restoration_plan, RestorationPlan
from visualization.map_view import create_disaster_command_map
from visualization.dashboard import (
    render_kpi_banner,
    render_before_after_comparison,
    render_dijkstra_astar_comparison,
    render_restoration_plan_dashboard,
    render_educational_panel,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# CACHED DATA LOADING FUNCTIONS
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading OpenStreetMap road network...")
def get_study_area_graph(area_choice: str, custom_query: str = "") -> Tuple[nx.MultiDiGraph, str]:
    """Retrieve and enrich the road network with local caching."""
    if area_choice == "Custom Search Location" and custom_query.strip():
        graph = load_osm_by_place(custom_query.strip(), network_type="drive")
        graph = enrich_graph_attributes(graph)
        return graph, custom_query.strip()

    preset = PRESET_STUDY_AREAS.get(area_choice, list(PRESET_STUDY_AREAS.values())[0])
    if "place_query" in preset:
        graph = load_osm_by_place(preset["place_query"], network_type=preset["network_type"])
    elif "point" in preset:
        graph = load_osm_by_point(
            preset["point"],
            dist=preset["dist"],
            network_type=preset["network_type"],
            identifier=area_choice,
        )
    else:
        raise ValueError("Invalid preset configuration.")

    graph = enrich_graph_attributes(graph)
    return graph, area_choice


@st.cache_data(show_spinner="Querying emergency facilities from OpenStreetMap...")
def get_cached_facilities(area_choice: str, custom_query: str = "") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Retrieve hospitals, fire stations, and clinics."""
    graph, _ = get_study_area_graph(area_choice, custom_query)
    preset = PRESET_STUDY_AREAS.get(area_choice)

    if area_choice == "Custom Search Location" and custom_query.strip():
        return load_facilities_for_graph(graph, place_name=custom_query.strip())
    elif preset and "place_query" in preset:
        return load_facilities_for_graph(graph, place_name=preset["place_query"])
    elif preset and "point" in preset:
        return load_facilities_for_graph(graph, point=preset["point"], dist=preset.get("dist", 1200))

    return load_facilities_for_graph(graph)


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
def main():
    # Header Banner
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1A237E 0%, #0D47A1 100%); padding: 18px 24px; border-radius: 8px; margin-bottom: 20px;">
            <h1 style="color: #FFFFFF; margin: 0; font-size: 26px;">
                🚨 ResQGraph — Disaster Response & Network Resilience Command Center
            </h1>
            <p style="color: #E0E0E0; margin: 4px 0 0 0; font-size: 13px;">
                Phase 1: Dynamic OpenStreetMap Simulation • Pure Custom Graph Algorithms (BFS, DFS, Dijkstra, A*, Union-Find, Tarjan, Kruskal MST)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # SIDEBAR: CONTROLS & SCENARIO CONFIGURATION
    # ------------------------------------------------------------------
    st.sidebar.markdown("## 🗺️ Study Area & Ingestion")

    area_options = list(PRESET_STUDY_AREAS.keys()) + ["Custom Search Location"]
    area_choice = st.sidebar.selectbox(
        "Select Geographic Sector",
        area_options,
        index=0,
        help="Select a pre-cached study area or input a custom municipality/city.",
    )

    custom_query = ""
    if area_choice == "Custom Search Location":
        custom_query = st.sidebar.text_input(
            "Enter City / District Name",
            value="Cambridge, UK",
            help="Queries Overpass API via OSMnx 2.x.",
        )

    # Load Road Network with Graceful Error Boundary
    try:
        base_graph, active_area_name = get_study_area_graph(area_choice, custom_query)
        total_nodes = len(base_graph.nodes)
        total_edges = len(base_graph.edges)
        facilities, facility_msg = get_cached_facilities(area_choice, custom_query)
    except (OSMDownloadError, EmptyGraphError) as e:
        st.error(f"❌ Failed to load OpenStreetMap network for '{area_choice}': {str(e)}")
        st.info("Tip: Please check spelling, internet connectivity, or choose one of the bundled preset sectors.")
        return
    except Exception as e:
        st.error(f"❌ Unexpected error loading study area: {str(e)}")
        return

    min_lat, min_lon, max_lat, max_lon = get_graph_bounds(base_graph)
    c_lat, c_lon = get_graph_centroid(base_graph)

    st.sidebar.caption(
        f"**Graph Size:** {total_nodes:,} nodes | {total_edges:,} road links  \n"
        f"**Speed Upper Bound (v_max):** {base_graph.graph.get('v_max_kph', 120):.0f} km/h (admissible for A*)"
    )

    # Disaster Scenario Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💥 Disaster Scenario Simulation")

    disaster_type = st.sidebar.selectbox(
        "Disaster Incident Type",
        ["Flood", "Earthquake", "Landslide", "Urban Fire", "Storm / Cyclone", "Custom Road Blockage"],
        index=0,
    )

    severity = st.sidebar.slider(
        "Disaster Severity Rating",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Minor localized event; 5 = Extreme widespread collapse.",
    )

    radius_m = st.sidebar.slider(
        "Simulated Impact Radius (meters)",
        min_value=100,
        max_value=2500,
        value=650,
        step=50,
    )

    propagation_mode = st.sidebar.radio(
        "Propagation Modeling Mode",
        options=["geographic", "bfs_topological", "hybrid"],
        format_func=lambda m: {
            "geographic": "Geographic Proximity (Distance Decay)",
            "bfs_topological": "BFS Network Propagation (Road Cascade)",
            "hybrid": "Hybrid (Max Severity Composite)",
        }[m],
        index=0,
        help="Maintains strict separation between proximity-based damage and topological BFS spread.",
    )

    # Epicenter selection: sliders centered on the study area
    with st.sidebar.expander("📍 Adjust Epicenter Coordinates", expanded=False):
        epi_lat = st.slider(
            "Epicenter Latitude",
            min_value=float(min_lat),
            max_value=float(max_lat),
            value=float(c_lat),
            step=0.0005,
            format="%.4f",
        )
        epi_lon = st.slider(
            "Epicenter Longitude",
            min_value=float(min_lon),
            max_value=float(max_lon),
            value=float(c_lon),
            step=0.0005,
            format="%.4f",
        )

    # Build Scenario Object
    scenario = DisasterScenario(
        disaster_type=disaster_type,
        epicenter_lat=epi_lat,
        epicenter_lon=epi_lon,
        radius_m=float(radius_m),
        severity=severity,
        propagation_mode=propagation_mode,
    )

    # ------------------------------------------------------------------
    # DISASTER IMPACT ENGINE EXECUTION
    # ------------------------------------------------------------------
    # Evaluates the scenario and builds shared ScenarioState
    scenario_state: ScenarioState = DisasterImpactEngine.apply_scenario(base_graph, scenario)

    # Instantiate Adapters (Both share the exact same scenario_state reference!)
    directed_adapter = DirectedGraphAdapter(base_graph, scenario_state)
    undirected_adapter = UndirectedGraphAdapter(base_graph, scenario_state)

    # Emergency Dispatch Origin & Destination Selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🚑 Emergency Dispatch Routing")

    all_node_list = list(base_graph.nodes)
    if not all_node_list:
        st.warning("Empty road network. No nodes available.")
        return

    # Select representative origin and destination
    default_origin = all_node_list[0]
    default_dest = all_node_list[-1] if len(all_node_list) > 1 else all_node_list[0]

    # Try mapping preset landmarks or facilities to origin/destination
    if facilities and len(facilities) >= 2:
        default_origin = facilities[0]["node"]
        default_dest = facilities[-1]["node"]

    origin_node = st.sidebar.selectbox(
        "Emergency Dispatch Origin",
        options=all_node_list,
        index=all_node_list.index(default_origin) if default_origin in all_node_list else 0,
        format_func=lambda n: f"Node {n} (Lat {get_node_coords(base_graph, n)[0]:.4f}, Lon {get_node_coords(base_graph, n)[1]:.4f})",
    )

    dest_node = st.sidebar.selectbox(
        "Incident Destination",
        options=all_node_list,
        index=all_node_list.index(default_dest) if default_dest in all_node_list else min(1, len(all_node_list) - 1),
        format_func=lambda n: f"Node {n} (Lat {get_node_coords(base_graph, n)[0]:.4f}, Lon {get_node_coords(base_graph, n)[1]:.4f})",
    )

    # ------------------------------------------------------------------
    # ALGORITHM EXECUTION & RESILIENCE ANALYSIS
    # ------------------------------------------------------------------
    # 1. Routing: Dijkstra vs A*
    t0_dijkstra = time.perf_counter()
    dijkstra_res: DijkstraResult = run_dijkstra(directed_adapter, origin_node, dest_node)
    t_dijkstra_ms = (time.perf_counter() - t0_dijkstra) * 1000.0

    t0_astar = time.perf_counter()
    astar_res: AStarResult = run_astar(directed_adapter, origin_node, dest_node)
    t_astar_ms = (time.perf_counter() - t0_astar) * 1000.0

    # 2. Reachability Analysis via DFS
    reachability_report: ReachabilityAnalysisReport = analyze_reachability(
        directed_adapter=directed_adapter,
        origin_node=origin_node,
        facilities=facilities,
    )

    # 3. Component & Isolation Analysis via Union-Find
    isolation_report: IsolationAnalysisReport = detect_isolated_regions(
        undirected_adapter=undirected_adapter,
        reference_origin_node=origin_node,
        facilities=facilities,
    )

    # 4. Criticality Analysis via Tarjan's Low-Link
    criticality_report: CriticalityAnalysisReport = analyze_criticality(undirected_adapter)

    # 5. Restoration Planning via Kruskal MST on Contracted Component Graph
    restoration_plan: RestorationPlan = build_restoration_plan(
        base_graph=base_graph,
        scenario_state=scenario_state,
        isolation_report=isolation_report,
        disaster_severity=severity,
    )

    # ------------------------------------------------------------------
    # MAIN DASHBOARD DISPLAY
    # ------------------------------------------------------------------
    # Top KPI Banner
    render_kpi_banner(
        total_nodes=total_nodes,
        total_edges=total_edges,
        scenario_state=scenario_state,
        isolation_report=isolation_report,
        reachability_report=reachability_report,
        dijkstra_res=dijkstra_res,
    )

    if facility_msg:
        st.info(f"ℹ️ {facility_msg}")

    # Main Interactive Command Map
    st.markdown("### 🗺️ Real-Time Disaster Operations Map")
    st.caption(
        "Interactive Folium command map. Toggle layers (top-right) to view Base Roads, Blockages, "
        "Emergency Routes, Isolated Regions, Bridges, and Recommended Restoration Links."
    )

    command_map = create_disaster_command_map(
        base_graph=base_graph,
        scenario=scenario,
        scenario_state=scenario_state,
        isolation_report=isolation_report,
        criticality_report=criticality_report,
        restoration_plan=restoration_plan,
        emergency_route=dijkstra_res.path if dijkstra_res.success else None,
        facilities=facilities,
        origin_node=origin_node,
        destination_node=dest_node,
        zoom_start=PRESET_STUDY_AREAS.get(area_choice, {}).get("default_zoom", 14),
    )

    st_folium(command_map, width="100%", height=560, returned_objects=[])

    # ------------------------------------------------------------------
    # TABBED OPERATIONAL & ACADEMIC ANALYSIS
    # ------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Damage Impact & Before/After",
            "🚑 Emergency Routing & Benchmark",
            "🏝️ Isolated Regions & Reachability",
            "⚠️ Structural Bottlenecks (Tarjan)",
            "🛠️ Kruskal MST Restoration",
            "📚 Graph Algorithm Guide",
        ]
    )

    with tab1:
        render_before_after_comparison(
            total_nodes=total_nodes,
            total_edges=total_edges,
            scenario_state=scenario_state,
            isolation_report=isolation_report,
            reachability_report=reachability_report,
            baseline_time_sec=dijkstra_res.total_travel_time_sec if dijkstra_res.success else None,
            post_time_sec=dijkstra_res.total_travel_time_sec if dijkstra_res.success else None,
        )

    with tab2:
        render_dijkstra_astar_comparison(
            dijkstra_res=dijkstra_res,
            astar_res=astar_res,
            dijkstra_runtime_ms=t_dijkstra_ms,
            astar_runtime_ms=t_astar_ms,
        )

    with tab3:
        st.markdown("### 🏝️ Isolated Network Regions & Stranded Facilities")
        st.write(
            f"**Total Network Components:** {isolation_report.total_components}  \n"
            f"**Main Component Size:** {isolation_report.main_component_size:,} intersections  \n"
            f"**Isolated Regions:** {isolation_report.isolated_regions_count} disconnected pockets "
            f"({isolation_report.total_isolated_nodes:,} intersections cut off from emergency access)"
        )

        if isolation_report.has_isolated_regions:
            iso_rows = []
            for reg in isolation_report.isolated_regions:
                fac_names = [f["name"] for f in reg.trapped_facilities]
                iso_rows.append(
                    {
                        "Region ID": reg.region_id,
                        "Intersections": reg.node_count,
                        "Trapped Facilities": ", ".join(fac_names) if fac_names else "None tagged",
                        "Centroid": f"({reg.centroid_lat:.4f}, {reg.centroid_lon:.4f})",
                        "Status": "Completely Severed from Main Access",
                    }
                )
            st.table(pd.DataFrame(iso_rows))
        else:
            st.success("✅ No isolated network pockets detected. All reachable nodes remain connected to the main grid.")

        st.markdown("#### Facility Reachability Audit")
        if reachability_report.facility_audits:
            fac_rows = [
                {
                    "Facility Name": fa.facility_name,
                    "Type": fa.facility_type.title(),
                    "Status": "✅ Reachable" if fa.is_reachable else "❌ Cut Off / Trapped",
                    "Nearest Intersection Node": fa.nearest_node,
                }
                for fa in reachability_report.facility_audits
            ]
            st.table(pd.DataFrame(fac_rows))
        else:
            st.caption("No critical facilities mapped in this study sector.")

    with tab4:
        st.markdown("### ⚠️ Structural Resilience Analysis (Tarjan's Low-Link)")
        st.caption(criticality_report.methodology_note)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Critical Structural Bridges", f"{criticality_report.bridge_count} roads")
        with col_b:
            st.metric("Critical Articulation Junctions", f"{criticality_report.articulation_point_count} intersections")

        if criticality_report.bridges:
            st.markdown("#### Identified Single-Point-of-Failure Roads (Bridges)")
            b_rows = [
                {
                    "Road Name": b.road_name,
                    "Length": f"{b.length_m:.0f} m",
                    "Nodes": f"{b.u} ↔ {b.v}",
                    "Vulnerability": "Severe: Severing this road increases disconnected components",
                }
                for b in criticality_report.bridges[:20]
            ]
            st.table(pd.DataFrame(b_rows))

        if criticality_report.articulation_points:
            st.markdown("#### Critical Intersection Nodes (Articulation Points)")
            st.write(
                f"Identified **{len(criticality_report.articulation_points)}** critical intersection junctions. "
                "Failure of any of these junctions will partition adjacent road corridors."
            )

    with tab5:
        render_restoration_plan_dashboard(restoration_plan)

    with tab6:
        render_educational_panel()


if __name__ == "__main__":
    main()
