"""Streamlit Command Center Dashboard Components for ResQGraph.

Renders high-level operational KPI cards, before/after network comparisons,
Dijkstra vs A* benchmarking tables, isolation breakdowns, Kruskal restoration
schedules, and educational DSA algorithm cards.
"""

from typing import Dict, List, Any, Optional
import streamlit as st
import pandas as pd
from disaster.road_conditions import ScenarioState, RoadCondition
from algorithms.dijkstra import DijkstraResult
from algorithms.astar import AStarResult
from analysis.isolation import IsolationAnalysisReport
from analysis.criticality import CriticalityAnalysisReport
from analysis.reachability import ReachabilityAnalysisReport
from restoration.restoration import RestorationPlan


def render_kpi_banner(
    total_nodes: int,
    total_edges: int,
    scenario_state: ScenarioState,
    isolation_report: Optional[IsolationAnalysisReport] = None,
    reachability_report: Optional[ReachabilityAnalysisReport] = None,
    dijkstra_res: Optional[DijkstraResult] = None,
):
    """Render top-level command-center status cards."""
    counts = scenario_state.count_by_condition()
    blocked_count = counts.get(RoadCondition.BLOCKED, 0)
    slowed_count = (
        counts.get(RoadCondition.SLOWED, 0) + counts.get(RoadCondition.SEVERELY_AFFECTED, 0)
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Infrastructure Status",
            value="CRITICAL" if blocked_count > 0 else "NORMAL",
            delta=f"-{blocked_count} blocked" if blocked_count > 0 else "100% open",
            delta_color="inverse",
        )

    with col2:
        st.metric(
            label="Damaged Roads",
            value=f"{blocked_count} Blocked",
            delta=f"+{slowed_count} slowed",
            delta_color="off",
        )

    with col3:
        iso_count = (
            isolation_report.isolated_regions_count if isolation_report else 0
        )
        st.metric(
            label="Isolated Regions",
            value=f"{iso_count} Pockets",
            delta=f"{isolation_report.total_isolated_nodes if isolation_report else 0} cut off",
            delta_color="inverse" if iso_count > 0 else "normal",
        )

    with col4:
        reach_pct = (
            reachability_report.reachability_percentage
            if reachability_report
            else 100.0
        )
        st.metric(
            label="Dispatch Reachability",
            value=f"{reach_pct:.1f}%",
            delta=f"{reachability_report.reachable_nodes_count if reachability_report else total_nodes} reachable",
        )

    with col5:
        if dijkstra_res and dijkstra_res.success:
            time_min = dijkstra_res.total_travel_time_sec / 60.0
            st.metric(
                label="Emergency ETA",
                value=f"{time_min:.1f} min",
                delta=f"{dijkstra_res.total_distance_m / 1000.0:.2f} km",
            )
        else:
            st.metric(
                label="Emergency Route",
                value="UNAVAILABLE" if blocked_count > 0 else "Select O/D",
                delta="Path Severed" if blocked_count > 0 else "Ready",
                delta_color="inverse" if blocked_count > 0 else "off",
            )


def render_before_after_comparison(
    total_nodes: int,
    total_edges: int,
    scenario_state: ScenarioState,
    isolation_report: Optional[IsolationAnalysisReport],
    reachability_report: Optional[ReachabilityAnalysisReport],
    baseline_time_sec: Optional[float] = None,
    post_time_sec: Optional[float] = None,
):
    """Render before vs after disaster impact comparison."""
    st.markdown("### 📊 Before vs. After Disaster Network Impact Analysis")

    counts = scenario_state.count_by_condition()
    blocked = counts.get(RoadCondition.BLOCKED, 0)
    slowed = counts.get(RoadCondition.SLOWED, 0) + counts.get(RoadCondition.SEVERELY_AFFECTED, 0)

    pre_components = 1  # Standard baseline urban assumption
    post_components = isolation_report.total_components if isolation_report else 1

    pre_reach_pct = 100.0
    post_reach_pct = reachability_report.reachability_percentage if reachability_report else 100.0

    data = [
        {
            "Metric": "Operational Network Intersections",
            "Pre-Disaster Baseline": f"{total_nodes:,}",
            "Post-Disaster State": f"{total_nodes - (isolation_report.total_isolated_nodes if isolation_report else 0):,}",
            "Change Impact": f"-{isolation_report.total_isolated_nodes if isolation_report else 0} isolated",
        },
        {
            "Metric": "Traversable Road Segments",
            "Pre-Disaster Baseline": f"{total_edges:,}",
            "Post-Disaster State": f"{total_edges - blocked:,}",
            "Change Impact": f"-{blocked} blocked ({slowed} slowed)",
        },
        {
            "Metric": "Connected Network Components",
            "Pre-Disaster Baseline": "1 (Unified Grid)",
            "Post-Disaster State": f"{post_components} Disjoint Pieces",
            "Change Impact": f"+{post_components - 1} fragmentations",
        },
        {
            "Metric": "Emergency Dispatch Reachability",
            "Pre-Disaster Baseline": "100.0%",
            "Post-Disaster State": f"{post_reach_pct:.1f}%",
            "Change Impact": f"-{100.0 - post_reach_pct:.1f}% accessible",
        },
    ]

    if baseline_time_sec is not None and post_time_sec is not None:
        data.append(
            {
                "Metric": "Primary Route Travel Time",
                "Pre-Disaster Baseline": f"{baseline_time_sec / 60.0:.1f} min",
                "Post-Disaster State": f"{post_time_sec / 60.0:.1f} min",
                "Change Impact": f"+{(post_time_sec - baseline_time_sec) / 60.0:.1f} min delay",
            }
        )

    df = pd.DataFrame(data)
    st.table(df)


def render_dijkstra_astar_comparison(
    dijkstra_res: DijkstraResult,
    astar_res: AStarResult,
    dijkstra_runtime_ms: float,
    astar_runtime_ms: float,
):
    """Render side-by-side benchmark comparison between Dijkstra and A*."""
    st.markdown("### 🚑 Emergency Routing: Dijkstra vs. A* Search Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔵 Dijkstra's Algorithm (Cost-Uniform)")
        st.markdown(
            "> Explores nodes strictly in order of known travel time from the origin. "
            "Guarantees the global shortest travel-time path."
        )
        st.write(f"**Route Status:** {'✅ Optimal Path Found' if dijkstra_res.success else '❌ No Route Available'}")
        if dijkstra_res.success:
            st.write(f"**Travel Time:** {dijkstra_res.total_travel_time_sec:.1f}s ({dijkstra_res.total_travel_time_sec / 60.0:.2f} min)")
            st.write(f"**Distance:** {dijkstra_res.total_distance_m / 1000.0:.2f} km ({dijkstra_res.total_distance_m:.0f} m)")
            st.write(f"**Nodes Explored:** {dijkstra_res.nodes_explored:,}")
            st.write(f"**Edges Examined:** {dijkstra_res.edges_examined:,}")
            st.write(f"**Execution Runtime:** {dijkstra_runtime_ms:.2f} ms")
        else:
            st.warning(dijkstra_res.failure_reason)

    with col2:
        st.markdown("#### 🟢 A* Search (Goal-Directed Heuristic)")
        st.markdown(
            "> Augments known cost with an admissible lower-bound heuristic: "
            "$$h(u) = \\frac{\\text{straight-line distance}}{v_{\\max}}$$. "
            "Prunes exploration towards the destination."
        )
        st.write(f"**Route Status:** {'✅ Optimal Path Found' if astar_res.success else '❌ No Route Available'}")
        if astar_res.success:
            st.write(f"**Travel Time:** {astar_res.total_travel_time_sec:.1f}s ({astar_res.total_travel_time_sec / 60.0:.2f} min)")
            st.write(f"**Distance:** {astar_res.total_distance_m / 1000.0:.2f} km ({astar_res.total_distance_m:.0f} m)")
            st.write(f"**Nodes Explored:** {astar_res.nodes_explored:,}")
            st.write(f"**Edges Examined:** {astar_res.edges_examined:,}")
            st.write(f"**Execution Runtime:** {astar_runtime_ms:.2f} ms")
        else:
            st.warning(astar_res.failure_reason)

    if dijkstra_res.success and astar_res.success:
        saved_nodes = dijkstra_res.nodes_explored - astar_res.nodes_explored
        pct_saved = (saved_nodes / dijkstra_res.nodes_explored) * 100.0 if dijkstra_res.nodes_explored > 0 else 0.0
        st.info(
            f"🎯 **Heuristic Efficiency:** A* examined **{saved_nodes} fewer nodes** "
            f"({pct_saved:.1f}% reduction in search space) while finding the mathematically identical "
            f"optimal travel-time ({astar_res.total_travel_time_sec:.1f}s)."
        )


def render_restoration_plan_dashboard(restoration_plan: RestorationPlan):
    """Render the Kruskal MST restoration schedule and synthetic cost breakdown."""
    st.markdown("### 🛠️ Recommended Infrastructure Restoration Plan (Kruskal MST)")

    # Prominent synthetic disclaimer
    st.warning(
        f"⚠️ **{restoration_plan.disclaimer}**",
        icon="⚠️",
    )

    if not restoration_plan.selected_connections:
        st.success("No isolated regions detected. The network remains structurally connected!")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Simulated Cost", f"{restoration_plan.total_synthetic_cost:,.1f} units")
    with col2:
        st.metric("Total Reconnection Span", f"{restoration_plan.total_distance_m:,.0f} m")
    with col3:
        st.metric("Debris Clearances", f"{restoration_plan.debris_clearance_count} roads")
    with col4:
        st.metric("Conceptual Links", f"{restoration_plan.hypothetical_links_count} bypasses")

    st.markdown("#### Prioritized Restoration Action Schedule")
    rows = []
    for idx, edge in enumerate(restoration_plan.selected_connections, 1):
        is_hypo = edge.metadata.get("is_hypothetical", False)
        rows.append(
            {
                "Priority": f"#{idx}",
                "Action ID": edge.edge_id,
                "Type": "Clear Existing Debris" if not is_hypo else "Simulated Conceptual Bypass",
                "Reconnecting": f"{edge.u_comp} ↔ {edge.v_comp}",
                "Length": f"{edge.length_m:.0f} m",
                "Simulated Cost": f"{edge.cost:,.1f}",
                "Description": edge.metadata.get("description", ""),
            }
        )
    st.table(pd.DataFrame(rows))


def render_educational_panel():
    """Render educational DSA guide with complexities and design rationale."""
    st.markdown("### 📚 Graph Algorithms & Data Structures in ResQGraph")
    st.markdown(
        "ResQGraph is built on custom, mathematically rigorous implementations of core "
        "computer science graph algorithms without black-box library wrappers."
    )

    algo_data = [
        {
            "Algorithm": "Breadth-First Search (BFS)",
            "Operational Purpose": "Models progressive physical disaster propagation outward from epicenter.",
            "Graph Representation": "Directed / Undirected",
            "Time Complexity": "O(V + E)",
            "Space Complexity": "O(V)",
            "Data Structure Used": "collections.deque (FIFO queue)",
        },
        {
            "Algorithm": "Depth-First Search (DFS)",
            "Operational Purpose": "Determines network reachability and accessible facilities from emergency origin.",
            "Graph Representation": "Directed (non-blocked)",
            "Time Complexity": "O(V + E)",
            "Space Complexity": "O(V)",
            "Data Structure Used": "Explicit iterative LIFO stack (prevents call-stack overflow)",
        },
        {
            "Algorithm": "Dijkstra's Algorithm",
            "Operational Purpose": "Computes optimal minimum-travel-time emergency response vehicle routes.",
            "Graph Representation": "Directed Multigraph (one-ways)",
            "Time Complexity": "O((V + E) log V)",
            "Space Complexity": "O(V + E)",
            "Data Structure Used": "heapq binary min-heap with lazy deletion",
        },
        {
            "Algorithm": "A* Search",
            "Operational Purpose": "Computes optimal routes faster using straight-line / v_max admissible heuristic.",
            "Graph Representation": "Directed Multigraph (one-ways)",
            "Time Complexity": "O((V + E) log V)",
            "Space Complexity": "O(V + E)",
            "Data Structure Used": "heapq min-heap with f(u) = g(u) + h(u)",
        },
        {
            "Algorithm": "Disjoint Set Union (Union-Find)",
            "Operational Purpose": "Identifies disconnected connected components and isolated neighborhood pockets.",
            "Graph Representation": "Undirected physical projection",
            "Time Complexity": "O(α(V)) amortized",
            "Space Complexity": "O(V)",
            "Data Structure Used": "Parent and Rank dictionaries with full path compression",
        },
        {
            "Algorithm": "Tarjan's Bridges",
            "Operational Purpose": "Discovers critical roads that are structural single points of failure.",
            "Graph Representation": "Undirected projection (multi-edge aware)",
            "Time Complexity": "O(V + E)",
            "Space Complexity": "O(V)",
            "Data Structure Used": "DFS low-link discovery time tracking",
        },
        {
            "Algorithm": "Tarjan's Articulation Points",
            "Operational Purpose": "Discovers critical intersections whose blockage fragments the network.",
            "Graph Representation": "Undirected projection",
            "Time Complexity": "O(V + E)",
            "Space Complexity": "O(V)",
            "Data Structure Used": "DFS tree children & low-link subtree evaluation",
        },
        {
            "Algorithm": "Kruskal's Algorithm (MST)",
            "Operational Purpose": "Finds minimum-cost set of restoration links to reconnect isolated regions.",
            "Graph Representation": "Contracted component super-graph",
            "Time Complexity": "O(E log E)",
            "Space Complexity": "O(V_comp + E_cand)",
            "Data Structure Used": "Greedy edge sorting + Custom Union-Find",
        },
    ]

    st.dataframe(pd.DataFrame(algo_data), hide_index=True)
