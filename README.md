# ResQGraph — Phase 1

## Dynamic OpenStreetMap Disaster Response & Road Network Resilience Simulator

ResQGraph is a disaster-response and road-network resilience simulator built on real OpenStreetMap (OSM) spatial networks. It simulates disaster scenarios (floods, earthquakes, landslides, urban fires, storms, and custom blockages), and uses pure custom Data Structures and Algorithms (DSA) to analyze physical network disruption, compute emergency routes, detect isolated regions, discover critical infrastructure bottlenecks, and generate minimum-cost restoration recommendations via Kruskal's MST.

---

## Key Features

1. **Real OpenStreetMap Integration**: Ingestion of real-world street networks using OSMnx 2.x API with local GraphML caching.
2. **Dual Graph Adapters (Shared Scenario State)**:
   - `DirectedGraphAdapter`: Directed multigraph routing respecting one-ways, parallel edge keys, and dynamic speed penalties.
   - `UndirectedGraphAdapter`: Physical structural connectivity projection for component and bridge analysis.
   - Both adapters share the exact same `ScenarioState` mapping `(u, v, key) -> RoadCondition`.
3. **Pure Custom DSA Implementations (No Library Black-Boxes)**:
   - **BFS (`algorithms/bfs.py`)**: Models topological disaster propagation level-by-level outward from epicenters.
   - **DFS (`algorithms/dfs.py`)**: Computes dispatch reachability and audits access to critical facilities (hospitals, fire stations).
   - **Dijkstra's Algorithm (`algorithms/dijkstra.py`)**: Computes optimal minimum-travel-time emergency routes using `heapq` with lazy deletion.
   - **A\* Search (`algorithms/astar.py`)**: Goal-directed routing with an admissible lower-bound travel-time heuristic ($h = \text{distance} / v_{\max}$).
   - **Union-Find / DSU (`algorithms/union_find.py`)**: Disjoint Set Union with full path compression and union by rank.
   - **Tarjan's Low-Link Algorithms (`algorithms/tarjan.py`)**: Multi-edge aware bridge detection and articulation point identification.
   - **Kruskal's Algorithm (`algorithms/kruskal.py`)**: Operates on a contracted component-level super-graph to generate minimum-cost restoration plans.
4. **Distinct Restoration Candidate Modeling**:
   - **Debris Clearance**: Reopening existing physical OSM roads currently blocked by disaster.
   - **Hypothetical Emergency Links**: Conceptual emergency bypasses across geographic gaps, explicitly labeled as simulated/conceptual.
5. **Interactive Folium Command Center**:
   - Dynamic map layers with visual hierarchy (Disaster zone, Blocked roads, Slowed roads, Emergency routes, Isolated regions, Bridges, Articulation points, Facilities, and MST restorations).
   - Persistent operational legend and OpenStreetMap attribution.
6. **Academic & Operational Dashboard**:
   - Top KPI metrics banner.
   - Before vs. After network impact analysis table.
   - Dijkstra vs. A* side-by-side performance benchmarking.
   - Educational algorithm reference with complexities ($O(V+E)$, $O((V+E)\log V)$, $\alpha(V)$, etc.).
7. **Phase 2 Extensibility**: Clean abstraction interfaces (`DisasterEventFeedInterface`) designed for future real-world weather and government feed integration.

---

## Synthetic Cost Disclaimer

> **SIMULATION ESTIMATE ONLY**: All restoration costs, distances, obstacle penalties, and rankings are synthetic simulation heuristics designed for educational and graph-optimization demonstration. They do not represent real civil engineering, governmental, or financial infrastructure estimates.

---

## Project Structure

```text
ADSA-PF/
├── app.py                     # Streamlit Command Center entry point
├── config.py                  # Simulation constants, fallback speeds, presets, palette
├── requirements.txt           # Project dependencies
├── README.md                  # System documentation
│
├── data/
│   └── osm/                   # Local GraphML network cache
│
├── graph/
│   ├── __init__.py
│   ├── osm_loader.py          # OSMnx 2.x data acquisition & caching
│   ├── graph_builder.py       # Speed model, travel times, dynamic v_max
│   ├── graph_adapter.py       # Directed & Undirected shared adapters
│   └── graph_utils.py         # Haversine distance, geometries, bounding boxes
│
├── algorithms/
│   ├── __init__.py
│   ├── bfs.py                 # Pure BFS propagation
│   ├── dfs.py                 # Pure DFS reachability
│   ├── dijkstra.py            # Dijkstra with lazy deletion
│   ├── astar.py               # A* with admissible travel-time heuristic
│   ├── union_find.py          # Union-Find (path compression & rank)
│   ├── tarjan.py              # Tarjan bridges & articulation points
│   └── kruskal.py             # Kruskal's MST on component graph
│
├── disaster/
│   ├── __init__.py
│   ├── road_conditions.py     # RoadCondition enum & ScenarioState container
│   ├── scenario.py            # DisasterScenario dataclass & Phase 2 interface
│   ├── simulator.py           # DisasterImpactEngine (geographic & hybrid)
│   └── propagation.py         # BFS topological cascade engine
│
├── analysis/
│   ├── __init__.py
│   ├── reachability.py        # Post-disaster dispatch reachability audit
│   ├── isolation.py           # Connected components & isolated region discovery
│   └── criticality.py         # Structural vulnerability & Tarjan metrics
│
├── restoration/
│   ├── __init__.py
│   ├── candidates.py          # Debris clearance vs hypothetical link generation
│   └── restoration.py         # Kruskal MST restoration plan builder
│
├── facilities/
│   ├── __init__.py
│   └── osm_facilities.py      # OSM POI facility acquisition & node mapping
│
├── visualization/
│   ├── __init__.py
│   ├── map_view.py            # Folium layered command map & legend
│   └── dashboard.py           # Streamlit metrics, before/after, benchmark UI
│
└── tests/
    ├── __init__.py
    ├── test_bfs.py
    ├── test_dfs.py
    ├── test_dijkstra.py
    ├── test_astar.py
    ├── test_union_find.py
    ├── test_tarjan.py
    ├── test_kruskal.py
    ├── test_isolation.py
    ├── test_candidates.py
    ├── test_adapter.py        # Shared scenario state integration test
    └── test_validation_reference.py  # NetworkX oracle validation tests
```

---

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Automated Test Suite**:
   ```bash
   pytest -v tests/
   ```

3. **Launch the ResQGraph Command Center**:
   ```bash
   streamlit run app.py
   ```

---

## Verification & Test Oracle

The test suite includes 33 comprehensive unit and integration tests across all algorithm modules and edge cases. To ensure academic rigor and correctness, `tests/test_validation_reference.py` compares our custom implementations against NetworkX reference implementations as a test oracle on benchmark graphs. NetworkX is strictly prohibited from production algorithm execution and is used solely in test fixtures.
