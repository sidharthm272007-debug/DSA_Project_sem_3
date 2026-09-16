"""Isolated Region Detection using Disjoint Set Union (Union-Find).

Purpose:
  Identifies disconnected clusters of the road network that have become physically
  cut off from the primary emergency-access network due to disaster damage.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from algorithms.union_find import UnionFind
from graph.graph_adapter import UndirectedGraphAdapter
from graph.graph_utils import get_node_coords


@dataclass
class IsolatedRegion:
    """Represents a disconnected pocket of the road network."""
    region_id: str  # e.g. "C_1", "C_2"
    root_node: Any
    nodes: List[Any]
    node_count: int
    centroid_lat: float
    centroid_lon: float
    trapped_facilities: List[Dict[str, Any]] = field(default_factory=list)
    boundary_nodes: List[Any] = field(default_factory=list)

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.centroid_lat, self.centroid_lon


@dataclass
class IsolationAnalysisReport:
    """Summary of network component partitioning and isolated regions."""
    total_components: int
    main_component_id: str
    main_component_nodes: List[Any]
    main_component_size: int
    isolated_regions: List[IsolatedRegion] = field(default_factory=list)
    total_isolated_nodes: int = 0
    all_components_map: Dict[str, List[Any]] = field(default_factory=dict)

    @property
    def has_isolated_regions(self) -> bool:
        return len(self.isolated_regions) > 0

    @property
    def isolated_regions_count(self) -> int:
        return len(self.isolated_regions)


def detect_isolated_regions(
    undirected_adapter: UndirectedGraphAdapter,
    reference_origin_node: Optional[Any] = None,
    facilities: Optional[List[Dict[str, Any]]] = None,
) -> IsolationAnalysisReport:
    """Detect all disconnected components and identify isolated regions using Union-Find.
    
    Args:
        undirected_adapter: UndirectedGraphAdapter with current scenario conditions.
        reference_origin_node: Emergency operations center node to anchor the Main Component.
        facilities: Optional list of facilities to audit for isolation.
        
    Returns:
        IsolationAnalysisReport with complete component census.
    """
    nodes = undirected_adapter.nodes

    # 1. Run Union-Find over the physical undirected projection (non-blocked edges)
    uf = UnionFind(nodes)

    for u, v in undirected_adapter.get_undirected_edge_list():
        uf.union(u, v)

    raw_components = uf.get_components()

    # 2. Determine the Main Component
    # If reference origin is specified and valid, its component is Main
    main_root = None
    if reference_origin_node is not None and reference_origin_node in uf.parent:
        main_root = uf.find(reference_origin_node)

    # Fallback to largest component by size if origin is unspecified or disconnected
    if main_root is None or main_root not in raw_components:
        if raw_components:
            main_root = max(raw_components.keys(), key=lambda r: len(raw_components[r]))

    # Map node -> facility for fast lookup
    node_to_facility = {}
    if facilities:
        for fac in facilities:
            n = fac.get("node")
            if n is not None:
                node_to_facility.setdefault(n, []).append(fac)

    # Build structured isolated region objects
    isolated_regions: List[IsolatedRegion] = []
    all_comps_map: Dict[str, List[Any]] = {}
    main_nodes = raw_components.get(main_root, []) if main_root else []
    all_comps_map["Main_Component"] = main_nodes

    region_idx = 1
    total_isolated = 0

    # Sort remaining components by size descending for clean presentation
    other_roots = [r for r in raw_components.keys() if r != main_root]
    other_roots.sort(key=lambda r: len(raw_components[r]), reverse=True)

    for r in other_roots:
        c_nodes = raw_components[r]
        if not c_nodes:
            continue

        c_id = f"Region_{region_idx}"
        all_comps_map[c_id] = c_nodes
        region_idx += 1
        total_isolated += len(c_nodes)

        # Compute centroid
        lats = []
        lons = []
        trapped_facs = []
        for n in c_nodes:
            lat, lon = undirected_adapter.get_node_coords(n)
            lats.append(lat)
            lons.append(lon)
            if n in node_to_facility:
                trapped_facs.extend(node_to_facility[n])

        centroid_lat = sum(lats) / len(lats) if lats else 0.0
        centroid_lon = sum(lons) / len(lons) if lons else 0.0

        # Boundary nodes: in Phase 1, boundary candidates will be refined by the restoration module
        boundary = c_nodes[: min(len(c_nodes), 10)]

        isolated_regions.append(
            IsolatedRegion(
                region_id=c_id,
                root_node=r,
                nodes=c_nodes,
                node_count=len(c_nodes),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                trapped_facilities=trapped_facs,
                boundary_nodes=boundary,
            )
        )

    return IsolationAnalysisReport(
        total_components=uf.component_count(),
        main_component_id="Main_Component",
        main_component_nodes=main_nodes,
        main_component_size=len(main_nodes),
        isolated_regions=isolated_regions,
        total_isolated_nodes=total_isolated,
        all_components_map=all_comps_map,
    )
