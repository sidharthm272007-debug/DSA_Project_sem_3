"""Tarjan's Low-Link Algorithms for Critical Bridges and Articulation Points.

Purpose:
  Answers the operational questions:
  1. Bridges: "Which roads are single points of failure that, if severed, disconnect portions of the network?"
  2. Articulation Points: "Which road junctions/intersections are critical single points of failure?"

Modeling Details:
  Operates on the UNDIRECTED projection of the road network. In physical reality,
  if a bridge or intersection is blocked, two-way structural transit is severed.
  Crucially handles parallel edges: if two parallel non-blocked roads connect intersection
  u and v, neither is a structural bridge because the parallel route maintains connectivity.

Complexity:
  - Time Complexity: O(V + E) where V is intersections and E is non-blocked road connections.
  - Space Complexity: O(V + E) for low-link metadata and recursion/stack state.

Educational Notes:
  Pure custom DFS low-link implementation with multi-edge awareness.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class CriticalityResult:
    """Encapsulates structural vulnerability analysis on the physical network."""
    bridges: List[Tuple[Any, Any]] = field(default_factory=list)  # List of (u, v) pairs
    articulation_points: Set[Any] = field(default_factory=set)  # Set of critical node IDs
    total_nodes_analyzed: int = 0
    total_edges_analyzed: int = 0

    @property
    def bridge_count(self) -> int:
        return len(self.bridges)

    @property
    def articulation_point_count(self) -> int:
        return len(self.articulation_points)


def find_bridges_and_articulation_points(
    undirected_adjacency: Dict[Any, List[Any]],
    edge_multiplicity: Optional[Dict[Tuple[Any, Any], int]] = None,
) -> CriticalityResult:
    """Execute Tarjan's low-link algorithm on an undirected graph.
    
    Args:
        undirected_adjacency: Dict mapping node u -> list of neighbor nodes v.
        edge_multiplicity: Optional dict mapping canonical (min(u,v), max(u,v)) -> count
          of non-blocked parallel edges. If count > 1, the connection cannot be a bridge.
          
    Returns:
        CriticalityResult containing identified bridges and articulation points.
    """
    tin: Dict[Any, int] = {}
    low: Dict[Any, int] = {}
    timer = 0

    bridges: List[Tuple[Any, Any]] = []
    articulation_points: Set[Any] = set()

    all_nodes = list(undirected_adjacency.keys())
    total_nodes = len(all_nodes)
    total_edges = sum(len(nbrs) for nbrs in undirected_adjacency.values()) // 2

    # Standard DFS using recursion (with sys.setrecursionlimit safety or iterative)
    # Since urban components can have high diameter, we use an iterative approach with explicit frame
    # or clean recursion with depth tracking. Let's use clean recursion with visited check.
    visited: Set[Any] = set()

    for root in all_nodes:
        if root in visited:
            continue

        # Iterative DFS for Tarjan to completely prevent RecursionError on deep paths:
        # We simulate the call stack: (u, p, neighbor_idx)
        # However, because low[u] = min(low[u], low[v]) must execute AFTER visiting child v,
        # we maintain an explicit post-visit traversal.
        # Let's use a two-phase stack approach:
        stack: List[Tuple[Any, Any, int]] = [(root, None, 0)]
        visited.add(root)
        timer += 1
        tin[root] = low[root] = timer
        root_children = 0

        # Maintain parent map for path tracking
        parent_map: Dict[Any, Any] = {root: None}
        # Maintain children in DFS tree for articulation point condition
        tree_children: Dict[Any, List[Any]] = {root: []}

        # DFS traversal loop
        call_stack = [root]
        idx_map = {root: 0}

        while call_stack:
            u = call_stack[-1]
            p = parent_map[u]
            nbrs = undirected_adjacency.get(u, [])
            i = idx_map[u]

            if i < len(nbrs):
                v = nbrs[i]
                idx_map[u] = i + 1

                if v == p:
                    continue  # Ignore direct edge back to DFS parent

                if v in visited:
                    # Back edge: update low[u] using discovery time tin[v]
                    low[u] = min(low[u], tin[v])
                else:
                    # Tree edge
                    visited.add(v)
                    parent_map[v] = u
                    tree_children.setdefault(u, []).append(v)
                    tree_children[v] = []
                    timer += 1
                    tin[v] = low[v] = timer
                    idx_map[v] = 0
                    call_stack.append(v)
            else:
                # Post-visit: backtrack and evaluate conditions
                call_stack.pop()
                if p is not None:
                    # Update parent's low-link
                    low[p] = min(low[p], low[u])

                    # Bridge condition: low[u] > tin[p]
                    if low[u] > tin[p]:
                        canonical = (min(str(p), str(u)), max(str(p), str(u)))
                        # If there are multiple parallel edges between p and u, not a bridge
                        multi = 1
                        if edge_multiplicity and canonical in edge_multiplicity:
                            multi = edge_multiplicity[canonical]
                        if multi <= 1:
                            bridges.append((p, u))

                    # Articulation point condition for non-root: low[u] >= tin[p]
                    if parent_map[p] is not None and low[u] >= tin[p]:
                        articulation_points.add(p)

        # Articulation point condition for root: >= 2 children in DFS tree
        if len(tree_children.get(root, [])) >= 2:
            articulation_points.add(root)

    return CriticalityResult(
        bridges=bridges,
        articulation_points=articulation_points,
        total_nodes_analyzed=total_nodes,
        total_edges_analyzed=total_edges,
    )
