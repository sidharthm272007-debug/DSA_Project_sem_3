"""Disjoint Set Union (Union-Find) with Path Compression and Union by Rank.

Purpose:
  Answers the operational question:
  "How many disconnected pieces does the road network contain after disaster damage?"

Algorithm Overview:
  Maintains a partition of graph vertices into disjoint connected components.
  Operates on the undirected physical road network projection.

Optimizations:
  1. Full Path Compression: Flattens the tree during find() so future queries are O(1).
  2. Union by Rank: Attaches shallower trees under deeper trees to minimize height growth.

Complexity:
  - Time Complexity: O(alpha(V)) amortized time per operation, where alpha is the
    inverse Ackermann function (alpha(V) <= 4 for all practical universe sizes).
  - Space Complexity: O(V) to store parent and rank dictionaries.
"""

from typing import Dict, List, Set, Any, Iterable


class UnionFind:
    """Disjoint Set Union (DSU) data structure with path compression and union by rank."""

    def __init__(self, elements: Iterable[Any] = None):
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}
        self.size: Dict[Any, int] = {}
        self._component_count: int = 0

        if elements is not None:
            for elem in elements:
                self.add(elem)

    def add(self, x: Any) -> None:
        """Register a new element if not already present."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1
            self._component_count += 1

    def find(self, x: Any) -> Any:
        """Find the representative root of element x with full path compression."""
        if x not in self.parent:
            self.add(x)
            return x

        # Path compression: point node directly to root
        root = x
        while root != self.parent[root]:
            root = self.parent[root]

        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt

        return root

    def union(self, x: Any, y: Any) -> bool:
        """Merge the sets containing elements x and y using union by rank.
        
        Returns:
            True if elements were in distinct sets and merged; False if already connected.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank: attach tree with lower rank under higher rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.size[root_x] += self.size[root_y]

        self._component_count -= 1
        return True

    def connected(self, x: Any, y: Any) -> bool:
        """Check if x and y belong to the same connected component."""
        return self.find(x) == self.find(y)

    def get_components(self) -> Dict[Any, List[Any]]:
        """Group elements by their canonical component root.
        
        Returns:
            Dict mapping root_node -> list of member nodes.
        """
        components: Dict[Any, List[Any]] = {}
        for elem in list(self.parent.keys()):
            root = self.find(elem)
            if root not in components:
                components[root] = []
            components[root].append(elem)
        return components

    def component_count(self) -> int:
        """Return the number of disjoint connected sets."""
        return self._component_count

    def get_component_size(self, x: Any) -> int:
        """Return total element count in x's component."""
        root = self.find(x)
        return self.size[root]
