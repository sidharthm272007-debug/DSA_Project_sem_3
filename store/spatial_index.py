"""Own KD-tree for snapping a clicked point to the nearest road node.

Build:  median split on the axis of largest spread, leaf size 16 -> O(N log N).
Query:  branch-and-bound descent with a bounded max-heap -> O(log N) on average.
Everything is iterative (no recursion), so it scales to city-sized inputs.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


class LocalProjection:
    """Equirectangular projection around a city centre. Error is far below 0.1% over 30 km."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._kx = EARTH_RADIUS_M * math.cos(math.radians(lat0))
        self._ky = EARTH_RADIUS_M

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        return np.radians(lon - self.lon0) * self._kx, np.radians(lat - self.lat0) * self._ky

    def to_latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return self.lat0 + np.degrees(y / self._ky), self.lon0 + np.degrees(x / self._kx)

    def distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Euclidean distance in metres between two (lat, lon) coordinates."""
        x1, y1 = self.to_xy(lat1, lon1)
        x2, y2 = self.to_xy(lat2, lon2)
        return float(math.hypot(x2 - x1, y2 - y1))


class KDTree:
    """2-D KD-tree over points in metres. ``xy`` has shape (N, 2)."""

    def __init__(self, xy: np.ndarray, leaf_size: int = 16):
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")
        if len(xy) == 0:
            raise ValueError("cannot build a KD-tree from zero points")
        self.xy = xy
        self.leaf_size = max(1, int(leaf_size))
        self.idx = np.arange(len(xy), dtype=np.int64)
        # per-node arrays (python lists: faster than numpy scalars inside the query loop)
        self._lo: list[int] = []
        self._hi: list[int] = []
        self._dim: list[int] = []
        self._split: list[float] = []
        self._left: list[int] = []
        self._right: list[int] = []
        self._build()

    def _new_node(self, lo: int, hi: int) -> int:
        self._lo.append(lo)
        self._hi.append(hi)
        self._dim.append(-1)
        self._split.append(0.0)
        self._left.append(-1)
        self._right.append(-1)
        return len(self._lo) - 1

    def _build(self) -> None:
        root = self._new_node(0, len(self.xy))
        stack = [root]
        while stack:
            nid = stack.pop()
            lo, hi = self._lo[nid], self._hi[nid]
            if hi - lo <= self.leaf_size:
                continue  # leaf: dim stays -1
            ids = self.idx[lo:hi]
            pts = self.xy[ids]
            d = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
            mid = (hi - lo) // 2
            part = np.argpartition(pts[:, d], mid)
            self.idx[lo:hi] = ids[part]
            split = float(self.xy[self.idx[lo + mid], d])
            left = self._new_node(lo, lo + mid)
            right = self._new_node(lo + mid, hi)
            self._dim[nid], self._split[nid] = d, split
            self._left[nid], self._right[nid] = left, right
            stack.extend((left, right))

    def query(self, x: float, y: float, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances_m) of the k nearest points, nearest first."""
        k = min(k, len(self.xy))
        q = (float(x), float(y))
        best: list[tuple[float, int]] = []  # max-heap via negated squared distance
        stack = [(0, 0.0)]                   # (node id, lower bound on squared distance)
        while stack:
            nid, bound = stack.pop()
            if len(best) == k and bound >= -best[0][0]:
                continue
            d = self._dim[nid]
            if d < 0:
                ids = self.idx[self._lo[nid]:self._hi[nid]]
                diff = self.xy[ids] - q
                d2 = (diff * diff).sum(axis=1)
                for j in range(len(ids)):
                    item = (-float(d2[j]), int(ids[j]))
                    if len(best) < k:
                        heapq.heappush(best, item)
                    elif item > best[0]:
                        heapq.heapreplace(best, item)
                continue
            delta = q[d] - self._split[nid]
            near, far = (self._left[nid], self._right[nid]) if delta <= 0 else (self._right[nid], self._left[nid])
            stack.append((far, max(bound, delta * delta)))
            stack.append((near, bound))
        best.sort(key=lambda t: (-t[0], t[1]))
        return (np.array([i for _, i in best], dtype=np.int64),
                np.sqrt(np.array([-d for d, _ in best], dtype=np.float64)))

    def query_radius(self, x: float, y: float, r: float) -> np.ndarray:
        """Return array of point indices within Euclidean distance r from (x, y)."""
        r2 = float(r) * float(r)
        q = (float(x), float(y))
        found: list[int] = []
        stack = [(0, 0.0)]  # (node_id, bound_squared_dist)
        while stack:
            nid, bound = stack.pop()
            if bound > r2:
                continue
            d = self._dim[nid]
            if d < 0:
                ids = self.idx[self._lo[nid]:self._hi[nid]]
                diff = self.xy[ids] - q
                d2 = (diff * diff).sum(axis=1)
                for j in range(len(ids)):
                    if d2[j] <= r2:
                        found.append(int(ids[j]))
                continue
            delta = q[d] - self._split[nid]
            near, far = (self._left[nid], self._right[nid]) if delta <= 0 else (self._right[nid], self._left[nid])
            stack.append((far, delta * delta))
            stack.append((near, bound))
        return np.array(found, dtype=np.int64)

    def nearest(self, x: float, y: float) -> tuple[int, float]:
        idx, dist = self.query(x, y, 1)
        return int(idx[0]), float(dist[0])


class NodeSnapper:
    """Convenience wrapper: (lat, lon) -> nearest graph node, using a LocalProjection + KDTree."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, centre: tuple[float, float] | None = None):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        c = centre or (float(lat.mean()), float(lon.mean()))
        self.proj = LocalProjection(*c)
        x, y = self.proj.to_xy(lat, lon)
        self.tree = KDTree(np.column_stack([x, y]))

    def snap(self, lat: float, lon: float) -> tuple[int, float]:
        """Return (node id, distance in metres). Callers should warn if the distance is large (> 500 m)."""
        x, y = self.proj.to_xy(lat, lon)
        return self.tree.nearest(float(x), float(y))
