"""Compact CSR (compressed sparse row) road graph.

Why CSR: a NetworkX-style dict-of-dicts graph needs hundreds of bytes per edge.
CSR keeps the whole city in a handful of NumPy arrays and loads from one .npz file.

Layout (N nodes, E directed edges, edges sorted by tail node):
    indptr[u] .. indptr[u+1]   -> range of edge ids leaving node u
    indices[e]                 -> head (target) node of edge e
    rev[e]                     -> id of the opposite-direction edge, or -1 for one-way roads

Design notes
- tt_s / length_m are float64 (the spec said float32) so oracle tests can demand exact equality.
- The graph is treated as IMMUTABLE. To change it (add a flyover, contraflow arcs) call
  ``with_extra_edges`` which returns a NEW graph plus index maps. "Undo" = drop the new graph.
- Per-scenario state (blocked / slowed) lives OUTSIDE the graph in ``ScenarioState``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

# ---- edge flag bits -------------------------------------------------------------------
FLAG_BRIDGE = 1
FLAG_TUNNEL = 2
FLAG_ONEWAY = 4
FLAG_FLYOVER = 8   # bridge with layer >= 1 on a road class that can carry a flyover
FLAG_ADDED = 16    # arc added by a what-if patch (flyover, contraflow)

# ---- road classes ---------------------------------------------------------------------
ROAD_CLASS_NAMES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
]
ROAD_CLASS_CODE = {name: i for i, name in enumerate(ROAD_CLASS_NAMES)}
ARTERIAL_CODES = frozenset(range(0, 4))  # motorway, trunk, primary, secondary

# ---- scenario condition codes (kept from Phase 1's RoadCondition tiers) ----------------
COND_NORMAL, COND_SLOWED, COND_SEVERE, COND_BLOCKED = 0, 1, 2, 3

_NODE_ARRAYS = ("lat", "lon", "elev_m", "slope_deg", "node_osm_id")
_EDGE_ARRAYS = ("tt_s", "length_m", "road_class", "lanes_dir", "flags", "way_id", "rev")


@dataclass
class CSRGraph:
    indptr: np.ndarray            # int32 [N+1]
    indices: np.ndarray           # int32 [E]
    tt_s: np.ndarray              # float64 [E]  free-flow travel time, seconds
    length_m: np.ndarray          # float64 [E]
    road_class: np.ndarray        # uint8 [E]
    lanes_dir: np.ndarray         # uint8 [E]
    flags: np.ndarray             # uint8 [E]
    way_id: np.ndarray            # int64 [E]
    rev: np.ndarray               # int32 [E]
    lat: np.ndarray               # float64 [N]
    lon: np.ndarray               # float64 [N]
    elev_m: np.ndarray | None = None      # float32 [N]
    slope_deg: np.ndarray | None = None   # float32 [N]
    node_osm_id: np.ndarray | None = None  # int64 [N]

    def __post_init__(self) -> None:
        n = len(self.indptr) - 1
        e = len(self.indices)
        for name in _EDGE_ARRAYS:
            if len(getattr(self, name)) != e:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {e}")
        for name in ("lat", "lon"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {n}")
        if self.elev_m is None:
            self.elev_m = np.zeros(n, dtype=np.float32)
        if self.slope_deg is None:
            self.slope_deg = np.zeros(n, dtype=np.float32)
        if self.node_osm_id is None:
            self.node_osm_id = np.arange(n, dtype=np.int64)

    # ---- basic properties ---------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.indptr) - 1

    @property
    def n_edges(self) -> int:
        return len(self.indices)

    @cached_property
    def edge_tails(self) -> np.ndarray:
        """Tail node of every edge (the row each edge id belongs to)."""
        return np.repeat(np.arange(self.n_nodes, dtype=np.int32), np.diff(self.indptr))

    @cached_property
    def is_arterial(self) -> np.ndarray:
        return np.isin(self.road_class, list(ARTERIAL_CODES))

    def out_edges(self, u: int) -> range:
        return range(int(self.indptr[u]), int(self.indptr[u + 1]))

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        tails,
        heads,
        tt_s,
        *,
        length_m=None,
        road_class=None,
        lanes_dir=None,
        flags=None,
        way_id=None,
        rev=None,
        lat=None,
        lon=None,
        elev_m=None,
        slope_deg=None,
        node_osm_id=None,
    ) -> "CSRGraph":
        """Build a graph from an unsorted edge list.

        ``rev`` (optional) gives, for each INPUT edge, the input index of its reverse edge or -1.
        """
        tails = np.asarray(tails, dtype=np.int64)
        heads = np.asarray(heads, dtype=np.int64)
        e = len(tails)
        tt_s = np.asarray(tt_s, dtype=np.float64)
        if len(heads) != e or len(tt_s) != e:
            raise ValueError("tails, heads and tt_s must have equal length")
        if e and (tails.min() < 0 or tails.max() >= n_nodes or heads.min() < 0 or heads.max() >= n_nodes):
            raise ValueError("edge endpoint out of range")
        if e and (tt_s < 0).any():
            raise ValueError("negative travel time")

        def _arr(x, dtype, default):
            return np.full(e, default, dtype=dtype) if x is None else np.asarray(x, dtype=dtype)

        length_m = _arr(length_m, np.float64, 0.0)
        road_class = _arr(road_class, np.uint8, ROAD_CLASS_CODE["residential"])
        lanes_dir = _arr(lanes_dir, np.uint8, 1)
        flags = _arr(flags, np.uint8, 0)
        way_id = _arr(way_id, np.int64, -1)
        rev_in = _arr(rev, np.int64, -1)

        order = np.argsort(tails, kind="stable")          # new position -> old index
        inv = np.empty(e, dtype=np.int64)
        inv[order] = np.arange(e)                          # old index -> new position
        rev_old = rev_in[order]
        rev_new = np.where(rev_old >= 0, inv[np.maximum(rev_old, 0)], -1)

        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(tails, minlength=n_nodes), out=indptr[1:])

        n = n_nodes
        return cls(
            indptr=indptr.astype(np.int32),
            indices=heads[order].astype(np.int32),
            tt_s=tt_s[order],
            length_m=length_m[order],
            road_class=road_class[order],
            lanes_dir=lanes_dir[order],
            flags=flags[order],
            way_id=way_id[order],
            rev=rev_new.astype(np.int32),
            lat=np.zeros(n) if lat is None else np.asarray(lat, dtype=np.float64),
            lon=np.zeros(n) if lon is None else np.asarray(lon, dtype=np.float64),
            elev_m=None if elev_m is None else np.asarray(elev_m, dtype=np.float32),
            slope_deg=None if slope_deg is None else np.asarray(slope_deg, dtype=np.float32),
            node_osm_id=None if node_osm_id is None else np.asarray(node_osm_id, dtype=np.int64),
        )

    # ---- derived graphs -----------------------------------------------------------------
    def reverse(self) -> tuple["CSRGraph", np.ndarray]:
        """Transpose graph. Returns (reversed_graph, edge_map) with edge_map[j] = original edge id
        of reversed edge j. Use ``cost[edge_map]`` to carry a cost vector across."""
        heads = self.indices.astype(np.int64)
        order = np.argsort(heads, kind="stable")
        indptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(heads, minlength=self.n_nodes), out=indptr[1:])
        rg = CSRGraph(
            indptr=indptr.astype(np.int32),
            indices=self.edge_tails[order].astype(np.int32),
            tt_s=self.tt_s[order],
            length_m=self.length_m[order],
            road_class=self.road_class[order],
            lanes_dir=self.lanes_dir[order],
            flags=self.flags[order],
            way_id=self.way_id[order],
            rev=np.full(self.n_edges, -1, dtype=np.int32),
            lat=self.lat, lon=self.lon, elev_m=self.elev_m,
            slope_deg=self.slope_deg, node_osm_id=self.node_osm_id,
        )
        return rg, order.astype(np.int32)

    def with_extra_edges(
        self,
        tails,
        heads,
        *,
        tt_s,
        length_m,
        road_class,
        lanes_dir,
        flags,
        way_id=None,
        new_rev=None,
    ) -> tuple["CSRGraph", np.ndarray, np.ndarray]:
        """Copy-on-write: return (new_graph, old_to_new, new_ids).

        old_to_new[e_old] = position of the old edge in the new graph.
        new_ids[i]        = position of the i-th added edge in the new graph.
        ``new_rev[i]`` (optional) = index (within the added list) of the reverse of added edge i, or -1.
        Carry any per-edge array across with ``align_edge_array``.
        """
        tails = np.asarray(tails, dtype=np.int64)
        heads = np.asarray(heads, dtype=np.int64)
        k = len(tails)
        e0 = self.n_edges
        if new_rev is None:
            new_rev = np.full(k, -1, dtype=np.int64)
        new_rev = np.asarray(new_rev, dtype=np.int64)
        rev_all = np.concatenate([self.rev.astype(np.int64), np.where(new_rev >= 0, new_rev + e0, -1)])

        def cat(old, new, dtype):
            return np.concatenate([old, np.asarray(new, dtype=dtype)])

        g = CSRGraph.from_edges(
            self.n_nodes,
            np.concatenate([self.edge_tails.astype(np.int64), tails]),
            np.concatenate([self.indices.astype(np.int64), heads]),
            cat(self.tt_s, tt_s, np.float64),
            length_m=cat(self.length_m, length_m, np.float64),
            road_class=cat(self.road_class, road_class, np.uint8),
            lanes_dir=cat(self.lanes_dir, lanes_dir, np.uint8),
            flags=cat(self.flags, flags, np.uint8),
            way_id=cat(self.way_id, np.full(k, -1) if way_id is None else way_id, np.int64),
            rev=rev_all,
            lat=self.lat, lon=self.lon, elev_m=self.elev_m,
            slope_deg=self.slope_deg, node_osm_id=self.node_osm_id,
        )
        # from_edges sorts stably by tail; recompute the permutation the same way.
        all_tails = np.concatenate([self.edge_tails.astype(np.int64), tails])
        order = np.argsort(all_tails, kind="stable")
        inv = np.empty(e0 + k, dtype=np.int64)
        inv[order] = np.arange(e0 + k)
        return g, inv[:e0].astype(np.int32), inv[e0:].astype(np.int32)

    @staticmethod
    def align_edge_array(arr, old_to_new, new_ids, new_values) -> np.ndarray:
        """Carry a per-edge array (flows, capacities, ...) onto a patched graph."""
        arr = np.asarray(arr)
        out = np.empty(len(arr) + len(new_ids), dtype=np.result_type(arr.dtype, np.asarray(new_values).dtype))
        out[old_to_new] = arr
        out[new_ids] = new_values
        return out

    def with_extra_nodes(
        self,
        lat,
        lon,
        *,
        elev_m=None,
        slope_deg=None,
        node_osm_id=None,
    ) -> tuple["CSRGraph", np.ndarray]:
        """Copy-on-write: append new nodes at the end with zero out-edges.

        Returns (new_graph, new_node_ids).
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        k = len(lat)
        if len(lon) != k:
            raise ValueError("lat and lon must have equal length")
        n0 = self.n_nodes
        new_ids = np.arange(n0, n0 + k, dtype=np.int32)

        new_indptr = np.pad(self.indptr, (0, k), mode="edge")
        new_lat = np.concatenate([self.lat, lat])
        new_lon = np.concatenate([self.lon, lon])

        elev = np.zeros(k, dtype=np.float32) if elev_m is None else np.asarray(elev_m, dtype=np.float32)
        slope = np.zeros(k, dtype=np.float32) if slope_deg is None else np.asarray(slope_deg, dtype=np.float32)
        osm_id = np.arange(-1, -1 - k, -1, dtype=np.int64) if node_osm_id is None else np.asarray(node_osm_id, dtype=np.int64)

        new_elev = np.concatenate([self.elev_m, elev])
        new_slope = np.concatenate([self.slope_deg, slope])
        new_osm = np.concatenate([self.node_osm_id, osm_id])

        g = CSRGraph(
            indptr=new_indptr,
            indices=self.indices.copy(),
            tt_s=self.tt_s.copy(),
            length_m=self.length_m.copy(),
            road_class=self.road_class.copy(),
            lanes_dir=self.lanes_dir.copy(),
            flags=self.flags.copy(),
            way_id=self.way_id.copy(),
            rev=self.rev.copy(),
            lat=new_lat,
            lon=new_lon,
            elev_m=new_elev,
            slope_deg=new_slope,
            node_osm_id=new_osm,
        )
        g.validate()
        return g, new_ids

    def without_edges(self, mask: np.ndarray) -> tuple["CSRGraph", np.ndarray]:
        """Copy-on-write: drop edges where mask is True.

        Returns (new_graph, old_to_new).
        old_to_new[e_old] = position of old edge in new graph, or -1 if dropped.
        """
        arr = np.asarray(mask)
        if arr.dtype == bool:
            if len(arr) != self.n_edges:
                raise ValueError(f"mask length {len(arr)} != n_edges {self.n_edges}")
            mask = arr
        else:
            idx = arr.astype(np.int64)
            mask = np.zeros(self.n_edges, dtype=bool)
            mask[idx] = True
        keep = ~mask
        old_to_new = np.full(self.n_edges, -1, dtype=np.int32)
        old_to_new[keep] = np.arange(np.sum(keep), dtype=np.int32)

        tails = self.edge_tails[keep]
        indices = self.indices[keep]
        tt_s = self.tt_s[keep]
        length_m = self.length_m[keep]
        road_class = self.road_class[keep]
        lanes_dir = self.lanes_dir[keep]
        flags = self.flags[keep]
        way_id = self.way_id[keep]

        old_rev = self.rev[keep]
        has_rev = (old_rev >= 0) & keep[np.maximum(0, old_rev)]
        new_rev = np.where(has_rev, old_to_new[np.maximum(0, old_rev)], -1).astype(np.int32)

        counts = np.bincount(tails, minlength=self.n_nodes)
        indptr = np.zeros(self.n_nodes + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        g = CSRGraph(
            indptr=indptr,
            indices=indices.astype(np.int32),
            tt_s=tt_s,
            length_m=length_m,
            road_class=road_class,
            lanes_dir=lanes_dir,
            flags=flags,
            way_id=way_id,
            rev=new_rev,
            lat=self.lat,
            lon=self.lon,
            elev_m=self.elev_m,
            slope_deg=self.slope_deg,
            node_osm_id=self.node_osm_id,
        )
        g.validate()
        return g, old_to_new

    def subgraph(self, edge_mask: np.ndarray) -> tuple["CSRGraph", np.ndarray, np.ndarray]:
        """Extract subgraph keeping edges where edge_mask is True.

        Compresses node space to only nodes with incident kept edges.
        Returns (sub_graph, node_map, edge_map).
        node_map[u_old] = new node index (or -1 if omitted).
        edge_map[e_old] = new edge index (or -1 if omitted).
        """
        keep_edges = np.asarray(edge_mask, dtype=bool)
        if len(keep_edges) != self.n_edges:
            raise ValueError(f"edge_mask length {len(keep_edges)} != n_edges {self.n_edges}")

        kept_tails = self.edge_tails[keep_edges]
        kept_heads = self.indices[keep_edges]

        used_nodes = np.unique(np.concatenate([kept_tails, kept_heads]))
        node_map = np.full(self.n_nodes, -1, dtype=np.int32)
        node_map[used_nodes] = np.arange(len(used_nodes), dtype=np.int32)

        new_tails = node_map[kept_tails]
        new_heads = node_map[kept_heads]

        edge_map = np.full(self.n_edges, -1, dtype=np.int32)
        edge_map[keep_edges] = np.arange(np.sum(keep_edges), dtype=np.int32)

        old_rev = self.rev[keep_edges]
        has_rev = (old_rev >= 0) & keep_edges[np.maximum(0, old_rev)]
        new_rev = np.where(has_rev, edge_map[np.maximum(0, old_rev)], -1).astype(np.int32)

        counts = np.bincount(new_tails, minlength=len(used_nodes))
        indptr = np.zeros(len(used_nodes) + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        g = CSRGraph(
            indptr=indptr,
            indices=new_heads.astype(np.int32),
            tt_s=self.tt_s[keep_edges],
            length_m=self.length_m[keep_edges],
            road_class=self.road_class[keep_edges],
            lanes_dir=self.lanes_dir[keep_edges],
            flags=self.flags[keep_edges],
            way_id=self.way_id[keep_edges],
            rev=new_rev,
            lat=self.lat[used_nodes],
            lon=self.lon[used_nodes],
            elev_m=self.elev_m[used_nodes],
            slope_deg=self.slope_deg[used_nodes],
            node_osm_id=self.node_osm_id[used_nodes],
        )
        g.validate()
        return g, node_map, edge_map

    # ---- I/O and validation -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        arrays = {name: getattr(self, name) for name in ("indptr", "indices", *_EDGE_ARRAYS, *_NODE_ARRAYS)}
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CSRGraph":
        with np.load(path) as z:
            return cls(**{k: z[k] for k in z.files})

    def validate(self) -> None:
        """Raise AssertionError if the structure is inconsistent. Cheap enough to run after every build."""
        assert self.indptr[0] == 0 and self.indptr[-1] == self.n_edges, "indptr ends"
        assert (np.diff(self.indptr) >= 0).all(), "indptr not monotonic"
        if self.n_edges:
            assert self.indices.min() >= 0 and self.indices.max() < self.n_nodes, "head out of range"
            assert (self.tt_s >= 0).all(), "negative travel time"
            has = self.rev >= 0
            r = self.rev[has]
            assert (self.rev[r] == np.flatnonzero(has)).all(), "rev is not an involution"
            assert (self.indices[r] == self.edge_tails[has]).all(), "rev head mismatch"


@dataclass
class ScenarioState:
    """Per-scenario road condition, shared by every algorithm (Phase 1's shared-state rule)."""

    cond: np.ndarray   # uint8 [E]  COND_*
    mult: np.ndarray   # float64 [E]  travel-time multiplier (inf = blocked)

    @classmethod
    def normal(cls, n_edges: int) -> "ScenarioState":
        return cls(np.zeros(n_edges, dtype=np.uint8), np.ones(n_edges, dtype=np.float64))

    @classmethod
    def from_probability(cls, p: np.ndarray, thresholds: dict, multipliers: dict) -> "ScenarioState":
        """Threshold failure probabilities into the four Phase 1 tiers."""
        p = np.asarray(p, dtype=np.float64)
        cond = np.zeros(len(p), dtype=np.uint8)
        cond[p >= thresholds["slowed"]] = COND_SLOWED
        cond[p >= thresholds["severe"]] = COND_SEVERE
        cond[p >= thresholds["blocked"]] = COND_BLOCKED
        mult = np.ones(len(p), dtype=np.float64)
        mult[cond == COND_SLOWED] = multipliers["slowed"]
        mult[cond == COND_SEVERE] = multipliers["severe"]
        mult[cond == COND_BLOCKED] = np.inf
        return cls(cond, mult)

    @property
    def blocked(self) -> np.ndarray:
        return self.cond == COND_BLOCKED

    def cost(self, tt_s: np.ndarray) -> np.ndarray:
        """Travel-time cost vector with inf on blocked edges."""
        return tt_s * self.mult
