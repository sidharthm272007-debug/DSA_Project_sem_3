"""CityData: everything the simulator and the route recommender need to know about a city, loaded once."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from store.csr_graph import FLAG_BRIDGE, FLAG_FLYOVER, FLAG_TUNNEL, ROAD_CLASS_NAMES, CSRGraph
from store.spatial_index import KDTree, LocalProjection

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "coimbatore"


@dataclass
class Facility:
    id: int
    name: str
    kind: str                      # hospital | fire_station | police | ambulance_post
    tier: int | None               # hospitals: 1 major, 2 general, 3 clinic/specialist
    node: int
    lat: float
    lon: float
    source: str = "osm"            # osm | manual | user
    approx: bool = False
    note: str = ""


@dataclass
class CityData:
    name: str
    g: CSRGraph
    node_pop: np.ndarray
    giant: np.ndarray
    facilities: list[Facility]
    landmarks: list[dict] = field(default_factory=list)
    way_names: dict[int, str] = field(default_factory=dict)
    structures: list[dict] = field(default_factory=list)
    water: dict[str, np.ndarray] = field(default_factory=dict)
    geom_indptr: np.ndarray | None = None
    geom_lonlat: np.ndarray | None = None
    centre: tuple[float, float] = (11.0168, 76.9558)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        g = self.g
        self.proj = LocalProjection(float(g.lat.mean()), float(g.lon.mean()))
        x, y = self.proj.to_xy(g.lat, g.lon)
        self.xy = np.column_stack([x, y])
        self.mid_xy = 0.5 * (self.xy[g.edge_tails] + self.xy[g.indices])
        # people living in disconnected data fragments are excluded from impact accounting
        self.pop = np.where(self.giant, self.node_pop, 0.0).astype(np.float64)
        self.total_pop = float(self.pop.sum())
        gi = np.flatnonzero(self.giant)
        self._giant_ids = gi
        self._giant_tree = KDTree(self.xy[gi])
        self.is_flyover = (g.flags & FLAG_FLYOVER) != 0
        # junctions that touch a bridge, flyover or tunnel: nothing new can be attached to a deck or a tube mid-span
        deck = (g.flags & (FLAG_BRIDGE | FLAG_FLYOVER | FLAG_TUNNEL)) != 0
        self.on_structure = np.zeros(g.n_nodes, dtype=bool)
        self.on_structure[g.edge_tails[deck]] = True
        self.on_structure[g.indices[deck]] = True
        self.is_bridge = (g.flags & FLAG_BRIDGE) != 0
        self.is_arterial = g.is_arterial
        self.is_residential = np.isin(g.road_class, [ROAD_CLASS_NAMES.index(c) for c in ("residential", "living_street", "unclassified")])
        self.edge_km = g.length_m / 1000.0
        self.by_id = {f.id: f for f in self.facilities}
        cx, cy = np.floor(self.xy[:, 0] / 500.0).astype(np.int64), np.floor(self.xy[:, 1] / 500.0).astype(np.int64)
        key = (cx - cx.min()) * 100_000 + (cy - cy.min())
        uniq, self.cell_id = np.unique(key, return_inverse=True)
        self.n_cells = len(uniq)
        cnt = np.bincount(self.cell_id, minlength=self.n_cells).astype(float)
        self.cell_lat = np.bincount(self.cell_id, weights=self.g.lat, minlength=self.n_cells) / cnt
        self.cell_lon = np.bincount(self.cell_id, weights=self.g.lon, minlength=self.n_cells) / cnt
        self.cell_pop = np.bincount(self.cell_id, weights=self.pop, minlength=self.n_cells)
        self._lm_tree = KDTree(np.column_stack(self.proj.to_xy([l["lat"] for l in self.landmarks], [l["lon"] for l in self.landmarks]))) if self.landmarks else None
        self._water_xy = {k: np.column_stack(self.proj.to_xy(v[:, 0], v[:, 1])) for k, v in self.water.items()}
        self._wd_cache: dict[str, np.ndarray] = {}

    # ---- facilities ---------------------------------------------------------------------------------
    def of_kind(self, kind: str, max_tier: int | None = None) -> list[Facility]:
        return [f for f in self.facilities if f.kind == kind and (max_tier is None or (f.tier is not None and f.tier <= max_tier))]

    def hospitals(self, max_tier: int = 2) -> list[Facility]:
        return self.of_kind("hospital", max_tier)

    # ---- snapping and naming ---------------------------------------------------------------------
    def snap(self, lat: float, lon: float) -> tuple[int, float]:
        x, y = self.proj.to_xy(lat, lon)
        j, d = self._giant_tree.nearest(float(x), float(y))
        return int(self._giant_ids[j]), float(d)

    def nearest_landmark(self, node: int) -> tuple[str, float]:
        if self._lm_tree is None:
            return f"node {node}", 0.0
        j, d = self._lm_tree.nearest(float(self.xy[node, 0]), float(self.xy[node, 1]))
        return self.landmarks[j]["name"], float(d)

    def describe_node(self, node: int) -> str:
        name, d = self.nearest_landmark(node)
        return f"{name}" if d < 300 else f"{name} ({d / 1000:.1f} km away)"

    def edge_name(self, e: int) -> str:
        nm = self.way_names.get(int(self.g.way_id[e]))
        if nm:
            return nm
        cls = ROAD_CLASS_NAMES[int(self.g.road_class[e])]
        return f"unnamed {cls} road near {self.describe_node(int(self.g.edge_tails[e]))}"

    def edge_polyline(self, e: int) -> np.ndarray:
        """(k, 2) lon/lat vertices of edge e (falls back to the straight segment)."""
        if self.geom_indptr is not None:
            return self.geom_lonlat[self.geom_indptr[e]:self.geom_indptr[e + 1]]
        g = self.g
        return np.array([[g.lon[g.edge_tails[e]], g.lat[g.edge_tails[e]]], [g.lon[g.indices[e]], g.lat[g.indices[e]]]])

    def dist_to_water(self, name: str | None = None, max_m: float = 2500.0) -> np.ndarray:
        """Distance (m) from every edge midpoint to the named water feature, or to the nearest of all.
        Beyond ``max_m`` the value is inf (no hazard here ever reaches that far). Cached per feature."""
        keys = [name] if name else list(self._water_xy)
        best = np.full(self.g.n_edges, np.inf)
        for k in keys:
            if k not in self._wd_cache:
                self._wd_cache[k] = self._compute_water_dist(self._water_xy[k], max_m)
            best = np.minimum(best, self._wd_cache[k])
        return best

    def water_distance_xy(self, pts: np.ndarray) -> np.ndarray:
        """Distance (m) from local-xy points to the nearest mapped waterbody line (densified to ~30 m). inf if there is no water."""
        if not hasattr(self, "_water_tree"):
            dense = []
            for xy in self._water_xy.values():
                for a, b in zip(xy[:-1], xy[1:]):
                    n = max(1, int(np.hypot(*(b - a)) // 30.0))
                    dense.extend(a + (b - a) * (t / n) for t in range(n + 1))
            self._water_tree = KDTree(np.array(dense)) if dense else None
        if self._water_tree is None:
            return np.full(len(pts), np.inf)
        return np.array([self._water_tree.nearest(float(x), float(y))[1] for x, y in pts])

    def _compute_water_dist(self, pts: np.ndarray, max_m: float) -> np.ndarray:
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):                  # densify to ~30 m so we measure distance to the line, not to vertices
            n = max(1, int(np.hypot(*(b - a)) // 30.0))
            for t in range(1, n + 1):
                dense.append(a + (b - a) * (t / n))
        tree = KDTree(np.array(dense))
        lo, hi = pts.min(axis=0) - max_m, pts.max(axis=0) + max_m
        m = self.mid_xy
        cand = np.flatnonzero((m[:, 0] >= lo[0]) & (m[:, 0] <= hi[0]) & (m[:, 1] >= lo[1]) & (m[:, 1] <= hi[1]))
        out = np.full(len(m), np.inf)
        for i in cand:
            d = tree.nearest(float(m[i, 0]), float(m[i, 1]))[1]
            if d <= max_m:
                out[i] = d
        return out


def load_city(data_dir: str | Path | None = None, extra_facilities: bool = True) -> CityData:
    d = Path(data_dir) if data_dir else DATA_DIR
    dd = d / "derived"
    g = CSRGraph.load(d / "graph.npz")
    fac = [Facility(**{k: f[k] for k in ("id", "name", "kind", "tier", "node", "lat", "lon", "source", "approx", "note")})
           for f in json.loads((dd / "facilities.json").read_text(encoding="utf-8"))]
    gp = d / "geometry.npz"
    gi = gl = None
    if gp.exists():
        with np.load(gp) as z:
            gi, gl = z["geom_indptr"], z["geom_lonlat"]
    water = {}
    wp = d / "water.json"
    if wp.exists():
        for k, v in json.loads(wp.read_text(encoding="utf-8")).items():
            arr = np.array(v, dtype=np.float64)
            if arr.ndim == 2 and len(arr) >= 2:
                water[k] = arr
    meta = json.loads((dd / "meta.json").read_text(encoding="utf-8"))
    city = CityData(
        name="Coimbatore", g=g, node_pop=np.load(dd / "node_pop.npy").astype(np.float64), giant=np.load(dd / "giant_mask.npy"),
        facilities=fac, landmarks=json.loads((dd / "landmarks.json").read_text(encoding="utf-8")),
        way_names={int(k): v for k, v in json.loads((dd / "way_names.json").read_text(encoding="utf-8")).items()},
        structures=json.loads((dd / "structures.json").read_text(encoding="utf-8")),
        water=water, geom_indptr=gi, geom_lonlat=gl, meta=meta,
    )
    return city
