"""Derive the app's data layer from the shipped OSM extract. Deterministic, no network.

    python -m scripts.build_derived

Inputs (data/coimbatore): graph.npz, geometry.npz, places.json, water.json, facilities_manual.csv,
                          data/raw/raw_roads.json (street names)
Outputs (data/coimbatore/derived): node_pop.npy, giant_mask.npy, facilities.json, landmarks.json,
                                   structures.json, way_names.json, meta.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.scc import strongly_connected_components  # noqa: E402
from assumptions import load_assumptions  # noqa: E402
from store.csr_graph import FLAG_BRIDGE, FLAG_FLYOVER, ROAD_CLASS_NAMES, CSRGraph  # noqa: E402
from store.spatial_index import KDTree, LocalProjection, NodeSnapper  # noqa: E402

TYPED_LANDMARKS = {  # hand-typed in an earlier build with coordinates up to 6.8 km off: dropped
    "Coimbatore Medical College Hospital (CMCH)", "PSG Hospitals Peelamedu", "Ganga Hospital Ramnagar",
    "GKNM Hospital Pappanaickenpalayam", "KG Hospital Arts College Road", "Coimbatore Junction Railway Station",
    "Gandhipuram Central Bus Stand", "Ukkadam Bus Stand", "Singanallur Bus Stand", "Coimbatore International Airport (CJB)",
    "Tidel Park Coimbatore", "Avinashi Road G.D. Naidu Expressway Entrance", "Lakshmi Mills Junction",
    "Hope College Junction Peelamedu", "RS Puram DB Road", "Saibaba Colony Junction", "Saravanampatti Tech Corridor",
    "Town Hall Heritage Commercial Core", "Singanallur Lake Boat House", "Periyakulam Lake Ukkadam",
}
TIER3_WORDS = re.compile(r"dental|dentist|eye|netra|vision|ortho|physio|clinic|nursing|maternity|scan|lab|pharma|pharmacy|blood|"
                         r"health cent|polyclinic|poly clinic|diagnos|skin|ivf|fertility|homeo|siddha|ayur|vaidya|veterinary|cosmetic|hair|de-?addiction", re.I)


def hospital_tier(name: str, kind: str, tier1_patterns: list[str]) -> int:
    if kind == "clinic" or TIER3_WORDS.search(name):
        return 3
    if any(re.search(p, name, re.I) for p in tier1_patterns):
        return 1
    return 2


def population_proxy(g: CSRGraph, total: float, weights: dict, cell_m: float = 250.0) -> np.ndarray:
    """Per-node population from OSM road density (metres of residential-type road), smoothed over 3x3 cells,
    scaled to ``total``. This is a PROXY: it follows where roads (hence buildings) are, not a census grid."""
    cw = np.array([weights.get(n, 0.3) for n in ROAD_CLASS_NAMES], dtype=np.float64)
    w = np.bincount(g.edge_tails, weights=g.length_m * cw[g.road_class], minlength=g.n_nodes)
    proj = LocalProjection(float(g.lat.mean()), float(g.lon.mean()))
    x, y = proj.to_xy(g.lat, g.lon)
    cx, cy = np.floor(x / cell_m).astype(np.int64), np.floor(y / cell_m).astype(np.int64)
    cx -= cx.min(); cy -= cy.min()
    nx, ny = cx.max() + 3, cy.max() + 3
    grid = np.zeros((ny, nx))
    np.add.at(grid, (cy + 1, cx + 1), w)
    sm = np.zeros_like(grid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sm[1:-1, 1:-1] += grid[1 + dy:ny - 1 + dy, 1 + dx:nx - 1 + dx]
    sm /= 9.0
    cell_sum = grid[cy + 1, cx + 1]
    dens = sm[cy + 1, cx + 1]
    share = np.divide(w, cell_sum, out=np.zeros_like(w), where=cell_sum > 0)
    node = share * dens
    return node * (total / node.sum())


def snap_to_giant(g, giant, nodes_xy_snapper: NodeSnapper, lat: float, lon: float) -> tuple[int, float]:
    n, d = nodes_xy_snapper.snap(lat, lon)
    return int(n), float(d)


def main(data_dir: str = "data/coimbatore", raw_roads: str = "data/raw/raw_roads.json") -> None:
    A = load_assumptions()
    d = ROOT / data_dir
    out = d / "derived"
    out.mkdir(parents=True, exist_ok=True)
    g = CSRGraph.load(d / "graph.npz")
    comp, _ = strongly_connected_components(g)
    giant = comp == int(np.argmax(np.bincount(comp)))
    gi = np.flatnonzero(giant)
    snapper = NodeSnapper(g.lat[gi], g.lon[gi])          # snaps ONLY to nodes inside the giant SCC

    def snap(lat, lon):
        j, dist = snapper.snap(lat, lon)
        return int(gi[j]), float(dist)

    # ---- street names and named structures -----------------------------------------------------
    names: dict[int, str] = {}
    rp = ROOT / raw_roads
    if rp.exists():
        for e in json.loads(rp.read_text(encoding="utf-8"))["elements"]:
            if e.get("type") == "way" and "name" in e.get("tags", {}):
                names[int(e["id"])] = e["tags"].get("name:en") or e["tags"]["name"]
    used = set(g.way_id.tolist())
    names = {k: v for k, v in names.items() if k in used}
    (out / "way_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")

    struct_mask = ((g.flags & (FLAG_BRIDGE | FLAG_FLYOVER)) != 0) & g.is_arterial
    groups: dict[str, list[int]] = {}
    for e in np.flatnonzero(struct_mask):
        nm = names.get(int(g.way_id[e]))
        if nm:
            groups.setdefault(nm, []).append(int(e))
    structures = []
    for nm, es in groups.items():
        es = np.array(es)
        length = float(g.length_m[es].sum() / 2.0)         # both directions counted
        if length < 60:
            continue
        mid = es[len(es) // 2]
        structures.append({"name": nm, "kind": "flyover" if (g.flags[es] & FLAG_FLYOVER).any() else "bridge",
                           "edges": es.tolist(), "length_m": round(length), "lat": float(g.lat[g.edge_tails[mid]]),
                           "lon": float(g.lon[g.edge_tails[mid]])})
    structures.sort(key=lambda s: -s["length_m"])
    (out / "structures.json").write_text(json.dumps(structures), encoding="utf-8")

    # ---- facilities ------------------------------------------------------------------------------
    places = [p for p in json.loads((d / "places.json").read_text(encoding="utf-8")) if p["name"] not in TYPED_LANDMARKS]
    tier1 = A["hospitals"]["tier1_patterns"]
    facilities, landmarks = [], []

    def add_fac(name, kind, tier, lat, lon, source, approx=False, note=""):
        node, dist = snap(lat, lon)
        facilities.append({"id": len(facilities), "name": name, "kind": kind, "tier": tier, "node": node, "lat": lat, "lon": lon,
                           "snap_m": round(dist), "source": source, "approx": approx, "note": note})

    for p in places:
        k, nm = p["kind"], p["name"]
        if k in ("hospital", "clinic"):
            if re.search(r"blood bank", nm, re.I):
                continue
            if nm == "Coimbatore Medical College Hospital":               # OSM point is the Avinashi Road campus
                add_fac("Coimbatore Medical College (Avinashi Rd campus)", "hospital", 2, p["lat"], p["lon"], "osm")
                continue
            add_fac(nm, "hospital", hospital_tier(nm, k, tier1), p["lat"], p["lon"], "osm")
        elif k == "fire_station":
            add_fac(nm, "fire_station", None, p["lat"], p["lon"], "osm")
        elif k == "police":
            add_fac(nm, "police", None, p["lat"], p["lon"], "osm")
    for fn, src in (("facilities_manual.csv", "manual"), ("facilities_extra.csv", "user")):
        fp = d / fn
        if not fp.exists():
            continue
        rows = [r for r in csv.DictReader(l for l in fp.read_text(encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#"))]
        for r in rows:
            lat, lon, approx = r.get("lat"), r.get("lon"), r.get("approx") == "1"
            if not lat and r.get("locality"):
                # no coordinates known: place the facility at the centre of the named OSM locality (approximate by construction)
                hit = next((p for p in places if p["kind"] in ("suburb", "neighbourhood", "town") and p["name"].lower() == r["locality"].strip().lower()), None)
                if hit is None:
                    print(f"  skipped {r['name']}: locality '{r['locality']}' not found in OSM places")
                    continue
                lat, lon, approx = hit["lat"], hit["lon"], True
            if not lat:
                continue
            add_fac(r["name"], r["kind"], int(r["tier"]) if r.get("tier") else None, float(lat), float(lon), src,
                    approx=approx, note=r.get("source", ""))
    # de-duplicate facilities of the same kind within 120 m with the same first word (OSM + user duplicates)
    keep, seen = [], []
    for f in facilities:
        dup = any(f["kind"] == s["kind"] and f["name"].split()[0].lower() == s["name"].split()[0].lower()
                  and abs(f["lat"] - s["lat"]) < 0.0011 and abs(f["lon"] - s["lon"]) < 0.0011 for s in seen)
        if not dup:
            keep.append(f); seen.append(f)
    for i, f in enumerate(keep):
        f["id"] = i
    (out / "facilities.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")

    # ---- landmarks for pickers (OSM only, named) -------------------------------------------------
    for p in places:
        if p["kind"] in ("suburb", "neighbourhood", "town", "station", "bus_station", "police", "fire_station"):
            node, dist = snap(p["lat"], p["lon"])
            landmarks.append({"name": p["name"], "kind": p["kind"], "lat": p["lat"], "lon": p["lon"], "node": node})
    for f in keep:
        if f["kind"] == "hospital" and f["tier"] in (1, 2):
            landmarks.append({"name": f["name"], "kind": "hospital", "lat": f["lat"], "lon": f["lon"], "node": f["node"]})
    for s in structures[:40]:
        node, _ = snap(s["lat"], s["lon"])
        landmarks.append({"name": s["name"], "kind": s["kind"], "lat": s["lat"], "lon": s["lon"], "node": node})
    seen_names, lm2 = set(), []
    for l in landmarks:
        key = (l["name"].lower(), l["kind"])
        if key not in seen_names and l["name"].strip():
            seen_names.add(key); lm2.append(l)
    (out / "landmarks.json").write_text(json.dumps(lm2, ensure_ascii=False), encoding="utf-8")

    # ---- population proxy ------------------------------------------------------------------------
    pop = population_proxy(g, float(A["population"]["total"]), A["population"]["class_weights"])
    np.save(out / "node_pop.npy", pop.astype(np.float32))
    np.save(out / "giant_mask.npy", giant)

    cnt = {"hospital_t1": sum(1 for f in keep if f["kind"] == "hospital" and f["tier"] == 1),
           "hospital_t2": sum(1 for f in keep if f["kind"] == "hospital" and f["tier"] == 2),
           "hospital_t3": sum(1 for f in keep if f["kind"] == "hospital" and f["tier"] == 3),
           "fire_station": sum(1 for f in keep if f["kind"] == "fire_station"),
           "police": sum(1 for f in keep if f["kind"] == "police"),
           "manual": sum(1 for f in keep if f["source"] == "manual")}
    meta = {"nodes": g.n_nodes, "arcs": g.n_edges, "giant_scc_share": float(giant.mean()), "population_total": float(pop.sum()),
            "population_outside_giant": float(pop[~giant].sum()), "facilities": cnt, "named_ways": len(names),
            "named_structures": len(structures), "landmarks": len(lm2),
            "population_source": "OSM road-density proxy scaled to the Corporation estimate (Budget 2023-24, ~22 lakh); not a census grid",
            "dropped_typed_landmarks": len(TYPED_LANDMARKS)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
