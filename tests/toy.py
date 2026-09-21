"""A small deterministic toy city (grid with two arterials, a river, hospitals) for fast tests."""
from __future__ import annotations

import math

import numpy as np

from sim.city import CityData, Facility
from store.csr_graph import FLAG_BRIDGE, FLAG_FLYOVER, ROAD_CLASS_CODE, CSRGraph


def toy_city(w: int = 21, h: int = 21, spacing: float = 250.0, lat0: float = 10.99, lon0: float = 76.94) -> CityData:
    dlat = spacing / 110_540.0
    dlon = spacing / (111_320.0 * math.cos(math.radians(lat0)))
    mid = w // 2
    tails, heads, tt, ln, rc, lanes, rev, flags, wid = [], [], [], [], [], [], [], [], []

    def nid(x, y):
        return y * w + x

    def add(a, b, art, flag=0, way=0):
        speed = 45.0 if art else 20.0
        i = len(tails)
        for u, v in ((a, b), (b, a)):
            tails.append(u); heads.append(v); tt.append(spacing / (speed / 3.6)); ln.append(spacing)
            rc.append(ROAD_CLASS_CODE["primary" if art else "residential"]); lanes.append(2 if art else 1); flags.append(flag); wid.append(way)
        rev.extend([i + 1, i])

    way = 1
    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                art = y == mid
                bridge = FLAG_BRIDGE | FLAG_FLYOVER if (art and x in (5, 6)) else 0          # a flyover on the east-west arterial
                add(nid(x, y), nid(x + 1, y), art, bridge, 100 + y if art else way)
            if y + 1 < h:
                add(nid(x, y), nid(x, y + 1), x == mid, 0, 200 + x if x == mid else way)
            way += 1
    g = CSRGraph.from_edges(w * h, tails, heads, tt, length_m=ln, road_class=rc, lanes_dir=lanes, flags=flags, way_id=wid, rev=rev,
                            lat=[lat0 + y * dlat for y in range(h) for _ in range(w)], lon=[lon0 + x * dlon for _ in range(h) for x in range(w)])
    pop = np.full(g.n_nodes, 2000000 / g.n_nodes)
    fac = [Facility(0, "Central Hospital", "hospital", 1, nid(mid, mid), g.lat[nid(mid, mid)], g.lon[nid(mid, mid)]),
           Facility(1, "North Hospital", "hospital", 1, nid(mid, h - 3), g.lat[nid(mid, h - 3)], g.lon[nid(mid, h - 3)]),
           Facility(2, "East General", "hospital", 2, nid(w - 3, mid + 2), g.lat[nid(w - 3, mid + 2)], g.lon[nid(w - 3, mid + 2)]),
           Facility(3, "West General", "hospital", 2, nid(2, mid - 3), g.lat[nid(2, mid - 3)], g.lon[nid(2, mid - 3)]),
           Facility(4, "Clinic", "hospital", 3, nid(4, 4), g.lat[nid(4, 4)], g.lon[nid(4, 4)]),
           Facility(5, "Central Fire", "fire_station", None, nid(mid + 1, mid + 1), g.lat[nid(mid + 1, mid + 1)], g.lon[nid(mid + 1, mid + 1)]),
           Facility(6, "South Police", "police", None, nid(mid, 3), g.lat[nid(mid, 3)], g.lon[nid(mid, 3)]),
           Facility(7, "West Police", "police", None, nid(3, mid), g.lat[nid(3, mid)], g.lon[nid(3, mid)])]
    lms = [{"name": "Town Centre", "kind": "suburb", "lat": float(g.lat[nid(mid, mid)]), "lon": float(g.lon[nid(mid, mid)]), "node": nid(mid, mid)},
           {"name": "North Suburb", "kind": "suburb", "lat": float(g.lat[nid(mid, h - 3)]), "lon": float(g.lon[nid(mid, h - 3)]), "node": nid(mid, h - 3)}]
    river = np.array([[lat0 + 4 * dlat, lon0 + x * dlon] for x in range(0, w)])            # river along y=4
    fly = [int(e) for e in np.flatnonzero(g.flags & FLAG_FLYOVER)]
    struct = [{"name": "Test Flyover", "kind": "flyover", "edges": fly,
               "length_m": 500, "lat": float(g.lat[nid(5, mid)]), "lon": float(g.lon[nid(5, mid)])}]
    return CityData("Toyville", g, pop, np.ones(g.n_nodes, dtype=bool), fac, lms, {100 + mid: "Main Street", 200 + mid: "Central Avenue"},
                    struct, {"Toy River": river}, centre=(float(g.lat[nid(mid, mid)]), float(g.lon[nid(mid, mid)])), meta={})
