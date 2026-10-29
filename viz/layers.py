"""pydeck layer builders. Pure functions: data in, layer out. No Streamlit here, so they can be tested and reused."""
from __future__ import annotations

import json
import math

import numpy as np
import pydeck as pdk
from pydeck.bindings.json_tools import default_serialize

from store.csr_graph import ROAD_CLASS_NAMES
from viz.theme import HAZARD_COLOR, PAL, TONE, hex_rgb

LEVEL_ALPHA = {1: 45, 2: 75, 3: 120}

# How much the browser has to draw. The UI replaces this with the user's choice ("light", "rich" or "off").
# light: flat 2-D, no extrusion, no rounded joins, no text labels. rich: the 3-D look. Tests and scripts get "rich".
mode_provider = lambda: "rich"  # noqa: E731


def mode() -> str:
    return mode_provider()


def view(lat: float, lon: float, zoom: float = 10.8, pitch: float = 48, bearing: float = -12) -> pdk.ViewState:
    return pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=pitch, bearing=bearing)


class CompactDeck(pdk.Deck):
    """pydeck pretty-prints its JSON (indent=2), which is about three times larger than it needs to be. Every map, and every animation tick, is sent to the browser as this JSON."""

    def to_json(self) -> str:
        return json.dumps(self, sort_keys=True, default=default_serialize, separators=(",", ":"))


def deck(layers: list, vs: pdk.ViewState, tooltip: dict | None = None) -> pdk.Deck:
    if mode() != "rich":                                                       # flat, top-down: far cheaper for a weak or software-rendered graphics stack
        vs = pdk.ViewState(latitude=vs.latitude, longitude=vs.longitude, zoom=vs.zoom, pitch=0, bearing=0)
    return CompactDeck(layers=[l for l in layers if l is not None], initial_view_state=vs, map_provider="carto", map_style="dark",
                       tooltip=tooltip or {"html": "{tip}", "style": {"backgroundColor": PAL["bg"], "color": PAL["text"], "fontSize": "12px", "border": f'1px solid {PAL["line"]}'}})


# ---- roads -----------------------------------------------------------------------------------------------
def poly(city, e: int) -> list[list[float]]:
    """Edge geometry as [lon, lat] pairs rounded to 5 decimals (~1 m): halves what is sent to the browser."""
    return np.round(city.edge_polyline(int(e)).astype(np.float64), 5).tolist()


def edge_paths(city, edges, cap: int = 7000, tip: str = "") -> list[dict]:
    edges = np.asarray(edges, dtype=np.int64)
    if len(edges) > cap:
        edges = edges[np.linspace(0, len(edges) - 1, cap).astype(np.int64)]
    return [{"path": poly(city, e), "tip": tip or city.edge_name(int(e))} for e in edges]


def simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker: drop vertices that lie within ``tol`` degrees of the straight line between their neighbours."""
    n = len(pts)
    if n <= 2:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, seg = pts[i], pts[j] - pts[i]
        length = float(np.hypot(*seg))
        q = pts[i + 1:j] - a
        d = np.abs(seg[0] * q[:, 1] - seg[1] * q[:, 0]) / length if length > 0 else np.hypot(q[:, 0], q[:, 1])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def base_network(city, classes=("trunk", "primary", "secondary")) -> list[dict]:
    """Trunk, primary and secondary roads, one direction per two-way pair: the context every map sits on.
    Simplified (about 9 m) and rounded to ~11 m: it is only a backdrop, and it is re-sent to the browser on every animation tick."""
    g = city.g
    keep = np.isin(g.road_class, [ROAD_CLASS_NAMES.index(k) for k in classes if k in ROAD_CLASS_NAMES])
    first = (g.rev < 0) | (np.arange(g.n_edges) < g.rev)
    idx = np.flatnonzero(keep & first)
    out = []
    for e in idx:
        pts = simplify(city.edge_polyline(int(e)).astype(np.float64), 8e-5)
        out.append({"path": np.round(pts, 4).tolist(), "tip": city.way_names.get(int(g.way_id[e]), "")})
    return out


def road_layer(paths: list[dict], color: str, width: float = 2.0, alpha: int = 255, id: str = "roads", pickable: bool = False, min_px: float = 1.0):
    if not paths:
        return None
    return pdk.Layer("PathLayer", data=paths, id=id, get_path="path", get_color=hex_rgb(color, alpha), get_width=width, width_units="'pixels'",
                     width_min_pixels=min_px, cap_rounded=(mode() == "rich"), joint_rounded=(mode() == "rich"), pickable=pickable, auto_highlight=(pickable and mode() == "rich"))


def glow_roads(paths: list[dict], color: str, id: str) -> list:
    """A wide translucent stroke under a bright thin one: roads that look lit."""
    return [road_layer(paths, color, 9, 55, id + "_glow"), road_layer(paths, color, 3, 255, id, pickable=True)]


# ---- hazards ---------------------------------------------------------------------------------------------
def _fill_alpha(level: int, radius_m: float | None) -> int:
    """Big footprints (an earthquake's felt zone is kilometres wide) must not paint the whole map: fainter fill the larger they are, ring kept."""
    a = LEVEL_ALPHA.get(level, 60)
    if radius_m is None or radius_m <= 1000:
        return a
    return max(6, int(a * (0.35 if radius_m <= 3000 else 0.2)))


def hazard_layers(prims: list[dict], id: str = "haz") -> list:
    circles, polys = [], []
    for p in prims:
        hexcol = HAZARD_COLOR.get(p.get("kind", "custom"), HAZARD_COLOR["custom"])
        line = hex_rgb(hexcol, 190)
        if p["type"] == "circle":
            circles.append({"lat": p["lat"], "lon": p["lon"], "r": p["radius_m"], "fill": hex_rgb(hexcol, _fill_alpha(p.get("level", 1), p["radius_m"])), "line": line,
                            "tip": f"{p.get('kind', '').title()} zone (level {p.get('level', 1)})"})
        else:
            polys.append({"polygon": p["coords"], "fill": hex_rgb(hexcol, _fill_alpha(p.get("level", 1), None)), "line": line, "tip": f"{p.get('kind', '').title()} zone (level {p.get('level', 1)})"})
    out = []
    if circles:
        out.append(pdk.Layer("ScatterplotLayer", data=circles, id=id + "_c", get_position=["lon", "lat"], get_radius="r", radius_units="'meters'", get_fill_color="fill",
                             get_line_color="line", stroked=True, line_width_min_pixels=1.5, pickable=True))
    if polys:
        out.append(pdk.Layer("PolygonLayer", data=polys, id=id + "_p", get_polygon="polygon", get_fill_color="fill", get_line_color="line", line_width_min_pixels=1.5,
                             stroked=True, filled=True, pickable=True))
    return out


# ---- people, facilities, incidents -------------------------------------------------------------------------
def columns(lat, lon, value, id: str, color_hi: str = PAL["red"], color_lo: str = PAL["red_tint"], radius: float = 230.0, scale: float = 1.0, tip_fmt: str = "{:,.0f} people",
            max_h: float = 1800.0, pickable: bool = True):
    value = np.asarray(value, dtype=float)
    if not len(value):
        return None
    vmax = float(value.max()) or 1.0
    lo, hi = np.array(hex_rgb(color_lo)[:3], float), np.array(hex_rgb(color_hi)[:3], float)
    t = (value / vmax) ** 0.6
    if mode() != "rich":                                                       # flat dots sized and coloured by value: same information, no 3-D geometry
        rows = [{"lon": float(o), "lat": float(a), "r": float(radius * (0.45 + 0.75 * tt)), "col": [int(x) for x in (lo + (hi - lo) * tt)] + [175], "tip": tip_fmt.format(v)}
                for a, o, v, tt in zip(lat, lon, value, t)]
        return pdk.Layer("ScatterplotLayer", data=rows, id=id, get_position=["lon", "lat"], get_radius="r", radius_units="'meters'", get_fill_color="col", stroked=False, pickable=pickable)
    rows = [{"lon": float(o), "lat": float(a), "h": float(v / vmax * max_h * scale), "col": [int(x) for x in (lo + (hi - lo) * tt)] + [205], "tip": tip_fmt.format(v)}
            for a, o, v, tt in zip(lat, lon, value, t)]
    return pdk.Layer("ColumnLayer", data=rows, id=id, get_position=["lon", "lat"], get_elevation="h", elevation_scale=1, radius=radius, get_fill_color="col",
                     extruded=True, pickable=pickable, auto_highlight=True, coverage=0.9)


def dots(rows: list[dict], id: str, radius_px: float = 6, stroke: bool = True, pickable: bool = True):
    if not rows:
        return None
    return pdk.Layer("ScatterplotLayer", data=rows, id=id, get_position=["lon", "lat"], get_fill_color="fill", get_line_color=[255, 255, 255, 235],
                     get_radius="r", radius_units="'pixels'", radius_min_pixels=radius_px * 0.6, stroked=stroke, line_width_min_pixels=1.4, pickable=pickable, billboard=True, parameters={"depthTest": False})


def _mercator_px(lat: float, lon: float, zoom: float) -> tuple[float, float]:
    """Standard Web Mercator projection to pixel space at a given zoom (the same one every slippy map, incl. deck.gl, uses).
    Approximate, but that is fine here: it only has to rank which labels would collide on screen, not render anything itself."""
    scale = 256.0 * (2.0 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    siny = min(max(math.sin(math.radians(lat)), -0.9999), 0.9999)
    y = (0.5 - math.log((1.0 + siny) / (1.0 - siny)) / (4.0 * math.pi)) * scale
    return x, y


def declutter_labels(rows: list[dict], zoom: float, font_px: int = 13, y_offset_px: int = 22, pad_px: float = 4.0) -> list[dict]:
    """Greedy label placement: keep a label only if its estimated screen box does not overlap one already kept.

    Map label placement is a classic, NP-hard-in-general computational-geometry problem (deciding whether even a FULL
    non-overlapping assignment exists is itself NP-hard); the standard practical answer is a priority-ordered greedy
    placement, which is what this is. ``rows`` should already be in priority order (most important label first) --
    this function only ever discards a label for colliding with one kept earlier, it never reorders or moves text, so
    the input order IS the priority. Width is estimated from character count (no font metrics available server-side);
    generous enough that it only ever under-places (skips a label that would actually have fit), never overlaps.
    """
    boxes: list[tuple[float, float, float, float]] = []       # accepted (x0, y0, x1, y1) in pixel space at `zoom`
    kept = []
    for r in rows:
        x, y = _mercator_px(r["lat"], r["lon"], zoom)
        w = len(r["text"]) * font_px * 0.62 + 10.0             # average glyph advance for a condensed sans at this size, plus the outline
        h = font_px * 1.35
        cy = y - y_offset_px                                   # TextLayer's get_pixel_offset shifts the label above the anchor point
        box = (x - w / 2 - pad_px, cy - h / 2 - pad_px, x + w / 2 + pad_px, cy + h / 2 + pad_px)
        if not any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in boxes):
            boxes.append(box)
            kept.append(r)
    return kept


def labels(rows: list[dict], id: str, size: int = 15, zoom: float = 11.0, declutter: bool = True):
    if not rows or mode() != "rich":                                           # text is the costliest thing to draw; names stay available on hover
        return None
    if declutter:
        rows = declutter_labels(rows, zoom, font_px=size)
        if not rows:
            return None
    return pdk.Layer("TextLayer", data=rows, id=id, get_position=["lon", "lat"], get_text="text", get_size=size, get_color=[248, 250, 252, 255],
                     get_pixel_offset=[0, -22], font_weight=700, outline_width=3, outline_color=[10, 16, 32, 255], billboard=True, parameters={"depthTest": False})


def hospital_rows(city, closed: set[int], ratio: dict[int, float | None], tiers=(1,)) -> list[dict]:
    rows = []
    for f in city.hospitals(2):
        if f.tier not in tiers:
            continue
        r = ratio.get(f.id)
        if f.id in closed:
            col, state = hex_rgb(PAL["red"]), "OUT OF SERVICE"
        elif r is not None and r < 0.7:
            col, state = hex_rgb(PAL["amber"]), f"cut off: reaches {r * 100:.0f}% of its normal catchment"
        else:
            col, state = hex_rgb(PAL["accent"]), "operating" + (f", {r * 100:.0f}% of normal catchment" if r is not None else "")
        rows.append({"lon": f.lon, "lat": f.lat, "fill": col, "r": 11 if f.tier == 1 else 6, "tip": f"{f.name} (tier {f.tier}): {state}", "text": (f.name.split("(")[0].replace(" Hospitals", "").replace(" Hospital", "").strip()[:20]) if f.tier == 1 else ""})
    return rows


INC_COLOR = {"pending": PAL["red"], "responding": PAL["amber"], "onscene": PAL["blue"], "transport": PAL["violet"], "done": PAL["green"]}


def incident_rows(city, incidents, t: float) -> list[dict]:
    rows = []
    for i in incidents:
        end = max([x for x in (i.t_done, i.t_delivered) if x is not None], default=None) if i.state == "done" else None
        if i.t > t or (end is not None and t > end + 8):
            continue
        if end is not None and t >= end:
            st = "done"
        elif i.t_arrive is not None and t >= i.t_arrive:
            st = "onscene"
        elif i.t_assign is not None and t >= i.t_assign:
            st = "responding"
        else:
            st = "pending"
        rows.append({"lon": float(city.g.lon[i.node]), "lat": float(city.g.lat[i.node]), "fill": hex_rgb(INC_COLOR[st]), "r": 4.5 + 1.5 * i.severity,
                     "tip": f"{i.kind} (severity {i.severity}), {st}"})
    return rows


def trips(trip_rows: list[dict], current_time: float, trail: float = 7.0, id: str = "trips"):
    if not trip_rows:
        return None
    return pdk.Layer("TripsLayer", data=trip_rows, id=id, get_path="path", get_timestamps="ts", get_color="color", opacity=0.95, width_min_pixels=4, rounded=True,
                     trail_length=trail, current_time=current_time)


def pick_layers(rows: list[dict], id: str = "picks") -> list:
    """rows: {lat, lon, n, tone, tip}. Numbered pins with a halo."""
    if not rows:
        return []
    halo = [{**r, "fill": hex_rgb(TONE[r["tone"]][1], 80), "r": 34} for r in rows]
    core = [{**r, "fill": hex_rgb(TONE[r["tone"]][1], 255), "r": 16} for r in rows]
    txt = [{**r, "text": str(r["n"])} for r in rows]
    return [dots(halo, id + "_halo", 20, stroke=False, pickable=False), dots(core, id + "_core", 10),
            pdk.Layer("TextLayer", data=txt, id=id + "_txt", get_position=["lon", "lat"], get_text="text", get_size=20, get_color=[8, 12, 24, 255], font_weight=800, billboard=True)]
