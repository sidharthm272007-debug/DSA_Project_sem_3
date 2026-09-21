"""Map layers: what the browser is sent must be readable, not just valid."""
from viz.layers import _fill_alpha, hazard_layers


def test_large_footprints_do_not_paint_the_whole_map():
    quake = [{"type": "circle", "kind": "quake", "level": lv, "lat": 11.0, "lon": 76.96, "radius_m": r} for lv, r in ((3, 2700), (2, 4950), (1, 7650))]
    layers = hazard_layers(quake)
    assert len(layers) == 1 and len(layers[0].data) == 3
    total_alpha = sum(_fill_alpha(p["level"], p["radius_m"]) for p in quake)
    assert total_alpha < 80                                   # stacked fills stay translucent (previously 240 of 255)
    assert all(_fill_alpha(1, r) <= _fill_alpha(1, 500) for r in (2000, 8000))
    assert _fill_alpha(3, 160) >= 100                          # a small collapse site stays clearly visible


def test_no_unit_or_enum_string_reaches_the_browser_as_a_javascript_expression():
    """pydeck sends bare strings as JS expressions ('@@=pixels'), which silently breaks widths and radii. Enumerations must be quoted."""
    import json

    import numpy as np

    from viz import layers as L

    rows = [{"lon": 76.9, "lat": 11.0, "fill": [1, 2, 3, 4], "r": 5, "tip": "x", "text": "a"}]
    paths = [{"path": [[76.9, 11.0], [76.91, 11.01]], "tip": "r"}]
    built = [L.road_layer(paths, "#ffffff", 3, 200, "a"), L.dots(rows, "b"), *L.glow_roads(paths, "#ff0000", "c"),
             *L.hazard_layers([{"type": "circle", "kind": "fire", "level": 1, "lat": 11.0, "lon": 76.9, "radius_m": 500}]),
             L.columns(np.array([11.0]), np.array([76.9]), np.array([10.0]), "d"), L.labels(rows, "e"),
             *L.link_layers([{"a": (11.0, 76.9), "b": (11.01, 76.91), "tone": "good", "label": "x"}]),
             *L.pick_layers([{"lat": 11.0, "lon": 76.9, "n": 1, "tone": "good", "tip": "t"}]), L.trips([{"path": paths[0]["path"], "ts": [0, 1], "color": [1, 2, 3, 255]}], 1.0)]
    for layer in built:
        spec = json.loads(layer.to_json()) if hasattr(layer, "to_json") else None
        text = json.dumps(spec)
        for bad in ("@@=pixels", "@@=meters", "@@=common"):
            assert bad not in text, (layer.id, bad)


def test_decks_are_sent_compactly_and_round_trip_to_the_same_content():
    import json

    import pydeck as pdk

    from viz import layers as L

    paths = [{"path": [[76.9 + i * 1e-3, 11.0], [76.91 + i * 1e-3, 11.01]], "tip": "r"} for i in range(200)]
    d = L.deck([L.road_layer(paths, "#ffffff", 3, 200, "a")], L.view(11.0, 76.9))
    compact = d.to_json()
    pretty = pdk.Deck.to_json(d)
    assert len(compact) < 0.6 * len(pretty) and "\n" not in compact
    assert json.loads(compact) == json.loads(pretty)                       # same content, less whitespace


def test_light_mode_draws_flat_cheap_geometry_and_rich_mode_draws_3d():
    """Light is the default in the app because some laptops cannot draw thousands of extruded columns (screen goes blank / browser freezes)."""
    import json

    import numpy as np

    from viz import layers as L

    lat, lon, val = np.array([11.0, 11.01]), np.array([76.9, 76.91]), np.array([10.0, 50.0])
    paths = [{"path": [[76.9, 11.0], [76.91, 11.01]], "tip": "r"}]
    links = [{"a": (11.0, 76.9), "b": (11.01, 76.91), "tone": "good", "label": "x"}]
    old = L.mode_provider
    try:
        L.mode_provider = lambda: "light"
        cols = json.loads(L.columns(lat, lon, val, "c").to_json())
        assert cols["@@type"] == "ScatterplotLayer" and "extruded" not in cols
        assert L.deck([], L.view(11.0, 76.9, 11, 55, -20)).initial_view_state.pitch == 0
        assert L.labels([{"lon": 76.9, "lat": 11.0, "text": "a"}], "t") is None
        assert [json.loads(x.to_json())["@@type"] for x in L.link_layers(links)] == ["PathLayer"]
        road = json.loads(L.road_layer(paths, "#ffffff", 3, 200, "r").to_json())
        assert not road.get("capRounded") and not road.get("jointRounded")
        L.mode_provider = lambda: "rich"
        assert json.loads(L.columns(lat, lon, val, "c").to_json())["@@type"] == "ColumnLayer"
        assert L.deck([], L.view(11.0, 76.9, 11, 55, -20)).initial_view_state.pitch == 55
        assert "ArcLayer" in [json.loads(x.to_json())["@@type"] for x in L.link_layers(links)]
    finally:
        L.mode_provider = old
