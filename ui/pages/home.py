import numpy as np
import streamlit as st

from sim.report import fmt_people
from ui.components import feature, kpi_row, legend, page_header, section, show_map
from ui.state import base_paths, core, visuals
from viz import layers as L
from viz.theme import PAL, hex_rgb, mi

A, city, _ = core()
g = city.g

page_header("location_city", "Overview",
            "Disaster-response routing and simulation for Coimbatore's real road network: the fastest and safest way through right now, and a minute-by-minute simulation of how the city responds when roads fail.")

r1, r2 = st.columns(2, gap="large")
with r1:
    st.markdown('<div class="card tall" style="min-height: 12.5rem"><h4>'
                + mi("route") + ' Best route now</h4>'
                '<p>Pick what has happened and where you need to go. Get the route that fits how careful you want to be, the alternatives it beat and why, the nearest '
                'hospital that is really reachable, and the roads the city is leaning on. Add the closures you know about and it is taken into account.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.page_link("ui/pages/route.py", label="Best route now", icon=":material/route:")
with r2:
    st.markdown('<div class="card tall" style="min-height: 12.5rem"><h4>'
                + mi("storm") + ' Disaster lab</h4>'
                '<p>Combine any hazards — earthquake, fire, gas leak, flood, bridge failure, storm, or a custom event — set the hour and the weather, and run a '
                'minute-by-minute simulation: damage, dispatch, re-routing, hospital outages, debris clearing.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.page_link("ui/pages/lab.py", label="Disaster lab", icon=":material/storm:")

st.write("")
n_h = len(city.hospitals(2))
n_fire = len(city.of_kind("fire_station"))
kpi_row([
    ("People modelled", fmt_people(city.total_pop), "OSM road-density proxy scaled to the city estimate"),
    ("Road network", f"{g.length_m.sum() / 2000:,.0f} km", f"{g.n_nodes:,} junctions"),
    ("Hospitals mapped", str(n_h), f"{len(city.hospitals(1))} major, rest general"),
    ("Fire stations mapped", str(n_fire), "far fewer than the real city: see Data & assumptions", PAL["amber"] if n_fire < A["standards"]["min_fire_stations"] else None),
    ("Named flyovers / bridges", str(len(city.structures)), "from OpenStreetMap"),
])

st.write("")
section("insights", "How the route recommendation is made")
h1, h2, h3, h4 = st.columns(4, gap="large")
for col, ic, t, d in ((h1, "troubleshoot", "Measure the damage", "Hazards close and slow real roads and put hospitals out of service; each stretch of road gets a risk score from what happened around it."),
                     (h2, "alt_route", "Search genuinely different routes", "Several routes are generated and scored on the damaged network with exact shortest-path maths, not looked up."),
                     (h3, "tune", "Choose by your priorities", "How careful you are on the road: fastest, balanced or cautious. Change it and the recommended route changes."),
                     (h4, "fact_check", "Explain and admit limits", "Each answer lists the numbers behind it and the data it rests on: what is measured, approximate or missing.")):
    col.markdown(feature(ic, t, d), unsafe_allow_html=True)

st.write("")
section("map", "The city at a glance")
legend([(PAL["red"], "where people live (bigger and redder = more people per square kilometre)" if visuals() != "rich" else "where people live (column height = people per 500 m cell)"),
        (PAL["accent"], "major hospital"), (PAL["muted"], "other hospital"), (PAL["orange"], "fire station")])


@st.cache_resource(show_spinner=False)
def overview_deck(detail: str):
    cells = np.flatnonzero(city.cell_pop > 30)
    if detail == "rich":
        cols = L.columns(city.cell_lat[cells], city.cell_lon[cells], city.cell_pop[cells], "pop", color_hi=PAL["red"], color_lo=PAL["red_tint"], radius=200, max_h=1700)
    else:                                              # light: merge 500 m cells into ~1 km cells, so a weak graphics card draws a quarter of the shapes
        kl, ko = np.round(city.cell_lat[cells] / 0.009).astype(np.int64), np.round(city.cell_lon[cells] / 0.0092).astype(np.int64)
        key = kl * 100000 + ko
        uk, inv = np.unique(key, return_inverse=True)
        pop = np.bincount(inv, weights=city.cell_pop[cells])
        lat = np.bincount(inv, weights=city.cell_lat[cells] * city.cell_pop[cells]) / np.maximum(pop, 1e-9)
        lon = np.bincount(inv, weights=city.cell_lon[cells] * city.cell_pop[cells]) / np.maximum(pop, 1e-9)
        cols = L.columns(lat, lon, pop, "pop", color_hi=PAL["red"], color_lo=PAL["red_tint"], radius=560, tip_fmt="{:,.0f} people in about 1 km²", pickable=False)
    hosp = [{"lon": f.lon, "lat": f.lat, "fill": hex_rgb(PAL["accent"]) if f.tier == 1 else hex_rgb(PAL["muted"], 200), "r": 9 if f.tier == 1 else 4, "tip": f.name}
            for f in city.hospitals(2)]
    fire = [{"lon": f.lon, "lat": f.lat, "fill": hex_rgb(PAL["orange"]), "r": 9, "tip": f.name} for f in city.of_kind("fire_station")]
    layers = [L.road_layer(base_paths(), PAL["line2"], 1.2, 150, "base"), cols, L.dots(hosp, "hosp"), L.dots(fire, "fire"),
              L.labels([{"lon": f.lon, "lat": f.lat, "text": f.name.split(",")[0]} for f in city.hospitals(1)], "hl", 13, zoom=10.7)]
    return L.deck(layers, L.view(city.centre[0], city.centre[1], 10.7, 45, -18))


if visuals() != "off":
    show_map(overview_deck(visuals()), height=620)
else:
    show_map(None)
st.caption("Drag to pan; scroll to zoom." + (" Right-drag or ctrl-drag to tilt and rotate." if visuals() == "rich" else " If maps are slow or blank on your computer, choose Light or No maps under Map detail in the sidebar.") + " Every road, hospital and person here is data from the model; nothing is illustrative.")
