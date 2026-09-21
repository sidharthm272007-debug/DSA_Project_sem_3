import numpy as np
import streamlit as st

from sim.report import fmt_people
from ui.components import hero, kpi_row, legend, show_map
from ui.state import base_paths, core, visuals
from viz import layers as L
from viz.theme import PAL

A, city, _ = core()
g = city.g

hero("Coimbatore · emergency access", "When roads fail: which way do I go, and what should the city build?",
     "One model of the real road network, two kinds of user. If you are on the road in an emergency, it recommends a route and explains it. "
     "If you plan the city, it recommends where to build and whether what exists is worth it. Every recommendation comes with its reasons and what it rests on.")

r1, r2 = st.columns(2)
with r1:
    st.markdown('<div class="card tall" style="border-color:#2DD4BF66; min-height: 13rem"><h4>🧭 I need to get somewhere, or I know these roads</h4>'
                '<p>Pick what has happened. Get the route that fits how careful you want to be, the alternatives it beat and why, the nearest hospital that is really reachable, '
                'and the roads the city is leaning on. Add the closures you know about and it takes them into account.</p></div>', unsafe_allow_html=True)
    st.page_link("ui/pages/route.py", label="Best route now", icon="🧭")
    st.page_link("ui/pages/lab.py", label="Disaster lab", icon="🌪️")
with r2:
    st.markdown('<div class="card tall" style="border-color:#F59E0B66; min-height: 13rem"><h4>🏗️ I plan or fund infrastructure</h4>'
                '<p>Where the next hospital, fire station, ambulance post, road or bridge would help most across a normal day and every kind of disaster; a verdict on any idea you put on the map; '
                'and a back-test of whether flyovers and hospitals already built are pulling their weight.</p></div>', unsafe_allow_html=True)
    st.page_link("ui/pages/build.py", label="Where to build", icon="🏗️")
    st.page_link("ui/pages/idea.py", label="Test an idea on the map", icon="💡")
    st.page_link("ui/pages/existing.py", label="Is what exists worth it?", icon="🔍")
st.write("")
n_h = len(city.hospitals(2))
n_fire = len(city.of_kind("fire_station"))
kpi_row([
    ("People modelled", fmt_people(city.total_pop), "OSM road-density proxy scaled to the city estimate"),
    ("Road network", f"{g.length_m.sum() / 2000:,.0f} km", f"{g.n_nodes:,} junctions"),
    ("Hospitals mapped", str(n_h), f"{len(city.hospitals(1))} major, rest general"),
    ("Fire stations mapped", str(n_fire), "far fewer than the real city: see Data & assumptions", PAL["amber"] if n_fire < A["advisor"]["min_fire_stations"] else None),
    ("Named flyovers / bridges", str(len(city.structures)), "from OpenStreetMap"),
])
st.write("")

st.subheader("How a recommendation is made")
h1, h2, h3, h4 = st.columns(4)
for col, n, t, d in ((h1, "1", "Measure the damage", "Hazards close and slow real roads and put hospitals out of service; each stretch of road gets a risk score from what happened around it."),
                     (h2, "2", "Search genuinely different options", "Routes, sites or roads are generated and scored on the damaged network with exact shortest-path maths, not looked up."),
                     (h3, "3", "Choose by your priorities", "How careful you are on the road, or how much a normal day versus each disaster counts for the city. Change them and the answer changes."),
                     (h4, "4", "Explain and admit limits", "Each answer lists the numbers behind it and the data it rests on: what is measured, approximate or missing.")):
    col.markdown(f'<div class="card tall" style="min-height: 12rem"><h4>{n} · {t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
st.write("")
st.subheader("The city at a glance")
legend([(PAL["red"], "where people live (bigger and redder = more people per square kilometre)" if visuals() != "rich" else "where people live (column height = people per 500 m cell)"), (PAL["teal"], "major hospital"), ("#7DD3FC", "other hospital"), ("#FB923C", "fire station")])


@st.cache_resource(show_spinner=False)
def overview_deck(detail: str):
    cells = np.flatnonzero(city.cell_pop > 30)
    if detail == "rich":
        cols = L.columns(city.cell_lat[cells], city.cell_lon[cells], city.cell_pop[cells], "pop", color_hi="#F43F5E", color_lo="#4C1D3D", radius=200, max_h=1700)
    else:                                              # light: merge 500 m cells into ~1 km cells, so a weak graphics card draws a quarter of the shapes
        kl, ko = np.round(city.cell_lat[cells] / 0.009).astype(np.int64), np.round(city.cell_lon[cells] / 0.0092).astype(np.int64)
        key = kl * 100000 + ko
        uk, inv = np.unique(key, return_inverse=True)
        pop = np.bincount(inv, weights=city.cell_pop[cells])
        lat = np.bincount(inv, weights=city.cell_lat[cells] * city.cell_pop[cells]) / np.maximum(pop, 1e-9)
        lon = np.bincount(inv, weights=city.cell_lon[cells] * city.cell_pop[cells]) / np.maximum(pop, 1e-9)
        cols = L.columns(lat, lon, pop, "pop", color_hi="#F43F5E", color_lo="#4C1D3D", radius=560, tip_fmt="{:,.0f} people in about 1 km²", pickable=False)
    hosp = [{"lon": f.lon, "lat": f.lat, "fill": [45, 212, 191, 255] if f.tier == 1 else [125, 211, 252, 200], "r": 9 if f.tier == 1 else 4, "tip": f.name}
            for f in city.hospitals(2)]
    fire = [{"lon": f.lon, "lat": f.lat, "fill": [251, 146, 60, 255], "r": 9, "tip": f.name} for f in city.of_kind("fire_station")]
    layers = [L.road_layer(base_paths(), "#3B4C6B", 1.2, 150, "base"), cols, L.dots(hosp, "hosp"), L.dots(fire, "fire"), L.labels([{"lon": f.lon, "lat": f.lat, "text": f.name.split(",")[0]} for f in city.hospitals(1)], "hl", 13)]
    return L.deck(layers, L.view(city.centre[0], city.centre[1], 10.7, 45, -18))


if visuals() != "off":
    show_map(overview_deck(visuals()), height=620)
else:
    show_map(None)
st.caption("Drag to pan; scroll to zoom." + (" Right-drag or ctrl-drag to tilt and rotate." if visuals() == "rich" else " If maps are slow or blank on your computer, choose Light or No maps under Map detail in the sidebar.") + " Every road, hospital and person here is data from the model; nothing is illustrative.")
