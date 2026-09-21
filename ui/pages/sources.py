from pathlib import Path

import pandas as pd
import streamlit as st

from sim.report import fmt_people
from ui.components import hero, kpi_row, note
from ui.state import core
from viz.theme import PAL

A, city, _ = core()
g = city.g
ROOT = Path(__file__).resolve().parents[2]

hero("Data & assumptions", "What the numbers rest on",
     "Nothing here is hidden. Each input is marked as measured, approximate, assumed or missing, so you know which results to trust and which to fix first.")

meta = city.meta
n_fire = len(city.of_kind("fire_station"))
approx = [f for f in city.facilities if f.approx]
kpi_row([("Junctions / road arcs", f"{g.n_nodes:,} / {g.n_edges:,}", "OpenStreetMap"), ("Named roads", f"{meta.get('named_ways', 0):,}", "used to name closures"),
         ("Facilities", str(len(city.facilities)), f"{len(approx)} with approximate positions"), ("Population modelled", fmt_people(city.total_pop), "proxy, not a census grid")])
st.write("")

rows = [
    ("Road network", "Measured", "OpenStreetMap: geometry, class, one-way, bridge / tunnel / flyover flags, speeds from class.", "Service, campus and informal roads are not loaded; railways act as barriers."),
    ("Water bodies", "Measured", f"OpenStreetMap: {len(city.water)} mapped water bodies (Noyyal River and lakes).", "Flood extent is a zone around the water, not a terrain simulation."),
    ("Named flyovers and bridges", "Measured", f"OpenStreetMap: {len(city.structures)} named structures.", "Costs of existing structures are estimates from one press-reported corridor."),
    ("Population", "Assumed", meta.get("population_source", "proxy"), "Distribution follows road density, so dense informal areas may be off."),
    ("Hospitals", "Mostly measured", f"{len(city.hospitals(2))} hospitals from OpenStreetMap; tier 1 by name pattern (assumptions.yaml).", "Approximate placements: " + (", ".join(f.name for f in approx if f.kind == 'hospital') or "none") + "."),
    ("Fire stations", "Missing", f"{n_fire} mapped (Ganapathy from OpenStreetMap, Peelamedu placed at its locality centre).", "The city has more. Fire-service numbers are excluded from headline advisor results until at least "
     f"{A['advisor']['min_fire_stations']} are mapped."),
    ("Ambulances", "Assumed", "Bases at major hospitals (3 each) plus general hospitals, spread out; default fleet of 60.", "Real 108 posts are not mapped."),
    ("Hazard physics", "Assumed", "Heuristic damage, spread and flood parameters in assumptions.yaml (earthquake damage by magnitude, fire growth, gas plume, flood extent).", "Not calibrated to a specific event."),
    ("Costs", "Placeholders", "Only the ₹177 crore/km flyover figure is press-sourced; every other unit cost is an order-of-magnitude placeholder.", "Benefit numbers do not depend on cost; value-for-money numbers do."),
    ("Standards", "Judgement", f"Any hospital within {A['impact']['slow_minutes']} min, a major hospital within {A['impact']['severe_minutes']} min, fire engine within {A['advisor']['sla_min']['fire']} min, ambulance within {A['advisor']['sla_min']['ambulance']} min.", "Planning conventions of this tool, not legal norms."),
]
df = pd.DataFrame(rows, columns=["Input", "Status", "Source", "Known limits"])
st.dataframe(df, hide_index=True, width="stretch", column_config={"Source": st.column_config.TextColumn(width="large"), "Known limits": st.column_config.TextColumn(width="large")})

st.subheader("Fix the biggest gap: add fire stations and ambulance posts")
st.markdown("Add rows to `data/coimbatore/facilities_extra.csv` (name, kind, and either lat/lon or a locality name from OpenStreetMap), then run:")
st.code("python -m scripts.build_derived", language="bash")
st.markdown("Every page picks the new stations up automatically. Fire numbers join the headline advisor results once enough stations are present.")

if approx:
    st.subheader("Facilities with approximate positions")
    st.dataframe(pd.DataFrame([{"Name": f.name, "Kind": f.kind, "Note": f.note} for f in approx]), hide_index=True, width="stretch")

st.subheader("All assumptions")
note("Every constant the simulator and advisor use lives in one file. Change it and every page follows.")
txt = (ROOT / "assumptions.yaml").read_text(encoding="utf-8")
with st.expander("assumptions.yaml", expanded=False):
    st.code(txt, language="yaml")
st.download_button("⬇ Download assumptions.yaml", txt, "assumptions.yaml", "text/yaml")
