import pandas as pd
import streamlit as st

from ui.components import bars, hero, note, priorities_panel, verdict_card, show_map
from ui.state import load_advisor_with_message, base_paths, core
from viz import layers as L
from viz.theme import PAL, TONE

A, city, _ = core()
ss = st.session_state

hero("Is it useful?", "Take it away and see what breaks",
     "For any flyover, bridge or hospital that already exists, the tool removes it, re-runs every scenario and everyday travel, and reports what the city would lose. "
     "That is the value of what is there today, measured the same way as a new idea.")
adv = load_advisor_with_message()
priorities_panel(A, adv)
tab_s, tab_h = st.tabs(["🌉 Flyovers & bridges", "🏥 Hospitals"])


def structure_map(a):
    paths = L.edge_paths(city, a.extra["edges"], 3000, a.title)
    e = a.extra["edges"]
    lat, lon = float(city.g.lat[city.g.edge_tails[e]].mean()), float(city.g.lon[city.g.edge_tails[e]].mean())
    zoom = 12.4 if a.extra["length_m"] < 1500 else 11.8 if a.extra["length_m"] < 4000 else 11.0
    layers = [L.road_layer(base_paths(), "#3B4C6B", 1.1, 130, "base")] + L.glow_roads(paths, TONE[a.verdict.tone][1], "st")
    return L.deck(layers, L.view(lat, lon, zoom, 40, -12))


with tab_s:
    idx = next((i for i, s in enumerate(city.structures) if "Naidu" in s["name"]), 0)
    st.markdown("##### Back-test: the Avinashi Road elevated corridor (G. D. Naidu flyover)")
    note("Press reports give about ₹1,791 crore for 10.1 km. The back-test asks: had it not been built, how much worse off would emergency access be, on a normal day and in each kind of disaster?")
    with st.spinner("Removing the flyover and re-scoring…"):
        feat = adv.assess_structure(idx)
    a, b = st.columns([1.1, 1])
    with a:
        verdict_card(feat.verdict, feat.title)
    with b:
        show_map(structure_map(feat), height=420)
    st.divider()
    st.markdown("##### Rank every named flyover and bridge")
    if st.button("Rank them all", type="primary", key="ex_s_go"):
        ss["ex_s_req"] = True
    if ss.get("ex_s_req"):
        bar = st.progress(0.0, text="Removing each structure in turn…")
        ranked = adv.rank_structures(progress=lambda f: bar.progress(min(f, 1.0), text="Removing each structure in turn…"))
        bar.empty()
        df = pd.DataFrame([{"Structure": r.title, "People at risk if lost": r.verdict.numbers["loss"], "Verdict": r.verdict.label, "Length (m)": round(r.extra["length_m"]),
                            "Worst case": r.verdict.numbers["worst_kind"] or "", "Extra minutes per resident": round(r.verdict.numbers["eve_minutes"], 3)} for r in ranked])
        st.altair_chart(bars(df.head(15), "Structure", "People at risk if lost", "people pushed outside emergency standards if it were lost"), width="stretch")
        st.dataframe(df.assign(**{"People at risk if lost": df["People at risk if lost"].round()}), hide_index=True, width="stretch")
        note("Structures shorter than 400 m are left out. A flyover already destroyed in a scenario is left out of its own average. 'People at risk' counts people who would fall outside a service standard, once per service (hospital, ambulance, and fire when enough stations are mapped).")
        pick = ranked[st.selectbox("Inspect", range(len(ranked)), format_func=lambda i: f"{ranked[i].title} · {ranked[i].verdict.label}", key="ex_s_insp")]
        c1, c2 = st.columns([1.1, 1])
        with c1:
            verdict_card(pick.verdict, pick.title)
        with c2:
            show_map(structure_map(pick), height=380)

with tab_h:
    st.markdown("Major hospitals are tested one at a time; general hospitals take about two minutes for the whole list.")
    b1, b2 = st.columns(2)
    if b1.button("Test the major hospitals", type="primary", key="ex_h1"):
        ss["ex_h"] = 1
    if b2.button("Test all general hospitals too", key="ex_h2"):
        ss["ex_h"] = 2
    if ss.get("ex_h"):
        tiers = [1] if ss["ex_h"] == 1 else [1, 2]
        bar = st.progress(0.0, text="Removing each hospital in turn…")
        ranked = []
        for n, t in enumerate(tiers):
            ranked += adv.rank_hospitals(t, progress=lambda f, n=n: bar.progress(min((n + f) / len(tiers), 1.0), text="Removing each hospital in turn…"))
        bar.empty()
        ranked.sort(key=lambda a: -a.verdict.numbers["loss"])
        df = pd.DataFrame([{"Hospital": r.title, "Tier": r.extra["fac"].tier, "People at risk if lost": round(r.verdict.numbers["loss"]), "Verdict": r.verdict.label,
                            "Where": "approximate position" if r.extra["fac"].approx else "mapped"} for r in ranked])
        st.altair_chart(bars(df.head(15), "Hospital", "People at risk if lost", "people pushed outside a hospital standard if it were lost", PAL["violet"]), width="stretch")
        st.dataframe(df, hide_index=True, width="stretch")
        note("Central hospitals overlap, so losing one rarely changes much; the ones on the edge of the city carry more. Tier 1 is judged on 30-min access to any hospital and 60-min access to a major one; tier 2 on 30-min access to any.")
        pick = ranked[st.selectbox("Inspect", range(len(ranked)), format_func=lambda i: f"{ranked[i].title} · {ranked[i].verdict.label}", key="ex_h_insp")]
        f = pick.extra["fac"]
        c1, c2 = st.columns([1.1, 1])
        with c1:
            verdict_card(pick.verdict, pick.title)
        with c2:
            show_map(L.deck([L.road_layer(base_paths(), "#3B4C6B", 1.1, 130, "base")] + L.pick_layers([{"lon": f.lon, "lat": f.lat, "n": "H", "tone": pick.verdict.tone, "tip": f.name}]),
                                    L.view(f.lat, f.lon, 12.0, 45, -10)), height=380)
