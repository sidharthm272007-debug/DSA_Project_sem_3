import pandas as pd
import streamlit as st

from advisor.service import IDEA_LABEL
from advisor.verdict import service_rows
from sim.report import fmt_people
from ui.components import by_kind_chart, hero, kpi_row, legend, note, place_picker, priorities_panel, verdict_card, show_map
from ui.state import load_advisor_with_message, base_paths, core, places
from viz import layers as L
from viz.theme import PAL, TONE

A, city, _ = core()
PLACES = places()
ss = st.session_state

hero("Test an idea", "Put your own idea on the map and see what it would really change",
     "Pick a spot for a hospital, fire station or ambulance post, or two places to join with a road or bridge. You get a verdict, who it helps in which kind of emergency, and what it rests on.")
adv = load_advisor_with_message()
priorities_panel(A, adv)
tab_f, tab_l = st.tabs(["🏥 A facility", "🛣️ A road or bridge"])

# ------------------------------------------------------------------------------------------ facility
with tab_f:
    a, b = st.columns([1, 1.4])
    with a:
        idea = st.radio("What would you build?", list(IDEA_LABEL), format_func=IDEA_LABEL.get, key="i_idea")
    with b:
        st.markdown("**Where?**")
        pname, (plat, plon) = place_picker("i_f", "Where", PLACES, default="Peelamedu")
    if st.button("Evaluate this idea", type="primary", key="i_f_go"):
        with st.spinner("Re-scoring every scenario with this facility added…"):
            ss["i_f_res"] = (idea, adv.assess_facility(idea, plat, plon))
            ss["i_f_better"] = None
    got = ss.get("i_f_res")
    if got:
        idea_r, r = got
        z = {"hospital": "hosp", "hospital_major": "t1", "fire_station": "fire", "ambulance_post": "amb"}[idea_r]
        kpi_row([("Verdict", r.verdict.label, "", TONE[r.verdict.tone][1]),
                 ("People newly inside the standard", fmt_people(r.verdict.numbers["gain"]), "weighted across scenarios"),
                 ("Normal day / avg. disaster", f"{fmt_people(r.verdict.numbers['normal'])} / {fmt_people(r.verdict.numbers['disaster'])}", ""),
                 ("Indicative cost", f"₹{r.cost_crore:,.0f} crore", "placeholder", PAL["amber"])])
        st.write("")
        left, right = st.columns([1.15, 1])
        with left:
            snap_names = [s.name for s in adv.portfolio]
            which = st.selectbox("Show where the minutes are saved in…", range(len(snap_names)), format_func=lambda i: snap_names[i], key="i_f_snap")
            gm = adv.gain_map(idea_r, r.extra["node"], which)
            same = [f for f in (city.hospitals(2) if idea_r == "hospital" else city.hospitals(1) if idea_r == "hospital_major" else city.of_kind("fire_station") if idea_r == "fire_station" else city.hospitals(2))]
            layers = [L.road_layer(base_paths(), "#3B4C6B", 1.1, 120, "base"),
                      L.columns(gm["lat"], gm["lon"], gm["people"], "gain", color_hi="#2DD4BF", color_lo="#134E4A", radius=240, max_h=1800, tip_fmt="{:,.0f} people get there faster"),
                      L.dots([{"lon": f.lon, "lat": f.lat, "fill": [148, 163, 184, 230], "r": 5, "tip": f"existing: {f.name}"} for f in same], "ex", 5)] + \
                     L.pick_layers([{"lon": r.extra["lon"], "lat": r.extra["lat"], "n": "★", "tone": r.verdict.tone, "tip": r.title}])
            show_map(L.deck(layers, L.view(r.extra["lat"], r.extra["lon"], 11.0, 50, -12)), height=520)
            legend([(PAL["teal"], "people who would reach the service faster (height = people per 500 m cell)"), ("#94A3B8", "existing facilities")])
            if gm["people"].sum() == 0:
                note("Nobody reaches the service faster in this particular scenario.")
        with right:
            verdict_card(r.verdict, r.title)
            st.markdown("##### Situation by situation")
            st.altair_chart(by_kind_chart(r.verdict.numbers["by_kind"], r.verdict.numbers["normal"]), width="stretch")
        if st.button("Look for a clearly better spot nearby", key="i_f_better_go"):
            with st.spinner("Trying the most-needed junctions within 2.5 km…"):
                ss["i_f_better"] = adv.better_nearby(idea_r, r.extra["node"])
        bt = ss.get("i_f_better")
        if bt is not None:
            if bt["better"]:
                st.success(f"A better spot exists close by: {bt['where']} ({bt['lat']:.4f}, {bt['lon']:.4f}) would bring about {fmt_people(bt['value'])} people inside the standard versus {fmt_people(bt['here'])} here.")
            else:
                st.info("No junction within 2.5 km beats this spot by a meaningful margin.")
    else:
        st.info("Choose a type and a place, then evaluate.")

# ------------------------------------------------------------------------------------------ link
with tab_l:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**From**")
        na, pa = place_picker("i_la", "From", PLACES, default="Ukkadam")
    with c2:
        st.markdown("**To**")
        nb, pb = place_picker("i_lb", "To", PLACES, default="Singanallur")
    if st.button("Evaluate this road / bridge", type="primary", key="i_l_go"):
        try:
            with st.spinner("Re-scoring every scenario and everyday travel with the new link…"):
                ss["i_l_res"] = adv.assess_link(pa, pb)
        except ValueError as e:
            ss["i_l_res"] = None
            st.error(str(e))
    r = ss.get("i_l_res")
    if r:
        x = r.extra
        kpi_row([("Verdict", r.verdict.label, "", TONE[r.verdict.tone][1]), ("Length", f"{x['length_m'] / 1000:.1f} km", "straight-ish, with a detour factor"),
                 ("Kind", "Bridge" if x["crosses_water"] else "Road", "priced as a bridge if it crosses mapped water"), ("Indicative cost", f"₹{r.cost_crore:,.0f} crore", "placeholder", PAL["amber"])])
        st.write("")
        left, right = st.columns([1.15, 1])
        with left:
            rows = [{"a": x["a"], "b": x["b"], "tone": r.verdict.tone, "w": 5, "label": r.title}]
            mid = ((x["a"][0] + x["b"][0]) / 2, (x["a"][1] + x["b"][1]) / 2)
            show_map(L.deck([L.road_layer(base_paths(), "#3B4C6B", 1.1, 130, "base")] + L.link_layers(rows), L.view(mid[0], mid[1], 12.0, 55, -12)), height=480)
        with right:
            verdict_card(r.verdict, r.title)
        st.altair_chart(by_kind_chart(r.verdict.numbers["by_kind"], r.verdict.numbers["normal"], "service-standard gains"), width="stretch")
        st.dataframe(pd.DataFrame([{"Service": q["service"], "Counted in headline": "yes" if q["in_headline"] else "no", "Normal day": round(q["normal_people"]),
                                    "Avg. disaster": round(q["disaster_people"]), "Mean minutes saved": round(q["minutes"], 3)} for q in service_rows(r.effect, adv.services_all)]),
                     hide_index=True, width="stretch")
    else:
        st.info("Choose two different places and evaluate.")
