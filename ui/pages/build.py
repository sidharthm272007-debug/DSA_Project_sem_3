import pandas as pd
import streamlit as st

from advisor.service import IDEA_LABEL
from sim.report import fmt_people
from advisor.verdict import service_rows
from ui.components import by_kind_chart, hero, kpi_row, legend, note, priorities_panel, verdict_card, show_map
from ui.state import load_advisor_with_message, base_paths, core
from viz import layers as L
from viz.theme import PAL, TONE

A, city, router = core()
ss = st.session_state

hero("Where to build", "Which new hospital, fire station, road or bridge would matter most?",
     "Every candidate is scored against the same set of futures: a normal day plus the worst moments of earthquakes, fires, floods, bridge failures and a storm. "
     "An idea that only helps on a good day loses to one that also helps when roads are cut.")
adv = load_advisor_with_message()
priorities_panel(A, adv)

tab_f, tab_l = st.tabs(["🏥 Hospitals, fire stations, ambulances", "🛣️ Roads & bridges"])


fmt = fmt_people


# ------------------------------------------------------------------------------------------- facilities
with tab_f:
    c1, c2, c3 = st.columns([2, 1, 1])
    idea = c1.radio("What to build", list(IDEA_LABEL), format_func=IDEA_LABEL.get, horizontal=True, key="b_idea")
    k = c2.slider("How many", 1, 8, 5, key="b_k")
    go = c3.button("Find the best sites", type="primary", width="stretch", key="b_go")
    if idea == "fire_station" and not adv.fire_ok:
        st.warning(f"Only {adv.n_fire} fire station(s) are mapped in this data, so the fire map below overstates the need: most of the city is 'uncovered' only because the stations are missing from OpenStreetMap. "
                   "Add them in data/coimbatore/facilities_extra.csv for decision-grade output.")
    if go:
        ss["b_req"] = (idea, k)
    req = ss.get("b_req")
    if req and req == (idea, k):
        bar = st.progress(0.0, text="Scoring candidate sites against every scenario…")
        recs = adv.recommend_sites(idea, k, progress=lambda f: bar.progress(min(f, 1.0) * 0.9, text="Scoring candidate sites against every scenario…"))
        bar.empty()
        nm = adv.need_map(idea)
        top = recs[0]
        tot_gain = sum(r.verdict.numbers["gain"] for r in recs)
        cost = sum(r.cost_crore for r in recs)
        kpi_row([("Sites recommended", str(len(recs)), f"{sum(r.verdict.tone == 'good' for r in recs)} rated Recommended"),
                 ("People newly inside the standard", fmt(tot_gain), "weighted average across scenarios, all sites together"),
                 ("Best single site", (lambda n: n if len(n) <= 20 else n[:19] + "…")(top.pick.where.split("(")[0].strip()), top.verdict.label, TONE[top.verdict.tone][1]),
                 ("Indicative cost", f"₹{cost:,.0f} crore", "placeholder unit costs", PAL["amber"])])
        st.write("")
        rows = [{"lon": r.pick.lon, "lat": r.pick.lat, "n": r.pick.rank, "tone": r.verdict.tone, "tip": f"#{r.pick.rank} {r.pick.where}: {r.verdict.label}"} for r in recs]
        existing = [{"lon": f.lon, "lat": f.lat, "fill": [148, 163, 184, 230], "r": 5, "tip": f"existing: {f.name}"}
                    for f in (city.hospitals(2 if idea == "hospital" else 1) if idea.startswith("hospital") else city.of_kind(idea if idea != "ambulance_post" else "hospital"))]
        layers = [L.road_layer(base_paths(), "#3B4C6B", 1.1, 120, "base"),
                  L.columns(nm["lat"], nm["lon"], nm["people"], "need", color_hi="#F43F5E", color_lo="#4C1D3D", radius=240, max_h=2000, tip_fmt="{:,.0f} people outside the standard, on average across scenarios"),
                  L.dots(existing, "exist", 5)] + L.pick_layers(rows)
        show_map(L.deck(layers, L.view(city.centre[0], city.centre[1], 10.6, 42, -14)), height=600)
        legend([(PAL["red"], "people outside the standard (weighted across scenarios)"), ("#94A3B8", "existing"), (TONE["good"][1], "recommended site"), (TONE["mid"][1], "worth a closer look")])

        st.markdown("##### The ranking")
        df = pd.DataFrame([{"#": r.pick.rank, "Where": r.pick.where, "Verdict": r.verdict.label, "Normal day": round(r.pick.effect["normal"].get(f"{r.pick.primary}_within", 0)),
                            "Avg. disaster": round(r.pick.effect["disaster_avg"].get(f"{r.pick.primary}_within", 0)), "Weighted": round(r.verdict.numbers["gain"]),
                            "Cost (₹ crore)": r.cost_crore, "People / crore": round(r.verdict.numbers["per_crore"], 1)} for r in recs])
        st.dataframe(df, hide_index=True, width="stretch")
        note("'People' = people newly inside the standard for that service (≤30 min to a hospital, ≤60 min to a major hospital, ≤8 min fire engine, ≤10 min ambulance). Later picks are measured after the earlier ones are built.")
        pick = recs[st.selectbox("Inspect a site", range(len(recs)), format_func=lambda i: f"#{recs[i].pick.rank} · {recs[i].pick.where} · {recs[i].verdict.label}", key="b_insp")]
        a, b = st.columns([1.2, 1])
        with a:
            verdict_card(pick.verdict, f"Site #{pick.pick.rank}: {IDEA_LABEL[idea]}")
        with b:
            st.markdown("##### Who it helps, situation by situation")
            st.altair_chart(by_kind_chart(pick.verdict.numbers["by_kind"], pick.pick.effect["normal"].get(f"{pick.pick.primary}_within", 0)), width="stretch")
    else:
        st.info("Choose what to build and press “Find the best sites”. The first run takes about 20–30 s; results are then remembered.")

# ------------------------------------------------------------------------------------------- links
with tab_l:
    c1, c2 = st.columns([3, 1])
    c1.markdown("Roads and bridges are found where two places are close as the crow flies but far apart by road at peak hour (a river, a railway, a missing link between arterials), "
                "then each shortlisted idea is scored like the facilities above. A new road is assumed to be lost in any scenario that damages the ground it crosses.")
    n = c2.slider("Show the best", 3, 12, 8, key="l_n")
    if c2.button("Find the best roads & bridges", type="primary", width="stretch", key="l_go"):
        ss["l_req"] = True
    if ss.get("l_req"):
        bar = st.progress(0.0, text="Searching for missing links and scoring them…")
        links = adv.recommend_links(n=n, progress=lambda f: bar.progress(min(f, 1.0), text="Searching for missing links and scoring them…"))
        bar.empty()
        if not links:
            st.warning("No candidate met the screening rules (a place pair much slower by road than by air).")
        else:
            g = city.g
            rows = [{"a": (float(g.lat[r.idea.u]), float(g.lon[r.idea.u])), "b": (float(g.lat[r.idea.v]), float(g.lon[r.idea.v])), "tone": r.verdict.tone,
                     "w": 3 + min(6, r.verdict.numbers["eve_minutes"] * 30), "label": f"{r.idea.where_u} ↔ {r.idea.where_v}: {r.verdict.label}"} for r in links]
            layers = [L.road_layer(base_paths(), "#3B4C6B", 1.1, 130, "base")] + L.link_layers(rows)
            show_map(L.deck(layers, L.view(city.centre[0], city.centre[1], 11.0, 55, -14)), height=560)
            legend([(TONE["good"][1], "recommended"), (TONE["mid"][1], "worth a closer look / everyday benefit"), (TONE["weak"][1], "weak case")])
            df = pd.DataFrame([{"Between": f"{r.idea.where_u}  ↔  {r.idea.where_v}", "Kind": "Bridge" if r.idea.crosses_water else "Road", "Length (m)": round(r.idea.length_m),
                                "Peak-hour drive today → with it (min)": f"{'25+' if r.idea.today_min >= 24.9 else format(r.idea.today_min, '.0f')} → {r.idea.with_min:.0f}", "Verdict": r.verdict.label,
                                "Emergency gains": round(r.verdict.numbers["gain"]), "Everyday (min/trip)": round(r.verdict.numbers["eve_minutes"], 3), "Cost (₹ crore)": round(r.idea.cost_crore)} for r in links])
            st.dataframe(df, hide_index=True, width="stretch")
            pick = links[st.selectbox("Inspect an idea", range(len(links)), format_func=lambda i: f"{links[i].idea.where_u} ↔ {links[i].idea.where_v} · {links[i].verdict.label}", key="l_insp")]
            a, b = st.columns([1.2, 1])
            with a:
                verdict_card(pick.verdict, "New road / bridge")
            with b:
                st.markdown("##### Situation by situation")
                st.altair_chart(by_kind_chart(pick.verdict.numbers["by_kind"], pick.verdict.numbers["normal"], "service-standard gains"), width="stretch")
                st.dataframe(pd.DataFrame([{"Service": r["service"], "Counted in headline": "yes" if r["in_headline"] else "no", "Normal day": round(r["normal_people"]), "Avg. disaster": round(r["disaster_people"])}
                                           for r in service_rows(pick.idea.effect, adv.services_all)]), hide_index=True, width="stretch")
    else:
        st.info("Press the button to search the whole city for missing links (about 30 s the first time).")


st.divider()
st.markdown("##### 📄 Take it to a meeting: the planning brief")
st.caption("One document with the top sites for each kind of facility, the best road and bridge ideas, and a back-test of the flyovers and major hospitals that already exist, each with its reasons and what it rests on. "
           "It uses the priorities you set above and takes about a minute the first time.")
if st.button("Build the planning brief", key="brief_go"):
    from advisor.brief import build_brief
    bar = st.progress(0.0, text="Assembling recommendations…")
    ss["brief_md"] = build_brief(adv, progress=lambda f: bar.progress(min(f, 1.0), text="Assembling recommendations…"))
    bar.empty()
if ss.get("brief_md"):
    st.download_button("⬇ Download the brief (Markdown)", ss["brief_md"], "planning_brief.md", "text/markdown", type="primary")
    with st.expander("Preview"):
        st.markdown(ss["brief_md"])
