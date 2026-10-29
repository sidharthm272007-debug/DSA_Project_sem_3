import bisect

import numpy as np
import pandas as pd
import streamlit as st

from ui.verdict import Verdict
from routing.lifelines import lifelines
from routing.recommend import advise, fmt_min, hospital_options
from sim.report import fmt_people
from sim.state import hazard_state, normal_state, state_from_frame
from ui.components import findings, kpi_row, legend, note, page_header, pareto_chart, place_picker, show_map, verdict_card
from ui.scenarios import make_scenario, presets
from ui.state import base_paths, core, edges_of_roads, places, road_index
from viz import layers as L
from viz.theme import PAL

A, city, router = core()
PLACES = places()
PRESETS = presets(city)
ss = st.session_state
g = city.g

page_header("route", "Best route now",
            "Tell it what has happened and where you need to go — get a recommended route with the reasons behind it, the alternatives it beat, and what it rests on.")

# ------------------------------------------------------------------------------------------------- what has happened
st.markdown("##### 1. What has happened?")
c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
sources = ["A disaster scenario", "Normal day"] + (["My last Disaster-lab run"] if ss.get("lab_res") is not None else [])
src = c1.radio("Situation", sources, key="rt_src", label_visibility="collapsed")
state = None
if src == "A disaster scenario":
    name = c2.selectbox("Scenario", list(PRESETS), key="rt_preset", label_visibility="collapsed")
    dur = int(PRESETS[name]["hours"] * 60)
    t_min = c3.slider("Minutes after it began", 0, dur, min(90, dur), 10, key="rt_t")
    key = ("preset", name, t_min)
    if ss.get("rt_key") != key:
        with st.spinner("Working out the damage at that moment…"):
            p = PRESETS[name]
            sc = make_scenario(city, name, p["specs"], p["ctx"], p["hours"], 1)
            ss["rt_state"] = hazard_state(city, A, sc.hazards, sc.context.background(A), float(t_min), seed=1, wind=(sc.context.wind_toward_deg, sc.context.wind_kmh), label=f"{name}, +{t_min} min")
            ss["rt_key"] = key
    state = ss["rt_state"]
    note("This shows the damage itself. It does not include emergency crews reopening roads; run the scenario in the Disaster lab and choose 'My last Disaster-lab run' to include them.")
elif src == "Normal day":
    tod = c2.selectbox("Traffic", ["off_peak", "peak", "night"], format_func={"off_peak": "Daytime, off-peak", "peak": "Rush hour", "night": "Night"}.get, key="rt_tod", label_visibility="collapsed")
    key = ("normal", tod)
    if ss.get("rt_key") != key:
        ss["rt_state"] = normal_state(city, A["context"]["time_of_day_speed"][tod] * A["context"]["weather_speed"]["clear"])
        ss["rt_key"] = key
    state = ss["rt_state"]
else:
    res = ss["lab_res"]
    T = int(res.times[-1])
    t_min = c3.slider("Minutes into the run", 0, T, min(90, T), 10, key="rt_t2")
    fi = max(0, bisect.bisect_right(res.times, t_min) - 1)
    c2.markdown(f"**{res.scenario_name}**")
    key = ("lab", id(res), fi)
    if ss.get("rt_key") != key:
        ss["rt_state"], ss["rt_key"] = state_from_frame(city, res, fi), key
    state = ss["rt_state"]
n_closed = int((state.F[(g.rev < 0) | (np.arange(g.n_edges) < g.rev)] == 0).sum())
note(f"At this moment: {n_closed:,} road segments closed, {len(state.closed_fac)} hospital(s) out of service, traffic and weather at {state.bg:.0%} of normal speed.")

tab_r, tab_l = st.tabs([":material/route: Recommend a route", ":material/traffic: Roads people depend on"])

# ------------------------------------------------------------------------------------------------- route advice
with tab_r:
    st.markdown("##### 2. Your trip")
    t1, t2, t3 = st.columns([1.2, 1.3, 1])
    with t1:
        st.markdown("**From**")
        na, pa = place_picker("rt_a", "From", PLACES, default="Saibaba Colony")
    with t2:
        st.markdown("**To**")
        goal = st.radio("To", ["A place", "Nearest working hospital", "Nearest working major hospital"], key="rt_goal", horizontal=True, label_visibility="collapsed")
        if goal == "A place":
            nb, pb = place_picker("rt_b", "Destination", PLACES, default="Peelamedu")
    with t3:
        st.markdown("**How careful?**")
        tol = st.select_slider("Attitude to risk", ["cautious", "balanced", "fastest"], value="balanced", key="rt_tol", label_visibility="collapsed",
                               format_func=str.title, help="Cautious accepts a longer trip to avoid dangerous roads; Fastest takes the quickest open route.")
        v = A["routing"]["tolerance_min_per_risk_km"][tol]
        note(f"{tol.title()}: add up to {v:g} min to avoid a km of full-risk road.")
    with st.expander("What you know locally (roads you know are blocked or unsafe)", icon=":material/pin_drop:"):
        names = sorted(road_index())
        k1, k2 = st.columns(2)
        blocked = k1.multiselect("Roads I know are blocked", names, key="rt_blk", help="Treated as closed, in both directions.")
        unsafe = k2.multiselect("Roads I know are unsafe or jammed", names, key="rt_unsafe", help="Still usable, but treated as maximum risk, so avoided when there is a reasonable alternative.")
    go = st.button("Recommend a route", type="primary", key="rt_go")

    if True:
        if not go and "rt_adv" not in ss:
            st.info("Set the situation, your trip and how careful you want to be, then press “Recommend a route”.")
        if go:
            o = city.snap(*pa)[0]
            be, ue = edges_of_roads(blocked), edges_of_roads(unsafe)
            hosp_note = None
            if goal == "A place":
                d = city.snap(*pb)[0]
                adv = advise(city, router, A, state, o, d, tol, be, ue)
                dest_name, dest_ll = nb, (float(g.lat[d]), float(g.lon[d]))
            else:
                tier = 1 if "major" in goal else None
                opts = hospital_options(city, router, A, state, o, tier, 4, be)
                if not opts["open"]:
                    adv, dest_name, dest_ll = None, None, None
                else:
                    tried = [(r, advise(city, router, A, state, o, r["fac"].node, tol, be, ue)) for r in opts["open"][:3]]
                    tried = [(r, a) for r, a in tried if a.recommended]
                    r, adv = min(tried, key=lambda x: x[1].recommended.objective) if tried else (opts["open"][0], None)
                    dest_name, dest_ll = r["fac"].name, (r["fac"].lat, r["fac"].lon)
                    near = opts["nearest_by_air"][0]
                    hosp_note = (opts, near, r)
            ss["rt_adv"] = dict(origin=na, adv=adv, o=o, a_ll=(float(g.lat[o]), float(g.lon[o])), dest=dest_name, dest_ll=dest_ll, hosp=hosp_note, tol=tol, blocked=blocked, state_label=state.label, goal=goal)
        R = ss.get("rt_adv")
        if R and R["adv"] is None:
            st.error("No open hospital can be reached from here in this scenario.")
        elif R:
            adv, rec = R["adv"], R["adv"].recommended
            walk = adv.walk_first
            if rec is None and walk:
                st.warning(f"You are cut off. {adv.reasons[-1]}")
                adv, rec = walk["advice"], walk["advice"].recommended
            if rec is None:
                st.error(adv.reasons[-1] if adv.reasons else "No open route exists.")
            else:
                steps = [s for s in rec.steps if s["m"] >= 250]
                named = [s for s in steps if s["road"] not in ("unnamed road", "flyover")]
                via = " → ".join(dict.fromkeys(s["road"] for s in sorted(named, key=lambda s: -s["m"])[:3])) or "local roads"
                dest_word = R["dest"]
                if R["hosp"]:
                    opts, near, chosen = R["hosp"]
                    reasons = list(adv.reasons)
                    if near["fac"].id != chosen["fac"].id:
                        reasons.append(f"The nearest hospital by distance, {near['fac'].name}, is {near['status']}" + (f" ({fmt_min(near['time_s'])} away)" if near["status"] == "open" else "")
                                       + f"; {chosen['fac'].name} is the better choice by time and risk.")
                else:
                    reasons = list(adv.reasons)
                head = f"{fmt_min(rec.time_s)} to {dest_word}: via {via}."
                if R["adv"].walk_first:
                    head = f"Walk about {walk['walk_m']:.0f} m to {walk['where']}, then {fmt_min(rec.time_s)} to {dest_word}: via {via}."
                v_ = Verdict("Recommended route", "good", head, reasons,
                             ["Every route shown is found by Yen's algorithm (the genuinely different shortest routes, ranked by time) plus one search biased hard toward safety, then narrowed to the "
                              "Pareto-optimal set: routes where no other option is both faster and safer. The recommendation is the point on that set that best fits how careful you asked to be — "
                              "picking from the full trade-off this way finds routes a simpler single-formula search can miss.",
                              "Risk is a judgement score built from the scenario (inside the hazard, slowed, near a closure, structure at risk); it is not a measured probability.",
                              "Times use the scenario's road speeds. The queues that many people re-routing at once would create are not modelled, so expect the popular detours to be slower.",
                              "Roads you marked as blocked were treated as closed; roads you marked unsafe were treated as maximum risk." if (R["blocked"] or ss.get("rt_unsafe")) else "Damage is from the chosen scenario; conditions on the ground may differ.",
                              f"Situation: {R['state_label']}."])
                verdict_card(v_, f"{R['origin']}  →  {dest_word}")
                if len(adv.options) > 1:
                    note("The time vs. risk trade-off among the routes actually considered — every point here beats all the others on at least one of the two:")
                    st.altair_chart(pareto_chart(adv.options, rec), width="stretch")
                # ---------------- map
                layers = [L.road_layer(base_paths(), PAL["line2"], 1.1, 110, "base")]
                ce = state.closed_edges
                ce = ce[(g.rev[ce] < 0) | (ce < g.rev[ce])]
                layers += L.glow_roads(L.edge_paths(city, ce, 5000, "closed road"), PAL["red"], "blk")
                layers += L.hazard_layers(state.prims)
                be_ = edges_of_roads(R["blocked"]) if R["blocked"] else []
                if len(be_):
                    layers.append(L.road_layer(L.edge_paths(city, be_, 1500, "blocked (you)"), PAL["violet"], 4, 255, "user"))
                if adv.usual and adv.usual.edges != rec.edges:
                    layers.append(L.road_layer([{"path": adv.usual.coords, "tip": "your usual route"}], PAL["muted"], 3, 200, "usual", True))
                for o in adv.options:
                    if o is rec:
                        continue
                    col = PAL["amber"] if "Fastest" in o.tags else PAL["green"] if "Safest" in o.tags else PAL["slate"]
                    layers.append(L.road_layer([{"path": o.coords, "tip": f"{'/'.join(o.tags) or 'Alternative'}: {fmt_min(o.time_s)}, {o.dist_m / 1000:.1f} km"}], col, 3.5, 230, f"alt{id(o)}", True))
                layers += [L.road_layer([{"path": rec.coords, "tip": f"Recommended: {fmt_min(rec.time_s)}"}], PAL["accent"], 12, 60, "rec_glow"),
                           L.road_layer([{"path": rec.coords, "tip": f"Recommended: {fmt_min(rec.time_s)}"}], PAL["accent"], 6, 255, "rec", True)]
                pins = [{"lat": R["a_ll"][0], "lon": R["a_ll"][1], "n": "A", "tone": "mid", "tip": "Start"}, {"lat": R["dest_ll"][0], "lon": R["dest_ll"][1], "n": "B", "tone": "good", "tip": dest_word}]
                if R["adv"].walk_first:
                    wn = walk["node"]
                    layers.append(L.road_layer([{"path": [[R["a_ll"][1], R["a_ll"][0]], [float(g.lon[wn]), float(g.lat[wn])]], "tip": "walk"}], PAL["violet"], 4, 255, "walk"))
                layers += L.pick_layers(pins)
                mid = ((R["a_ll"][0] + R["dest_ll"][0]) / 2, (R["a_ll"][1] + R["dest_ll"][1]) / 2)
                show_map(L.deck(layers, L.view(mid[0], mid[1], 11.6, 40, -8)), height=560)
                legend([(PAL["accent"], "recommended"), (PAL["amber"], "fastest (if different)"), (PAL["green"], "safest (if different)"), (PAL["muted"], "your usual route"), (PAL["red"], "closed road"), (PAL["violet"], "blocked by you / walk")])
                # ---------------- options and steps
                st.markdown("##### The routes on the trade-off" if len(adv.options) > 1 else "##### The route")
                rows = [{"Route": " / ".join(o.tags) or "On the frontier", "Time": fmt_min(o.time_s), "Distance (km)": round(o.dist_m / 1000, 1), "Risk-km": round(o.risk_km, 2),
                         "High-risk road (km)": round(o.hazard_km, 1), "Extra vs fastest": f"{(o.time_s - min(x.time_s for x in adv.options)) / 60:+.1f} min"} for o in adv.options]
                if adv.usual:
                    rows.append({"Route": "Usual route on a normal day", "Time": fmt_min(adv.usual.time_s), "Distance (km)": round(adv.usual.dist_m / 1000, 1), "Risk-km": round(adv.usual.risk_km, 2),
                                 "High-risk road (km)": round(adv.usual.hazard_km, 1), "Extra vs fastest": "cut" if adv.usual_cut_at else ""})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                with st.expander("Step by step (recommended route)", expanded=False):
                    st.dataframe(pd.DataFrame([{"Road": s["road"] + (" (flyover/bridge)" if s["deck"] else ""), "Distance (m)": round(s["m"]), "Minutes": round(s["s"] / 60, 1), "Risk": "High" if s["risky"] else "Low"}
                                               for s in rec.steps]), hide_index=True, width="stretch")
                if R["hosp"]:
                    opts = R["hosp"][0]
                    with st.expander("Hospitals compared"):
                        st.dataframe(pd.DataFrame([{"Hospital": r["fac"].name, "Tier": r["fac"].tier, "Minutes": round(r["time_s"] / 60, 1), "Straight-line km": round(r["air_m"] / 1000, 1)} for r in opts["open"]]), hide_index=True, width="stretch")
                        bad = [r for r in opts["nearest_by_air"] if r["status"] != "open"]
                        if bad:
                            note("Close by but unusable: " + ", ".join(f"{r['fac'].name} ({r['status']})" for r in bad))

# ------------------------------------------------------------------------------------------------- lifelines
with tab_l:
    st.markdown("Every home has a fastest route to its nearest open hospital. **A road's load is the number of people whose fastest route uses it.** Compared with a normal day, "
                "this shows which roads the city depends on, which become the detours everyone takes, and which have been cut.")
    tier_lbl = st.radio("Which hospitals?", ["Any hospital", "Major hospitals only"], horizontal=True, key="ll_tier")
    if st.button("Show the roads people depend on", type="primary", key="ll_go"):
        with st.spinner("Tracing every home's fastest route to a hospital, now and on a normal day…"):
            ss["ll"] = (state.label, lifelines(city, router, state, 1 if tier_lbl.startswith("Major") else None), tier_lbl)
    got = ss.get("ll")
    if got:
        label, LL, tl = got
        now, norm, ch = LL["now"], LL["normal"], LL["change"]
        kpi_row([("People who can reach a hospital", fmt_people(now.reach), f"normal: {fmt_people(norm.reach)}"),
                 ("People with no route", fmt_people(now.cut), f"normal: {fmt_people(norm.cut)}", PAL["red"] if now.cut > norm.cut + 300 else None),
                 ("Busiest lifeline", LL["top_load"][0]["road"] if LL["top_load"] else "-", f"{fmt_people(LL['top_load'][0]['now'])} people depend on it" if LL["top_load"] else ""),
                 ("Biggest surge", LL["top_rise"][0]["road"] if LL["top_rise"] else "none", f"+{fmt_people(LL['top_rise'][0]['rise'])} people" if LL["top_rise"] else "", PAL["amber"] if LL["top_rise"] else None)])
        st.write("")
        fl = now.flow
        keep = np.flatnonzero((fl > 0) & ((g.rev < 0) | (fl >= fl[np.maximum(g.rev, 0)])))
        keep = keep[np.argsort(-fl[keep])][:3500]
        fmax = float(fl[keep].max()) if len(keep) else 1.0
        rows = []
        for e in keep:
            base_f, c = float(norm.flow[e]), float(ch[e])
            up = c / base_f if base_f > 0 else (9.0 if c > 0 else 0.0)
            col = PAL["red"] if up >= 1.0 else PAL["amber"] if up >= 0.3 else PAL["accent"]
            rows.append({"path": L.poly(city, e), "w": 1.2 + 8.0 * (float(fl[e]) / fmax) ** 0.5, "col": L.hex_rgb(col, 235),
                         "tip": f"{city.edge_name(int(e))}: {fl[e]:,.0f} people depend on it (normal day {base_f:,.0f})"})
        import pydeck as pdk
        lost = np.flatnonzero((norm.flow > 0) & (now.flow == 0) & ((g.rev < 0) | (norm.flow >= norm.flow[np.maximum(g.rev, 0)])))
        lost = lost[np.argsort(-norm.flow[lost])][:1200]
        lrows = [{"path": L.poly(city, e), "tip": f"{city.edge_name(int(e))}: lost its load ({norm.flow[e]:,.0f} people normally)"} for e in lost]
        layers = [L.road_layer(base_paths(), PAL["line2"], 1.0, 100, "base"),
                  pdk.Layer("PathLayer", data=rows, id="ll", get_path="path", get_color="col", get_width="w", width_units="pixels", width_min_pixels=1, cap_rounded=True, pickable=True, auto_highlight=True)]
        if lrows:
            layers.insert(1, L.road_layer(lrows, PAL["slate"], 2, 200, "lost", True))
        layers += L.hazard_layers(state.prims)
        show_map(L.deck(layers, L.view(city.centre[0], city.centre[1], 10.7, 45, -12)), height=600)
        legend([(PAL["accent"], "load about the same as a normal day"), (PAL["amber"], "load up 30%+"), (PAL["red"], "load at least doubled"), (PAL["slate"], "lost its load (cut or bypassed)")])
        a, b, c = st.columns(3)
        for col, key, title, val in ((a, "top_load", "Roads most people depend on", "now"), (b, "top_rise", "Roads whose load surges", "rise"), (c, "top_fall", "Roads that lost their load", "fall")):
            with col:
                st.markdown(f"**{title}**")
                if LL[key]:
                    st.dataframe(pd.DataFrame([{"Road": r["road"], "People": round(abs(r[val])), "Normal": round(r["normal"])} for r in LL[key]]), hide_index=True, width="stretch")
                else:
                    note("None in this situation.")
        if LL["top_rise"]:
            findings([f"{LL['top_rise'][0]['road']} takes on {fmt_people(LL['top_rise'][0]['rise'])} more people's route to a hospital than on a normal day: it is the detour a lot of people will choose, so expect it to slow down and consider staying off it if you have a choice.",
                      "For a planner, the roads at the top of the first list are the ones to protect and keep clear; the surge list shows where a second route is missing."])
        note("Load counts people, not vehicles, and each home is assigned to its single fastest route; the queues this would cause are not modelled.")
    else:
        st.info("Press the button to trace the routes for the situation chosen above.")
