import bisect
import json

import numpy as np
import pandas as pd
import streamlit as st

from sim.engine import Simulation
from sim.report import fmt_people
from ui.components import findings, hero, kpi_row, legend, note, place_picker, timeline_chart, show_map
from ui.scenarios import HAZARD_TYPES, NEEDS_PLACE, make_scenario, presets
from ui.state import base_paths, core, places
from viz import layers as L
from viz.anim import TripBook
from viz.theme import HAZARD_COLOR, PAL, UNIT_COLOR

A, city, router = core()
PLACES = places()
PRESETS = presets(city)
ss = st.session_state
ss.setdefault("lab_specs", [])
ss.setdefault("lab_ctx", {"time_of_day": "off_peak", "weather": "clear", "wind_toward_deg": 90, "wind_kmh": 12})
ss.setdefault("lab_hours", 8)
ss.setdefault("lab_name", "My scenario")
ss.setdefault("lab_res", None)
ss.setdefault("lab_playing", False)
ss.setdefault("lab_t", 0)

hero("Disaster lab", "Break the city, then watch how help responds",
     "Combine any hazards, choose the hour and the weather, and run a minute-by-minute simulation on the real roads: damage, dispatch, re-routing, hospital outages, debris clearing.")

t_build, t_watch, t_find = st.tabs(["1 · Build the scenario", "2 · Watch it unfold", "3 · Findings & data"])


def describe(spec: dict) -> str:
    t = spec["type"]
    where = spec.get("place") or (spec.get("water") if t == "flood_zone" and spec.get("water") else "")
    if t == "structure_failure":
        where = city.structures[spec["structure"]]["name"]
    bits = {"earthquake": f"magnitude {spec.get('magnitude', 6.0)}", "storm": f"severity {spec.get('severity', 3)}", "custom": f"radius {spec.get('radius_m', 300):.0f} m",
            "structure_failure": spec.get("mode", "collapse")}.get(t, f"severity {spec.get('severity', 3)}")
    return f"**{HAZARD_TYPES[t]}**" + (f" · {where}" if where else "") + f" · {bits} · starts at minute {spec.get('start_min', 0):.0f}"


# ================================================================================================= BUILD
with t_build:
    st.markdown("##### Start from a real-world case")
    cols = st.columns(3)
    for i, (name, p) in enumerate(PRESETS.items()):
        with cols[i % 3]:
            st.markdown(f'<div class="card tall"><h4>{name}</h4><p>{p["blurb"]}</p></div>', unsafe_allow_html=True)
            if st.button("Load this scenario", key=f"pre{i}", width="stretch"):
                ss.lab_specs = [dict(s) for s in p["specs"]]
                ss.lab_ctx = {"time_of_day": "off_peak", "weather": "clear", "wind_toward_deg": 90, "wind_kmh": 12, **p["ctx"]}
                ss.lab_hours = p["hours"]
                ss.lab_name = name
                st.rerun()
        if i % 3 == 2:
            st.write("")

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("##### Hazards in this scenario")
        if not ss.lab_specs:
            note("Nothing yet. Load a case above or add a hazard below. Hazards can overlap in time and space.")
        for i, s in enumerate(list(ss.lab_specs)):
            a, b = st.columns([8, 1])
            a.markdown(describe(s))
            if b.button("✕", key=f"rm{i}", help="Remove"):
                ss.lab_specs.pop(i)
                st.rerun()

        with st.expander("➕ Add a hazard", expanded=not ss.lab_specs):
            htype = st.selectbox("Type", list(HAZARD_TYPES), format_func=HAZARD_TYPES.get, key="add_type")
            spec = {"type": htype}
            if htype in NEEDS_PLACE:
                pname, (plat, plon) = place_picker("add_place", "Where", PLACES, default="Gandhipuram")
                spec.update(place=pname, lat=plat, lon=plon)
            if htype == "earthquake":
                spec["magnitude"] = st.slider("Magnitude", 4.5, 7.5, 6.0, 0.1, key="add_mag")
                spec["secondary_fires"] = st.checkbox("Ignite secondary fires after the shaking", True, key="add_sf")
            elif htype == "flood_zone":
                how = st.radio("Where does the water come from?", ["Along a mapped water body", "Around a point"], horizontal=True, key="add_flood_how")
                if how.startswith("Along"):
                    spec["water"] = st.selectbox("Water body", list(city.water), key="add_water")
                else:
                    pname, (plat, plon) = place_picker("add_fplace", "Centre", PLACES, default="Ukkadam")
                    spec.update(place=pname, lat=plat, lon=plon)
                spec["severity"] = st.slider("Severity", 1, 5, 3, key="add_fsev")
            elif htype == "structure_failure":
                idx = st.selectbox("Structure", range(len(city.structures)), format_func=lambda i: f"{city.structures[i]['name']} ({city.structures[i]['length_m']:.0f} m)", key="add_struct")
                spec["structure"] = int(idx)
                spec["mode"] = st.radio("What happens", ["collapse", "closure", "partial"], horizontal=True, key="add_mode",
                                        help="Collapse: closed, with casualties. Closure: closed, no casualties. Partial: reduced capacity.")
            elif htype == "custom":
                spec["radius_m"] = st.slider("Radius (m)", 50, 2000, 300, 50, key="add_rad")
                spec["closed"] = st.checkbox("Roads inside are closed (otherwise only slowed)", True, key="add_closed")
                if not spec["closed"]:
                    spec["slow_factor"] = st.slider("Speed left inside (share of normal)", 0.05, 0.9, 0.3, 0.05, key="add_slow")
                spec["duration_min"] = st.slider("Lasts (minutes)", 30, 720, 240, 30, key="add_dur")
                spec["casualties"] = st.number_input("People needing rescue", 0, 200, 0, key="add_cas")
            else:
                spec["severity"] = st.slider("Severity", 1, 5, 3, key="add_sev")
            spec["start_min"] = st.number_input("Starts (minutes after the scenario begins)", 0, 720, 0, 10, key="add_start")
            if st.button("Add to scenario", type="primary", key="add_go"):
                ss.lab_specs.append(spec)
                st.rerun()

    with right:
        st.markdown("##### Conditions")
        ctx = ss.lab_ctx
        c1, c2 = st.columns(2)
        ctx["time_of_day"] = c1.selectbox("Time of day", ["night", "off_peak", "peak"], index=["night", "off_peak", "peak"].index(ctx["time_of_day"]),
                                          format_func=lambda x: {"night": "Night", "off_peak": "Daytime, off-peak", "peak": "Rush hour"}[x])
        ctx["weather"] = c2.selectbox("Weather", ["clear", "light_rain", "heavy_rain", "storm"], index=["clear", "light_rain", "heavy_rain", "storm"].index(ctx["weather"]),
                                      format_func=lambda x: x.replace("_", " ").title())
        c3, c4 = st.columns(2)
        ctx["wind_toward_deg"] = c3.slider("Wind blows towards (° from north)", 0, 359, int(ctx["wind_toward_deg"]), 15, help="Matters for gas and smoke.")
        ctx["wind_kmh"] = c4.slider("Wind speed (km/h)", 0, 60, int(ctx["wind_kmh"]), 2)
        ss.lab_hours = st.slider("Simulate (hours)", 2, 24, int(ss.lab_hours))
        c5, c6 = st.columns(2)
        ambulances = c5.slider("Ambulances", 10, 150, 60, 5)
        crews = c6.slider("Debris-clearing crews", 1, 12, 3)
        c7, c8 = st.columns(2)
        seed = c7.number_input("Random seed", 0, 9999, 1, help="Same seed, same scenario: the run is reproducible.")
        inc_scale = c8.slider("Call volume", 0.5, 3.0, 1.0, 0.25, help="Multiplies how many people call for help.")

    st.divider()
    b1, b2 = st.columns([1, 3])
    run = b1.button("▶ Run the simulation", type="primary", width="stretch", disabled=not ss.lab_specs)
    if not ss.lab_specs:
        b2.markdown('<div class="note" style="padding-top:8px">Add at least one hazard to run.</div>', unsafe_allow_html=True)
    if run:
        sc = make_scenario(city, ss.lab_name, ss.lab_specs, ss.lab_ctx, ss.lab_hours, int(seed), {"ambulances": ambulances, "crews": crews}, inc_scale)
        bar = st.progress(0.0, text="Starting…")
        res = Simulation(city, sc, A, router).run(progress=lambda f, msg: bar.progress(min(f, 1.0), text=f"Simulating {msg}"))
        bar.empty()
        ss.lab_res, ss.lab_trips, ss.lab_specs_run = res, TripBook(city, res.legs), [dict(s) for s in ss.lab_specs]
        ss.lab_t, ss.lab_playing, ss.lab_cache = 0, False, {}
        st.success(f"Done in {res.runtime_s:.1f} s. Open “2 · Watch it unfold”.")

res = ss.lab_res

# ================================================================================================= WATCH
with t_watch:
    if res is None:
        st.info("Run a scenario in the first tab and it will play here.")
    else:
        T = int(res.times[-1])
        base = res.baseline["summary"]
        hp = res.hazard_points
        clat = float(np.mean([h["lat"] for h in hp])) if hp else city.centre[0]
        clon = float(np.mean([h["lon"] for h in hp])) if hp else city.centre[1]

        top = st.columns([1, 1, 4])
        if top[0].button("⏸ Pause" if ss.lab_playing else "▶ Play", width="stretch"):
            ss.lab_playing = not ss.lab_playing
            if ss.lab_playing and ss.lab_t >= T:
                ss.lab_t = 0
            st.rerun()
        speed = top[1].selectbox("Speed", [2, 5, 10, 20], index=1, format_func=lambda x: f"{x} min / s", label_visibility="collapsed")
        LAYERS = ["Vehicles", "Calls", "Closed roads", "Slow roads", "People cut off", "Hazard", "Hospitals"]
        top[2].markdown('<div class="note" style="padding-top:10px">Choose what to draw below the slider. Hover any road, dot or vehicle for details.</div>', unsafe_allow_html=True)
        on = st.pills("Layers", LAYERS, selection_mode="multi", default=[x for x in LAYERS if x != "Slow roads"], label_visibility="collapsed", key="lab_layers") or []
        show_units, show_inc, show_blk, show_slow, show_cut, show_haz, show_hosp = (x in on for x in LAYERS)

        def watch():
            if ss.lab_playing:
                ss.lab_t = min(T, int(ss.lab_t) + int(speed))
            t = st.slider("Time", 0, T, key="lab_t", format="%d min", label_visibility="collapsed")
            fi = max(0, bisect.bisect_right(res.times, t) - 1)
            fr, m = res.frames[fi], res.metrics[fi]
            hh, mm = divmod(int(t), 60)
            kpi_row([
                ("Clock", f"{hh}:{mm:02d}", f"hour {t / 60:.1f} of {T / 60:.0f}"),
                ("30+ min from a hospital", fmt_people(m["gt30_any"]), f"normal: {fmt_people(base['gt30_any'])}", PAL["amber"]),
                ("Cut off from hospitals", fmt_people(m["cut_any"]), "no road route", PAL["red"] if m["cut_any"] > 300 else None),
                ("Roads closed", f"{m['blocked_km']:.1f} km", f"{m['repaired_km']:.1f} km reopened"),
                ("Open calls", str(m["incidents_open"]), f"{m['served']} served so far"),
                ("Hospitals out", str(m["hospitals_closed"]), ""),
            ])
            cache = ss.setdefault("lab_cache", {})
            if ("blk", fi) not in cache:
                cache[("blk", fi)] = L.edge_paths(city, fr["blocked"], 5500, "closed road")
                cache[("slow", fi)] = L.edge_paths(city, fr["slowed"], 2500, "slowed road")
            layers = [L.road_layer(base_paths(), "#3B4C6B", 1.1, 130, "base")]
            if show_cut and len(fr["cut_cells"][0]):
                idx, ppl = fr["cut_cells"]
                layers.append(L.columns(city.cell_lat[idx], city.cell_lon[idx], ppl, "cut", color_hi="#F43F5E", color_lo="#7F1D1D", radius=240, max_h=1800, tip_fmt="{:,.0f} people cut off"))
            if show_haz:
                layers += L.hazard_layers(fr["prims"])
            if show_slow:
                layers.append(L.road_layer(cache[("slow", fi)], PAL["amber"], 2.2, 220, "slow", True))
            if show_blk:
                layers += L.glow_roads(cache[("blk", fi)], "#FF3B5C", "blk")
            if show_hosp:
                rows = L.hospital_rows(city, set(fr["closed_fac"]), fr["hospital_ratio"], tiers=(1,)) + [r for r in L.hospital_rows(city, set(fr["closed_fac"]), {}, tiers=(2,)) if "OUT OF" in r["tip"]]
                layers += [L.dots(rows, "hosp", 9), L.labels([r for r in rows if r["text"] and ("OUT" in r["tip"] or "cut off" in r["tip"])], "hlab", 13)]
            if show_inc:
                layers.append(L.dots(L.incident_rows(city, res.incidents, t), "inc", 5, stroke=True))
            if show_units:
                layers.append(L.trips(ss.lab_trips.window(t, 7.0), t, 7.0))
            show_map(L.deck(layers, L.view(clat, clon, 11.3, 45, -14)), height=640, key="labmap")
            legend([("#FF3B5C", "closed road"), (PAL["amber"], "slowed"), (PAL["red"], "people with no route to a hospital"), (PAL["teal"], "hospital working"), (PAL["amber"], "hospital cut off"),
                    ("#F43F5E", "hospital out"), (UNIT_COLOR["ambulance"], "ambulance"), (UNIT_COLOR["fire"], "fire engine"), (UNIT_COLOR["rescue"], "rescue"), (UNIT_COLOR["crew"], "clearing crew")])
            st.altair_chart(timeline_chart(res.metrics, base, t), width="stretch")
            if ss.lab_playing and t >= T:
                ss.lab_playing = False
                st.rerun()

        frag = st.fragment(run_every=(1.0 if ss.lab_playing else None))(watch)
        frag()

# ================================================================================================= FINDINGS
with t_find:
    if res is None:
        st.info("Findings appear here after a run.")
    else:
        s = res.summary
        base = s["baseline"]
        pk = res.peak
        st.subheader(res.scenario_name)
        kpi_row([
            ("Worst moment", f"{pk['t'] / 60:.1f} h", "when most people were far from care"),
            ("Extra people 30+ min away", fmt_people(max(pk["gt30_any"] - base["gt30_any"], 0)), f"{max(pk['gt30_any'] - base['gt30_any'], 0) / s['population']:.1%} of the city", PAL["amber"]),
            ("Person-hours without timely care", fmt_people(s["person_hours_without_timely_care"]), "on top of a normal day"),
            ("Recovered", "no" if s["recovered_at_min"] is None and not s["never_hit"] else ("never hit" if s["never_hit"] else f"hour {s['recovered_at_min'] / 60:.1f}"), "to within 10% of the peak burden"),
            ("Calls", str(s["incidents_total"]), f"{s['incidents_failed']} never served or delivered"),
        ])
        st.write("")
        a, b = st.columns([1.3, 1])
        with a:
            st.markdown("##### What happened")
            findings(s["findings"])
            if s["top_blocked_roads"]:
                note("Roads with the longest closed stretches at the peak: " + ", ".join(s["top_blocked_roads"]))
        with b:
            st.markdown("##### Response by kind of call")
            rows = []
            for k, label in (("medical", "Medical"), ("fire", "Fire"), ("rescue", "Rescue")):
                bk = s["by_kind"][k]
                rm = bk["response_min"]
                rows.append({"Calls": label, "Total": bk["total"], "Served": bk["served"], "Never served": bk["unserved"] + bk["stranded"],
                             "Median first arrival (min)": None if rm["median"] is None else round(rm["median"], 1), "90th pct (min)": None if rm["p90"] is None else round(rm["p90"], 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            gd = s["golden"]
            if gd["critical_total"]:
                st.metric(f"Critical patients delivered within {gd['minutes']} min", f"{gd['within']} of {gd['critical_total']}")
            st.markdown("##### Major hospitals")
            hh = [{"Hospital": n, "Worst reach (share of normal)": f"{r:.0%}", "Status": "out of service" if c else "open, cut off"} for n, r, c in s["hospitals_hit"] if r < 0.7]
            st.dataframe(pd.DataFrame(hh), hide_index=True, width="stretch") if hh else note("No major hospital lost more than 30% of its usual reach.")
        st.markdown("##### Data")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Timeline (CSV)", pd.DataFrame(res.metrics).to_csv(index=False), "timeline.csv", "text/csv", width="stretch")
        d2.download_button("⬇ Summary (JSON)", json.dumps({k: v for k, v in s.items() if k not in ("baseline",)}, default=float, indent=1), "summary.json", "application/json", width="stretch")
        if s["data_flags"]["fire_stations_mapped"] < A["advisor"]["min_fire_stations"]:
            st.warning(f"Only {s['data_flags']['fire_stations_mapped']} fire station(s) are mapped, so fire-response numbers are pessimistic. See Data & assumptions.")
