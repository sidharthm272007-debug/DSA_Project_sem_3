"""Reusable pieces of interface. Pages compose these; nothing here knows about a specific page."""
from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from advisor.portfolio import KIND_LABEL
from advisor.verdict import Verdict
from sim.report import fmt_people
from ui.state import visuals
from viz.theme import CSS, HAZARD_COLOR, PAL, TONE


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, sub: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", color: str | None = None) -> str:
    c = f"color:{color};" if color else ""
    return f'<div class="kpi"><div class="v" style="{c}">{html.escape(value)}</div><div class="l">{html.escape(label)}</div><div class="d" style="color:{PAL["muted"]}">{html.escape(note)}</div></div>'


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def badge(text: str, tone: str) -> str:
    bg, fg = TONE[tone]
    return f'<span class="badge" style="background:{bg};color:{fg};border:1px solid {fg}55">{html.escape(text)}</span>'


def verdict_card(v: Verdict, title: str | None = None) -> None:
    bg, fg = TONE[v.tone]
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in v.reasons)
    t = f'<div style="color:{PAL["muted"]};font-size:.8rem;margin-bottom:6px">{html.escape(title)}</div>' if title else ""
    st.markdown(f'<div class="verdict" style="background:{bg};border-color:{fg}66">{t}{badge(v.label, v.tone)}<div class="head">{html.escape(v.headline)}</div><ul>{reasons}</ul></div>',
                unsafe_allow_html=True)
    with st.expander("What this rests on"):
        for c in v.caveats:
            st.markdown(f"- {c}")


def findings(items: list[str]) -> None:
    for f in items:
        st.markdown(f'<div class="finding">{html.escape(f)}</div>', unsafe_allow_html=True)


def legend(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{html.escape(t)}</span>' for c, t in items) + "</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{html.escape(text)}</div>', unsafe_allow_html=True)


# ---- charts -------------------------------------------------------------------------------------------------
def dark(ch: alt.Chart, height: int = 230) -> alt.Chart:
    return (ch.properties(height=height, background="transparent")
            .configure_axis(labelColor=PAL["muted"], titleColor=PAL["muted"], gridColor=PAL["line"], domainColor=PAL["line"], tickColor=PAL["line"])
            .configure_legend(labelColor=PAL["muted"], titleColor=PAL["muted"], orient="top")
            .configure_view(strokeOpacity=0))


def by_kind_chart(by_kind: dict[str, float], normal: float | None = None, unit: str = "people newly within the standard") -> alt.Chart:
    rows = ([{"situation": "Normal day", "value": normal, "kind": "baseline"}] if normal is not None else []) + \
           [{"situation": KIND_LABEL.get(k, k), "value": v, "kind": k} for k, v in by_kind.items()]
    df = pd.DataFrame(rows)
    order = list(df["situation"])
    colors = {"baseline": PAL["teal"], "earthquake": HAZARD_COLOR["earthquake"], "fire": HAZARD_COLOR["fire"], "flood": HAZARD_COLOR["flood"],
              "structure": HAZARD_COLOR["structure"], "storm": HAZARD_COLOR["storm"]}
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("situation:N", sort=order, title=None), x=alt.X("value:Q", title=unit),
        color=alt.Color("kind:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        tooltip=[alt.Tooltip("situation:N"), alt.Tooltip("value:Q", format=",.0f", title=unit)])
    return dark(ch, height=max(120, 34 * len(df)))


def timeline_chart(metrics: list[dict], baseline: dict, t_now: float | None) -> alt.Chart:
    df = pd.DataFrame(metrics)
    hours = df["t"] / 60.0
    long = pd.concat([
        pd.DataFrame({"h": hours, "people": df["gt30_any"], "series": "More than 30 min from any hospital"}),
        pd.DataFrame({"h": hours, "people": df["cut_any"], "series": "No road route to any hospital"}),
        pd.DataFrame({"h": hours, "people": df["gt60_t1"], "series": "More than 60 min from a major hospital"}),
    ])
    ch = alt.Chart(long).mark_line(strokeWidth=2.4, interpolate="monotone").encode(
        x=alt.X("h:Q", title="hours since the event"), y=alt.Y("people:Q", title="people"),
        color=alt.Color("series:N", scale=alt.Scale(range=[PAL["amber"], PAL["red"], PAL["violet"]])),
        tooltip=[alt.Tooltip("h:Q", format=".1f", title="hour"), "series:N", alt.Tooltip("people:Q", format=",.0f")])
    layers = [ch]
    if t_now is not None:
        layers.append(alt.Chart(pd.DataFrame({"h": [t_now / 60.0]})).mark_rule(color=PAL["teal"], strokeDash=[5, 4], strokeWidth=2).encode(x="h:Q"))
    return dark(alt.layer(*layers), 230)


def bars(df: pd.DataFrame, label: str, value: str, title: str, color: str = PAL["teal"], height: int | None = None, fmt: str = ",.0f") -> alt.Chart:
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=color).encode(
        y=alt.Y(f"{label}:N", sort="-x", title=None), x=alt.X(f"{value}:Q", title=title), tooltip=[label, alt.Tooltip(f"{value}:Q", format=fmt)])
    return dark(ch, height or max(140, 28 * len(df)))


# ---- inputs --------------------------------------------------------------------------------------------------
def place_picker(key: str, label: str, places: dict[str, tuple[float, float]], default: str | None = None) -> tuple[str, tuple[float, float]]:
    """Choose a named place (type to search) or, optionally, enter coordinates. Returns (name, (lat, lon))."""
    names = list(places)
    if f"{key}_sel" not in st.session_state:
        st.session_state[f"{key}_sel"] = default if default in places else names[0]
    name = st.selectbox(label, names, key=f"{key}_sel", label_visibility="collapsed", help="Type to search localities, hospitals, stations and flyovers.")
    if not st.checkbox("Enter coordinates instead", key=f"{key}_c"):
        return name, places[name]
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", value=float(places[name][0]), format="%.5f", key=f"{key}_lat")
    lon = c2.number_input("Longitude", value=float(places[name][1]), format="%.5f", key=f"{key}_lon")
    return f"{lat:.4f}, {lon:.4f}", (float(lat), float(lon))


PRIORITY_PRESETS = {
    "Balanced": None,
    "Everyday first": {"baseline": 0.6, "earthquake": 0.1, "fire": 0.1, "flood": 0.08, "structure": 0.08, "storm": 0.04},
    "Disaster-proofing": {"baseline": 0.05, "earthquake": 0.3, "fire": 0.2, "flood": 0.2, "structure": 0.15, "storm": 0.1},
    "Flood & storm focus": {"baseline": 0.1, "earthquake": 0.05, "fire": 0.05, "flood": 0.5, "structure": 0.05, "storm": 0.25},
}


def priorities_panel(A: dict, adv) -> None:
    """What matters most? The advisor judges every idea across a normal day and several kinds of disaster; this sets how much each counts.

    State lives in non-widget session keys: Streamlit forgets widget state for widgets that were not drawn on the previous page,
    so the values are handed back to the sliders each time instead of being read from them.
    """
    ss = st.session_state
    kinds = ["baseline", "earthquake", "fire", "flood", "structure", "storm"]
    default = {"baseline": A["advisor"]["portfolio"]["baseline_weight"], **A["advisor"]["portfolio"]["kind_weight"]}
    names = list(PRIORITY_PRESETS)
    with st.expander("⚖️ What matters most to you? (how much each situation counts)", expanded=False):
        choice = st.radio("Preset", names, index=names.index(ss.get("prio_choice", "Balanced")), horizontal=True, key="prio_preset")
        if choice != ss.get("prio_choice", "Balanced") or "prio_vals" not in ss:
            ss["prio_choice"] = choice
            ss["prio_vals"] = {k: int(round(v * 100)) for k, v in (PRIORITY_PRESETS[choice] or default).items()}
        cols = st.columns(len(kinds))
        vals = {k: col.slider(KIND_LABEL[k], 0, 100, value=int(ss["prio_vals"].get(k, 0)), key=f"prio_s_{choice}_{k}") for col, k in zip(cols, kinds)}
        ss["prio_vals"] = vals
        tot = sum(vals.values())
        if tot <= 0:
            st.warning("At least one situation has to count. Using the balanced defaults until you raise one.")
            ss.kind_weights = None
        else:
            ss.kind_weights = {k: v / tot for k, v in vals.items()}
        adv.reweight(ss.kind_weights)
        note("Weights are relative and are normalised. Every result on this page is re-scored when they change; the answers are only as good as these choices, so they are yours to set.")


def show_map(deck, **kw) -> None:
    """st.pydeck_chart, unless the user has switched maps off (some laptops cannot draw them; everything else still works)."""
    if visuals() == "off":
        st.info("Maps are switched off (sidebar: Map detail). The tables and text on this page carry the same results.")
        return
    st.pydeck_chart(deck, **kw)
