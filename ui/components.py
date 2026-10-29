"""Reusable pieces of interface. Pages compose these; nothing here knows about a specific page."""
from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from ui.state import visuals
from ui.verdict import Verdict
from viz.theme import CSS, PAL, TONE, TONE_ICON, mi


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, sub: str) -> None:
    """The page's title bar: one icon, the page name (matching its entry in the sidebar), one line of context. Deliberately
    compact — a working tool is opened dozens of times a day, so the header should not repeat a full landing-page pitch
    every time; it should say what page this is and get out of the way."""
    st.markdown(f'<div class="pageh"><div class="row">{mi(icon)}<h1>{html.escape(title)}</h1></div><p>{html.escape(sub)}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", color: str | None = None) -> str:
    c = f"color:{color};" if color else ""
    return f'<div class="kpi"><div class="v" style="{c}">{html.escape(value)}</div><div class="l">{html.escape(label)}</div><div class="d">{html.escape(note)}</div></div>'


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def badge(text: str, tone: str) -> str:
    bg, fg = TONE[tone]
    return f'<span class="badge" style="background:{bg};color:{fg}">{mi(TONE_ICON[tone])}{html.escape(text)}</span>'


def feature(icon: str, title: str, text: str) -> str:
    """One entry in a plain feature grid: icon, bold title, one line of muted text. No card border or background —
    used where several short items sit side by side purely to explain, not to compare discrete data (see .feature in theme.py)."""
    return f'<div class="feature"><div class="head">{mi(icon)}<b>{html.escape(title)}</b></div><p>{html.escape(text)}</p></div>'


def verdict_card(v: Verdict, title: str | None = None) -> None:
    bg, fg = TONE[v.tone]
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in v.reasons)
    t = f'<div class="kicker">{html.escape(title)}</div>' if title else ""
    st.markdown(f'<div class="verdict" style="border-left-color:{fg}">{t}{badge(v.label, v.tone)}<div class="head">{html.escape(v.headline)}</div><ul>{reasons}</ul></div>',
                unsafe_allow_html=True)
    with st.expander("What this rests on"):
        for c in v.caveats:
            st.markdown(f"- {c}")


def findings(items: list[str]) -> None:
    st.markdown("".join(f'<div class="finding">{mi("arrow_right_alt")}<div>{html.escape(f)}</div></div>' for f in items), unsafe_allow_html=True)


def legend(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{html.escape(t)}</span>' for c, t in items) + "</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{html.escape(text)}</div>', unsafe_allow_html=True)


def section(icon: str, text: str) -> None:
    """A small heading used to break up a page without the weight of st.subheader — icon plus label, one accent per section."""
    st.markdown(f'<div class="section-label">{mi(icon)}{html.escape(text)}</div>', unsafe_allow_html=True)


# ---- charts -------------------------------------------------------------------------------------------------
def dark(ch: alt.Chart, height: int = 230) -> alt.Chart:
    return (ch.properties(height=height, background="transparent")
            .configure_axis(labelColor=PAL["muted"], titleColor=PAL["muted"], gridColor=PAL["line"], domainColor=PAL["line"], tickColor=PAL["line"], labelFont="Public Sans", titleFont="Public Sans")
            .configure_legend(labelColor=PAL["muted"], titleColor=PAL["muted"], orient="top", labelFont="Public Sans", titleFont="Public Sans")
            .configure_view(strokeOpacity=0))


def pareto_chart(options: list, recommended) -> alt.Chart:
    """The actual trade-off curve behind the route recommendation: every point is a genuinely different route that
    beats everything else on at least one of time or risk (see routing/pareto.py) -- not three arbitrary picks."""
    rows = [{"time": o.time_s / 60.0, "risk": o.risk_km, "label": "Recommended" if o is recommended else ("/".join(o.tags) or "On the frontier"),
             "rec": o is recommended} for o in options]
    df = pd.DataFrame(rows)
    line = alt.Chart(df).mark_line(color=PAL["line2"], strokeDash=[3, 3], strokeWidth=1.2).encode(x="time:Q", y="risk:Q") if len(df) > 1 else alt.Chart(df).mark_point(opacity=0)
    pts = alt.Chart(df).mark_circle().encode(
        x=alt.X("time:Q", title="time (min)"), y=alt.Y("risk:Q", title="risk-km (lower is safer)"),
        size=alt.condition(alt.datum.rec, alt.value(220), alt.value(90)),
        color=alt.condition(alt.datum.rec, alt.value(PAL["accent"]), alt.value(PAL["muted"])),
        tooltip=[alt.Tooltip("label:N", title="Route"), alt.Tooltip("time:Q", format=".1f", title="Time (min)"), alt.Tooltip("risk:Q", format=".2f", title="Risk-km")])
    return dark(line + pts, 190)


def timeline_chart(metrics: list[dict], baseline: dict, t_now: float | None) -> alt.Chart:
    df = pd.DataFrame(metrics)
    hours = df["t"] / 60.0
    long = pd.concat([
        pd.DataFrame({"h": hours, "people": df["gt30_any"], "series": "More than 30 min from any hospital"}),
        pd.DataFrame({"h": hours, "people": df["cut_any"], "series": "No road route to any hospital"}),
        pd.DataFrame({"h": hours, "people": df["gt60_t1"], "series": "More than 60 min from a major hospital"}),
    ])
    ch = alt.Chart(long).mark_line(strokeWidth=2.2, interpolate="monotone").encode(
        x=alt.X("h:Q", title="hours since the event"), y=alt.Y("people:Q", title="people"),
        color=alt.Color("series:N", scale=alt.Scale(range=[PAL["amber"], PAL["red"], PAL["violet"]])),
        tooltip=[alt.Tooltip("h:Q", format=".1f", title="hour"), "series:N", alt.Tooltip("people:Q", format=",.0f")])
    layers = [ch]
    if t_now is not None:
        layers.append(alt.Chart(pd.DataFrame({"h": [t_now / 60.0]})).mark_rule(color=PAL["accent"], strokeDash=[5, 4], strokeWidth=1.6).encode(x="h:Q"))
    return dark(alt.layer(*layers), 230)


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


def show_map(deck, **kw) -> None:
    """st.pydeck_chart, unless the user has switched maps off (some laptops cannot draw them; everything else still works)."""
    if visuals() == "off":
        st.info("Maps are switched off (sidebar: Map detail). The tables and text on this page carry the same results.")
        return
    st.pydeck_chart(deck, **kw)
