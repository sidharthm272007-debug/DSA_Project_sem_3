"""Colours and CSS shared by every page. One palette so the maps, charts and cards look like one product."""
from __future__ import annotations

PAL = {
    "bg": "#0A1020", "panel": "#111B30", "panel2": "#16233D", "line": "#22314F", "text": "#E2E8F0", "muted": "#8FA0BC",
    "teal": "#2DD4BF", "amber": "#F59E0B", "red": "#F43F5E", "blue": "#38BDF8", "violet": "#A78BFA", "green": "#4ADE80", "slate": "#64748B",
}
HAZARD_COLOR = {"earthquake": "#F97316", "fire": "#F43F5E", "plume": "#A3E635", "flood_zone": "#38BDF8", "flood": "#38BDF8", "structure": "#E879F9",
                "structure_failure": "#E879F9", "accident": "#FBBF24", "storm": "#94A3B8", "custom": "#FACC15", "quake": "#F97316"}
UNIT_COLOR = {"ambulance": "#F8FAFC", "fire": "#FB923C", "rescue": "#A78BFA", "crew": "#FACC15"}
TONE = {"good": ("#052E2B", "#2DD4BF"), "mid": ("#3A2A05", "#F59E0B"), "weak": ("#1E293B", "#94A3B8")}


def hex_rgb(h: str, a: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


CSS = f"""
<style>
.block-container {{ padding-top: 4rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0D1528 0%,#0A1020 100%); border-right: 1px solid {PAL['line']}; }}
.hero {{ position: relative; overflow: hidden; border: 1px solid {PAL['line']}; border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;
        background: radial-gradient(1200px 320px at 85% -20%, rgba(45,212,191,.18), transparent 60%), radial-gradient(900px 280px at 0% 120%, rgba(244,63,94,.14), transparent 60%), {PAL['panel']}; }}
.hero .eyebrow {{ color: {PAL['teal']}; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }}
.hero h1 {{ margin: .25rem 0 .35rem 0; font-size: 2.15rem; line-height: 1.15; }}
.hero p {{ color: {PAL['muted']}; font-size: 1.02rem; max-width: 60rem; margin: 0; }}
.card {{ border: 1px solid {PAL['line']}; background: {PAL['panel']}; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.card.tall {{ min-height: 9.6rem; }}
.card h4 {{ margin: 0 0 .35rem 0; font-size: 1.02rem; }}
.card p, .card li {{ color: {PAL['muted']}; font-size: .92rem; margin: 0; }}
.kpi {{ border: 1px solid {PAL['line']}; background: {PAL['panel']}; border-radius: 14px; padding: 14px 16px; min-height: 7.4rem; }}
.kpi .v {{ font-size: 1.6rem; font-weight: 750; line-height: 1.15; overflow-wrap: anywhere; }}
.kpi .l {{ color: {PAL['muted']}; font-size: .82rem; margin-top: 4px; word-break: normal; hyphens: none; }}
.kpi .d {{ font-size: .82rem; margin-top: 4px; }}
.badge {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 700; letter-spacing: .04em; }}
.verdict {{ border-radius: 16px; padding: 18px 20px; border: 1px solid; margin-bottom: 10px; }}
.verdict .head {{ font-size: 1.08rem; font-weight: 600; margin-top: 8px; line-height: 1.4; }}
.verdict ul {{ margin: .6rem 0 0 1.1rem; padding: 0; color: {PAL['text']}; font-size: .93rem; }}
.verdict li {{ margin-bottom: .25rem; }}
.pill {{ display:inline-block; padding: 2px 9px; margin: 0 6px 6px 0; border-radius: 8px; background: {PAL['panel2']}; border: 1px solid {PAL['line']}; font-size: .8rem; color: {PAL['muted']}; }}
.finding {{ border-left: 3px solid {PAL['teal']}; padding: 6px 12px; margin: 6px 0; background: rgba(45,212,191,.05); border-radius: 0 8px 8px 0; font-size: .95rem; }}
.note {{ color: {PAL['muted']}; font-size: .82rem; }}
.legend span {{ display:inline-flex; align-items:center; margin-right:14px; font-size:.8rem; color:{PAL['muted']}; }}
.legend i {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; }}
div[data-testid="stMetric"] {{ background: {PAL['panel']}; border: 1px solid {PAL['line']}; padding: 10px 14px; border-radius: 12px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
"""
