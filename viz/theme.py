"""Design tokens shared by every page: one palette, one type system, one icon system.

Design direction (see DESIGN_NOTES.md for the full plan): a civic infrastructure and emergency-operations tool should read
like the serious software professionals already use for this work -- engineering and planning tools, operations consoles --
not like a generated startup dashboard. Concretely that means: a blueprint-adjacent steel-blue accent instead of a neon
glow colour, status colour (red/amber/green) used only for status and never as decoration, Public Sans (the U.S. civic
design system's typeface) for every size of text, no gradient hero washes, and Material Symbols -- the same icon set
Streamlit's own widgets already render -- used everywhere instead of emoji.
"""
from __future__ import annotations

from viz._public_sans_b64 import PUBLIC_SANS_WOFF2_B64

PAL = {
    "bg": "#0B0E14", "panel": "#12151E", "panel2": "#1A1E2A", "line": "#262B3A", "line2": "#333A4D",
    "text": "#E8EAEE", "muted": "#8B93A7", "faint": "#5C6379",
    "accent": "#5B9BD8", "accent_dim": "#3A6E9E", "accent_tint": "#1B2A3A",
    "red": "#D65C55", "red_tint": "#2A1616", "amber": "#C9973F", "green": "#4C9F63", "blue": "#5B9BD8", "violet": "#9C8CD8",
    "slate": "#6B7386", "orange": "#C97A3F",
}
# Map-only hues (hazard footprints, unit trails): a wider, slightly more saturated set is legitimate here because it is
# functional legend colour on a map, not brand chrome -- but every one of them is still muted, none of them glow.
HAZARD_COLOR = {"earthquake": "#C9793E", "fire": "#D65C55", "plume": "#8FAF56", "flood_zone": "#5B9BD8", "flood": "#5B9BD8",
                "structure": "#B27BC4", "structure_failure": "#B27BC4", "accident": "#C9973F", "storm": "#8A93A8", "custom": "#C9AA3F", "quake": "#C9793E"}
UNIT_COLOR = {"ambulance": "#E8EAEE", "fire": "#D68A55", "rescue": "#9C8CD8", "crew": "#C9AA3F"}
TONE = {"good": ("#16241A", "#4C9F63"), "mid": ("#2A2114", "#C9973F"), "weak": ("#1A1E2A", "#8B93A7")}
TONE_ICON = {"good": "check_circle", "mid": "error", "weak": "remove_circle"}


def hex_rgb(h: str, a: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def mi(name: str, size: str = "1em", color: str | None = None, extra: str = "") -> str:
    """One Material Symbols glyph as an HTML fragment, for use inside hand-built HTML (cards, headers, verdict panels).
    Renders with the exact same bundled font Streamlit's own icon=":material/...:" widgets use, so it always matches them."""
    style = f"font-size:{size};" + (f"color:{color};" if color else "") + extra
    return f'<span class="mi" style="{style}">{name}</span>'


CSS = f"""
<style>
/* Public Sans, embedded (see viz/_public_sans_b64.py): no request to Google Fonts, so the typeface still renders on a
   network that blocks it -- a plain "@import url(fonts.googleapis.com/...)" silently falls back to a generic system
   font with no warning on such a network, which is a real risk for a typeface this deliberately chosen. One variable
   file (weight 100-900) covers every weight this stylesheet uses, including the in-between values (550, 650) below. */
@font-face {{
  font-family: "Public Sans"; font-style: normal; font-weight: 100 900; font-display: swap;
  src: url(data:font/woff2;base64,{PUBLIC_SANS_WOFF2_B64}) format("woff2-variations");
}}

:root {{
  --bg: {PAL['bg']}; --panel: {PAL['panel']}; --panel2: {PAL['panel2']}; --line: {PAL['line']}; --line2: {PAL['line2']};
  --text: {PAL['text']}; --muted: {PAL['muted']}; --faint: {PAL['faint']}; --accent: {PAL['accent']}; --accent-dim: {PAL['accent_dim']}; --accent-tint: {PAL['accent_tint']};
}}

html, body {{ font-family: "Public Sans", -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.block-container {{ padding-top: 2.75rem; padding-bottom: 3rem; max-width: 1420px; }}
/* Streamlit's own stylesheet pins headings to its default typeface with a selector more specific than a plain "h1{{}}" rule,
   so headings need "!important" to actually take Public Sans -- but that must never reach the icon glyphs Streamlit renders
   as <span role="img"> with the Material Symbols font, or every native icon in the app silently turns back into literal text
   (found by screenshot: this exact regression happened once already). Everything except headings and those icon spans
   already inherits "Public Sans" correctly from the plain html/body rule above, with no need for "!important" there. */
h1, h2, h3, h4, h5, h6 {{
  font-family: "Public Sans", -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; letter-spacing: -0.012em !important; font-weight: 650 !important;
}}
h1 span[role="img"], h2 span[role="img"], h3 span[role="img"], h4 span[role="img"], h5 span[role="img"], h6 span[role="img"], .mi {{
  font-family: "Material Symbols Rounded", "Material Icons" !important; letter-spacing: normal !important; font-weight: 400 !important;
}}
::selection {{ background: {PAL['accent_dim']}; color: #fff; }}

/* Material Symbols glyph, reusing the exact font Streamlit's own icon=":material/...:" already loads (see viz/theme.py:mi()). */
.mi {{ font-family: "Material Symbols Rounded", "Material Icons"; font-weight: 400; font-style: normal; display: inline-block;
      line-height: 1; vertical-align: -14%; white-space: nowrap; -webkit-font-smoothing: antialiased; }}

/* ---------------------------------------------------------------------------------------------- sidebar / nav */
[data-testid="stSidebar"] {{ background: var(--bg); border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: var(--muted); }}
.brand {{ display: flex; align-items: center; gap: 10px; padding: 2px 0 2px 0; }}
.brand .mi {{ font-size: 1.35rem; color: var(--accent); }}
.brand .name {{ font-size: 1.05rem; font-weight: 700; color: var(--text); line-height: 1.1; }}
.brand .sub {{ font-size: .76rem; color: var(--faint); margin-top: 1px; }}

/* ---------------------------------------------------------------------------------------------- page header (replaces the old gradient "hero") */
.pageh {{ padding-bottom: 14px; margin-bottom: 20px; border-bottom: 1px solid var(--line); }}
.pageh .row {{ display: flex; align-items: center; gap: 10px; }}
.pageh .row .mi {{ font-size: 1.3rem; color: var(--accent); }}
.pageh h1 {{ margin: 0; font-size: 1.55rem; line-height: 1.25; font-weight: 650; }}
.pageh p {{ color: var(--muted); font-size: .95rem; max-width: 62rem; margin: 6px 0 0 0; line-height: 1.55; }}

/* ---------------------------------------------------------------------------------------------- section (a light-weight substitute for wrapping every block in a card) */
.section-label {{ display: flex; align-items: center; gap: 8px; font-size: 1.05rem; font-weight: 650; color: var(--text); margin: 4px 0 12px 0; }}
.section-label .mi {{ font-size: 1.05rem; color: var(--muted); }}
.feature {{ padding-right: 8px; }}
.feature .head {{ display: flex; align-items: center; gap: 9px; margin-bottom: 5px; }}
.feature .head .mi {{ font-size: 1.05rem; color: var(--accent); }}
.feature .head b {{ font-size: .95rem; font-weight: 650; color: var(--text); }}
.feature p {{ color: var(--muted); font-size: .875rem; line-height: 1.55; margin: 0; }}

/* ---------------------------------------------------------------------------------------------- cards: reserved for discrete, comparable units (KPIs, verdicts, entry points) */
/* A soft, low drop-shadow (not a glow) so panels read as raised surfaces against the near-black page background --
   the one depth cue a flat outline-only dark theme is missing; kept subtle enough to still look like engineering
   software, not a glossy consumer app. */
.card, .kpi, .verdict, div[data-testid="stMetric"] {{ box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 6px 16px -10px rgba(0,0,0,.55); }}
.card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 10px; padding: 16px 18px; height: 100%; }}
.card.tall {{ min-height: 9.6rem; }}
.card h4 {{ margin: 0 0 .35rem 0; font-size: .95rem; font-weight: 650; display: flex; align-items: center; gap: 8px; }}
.card h4 .mi {{ font-size: 1.05rem; color: var(--accent); }}
.card p, .card li {{ color: var(--muted); font-size: .875rem; margin: 0; line-height: 1.55; }}

.kpi {{ border: 1px solid var(--line); background: var(--panel); border-radius: 10px; padding: 13px 16px; min-height: 7.2rem; }}
.kpi .v {{ font-size: 1.5rem; font-weight: 650; line-height: 1.2; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }}
.kpi .l {{ color: var(--muted); font-size: .8rem; margin-top: 3px; word-break: normal; hyphens: none; }}
.kpi .d {{ font-size: .78rem; margin-top: 3px; color: var(--faint); }}

.badge {{ display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 5px; font-size: .78rem; font-weight: 600; }}
.badge .mi {{ font-size: .95rem; }}

.verdict {{ border-radius: 10px; padding: 16px 18px; border: 1px solid var(--line); border-left: 3px solid; margin-bottom: 10px; background: var(--panel); }}
.verdict .kicker {{ color: var(--faint); font-size: .78rem; margin-bottom: 8px; }}
.verdict .head {{ font-size: 1rem; font-weight: 600; margin-top: 10px; line-height: 1.5; color: var(--text); }}
.verdict ul {{ margin: .65rem 0 0 1.05rem; padding: 0; color: var(--text); font-size: .89rem; }}
.verdict li {{ margin-bottom: .3rem; line-height: 1.5; }}

.pill {{ display: inline-block; padding: 2px 9px; margin: 0 6px 6px 0; border-radius: 5px; background: var(--panel2); border: 1px solid var(--line); font-size: .78rem; color: var(--muted); }}
.finding {{ display: flex; gap: 9px; padding: 7px 2px; border-bottom: 1px solid var(--line); font-size: .89rem; line-height: 1.5; }}
.finding:last-child {{ border-bottom: none; }}
.finding .mi {{ font-size: 1rem; color: var(--accent); margin-top: .1rem; flex: none; }}
.note {{ color: var(--muted); font-size: .81rem; line-height: 1.5; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 16px; }}
.legend span {{ display: inline-flex; align-items: center; font-size: .78rem; color: var(--muted); }}
.legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; flex: none; }}

/* ---------------------------------------------------------------------------------------------- native Streamlit widgets: quieter chrome, same one accent everywhere */
div[data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--line); padding: 10px 14px; border-radius: 10px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 550; font-size: .9rem; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--accent); }}
.stTabs [aria-selected="true"] {{ color: var(--text) !important; }}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{ background: var(--accent); border-color: var(--accent); }}
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover {{ background: var(--accent-dim); border-color: var(--accent-dim); }}
.stButton button, .stDownloadButton button {{ border-radius: 7px; }}
[data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 8px; }}
[role="radiogroup"] label, .stCheckbox label {{ font-size: .9rem; }}
a {{ color: var(--accent); }}
hr {{ border-color: var(--line); }}
</style>
"""
