"""ResQGraph: emergency access and infrastructure planning for Coimbatore.

Start it with:   python -m streamlit run app.py        (or double-click run_windows.bat / run_mac_linux.sh)
"""
import importlib.util
import sys

HELP = """
ResQGraph is a Streamlit app. Start it like this, from the folder that contains app.py:

    python -m pip install -r requirements.txt      (first time only)
    python -m streamlit run app.py

On Windows you can also double-click  run_windows.bat
"""

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed." + HELP)
    raise SystemExit(1)

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    _under_streamlit = get_script_run_ctx() is not None
except Exception:                                   # a different Streamlit layout: assume we are being run properly
    _under_streamlit = True
if not _under_streamlit:                            # someone ran `python app.py`
    print("This file must be started with Streamlit, not with plain python." + HELP)
    raise SystemExit(1)

st.set_page_config(page_title="ResQGraph · Coimbatore", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

_REQUIRED = {"numpy": "numpy", "scipy": "scipy", "yaml": "pyyaml", "pandas": "pandas", "pydeck": "pydeck", "altair": "altair"}
_missing = [pkg for mod, pkg in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _missing:
    st.error(f"Some packages this app needs are not installed: **{', '.join(_missing)}**.")
    st.markdown("Stop the app (Ctrl+C in the terminal), then run this in the folder that contains `app.py`:")
    st.code("python -m pip install -r requirements.txt\npython -m streamlit run app.py", language="bash")
    st.caption(f"Python in use: {sys.executable}")
    st.stop()

from ui.components import inject_css  # noqa: E402

inject_css()
from ui.state import warm_up  # noqa: E402

warm_up()
nav = st.navigation({
    "": [st.Page("ui/pages/home.py", title="Overview", icon="🏙️", default=True)],
    "In an emergency": [st.Page("ui/pages/route.py", title="Best route now", icon="🧭"), st.Page("ui/pages/lab.py", title="Disaster lab", icon="🌪️")],
    "Planning the city": [st.Page("ui/pages/build.py", title="Where to build", icon="🏗️"), st.Page("ui/pages/idea.py", title="Test an idea", icon="💡"),
                          st.Page("ui/pages/existing.py", title="Is it useful?", icon="🔍")],
    "About": [st.Page("ui/pages/sources.py", title="Data & assumptions", icon="🧾")],
})
import os  # noqa: E402

with st.sidebar:
    st.markdown("### ResQGraph")
    st.caption("Emergency access planning · Coimbatore")
    st.session_state.setdefault("visuals", os.environ.get("RESQGRAPH_VISUALS", "light") if os.environ.get("RESQGRAPH_VISUALS", "light") in ("light", "rich", "off") else "light")
    st.radio("Map detail", ["light", "rich", "off"], key="visuals", format_func={"light": "Light: flat 2D (recommended)", "rich": "Rich: 3D (needs a good graphics card)", "off": "No maps: text and tables only"}.get,
             help="If the screen goes blank, freezes or the maps are slow, pick Light, or No maps. Everything except the map pictures works the same.")
nav.run()
