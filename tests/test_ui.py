"""Whole-app behaviour through Streamlit's headless runner, on the real city: navigation must not lose the user's settings."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    os.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    return at


def test_every_page_renders_without_errors(app):
    for pg in ("ui/pages/route.py", "ui/pages/lab.py", "ui/pages/build.py", "ui/pages/idea.py", "ui/pages/existing.py", "ui/pages/sources.py"):
        app.switch_page(pg).run()
        assert not app.exception, (pg, [e.value for e in app.exception])


def test_priorities_survive_visiting_pages_without_the_sliders(app):
    """Regression: Streamlit drops widget state for widgets not drawn on the previous page; the sliders used to come back at zero."""
    app.switch_page("ui/pages/build.py").run()
    app.radio(key="prio_preset").set_value("Everyday first").run()
    w1 = dict(app.session_state["kind_weights"])
    assert sum(w1.values()) == pytest.approx(1.0) and w1["baseline"] > 0.5
    app.switch_page("ui/pages/route.py").run()                 # a page with no priority sliders
    app.switch_page("ui/pages/existing.py").run()
    w2 = dict(app.session_state["kind_weights"])
    assert w2 == pytest.approx(w1) and sum(w2.values()) == pytest.approx(1.0)


def test_route_page_recommends_and_lifelines_run(app):
    app.switch_page("ui/pages/route.py").run()
    [b for b in app.button if b.label == "Recommend a route"][0].click().run()
    assert not app.exception and app.session_state["rt_adv"]["adv"].recommended is not None
    [b for b in app.button if b.label == "Show the roads people depend on"][0].click().run()
    assert not app.exception and app.session_state["ll"][1]["top_load"]


def test_missing_package_gives_instructions_not_a_traceback(monkeypatch):
    """What a user sees when they forgot `pip install -r requirements.txt` (this is exactly how it failed on a fresh Windows machine)."""
    import sys

    monkeypatch.setitem(sys.modules, "yaml", None)                       # the package is 'not installed'
    os.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.error and "pyyaml" in at.error[0].value
    assert any("pip install -r requirements.txt" in c.value for c in at.code)


def test_running_with_plain_python_explains_how_to_start_it():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "app.py"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "streamlit run app.py" in out and "Traceback" not in out


def test_maps_can_be_switched_off_and_light_is_the_default(app):
    """A laptop that cannot draw maps must still get a working app: 'off' replaces every map with a note and nothing else changes."""
    app.switch_page("ui/pages/home.py").run()
    assert app.session_state["visuals"] == "light"
    assert len(app.get("deck_gl_json_chart")) == 1
    app.radio(key="visuals").set_value("off").run()
    assert not app.exception and len(app.get("deck_gl_json_chart")) == 0
    assert any("Maps are switched off" in i.value for i in app.info)
    app.switch_page("ui/pages/route.py").run()                      # a page with a real result: the recommendation still appears without any map
    [b for b in app.button if b.label == "Recommend a route"][0].click().run()
    assert not app.exception and app.session_state["rt_adv"]["adv"].recommended is not None and len(app.get("deck_gl_json_chart")) == 0
    app.radio(key="visuals").set_value("light").run()
