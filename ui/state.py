"""Shared, expensive objects. Built once per server process, thread-safely, and warmed up in the background at start-up
so the first visit to a planning page does not wait for them."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
import streamlit as st

from advisor.service import Advisor
from assumptions import load_assumptions
from sim.city import load_city
from sim.router import Router
from ui.scenarios import place_lookup
from viz import layers as L

_LOCK = threading.RLock()
_WARM_STARTED = False


@lru_cache(maxsize=1)
def _core():
    A = load_assumptions()
    city = load_city()
    return A, city, Router(city.g)


def core():
    with _LOCK:
        return _core()


def visuals() -> str:
    """How much map detail to draw: "light" (flat 2-D, default), "rich" (3-D) or "off". The sidebar sets it; RESQGRAPH_VISUALS sets the starting value."""
    v = st.session_state.get("visuals") or os.environ.get("RESQGRAPH_VISUALS", "light")
    return v if v in ("light", "rich", "off") else "light"


L.mode_provider = visuals


@lru_cache(maxsize=2)
def _base_paths(detail: str):
    _, city, _ = core()
    return L.base_network(city) if detail == "rich" else L.base_network(city, ("trunk", "primary"))


def base_paths():
    with _LOCK:
        return _base_paths("rich" if visuals() == "rich" else "light")


@lru_cache(maxsize=1)
def _places():
    _, city, _ = core()
    return place_lookup(city)


def places():
    with _LOCK:
        return _places()


@lru_cache(maxsize=1)
def _shared_advisor():
    A, city, router = core()
    return Advisor(city, A, router=router)


def advisor_ready() -> bool:
    return _shared_advisor.cache_info().currsize > 0


def advisor() -> Advisor:
    """This session's advisor: a cheap fork of the shared one with the session's own priority weights.
    Blocks (thread-safely) until the shared advisor exists."""
    ss = st.session_state
    if "_adv" not in ss:
        with _LOCK:
            ss["_adv"] = _shared_advisor().fork()
    adv = ss["_adv"]
    adv.reweight(ss.get("kind_weights"))
    return adv


def load_advisor_with_message(what: str = "Preparing the scenario set every idea is judged against") -> Advisor:
    """Call after the page heading is on screen: shows a visible message instead of a blank page while the advisor is built."""
    try:
        if advisor_ready():
            return advisor()
        with st.spinner(f"{what} (about 15 s, once per server start)…"):
            return advisor()
    except MemoryError:
        st.error("This computer ran out of free memory while preparing the planning tools (they need about 0.6 GB free).")
        st.markdown("Close other programs and browser tabs and reload this page. **Best route now** and **Disaster lab** need much less memory and still work.")
        st.stop()


def warm_up() -> None:
    """Start building the road network, the base map and the advisor in the background (once per process)."""
    global _WARM_STARTED
    if _WARM_STARTED or os.environ.get("RESQGRAPH_WARMUP") != "1":       # off by default: it competes for memory and processor while the first page loads
        return
    _WARM_STARTED = True

    def work():
        try:
            core()
            _base_paths("light")
            places()
            with _LOCK:
                _shared_advisor()
        except Exception:                      # a failure here must not take the app down; the page will surface it when it needs the object
            pass

    threading.Thread(target=work, name="resqgraph-warmup", daemon=True).start()


@lru_cache(maxsize=1)
def _road_index():
    _, city, _ = core()
    idx: dict[str, list[int]] = {}
    for wid, name in city.way_names.items():
        if name:
            idx.setdefault(name, []).append(int(wid))
    return idx


def road_index():
    with _LOCK:
        return _road_index()


def edges_of_roads(names: list[str]):
    _, city, _ = core()
    ids = [w for n in names for w in road_index().get(n, [])]
    return np.flatnonzero(np.isin(city.g.way_id, ids)) if ids else np.zeros(0, dtype=np.int64)
