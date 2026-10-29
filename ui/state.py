"""Shared, expensive objects. Built once per server process, thread-safely, and warmed up in the background at start-up."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
import streamlit as st

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


def warm_up() -> None:
    """Start building the road network and the base map in the background (once per process)."""
    global _WARM_STARTED
    if _WARM_STARTED or os.environ.get("RESQGRAPH_WARMUP") != "1":       # off by default: it competes for memory and processor while the first page loads
        return
    _WARM_STARTED = True

    def work():
        try:
            core()
            _base_paths("light")
            places()
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
