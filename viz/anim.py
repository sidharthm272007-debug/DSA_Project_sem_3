"""Vehicle trips for the deck.gl TripsLayer. Built lazily, only for the trips near the playback time, then cached."""
from __future__ import annotations

import numpy as np

from viz.theme import UNIT_COLOR, hex_rgb


class TripBook:
    def __init__(self, city, legs):
        self.city = city
        self.legs = legs
        self.span = []                                 # (t_first_min, t_last_min) per leg
        for L in legs:
            self.span.append((L.edges[0][1] / 60.0, L.edges[-1][2] / 60.0) if L.edges else (1e18, -1e18))
        self.span_arr = np.array(self.span) if self.span else np.zeros((0, 2))
        self._cache: dict[int, dict] = {}

    def _build(self, i: int) -> dict | None:
        L = self.legs[i]
        path: list[list[float]] = []
        ts: list[float] = []
        for e, t0, t1 in L.edges:
            pts = self.city.edge_polyline(int(e))
            if len(pts) < 2:
                continue
            seg = np.hypot(*np.diff(pts, axis=0).T) + 1e-12
            frac = np.concatenate([[0.0], np.cumsum(seg) / seg.sum()])
            times = (t0 + frac * (t1 - t0)) / 60.0
            start = 1 if path else 0
            path.extend(np.round(pts[start:].astype(np.float64), 5).tolist())
            ts.extend(np.round(times[start:], 3).tolist())
        if len(path) < 2:
            return None
        col = hex_rgb(UNIT_COLOR.get(L.kind, "#FFFFFF"), 255)
        return {"path": path, "ts": ts, "color": col, "kind": L.kind, "tip": f"{L.kind} ({L.leg.replace('_', ' ')})"}

    def window(self, t: float, trail: float = 7.0, limit: int = 900) -> list[dict]:
        if not len(self.span_arr):
            return []
        idx = np.flatnonzero((self.span_arr[:, 0] <= t + 0.5) & (self.span_arr[:, 1] >= t - trail))
        if len(idx) > limit:
            idx = idx[np.linspace(0, len(idx) - 1, limit).astype(int)]
        out = []
        for i in idx:
            i = int(i)
            if i not in self._cache:
                self._cache[i] = self._build(i)
            if self._cache[i] is not None:
                out.append(self._cache[i])
        return out
