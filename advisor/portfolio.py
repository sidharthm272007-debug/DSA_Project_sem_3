"""The scenario portfolio: a normal day plus the *worst moment* of several different disasters.

Infrastructure decisions have to hold up across many futures, not one. Every idea the advisor scores is judged against
the same portfolio, so ideas are comparable and no single hand-picked disaster drives a recommendation.

Each snapshot freezes one hazard at its worst state (no repairs yet, no dispatch dynamics): the road network as the
hazard leaves it and the hospitals it knocks out. Locations are drawn deterministically from real geography
(quakes over the built-up area, fires where people live, floods on the mapped waterbodies, failures of the named
flyovers/bridges), so the same seed always gives the same portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.city import CityData
from sim.hazards import Earthquake, Fire, FloodZone, Hazard, HazardContext, Storm, StructureFailure, WorldView

KINDS = ("baseline", "earthquake", "fire", "flood", "structure", "storm")
KIND_LABEL = {"baseline": "Normal day", "earthquake": "Earthquake", "fire": "Fire", "flood": "Flood", "structure": "Bridge / flyover failure", "storm": "Storm"}


@dataclass
class Snapshot:
    name: str
    kind: str
    F: np.ndarray | None                # per-edge speed factor at the worst moment (None = normal)
    closed_fac: frozenset[int]          # facility ids knocked out
    bg: float                           # background speed factor (traffic + weather)
    weight: float = 0.0
    loc: tuple[float, float] | None = None
    detail: str = ""
    closed_edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


class _Idle(WorldView):
    pass


def _peak_state(h: Hazard, city: CityData, A: dict, seed: int, probes: list[float]) -> tuple[np.ndarray, frozenset[int]]:
    """Step the hazard through ``probes`` (minutes) and return the state with the most closed road length."""
    h.reset(HazardContext(city, A, seed))
    best, best_len, best_fac = None, -1.0, frozenset()
    world = _Idle()
    for t in probes:
        h.step(float(t), world)
        f = h.factor()
        F = np.ones(city.g.n_edges, dtype=np.float32) if f is None else np.asarray(f, dtype=np.float32).copy()
        closed_len = float(city.g.length_m[F == 0].sum())
        fac = frozenset(fid for fid, cap in h.outages().items() if cap <= 0)
        score = closed_len + 1e5 * len(fac)
        if score > best_len:
            best, best_len, best_fac = F, score, fac
    return best, best_fac


def build_portfolio(city: CityData, A: dict, weights: dict | None = None, seed: int | None = None, counts: dict | None = None) -> list[Snapshot]:
    P = A["advisor"]["portfolio"]
    seed = P["seed"] if seed is None else seed
    counts = counts or P["counts"]
    rng = np.random.default_rng(seed)
    giant = np.flatnonzero(city.giant)
    g = city.g
    bg_normal = A["context"]["time_of_day_speed"]["off_peak"] * A["context"]["weather_speed"]["clear"]
    snaps: list[Snapshot] = [Snapshot("Normal day", "baseline", None, frozenset(), bg_normal)]

    def pick(n: int, weighted: bool) -> np.ndarray:
        p = city.pop[giant] / city.pop[giant].sum() if weighted else None
        return rng.choice(giant, size=n, replace=False, p=p)

    for i, node in enumerate(pick(counts.get("earthquake", 0), False)):
        mag = float(P["quake_magnitudes"][i % len(P["quake_magnitudes"])])
        lat, lon = float(g.lat[node]), float(g.lon[node])
        h = Earthquake(lat, lon, magnitude=mag, secondary_fires=False, name=f"M{mag:.1f} quake near {city.describe_node(int(node))}")
        F, fac = _peak_state(h, city, A, seed + i, [1.0])
        snaps.append(Snapshot(h.name, "earthquake", F, fac, bg_normal, loc=(lat, lon), detail=f"magnitude {mag:.1f}"))

    for i, node in enumerate(pick(counts.get("fire", 0), True)):
        lat, lon = float(g.lat[node]), float(g.lon[node])
        h = Fire(lat, lon, severity=int(P["fire_severity"]), start_min=0.0, name=f"Fire near {city.describe_node(int(node))}")
        F, fac = _peak_state(h, city, A, seed + 100 + i, [30.0, 60.0, 120.0, 180.0])
        snaps.append(Snapshot(h.name, "fire", F, fac, bg_normal, loc=(lat, lon), detail=f"severity {P['fire_severity']}, uncontained after up to 3 h"))

    waters = sorted(city.water, key=lambda k: -float(np.hypot(*np.diff(city._water_xy[k], axis=0).T).sum()))
    for i in range(min(counts.get("flood", 0), len(waters))):
        sev = int(P["flood_severity"][i % len(P["flood_severity"])])
        h = FloodZone(waters[i], severity=sev, start_min=0.0, name=f"Flood: {waters[i]}")
        F, fac = _peak_state(h, city, A, seed + 200 + i, [60.0, 120.0, 240.0, 360.0, 480.0])
        snaps.append(Snapshot(h.name, "flood", F, fac, bg_normal * A["context"]["weather_speed"]["heavy_rain"], loc=(h.lat, h.lon), detail=f"severity {sev}"))

    for i, s in enumerate(city.structures[: counts.get("structure", 0)]):
        h = StructureFailure(structure=s, mode="collapse", name=f"Failure of {s['name']}")
        F, fac = _peak_state(h, city, A, seed + 300 + i, [1.0])
        snaps.append(Snapshot(h.name, "structure", F, fac, bg_normal, loc=(s["lat"], s["lon"]), detail=f"{s['length_m']:.0f} m closed"))

    for i in range(counts.get("storm", 0)):
        h = Storm(severity=int(P["storm_severity"]), start_min=0.0, name="Severe storm")
        F, fac = _peak_state(h, city, A, seed + 400 + i, [60.0, 120.0, 240.0])
        snaps.append(Snapshot(h.name, "storm", F, fac, bg_normal * A["context"]["weather_speed"]["storm"], detail="city-wide fallen trees"))

    for s in snaps:
        s.closed_edges = np.flatnonzero(s.F == 0) if s.F is not None else np.zeros(0, dtype=np.int64)
    assign_weights(snaps, weights, A)
    return snaps


def assign_weights(snaps: list[Snapshot], weights: dict | None, A: dict) -> None:
    """weight of a kind is split equally among its snapshots; all weights sum to 1. ``weights`` maps kind -> relative importance."""
    P = A["advisor"]["portfolio"]
    w = {"baseline": P["baseline_weight"], **{("flood" if k == "flood" else k): v for k, v in P["kind_weight"].items()}}
    if weights:
        w.update(weights)
    present = {k: sum(1 for s in snaps if s.kind == k) for k in KINDS}
    total = sum(w.get(k, 0.0) for k in KINDS if present[k]) or 1.0
    for s in snaps:
        s.weight = w.get(s.kind, 0.0) / total / present[s.kind]
