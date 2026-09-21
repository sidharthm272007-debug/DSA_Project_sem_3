"""How risky is each stretch of road in a scenario? Built from what the scenario actually did, with the weights in assumptions.yaml."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from sim.city import CityData
from sim.state import WorldState
from store.csr_graph import FLAG_BRIDGE, FLAG_FLYOVER, FLAG_TUNNEL


def edge_risk(city: CityData, A: dict, state: WorldState, avoid_edges: np.ndarray | None = None) -> np.ndarray:
    """Per-edge risk in 0..1 (closed roads are 1). ``avoid_edges`` are roads the traveller knows to be unsafe."""
    R = A["routing"]["risk"]
    g = city.g
    F = state.F.astype(np.float64)
    expo = np.maximum(state.expo[g.edge_tails], state.expo[g.indices])
    degraded = np.clip(1.0 - F / R["degraded_ref"], 0.0, 1.0)
    closed = F == 0
    near = np.zeros(g.n_edges)
    if closed.any():
        mid = 0.5 * (city.xy[g.edge_tails] + city.xy[g.indices])
        d, _ = cKDTree(mid[closed]).query(mid, distance_upper_bound=R["near_closure_m"])
        near = np.isfinite(d).astype(np.float64)
    deck = (g.flags & (FLAG_BRIDGE | FLAG_FLYOVER | FLAG_TUNNEL)) != 0
    fragile = (deck & (expo >= R["fragile_reach"])).astype(np.float64)
    ok = 1.0 - (1.0 - R["exposure_weight"] * expo) * (1.0 - R["degraded_weight"] * degraded) * (1.0 - R["near_closure_weight"] * near) * (1.0 - R["fragile_weight"] * fragile)
    ok[closed] = 1.0
    if avoid_edges is not None and len(avoid_edges):
        ok[np.asarray(avoid_edges, dtype=np.int64)] = 1.0
    return ok.astype(np.float32)
