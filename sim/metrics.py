"""Access metrics shared by the simulator and the advisor."""
from __future__ import annotations

import numpy as np

from sim.router import Costs, Router


def compute_access(city, router: Router, costs: Costs, any_nodes, t1_nodes, fire_nodes=None, limit: float = float("inf")):
    """Seconds from every node to the nearest open hospital (any / major) and to the nearest fire station."""
    d_any, lab = router.reverse_multi(costs, {n: 0.0 for n in any_nodes}, limit) if any_nodes else ([np.inf] * city.g.n_nodes, [-1] * city.g.n_nodes)
    d_t1, _ = router.reverse_multi(costs, {n: 0.0 for n in t1_nodes}, limit) if t1_nodes else ([np.inf] * city.g.n_nodes, None)
    d_fire = None
    if fire_nodes is not None:
        d_fire, _ = router.reverse_multi(costs, {n: 0.0 for n in fire_nodes}, limit) if fire_nodes else ([np.inf] * city.g.n_nodes, None)
    return np.asarray(d_any), np.asarray(d_t1), (np.asarray(d_fire) if d_fire is not None else None), np.asarray(lab)


def summarize(city, A: dict, d_any: np.ndarray, d_t1: np.ndarray, d_fire: np.ndarray | None = None) -> dict:
    pop, tot = city.pop, max(city.total_pop, 1.0)
    s30, s60, cap = A["impact"]["slow_minutes"] * 60, A["impact"]["severe_minutes"] * 60, A["impact"]["cap_minutes"] * 60
    out = {
        "cut_any": float(pop[~np.isfinite(d_any)].sum()),
        "gt30_any": float(pop[d_any > s30].sum()),
        "gt60_any": float(pop[d_any > s60].sum()),
        "mean_any_min": float((pop * np.minimum(d_any, cap)).sum() / tot / 60.0),
        "cut_t1": float(pop[~np.isfinite(d_t1)].sum()),
        "gt30_t1": float(pop[d_t1 > s30].sum()),
        "gt60_t1": float(pop[d_t1 > s60].sum()),
        "mean_t1_min": float((pop * np.minimum(d_t1, cap)).sum() / tot / 60.0),
    }
    if d_fire is not None:
        sla = A["advisor"]["sla_min"]["fire"] * 60
        out["gt_fire_sla"] = float(pop[d_fire > sla].sum())
        out["mean_fire_min"] = float((pop * np.minimum(d_fire, cap)).sum() / tot / 60.0)
    return out


def catchments(city, d_any: np.ndarray, lab: np.ndarray, within_s: float) -> dict[int, float]:
    """People whose nearest open hospital (within ``within_s``) is each hospital node."""
    ok = (d_any <= within_s) & (lab >= 0)
    w = np.bincount(lab[ok], weights=city.pop[ok], minlength=city.g.n_nodes)
    return {int(n): float(w[n]) for n in np.flatnonzero(w)}
