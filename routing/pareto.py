"""The Pareto-optimal front of a set of (time, risk) route options.

A route is Pareto-optimal (non-dominated) if no other option is at least as good on BOTH objectives and strictly
better on at least one. This is the complete, correct answer to "which of these routes are genuinely worth
considering" for two objectives that trade off against each other (faster vs. safer) -- anything not on the front
is strictly worse than some option that IS on it, on both counts at once, so it can never be the right choice for
any traveller, however they weigh time against risk.

Why this matters here specifically: picking "the best route" by minimising a single number like
``time + v * risk`` (a linear scalarisation) can only ever land on a point that is optimal for SOME straight-line
trade-off. If the true frontier has a concave section -- a route that is meaningfully safer for only a little
extra time, sitting strictly inside the hull of cheaper/riskier and slower/safer alternatives -- no value of `v`
will ever select it, even though it dominates nothing and is dominated by nothing. Computing the actual frontier
first, then choosing a point ON it, closes that gap: routing/recommend.py still uses the traveller's `v` to pick
one point, but it chooses from the complete set of genuinely-worth-considering routes, not a subset that happens
to be reachable by a linear weighting.

O(n log n): sort by the first objective, then sweep keeping only strictly-improving values of the second -- the
standard algorithm for the 2-D case (the general, k-dimensional version is O(n log^{d-2} n) via Kung et al. 1975,
not needed here since there are only two objectives).
"""
from __future__ import annotations

from typing import Protocol, TypeVar


class _Scored(Protocol):
    time_s: float
    risk_km: float


T = TypeVar("T", bound=_Scored)


def pareto_front(options: list[T]) -> list[T]:
    """The non-dominated subset of ``options`` (minimising both ``time_s`` and ``risk_km``), sorted by increasing time."""
    ordered = sorted(options, key=lambda o: (o.time_s, o.risk_km))
    front: list[T] = []
    best_risk = float("inf")
    for o in ordered:
        if o.risk_km < best_risk - 1e-9:                 # strictly safer than every option already kept (all of them faster-or-tied)
            front.append(o)
            best_risk = o.risk_km
    return front


def is_dominated(o: T, others: list[T]) -> bool:
    """True if some option in ``others`` is at least as good on both objectives and strictly better on one -- the
    direct (O(n), brute-force) definition, used in tests to check ``pareto_front`` against its own specification."""
    for q in others:
        if q is o:
            continue
        if q.time_s <= o.time_s and q.risk_km <= o.risk_km and (q.time_s < o.time_s or q.risk_km < o.risk_km):
            return True
    return False
