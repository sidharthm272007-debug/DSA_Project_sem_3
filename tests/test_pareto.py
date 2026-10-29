"""The Pareto frontier sweep, checked against its own brute-force (O(n^2)) specification."""
from dataclasses import dataclass

from routing.pareto import is_dominated, pareto_front


@dataclass
class Pt:
    time_s: float
    risk_km: float
    tag: str = ""


def test_matches_brute_force_definition_on_random_sets():
    import random

    rng = random.Random(3)
    for _ in range(200):
        n = rng.randint(1, 25)
        pts = [Pt(rng.uniform(0, 100), rng.uniform(0, 50)) for _ in range(n)]
        front = pareto_front(pts)
        # everything on the front must be non-dominated, by the independent brute-force check
        for p in front:
            assert not is_dominated(p, pts)
        # everything NOT on the front must be dominated by something (brute force agrees it was right to drop it)
        for p in pts:
            if p not in front:
                assert is_dominated(p, pts)


def test_no_two_points_on_the_front_dominate_each_other():
    import random

    rng = random.Random(9)
    pts = [Pt(rng.uniform(0, 100), rng.uniform(0, 50)) for _ in range(40)]
    front = pareto_front(pts)
    for a in front:
        for b in front:
            if a is not b:
                assert not (b.time_s <= a.time_s and b.risk_km <= a.risk_km and (b.time_s < a.time_s or b.risk_km < a.risk_km))


def test_a_single_best_in_everything_is_the_whole_front():
    pts = [Pt(10, 10), Pt(20, 5), Pt(30, 1)]      # a genuine trade-off: time rises as risk falls -> none dominated, all on the front
    front = pareto_front(pts)
    assert len(front) == 3


def test_a_point_worse_on_both_axes_is_dropped():
    good = Pt(10, 1)
    bad = Pt(20, 5)                                # slower and riskier than `good` -> dominated, must be dropped
    front = pareto_front([good, bad])
    assert front == [good]


def test_ties_on_time_keep_only_the_lower_risk():
    a, b = Pt(10, 3), Pt(10, 1)
    front = pareto_front([a, b])
    assert front == [b]


def test_exact_ties_on_both_axes_keep_exactly_one():
    a, b = Pt(10, 3, "a"), Pt(10, 3, "b")
    front = pareto_front([a, b])
    assert len(front) == 1


def test_concave_frontier_point_is_kept_even_though_no_linear_weight_would_ever_pick_it():
    """The whole reason this module exists instead of just minimising time + v*risk: a point that sits strictly
    inside the hull of its neighbours on a concave stretch of the trade-off curve is still genuinely non-dominated,
    and the frontier must include it even though sweeping any single weighted sum would skip straight past it."""
    cheap_risky = Pt(10, 10)
    knee = Pt(15, 9)            # above the straight line from cheap_risky to safe_slow at time=15 (that line sits at risk=8.5) --
    safe_slow = Pt(40, 1)       # genuinely non-dominated, but "hidden" behind the hull of the other two, so no straight-line
                                 # trade-off (i.e. no value of v in time + v*risk) can ever prefer it to both neighbours at once.
    front = pareto_front([cheap_risky, knee, safe_slow])
    assert knee in front and cheap_risky in front and safe_slow in front
    for v in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 3.12, 5.0, 100.0):
        scored = {"cheap_risky": cheap_risky.time_s + v * cheap_risky.risk_km, "knee": knee.time_s + v * knee.risk_km,
                  "safe_slow": safe_slow.time_s + v * safe_slow.risk_km}
        assert min(scored, key=scored.get) != "knee"


def test_empty_and_single_point():
    assert pareto_front([]) == []
    p = Pt(5, 5)
    assert pareto_front([p]) == [p]
