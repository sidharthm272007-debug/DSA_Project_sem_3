"""The plain-language findings must be built from computed numbers and must say when the data is thin."""
import copy

from sim.engine import Context, Scenario, Simulation
from sim.hazards import Earthquake, Fire, Storm


def run(toy, A, hazards, hours=4):
    return Simulation(toy, Scenario("t", hazards, duration_h=hours, seed=3, context=Context()), A).run()


def test_findings_flag_thin_fire_data_only_when_fire_calls_exist(toy, A):
    g = toy.g
    mid = 10 * 21 + 10
    with_fire = run(toy, A, [Fire(float(g.lat[mid]), float(g.lon[mid]), severity=3)])
    assert any("Data gap" in f and "fire station" in f for f in with_fire.summary["findings"])
    A2 = copy.deepcopy(A)
    A2["standards"]["min_fire_stations"] = 1
    ok = run(toy, A2, [Fire(float(g.lat[mid]), float(g.lon[mid]), severity=3)])
    assert not any("Data gap" in f for f in ok.summary["findings"])


def test_no_zero_repair_boilerplate_and_correct_grammar(toy, A):
    import re
    res = run(toy, A, [Earthquake(float(toy.g.lat[10 * 21 + 10]), float(toy.g.lon[10 * 21 + 10]), magnitude=6.5)])
    text = " ".join(res.summary["findings"])
    assert "0 debris-clearing jobs" not in text and "0 clearing jobs" not in text
    assert not re.search(r"(?<!\d)1 were never", text)


def test_response_time_findings_match_an_independent_recount(toy, A):
    import numpy as np
    res = run(toy, A, [Storm(severity=4)], hours=6)
    text = " ".join(res.summary["findings"])
    checked = 0
    for kind, label in (("medical", "medical calls"), ("fire", "fire calls"), ("rescue", "rescue calls")):
        xs = [i for i in res.incidents if i.kind == kind]
        arrive = [i.t_arrive - i.t for i in xs if i.t_arrive is not None]
        if len(xs) >= 3 and arrive:
            want = f"{len(xs)} {label}: median first-unit arrival {np.median(arrive):.0f} min, 90th percentile {np.percentile(arrive, 90):.0f} min"
            assert want in text, want
            checked += 1
    assert checked >= 1


def test_fmt_people_is_sign_symmetric_and_never_prints_negative_zero():
    from sim.report import fmt_people
    assert fmt_people(0.0) == "0" and fmt_people(-0.0) == "0" and fmt_people(-1e-9) == "0" and fmt_people(-0.3) == "0"
    assert fmt_people(150000) == "1.5 lakh" and fmt_people(-150000) == "-1.5 lakh"
    assert fmt_people(12345) == "12.3k" and fmt_people(-12345) == "-12.3k"
    assert fmt_people(42) == "42" and fmt_people(-42) == "-42"
