"""Invariants of the disaster simulator, checked on the small toy city."""
import pytest

from sim.engine import Context, Scenario, Simulation
from sim.hazards import Earthquake, Fire, Hazard, IncidentSpec, StructureFailure
from sim.response import Resources


def quake(toy, mag=6.5, **kw):
    return Earthquake(toy.g.lat[220], toy.g.lon[220], magnitude=mag, **kw)


def run(toy, A, hazards, hours=3, seed=3, **kw):
    res = kw.pop("resources", Resources(ambulances=8, crews=2))
    sc = Scenario("t", hazards, duration_h=hours, seed=seed, resources=res, **kw)
    return Simulation(toy, sc, A).run()


def test_same_seed_same_result(toy, A):
    a, b = run(toy, A, [quake(toy)]), run(toy, A, [quake(toy)])
    assert a.metrics == b.metrics
    assert [(i.kind, i.node, i.severity, i.state, i.t_arrive) for i in a.incidents] == [(i.kind, i.node, i.severity, i.state, i.t_arrive) for i in b.incidents]
    c = run(toy, A, [quake(toy)], seed=4)
    assert [i.node for i in c.incidents] != [i.node for i in a.incidents]          # a different seed really is different


def test_no_hazard_means_nothing_changes(toy, A):
    r = run(toy, A, [], hours=2)
    base = r.baseline["summary"]
    for m in r.metrics:
        assert m["blocked_km"] == 0 and m["hospitals_closed"] == 0
        assert m["cut_any"] == pytest.approx(base["cut_any"]) and m["gt30_t1"] == pytest.approx(base["gt30_t1"])
    assert all(i.tag == "baseline" for i in r.incidents)
    assert r.summary["never_hit"]


def test_earthquake_hurts_access_and_more_so_when_stronger(toy, A):
    weak, strong = run(toy, A, [quake(toy, 5.0)]), run(toy, A, [quake(toy, 7.0)])
    base = strong.baseline["summary"]
    assert strong.peak["blocked_km"] > weak.peak["blocked_km"] >= 0
    assert strong.peak["gt30_t1"] >= weak.peak["gt30_t1"] >= base["gt30_t1"]
    assert strong.peak["hospitals_closed"] >= weak.peak["hospitals_closed"]
    assert strong.peak["gt30_t1"] > base["gt30_t1"] or strong.peak["cut_any"] > 0


def test_vehicles_never_enter_a_closed_road(toy, A):
    r = run(toy, A, [quake(toy, 6.8)], hours=3)
    step = r.times[1] - r.times[0]
    repaired_at = {}
    for job in r.repair_log:
        for e in job["edges"]:
            repaired_at[e] = min(repaired_at.get(e, 1e9), job["t"])
    frames = {f["t"]: set(f["blocked"].tolist()) | {int(toy.g.rev[e]) for e in f["blocked"] if toy.g.rev[e] >= 0} for f in r.frames}
    checked = 0
    for leg in r.legs:
        for e, t_in, t_out in leg.edges:
            minute = t_in / 60.0
            frame_t = int(minute // step * step)
            closed = e in frames[frame_t] and not (e in repaired_at and repaired_at[e] <= minute + 1e-6)
            assert not closed, f"unit {leg.unit} entered closed edge {e} at minute {minute:.1f}"
            checked += 1
    assert checked > 50                                         # the check had something to check


def test_no_vehicle_beats_free_flow_speed(toy, A):
    r = run(toy, A, [quake(toy)])
    for leg in r.legs:
        for e, t_in, t_out in leg.edges:
            assert t_out - t_in >= toy.g.tt_s[e] - 1e-6
        assert all(leg.edges[i][2] <= leg.edges[i + 1][1] + 1e-6 for i in range(len(leg.edges) - 1))   # time flows forward
        assert all(toy.g.indices[leg.edges[i][0]] == toy.g.edge_tails[leg.edges[i + 1][0]] for i in range(len(leg.edges) - 1))  # contiguous


def test_incident_timeline_is_consistent(toy, A):
    r = run(toy, A, [quake(toy)])
    for i in r.incidents:
        if i.t_arrive is not None:
            assert i.t_arrive >= i.t and i.t_assign is not None and i.t_assign >= i.t
        if i.t_delivered is not None:
            assert i.t_arrive is not None and i.t_delivered >= i.t_arrive
        assert i.state in ("done", "unserved", "stranded", "pending", "responding", "onscene", "transport")
    assert len(r.incidents) <= A["incidents"]["max_incidents"] + 20


class TwoCalls(Hazard):
    """Test hazard: a minor and a critical medical call at the same instant, far apart."""
    kind = "custom"

    def __init__(self, a, b):
        super().__init__("two calls", 0.0)
        self.a, self.b, self._done = a, b, False

    def incidents(self, t0, t1):
        if self._done or not (t0 <= 0 < t1 + 1e-9):
            return []
        self._done = True
        return [IncidentSpec("medical", self.a, 1, 1.0, self.name), IncidentSpec("medical", self.b, 3, 1.0, self.name)]


def test_critical_patient_gets_the_only_ambulance_first(toy, A):
    res = Resources(ambulances=1, crews=0, fire_engines=0, rescue_units=0)
    r = run(toy, A, [TwoCalls(10 * 21 + 12, 10 * 21 + 18)], hours=2, resources=res, incident_scale=0.0)
    med = sorted((i for i in r.incidents if i.tag == "two calls"), key=lambda i: i.severity)
    minor, critical = med
    assert critical.t_assign is not None
    assert minor.t_assign is None or critical.t_assign <= minor.t_assign


def test_fire_is_contained_only_if_engines_arrive(toy, A):
    def burn(engines):
        f = Fire(toy.g.lat[30 * 7], toy.g.lon[30 * 7], severity=3, start_min=5)
        r = run(toy, A, [f], hours=6, resources=Resources(ambulances=4, fire_engines=engines, crews=0), incident_scale=0.0)
        return f, r
    f_eng, r_eng = burn(3)
    f_none, r_none = burn(0)
    assert f_eng.contained_at is not None
    assert f_none.contained_at is None
    fire_eng = [i for i in r_eng.incidents if i.kind == "fire"]
    assert fire_eng and fire_eng[0].arrived >= 1
    assert max(m["blocked_km"] for m in r_none.metrics) >= max(m["blocked_km"] for m in r_eng.metrics)


def test_repair_crews_reopen_roads_only_when_they_exist(toy, A):
    with_crews = run(toy, A, [quake(toy, 6.8)], hours=6, resources=Resources(ambulances=8, crews=3))
    no_crews = run(toy, A, [quake(toy, 6.8)], hours=6, resources=Resources(ambulances=8, crews=0))
    assert no_crews.summary["repairs"] == 0 and no_crews.metrics[-1]["repaired_km"] == 0
    assert with_crews.summary["repairs"] > 0 and with_crews.metrics[-1]["repaired_km"] > 0
    assert with_crews.metrics[-1]["blocked_km"] <= no_crews.metrics[-1]["blocked_km"] + 1e-9


def test_structure_failure_closes_exactly_the_structure(toy, A):
    s = toy.structures[0]
    h = StructureFailure(structure=s, mode="collapse", start_min=0)
    r = run(toy, A, [h], hours=1, incident_scale=0.0, resources=Resources(ambulances=4, crews=0))
    assert r.peak["blocked_km"] > 0
    blocked = set(r.frames[0]["blocked"].tolist()) | {int(toy.g.rev[e]) for e in r.frames[0]["blocked"]}
    assert blocked <= set(s["edges"]) | {int(toy.g.rev[e]) for e in s["edges"]}


def test_background_traffic_and_weather_slow_the_baseline(toy, A):
    fast = run(toy, A, [], hours=1, context=Context(time_of_day="night", weather="clear"))
    slow = run(toy, A, [], hours=1, context=Context(time_of_day="peak", weather="storm"))
    assert slow.baseline["summary"]["mean_any_min"] > fast.baseline["summary"]["mean_any_min"]
