"""Behavioural tests for the hazard engine, on the toy city."""
import numpy as np

from sim.hazards import (Accident, CustomEvent, Earthquake, Fire, FloodZone, GasLeak, HazardContext, Storm, StructureFailure, WorldView)


class Engines(WorldView):
    def __init__(self, n=0): self.n = n
    def engines_on_scene(self, tag): return self.n


def ctx(toy, A, seed=1, wind=90.0, kmh=12.0):
    return HazardContext(toy, A, seed, wind_toward_deg=wind, wind_kmh=kmh)


def run(h, toy, A, t, world=None, **kw):
    h.reset(ctx(toy, A, **kw)); h.step(t, world or Engines()); return h


def closed(h): 
    f = h.factor(); return np.zeros(0, bool) if f is None else (f == 0)


def test_earthquake_bigger_magnitude_closes_more_roads(toy, A):
    lat, lon = toy.centre
    n = [int(closed(run(Earthquake(lat, lon, m, secondary_fires=False), toy, A, 1)).sum()) for m in (5.0, 6.0, 7.0)]
    assert n[0] <= n[1] <= n[2] and n[2] > n[0]


def test_earthquake_damage_decays_with_distance(toy, A):
    lat, lon = toy.centre
    h = run(Earthquake(lat, lon, 6.8, secondary_fires=False), toy, A, 1)
    near, far = h.r_edge < 700, h.r_edge > 1800
    assert closed(h)[near].mean() > closed(h)[far].mean()


def test_earthquake_deterministic_and_seed_sensitive(toy, A):
    lat, lon = toy.centre
    a = closed(run(Earthquake(lat, lon, 6.5), toy, A, 1, seed=3)); b = closed(run(Earthquake(lat, lon, 6.5), toy, A, 1, seed=3))
    c = closed(run(Earthquake(lat, lon, 6.5), toy, A, 1, seed=4))
    assert (a == b).all() and (a != c).any()


def test_earthquake_nothing_before_start_and_aftershock_adds_damage(toy, A):
    lat, lon = toy.centre
    h = Earthquake(lat, lon, 6.5, start_min=60, secondary_fires=False)
    h.reset(ctx(toy, A)); h.step(30, Engines())
    assert h.factor() is None
    h.step(70, Engines()); n1 = closed(h).sum()
    h.step(70 + 600, Engines()); n2 = closed(h).sum()
    assert n2 >= n1


def test_earthquake_spawns_fires_once_and_debris_is_repairable(toy, A):
    lat, lon = toy.centre
    h = run(Earthquake(lat, lon, 6.5), toy, A, 1)
    kids = h.spawn()
    assert len(kids) >= 1 and all(isinstance(k, Fire) for k in kids) and h.spawn() == []
    assert h.repairable().sum() == closed(h).sum()


def test_fire_grows_then_dies_and_blocks_core(toy, A):
    lat, lon = toy.centre
    f = Fire(lat, lon, severity=4)
    f.reset(ctx(toy, A))
    radii = []
    for t in (5, 60, 200, 400, 700):
        f.step(t, Engines()); radii.append(f.core)
    assert radii[0] < radii[1] <= radii[2] and radii[-1] < radii[2]          # grows, then burns out
    f.step(120, Engines()); assert closed(f).sum() > 0 and f.footprint()


def test_fire_containment_stops_growth(toy, A):
    lat, lon = toy.centre
    free, held = Fire(lat, lon, severity=4), Fire(lat, lon, severity=4)
    free.reset(ctx(toy, A)); held.reset(ctx(toy, A))
    for t in range(0, 300, 10):
        free.step(t, Engines(0)); held.step(t, Engines(2 if t >= 30 else 0))
    assert held.contained_at is not None and free.contained_at is None
    free.step(150, Engines(0)); held.step(150, Engines(2))
    assert held.core < free.core


def test_fire_stretches_downwind(toy, A):
    lat, lon = toy.centre
    f = run(Fire(lat, lon, severity=5), toy, A, 200, wind=90.0, kmh=30.0)        # blowing east
    ex, ey = toy.mid_xy[:, 0] - f.cxy[0], toy.mid_xy[:, 1] - f.cxy[1]
    c = closed(f)
    assert c[(ex > 0) & (abs(ey) < 100)].sum() >= c[(ex < 0) & (abs(ey) < 100)].sum()
    assert f.elong > 1.0


def test_fire_emits_one_incident(toy, A):
    lat, lon = toy.centre
    f = run(Fire(lat, lon, severity=3, start_min=10), toy, A, 0)
    assert len(f.incidents(0, 5)) == 0 and len(f.incidents(5, 15)) == 1 and len(f.incidents(5, 15)) == 0


def test_gas_leak_is_asymmetric_and_ends(toy, A):
    lat, lon = toy.centre
    g = run(GasLeak(lat, lon, severity=4), toy, A, 100, wind=90.0)
    ex, ey = toy.mid_xy[:, 0] - g.cxy[0], toy.mid_xy[:, 1] - g.cxy[1]
    c = closed(g)
    assert c[(ex > 200) & (abs(ey) < 150)].sum() > c[(ex < -200) & (abs(ey) < 150)].sum()
    g.step(1200, Engines()); assert closed(g).sum() == 0                       # leak stopped, plume dispersed


def test_flood_rises_recedes_and_bridges_stay_open_below_threshold(toy, A):
    fz = FloodZone("Toy River", severity=3); fz.reset(ctx(toy, A))
    ext = []
    for t in (30, 120, 200, 400, 900):
        fz.step(t, Engines()); ext.append(fz.ext)
    assert ext[0] < ext[1] and ext[3] < ext[2] and ext[-1] < ext[3]
    fz.step(150, Engines()); f = fz.factor()
    assert (f == 0).sum() > 0
    strong = FloodZone("Toy River", severity=5); strong.reset(ctx(toy, A)); strong.step(150, Engines())
    assert (strong.factor() == 0).sum() > (f == 0).sum()


def test_flood_footprint_and_exposure(toy, A):
    fz = FloodZone("Toy River", severity=4); fz.reset(ctx(toy, A)); fz.step(150, Engines())
    assert fz.footprint() and fz.exposure().max() == 1.0


def test_structure_failure_blocks_exactly_the_structure_and_reopens(toy, A):
    s = toy.structures[0]
    sf = StructureFailure(s, mode="closure", start_min=30, duration_min=120); sf.reset(ctx(toy, A))
    sf.step(10, Engines()); assert sf.factor() is None
    sf.step(60, Engines()); assert set(np.flatnonzero(closed(sf))) == set(sf.edges.tolist())
    sf.step(200, Engines()); assert sf.factor() is None
    part = StructureFailure(s, mode="partial"); part.reset(ctx(toy, A)); part.step(5, Engines())
    assert (part.factor()[part.edges] > 0).all() and (part.factor()[part.edges] < 1).all()


def test_structure_collapse_creates_casualties(toy, A):
    sf = StructureFailure(toy.structures[0], mode="collapse"); sf.reset(ctx(toy, A))
    assert len(sf.incidents(0, 5)) >= 6 and sf.incidents(0, 5) == []


def test_accident_blocks_and_queues_upstream(toy, A):
    lat, lon = toy.centre
    a = run(Accident(lat, lon, severity=4, duration_min=90), toy, A, 10)
    f = a.factor()
    assert (f == 0).sum() >= 2 and ((f > 0) & (f < 1)).sum() > 0
    a.step(200, Engines()); assert a.factor() is None


def test_storm_slows_everything_and_fells_trees_progressively(toy, A):
    s = Storm(severity=5, start_min=0, duration_min=300); s.reset(ctx(toy, A))
    s.step(150, Engines()); f = s.factor(); assert f[f > 0].max() < 1.0
    early = closed(s).sum(); s.step(280, Engines()); assert closed(s).sum() >= early and s.repairable().sum() > 0


def test_custom_event_closed_vs_slowed_and_casualties(toy, A):
    lat, lon = toy.centre
    cl = run(CustomEvent(lat, lon, radius_m=400, closed=True, casualties=8), toy, A, 5)
    sl = run(CustomEvent(lat, lon, radius_m=400, closed=False, slow_factor=0.3), toy, A, 5)
    assert closed(cl).sum() > 0 and closed(sl).sum() == 0 and (sl.factor() < 1).sum() > 0
    assert len(cl.incidents(0, 10)) == 8
    grow = CustomEvent(lat, lon, radius_m=600, growth_m_per_min=10, duration_min=200); grow.reset(ctx(toy, A))
    grow.step(5, Engines()); r1 = grow.core; grow.step(50, Engines()); assert grow.core > r1
