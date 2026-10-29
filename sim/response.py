"""Emergency units, incidents, and helpers to place them."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.city import CityData


@dataclass
class Unit:
    id: int
    kind: str                     # ambulance | fire | rescue | crew
    base: int
    node: int
    label: str = ""
    state: str = "idle"           # idle | enroute | onscene | transport | handover | returning | working
    path: list[int] = field(default_factory=list)
    pi: int = 0
    left_s: float | None = None
    incident: int | None = None
    leg: str = ""
    busy_until: float = 0.0
    target: object | None = None
    edge_in_s: float = 0.0
    trip: list[tuple[int, float, float]] = field(default_factory=list)
    busy_min: float = 0.0
    blocked_wait_s: float = 0.0
    replan_at: float = 0.0


@dataclass
class Incident:
    id: int
    kind: str                     # medical | fire | rescue
    node: int
    severity: int
    t: float                      # minute created
    tag: str = ""
    needs: int = 1
    state: str = "pending"        # pending | responding | onscene | transport | done | unserved | stranded
    units: list[int] = field(default_factory=list)
    arrived: int = 0
    t_assign: float | None = None
    t_arrive: float | None = None
    t_done: float | None = None
    t_delivered: float | None = None
    hospital: int | None = None
    retry_at: float = 0.0
    service_end: float | None = None


def farthest_point_sample(city: CityData, nodes: list[int], k: int) -> list[int]:
    """Pick k of the given nodes spread as far apart as possible (greedy farthest-point)."""
    if k >= len(nodes):
        return list(nodes)
    xy = city.xy[nodes]
    c = xy.mean(axis=0)
    first = int(np.argmin(np.hypot(*(xy - c).T)))
    chosen = [first]
    dist = np.hypot(*(xy - xy[first]).T)
    while len(chosen) < k:
        j = int(np.argmax(dist))
        chosen.append(j)
        dist = np.minimum(dist, np.hypot(*(xy - xy[j]).T))
    return [nodes[j] for j in chosen]


@dataclass
class Resources:
    ambulances: int = 60
    fire_engines: int | None = None          # None => fire_engines_per_station x mapped stations
    rescue_units: int | None = None          # None => one per police station
    crews: int = 3
    extra: list[tuple[str, int]] = field(default_factory=list)      # (kind, node) added by the user


def build_units(city: CityData, res: Resources, A: dict) -> list[Unit]:
    R = A["response"]
    units: list[Unit] = []

    def add(kind, node, label):
        units.append(Unit(len(units), kind, node, node, label))

    # ambulances: 3 at every major hospital, the rest spread across general hospitals (farthest-point sampling)
    t1 = city.hospitals(1)
    n_left = res.ambulances
    for f in t1:
        for _ in range(min(R["ambulances_per_tier1"], n_left)):
            add("ambulance", f.node, f.name); n_left -= 1
    t2 = [f for f in city.hospitals(2) if f.tier == 2]
    posts = city.of_kind("ambulance_post")
    pool = [f.node for f in t2] + [f.node for f in posts]
    for node in farthest_point_sample(city, pool, max(0, n_left)) if pool else []:
        add("ambulance", node, "ambulance post")
    fire_nodes = [f.node for f in city.of_kind("fire_station")]
    n_fire = res.fire_engines if res.fire_engines is not None else R["fire_engines_per_station"] * len(fire_nodes)
    for i in range(n_fire):
        if fire_nodes:
            add("fire", fire_nodes[i % len(fire_nodes)], "fire station")
    police = [f.node for f in city.of_kind("police")]
    n_resc = res.rescue_units if res.rescue_units is not None else R["rescue_units_per_police_station"] * len(police)
    for node in (farthest_point_sample(city, police, n_resc) if police else []):
        add("rescue", node, "police / rescue")
    bases = farthest_point_sample(city, police or [f.node for f in t1], max(1, res.crews))
    for i in range(res.crews):
        add("crew", bases[i % len(bases)], "repair crew")
    for kind, node in res.extra:
        add(kind, node, "added")
    return units
