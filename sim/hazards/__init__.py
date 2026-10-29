"""Hazard events -- package index.

Every disaster is a footprint x time profile x effect on roads and facilities. The shared contract and the
area-hazard base class live in ``base.py``; each concrete hazard type has its own file:

    base.py                Hazard / AreaHazard contracts, IncidentSpec, HazardContext, WorldView
    earthquake.py           Earthquake
    fire.py                 Fire (urban fire)
    gas_leak.py             GasLeak (chemical release)
    flood.py                FloodZone
    custom_event.py         CustomEvent (building collapse, crowd surge, cordon, ...)
    structure_failure.py    StructureFailure (a specific bridge/flyover collapses or closes)
    accident.py             Accident (major road accident + queue spillback)
    storm.py                Storm (city-wide tree-fall / cyclone)

Importing ``sim.hazards`` (as the rest of the codebase does: ``from sim.hazards import Fire``) re-exports
every one of these names, so this split is invisible to every other module -- only the implementation moved.
"""
from sim.hazards.accident import Accident
from sim.hazards.base import AreaHazard, Hazard, HazardContext, IncidentSpec, WorldView, circle_polygon, ellipse_polygon
from sim.hazards.custom_event import CustomEvent
from sim.hazards.earthquake import Earthquake
from sim.hazards.fire import Fire
from sim.hazards.flood import FloodZone
from sim.hazards.gas_leak import GasLeak
from sim.hazards.storm import Storm
from sim.hazards.structure_failure import StructureFailure

__all__ = [
    "IncidentSpec", "HazardContext", "WorldView", "circle_polygon", "ellipse_polygon",
    "Hazard", "AreaHazard",
    "Earthquake", "Fire", "GasLeak", "FloodZone", "CustomEvent", "StructureFailure", "Accident", "Storm",
]
