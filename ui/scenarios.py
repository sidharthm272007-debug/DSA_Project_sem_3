"""Scenario specs (plain dicts the UI can keep in session state) -> real Hazard objects; plus ready-made presets."""
from __future__ import annotations

from sim.engine import Context, Scenario
from sim.hazards import Accident, CustomEvent, Earthquake, Fire, FloodZone, GasLeak, StructureFailure, Storm
from sim.response import Resources

HAZARD_TYPES = {
    "earthquake": "Earthquake", "fire": "Urban fire", "plume": "Gas leak / chemical release", "flood_zone": "Flood",
    "structure_failure": "Bridge / flyover failure", "accident": "Major road accident", "storm": "Severe storm / cyclone", "custom": "Custom event (anything else)",
}
NEEDS_PLACE = {"earthquake", "fire", "plume", "accident", "custom"}


def place_lookup(city) -> dict[str, tuple[float, float]]:
    """Every named place a user can choose: localities, stations, hospitals, flyovers. Name -> (lat, lon)."""
    out: dict[str, tuple[float, float]] = {}
    for l in sorted(city.landmarks, key=lambda l: (l["kind"] not in ("suburb", "town", "neighbourhood"), l["name"])):
        out.setdefault(l["name"], (l["lat"], l["lon"]))
    return out


def make_hazard(city, spec: dict):
    t, start = spec["type"], float(spec.get("start_min", 0.0))
    lat, lon = spec.get("lat"), spec.get("lon")
    sev = int(spec.get("severity", 3))
    name = spec.get("name", "")
    if t == "earthquake":
        return Earthquake(lat, lon, magnitude=float(spec.get("magnitude", 6.0)), start_min=start, secondary_fires=bool(spec.get("secondary_fires", True)), name=name)
    if t == "fire":
        return Fire(lat, lon, severity=sev, start_min=start, name=name)
    if t == "plume":
        return GasLeak(lat, lon, severity=sev, start_min=start, name=name)
    if t == "flood_zone":
        return FloodZone(spec.get("water"), severity=sev, start_min=start, name=name) if spec.get("water") else FloodZone(None, severity=sev, start_min=start, lat=lat, lon=lon, name=name)
    if t == "structure_failure":
        return StructureFailure(structure=city.structures[int(spec["structure"])], mode=spec.get("mode", "collapse"), start_min=start, name=name)
    if t == "accident":
        return Accident(lat, lon, severity=sev, start_min=start, name=name)
    if t == "storm":
        return Storm(severity=sev, start_min=start, name=name)
    if t == "custom":
        return CustomEvent(lat, lon, radius_m=float(spec.get("radius_m", 300)), closed=bool(spec.get("closed", True)), slow_factor=float(spec.get("slow_factor", 0.3)),
                           start_min=start, duration_min=float(spec.get("duration_min", 240)), casualties=int(spec.get("casualties", 0)), name=name or "Custom event")
    raise ValueError(f"unknown hazard type {t!r}")


def make_scenario(city, name: str, specs: list[dict], ctx: dict, duration_h: float, seed: int, resources: dict | None = None, incident_scale: float = 1.0) -> Scenario:
    r = resources or {}
    return Scenario(name=name, hazards=[make_hazard(city, s) for s in specs], duration_h=float(duration_h), seed=int(seed),
                    context=Context(time_of_day=ctx.get("time_of_day", "off_peak"), weather=ctx.get("weather", "clear"), wind_toward_deg=float(ctx.get("wind_toward_deg", 90.0)),
                                    wind_kmh=float(ctx.get("wind_kmh", 12.0))),
                    resources=Resources(ambulances=int(r.get("ambulances", 60)), crews=int(r.get("crews", 3))), incident_scale=float(incident_scale))


def _at(city, place: str):
    lat, lon = place_lookup(city)[place]
    return {"lat": lat, "lon": lon, "place": place}


def presets(city) -> dict[str, dict]:
    """Ready-made scenarios on real places. Each is only a starting point: every field can be changed in the builder."""
    flyover = next((i for i, s in enumerate(city.structures) if "Naidu" in s["name"]), 0)
    return {
        "M6.0 earthquake under the city centre": {
            "specs": [{"type": "earthquake", "magnitude": 6.0, "secondary_fires": True, "start_min": 0, **_at(city, "Gandhipuram")}],
            "ctx": {"time_of_day": "off_peak", "weather": "clear"}, "hours": 8,
            "blurb": "Roads, flyovers and hospitals fail together and fires start afterwards. How fast does help recover?"},
        "Noyyal river in flood, heavy rain": {
            "specs": [{"type": "flood_zone", "water": "Noyyal River", "severity": 4, "start_min": 0}],
            "ctx": {"time_of_day": "off_peak", "weather": "heavy_rain"}, "hours": 12,
            "blurb": "Water rises along the Noyyal River during heavy rain and cuts the low crossings."},
        "Fire with toxic smoke near Singanallur": {
            "specs": [{"type": "fire", "severity": 4, "start_min": 0, **_at(city, "Singanallur")},
                      {"type": "plume", "severity": 3, "start_min": 10, **_at(city, "Singanallur")}],
            "ctx": {"time_of_day": "peak", "weather": "clear", "wind_toward_deg": 60, "wind_kmh": 14}, "hours": 6,
            "blurb": "A spreading fire and a wind-driven gas cloud at evening peak. Does the fire get contained, and who is trapped?"},
        "Avinashi Road flyover fails at rush hour": {
            "specs": [{"type": "structure_failure", "structure": flyover, "mode": "collapse", "start_min": 0}],
            "ctx": {"time_of_day": "peak", "weather": "clear"}, "hours": 6,
            "blurb": "The G.D. Naidu elevated corridor is lost. What does the rest of the road network absorb?"},
        "Severe cyclone / storm": {
            "specs": [{"type": "storm", "severity": 4, "start_min": 0}],
            "ctx": {"time_of_day": "night", "weather": "storm"}, "hours": 10,
            "blurb": "City-wide fallen trees and slow roads at night; crews clear debris while calls pile up."},
        "Earthquake during the monsoon (compound)": {
            "specs": [{"type": "earthquake", "magnitude": 6.2, "secondary_fires": True, "start_min": 0, **_at(city, "Ukkadam")},
                      {"type": "flood_zone", "water": "Periyakulam (Ukkadam Lake)", "severity": 3, "start_min": 60}],
            "ctx": {"time_of_day": "off_peak", "weather": "heavy_rain"}, "hours": 10,
            "blurb": "Shaking closes roads and hospitals near Ukkadam, then the lake floods an hour later in heavy rain."},
    }
