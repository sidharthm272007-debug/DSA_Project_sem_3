"""Turn the evaluator's numbers into a recommendation a decision-maker can read.

Every sentence is assembled from numbers that were computed; nothing is asserted that the numbers do not show.
The thresholds (what counts as strong, robust, critical) are in ``assumptions.yaml`` and are judgement calls of this
tool, not engineering standards: the verdict text says so.

Sign convention used here (the evaluator reports raw new-minus-old differences):
  * building something: ``people`` = people newly inside the service standard, ``minutes`` = mean minutes saved per resident
  * removing something (how we test whether existing infrastructure matters): the same words with the sign flipped, so
    ``people`` = people who would fall outside the standard and ``minutes`` = mean minutes added per resident.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from advisor.evaluator import SERVICE_LABEL
from advisor.portfolio import KIND_LABEL
from sim.report import fmt_people

SERVICE_SHORT = {"hosp": "a hospital", "t1": "a major hospital", "fire": "a fire engine", "amb": "an ambulance"}
LINK_SERVICES = ("hosp", "fire", "amb")


@dataclass
class Verdict:
    label: str
    tone: str                                   # good | mid | weak
    headline: str
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)


def people(sec: dict, z: str, remove: bool = False) -> float:
    v = float(sec.get(f"{z}_within", 0.0))
    return -v if remove else v


def minutes(sec: dict, z: str, remove: bool = False) -> float:
    v = float(sec.get(f"{z}_mean", 0.0))
    return v if remove else -v


def care_people(sec: dict, services, remove: bool = False) -> float:
    """People brought inside (or pushed outside) a standard, counted once per service."""
    return sum(people(sec, z, remove) for z in services)


def kind_effects(agg: dict, services, remove: bool = False) -> dict[str, float]:
    return {k: care_people(v, services, remove) for k, v in agg["by_kind"].items() if k != "baseline"}


def _joined(items: list[str]) -> str:
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


COST_NOTE = "Cost is a placeholder estimate (assumptions.yaml), not a quote; benefit numbers do not depend on it."
MODEL_NOTE = ("Benefits are averaged over a fixed set of {n} scenarios (a normal day plus the worst moments of earthquakes, fires, floods, "
              "bridge failures and a storm) with the importance weights you chose; they are model outputs on OSM roads and a population proxy, not forecasts. "
              "'People inside the standard' counts people crossing a time threshold, so a small change in travel time near the threshold can move many people in one scenario; "
              "the average minutes saved is the steadier companion figure.")


def judge_build(A: dict, agg: dict, services, cost_crore: float, *, primary: str | None = None, everyday: dict | None = None,
                peer_per_crore: list[float] | None = None, what: str = "This idea", n_scen: int = 0) -> Verdict:
    """Should we build it? ``services`` are the services the idea can improve; ``primary`` the one used for the headline."""
    V, noise = A["advisor"]["verdict"], float(A["advisor"]["noise_floor_people"])
    us = A["advisor"]["usefulness"]
    services = tuple(services)
    gain = care_people(agg["weighted"], services)
    normal = care_people(agg["normal"], services) if agg["normal"] else 0.0
    disaster = care_people(agg["disaster_avg"], services)
    by_kind = kind_effects(agg, services)
    helped = [k for k, v in by_kind.items() if v >= noise]
    unhelped = [k for k, v in by_kind.items() if v < noise]
    eve = everyday or {}
    eve_min = float(eve.get("hub_minutes", 0.0))
    eve_useful = eve_min >= us["everyday_minutes"]
    robust_n = len(helped) + (1 if normal >= noise else 0) + (1 if eve_useful else 0)
    strong = gain >= V["strong_multiple"] * noise or (eve_useful and eve_min >= V["strong_multiple"] * us["everyday_minutes"])
    robust = robust_n >= V["robust_kinds"]
    per_crore = gain / cost_crore if cost_crore > 0 else 0.0
    good_value = None
    if peer_per_crore:
        cut = sorted(peer_per_crore)[min(len(peer_per_crore) - 1, int(len(peer_per_crore) * (1 - V["good_value_share"])))]
        good_value = bool(per_crore >= cut)

    p = primary or services[0]
    unit = "people newly within the standard" if len(services) == 1 else "service-standard gains (people, counted once per service)"
    if gain < noise and not eve_useful:
        label, tone = "Weak case", "weak"
        head = f"{what} would change very little: about {fmt_people(gain)} {unit}, below the {fmt_people(noise)} we treat as noise."
    elif gain < noise and eve_useful:
        label, tone = "Everyday benefit only", "mid"
        head = (f"{what} does little for emergency access ({fmt_people(gain)} {unit}, below the {fmt_people(noise)} we treat as noise) "
                f"but it does shorten everyday peak-hour trips by {eve_min:.2f} min per resident.")
    elif strong and robust:
        label, tone = "Recommended", "good"
        head = f"{what} is a strong, robust addition: about {fmt_people(gain)} {unit} on average, and it helps in {robust_n} different situations."
    else:
        label, tone = "Worth a closer look", "mid"
        why = "the benefit is modest" if not strong else "the benefit is concentrated in few situations"
        head = f"{what} helps, but {why}: about {fmt_people(gain)} {unit} on average."

    reasons: list[str] = []
    if len(services) == 1:
        reasons.append(f"Standard used: {SERVICE_LABEL[p]}. Average resident saves {minutes(agg['weighted'], p):.2f} min to reach {SERVICE_SHORT[p]}.")
    else:
        reasons.append("Standards used: " + "; ".join(SERVICE_LABEL[z] for z in services) + ".")
    if agg["normal"]:
        reasons.append(f"On a normal day: {fmt_people(normal)} people newly within the standard; averaged over the disasters: {fmt_people(disaster)}.")
    if helped:
        reasons.append("Helps in " + _joined([f"{KIND_LABEL.get(k, k).lower()} ({fmt_people(by_kind[k])})" for k in helped]) + ".")
    if unhelped:
        reasons.append("Little or no effect in " + _joined([KIND_LABEL.get(k, k).lower() for k in unhelped]) + ".")
    if eve:
        if eve_useful:
            reasons.append(f"It also helps every day: the average resident saves {eve_min:.2f} min per trip to the main destinations at peak hour "
                           f"(~{fmt_people(eve['hub_hours_per_day'])} person-hours per day city-wide).")
        else:
            reasons.append("It does not noticeably change everyday peak-hour trips to the main destinations.")
    if cost_crore > 0:
        vs = ""
        if good_value is True:
            vs = " This is in the best third of the ideas compared, per rupee."
        elif good_value is False:
            vs = " Other ideas compared give more per rupee."
        reasons.append(f"About {fmt_people(per_crore)} people-gains per crore (placeholder cost Rs {cost_crore:,.0f} crore).{vs}")
    if agg.get("n_hit"):
        reasons.append(f"In {agg['n_hit']} scenario(s) the new road lies inside the damaged zone, so it is assumed lost there and gets no credit.")
    caveats = [MODEL_NOTE.format(n=n_scen or len(agg["per_snapshot"])), COST_NOTE]
    nums = {"gain": gain, "normal": normal, "disaster": disaster, "per_crore": per_crore, "robust_n": robust_n, "strong": strong,
            "eve_minutes": eve_min, "good_value": good_value, "by_kind": by_kind}
    return Verdict(label, tone, head, reasons, caveats, nums)


def judge_existing(A: dict, agg: dict, services, name: str, *, everyday: dict | None = None, kind_word: str = "it") -> Verdict:
    """Is something that already exists useful? Measured by what the city would lose if it were not there."""
    noise = float(A["advisor"]["noise_floor_people"])
    us = A["advisor"]["usefulness"]
    services = tuple(services)
    loss = care_people(agg["weighted"], services, remove=True)
    by_kind = kind_effects(agg, services, remove=True)
    worst_kind = max(by_kind, key=by_kind.get) if by_kind else None
    worst = by_kind.get(worst_kind, 0.0) if worst_kind else 0.0
    normal = care_people(agg["normal"], services, remove=True) if agg["normal"] else 0.0
    eve = everyday or {}
    eve_min = -float(eve.get("hub_minutes", 0.0))              # removal: negative gain = minutes added
    eve_useful = eve_min >= us["everyday_minutes"]
    critical = loss >= us["critical_people"]                          # critical on the weighted average across all situations ...
    one_situation = (not critical) and worst >= us["critical_people"]  # ... or only when one particular kind of emergency strikes
    min_added = sum(minutes(agg["weighted"], z, remove=True) for z in services) / max(len(services), 1)

    if critical:
        label, tone = "Critical", "good"
        head = f"{name} matters a lot: without {kind_word}, about {fmt_people(loss)} more people would fall outside the emergency-access standards on average across the scenarios" + (f", and about {fmt_people(worst)} in the worst kind ({KIND_LABEL.get(worst_kind, worst_kind).lower()})." if worst > loss * 1.2 else ".")
    elif one_situation:
        label, tone = "Critical in one situation", "mid"
        head = (f"{name} is not critical on average (about {fmt_people(loss)} people), but in {KIND_LABEL.get(worst_kind, worst_kind).lower()} about {fmt_people(worst)} more people "
                f"would fall outside the emergency-access standards without {kind_word}.")
    elif loss >= noise or eve_useful:
        label, tone = "Useful", "mid"
        head = f"{name} is useful but not critical: losing {kind_word} would cost about {fmt_people(loss)} people their standard of emergency access" + (" and slow everyday trips." if eve_useful else ".")
    else:
        label, tone = "Little measurable effect", "weak"
        head = f"Losing {name} would change emergency access for fewer than {fmt_people(noise)} people, because other roads or facilities already cover the same ground."
    reasons = []
    if agg["normal"]:
        reasons.append(f"On a normal day, {fmt_people(normal)} people would fall outside the standards without {kind_word}.")
    hurt = [k for k, v in by_kind.items() if v >= noise]
    if hurt:
        reasons.append("It matters most in " + _joined([f"{KIND_LABEL.get(k, k).lower()} ({fmt_people(by_kind[k])} people)" for k in sorted(hurt, key=lambda k: -by_kind[k])[:3]]) + ".")
    if eve:
        reasons.append(f"Everyday: without {kind_word}, the average resident's peak-hour trip to the main destinations would take {eve_min:.2f} min longer." if eve_useful
                       else "Everyday peak-hour trips to the main destinations would barely change.")
    reasons.append(f"Average extra time to reach the services if removed: {min_added:.2f} min per resident.")
    caveats = ["Measured by removing it and re-computing access under every scenario, so it shows the value of what is there today, not what it cost to build.",
               "Scenarios in which it is already destroyed are left out of its average.", MODEL_NOTE.format(n=len(agg["per_snapshot"]))]
    return Verdict(label, tone, head, reasons, caveats, {"loss": loss, "worst": worst, "worst_kind": worst_kind, "normal": normal, "eve_minutes": eve_min, "by_kind": by_kind})


def service_rows(agg: dict, headline_services, remove: bool = False) -> list[dict]:
    """One row per emergency service for the detail table (all four are shown; only some feed the headline)."""
    rows = []
    for z in ("hosp", "t1", "fire", "amb"):
        rows.append({"service": SERVICE_LABEL[z], "key": z,
                     "normal_people": people(agg["normal"], z, remove) if agg["normal"] else 0.0,
                     "disaster_people": people(agg["disaster_avg"], z, remove),
                     "weighted_people": people(agg["weighted"], z, remove),
                     "minutes": minutes(agg["weighted"], z, remove),
                     "in_headline": z in headline_services})
    return rows
