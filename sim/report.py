"""Turn a finished simulation into a summary a decision-maker can read.

Every number here is computed from the run. Every sentence in ``findings`` is assembled from those numbers.
Nothing is hard-coded, and if something did not happen (no incidents, no recovery) the summary says so.
"""
from __future__ import annotations

import numpy as np


def _pct(a: list[float], q: float) -> float | None:
    return float(np.percentile(a, q)) if a else None


def _stats(vals: list[float]) -> dict:
    return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None, "median": _pct(vals, 50), "p90": _pct(vals, 90),
            "max": float(max(vals)) if vals else None}


def fmt_people(x: float) -> str:
    x = float(x)
    if x >= 100_000:
        return f"{x / 100_000:.1f} lakh"
    if x >= 1000:
        return f"{x / 1000:.1f}k"
    return f"{x:.0f}"


def summarize_run(city, A: dict, res) -> tuple[dict, dict]:
    """Returns (summary, peak). ``res`` is a SimResult (only its plain data fields are used)."""
    M = res.metrics
    base = res.baseline["summary"]
    step = (res.times[1] - res.times[0]) if len(res.times) > 1 else 10.0
    golden = A["impact"]["golden_minutes"]

    # ---- peak moment ----------------------------------------------------------------------------
    score = [m["gt60_any"] + 1e-3 * m["gt30_any"] for m in M]
    ip = int(np.argmax(score))
    pk = M[ip]
    peak = {"index": ip, "t": pk["t"], **{k: pk[k] for k in ("cut_any", "gt30_any", "gt60_any", "cut_t1", "gt30_t1", "gt60_t1",
                                                              "mean_any_min", "mean_t1_min", "blocked_km", "hospitals_closed", "exposed")}}

    # ---- population-hours without timely care (excess over the no-disaster baseline) ----------------
    extra30 = [max(0.0, m["gt30_any"] - base["gt30_any"]) for m in M]
    person_hours = float(sum(extra30) * step / 60.0)
    extra60 = [max(0.0, m["gt60_any"] - base["gt60_any"]) for m in M]
    noise = float(A["advisor"]["noise_floor_people"])
    peak_extra = extra30[ip]
    never_hit = peak_extra < noise and pk["cut_any"] < noise
    recovered_at = None                                    # first time after the peak that the added burden is down to 10% of its peak
    if not never_hit:
        for i in range(ip, len(M)):
            if extra30[i] <= 0.1 * peak_extra + noise and M[i]["cut_any"] < noise + 0.1 * pk["cut_any"]:
                recovered_at = M[i]["t"]
                break

    # ---- incidents ---------------------------------------------------------------------------------
    inc = res.incidents
    by_kind: dict[str, dict] = {}
    for k in ("medical", "fire", "rescue"):
        xs = [i for i in inc if i.kind == k]
        arrive = [i.t_arrive - i.t for i in xs if i.t_arrive is not None]
        done = [i for i in xs if i.state == "done"]
        by_kind[k] = {"total": len(xs), "served": len(done), "unserved": sum(1 for i in xs if i.state == "unserved"),
                      "stranded": sum(1 for i in xs if i.state == "stranded"), "response_min": _stats(arrive)}
    med3 = [i for i in inc if i.kind == "medical" and i.severity == 3]
    golden_ok = [i for i in med3 if i.t_delivered is not None and i.t_delivered - i.t <= golden]
    to_hosp = [i.t_delivered - i.t for i in inc if i.kind == "medical" and i.t_delivered is not None]
    total_inc = len(inc)
    failed = sum(1 for i in inc if i.state in ("unserved", "stranded"))

    # ---- hospitals -----------------------------------------------------------------------------------
    worst_ratio: dict[int, float] = {}
    for fr in res.frames:
        for fid, r in fr["hospital_ratio"].items():
            if r is not None:
                worst_ratio[fid] = min(worst_ratio.get(fid, 1.0), r)
    closed_ids = {fid for fr in res.frames for fid in fr["closed_fac"]}
    hosp_hit = sorted(((city.by_id[fid].name, r, fid in closed_ids) for fid, r in worst_ratio.items() if r < 0.7), key=lambda x: x[1])
    closed_major = sorted(city.by_id[f].name for f in closed_ids if city.by_id[f].tier == 1)
    closed_other = sum(1 for f in closed_ids if city.by_id[f].tier != 1)

    # ---- roads -----------------------------------------------------------------------------------------
    pf = res.frames[ip]
    names: dict[str, float] = {}
    for e in pf["blocked"][:4000]:
        nm = city.way_names.get(int(city.g.way_id[e]))
        if nm:
            names[nm] = names.get(nm, 0.0) + float(city.g.length_m[e])
    top_blocked = [n for n, _ in sorted(names.items(), key=lambda x: -x[1])[:6]]
    rl = res.repair_log
    repaired_km = float(M[-1]["repaired_km"])

    # ---- fleet ---------------------------------------------------------------------------------------------
    by_unit: dict[str, list[float]] = {}
    for u in res.units:
        by_unit.setdefault(u.kind, []).append(u.busy_min)
    util = {k: float(np.mean(v)) / max(res.times[-1], 1.0) for k, v in by_unit.items() if v}
    stuck = float(sum(u.blocked_wait_s for u in res.units) / 60.0)

    summary = {
        "duration_min": res.times[-1], "state_step_min": step, "runtime_s": res.runtime_s,
        "population": city.total_pop, "baseline": base,
        "person_hours_without_timely_care": person_hours, "recovered_at_min": recovered_at, "never_hit": never_hit,
        "incidents_total": total_inc, "incidents_failed": failed, "by_kind": by_kind,
        "golden": {"critical_total": len(med3), "within": len(golden_ok), "minutes": golden},
        "to_hospital_min": _stats(to_hosp),
        "hospitals_hit": hosp_hit, "closed_major": closed_major, "closed_other": closed_other,
        "data_flags": {"fire_stations_mapped": len(city.of_kind("fire_station")), "ambulance_posts_mapped": len(city.of_kind("ambulance_post"))},
        "top_blocked_roads": top_blocked, "repairs": len(rl), "repaired_km": repaired_km,
        "unit_utilisation": util, "unit_minutes_stuck": stuck,
        "extra60_peak": max(extra60) if extra60 else 0.0,
    }
    summary["findings"] = findings(city, A, summary, peak)
    return summary, peak


def findings(city, A: dict, s: dict, pk: dict) -> list[str]:
    """Plain-language findings, each one traceable to a number in ``summary``."""
    out: list[str] = []
    base = s["baseline"]
    tot = s["population"]
    slow = A["impact"]["slow_minutes"]
    h, m = divmod(int(pk["t"]), 60)
    extra = pk["gt30_any"] - base["gt30_any"]
    if s["never_hit"]:
        out.append(f"No measurable loss of hospital access: fewer than {A['advisor']['noise_floor_people']} extra people fell beyond {slow} minutes of a hospital.")
    else:
        line = f"At the worst moment (hour {h}:{m:02d}) about {fmt_people(max(extra, 0))} more people than normal ({max(extra, 0) / tot:.1%} of the city) were more than {slow} minutes from the nearest open hospital"
        line += f", and {fmt_people(pk['cut_any'])} had no road route to any hospital at all." if pk["cut_any"] >= A["advisor"]["noise_floor_people"] else "."
        out.append(line)
        if pk["gt30_t1"] - base["gt30_t1"] > 0.02 * tot:
            out.append(f"For major (tier-1) hospitals the effect is larger: {fmt_people(pk['gt30_t1'])} people beyond {slow} minutes versus {fmt_people(base['gt30_t1'])} normally.")
        if s["recovered_at_min"] is not None:
            rh, rm = divmod(int(s["recovered_at_min"]), 60)
            work = f"{s['repairs']} debris-clearing jobs were completed ({s['repaired_km']:.1f} km reopened)" if s["repairs"] else "no debris-clearing jobs were needed or completed"
            out.append(f"Access recovered to within 10% of the peak burden at hour {rh}:{rm:02d}; {work}.")
        else:
            work = f"{s['repairs']} clearing jobs were completed ({s['repaired_km']:.1f} km reopened)" if s["repairs"] else "no clearing jobs were completed"
            out.append(f"Access had not recovered by the end of the run ({s['duration_min'] / 60:.0f} h); {work}.")
        out.append(f"That is about {fmt_people(s['person_hours_without_timely_care'])} person-hours beyond {slow} minutes of a hospital, on top of normal.")
    if s["closed_major"]:
        out.append("Major hospitals put out of service: " + ", ".join(s["closed_major"][:4]) + ".")
    if s["closed_other"]:
        out.append(f"{s['closed_other']} other hospitals or nursing homes were put out of service.")
    cut_off = [(n, r) for n, r, closed in s["hospitals_hit"] if not closed]
    if cut_off:
        out.append("Major hospitals still open but with a much smaller reachable catchment: " + ", ".join(f"{n} ({r:.0%} of normal)" for n, r in cut_off[:3]) + ".")
    for k, label in (("medical", "medical calls"), ("fire", "fire calls"), ("rescue", "rescue calls")):
        b = s["by_kind"][k]
        if b["total"] >= 3 and b["response_min"]["median"] is not None:
            nf = b["unserved"] + b["stranded"]
            out.append(f"{b['total']} {label}: median first-unit arrival {b['response_min']['median']:.0f} min, 90th percentile {b['response_min']['p90']:.0f} min; "
                       f"{nf} {'was' if nf == 1 else 'were'} never served or could not be delivered to care.")
        elif b["total"] >= 3:
            out.append(f"{b['total']} {label}: none received a unit ({b['unserved'] + b['stranded']} unserved).")
    nfs = s["data_flags"]["fire_stations_mapped"]
    if s["by_kind"]["fire"]["total"] >= 1 and nfs < int(A["advisor"]["min_fire_stations"]):
        out.append(f"Data gap: only {nfs} fire station(s) are mapped, so fire-response times and unserved fire calls here are worse than the real city would see. "
                   "Add the missing stations in data/coimbatore/facilities_extra.csv.")
    g = s["golden"]
    if g["critical_total"] >= 3:
        out.append(f"{g['within']} of {g['critical_total']} critical patients reached a hospital within {g['minutes']} minutes.")
    if s["top_blocked_roads"]:
        out.append("Named roads with blocked stretches at the peak: " + ", ".join(s["top_blocked_roads"][:5]) + ".")
    if s["unit_minutes_stuck"] > 30:
        out.append(f"Responding vehicles lost about {s['unit_minutes_stuck']:.0f} vehicle-minutes waiting at closed roads before finding another route.")
    return out
