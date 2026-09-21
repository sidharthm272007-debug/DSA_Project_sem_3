"""A planning brief: the advisor's recommendations for the whole city in one document a decision-maker can read, share and question.

Everything is assembled from the same computed verdicts the app shows; nothing is written by hand. Each recommendation keeps its reasons and the caveats it rests on.
"""
from __future__ import annotations

from datetime import date

from advisor.portfolio import KIND_LABEL
from advisor.service import IDEA_LABEL, Advisor


def _block(title: str, v, extra: str = "") -> list[str]:
    out = [f"### {title}", "", f"**{v.label}.** {v.headline}", ""]
    out += [f"- {r}" for r in v.reasons]
    if extra:
        out.append(f"- {extra}")
    return out + [""]


def build_brief(adv: Advisor, k_sites: int = 3, n_links: int = 5, n_structures: int = 5, progress=None) -> str:
    city, A = adv.city, adv.A
    steps = 3 + (1 if adv.fire_ok else 0) + 3
    done = [0]

    def tick():
        done[0] += 1
        if progress:
            progress(min(done[0] / steps, 1.0))

    md = [f"# Emergency-access planning brief: Coimbatore", "", f"Generated {date.today().isoformat()} by ResQGraph from the model and data described in the last section. "
          "It is decision support: each recommendation is a model result with its reasons and limits, not a forecast or an engineering design.", "",
          "## How much each situation counted", ""]
    md += [f"- {KIND_LABEL.get(k, k)}: {w:.0%}" for k, w in adv.weights_now().items()] + [""]

    md += ["## Where to build", ""]
    for idea in ("hospital", "ambulance_post") + (("fire_station",) if adv.fire_ok else ()):
        recs = adv.recommend_sites(idea, k_sites)
        tick()
        md += [f"## {IDEA_LABEL[idea]}s", ""]
        for r in recs:
            md += _block(f"#{r.pick.rank} {r.pick.where} (about Rs {r.cost_crore:,.0f} crore, placeholder)", r.verdict)
    if not adv.fire_ok:
        md += ["## Fire stations", "", f"Not ranked: only {adv.n_fire} fire station(s) are mapped, far fewer than the city has. Add the missing ones to data/coimbatore/facilities_extra.csv and re-run.", ""]

    md += ["## New roads and bridges", ""]
    links = adv.recommend_links(n=n_links)
    tick()
    for r in links:
        i = r.idea
        md += _block(f"{i.where_u} <-> {i.where_v} ({'bridge' if i.crosses_water else 'road'}, {i.length_m / 1000:.1f} km, about Rs {i.cost_crore:,.0f} crore, placeholder)", r.verdict)

    md += ["## Is what exists worth it?", "", "Each flyover or bridge was removed in turn and every scenario re-scored; what the city would lose is its value today.", ""]
    ranked = adv.rank_structures()
    tick()
    for a in ranked[:n_structures]:
        md += _block(a.title, a.verdict)
    md += ["### Major hospitals", ""]
    for a in adv.rank_hospitals(1):
        md += [f"- **{a.title}**: {a.verdict.label}. {a.verdict.headline}"]
    tick()
    md += ["", "## What all of this rests on", ""]
    md += [f"- {n}" for n in adv.data_notes()] + [f"- Costs are placeholders except the press-reported Rs {A['advisor']['cost_crore']['flyover_per_km']} crore/km flyover figure.",
                                                    f"- Benefits are averaged over {adv.n_scen} scenarios; the sample and the weights above shape the answer.", ""]
    return "\n".join(md)
