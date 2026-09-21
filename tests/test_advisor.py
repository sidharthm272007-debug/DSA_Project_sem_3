"""Siting, link ideas, verdicts and the advisor service, checked against brute force and hand-built cases."""
import copy

import numpy as np
import pytest

from advisor.evaluator import Evaluator
from advisor.links import generate_links, link_cost_crore, link_geometry, score_links
from advisor.portfolio import build_portfolio
from advisor.service import Advisor
from advisor.siting import PRIMARY, candidate_sites, greedy_sites
from advisor.verdict import judge_build, judge_existing, service_rows

COUNTS = {"earthquake": 2, "fire": 2, "flood": 1, "structure": 1, "storm": 1}


@pytest.fixture(scope="module")
def A2(A):
    a = copy.deepcopy(A)
    a["advisor"]["min_fire_stations"] = 1          # the toy city has one fire station: enough to switch fire numbers on
    return a


@pytest.fixture(scope="module")
def adv(toy, A2):
    return Advisor(toy, A2, portfolio=build_portfolio(toy, A2, counts=COUNTS))


# ---------------------------------------------------------------------------------- siting
def test_first_greedy_pick_is_the_brute_force_best(adv):
    ev = adv.ev
    cands = candidate_sites(ev, "hospital", 30, min_sep_m=250.0)
    assert len(cands) >= 3
    picks = greedy_sites(ev, "hospital", cands, 1)
    # greedy objective = weighted person-minutes saved = -weighted mean minutes; recompute it by brute force for every candidate
    saved = {c: -ev.eval_facility("hospital", c)["weighted"]["hosp_mean"] for c in cands}
    best = max(saved.values())
    assert saved[picks[0].node] == pytest.approx(best, rel=1e-9, abs=1e-12)


def test_later_picks_never_gain_more_than_they_would_alone(adv):
    ev = adv.ev
    cands = candidate_sites(ev, "hospital", 30, min_sep_m=250.0)
    picks = greedy_sites(ev, "hospital", cands, 3)
    for p in picks[1:]:
        alone = -ev.eval_facility("hospital", p.node)["weighted"]["hosp_mean"]
        marginal = -p.effect["weighted"]["hosp_mean"]
        assert marginal <= alone + 1e-9


def test_candidates_are_real_junctions_on_major_roads_and_spread_out(adv):
    ev = adv.ev
    cands = candidate_sites(ev, "fire_station", 20, min_sep_m=400.0)
    xy = ev.city.xy[cands]
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    assert (d[np.triu_indices(len(cands), 1)] >= 400.0).all()
    assert ev.city.giant[cands].all()


def test_need_map_never_exceeds_population(adv):
    nm = adv.need_map("hospital")
    assert nm["people"].sum() <= adv.city.total_pop + 1e-6
    assert (nm["people"] > 0).all()


# ---------------------------------------------------------------------------------- links
def test_link_geometry_flags_river_crossings(adv):
    c = adv.city
    nid = lambda x, y: y * 21 + x                       # toy grid ids; the river runs along y = 4
    _, tau_across, crosses = link_geometry(adv.ev, nid(6, 2), nid(6, 7))
    _, _, same_side = link_geometry(adv.ev, nid(6, 10), nid(6, 15))
    assert crosses and not same_side and tau_across > 0


def test_bridge_costs_more_than_road(A):
    assert link_cost_crore(A, 1000.0, True) > link_cost_crore(A, 1000.0, False) > 0


def test_generated_links_are_sane_and_scoring_matches_eval_link(adv):
    ev = adv.ev
    ideas = generate_links(ev, n_shortlist=6, min_len=250.0, max_len=2500.0, min_ratio=1.2, min_saved_s=30.0)
    assert ideas, "the toy city has a river, so some pair must be much further by road than by air"
    for i in ideas:
        assert i.u != i.v and 250.0 <= i.length_m / ev.ev["link_detour_factor"] <= 2500.0 + 1
        assert i.today_min > i.with_min > 0 and i.cost_crore > 0
    score_links(ev, ideas[:2])
    for i in ideas[:2]:
        ref = ev.eval_link(i.u, i.v, i.tau_s)
        assert i.effect["weighted"] == ref["weighted"]


# ---------------------------------------------------------------------------------- verdicts
def _agg(normal=0.0, by_kind=None, weighted=None, eve=None):
    """Hand-built evaluator output: only the 'hosp' service, with the given people gains."""
    by_kind = by_kind or {}
    z = lambda v: {"hosp_within": v, "hosp_mean": -v / 1000.0}
    kinds = {"baseline": z(normal), **{k: z(v) for k, v in by_kind.items()}}
    dis = float(np.mean(list(by_kind.values()))) if by_kind else 0.0
    w = normal * 0.25 + dis * 0.75 if weighted is None else weighted
    return {"per_snapshot": [0] * 8, "weighted": z(w), "normal": z(normal), "disaster_avg": z(dis), "by_kind": kinds, "n_hit": 0}


def test_verdict_labels_follow_the_thresholds(A):
    noise = A["advisor"]["noise_floor_people"]
    V = A["advisor"]["verdict"]
    strong = _agg(normal=noise * V["strong_multiple"] * 2, by_kind={"earthquake": noise * 4, "flood": noise * 4})
    assert judge_build(A, strong, ("hosp",), 10.0).label == "Recommended"
    nothing = _agg(normal=noise / 10, by_kind={"earthquake": noise / 10})
    assert judge_build(A, nothing, ("hosp",), 10.0).label == "Weak case"
    narrow = _agg(normal=0.0, by_kind={"earthquake": noise * V["strong_multiple"] * 5, "flood": 0.0}, weighted=noise * V["strong_multiple"] * 1.5)
    v = judge_build(A, narrow, ("hosp",), 10.0)
    assert v.label == "Worth a closer look"


def test_verdict_text_quotes_only_computed_numbers(A):
    agg = _agg(normal=1234.0, by_kind={"earthquake": 4321.0})
    v = judge_build(A, agg, ("hosp",), 50.0)
    text = " ".join([v.headline] + v.reasons)
    assert "1.2k" in text and "4.3k" in text                      # numbers appear as computed
    assert v.numbers["per_crore"] == pytest.approx(v.numbers["gain"] / 50.0)


def test_existing_infrastructure_labels(A):
    noise, crit = A["advisor"]["noise_floor_people"], A["advisor"]["usefulness"]["critical_people"]
    hurt = lambda v: _agg(normal=-v, by_kind={"earthquake": -v})           # removal: people fall outside => negative gains
    assert judge_existing(A, hurt(crit * 2), ("hosp",), "X").label == "Critical"
    assert judge_existing(A, hurt(noise * 2), ("hosp",), "X").label == "Useful"
    assert judge_existing(A, hurt(noise / 10), ("hosp",), "X").label == "Little measurable effect"


def test_service_rows_flag_which_services_feed_the_headline(adv):
    eff = adv.ev.eval_facility("hospital", 5 * 21 + 5)
    rows = service_rows(eff, ("hosp",))
    assert [r["key"] for r in rows] == ["hosp", "t1", "fire", "amb"]
    assert [r["in_headline"] for r in rows] == [True, False, False, False]


# ---------------------------------------------------------------------------------- service
def test_fire_numbers_switch_on_only_with_enough_stations(toy, A, A2):
    strict = Advisor(toy, A, portfolio=build_portfolio(toy, A, counts=COUNTS))
    assert not strict.fire_ok and "fire" not in strict.services_all
    assert any("fire station" in n for n in strict.data_notes())
    lax = Advisor(toy, A2, portfolio=build_portfolio(toy, A2, counts=COUNTS))
    assert lax.fire_ok and "fire" in lax.services_all


def test_assess_structure_drops_scenarios_where_it_is_already_destroyed(adv):
    a = adv.assess_structure(0)
    assert a.extra["excluded_scenarios"] >= 1                       # the portfolio includes "failure of this structure"
    assert a.verdict.label in ("Critical", "Useful", "Little measurable effect")
    assert a.verdict.numbers["loss"] >= 0                           # removing something never helps


def test_reweighting_changes_weights_and_clears_cache(adv):
    before = adv.weights_now()
    adv.recommend_sites("hospital", k=1, n_candidates=10)
    assert adv._cache
    adv.reweight({"baseline": 0.0, "earthquake": 1.0, "fire": 0.0, "flood": 0.0, "structure": 0.0, "storm": 0.0})
    assert not adv._cache and adv.weights_now()["earthquake"] == pytest.approx(1.0)
    adv.reweight(None)
    assert adv.weights_now()["baseline"] == pytest.approx(before["baseline"])


def test_assess_link_rejects_identical_ends_and_prices_bridges(adv):
    g = adv.city.g
    a = (float(g.lat[3 * 21 + 3]), float(g.lon[3 * 21 + 3]))
    with pytest.raises(ValueError):
        adv.assess_link(a, a)
    across = adv.assess_link(a, (float(g.lat[8 * 21 + 3]), float(g.lon[8 * 21 + 3])))
    assert across.extra["crosses_water"] and across.cost_crore == pytest.approx(link_cost_crore(adv.A, across.extra["length_m"], True))


def test_better_nearby_never_recommends_a_worse_spot(adv):
    node = 2 * 21 + 2
    r = adv.better_nearby("hospital", node, n=5)
    if r["better"]:
        assert r["value"] > r["here"]


# ---------------------------------------------------------------------------------- elevated structures
def test_nothing_is_ever_attached_to_a_deck_or_tunnel(adv):
    from advisor.links import connectors
    c = adv.city
    assert c.on_structure.any(), "the toy city has a flyover, so some junctions sit on a structure"
    assert not c.on_structure[connectors(adv.ev)].any()
    for idea in ("hospital", "fire_station"):
        assert not c.on_structure[candidate_sites(adv.ev, idea, 40, min_sep_m=100.0)].any()
    ideas = generate_links(adv.ev, n_shortlist=10, min_ratio=1.2, min_saved_s=30.0)
    assert not any(c.on_structure[[i.u, i.v]].any() for i in ideas)


def test_link_verdicts_carry_the_osm_completeness_caveat(adv):
    g = adv.city.g
    a = adv.assess_link((float(g.lat[3 * 21 + 3]), float(g.lon[3 * 21 + 3])), (float(g.lat[8 * 21 + 3]), float(g.lon[8 * 21 + 3])))
    assert any("OpenStreetMap roads only" in c for c in a.verdict.caveats)


def test_link_shortlist_is_spread_across_the_city(adv):
    ideas = generate_links(adv.ev, n_shortlist=12, min_ratio=1.2, min_saved_s=30.0, mid_sep_m=800.0)
    c = adv.city
    mids = np.array([0.5 * (c.xy[i.u] + c.xy[i.v]) for i in ideas])
    if len(mids) > 1:
        d = np.hypot(mids[:, None, 0] - mids[None, :, 0], mids[:, None, 1] - mids[None, :, 1])
        assert (d[np.triu_indices(len(mids), 1)] >= 800.0 - 1e-6).all()


def test_critical_in_one_situation_is_not_called_critical_overall(A):
    crit = A["advisor"]["usefulness"]["critical_people"]
    only_storm = _agg(normal=0.0, by_kind={"earthquake": 0.0, "storm": -crit * 3.0}, weighted=-crit / 10)
    v = judge_existing(A, only_storm, ("hosp",), "X")
    assert v.label == "Critical in one situation" and v.numbers["worst_kind"] == "storm"


def test_planning_brief_is_assembled_from_the_verdicts(adv):
    from advisor.brief import build_brief
    md = build_brief(adv, k_sites=2, n_links=2, n_structures=2)
    top = adv.recommend_sites("hospital", 2)[0]
    assert top.verdict.headline in md and top.verdict.label in md
    assert "## What all of this rests on" in md and "placeholder" in md
    assert "## Fire stations" in md and "Not ranked" not in md              # this fixture has enough fire stations, so they are ranked
    for v in adv.weights_now().values():
        assert f"{v:.0%}" in md


def test_brief_refuses_to_rank_fire_stations_when_data_is_thin(toy, A):
    from advisor.brief import build_brief
    strict = Advisor(toy, A, portfolio=build_portfolio(toy, A, counts=COUNTS))
    md = build_brief(strict, k_sites=1, n_links=1, n_structures=1)
    assert "Not ranked: only" in md and "Fire stations" in md


def test_forked_advisors_do_not_disturb_each_other(toy, A2):
    base = Advisor(toy, A2, portfolio=build_portfolio(toy, A2, counts=COUNTS))
    a, b = base.fork(), base.fork()
    quake_only = {"baseline": 0.0, "earthquake": 1.0, "fire": 0.0, "flood": 0.0, "structure": 0.0, "storm": 0.0}
    everyday = {"baseline": 1.0, "earthquake": 0.0, "fire": 0.0, "flood": 0.0, "structure": 0.0, "storm": 0.0}
    a.reweight(quake_only)
    b.reweight(everyday)
    assert a.weights_now()["earthquake"] == pytest.approx(1.0) and b.weights_now()["baseline"] == pytest.approx(1.0)
    assert base.weights_now()["baseline"] < 1.0 and base.weights_now()["earthquake"] < 1.0          # the shared one was not touched
    # each fork answers exactly as an advisor built from scratch with those weights would
    for fork, w in ((a, quake_only), (b, everyday)):
        fresh = Advisor(toy, A2, portfolio=build_portfolio(toy, A2, counts=COUNTS, weights=w))
        got = fork.recommend_sites("hospital", 2, n_candidates=20)
        ref = fresh.recommend_sites("hospital", 2, n_candidates=20)
        assert [r.pick.node for r in got] == [r.pick.node for r in ref]
        assert [round(r.verdict.numbers["gain"], 6) for r in got] == [round(r.verdict.numbers["gain"], 6) for r in ref]
    c = base.fork()                                                            # default weights: same as the shared advisor, and non-trivial
    same = [r.pick.node for r in c.recommend_sites("hospital", 2, n_candidates=20)]
    assert same and same == [r.pick.node for r in base.recommend_sites("hospital", 2, n_candidates=20)]


def _plain_greedy(ev, idea, candidates, k):
    """The textbook version (recompute every candidate's gain every round), kept here only as a reference for the lazy one."""
    from advisor.evaluator import IDEA_SERVICES
    z, services = PRIMARY[idea], IDEA_SERVICES[idea]
    cur = [{zz: st.d[zz].copy() for zz in services} for st in ev.snaps]
    w = [st.snap.weight for st in ev.snaps]
    chosen, gains, avail = [], [], set(range(len(candidates)))
    for _ in range(k):
        best, bg = None, 0.0
        for ci in sorted(avail):
            g = 0.0
            for si, st in enumerate(ev.snaps):
                R = ev._reach(st, candidates[ci])
                g += w[si] * float((ev.pop * np.maximum(np.minimum(cur[si][z], ev.cap) - np.minimum(R, ev.cap), 0.0)).sum())
            if g > bg + 1e-12:
                best, bg = ci, g
        if best is None:
            break
        for si, st in enumerate(ev.snaps):
            cur[si] = {zz: np.minimum(cur[si][zz], ev._reach(st, candidates[best])) for zz in services}
        chosen.append(candidates[best]); gains.append(bg); avail.discard(best)
    return chosen, gains


def test_lazy_greedy_gives_the_same_sequence_of_gains_as_plain_greedy(adv):
    ev = adv.ev
    for idea in ("hospital", "ambulance_post"):
        cands = candidate_sites(ev, idea, 30, min_sep_m=250.0)
        ref_nodes, ref_gains = _plain_greedy(ev, idea, cands, 4)
        got = greedy_sites(ev, idea, cands, 4)
        assert len(got) == len(ref_nodes)
        z = PRIMARY[idea]
        # the marginal person-time gain of each pick (from its own effect record) must equal the reference gain at that step
        for pick, g in zip(got, ref_gains):
            saved = -pick.effect["weighted"][f"{z}_mean"] * ev.tot * 60.0          # weighted person-seconds saved by this pick
            assert saved == pytest.approx(g, rel=1e-9, abs=1e-6)
