"""expB-304a — invariants of the dispatch pricer, pinned against the engine.

Three classes of test:
  1. the cadence/stream arithmetic matches what `kaggriculture` actually does
     (counted by simulating the engine's own refresh, not by restating my
     formula);
  2. the pricer carries NO hard-coded per-commodity PRICE (ruling-02
     addendum-02 A-7) and no absolute-day constant;
  3. the whole arm still degrades safely and the CONTROL lineage is untouched.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agents" / "team_b"))
sys.path.insert(0, str(REPO / "scripts"))

from millrace import pricer as PR          # noqa: E402
from millrace.feedchain import make_feedchain_agent  # noqa: E402
from millrace.probe_core import CROPS      # noqa: E402

import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402


CHAMPION = {"melon_replant": True, "melon_rule": "mr", "strawberry_tiles": 16}
ARM = dict(CHAMPION, dispatch="priced", dig=True, dedupe_ontile=True,
           ongoing_water_span=True)


class FakeModel:
    """Flat $10 market: every marginal revenue/cost is exactly n * 10."""

    def marginal_revenue(self, product, n):
        return 10.0 * int(n)

    def marginal_buy_cost(self, product, n, skip=0):
        return 10.0 * int(n)


def _pricer(day, last_day=29, lam=0.0):
    return PR.Pricer(FakeModel(), CROPS, day, 0, last_day, lam, True)


# ---------------------------------------------------------------- engine truth
@pytest.mark.parametrize("animal", ["GOOSE", "COW", "SHEEP"])
@pytest.mark.parametrize("placed_day,day", [(0, 0), (0, 10), (3, 3), (12, 20), (25, 26)])
def test_animal_events_left_matches_engine(animal, placed_day, day):
    """Count the engine's own production events by replaying
    `_daily_refresh_animals`'s arithmetic day by day, and require the closed
    form to agree. An event that resolves at the end of day d is only bankable
    if there is a day d+1 to harvest on."""
    last_day = 29
    a = K.ANIMALS[animal]
    want = 0
    for d in range(day, last_day):
        next_day = d + 1
        k = next_day - placed_day - a["first_yield_day"]
        if k >= 0 and k % a["interval"] == 0:
            want += 1
    assert _pricer(day, last_day).animal_events_left(animal, placed_day) == want


def test_engine_tables_match_the_pricer_copy():
    """`ANIMAL_ENGINE` / `CROP_MAX_YIELD` are vendored engine constants (the
    observation does not expose the engine's tables). Pin them, so an engine
    bump breaks a test instead of silently mispricing every animal."""
    for animal, ae in PR.ANIMAL_ENGINE.items():
        e = K.ANIMALS[animal]
        assert (ae["first"], ae["interval"], ae["max_held"]) == \
               (e["first_yield_day"], e["interval"], e["max_held"]), animal
    for crop, cap in PR.CROP_MAX_YIELD.items():
        assert cap == K.CROPS[crop]["max_yield"], crop
    for animal, product in PR.ANIMAL_PRODUCT.items():
        assert product == K.ANIMALS[animal]["product"], animal


def test_cadence_count_brute_force():
    for first_ok in range(0, 8):
        for interval in (1, 2, 3):
            for lo in range(0, 10):
                for hi in range(-1, 12):
                    want = sum(1 for d in range(lo, hi + 1)
                               if d >= first_ok and (d - first_ok) % interval == 0)
                    assert PR._cadence_count(first_ok, interval, lo, hi) == want


def test_strawberry_plant_value_counts_four_events_when_the_season_allows():
    """STRAWBERRY: first 10, interval 2, max_yield 4 -> eves at ages 9/11/13/15,
    each bankable the following day. Planted on day 0 all four fit; planted late
    enough, none do."""
    p = _pricer(0, last_day=29, lam=0.0)
    # 4 events x 1 unit x $10, less the $100 seed, less zero action cost
    assert p.fresh_plant_value("STRAWBERRY") == pytest.approx(4 * 10.0 - 100)
    late = _pricer(25, last_day=29, lam=0.0)
    assert late.fresh_plant_value("STRAWBERRY") == pytest.approx(-100)


def test_routine_feed_is_refused_when_the_care_bonus_does_not_pay():
    """Engine: base production and `fertilizer_available` are unconditional;
    `fed_today` only buys the CARE bonus. So a routine feed whose bonus is worth
    less than the wheat must price negative — the legal half-cost keep-alive."""
    p = _pricer(5, lam=5.0)
    animals = [((1, 1), {"animal": "GOOSE", "placed_day": 0, "fed_today": False,
                         "cared_today": False, "consecutive_unfed": 0,
                         "yield_units": 0, "fertilizer_available": False})]
    urgent = set()
    rows = p.animal_tasks(animals, wheat_stock=50, urgent=urgent)
    feed = [r for r in rows if r[2] == ["FEED"]][0]
    # value = bonus(10) - wheat(10) - lam(5) = -5  -> prio = +5
    assert feed[0] == pytest.approx(5.0)
    assert ("F", 1, 1) not in urgent


def test_critical_feed_is_urgent_and_priced_at_the_whole_stream():
    """The only irrecoverable failure in this game. It must be flagged urgent
    (so it bypasses the lambda cut) and priced above ordinary work."""
    p = _pricer(5, lam=5.0)
    tile = {"animal": "COW", "placed_day": 0, "fed_today": False,
            "cared_today": False, "consecutive_unfed": 1, "yield_units": 0,
            "fertilizer_available": False}
    urgent = set()
    rows = p.animal_tasks([((2, 2), tile)], wheat_stock=50, urgent=urgent)
    feed = [r for r in rows if r[2] == ["FEED"]][0]
    assert ("F", 2, 2) in urgent
    assert feed[0] == pytest.approx(-p.animal_stream("COW", 0))
    assert feed[0] < 0


def test_dig_is_offered_only_when_the_tile_unlocks_something():
    p = _pricer(5, lam=1.0)
    assert p.dig_tasks([(0, 0)], plant_value=0.0, build_value=0.0) == []
    rows = p.dig_tasks([(0, 0), (1, 0)], plant_value=50.0, build_value=0.0)
    assert len(rows) == 2
    assert rows[0][2] == ["DIG"] and rows[0][0] == pytest.approx(-(50.0 - 1.0))


# ---------------------------------------------------------------- section 0
def test_pricer_carries_no_price_constant():
    """ruling-02 addendum-02 A-7. Every dollar figure the pricer uses comes from
    M1's live observed table; the only numeric literals in the module are engine
    cadences, caps and small integer arithmetic. This is what makes the agent
    immune to pricing a pool the field has abandoned."""
    tree = ast.parse((REPO / "agents" / "team_b" / "millrace" / "pricer.py").read_text())
    nums = {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)}
    # 0,1,2,3,4,6,8 = first_yield_day (4/8/6), interval (1/2/3), max_held (4/6),
    # max_yield (4/6) and the engine's 3-day fertilizer window.
    assert nums <= {0, 1, 2, 3, 4, 6, 8}, sorted(nums)
    assert "marginal_revenue" in (REPO / "agents" / "team_b" / "millrace" / "pricer.py").read_text()


# ---------------------------------------------------------------- behaviour
def _obs_smoke(params, seed=0, steps=120):
    import kaggle_environments as ke
    env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed},
                  debug=True)
    ag = make_feedchain_agent(params)
    env.reset()
    env.run([ag, ag])
    return env, ag


def test_arm_runs_clean_and_the_control_lineage_is_untouched():
    env_a, ag_a = _obs_smoke(ARM)
    assert [s.status for s in env_a.state] == ["DONE", "DONE"]
    env_c, _ = _obs_smoke(CHAMPION)
    assert [s.status for s in env_c.state] == ["DONE", "DONE"]
    # the two must differ: the flags are not inert
    assert [s.reward for s in env_a.state] != [s.reward for s in env_c.state]


def test_both_lambda_series_are_logged_and_lambda_opp_is_not_degenerate():
    """ruling-02 section 2.7: a lambda_opp pinned at the lambda_wage floor means
    the pricer is silently a wage gate and the arm is VOID, not failed. The
    assertion has to be checkable from outside the agent."""
    _, ag = _obs_smoke(ARM, steps=720)
    log = ag.state["st"]["lam_log"]
    assert len(log) >= 25
    for row in log:
        assert {"day", "lam_wage", "lam_opp", "lam_cut", "pinned"} <= set(row)
    pinned = sum(1 for r in log if r["pinned"])
    assert pinned < len(log) // 2, f"lambda_opp pinned on {pinned}/{len(log)} days"
    assert max(r["lam_cut"] for r in log) > 0.0


def test_mirror_is_deterministic_and_seat_symmetric():
    import kaggle_environments as ke
    banks = []
    for _ in range(2):
        env = ke.make("kaggriculture", configuration={"episodeSteps": 240, "seed": 7},
                      debug=False)
        env.reset()
        env.run([make_feedchain_agent(ARM), make_feedchain_agent(ARM)])
        banks.append([s.reward for s in env.state])
    assert banks[0] == banks[1]
    assert banks[0][0] == banks[0][1]
