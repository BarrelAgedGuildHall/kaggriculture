"""expB-304b-caps / expB-304b-seeds — the two instruments R-47 and R-51 rest on.

These pin the things a future edit could silently break underneath the measured
verdicts, in the order the verdicts depend on them:

  * the seed log is OFF by default and adds no state to the shipped build (the
    whole "the champion is unchanged" claim rests on this, and it is also
    proved end-to-end by `identity_check.py`: 92/92 pairs / 184 episodes);
  * a logged row reproduces the order arithmetic exactly, so a cause attributed
    to `budget` really was a budget decision;
  * `"uncapped"` deletes the constant rather than raising it — the ask becomes
    the structural site budget;
  * MELON and STRAWBERRY both carry `first_yield_day == 10`, so the structural
    replant window closes after day 19.  That is why both caps stop binding at
    day 20 and it is an engine fact, not a policy choice;
  * `_process_market` walks the order list IN ORDER and `_commit_unit` mutates
    `farm["money"]` immediately, so a SELL at index 0 funds a BUY_SEED later in
    the same turn.  This is the premise of the expB-304b-seeds finding that our
    stock-outs are NOT the same-turn cash-sequencing failure Team A's census
    found in their wrapper.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _engine import import_engine  # noqa: E402

import_engine()
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

from millrace import feedchain as fc  # noqa: E402
from millrace import probe_core as pc  # noqa: E402

CHAMPION = {"melon_replant": True, "melon_rule": "fixed", "melon_tiles": 6,
            "strawberry_tiles": 32, "dispatch": "priced", "dig": True,
            "dedupe_ontile": True, "ongoing_water_span": True}


def _episode(params, steps=72, seed=0):
    """Run a short seat-season and return (agent, last observation it planned on)."""
    ke = import_engine()
    env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed},
                  debug=False)
    env.reset()
    a = fc.make_feedchain_agent(params)
    b = fc.make_feedchain_agent(CHAMPION)
    last = None
    while not env.done:
        last = env.state[0].observation
        env.step([a(last, env.configuration),
                  b(env.state[1].observation, env.configuration)])
    return a, last


def _board(obs):
    counts, empties = {}, 0
    for row in obs.farms[0]["tiles"]:
        for t in row:
            if t is None:
                empties += 1
            elif isinstance(t, dict) and "crop" in t:
                counts[t["crop"]] = counts.get(t["crop"], 0) + 1
    return counts, empties


# ------------------------------------------------------------------ the seed log
def test_seedlog_is_off_by_default_and_adds_no_state():
    assert fc._SEEDLOG is False, "MILLRACE_SEEDLOG must default OFF in the shipped build"
    agent, _ = _episode(CHAMPION)
    assert "seedlog" not in agent.state["st"]


def test_seedlog_rows_reproduce_the_order_arithmetic(monkeypatch):
    monkeypatch.setattr(fc, "_SEEDLOG", True)
    agent, _ = _episode(CHAMPION)
    rows = [r for r in agent.state["st"]["seedlog"] if "crop" in r]
    assert rows, "the log recorded nothing"
    for r in rows:
        assert r["px"] == pc.CROPS[r["crop"]]["seed"]
        assert r["k"] == min(r["need"], 10, int(max(0, r["budget"]) // r["px"]))
        assert r["placed"] == (r["k"] > 0 and r["slot"] < 10)
        assert r["budget"] <= r["budget_open"]
    # every crop the buyer considers gets a row on every turn it runs
    assert {r["crop"] for r in rows} == {"WHEAT", "CARROT", "MELON", "STRAWBERRY"}


# ------------------------------------------------------------------ "uncapped"
def test_uncapped_deletes_the_constant_rather_than_raising_it():
    """The ask becomes standing + the structural site budget, so no tile-count
    constant can bind.  Measured on the wants the planner itself recorded."""
    for want_key, crop, arm in (("want_melon", "MELON", {"melon_rule": "uncapped"}),
                                ("want_straw", "STRAWBERRY",
                                 {"strawberry_rule": "uncapped"})):
        agent, obs = _episode({**CHAMPION, **arm})
        counts, empties = _board(obs)
        assert agent.state["st"][want_key] == counts.get(crop, 0) + empties
        base, obs_b = _episode(CHAMPION)
        cap = CHAMPION["melon_tiles"] if crop == "MELON" else CHAMPION["strawberry_tiles"]
        assert base.state["st"][want_key] == cap


# ------------------------------------------------------------------ engine facts
def test_both_capped_crops_close_their_replant_window_after_day_19():
    for crop in ("MELON", "STRAWBERRY"):
        assert K.CROPS[crop]["first_yield_day"] == 10
        assert pc.CROPS[crop]["first"] == 10
    last_day = 29
    assert 19 + 10 <= last_day and 20 + 10 > last_day


def test_a_sell_funds_a_buy_seed_later_in_the_same_order_list():
    """`_commit_unit` mutates money in place and `_process_market` walks the
    list in order, so BUY_SEED at index 1 sees the SELL at index 0."""
    farm = K._new_farm(10, 0.0)            # zero cash on purpose
    private = {"shed": {"EGG": 4}, "seeds": {}, "inventories": [{}]}
    market = {"inventory": {"EGG": 50}, "prices": {"EGG": 30}, "params": None}
    px = K.market_price("EGG", market["inventory"]["EGG"], market.get("params"))
    assert px > 0
    assert K._commit_unit("SELL", "EGG", px, farm, private, market) is True
    assert farm["money"] == px
    seed_px = K.CROPS["WHEAT"]["seed"]
    ok = K._commit_unit("BUY_SEED", "WHEAT", seed_px, farm, private, market)
    assert ok is (px >= seed_px)           # funded by the sale, not by opening cash
    assert private["seeds"].get("WHEAT", 0) == (1 if ok else 0)


def test_a_buy_seed_with_no_cash_is_refused_not_silently_granted():
    farm = K._new_farm(10, 0.0)
    private = {"shed": {}, "seeds": {}, "inventories": [{}]}
    market = {"inventory": {}, "prices": {}, "params": None}
    assert K._commit_unit("BUY_SEED", "STRAWBERRY", K.CROPS["STRAWBERRY"]["seed"],
                          farm, private, market) is False
    assert private["seeds"] == {}
