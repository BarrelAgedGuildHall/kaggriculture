"""The terminal-bank line decomposition is EXACT, and the accountant is invisible.

exp-014's factored critic is only a decomposition if the lines sum to the bank. This pins the
identity against the live engine (so an engine bump that adds a cash path fails here rather
than silently mis-attributing every critic target), and pins that instrumenting an episode
changes no action -- the same discipline the harness/Pit calibration establishes for banks.
"""
from __future__ import annotations

import numpy as np

from agents.team_a.ppo import harness as H
from agents.team_a.ppo import lines as L
from arena import engine as arena_engine

SEED = 3000
STEPS = 720
START = 3000.0


def _play(seed, seat, accountant=False, steps=STEPS):
    ke = arena_engine.import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    env = ke.make("kaggriculture",
                  configuration={"seed": int(seed), "episodeSteps": int(steps)}, debug=False)
    actor = H.make_policy("pi0", safety=3, greedy=True)
    opp = H.make_policy("base", safety=3, greedy=True)
    agents = [None, None]
    agents[seat] = lambda obs, config=None: actor.act(obs)
    agents[1 - seat] = lambda obs, config=None: opp.act(obs)
    if not accountant:
        env.run(agents)
        last = env.steps[-1]
        return [float(last[0]["observation"]["farms"][i]["money"]) for i in (0, 1)], None
    acc = L.Accountant(K).attach(env)
    with acc:
        env.run(agents)
    last = env.steps[-1]
    return [float(last[0]["observation"]["farms"][i]["money"]) for i in (0, 1)], acc


def test_line_of_covers_every_cash_moving_op():
    for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                 "EGG", "MILK", "WOOL", "FERTILIZER"):
        assert L.line_of("SELL", item) is not None, item
    assert L.line_of("BUY_PRODUCT", "WHEAT") == L.line_of("BUY_SEED", "MELON")
    assert L.line_of("BUY_ANIMAL", "COW") == L.line_of("BUY_LAND", "-")
    assert L.line_of("HIRE", "-") is not None
    # ops that move no cash must map to None, not to line 0
    assert L.line_of("PASS", "-") is None
    assert L.line_of("SELL", "NOT_A_PRODUCT") is None


def test_the_lines_sum_to_the_bank_to_the_cent():
    money, acc = _play(SEED, 0, accountant=True)
    for seat in (0, 1):
        r = acc.residual(seat, money[seat], START)
        assert abs(r) < 1e-6, (
            f"seat {seat}: bank {money[seat]:.2f} vs start+lines "
            f"{START + acc.totals(seat).sum():.2f} (residual {r:.6f}) -- the engine has a "
            f"cash path lines.py does not know about")
    tot = acc.totals(0)
    assert tot.shape == (L.N_LINES,)
    # a real episode must actually touch several lines, or the identity is vacuous
    assert int((tot != 0).sum()) >= 4


def test_the_accountant_changes_no_action():
    clean, _ = _play(SEED, 0, accountant=False)
    instrumented, _acc = _play(SEED, 0, accountant=True)
    assert clean == instrumented
    # and the residual on a clean re-run is still nested-safe: no accountant is active
    assert not L._ACTIVE


def test_sell_and_cost_lines_have_the_expected_signs():
    _money, acc = _play(SEED, 1, accountant=True)
    tot = acc.totals(1)
    idx = {n: i for i, n in enumerate(L.LINES)}
    for n in ("sell_staple", "sell_premium", "sell_animal", "sell_fertilizer"):
        assert tot[idx[n]] >= 0.0, n
    for n in ("cost_inputs", "cost_capital", "cost_labour"):
        assert tot[idx[n]] <= 0.0, n
    assert np.isfinite(tot).all()
