"""Packaging parity for the Team A behaviour-cloning agent.

The Pokemon-challenge team shipped a wrapper that bypassed the policy's feature path and
agreed with it on only 62.9% of decisions -- "a correct model behind an incorrect interface
is a different policy". These tests assert that the packaged entry point and the policy
object make the SAME decisions, that decisions are deterministic, and that every emitted
action is well formed under the engine's rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BC = Path(__file__).resolve().parents[1] / "agents" / "team_a" / "bc"
sys.path.insert(0, str(BC))

pytest.importorskip("numpy")
if not list(BC.glob("weights/*.npz")):
    pytest.skip("no trained BC weights present", allow_module_level=True)

import features as F      # noqa: E402
import main as entry      # noqa: E402
import policy as P        # noqa: E402


def _observations(n=40, seed=0):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 240, "seed": seed}, debug=False)
    env.run(["random", "random"])
    return [s[0]["observation"] for s in env.steps[:n] if isinstance(s[0].get("observation"), dict)]


@pytest.fixture(scope="module")
def obss():
    return _observations()


def test_packaged_agent_matches_policy(obss):
    pol = P.load_default()
    for obs in obss:
        assert entry.agent(obs) == pol.act(obs)


def test_deterministic(obss):
    pol = P.load_default()
    for obs in obss:
        assert pol.act(obs) == pol.act(obs)


def test_actions_well_formed(obss):
    pol = P.load_default()
    for obs in obss:
        act = pol.act(obs)
        ctx = F.TurnContext(obs)
        assert set(act) == {"farmer", "hands", "market"}
        assert isinstance(act["farmer"], list) and act["farmer"][0] in F.OPS
        assert len(act["hands"]) == ctx.n_units - 1
        assert all(h[0] in F.OPS for h in act["hands"])
        assert len(act["market"]) <= F.MAX_MARKET_ORDERS
        # The engine drops EVERY PLANT for a crop whose requests exceed the seeds held.
        want = {}
        for op in [act["farmer"]] + act["hands"]:
            if op[0] == "PLANT":
                want[op[1]] = want.get(op[1], 0) + 1
        for crop, k in want.items():
            assert k <= ctx.seeds.get(crop, 0), f"{k} PLANT {crop} but {ctx.seeds.get(crop,0)} seeds"


def test_never_raises_on_degenerate_obs():
    """A crashed agent is scored as a loss, so the entry point must swallow everything."""
    assert entry.agent({"player": 0}) == {"farmer": ["PASS"], "hands": [], "market": []}


# --------------------------------------------------------------------- exp-022a
def _liq_probe(order):
    """`_liquidate` on a hand-built context: 4 animals (reserve 12 WHEAT), a shed that can
    fund three SELL lines, and two buy-side orders already in the list."""
    import types

    class Ctx:
        tiles = [[{"animal": "COW"}] * 2, [{"animal": "COW"}] * 2]
        shed = {"WHEAT": 20, "MILK": 5, "WOOL": 7}
        prices = {"WHEAT": 40, "MILK": 170, "WOOL": 160}

    pol = types.SimpleNamespace(liquidate_order=order)
    orders = [["BUY_LAND"], ["HIRE"], ["SELL", "MILK", 1]]
    return P.BCPolicy._liquidate(pol, Ctx(), orders)


def test_liquidate_default_is_buys_first():
    """The wrapper pin (integration-ruling I-3): the shipped default must stay `buys_first`
    until the exp-013b/exp-014 pair closes, because every PPO rollout runs on it."""
    import inspect
    sig = inspect.signature(P.BCPolicy.__init__)
    assert sig.parameters["liquidate_order"].default == "buys_first"
    assert inspect.signature(P.load_default).parameters["liquidate_order"].default == "buys_first"


def test_liquidate_order_only_reorders():
    """exp-022a changes the SEQUENCE, never the CONTENTS: the same orders in both arms, and
    the truncation budget (`room = MAX - len(kept)`) is computed the same way. If this ever
    stops holding, the arm is measuring two variables."""
    buys = _liq_probe("buys_first")
    sells = _liq_probe("sells_first")
    assert sorted(map(repr, buys)) == sorted(map(repr, sells))
    assert len(buys) <= F.MAX_MARKET_ORDERS
    # the net's own SELL orders are dropped by the rule in BOTH arms
    assert [o for o in buys if o[0] == "SELL"] == [o for o in sells if o[0] == "SELL"]
    kept = [o for o in buys if o[0] != "SELL"]
    n_sell = len(buys) - len(kept)
    assert [o[0] for o in buys[:len(kept)]] == [o[0] for o in kept]
    assert all(o[0] == "SELL" for o in sells[:n_sell])
    assert [o[0] for o in sells[n_sell:]] == [o[0] for o in kept]


def test_liquidate_order_rejects_unknown_value():
    import pytest as _pytest
    members = sorted(str(p) for p in BC.glob("weights/*.npz"))
    with _pytest.raises(ValueError):
        P.BCPolicy(members, safety=3, liquidate_order="cheapest_first")
