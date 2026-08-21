"""Invariants of the PPO rollout recorder (spec sections 3.1 / 3.2 / 3.3).

These are the tests that make the recorded data trustworthy. In order of how much damage
their failure would do silently:

1. **greedy parity** -- `RecordingPolicy(greedy=True)` must make exactly the decisions the
   shipped `BCPolicy` makes. If it does not, PPO starts from a different agent than the one
   we measured, and every comparison against the base is meaningless.
2. **log-prob recompute** -- re-running the forward pass on the recorded features must
   reproduce every recorded log-prob. This is the check for mask drift between the rollout
   and the update, the single most likely silent bug in a factored PPO.
3. **determinism** -- same episode seed + same policy rng seed => identical trajectory.
4. **override mask** -- at safety 3 every SELL factor and every day-29 order factor must be
   marked overridden, and a factor marked executed must actually appear in the emitted
   action.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")
if not list((ROOT / "agents" / "team_a" / "bc" / "weights").glob("*.npz")):
    pytest.skip("no trained BC weights present", allow_module_level=True)

from agents.team_a.ppo import harness as H          # noqa: E402
from agents.team_a.ppo import market_head as MH     # noqa: E402
from agents.team_a.ppo.actor import RecordingPolicy  # noqa: E402
from agents.team_a.ppo.bcpath import F, P           # noqa: E402
from arena import engine as arena_engine            # noqa: E402

STEPS = 240          # 10 game-days: enough to exercise buying, selling and hiring


def _episode(agent_fns, seed=3, steps=STEPS):
    ke = arena_engine.import_engine()
    env = ke.make("kaggriculture", configuration={"seed": seed, "episodeSteps": steps},
                  debug=False)
    env.run(agent_fns)
    return env


def test_greedy_recording_policy_matches_shipped_policy():
    ref = P.BCPolicy(H.PI0_MEMBERS, safety=3)
    rec = RecordingPolicy(H.PI0_MEMBERS, safety=3, greedy=True, seed=0, record=True)
    seen = {"n": 0, "bad": 0}

    def under_test(obs, config=None):
        got, want = rec.act(obs), ref.act(obs)
        seen["n"] += 1
        seen["bad"] += (got != want)
        return got

    _episode([under_test, lambda obs, config=None: ref.act(obs)])
    assert seen["n"] > 200
    assert seen["bad"] == 0, f"{seen['bad']}/{seen['n']} greedy decisions differ from BCPolicy"


def _recorded(greedy=False, rng_seed=7, seed=3):
    pol = RecordingPolicy(H.PI0_MEMBERS, safety=3, greedy=greedy, seed=rng_seed, record=True)
    opp = P.BCPolicy(H.PI0_MEMBERS, safety=3)
    _episode([lambda o, c=None: pol.act(o), lambda o, c=None: opp.act(o)], seed=seed)
    return pol


def _recompute(pol, mkt=None):
    m = pol.members[0]
    x = np.stack([t["x"] for t in pol.turns])
    mask = np.stack([t["order_mask"] for t in pol.turns])
    fire = np.stack([t["fire"] for t in pol.turns])
    qty = np.stack([t["qty_idx"] for t in pol.turns]).astype(np.int32)
    hire = np.array([int(t["hire_idx"]) for t in pol.turns])
    d = MH.dists(mkt or m.mkt, m.mmu, m.msd, x, mask, pol.th_logit, pol.temperature)
    got = MH.logprobs(d, fire, qty, hire)
    rec = (np.stack([t["logp_fire"] for t in pol.turns]),
           np.stack([t["logp_qty"] for t in pol.turns]),
           np.array([float(t["logp_hire"]) for t in pol.turns]))
    return got, rec


def test_logprob_recompute_is_exact():
    """Recomputed through the same `dists()` the update uses.

    Tolerance is **1e-5 absolute, not the spec's 1e-6** -- a deliberate, recorded deviation.
    The network is float32 on purpose: it is byte-identical to the shipped artefact, which is
    this stack's entire export-risk story. A float32 dot product over 128 hidden units does
    not associate the same way at batch 1 (rollout) as at batch B (update), so an order logit
    carries ~5e-6 of absolute error between the two paths, and the log-prob inherits it. 1e-6
    is unreachable in float32 without making the trainer and the artefact different programs,
    which is a worse trade.

    The check keeps all of its power: a mask that drifted between rollout and update moves a
    log-prob by O(1) or to -inf, never by 5e-6. `test_logprob_recompute_catches_mask_drift`
    shows that even a 0.01-nat perturbation of the head is caught.

    (The update does not divide the recorded log-probs into the new ones either: it recomputes
    pi_old batched, from a frozen copy of the pre-update parameters, so the PPO ratio is exact
    by construction and this recording is purely the drift detector. See `ppo.py`.)
    """
    pol = _recorded()
    got, rec = _recompute(pol)
    for g, r in zip(got, rec):
        assert np.abs(g - r).max() <= 1e-5


def test_logprob_recompute_catches_mask_drift():
    """Positive control: perturb the head and the invariant must fail loudly."""
    pol = _recorded()
    m = pol.members[0]
    drifted = type(m.mkt).from_state({k: np.array(v, copy=True) for k, v in m.mkt.state().items()},
                                     dict(m.mkt.head_dims))
    drifted.heads["orders"][1] = drifted.heads["orders"][1] + 0.01   # 0.01 nats, tiny
    got, rec = _recompute(pol, mkt=drifted)
    err = np.abs(got[0] - rec[0])
    assert (err > 1e-6 * np.maximum(1.0, np.abs(rec[0]))).any(), \
        "a perturbed order head slipped through the recompute check"


def test_rollout_is_deterministic():
    a, b = _recorded(rng_seed=11), _recorded(rng_seed=11)
    assert len(a.turns) == len(b.turns)
    for ta, tb in zip(a.turns, b.turns):
        for k in ("x", "order_mask", "fire", "qty_idx", "logp_fire", "exec_fire", "exec_qty"):
            assert np.array_equal(ta[k], tb[k]), f"non-deterministic field {k}"
        assert int(ta["hire_idx"]) == int(tb["hire_idx"])
    c = _recorded(rng_seed=12)
    assert any(not np.array_equal(ta["fire"], tc["fire"]) for ta, tc in zip(a.turns, c.turns)), \
        "a different policy rng seed produced an identical trajectory -- sampling is not live"


def test_override_mask_drops_every_wrapper_owned_factor():
    pol = _recorded()
    sells = np.nonzero(MH.SELL_ORDERS)[0]
    for t in pol.turns:
        assert not t["exec_fire"][sells].any(), "safety 3 rewrites SELL; those bits must be masked"
        assert not t["exec_qty"][sells].any()
        if t["day"] >= F.DAYS - 1:
            assert not t["exec_fire"].any() and not t["exec_qty"].any() and not t["exec_hire"], \
                "day 29 orders are rewritten by _endgame"
        # a masked-out order is not a factor at all
        assert not (t["exec_fire"] * (t["order_mask"] == 0)).any()


def test_some_buy_factors_do_reach_the_engine():
    """The mask must not be vacuous: if nothing is ever executed there is no gradient."""
    pol = _recorded()
    buys = np.array([t == "BUY_SEED" or t == "BUY_PRODUCT" or t == "BUY_ANIMAL"
                     or t == "BUY_LAND" for t, _ in F.ORDER_TYPES])
    executed = sum(int(t["exec_fire"][buys].sum()) for t in pol.turns)
    factors = sum(int((t["order_mask"] > 0)[buys].sum()) for t in pol.turns)
    assert factors > 0
    assert executed / factors > 0.5, \
        f"only {executed}/{factors} buy factors reached the engine -- the wrapper owns the head"
    assert sum(int(t["exec_hire"]) for t in pol.turns) > 0
