"""Phase P1: the unit head as a stochastic policy, and its hand-written gradients.

Three obligations, each of which has already cost this line a run when it was assumed
instead of asserted:

1. **Greedy parity.** `RecordingPolicy(greedy=True, record_units=True)` must emit exactly
   what `BCPolicy` emits. If it does not, iteration 0 of PPO is already a different agent and
   every later number is about a policy we never evaluated.
2. **Log-prob recompute.** The batched update must reproduce the batch-1 log-probs the
   rollout recorded. Mask drift between the two paths is the single most likely silent bug --
   it would make every importance ratio wrong without raising anything.
3. **Gradient correctness.** Central finite differences on the scalar the analytic gradients
   claim to belong to, plus the advantage-sign pin. A wrong sign here does not crash; it
   produces a clean-looking run that reports "money did not move", which we would read as a
   fact about the game.
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

from agents.team_a.ppo import ppo as PPO            # noqa: E402
from agents.team_a.ppo import unit_head as UH       # noqa: E402
from agents.team_a.ppo.actor import RecordingPolicy  # noqa: E402
from agents.team_a.ppo.bcpath import P, WEIGHTS_DIR  # noqa: E402

M0 = str(WEIGHTS_DIR / "rebuild-m0.npz")


def _f64(mlp):
    return type(mlp).from_state(
        {k: np.asarray(v, dtype=np.float64) for k, v in mlp.state().items()},
        dict(mlp.head_dims))


@pytest.fixture(scope="module")
def rollout_rows():
    from agents.team_a.ppo import harness as H
    turns, urows, _r, _ro = H.rollout(11, 0, [M0], "pi0", 5, temperature=2.0, steps=240,
                                      record_units=True, unit_subsample=1.0)
    assert urows, "phase P1 recorded no unit rows"
    return turns, urows


def _batch(urows, rng, n=64):
    rows = [urows[i] for i in rng.choice(len(urows), size=min(n, len(urows)), replace=False)]
    return {
        "x": np.stack([r["x"] for r in rows]),
        "op_mask": np.stack([r["op_mask"] for r in rows]).astype(np.float32),
        "crop_mask": np.stack([r["crop_mask"] for r in rows]).astype(np.float32),
        "item_mask": np.stack([r["item_mask"] for r in rows]).astype(np.float32),
        "op_idx": np.array([int(r["op_idx"]) for r in rows], dtype=np.int32),
        "crop_idx": np.array([int(r["crop_idx"]) for r in rows], dtype=np.int32),
        "item_idx": np.array([int(r["item_idx"]) for r in rows], dtype=np.int32),
        "qty_idx": np.array([int(r["qty_idx"]) for r in rows], dtype=np.int32),
        "logp_op": np.array([float(r["logp_op"]) for r in rows]),
        "logp_crop": np.array([float(r["logp_crop"]) for r in rows]),
        "logp_item": np.array([float(r["logp_item"]) for r in rows]),
        "logp_qty": np.array([float(r["logp_qty"]) for r in rows]),
        "exec_op": np.array([int(r["exec_op"]) for r in rows]),
        "exec_crop": np.array([int(r["exec_crop"]) for r in rows]),
        "exec_item": np.array([int(r["exec_item"]) for r in rows]),
        "exec_qty": np.array([int(r["exec_qty"]) for r in rows]),
    }


def test_greedy_unit_emission_is_the_bc_policy(rollout_rows):
    """Same observations, same actions -- ops and market alike."""
    from arena import engine as arena_engine
    ke = arena_engine.import_engine()
    env = ke.make("kaggriculture", configuration={"seed": 3, "episodeSteps": 240}, debug=False)
    ref = P.BCPolicy([M0], safety=3)
    rec = RecordingPolicy([M0], safety=3, greedy=True, temperature=1.0, seed=0,
                          record=True, record_units=True)
    seen = 0
    env.run([lambda obs, cfg=None: ref.act(obs), "pass"])
    for step in env.steps:
        obs = step[0]["observation"]
        if obs.get("step", 0) >= 239:
            break
        a, b = ref.act(obs), rec.act(obs)
        assert a["farmer"] == b["farmer"], f"farmer differs at step {obs.get('step')}"
        assert a["hands"] == b["hands"], f"hands differ at step {obs.get('step')}"
        assert a["market"] == b["market"], f"market differs at step {obs.get('step')}"
        seen += 1
    assert seen > 100


def test_recorded_logprobs_recompute(rollout_rows):
    """The batched update reproduces the batch-1 recording. Threshold 1e-5, not 1e-6:
    float32 matmul does not associate the same way at batch 1 and batch B (exp-011 [t3]
    deviation 4). The ratio never uses the recorded value, so this is a drift detector."""
    _turns, urows = rollout_rows
    b = _batch(urows, np.random.default_rng(1), n=256)
    m = P.Member(M0)
    t = PPO.UnitPPO(m.unit, m.umu, m.usd, temperature=2.0)
    lo, lc, li, lq = t.old_logprobs(b, m.unit)
    for name, got, want in (("op", lo, b["logp_op"]), ("crop", lc, b["logp_crop"]),
                            ("item", li, b["logp_item"]), ("qty", lq, b["logp_qty"])):
        d = np.abs(got - want).max()
        assert d < 1e-5, f"{name} log-prob drift {d:.3e}"


def test_conditional_factors_exist_only_where_their_op_fired(rollout_rows):
    _turns, urows = rollout_rows
    from agents.team_a.ppo.bcpath import F
    for r in urows:
        op = F.OPS[int(r["op_idx"])]
        assert (int(r["crop_idx"]) >= 0) == (op == "PLANT")
        assert (int(r["item_idx"]) >= 0) == (op in ("PICKUP", "PLACE"))
        assert (int(r["qty_idx"]) >= 0) == (op == "PICKUP")


def test_unit_gradients_match_finite_differences(rollout_rows):
    _turns, urows = rollout_rows
    rng = np.random.default_rng(3)
    b = _batch(urows, rng, n=192)
    m = P.Member(M0)
    t = PPO.UnitPPO(_f64(m.unit), m.umu, m.usd, entropy_coef=1e-2, kl_beta=0.5,
                    temperature=2.0)
    anchor = UH.mixture_dists([P.Member(p) for p in sorted(
        str(p) for p in WEIGHTS_DIR.glob("*.npz"))], b["x"], b["op_mask"],
        b["crop_mask"], b["item_mask"])
    adv = rng.normal(0, 1, len(b["x"]))
    logp_old = list(t.old_logprobs(b, PPO.clone_mlp(t.unit)))
    grads, stats = t.loss_grads(b, adv, logp_old, anchor)
    assert stats["n_factors"] > 50, "degenerate fixture: nothing survived the override mask"

    def loss():
        return t.loss_grads(b, adv, logp_old, anchor)[1]["loss"]

    params = PPO.head_params(t.unit, t.HEADS)
    eps, checked = 1e-6, 0
    for pi, (par, g) in enumerate(zip(params, grads)):
        flat, gflat = par.reshape(-1), np.asarray(g).reshape(-1)
        for j in rng.choice(len(flat), size=min(5, len(flat)), replace=False):
            keep = flat[j]
            flat[j] = keep + eps
            lp = loss()
            flat[j] = keep - eps
            lm = loss()
            flat[j] = keep
            fd, an = (lp - lm) / (2 * eps), float(gflat[j])
            scale = max(1e-8, abs(fd), abs(an))
            assert abs(fd - an) / scale < 2e-4, (
                f"unit param block {pi} entry {j}: analytic {an:.6e} vs fd {fd:.6e}")
            checked += 1
    assert checked >= 20


def test_unit_advantage_sign(rollout_rows):
    _turns, urows = rollout_rows
    b = _batch(urows, np.random.default_rng(5), n=256)

    def after_one_step(sign):
        m = P.Member(M0)
        t = PPO.UnitPPO(_f64(m.unit), m.umu, m.usd, lr=1e-2, entropy_coef=0.0, kl_beta=0.0,
                        temperature=2.0)
        old = list(t.old_logprobs(b, PPO.clone_mlp(t.unit)))
        t.step(b, sign * np.ones(len(b["x"])), old)
        return old, list(t.old_logprobs(b, t.unit))

    for sign in (+1.0, -1.0):
        old, new = after_one_step(sign)
        m = (b["op_idx"] >= 0) & (b["exec_op"] > 0)
        d = float(((new[0] - old[0]) * m).sum() / max(1, m.sum()))
        assert np.sign(d) == sign, f"op head moved the wrong way at advantage {sign}"


def test_overridden_unit_factors_carry_no_gradient(rollout_rows):
    _turns, urows = rollout_rows
    b = _batch(urows, np.random.default_rng(9), n=128)
    dead = dict(b)
    for k in ("exec_op", "exec_crop", "exec_item", "exec_qty"):
        dead[k] = np.zeros_like(b[k])
    m = P.Member(M0)
    t = PPO.UnitPPO(m.unit, m.umu, m.usd, temperature=2.0)
    grads, stats = t.loss_grads(dead, np.ones(len(b["x"])), list(t.old_logprobs(b, m.unit)))
    assert grads is None and stats.get("skipped") == 1
