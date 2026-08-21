"""The PPO update's gradients are hand-written; this checks them against the loss.

`ppo.py` differentiates the clipped surrogate, the entropy bonus and the KL anchor
analytically w.r.t. the head logits and pushes them through `MLP`'s backward. There is no
autodiff to catch a sign slip, and the failure mode of a wrong gradient in this programme is
not a crash -- it is a run that reports "money did not move", which we would then wrongly
read as a fact about the game. So: central finite differences on the scalar the gradients
claim to belong to.

Run in float64 (a float32 round-trip would swamp a 1e-6 perturbation); production stays
float32, byte-identical to the artefact.
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

from agents.team_a.ppo import market_head as MH   # noqa: E402
from agents.team_a.ppo import ppo as PPO          # noqa: E402
from agents.team_a.ppo.bcpath import F, P, WEIGHTS_DIR  # noqa: E402


def _f64(mlp):
    return type(mlp).from_state({k: np.asarray(v, dtype=np.float64) for k, v in mlp.state().items()},
                                dict(mlp.head_dims))


@pytest.fixture(scope="module")
def fixture():
    """A REAL batch, not synthetic noise.

    An earlier version drew Gaussian features. After the policy's own normalisation those
    are wildly out of distribution: hidden activations reached ~2e3, the 13-way hire softmax
    saturated to a one-hot, and the loss became numerically flat while the analytic gradient
    stayed finite -- the finite-difference check failed on the *fixture*, not on the code.
    A short recorded episode keeps every head in the regime PPO actually trains in.
    """
    from agents.team_a.ppo import harness as H
    rng = np.random.default_rng(0)
    m = P.Member(str(WEIGHTS_DIR / "rebuild-m0.npz"))
    th = m.ens_thresholds if m.ens_thresholds is not None else m.thresholds
    thl = MH.logit(np.asarray(th, dtype=np.float64))
    trainer = PPO.MarketPPO(_f64(m.mkt), m.mmu, m.msd, thl, entropy_coef=1e-2,
                            kl_beta=0.5, temperature=2.0)
    trainer.anchor = PPO.clone_mlp(trainer.mkt)
    for k in ("orders", "qty", "hire"):     # a perturbed anchor -> a live KL gradient
        trainer.anchor.heads[k][1] = (trainer.anchor.heads[k][1]
                                      + rng.normal(0, 0.05, trainer.anchor.heads[k][1].shape))

    turns, _, _ = H.rollout(5, 0, [str(WEIGHTS_DIR / "rebuild-m0.npz")], "pi0", 3,
                            temperature=2.0, steps=240)
    turns = [turns[i] for i in rng.choice(len(turns), size=48, replace=False)]
    b = {"x": np.stack([t["x"] for t in turns]),
         "order_mask": np.stack([t["order_mask"] for t in turns]),
         "fire": np.stack([t["fire"] for t in turns]).astype(np.int32),
         "qty_idx": np.stack([t["qty_idx"] for t in turns]).astype(np.int32),
         "hire_idx": np.array([int(t["hire_idx"]) for t in turns]),
         "exec_fire": np.stack([t["exec_fire"] for t in turns]),
         "exec_qty": np.stack([t["exec_qty"] for t in turns]),
         "exec_hire": np.array([int(t["exec_hire"]) for t in turns])}
    adv = rng.normal(0, 1, len(turns))
    logp_old = list(trainer.old_logprobs(b, PPO.clone_mlp(trainer.mkt)))
    anchor_d = trainer.dists(trainer.anchor, b["x"], b["order_mask"])
    return trainer, b, adv, logp_old, anchor_d


def _loss(trainer, b, adv, logp_old, anchor_d):
    _, stats = trainer.loss_grads(b, adv, logp_old, anchor_d)
    return stats["loss"]


def test_analytic_head_gradients_match_finite_differences(fixture):
    trainer, b, adv, logp_old, anchor_d = fixture
    grads, stats = trainer.loss_grads(b, adv, logp_old, anchor_d)
    assert stats["n_factors"] > 100, "degenerate fixture: nothing survives the override mask"

    params = PPO.head_params(trainer.mkt, trainer.HEADS)
    rng = np.random.default_rng(7)
    eps = 1e-6
    checked = 0
    for pi, (par, g) in enumerate(zip(params, grads)):
        flat = par.reshape(-1)
        gflat = np.asarray(g).reshape(-1)
        picks = rng.choice(len(flat), size=min(6, len(flat)), replace=False)
        for j in picks:
            keep = flat[j]
            flat[j] = keep + eps
            lp = _loss(trainer, b, adv, logp_old, anchor_d)
            flat[j] = keep - eps
            lm = _loss(trainer, b, adv, logp_old, anchor_d)
            flat[j] = keep
            fd = (lp - lm) / (2 * eps)
            an = float(gflat[j])
            scale = max(1e-8, abs(fd), abs(an))
            assert abs(fd - an) / scale < 2e-4, (
                f"param block {pi} entry {j}: analytic {an:.6e} vs finite-difference {fd:.6e}")
            checked += 1
    assert checked >= 30


def test_clipped_factors_get_no_gradient(fixture):
    """Above the trust region a positive-advantage factor is flat: no gradient there.

    (The asymmetry is PPO's, not a bug: for a *negative* advantage the pessimistic `min`
    keeps the unclipped branch however large the ratio grows, which is what stops the policy
    running away from a bad action.)
    """
    trainer, b, adv, logp_old, _ = fixture
    pos = np.abs(adv)
    plain = PPO.MarketPPO(trainer.mkt, trainer.mmu, trainer.msd, trainer.th_logit,
                          entropy_coef=0.0, kl_beta=0.0, temperature=trainer.temperature)
    far = [lp - 5.0 for lp in logp_old]        # ratio ~ e^5, far above 1 + clip
    g_far, st_far = plain.loss_grads(b, pos, far, None)
    g_near, st_near = plain.loss_grads(b, pos, logp_old, None)
    assert st_far["clipfrac"] > 0.9
    far_norm = sum(float(np.abs(x).sum()) for x in g_far)
    near_norm = sum(float(np.abs(x).sum()) for x in g_near)
    assert far_norm < 1e-9, f"clipped factors still carry gradient ({far_norm:.3e})"
    assert near_norm > 0


def test_override_mask_zeroes_the_gradient_of_overridden_factors(fixture):
    """A batch where nothing was executed must produce no policy gradient at all."""
    trainer, b, adv, logp_old, anchor_d = fixture
    dead = dict(b)
    dead["exec_fire"] = np.zeros_like(b["exec_fire"])
    dead["exec_qty"] = np.zeros_like(b["exec_qty"])
    dead["exec_hire"] = np.zeros_like(b["exec_hire"])
    grads, stats = trainer.loss_grads(dead, adv, logp_old, anchor_d)
    assert grads is None and stats.get("skipped") == 1
