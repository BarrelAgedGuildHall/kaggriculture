"""The market head as a *stochastic policy*, plus its exact log-prob recompute.

Two things this module has to get right, because everything downstream rests on them.

**1. The stochastic policy's greedy mode must be the shipped policy, bit for bit.**
`policy.BCPolicy._market` does not argmax the order head -- it fires order `i` when the
averaged probability exceeds a calibrated threshold `th[i]` (F1-tuned; see
`calibrate_ensemble.py`). Sampling a Bernoulli on the raw probability would therefore be
sampling around a *different* decision boundary than the one we ship, and iteration 0 of PPO
would already be a different agent. Instead the threshold is folded in as a constant logit
offset:

    z_i = logit(p_i) - logit(th_i)   =>   sigmoid(z_i) > 0.5  <=>  p_i > th_i

so `greedy=True` reproduces the BC decision exactly and `greedy=False` samples around it.
`test_ppo_market_head.py` asserts the equivalence on real observations.

**2. Recompute must be exact.** The rollout stores features, masks and sampled indices; the
update recomputes the distribution from the *current* parameters. If the recompute disagrees
with the rollout at pi_old, the ratio is silently wrong. `dists()` is the single
implementation used by both paths, which is what makes the <=1e-6 invariant meaningful.
"""
from __future__ import annotations

import numpy as np

from . import bcpath
from .bcpath import F

N_ORDERS = F.N_ORDERS
N_QTY = F.N_QTY
MAX_HIRES = F.MAX_HIRES
SELL_ORDERS = np.array([t == "SELL" for t, _ in F.ORDER_TYPES], dtype=bool)


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def dists(mkt, mmu, msd, x, order_mask, th_logit, temperature=1.0, cache=False):
    """Batched forward -> the three factor distributions.

    x            (B, D) raw market-turn features
    order_mask   (B, N_ORDERS) 1 = the order is emittable this turn
    th_logit     (N_ORDERS,) constant offset folded in from the calibrated thresholds
    returns dict: p_fire (B,20), p_qty (B,20,18), p_hire (B,13), plus logits and the cache
    """
    xn = ((np.asarray(x, dtype=np.float32) - mmu) / msd)
    if cache:
        out, cch = mkt.forward(xn, cache=True)
    else:
        out, cch = mkt.forward(xn), None
    z_fire = (out["orders"].astype(np.float64) - th_logit[None, :]) / temperature
    p_fire = 1.0 / (1.0 + np.exp(-np.clip(z_fire, -30.0, 30.0)))
    p_fire = np.where(np.asarray(order_mask) > 0, p_fire, 0.0)

    zq = out["qty"].reshape(-1, N_ORDERS, N_QTY).astype(np.float64) / temperature
    zq = zq - zq.max(axis=2, keepdims=True)
    eq = np.exp(zq)
    p_qty = eq / np.maximum(eq.sum(axis=2, keepdims=True), 1e-12)

    zh = out["hire"].astype(np.float64) / temperature
    zh = zh - zh.max(axis=1, keepdims=True)
    eh = np.exp(zh)
    p_hire = eh / np.maximum(eh.sum(axis=1, keepdims=True), 1e-12)
    return {"p_fire": p_fire, "p_qty": p_qty, "p_hire": p_hire,
            "z_orders": out["orders"], "cache": cch}


def logprobs(d, fire, qty_idx, hire_idx):
    """Per-factor log-probabilities of the recorded actions. Shapes mirror the recording.

    `qty_idx < 0` marks "no qty factor this turn for this order" (it did not fire).
    Masked-out orders carry log-prob 0 and are excluded by the caller's factor mask.
    """
    eps = 1e-12
    p = d["p_fire"]
    lf = np.where(fire > 0, np.log(np.maximum(p, eps)), np.log(np.maximum(1.0 - p, eps)))
    b = np.arange(p.shape[0])
    lq = np.zeros_like(lf)
    has_q = qty_idx >= 0
    if has_q.any():
        bi, oi = np.nonzero(has_q)
        lq[bi, oi] = np.log(np.maximum(d["p_qty"][bi, oi, qty_idx[bi, oi]], eps))
    lh = np.log(np.maximum(d["p_hire"][b, hire_idx], eps))
    return lf, lq, lh


def entropy(d, order_mask):
    """Mean per-factor entropy over emittable orders + the qty and hire heads."""
    eps = 1e-12
    p = np.clip(d["p_fire"], eps, 1 - eps)
    hb = -(p * np.log(p) + (1 - p) * np.log(1 - p)) * (np.asarray(order_mask) > 0)
    hq = -(d["p_qty"] * np.log(np.maximum(d["p_qty"], eps))).sum(2)
    hh = -(d["p_hire"] * np.log(np.maximum(d["p_hire"], eps))).sum(1)
    return hb, hq, hh
