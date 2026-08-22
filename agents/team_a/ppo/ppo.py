"""The PPO update for the market head (phase P0), in numpy.

Three things here are decisions, not implementation detail.

**Per-factor ratios.** A turn emits ~20 order Bernoullis + the qty softmax of each fired
order + one hire softmax. A joint ratio `exp(sum of ~35 log-prob deltas)` saturates the clip
on essentially every sample and carries no usable gradient, so the surrogate is the factored
form: one ratio and one clip per factor, all sharing the turn's advantage (spec section 3.4).

**pi_old is recomputed, not replayed.** The rollout's recorded log-probs are the *invariant
check* (`test_ppo_rollout.py`); the ratio itself divides two log-probs computed in the same
batched float32 code path, from a frozen copy of the pre-update parameters. Dividing a
batch-B number by a batch-1 number would put ~5e-6 of float32 noise into every ratio for no
reason.

**Overridden factors carry no gradient.** A factor the safety layer rewrote did not
influence the outcome; leaving it in gives it advantage x grad-log-pi on an action that never
happened -- noise, aimed at the heads the wrapper is protecting. Excluded from the surrogate,
the entropy term and the KL term alike (spec section 3.3).

Gradients are analytic w.r.t. the logits and handed to the existing `MLP.backward`; in P0
only the three head matrices move, so the trunk backward is skipped entirely.
"""
from __future__ import annotations

import numpy as np

from . import market_head as MH
from . import unit_head as UH
from .bcpath import N

EPS = 1e-12


def clone_mlp(mlp):
    return type(mlp).from_state({k: np.array(v, copy=True) for k, v in mlp.state().items()},
                                dict(mlp.head_dims))


def head_params(mlp, heads=("hire", "orders", "qty")):
    ps = []
    for k in heads:
        ps += mlp.heads[k]
    return ps


def _head_grads(h2, glogits, heads):
    """dL/dW, dL/db for the listed heads only (the trunk is frozen in P0)."""
    out = []
    for k in heads:
        g = glogits[k]
        out += [h2.T @ g, g.sum(0)]
    return out


class Critic:
    """Train-only value net on the market-turn features. Never exported, so unconstrained."""

    def __init__(self, in_dim, hidden=256, lr=1e-3, seed=0):
        self.net = N.MLP(in_dim, hidden, {"v": 1}, seed=seed)
        self.opt = N.Adam(self.net.params(), lr=lr)
        self.mu = np.zeros(in_dim, np.float32)
        self.sd = np.ones(in_dim, np.float32)

    def fit_norm(self, x):
        self.mu = x.mean(0).astype(np.float32)
        self.sd = np.maximum(x.std(0), 1e-3).astype(np.float32)

    def value(self, x):
        return self.net.forward((x - self.mu) / self.sd)["v"][:, 0]

    def update(self, x, target):
        xn = (x - self.mu) / self.sd
        out, cache = self.net.forward(xn, cache=True)
        pred = out["v"][:, 0]
        diff = pred - target
        loss = float((diff ** 2).mean())
        g = (2.0 * diff / len(target)).astype(np.float32)[:, None]
        grads = self.net.backward(cache, {"v": g})
        self.opt.step(self.net.params(), grads)
        return loss, pred


class MarketPPO:
    """Phase P0: the market net's `orders` / `qty` / `hire` heads. Everything else frozen."""

    HEADS = ("hire", "orders", "qty")     # sorted() order, to match MLP.params()

    def __init__(self, mkt, mmu, msd, th_logit, lr=3e-5, wd=1e-5, clip=0.1,
                 entropy_coef=1e-3, kl_target=0.02, kl_beta=1.0, temperature=1.0,
                 train_trunk=False):
        self.mkt = mkt
        self.mmu, self.msd, self.th_logit = mmu, msd, th_logit
        self.anchor = clone_mlp(mkt)       # the frozen BC head (P0 fallback anchor)
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.kl_target = kl_target
        self.beta = kl_beta
        self.temperature = temperature
        self.train_trunk = bool(train_trunk)   # P1: the market trunk joins the parameters
        self.opt = N.Adam(self.params(), lr=lr, wd=wd)

    def params(self):
        return self.mkt.params() if self.train_trunk else head_params(self.mkt, self.HEADS)

    def _grads(self, cache, glogits):
        if self.train_trunk:
            return self.mkt.backward(cache, glogits)
        return _head_grads(cache[4], glogits, self.HEADS)

    # -- distributions -------------------------------------------------------
    def dists(self, mkt, x, mask, cache=False):
        return MH.dists(mkt, self.mmu, self.msd, x, mask, self.th_logit,
                        self.temperature, cache=cache)

    def old_logprobs(self, batch, frozen):
        d = self.dists(frozen, batch["x"], batch["order_mask"])
        return MH.logprobs(d, batch["fire"], batch["qty_idx"], batch["hire_idx"])

    # -- one minibatch -------------------------------------------------------
    def step(self, b, adv, logp_old, anchor_d=None, aux=None, alpha=0.0):
        """Compute the gradient and take one Adam step. See `loss_grads` for the objective."""
        grads, stats = self.loss_grads(b, adv, logp_old, anchor_d)
        if aux is not None and alpha:
            g2, s2 = self.aux_grads(aux, alpha)
            grads = g2 if grads is None else [a + b_ for a, b_ in zip(grads, g2)]
            stats.update(s2)
        if grads is not None:
            self.opt.step(self.params(), grads)
        return stats

    def aux_grads(self, aux, alpha):
        """BC-auxiliary: soft cross-entropy to the FROZEN MIXTURE on corpus states.

        Not to `pi0` (addendum 01 section 2.1) and not to the corpus's hard labels: the
        anchor we want to hold is the ensemble's own distribution, and `exp-012` measured
        that the hard-label target and the mixture's argmax are different targets.
        """
        d = self.dists(self.mkt, aux["x"], aux["order_mask"], cache=True)
        cch = d.pop("cache")
        n = float(len(aux["x"]))
        t = aux["teacher"]
        p = np.clip(d["p_fire1"], EPS, 1 - EPS)
        q = np.clip(t["p_fire1"], EPS, 1 - EPS)
        m = (np.asarray(aux["order_mask"]) > 0)
        loss = float((-(q * np.log(p) + (1 - q) * np.log(1 - p)) * m).sum() / max(1.0, m.sum()))
        gz_fire = alpha * (p - q) * m / n
        gz_qty = alpha * 0.5 * (d["p_qty1"] - t["p_qty1"]) / n
        gz_hire = alpha * 0.5 * (d["p_hire1"] - t["p_hire1"]) / n
        dt = cch[4].dtype
        glogits = {"orders": gz_fire.astype(dt),
                   "qty": gz_qty.reshape(len(aux["x"]), -1).astype(dt),
                   "hire": gz_hire.astype(dt)}
        return self._grads(cch, glogits), {"aux_mkt": loss}

    def loss_grads(self, b, adv, logp_old, anchor_d=None):
        """Analytic gradients of the full objective w.r.t. the three head matrices.

        `stats["loss"]` is the scalar the gradients belong to, so
        `tests/test_ppo_update.py` can finite-difference it. An unverified hand-written
        gradient is exactly the kind of bug that shows up as "money did not move".
        """
        mask_f = (b["order_mask"] > 0) & (b["exec_fire"] > 0)
        mask_q = (b["qty_idx"] >= 0) & (b["exec_qty"] > 0)
        mask_h = (b["exec_hire"] > 0)
        n_factors = float(mask_f.sum() + mask_q.sum() + mask_h.sum())
        if n_factors == 0:
            return None, {"skipped": 1}

        d = self.dists(self.mkt, b["x"], b["order_mask"], cache=True)
        cch = d.pop("cache")
        h2 = cch[4]
        lf, lq, lh = MH.logprobs(d, b["fire"], b["qty_idx"], b["hire_idx"])
        T = self.temperature

        gz_fire = np.zeros_like(d["p_fire"])
        gz_qty = np.zeros_like(d["p_qty"])
        gz_hire = np.zeros_like(d["p_hire"])
        stats = {}

        # ---- clipped surrogate, per factor --------------------------------
        def surrogate(logp_new, logp_o, a):
            r = np.exp(np.clip(logp_new - logp_o, -20.0, 20.0))
            u = r * a
            c = np.clip(r, 1 - self.clip, 1 + self.clip) * a
            take = (u <= c)                       # min(): grad flows through the clip only inside
            loss = -np.where(take, u, c)
            # dL/dr, zero wherever the clip binds
            return loss, (-a * take), r

        a_f = adv[:, None]
        loss_f, dr_f, r_f = surrogate(lf, logp_old[0], a_f)
        loss_q, dr_q, r_q = surrogate(lq, logp_old[1], a_f)
        loss_h, dr_h, r_h = surrogate(lh, logp_old[2], adv)

        # dlogp/dz: Bernoulli (a - p)/T ; categorical (onehot - p)/T
        gz_fire += (dr_f * r_f * (b["fire"] - d["p_fire"]) / T) * mask_f / n_factors
        gz_hire += _cat_grad(d["p_hire"], b["hire_idx"],
                             (dr_h * r_h / T) * mask_h / n_factors)
        gz_qty += _cat_grad_2d(d["p_qty"], b["qty_idx"],
                               (dr_q * r_q / T) * mask_q / n_factors)

        stats["surrogate"] = float((loss_f * mask_f).sum() + (loss_q * mask_q).sum()
                                   + (loss_h * mask_h).sum()) / n_factors
        total = stats["surrogate"]
        stats["clipfrac"] = float(((np.abs(r_f - 1) > self.clip) * mask_f).sum()
                                  + ((np.abs(r_q - 1) > self.clip) * mask_q).sum()) \
            / max(1.0, float(mask_f.sum() + mask_q.sum()))

        # ---- entropy bonus -------------------------------------------------
        if self.entropy_coef:
            p = np.clip(d["p_fire"], EPS, 1 - EPS)
            s = np.log(p / (1 - p))
            # loss term is -c*H, so dL/dz = +c * s * p(1-p) / T
            gz_fire += self.entropy_coef * (s * p * (1 - p) / T) * mask_f / n_factors
            gz_hire -= self.entropy_coef * _ent_grad(d["p_hire"]) * mask_h[:, None] / n_factors / T
            gz_qty -= self.entropy_coef * _ent_grad(d["p_qty"]) * mask_q[..., None] / n_factors / T
            hb, hq, hh = MH.entropy(d, b["order_mask"])
            stats["entropy_orders"] = float((hb * (b["order_mask"] > 0)).sum()
                                            / max(1, (b["order_mask"] > 0).sum()))
            stats["entropy_hire"] = float(hh.mean())
            total -= self.entropy_coef * float((hb * mask_f).sum() + (hq * mask_q).sum()
                                               + (hh * mask_h).sum()) / n_factors

        # ---- adaptive KL anchor to the FROZEN 3-MEMBER MIXTURE --------------
        # Evaluated at T = 1 on BOTH sides (addendum 01 section 2.1 + 6.3(i)): the anchor is
        # the mixture, which has no temperature knob, so measuring a T = 2 policy against it
        # would turn the anchor into a sharpening force fighting the exploration schedule
        # instead of a drift control. At T = 1 the term is exactly "how far has the deployed
        # (greedy-relevant) distribution moved from the BC prior", which is what kappa means.
        if self.beta and anchor_d is not None:
            q = np.clip(anchor_d["p_fire1"], EPS, 1 - EPS)
            p = np.clip(d["p_fire1"], EPS, 1 - EPS)
            kl_b = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
            dkl_dp = np.log(p / q) - np.log((1 - p) / (1 - q))
            gz_fire += self.beta * (dkl_dp * p * (1 - p)) * mask_f / n_factors
            kl_h, gh = _cat_kl(d["p_hire1"], anchor_d["p_hire1"])
            gz_hire += self.beta * gh * mask_h[:, None] / n_factors
            kl_q, gq = _cat_kl(d["p_qty1"], anchor_d["p_qty1"])
            gz_qty += self.beta * gq * mask_q[..., None] / n_factors
            stats["kl"] = float((kl_b * mask_f).sum() + (kl_q * mask_q).sum()
                                + (kl_h * mask_h).sum()) / n_factors
            total += self.beta * stats["kl"]

        # match the parameter dtype (float32 in production; float64 in the gradient test,
        # where a float32 round-trip would swamp the finite differences)
        dt = h2.dtype
        glogits = {"orders": gz_fire.astype(dt),
                   "qty": gz_qty.reshape(len(b["x"]), -1).astype(dt),
                   "hire": gz_hire.astype(dt)}
        grads = self._grads(cch, glogits)
        stats["n_factors"] = n_factors
        stats["loss"] = total
        return grads, stats

    def adapt_beta(self, kl):
        """Adaptive penalty: PPO's clip bounds the change per update, not cumulative drift."""
        if kl > 1.5 * self.kl_target:
            self.beta = min(self.beta * 2.0, 64.0)
        elif kl < 0.5 * self.kl_target:
            self.beta = max(self.beta / 2.0, 1e-3)
        return self.beta


class UnitPPO:
    """Phase P1: the unit net's four action heads. The unit TRUNK stays frozen.

    Why the trunk is frozen here while the market trunk is not: the market net is where
    `exp-011` already moved money, so its representation has earned the right to move; the
    unit trunk is the shared representation that all three ensemble members' behaviour was
    distilled through, and moving it is `P2` (`exp-014`) with its own row. This keeps
    `exp-013` a one-variable step over `exp-011` -- action policy, not representation.
    """

    HEADS = ("crop", "item", "op", "qty")     # sorted(), to match MLP.params()

    def __init__(self, unit, umu, usd, lr=3e-5, wd=1e-5, clip=0.1, entropy_coef=1e-3,
                 kl_beta=1.0, temperature=1.0):
        self.unit = unit
        self.umu, self.usd = umu, usd
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.beta = kl_beta
        self.temperature = temperature
        self.opt = N.Adam(head_params(self.unit, self.HEADS), lr=lr, wd=wd)

    def dists(self, unit, b, cache=False):
        return UH.dists(unit, self.umu, self.usd, b["x"], b["op_mask"], b["crop_mask"],
                        b["item_mask"], self.temperature, cache=cache)

    def old_logprobs(self, b, frozen):
        d = self.dists(frozen, b)
        return UH.logprobs(d, b["op_idx"], b["crop_idx"], b["item_idx"], b["qty_idx"])

    def step(self, b, adv, logp_old, anchor_d=None, aux=None, alpha=0.0):
        grads, stats = self.loss_grads(b, adv, logp_old, anchor_d)
        if aux is not None and alpha:
            g2, s2 = self.aux_grads(aux, alpha)
            grads = g2 if grads is None else [a + c for a, c in zip(grads, g2)]
            stats.update(s2)
        if grads is not None:
            self.opt.step(head_params(self.unit, self.HEADS), grads)
        return stats

    def loss_grads(self, b, adv, logp_old, anchor_d=None):
        """Same objective as `MarketPPO.loss_grads`, over the four conditional unit factors.

        `stats["loss"]` is the scalar the returned gradients belong to, so
        `tests/test_ppo_unit_update.py` can finite-difference every head against it.
        """
        masks = {
            "op": (b["op_idx"] >= 0) & (b["exec_op"] > 0),
            "crop": (b["crop_idx"] >= 0) & (b["exec_crop"] > 0),
            "item": (b["item_idx"] >= 0) & (b["exec_item"] > 0),
            "qty": (b["qty_idx"] >= 0) & (b["exec_qty"] > 0),
        }
        n_factors = float(sum(m.sum() for m in masks.values()))
        if n_factors == 0:
            return None, {"skipped": 1}

        d = self.dists(self.unit, b, cache=True)
        cch = d.pop("cache")
        h2 = cch[4]
        lp = dict(zip(("op", "crop", "item", "qty"),
                      UH.logprobs(d, b["op_idx"], b["crop_idx"], b["item_idx"], b["qty_idx"])))
        old = dict(zip(("op", "crop", "item", "qty"), logp_old))
        idx = {"op": b["op_idx"], "crop": b["crop_idx"], "item": b["item_idx"],
               "qty": b["qty_idx"]}
        T = self.temperature
        gz = {k: np.zeros_like(d["p_" + k]) for k in masks}
        stats, total, clipped, n_clip = {}, 0.0, 0.0, 0.0

        for k, m in masks.items():
            p = d["p_" + k]
            r = np.exp(np.clip(lp[k] - old[k], -20.0, 20.0))
            u, c = r * adv, np.clip(r, 1 - self.clip, 1 + self.clip) * adv
            take = (u <= c)
            total += float((-np.where(take, u, c) * m).sum()) / n_factors
            coef = (-adv * take) * r / T * m / n_factors
            gz[k] += _cat_grad(p, np.maximum(idx[k], 0), coef) * m[:, None]
            clipped += float(((np.abs(r - 1) > self.clip) * m).sum())
            n_clip += float(m.sum())

        stats["surrogate"] = total
        stats["clipfrac"] = clipped / max(1.0, n_clip)

        if self.entropy_coef:
            hs = dict(zip(("op", "crop", "item", "qty"), UH.entropy(d)))
            for k, m in masks.items():
                gz[k] -= self.entropy_coef * _ent_grad(d["p_" + k]) * m[:, None] / n_factors / T
                total -= self.entropy_coef * float((hs[k] * m).sum()) / n_factors
            stats["entropy_op"] = float((hs["op"] * masks["op"]).sum()
                                        / max(1.0, masks["op"].sum()))

        if self.beta and anchor_d is not None:
            d1 = UH.unit_probs_at_one(d)
            kl_total = 0.0
            for k, m in masks.items():
                kl, g = _cat_kl(d1["p_" + k], anchor_d["p_" + k])
                gz[k] += self.beta * g * m[:, None] / n_factors     # KL at T = 1, no 1/T
                kl_total += float((kl * m).sum())
            stats["kl"] = kl_total / n_factors
            total += self.beta * stats["kl"]

        dt = h2.dtype
        glogits = {k: gz[k].astype(dt) for k in masks}
        stats["n_factors"] = n_factors
        stats["loss"] = total
        return _head_grads(h2, glogits, self.HEADS), stats

    def aux_grads(self, aux, alpha):
        """BC-auxiliary: soft cross-entropy to the frozen mixture on corpus states."""
        d = UH.dists(self.unit, self.umu, self.usd, aux["x"], aux["op_mask"],
                     None, None, 1.0, cache=True)
        cch = d.pop("cache")
        n = float(len(aux["x"]))
        t = aux["teacher"]
        w = {"op": 1.0, "crop": 0.3, "item": 0.3, "qty": 0.3}
        glogits, loss = {}, 0.0
        for k, wk in w.items():
            p, q = d["p_" + k], t["p_" + k]
            glogits[k] = (alpha * wk * (p - q) / n).astype(cch[4].dtype)
            if k == "op":
                loss = float((-(q * np.log(np.maximum(p, EPS))).sum(1)).mean())
        return _head_grads(cch[4], glogits, self.HEADS), {"aux_unit": loss}

    def adapt_beta(self, kl, kl_target):
        if kl > 1.5 * kl_target:
            self.beta = min(self.beta * 2.0, 64.0)
        elif kl < 0.5 * kl_target:
            self.beta = max(self.beta / 2.0, 1e-3)
        return self.beta


def _cat_grad(p, idx, coef):
    """coef * (onehot(idx) - p) for a (B, K) categorical."""
    g = -p * coef[:, None]
    g[np.arange(len(idx)), idx] += coef
    return g


def _cat_grad_2d(p, idx, coef):
    """Same for (B, N_ORDERS, K); rows with idx < 0 contribute nothing."""
    g = -p * coef[..., None]
    bi, oi = np.nonzero(idx >= 0)
    g[bi, oi, idx[bi, oi]] += coef[bi, oi]
    g[idx < 0] = 0.0
    return g


def _ent_grad(p):
    """dH/dz for a categorical, up to the temperature factor: -p (log p + H)."""
    lp = np.log(np.maximum(p, EPS))
    h = -(p * lp).sum(-1, keepdims=True)
    return -p * (lp + h)


def _cat_kl(p, q):
    lp = np.log(np.maximum(p, EPS))
    lq = np.log(np.maximum(q, EPS))
    kl = (p * (lp - lq)).sum(-1)
    g = p * ((lp - lq) - kl[..., None])
    return kl, g
