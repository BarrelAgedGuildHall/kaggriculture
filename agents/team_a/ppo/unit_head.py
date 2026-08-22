"""The unit net as a *stochastic policy*, plus its exact log-prob recompute (phase P1).

The market twin of this module is `market_head.py`; the same two obligations apply and the
same reasoning as its docstring, with one extra wrinkle that is worth stating because it is
where a silent mismatch would live.

**Greedy mode must be `BCPolicy._unit_op`, decision for decision.** The shipped policy takes
`argmax` of a *masked* softmax on the op head, and for the argument heads it takes
`argmax(p * mask)` of an **unmasked** softmax. Those two are the same decision --
`argmax(softmax(z) * m) == argmax(masked_softmax(z, m))` whenever `m` is 0/1 and not all
zero -- so the stochastic policy uses the masked softmax throughout and its greedy mode
reproduces the BC decision exactly. `tests/test_ppo_unit_head.py` asserts it on real
observations rather than taking my word for it.

**The factor set is conditional.** A unit row always has an `op` factor. It has a `crop`
factor only if the sampled op is PLANT, an `item` factor only for PICKUP/PLACE, and a `qty`
factor only for PICKUP. A factor that does not exist carries index -1 and is excluded from
every term of the loss -- it is not a zero-probability factor, it is no factor at all.

**Temperature.** `p` is at the sampling temperature `T`; `z` (the raw, mask-applied logits)
is returned as well so the KL anchor and the BC-auxiliary term can be evaluated at T = 1.
That matters: the anchor is the frozen 3-member mixture, which has no temperature knob, so
comparing a T = 2 policy against it would make the anchor a *sharpening* force fighting the
exploration schedule rather than a drift control (spec addendum 01 section 6.3(i)).
"""
from __future__ import annotations

import numpy as np

from .bcpath import F

N_OPS = F.N_OPS
N_CROPS = F.N_CROPS
N_ITEMS = F.N_ITEMS
N_UQTY = F.N_UQTY

IX_PLANT = F.OPS.index("PLANT")
IX_PICKUP = F.OPS.index("PICKUP")
IX_PLACE = F.OPS.index("PLACE")

HEADS = ("crop", "item", "op", "qty")      # sorted(), to match MLP.params()
NEG = -1e9
EPS = 1e-12


def _masked_softmax(z, mask, temperature=1.0):
    """Masked softmax at temperature `T`, in float64 (log-probs reach ~-30)."""
    z = np.asarray(z, dtype=np.float64)
    if mask is not None:
        z = np.where(np.asarray(mask) > 0, z, NEG)
    z = z / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), EPS)


def dists(unit, umu, usd, x, op_mask, crop_mask, item_mask, temperature=1.0, cache=False):
    """Batched forward -> the four factor distributions.

    x           (B, D) raw unit features
    op_mask     (B, N_OPS)   1 = legal op
    crop_mask   (B, N_CROPS) 1 = a seed of that crop is held
    item_mask   (B, N_ITEMS) 1 = that item is available for the op in question

    Returns p_* at temperature `T` and z_* = the masked logits at T = 1.
    """
    xn = (np.asarray(x, dtype=np.float32) - umu) / usd
    if cache:
        out, cch = unit.forward(xn, cache=True)
    else:
        out, cch = unit.forward(xn), None
    def m(z, mask):
        z = z.astype(np.float64)
        return z if mask is None else np.where(np.asarray(mask) > 0, z, NEG)

    z_op = m(out["op"], op_mask)
    z_crop = m(out["crop"], crop_mask)
    z_item = m(out["item"], item_mask)
    z_qty = out["qty"].astype(np.float64)
    return {
        "p_op": _masked_softmax(z_op, None, temperature),
        "p_crop": _masked_softmax(z_crop, None, temperature),
        "p_item": _masked_softmax(z_item, None, temperature),
        "p_qty": _masked_softmax(z_qty, None, temperature),
        "z_op": z_op, "z_crop": z_crop, "z_item": z_item, "z_qty": z_qty,
        "cache": cch,
    }


def unit_probs_at_one(d):
    """The same distributions at T = 1, from the cached logits. Used by the KL anchor."""
    return {"p_op": _masked_softmax(d["z_op"], None, 1.0),
            "p_crop": _masked_softmax(d["z_crop"], None, 1.0),
            "p_item": _masked_softmax(d["z_item"], None, 1.0),
            "p_qty": _masked_softmax(d["z_qty"], None, 1.0)}


def logprobs(d, op_idx, crop_idx, item_idx, qty_idx):
    """Per-factor log-probabilities of the recorded actions; index < 0 => no factor (0.0)."""
    b = np.arange(len(op_idx))
    def pick(p, idx):
        out = np.zeros(len(idx), dtype=np.float64)
        ok = idx >= 0
        if ok.any():
            j = np.nonzero(ok)[0]
            out[j] = np.log(np.maximum(p[j, idx[j]], EPS))
        return out
    return (pick(d["p_op"], op_idx), pick(d["p_crop"], crop_idx),
            pick(d["p_item"], item_idx), pick(d["p_qty"], qty_idx))


def entropy(d):
    def h(p):
        return -(p * np.log(np.maximum(p, EPS))).sum(-1)
    return h(d["p_op"]), h(d["p_crop"]), h(d["p_item"]), h(d["p_qty"])


def mixture_dists(members, x, op_mask, crop_mask, item_mask):
    """The frozen 3-member mixture's unit distributions on the same rows, at T = 1.

    Formed exactly as `BCPolicy._unit_op` forms them -- each member's masked softmax over the
    SAME legal set, then the probability vectors averaged (never logits, never weights) -- and
    then renormalised over the argument masks, because the shipped policy takes
    `argmax(p_mix * mask)`. That renormalisation is what makes this the mixture's actual
    decision distribution rather than a distribution over illegal arguments.
    """
    acc = None
    for m in members:
        o = m.unit.forward((np.asarray(x, dtype=np.float32) - m.umu) / m.usd)
        p = {"p_op": _masked_softmax(o["op"], op_mask),
             "p_crop": _masked_softmax(o["crop"], None),
             "p_item": _masked_softmax(o["item"], None),
             "p_qty": _masked_softmax(o["qty"], None)}
        acc = p if acc is None else {k: acc[k] + p[k] for k in acc}
    k = float(len(members))
    out = {kk: v / k for kk, v in acc.items()}
    for key, mask in (("p_crop", crop_mask), ("p_item", item_mask)):
        if mask is None:
            continue
        w = out[key] * (np.asarray(mask) > 0)
        s = w.sum(-1, keepdims=True)
        out[key] = np.where(s > EPS, w / np.maximum(s, EPS), out[key])
    return out
