"""exp-012 -- distil the 3-member BC ensemble into ONE net (`pi0`).

Why this exists (spec section 2.2): PPO needs a single differentiable stochastic policy.
Running PPO through the 3-member probability mixture is *feasible* (log p_mix is exactly
computable) but it triples rollout and update cost and it co-adapts the three members,
destroying the independence that is the entire justification for the ensemble.

What is distilled, precisely. The teacher is the mixture AS IT PLAYS, i.e. exactly what
`policy.BCPolicy` computes: each member's probabilities are averaged (not logits, not
weights), under the same legality masks used at inference. The student is a net of identical
architecture, so the result is a drop-in `Member` -- `BCPolicy([pi0.npz])` is a legal
one-member ensemble and the packaged agent is unchanged apart from the weights file.

Loss: per head, KL(mixture || student) = cross-entropy against the mixture's soft targets
(the mixture's entropy is constant w.r.t. the student, so the gradient is exactly
p_student - p_teacher). No hard labels by default -- `--alpha-hard` enables the spec's
fallback path (train on the union of all three chunks WITH the corpus labels plus the
distillation term) if plain distillation misses the agreement gate.

CONDITIONAL HEAD WEIGHTING. Four of the seven heads are read only conditionally at
inference: `crop` only when the op is PLANT, `item` only on PICKUP/PLACE, unit `qty` only on
PICKUP, and market `qty` only for an order that actually fires. Matching the teacher on rows
where a head is never read spends capacity on outputs no one ever looks at, so each of those
heads is weighted by the teacher's own probability that it gets read. This mirrors the BC
trainer, which supervised those heads only on rows that carried the argument, and it lines
the loss up with how agreement is measured below.

AGREEMENT IS MEASURED THE WAY THE HEAD IS READ. The gate is "the student takes the same
action as the ensemble", so per-head agreement is conditioned the same way (crop scored on
rows where the mixture's op is PLANT, etc.), and the honest composite -- the whole emitted
unit action, and the whole emitted order set -- is reported alongside. The order head is not
an argmax head: it is 20 independent threshold decisions, so its agreement is the fired bit
under the ENSEMBLE-calibrated thresholds, which the student inherits unchanged (matching
probabilities under a shared cutoff is the point; see calibrate_ensemble.py).

Usage:
    uv run python agents/team_a/bc/distil.py \
        --members agents/team_a/bc/weights/rebuild-m0.npz \
                  agents/team_a/bc/weights/rebuild-m1.npz \
                  agents/team_a/bc/weights/rebuild-m2.npz \
        --rows <wh>/bc-rows/rebuild-chunk0.npz <wh>/.../rebuild-chunk1.npz <wh>/.../rebuild-chunk2.npz \
        --out agents/team_a/bc/weights/pi0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from net import MLP, Adam, masked_softmax, sigmoid  # noqa: E402
from policy import Member  # noqa: E402

UNIT_HEADS = {"op": F.N_OPS, "crop": F.N_CROPS, "item": F.N_ITEMS, "qty": F.N_UQTY}
MKT_HEADS = {"orders": F.N_ORDERS, "qty": F.N_ORDERS * F.N_QTY, "hire": F.MAX_HIRES}
UNIT_W = {"op": 1.0, "crop": 0.3, "item": 0.3, "qty": 0.3}
MKT_W = {"orders": 1.0, "qty": 0.5, "hire": 0.5}

IX_PLANT = F.OPS.index("PLANT")
IX_PICKUP = F.OPS.index("PICKUP")
IX_PLACE = F.OPS.index("PLACE")


# ---------------------------------------------------------------- teacher ---
def teacher_unit(members, x, mask, batch=8192):
    """Mixture probabilities for the unit net, exactly as policy.BCPolicy._unit_op forms
    them: masked softmax on `op`, plain softmax on the argument heads, averaged over
    members."""
    n = len(x)
    out = {"op": np.empty((n, F.N_OPS), np.float32),
           "crop": np.empty((n, F.N_CROPS), np.float32),
           "item": np.empty((n, F.N_ITEMS), np.float32),
           "qty": np.empty((n, F.N_UQTY), np.float32)}
    for i in range(0, n, batch):
        xb, mb = x[i:i + batch], mask[i:i + batch]
        acc = None
        for m in members:
            o = m.unit.forward((xb - m.umu) / m.usd)
            p = {"op": masked_softmax(o["op"], mb), "crop": masked_softmax(o["crop"]),
                 "item": masked_softmax(o["item"]), "qty": masked_softmax(o["qty"])}
            acc = p if acc is None else {k: acc[k] + p[k] for k in acc}
        for k in out:
            out[k][i:i + batch] = acc[k] / len(members)
    return out


def teacher_mkt(members, x, batch=8192):
    """Mixture probabilities for the market net, as policy.BCPolicy._market forms them."""
    n = len(x)
    out = {"orders": np.empty((n, F.N_ORDERS), np.float32),
           "qty": np.empty((n, F.N_ORDERS, F.N_QTY), np.float32),
           "hire": np.empty((n, F.MAX_HIRES), np.float32)}
    for i in range(0, n, batch):
        xb = x[i:i + batch]
        acc = None
        for m in members:
            o = m.mkt.forward((xb - m.mmu) / m.msd)
            q = o["qty"].reshape(len(xb), F.N_ORDERS, F.N_QTY)
            p = {"orders": sigmoid(o["orders"]),
                 "qty": masked_softmax(q.reshape(-1, F.N_QTY)).reshape(q.shape),
                 "hire": masked_softmax(o["hire"])}
            acc = p if acc is None else {k: acc[k] + p[k] for k in acc}
        for k in out:
            out[k][i:i + batch] = acc[k] / len(members)
    return out


# ---------------------------------------------------------------- student ---
def student_unit_probs(net, mu, sd, x, mask, batch=8192):
    n = len(x)
    out = {"op": np.empty((n, F.N_OPS), np.float32),
           "crop": np.empty((n, F.N_CROPS), np.float32),
           "item": np.empty((n, F.N_ITEMS), np.float32),
           "qty": np.empty((n, F.N_UQTY), np.float32)}
    for i in range(0, n, batch):
        o = net.forward((x[i:i + batch] - mu) / sd)
        out["op"][i:i + batch] = masked_softmax(o["op"], mask[i:i + batch])
        out["crop"][i:i + batch] = masked_softmax(o["crop"])
        out["item"][i:i + batch] = masked_softmax(o["item"])
        out["qty"][i:i + batch] = masked_softmax(o["qty"])
    return out


def student_mkt_probs(net, mu, sd, x, batch=8192):
    n = len(x)
    out = {"orders": np.empty((n, F.N_ORDERS), np.float32),
           "qty": np.empty((n, F.N_ORDERS, F.N_QTY), np.float32),
           "hire": np.empty((n, F.MAX_HIRES), np.float32)}
    for i in range(0, n, batch):
        o = net.forward((x[i:i + batch] - mu) / sd)
        q = o["qty"].reshape(-1, F.N_ORDERS, F.N_QTY)
        out["orders"][i:i + batch] = sigmoid(o["orders"])
        out["qty"][i:i + batch] = masked_softmax(q.reshape(-1, F.N_QTY)).reshape(q.shape)
        out["hire"][i:i + batch] = masked_softmax(o["hire"])
    return out


# ------------------------------------------------------------------ train ---
def standardise(x):
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def lr_at(args, ep, total):
    """Cosine decay to `--lr-final` x peak. Distillation is pure function approximation --
    the teacher is a deterministic function of x, so there is no label noise to average out
    and the last points of argmax agreement live near the decision boundaries, which a
    constant Adam step size bounces across rather than settling into."""
    if not args.lr_final or total <= 1:
        return args.lr
    f = args.lr_final + (1.0 - args.lr_final) * 0.5 * (1.0 + np.cos(np.pi * ep / (total - 1)))
    return args.lr * float(f)


def sharpen(p, a):
    """Blend the mixture's own argmax one-hot into its soft target.

    The gate is ARGMAX agreement with the mixture, but plain KL optimises the whole
    distribution -- and the mixture is very sharp (the order head measures 0.040 nats), so
    KL is dominated by the top class while the residual disagreement sits on near-ties that
    the loss barely charges for. Mixing in the argmax one-hot spends the remaining capacity
    exactly where the gate is scored. a=0 recovers pure distillation.

    Note this targets the MIXTURE's argmax, not the corpus's action (`--alpha-hard`): the
    student is being asked to reproduce the ensemble, not to re-clone the teachers.
    """
    if a <= 0.0:
        return p
    oh = np.zeros_like(p)
    idx = p.argmax(-1)
    np.put_along_axis(oh, idx[..., None], 1.0, axis=-1)
    return (1.0 - a) * p + a * oh


def soft_ce(p_student, p_teacher):
    """KL(teacher || student) up to the teacher's constant entropy, and its gradient."""
    loss = -(p_teacher * np.log(np.maximum(p_student, 1e-12))).sum(-1)
    return loss, p_student - p_teacher


def _probe_unit(members, net, mu, sd, val, k=40000):
    """Cheap agreement probe on a fixed slice of the frozen val split (telemetry only --
    the gate is always computed on the whole split)."""
    x, m = val["unit_x"][:k], val["unit_mask"][:k]
    t = teacher_unit(members, x, m)
    s = student_unit_probs(net, mu, sd, x, m)
    t_op, s_op = t["op"].argmax(1), s["op"].argmax(1)
    out = {"op": float((t_op == s_op).mean())}
    for h, sel in (("crop", t_op == IX_PLANT), ("item", (t_op == IX_PICKUP) | (t_op == IX_PLACE)),
                   ("qty", t_op == IX_PICKUP)):
        out[h] = float((t[h].argmax(1)[sel] == s[h].argmax(1)[sel]).mean()) if sel.any() else 1.0
    return out


def _probe_mkt(members, net, mu, sd, val, th, k=20000):
    x = val["mkt_x"][:k]
    mm = val["mkt_mask"][:k].astype(bool)
    t, s = teacher_mkt(members, x), student_mkt_probs(net, mu, sd, x)
    tf = (t["orders"] >= th[None, :]) & mm
    sf = (s["orders"] >= th[None, :]) & mm
    nf = int(tf.sum())
    return {"orders": float((tf == sf).mean()),
            "qty": float((t["qty"].argmax(2)[tf] == s["qty"].argmax(2)[tf]).mean()) if nf else 1.0,
            "hire": float((t["hire"].argmax(1) == s["hire"].argmax(1)).mean())}


def train_unit(members, x, mask, hard, args, val=None):
    mu, sd = standardise(x)
    net = MLP(x.shape[1], args.hidden, UNIT_HEADS, seed=args.seed)
    opt = Adam(net.params(), lr=args.lr, wd=args.wd)
    n = len(x)
    rng = np.random.default_rng(args.seed)
    steps = max(1, n // args.batch)
    print(f"[unit] {n:,} rows, dim {x.shape[1]}, {steps} steps/epoch, {args.epochs} epochs",
          flush=True)
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        t0, tot = time.perf_counter(), 0.0
        opt.lr = lr_at(args, ep, args.epochs)
        for s in range(steps):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            xb, mb = x[idx], mask[idx]
            tp = teacher_unit(members, xb, mb, batch=len(idx))
            if args.alpha_mix > 0.0:
                tp = {k: sharpen(v, args.alpha_mix) for k, v in tp.items()}
            out, cache = net.forward((xb - mu) / sd, cache=True)
            b = len(idx)
            g, loss = {}, 0.0
            ps = masked_softmax(out["op"], mb)
            lo, go = soft_ce(ps, tp["op"])
            # the argument heads are read only when the op selects them; weight each row by
            # the teacher's probability that it IS read (see module docstring)
            w = {"crop": tp["op"][:, IX_PLANT],
                 "item": tp["op"][:, IX_PICKUP] + tp["op"][:, IX_PLACE],
                 "qty": tp["op"][:, IX_PICKUP]}
            if args.alpha_hard > 0.0:
                oh = np.zeros_like(ps)
                lab = hard["op"][idx]
                ok = lab >= 0
                oh[np.nonzero(ok)[0], lab[ok]] = 1.0
                go = (1.0 - args.alpha_hard) * go + args.alpha_hard * (ps - oh) * ok[:, None]
            g["op"] = go * (UNIT_W["op"] / b)
            loss += UNIT_W["op"] * float(lo.mean())
            for k in ("crop", "item", "qty"):
                psk = masked_softmax(out[k])
                lk, gk = soft_ce(psk, tp[k])
                wk = w[k].astype(np.float32)
                denom = max(1e-6, float(wk.sum()))
                g[k] = (gk * wk[:, None]) * (UNIT_W[k] / denom)
                loss += UNIT_W[k] * float((lk * wk).sum() / denom)
            opt.step(net.params(), net.backward(cache, g))
            tot += loss
        msg = ""
        if val is not None and args.eval_every and (ep + 1) % args.eval_every == 0:
            p = _probe_unit(members, net, mu, sd, val)
            msg = " | " + " ".join(f"{k}={v:.4f}" for k, v in p.items())
        print(f"  [unit] epoch {ep+1}/{args.epochs} loss={tot/steps:.4f} "
              f"({time.perf_counter()-t0:.0f}s){msg}", flush=True)
    return net, mu, sd


def train_mkt(members, x, args, val=None, th=None):
    mu, sd = standardise(x)
    net = MLP(x.shape[1], args.mkt_hidden, MKT_HEADS, seed=args.seed + 1)
    opt = Adam(net.params(), lr=args.lr, wd=args.wd)
    n = len(x)
    rng = np.random.default_rng(args.seed + 1)
    steps = max(1, n // args.batch)
    print(f"[market] {n:,} rows, dim {x.shape[1]}, {steps} steps/epoch, "
          f"{args.mkt_epochs} epochs", flush=True)
    for ep in range(args.mkt_epochs):
        perm = rng.permutation(n)
        t0, tot = time.perf_counter(), 0.0
        opt.lr = lr_at(args, ep, args.mkt_epochs)
        for s in range(steps):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            xb = x[idx]
            tp = teacher_mkt(members, xb, batch=len(idx))
            if args.alpha_mix > 0.0:
                tp = dict(tp, qty=sharpen(tp["qty"], args.alpha_mix),
                          hire=sharpen(tp["hire"], args.alpha_mix))
            out, cache = net.forward((xb - mu) / sd, cache=True)
            b = len(idx)
            g, loss = {}, 0.0
            # orders: 20 independent Bernoullis matched in probability
            po = sigmoid(out["orders"])
            g["orders"] = ((po - tp["orders"]) * (MKT_W["orders"] / b)).astype(np.float32)
            loss += MKT_W["orders"] * float(np.mean(
                -(tp["orders"] * np.log(np.maximum(po, 1e-12))
                  + (1 - tp["orders"]) * np.log(np.maximum(1 - po, 1e-12))).sum(1)))
            # qty: one softmax per order, weighted by the teacher's probability that the
            # order fires at all (a quantity nobody reads is not worth capacity)
            q = out["qty"].reshape(b, F.N_ORDERS, F.N_QTY)
            pq = masked_softmax(q.reshape(-1, F.N_QTY)).reshape(q.shape)
            lq, gq = soft_ce(pq, tp["qty"])
            wq = tp["orders"].astype(np.float32)
            denom = max(1e-6, float(wq.sum()))
            g["qty"] = ((gq * wq[:, :, None]) * (MKT_W["qty"] / denom)
                        ).reshape(b, -1).astype(np.float32)
            loss += MKT_W["qty"] * float((lq * wq).sum() / denom)
            ph = masked_softmax(out["hire"])
            lh, gh = soft_ce(ph, tp["hire"])
            g["hire"] = gh * (MKT_W["hire"] / b)
            loss += MKT_W["hire"] * float(lh.mean())
            opt.step(net.params(), net.backward(cache, g))
            tot += loss
        msg = ""
        if val is not None and th is not None and args.eval_every \
                and (ep + 1) % (args.eval_every * 5) == 0:
            p = _probe_mkt(members, net, mu, sd, val, th)
            msg = " | " + " ".join(f"{k}={v:.4f}" for k, v in p.items())
        print(f"  [market] epoch {ep+1}/{args.mkt_epochs} loss={tot/steps:.4f} "
              f"({time.perf_counter()-t0:.0f}s){msg}", flush=True)
    return net, mu, sd


# --------------------------------------------------------------- agreement ---
def agreement(members, unet, umu, usd, mnet, mmu, msd, val, thresholds):
    """Argmax/fired agreement between the student and the mixture on the FROZEN shared
    validation split -- the common yardstick. Every head is scored the way it is read."""
    ux, umask = val["unit_x"], val["unit_mask"]
    tu = teacher_unit(members, ux, umask)
    su = student_unit_probs(unet, umu, usd, ux, umask)
    t_op, s_op = tu["op"].argmax(1), su["op"].argmax(1)
    res = {"unit_op": {"n": int(len(t_op)), "agree": float((t_op == s_op).mean())}}
    cond = {"crop": t_op == IX_PLANT,
            "item": (t_op == IX_PICKUP) | (t_op == IX_PLACE),
            "qty": t_op == IX_PICKUP}
    same_arg = np.ones(len(t_op), bool)
    for k, sel in cond.items():
        tk, sk = tu[k].argmax(1), su[k].argmax(1)
        res[f"unit_{k}"] = {"n": int(sel.sum()),
                            "agree": float((tk[sel] == sk[sel]).mean()) if sel.any() else 1.0,
                            "agree_uncond": float((tk == sk).mean())}
        same_arg &= ~sel | (tk == sk)
    # the honest composite: the whole emitted unit action (op AND the argument it selects).
    # Note the inference-time ctx masks (crop_mask/item_mask_*) are not in the corpus, so
    # this is the pre-ctx-mask action; a ctx mask can only make two policies agree MORE.
    res["unit_action"] = {"n": int(len(t_op)),
                          "agree": float(((t_op == s_op) & same_arg).mean())}

    mx = val["mkt_x"]
    mmask = val["mkt_mask"].astype(bool)
    tm = teacher_mkt(members, mx)
    sm = student_mkt_probs(mnet, mmu, msd, mx)
    th = thresholds[None, :]
    t_fire = (tm["orders"] >= th) & mmask
    s_fire = (sm["orders"] >= th) & mmask
    res["mkt_orders_bitwise"] = {"n": int(t_fire.size), "agree": float((t_fire == s_fire).mean())}
    res["mkt_orders_setexact"] = {"n": int(len(mx)),
                                  "agree": float((t_fire == s_fire).all(1).mean())}
    res["mkt_orders_fired"] = {"teacher": int(t_fire.sum()), "student": int(s_fire.sum())}
    tq, sq = tm["qty"].argmax(2), sm["qty"].argmax(2)
    n_f = int(t_fire.sum())
    res["mkt_qty_on_fired"] = {"n": n_f,
                               "agree": float((tq[t_fire] == sq[t_fire]).mean()) if n_f else 1.0}
    t_h, s_h = tm["hire"].argmax(1), sm["hire"].argmax(1)
    res["mkt_hire"] = {"n": int(len(t_h)), "agree": float((t_h == s_h).mean())}
    res["mkt_turn_exact"] = {
        "n": int(len(mx)),
        "agree": float(((t_fire == s_fire).all(1)
                        & (t_h == s_h)
                        & ((tq == sq) | ~t_fire).all(1)).mean())}
    return res


# ------------------------------------------------------------------- main ---
def load_rows(paths):
    """Union of the training splits + the ONE shared validation split (identical in every
    chunk by construction -- extract_rebuild.py writes the same val rows into each)."""
    tr_u, tr_um, tr_m, hard_op = [], [], [], []
    val = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        tr_u.append(d["train_unit_x"])
        tr_um.append(d["train_unit_mask"].astype(np.float32))
        tr_m.append(d["train_mkt_x"])
        hard_op.append(d["train_unit_op"].astype(np.int32))
        v = {"unit_x": d["val_unit_x"], "unit_mask": d["val_unit_mask"].astype(np.float32),
             "mkt_x": d["val_mkt_x"], "mkt_mask": d["val_mkt_mask"].astype(np.float32)}
        if val is None:
            val = v
        else:
            if v["unit_x"].shape != val["unit_x"].shape or \
                    not np.array_equal(v["unit_x"][:64], val["unit_x"][:64]):
                raise SystemExit(f"{p}: validation split differs -- not a shared yardstick")
    return (np.concatenate(tr_u), np.concatenate(tr_um), np.concatenate(tr_m),
            {"op": np.concatenate(hard_op)}, val)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--mkt-hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--mkt-epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=707)
    ap.add_argument("--alpha-hard", type=float, default=0.0,
                    help="spec fallback: blend the corpus hard labels into the op head")
    ap.add_argument("--alpha-mix", type=float, default=0.0,
                    help="blend the MIXTURE's argmax one-hot into its soft target (0-1); "
                         "spends capacity where the argmax-agreement gate is scored")
    ap.add_argument("--lr-final", type=float, default=0.05,
                    help="cosine-decay the LR to this fraction of peak (0 = constant)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="agreement probe on a val slice every N unit epochs (0 = off)")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    members = [Member(p) for p in args.members]
    ref = members[0]
    th = ref.ens_thresholds if ref.ens_thresholds is not None else ref.thresholds
    if ref.ens_thresholds is None:
        print("WARNING: no m_ens_thresholds on member 0 -- falling back to per-member "
              "thresholds; run calibrate_ensemble.py first", flush=True)
    print(f"teacher: {len(members)} members, hidden={ref.unit.hidden}/{ref.mkt.hidden}",
          flush=True)

    ux, umask, mx, hard, val = load_rows(args.rows)
    print(f"union: unit {len(ux):,} rows, market {len(mx):,} rows | "
          f"val (frozen yardstick): unit {len(val['unit_x']):,}, mkt {len(val['mkt_x']):,}",
          flush=True)

    unet, umu, usd = train_unit(members, ux, umask, hard, args, val=val)
    mnet, mmu, msd = train_mkt(members, mx, args, val=val, th=th)

    agree = agreement(members, unet, umu, usd, mnet, mmu, msd, val, th)
    print("\n=== agreement with the mixture on the frozen shared validation split ===")
    for k, v in agree.items():
        if "agree" in v:
            print(f"  {k:24s} n={v['n']:>9,} agree={v['agree']:.4f}"
                  + (f"  (uncond {v['agree_uncond']:.4f})" if "agree_uncond" in v else ""))
    gate_heads = ["unit_op", "unit_crop", "unit_item", "unit_qty",
                  "mkt_orders_bitwise", "mkt_qty_on_fired", "mkt_hire"]
    worst = min(agree[k]["agree"] for k in gate_heads)
    gate = worst >= 0.99
    print(f"\nGATE (>=99% per head): {'PASS' if gate else 'FAIL'} (worst head {worst:.4f})")

    out = {}
    for k, v in unet.state().items():
        out["u_" + k] = v
    for k, v in mnet.state().items():
        out["m_" + k] = v
    out["u_mu"], out["u_sd"] = umu, usd
    out["m_mu"], out["m_sd"] = mmu, msd
    # the student inherits the ENSEMBLE-calibrated thresholds and slot order unchanged: it is
    # trained to reproduce the mixture's probabilities, so the mixture's cutoffs are the ones
    # that reproduce the mixture's decisions.
    out["m_thresholds"] = np.asarray(th, np.float32)
    out["m_ens_thresholds"] = np.asarray(th, np.float32)
    out["slot_order"] = np.asarray(ref.slot_order, np.float32)
    card = {
        "kind": "distilled", "experiment": "exp-012",
        "teacher_members": [str(p) for p in args.members],
        "teacher_cards": [m.card.get("teacher", "?") for m in members],
        "rows": [str(p) for p in args.rows],
        "unit_dim": int(ux.shape[1]), "market_dim": int(mx.shape[1]),
        "hidden": args.hidden, "mkt_hidden": args.mkt_hidden,
        "epochs": args.epochs, "mkt_epochs": args.mkt_epochs, "batch": args.batch,
        "lr": args.lr, "wd": args.wd, "seed": args.seed, "alpha_hard": args.alpha_hard, "alpha_mix": args.alpha_mix,
        "rows_train_unit": int(len(ux)), "rows_train_mkt": int(len(mx)),
        "rows_val_unit": int(len(val["unit_x"])), "rows_val_mkt": int(len(val["mkt_x"])),
        "agreement": agree, "gate_pass": bool(gate), "worst_head": round(float(worst), 4),
        "engine": members[0].card.get("engine", "?"),
        "train_wall_s": round(time.perf_counter() - t_start, 1),
    }
    out["card"] = np.asarray([json.dumps(card)])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({path.stat().st_size/1e6:.2f} MB) in {card['train_wall_s']}s")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
