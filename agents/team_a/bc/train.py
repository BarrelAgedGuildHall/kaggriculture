"""Train one behaviour-cloned policy (one teacher) from extracted rows.

Two networks, six heads:
  unit net    x(UNIT_DIM)   -> op(18) | crop(5) | item(12)
  market net  x(MARKET_DIM) -> type(7) | item(12) | qty(18)

Validation is by held-out *episodes* (the split was made in extract.py), and the metric
reported per head is top-1 accuracy plus macro (class-balanced) accuracy, because the op
head is dominated by PASS and the four moves and a plain accuracy would hide a policy that
has learned nothing but "walk".

Offline accuracy is a FILTER, not a promotion criterion. The gate is head-to-head play in
scripts/evaluate.py.

Usage:
    uv run python agents/team_a/bc/train.py --rows <npz> --out agents/team_a/bc/weights/tetsuya.npz
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
from net import MLP, Adam, bce_grad, ce_grad, masked_softmax, sigmoid  # noqa: E402

UNIT_HEADS = {"op": F.N_OPS, "crop": F.N_CROPS, "item": F.N_ITEMS, "qty": F.N_UQTY}
MKT_HEADS = {"orders": F.N_ORDERS, "qty": F.N_ORDERS * F.N_QTY, "hire": F.MAX_HIRES}
# crop/item are only supervised on the minority of rows that carry an argument, so they get
# a smaller share of the gradient than the op head.
UNIT_W = {"op": 1.0, "crop": 0.3, "item": 0.3, "qty": 0.3}
MKT_W = {"orders": 1.0, "qty": 0.5, "hire": 0.5}


def standardise(x):
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def accuracy(pred, labels, n_classes):
    valid = labels >= 0
    if not valid.any():
        return 0.0, 0.0, 0
    p, y = pred[valid], labels[valid]
    acc = float((p == y).mean())
    per = []
    for c in range(n_classes):
        sel = y == c
        if sel.sum() >= 5:
            per.append(float((p[sel] == c).mean()))
    return acc, (float(np.mean(per)) if per else 0.0), int(valid.sum())


def evaluate(net, x, labels, masks, heads, mu, sd, batch=8192):
    xs = (x - mu) / sd
    preds = {k: np.empty(len(x), np.int16) for k in heads}
    for i in range(0, len(xs), batch):
        out = net.forward(xs[i:i + batch])
        for k in heads:
            m = masks.get(k)
            logits = out[k]
            if m is not None:
                logits = np.where(m[i:i + batch] > 0, logits, -1e9)
            preds[k][i:i + batch] = logits.argmax(1)
    return {k: accuracy(preds[k], labels[k], heads[k]) for k in heads}


def train_net(name, xtr, ytr, mtr, xva, yva, mva, heads, weights, args, seed):
    mu, sd = standardise(xtr)
    xs = (xtr - mu) / sd
    net = MLP(xtr.shape[1], args.hidden, heads, seed=seed)
    opt = Adam(net.params(), lr=args.lr, wd=args.wd)
    n = len(xs)
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, n // args.batch)
    print(f"[{name}] {n} train rows, dim {xtr.shape[1]}, {steps_per_epoch} steps/epoch")
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        t0 = time.perf_counter()
        tot, cnt = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            xb = xs[idx]
            out, cache = net.forward(xb, cache=True)
            grads_out, loss_sum, nvalid = {}, 0.0, 0
            for k in heads:
                mb = mtr[k][idx] if k in mtr else None
                ls, nk, g = ce_grad(out[k], ytr[k][idx], mb)
                if nk:
                    grads_out[k] = g * (weights[k] / len(idx))
                    loss_sum += weights[k] * ls / nk
                    nvalid += 1
            grads = net.backward(cache, grads_out)
            opt.step(net.params(), grads)
            tot += loss_sum
            cnt += 1
        va = evaluate(net, xva, yva, mva, heads, mu, sd) if len(xva) else {}
        msg = " ".join(f"{k}={va[k][0]:.3f}/{va[k][1]:.3f}" for k in heads) if va else ""
        print(f"  [{name}] epoch {ep+1}/{args.epochs} loss={tot/max(1,cnt):.4f} "
              f"val(top1/macro) {msg}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return net, mu, sd



def train_market(xtr, mtr, qtr, htr, xva, mva, qva, hva, args, seed):
    """Per-turn market head: one sigmoid per concrete order type (multi-label), a quantity
    softmax per order type, and a count head for HIRE.

    The rare orders are the decisive ones -- BUY_ANIMAL is 0.7% of turns and starts the
    entire animal economy -- so each class gets its own decision boundary (pos_weight in
    the loss, then a per-class threshold tuned on validation F1) instead of competing in
    one softmax it can never win.
    """
    mu, sd = standardise(xtr)
    xs = (xtr - mu) / sd
    net = MLP(xtr.shape[1], args.mkt_hidden, MKT_HEADS, seed=seed)
    opt = Adam(net.params(), lr=args.lr, wd=args.wd)
    n = len(xs)
    pos = mtr.sum(0).astype(np.float64)
    pos_weight = np.clip(np.sqrt((n - pos) / np.maximum(pos, 1.0)), 1.0, 20.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    steps = max(1, n // args.batch)
    print(f"[market] {n} turns, dim {xtr.shape[1]}, {steps} steps/epoch, "
          f"positives per order: {pos.astype(int).tolist()}")
    for ep in range(args.mkt_epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for s_ in range(steps):
            idx = perm[s_ * args.batch:(s_ + 1) * args.batch]
            out, cache = net.forward(xs[idx], cache=True)
            g = {}
            lo, go = bce_grad(out["orders"], mtr[idx].astype(np.float32), pos_weight)
            g["orders"] = go * MKT_W["orders"]
            # quantity: one softmax per order type, supervised only where that order fired
            ql = out["qty"].reshape(len(idx), F.N_ORDERS, F.N_QTY)
            qlab = qtr[idx]
            gq = np.zeros_like(ql)
            lq, nq = 0.0, 0
            for oi in range(F.N_ORDERS):
                sel = qlab[:, oi] >= 0
                if not sel.any():
                    continue
                l_, n_, gg = ce_grad(ql[sel, oi, :], qlab[sel, oi].astype(np.int32))
                gq[sel, oi, :] = gg
                lq += l_
                nq += n_
            g["qty"] = (gq.reshape(len(idx), -1) * (MKT_W["qty"] / len(idx))).astype(np.float32)
            lh, nh_, gh = ce_grad(out["hire"], htr[idx].astype(np.int32))
            g["hire"] = gh * (MKT_W["hire"] / len(idx))
            opt.step(net.params(), net.backward(cache, g))
            tot += lo + (lq / max(1, nq)) + (lh / max(1, nh_))
        print(f"  [market] epoch {ep+1}/{args.mkt_epochs} loss={tot/steps:.4f}", flush=True)

    # per-class threshold, tuned on validation F1 (0.5 is useless for a 0.7% class)
    pv = sigmoid(net.forward((xva - mu) / sd)["orders"]) if len(xva) else np.zeros((0, F.N_ORDERS))
    thresholds = np.full(F.N_ORDERS, 0.5, np.float32)
    report = {}
    for oi in range(F.N_ORDERS):
        if not len(pv):
            break
        y = mva[:, oi].astype(bool)
        best = (0.0, 0.5, 0.0, 0.0)
        for th in np.arange(0.05, 0.96, 0.025):
            pred = pv[:, oi] >= th
            tp = int((pred & y).sum())
            if tp == 0:
                continue
            prec = tp / max(1, int(pred.sum()))
            rec = tp / max(1, int(y.sum()))
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            if f1 > best[0]:
                best = (f1, float(th), prec, rec)
        thresholds[oi] = best[1]
        name = f"{F.ORDER_TYPES[oi][0]}:{F.ORDER_TYPES[oi][1]}"
        report[name] = {"n_val": int(y.sum()), "f1": round(best[0], 3),
                        "th": round(best[1], 3), "prec": round(best[2], 3),
                        "rec": round(best[3], 3)}
    hire_acc = 0.0
    if len(xva):
        ph = net.forward((xva - mu) / sd)["hire"].argmax(1)
        hire_acc = float((ph == hva).mean())
    return net, mu, sd, thresholds, {"orders": report, "hire_top1": round(hire_acc, 4)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--mkt-hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--mkt-epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    d = np.load(args.rows, allow_pickle=True)
    meta = json.loads(str(d["meta"][0]))
    print(f"teacher={meta['teacher']} engine={meta['engine']} "
          f"train_eps={len(meta['episodes_train'])} val_eps={len(meta['episodes_val'])}")

    ux, uxv = d["train_unit_x"], d["val_unit_x"]
    uy = {"op": d["train_unit_op"].astype(np.int32),
          "crop": d["train_unit_crop"].astype(np.int32),
          "item": d["train_unit_item"].astype(np.int32),
          "qty": d["train_unit_qty"].astype(np.int32)}
    uyv = {"op": d["val_unit_op"].astype(np.int32),
           "crop": d["val_unit_crop"].astype(np.int32),
           "item": d["val_unit_item"].astype(np.int32),
           "qty": d["val_unit_qty"].astype(np.int32)}
    um = {"op": d["train_unit_mask"].astype(np.float32)}
    umv = {"op": d["val_unit_mask"].astype(np.float32)}

    mx, mxv = d["train_mkt_x"], d["val_mkt_x"]

    t0 = time.perf_counter()
    unet, umu, usd = train_net("unit", ux, uy, um, uxv, uyv, umv,
                               UNIT_HEADS, UNIT_W, args, args.seed)
    mnet, mmu, msd, thresholds, mreport = train_market(
        mx, d["train_mkt_multi"], d["train_mkt_qty"], d["train_mkt_hire"].astype(np.int32),
        mxv, d["val_mkt_multi"], d["val_mkt_qty"], d["val_mkt_hire"].astype(np.int32),
        args, args.seed + 1)

    uva = evaluate(unet, uxv, uyv, umv, UNIT_HEADS, umu, usd)

    out = {}
    for k, v in unet.state().items():
        out["u_" + k] = v
    for k, v in mnet.state().items():
        out["m_" + k] = v
    out["u_mu"], out["u_sd"] = umu, usd
    out["m_mu"], out["m_sd"] = mmu, msd
    out["m_thresholds"] = thresholds
    out["slot_order"] = np.asarray(meta["slot_order"] + [meta["hire_slot"]], np.float32)
    card = {
        "teacher": meta["teacher"], "engine": meta["engine"],
        "unit_dim": int(ux.shape[1]), "market_dim": int(mx.shape[1]),
        "hidden": args.hidden, "mkt_hidden": args.mkt_hidden,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "wd": args.wd,
        "seed": args.seed,
        "rows_train_unit": int(len(ux)), "rows_val_unit": int(len(uxv)),
        "rows_train_mkt": int(len(mx)), "rows_val_mkt": int(len(mxv)),
        "episodes_train": meta["episodes_train"], "episodes_val": meta["episodes_val"],
        "extract_stats": meta["stats"],
        "val_unit": {k: {"top1": round(v[0], 4), "macro": round(v[1], 4), "n": v[2]}
                     for k, v in uva.items()},
        "val_market": mreport,
        "train_wall_s": round(time.perf_counter() - t0, 1),
    }
    out["card"] = np.asarray([json.dumps(card)])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print("\n" + json.dumps({k: card[k] for k in ("val_unit", "val_market")}, indent=1))
    print(f"wrote {path} ({path.stat().st_size/1e6:.2f} MB) in {card['train_wall_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
