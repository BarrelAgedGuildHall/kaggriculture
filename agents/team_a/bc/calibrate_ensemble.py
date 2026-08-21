"""Re-tune the market order thresholds on the ENSEMBLE's averaged probabilities.

Why this exists. Each member's per-order thresholds are tuned (in `train.py`) against that
member's *own* sigmoid outputs. But `policy.BCPolicy._market` fires orders from the mean of
all members' probabilities, and averaging pulls extremes toward the middle -- so a threshold
calibrated on one member's sharper distribution is systematically wrong for the mixture. With
three members the observed per-order spread reached 0.625, which is far too large to ignore
for orders like BUY_ANIMAL that start the entire animal economy.

This is a calibration fix, not a method change: jun177's ensemble takes a plain argmax over a
mixture and has no thresholds to mis-set, so it has no counterpart to get wrong.

The tuned vector is written back into member 0's .npz as `m_ens_thresholds`, which
`policy.Member` prefers over the per-member `m_thresholds` when present. It is tuned on the
shared held-out validation episodes -- the same rows every member was selected on, and rows no
member trained on.

    uv run python agents/team_a/bc/calibrate_ensemble.py \
        --rows "$GUILDHALL_WAREHOUSE/datasets/kaggriculture/bc-rows/rebuild-chunk0.npz"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import policy as P  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, help="any rebuild chunk npz (val_* rows are shared)")
    p.add_argument("--weights", default="")
    p.add_argument("--report", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    pol = P.load_default(args.weights or None)
    members = pol.members
    print(f"{len(members)} members")

    d = np.load(args.rows, allow_pickle=True)
    xva, yva = d["val_mkt_x"], d["val_mkt_multi"].astype(bool)
    print(f"{len(xva):,} held-out market turns")

    # Exactly the mixture policy._market uses.
    probs = None
    for m in members:
        pm = P.sigmoid(m.mkt.forward((xva - m.mmu) / m.msd)["orders"])
        probs = pm if probs is None else probs + pm
    probs /= len(members)

    old = members[0].thresholds
    thresholds = np.full(F.N_ORDERS, 0.5, np.float32)
    report = {}
    for oi in range(F.N_ORDERS):
        y = yva[:, oi]
        best = (0.0, 0.5, 0.0, 0.0)
        for th in np.arange(0.05, 0.96, 0.025):
            pred = probs[:, oi] >= th
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
                        "th_ensemble": round(best[1], 3), "th_member0": round(float(old[oi]), 3),
                        "prec": round(best[2], 3), "rec": round(best[3], 3)}
        if int(y.sum()):
            print(f"  {name:28} n={int(y.sum()):6d} f1={best[0]:.3f} "
                  f"th {float(old[oi]):.3f} -> {best[1]:.3f}")

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.dry_run:
        print("dry run; nothing written")
        return 0

    target = Path(members[0].path)
    payload = dict(np.load(target, allow_pickle=True))
    payload["m_ens_thresholds"] = thresholds
    np.savez_compressed(target, **payload)
    print(f"wrote m_ens_thresholds into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
