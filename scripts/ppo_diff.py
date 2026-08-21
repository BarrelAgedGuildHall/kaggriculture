"""How much did the greedy policy actually move? (exp-011 sanity check before the screen.)

The order head is extremely sharp -- mean per-factor entropy 0.040 nats at temperature 1 --
so a fine-tune can shift logits measurably while flipping *no* greedy decision at all. In
that case the acceptance screen would return an exact zero and we would waste an hour of Pit
time learning nothing. This tells us in ~20 seconds instead: replay recorded states through
both policies and count greedy decision changes per head.

    uv run python scripts/ppo_diff.py --checkpoint ...npz --episodes 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo import market_head as MH        # noqa: E402
from agents.team_a.ppo import harness as H             # noqa: E402
from agents.team_a.ppo.bcpath import P, WEIGHTS_DIR    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    m0 = str(WEIGHTS_DIR / "rebuild-m0.npz")
    ref, cand = P.Member(m0), P.Member(args.checkpoint)
    th = ref.ens_thresholds if ref.ens_thresholds is not None else ref.thresholds
    thl = MH.logit(np.asarray(th, dtype=np.float64))

    xs, masks = [], []
    for i in range(args.episodes):
        turns, _, _ = H.rollout(9000 + i, i % 2, [m0], "pi0", 1234 + i,
                                temperature=1.0, steps=args.steps)
        xs.append(np.stack([t["x"] for t in turns]))
        masks.append(np.stack([t["order_mask"] for t in turns]))
    x, mask = np.concatenate(xs), np.concatenate(masks)

    a = MH.dists(ref.mkt, ref.mmu, ref.msd, x, mask, thl, 1.0)
    b = MH.dists(cand.mkt, ref.mmu, ref.msd, x, mask, thl, 1.0)
    em = mask > 0
    fire_a, fire_b = (a["p_fire"] >= 0.5) & em, (b["p_fire"] >= 0.5) & em
    qa, qb = a["p_qty"].argmax(2), b["p_qty"].argmax(2)
    ha, hb = a["p_hire"].argmax(1), b["p_hire"].argmax(1)
    fired = fire_a | fire_b

    out = {
        "checkpoint": args.checkpoint, "rows": int(len(x)),
        "order_factors": int(em.sum()),
        "order_flips": int((fire_a != fire_b).sum()),
        "order_flip_rate": round(float((fire_a != fire_b).sum()) / max(1, em.sum()), 5),
        "orders_fired_ref": int(fire_a.sum()), "orders_fired_cand": int(fire_b.sum()),
        "qty_changes_on_fired": int(((qa != qb) & fired).sum()),
        "hire_changes": int((ha != hb).sum()),
        "mean_hire_ref": round(float(ha.mean()), 4), "mean_hire_cand": round(float(hb.mean()), 4),
        "max_abs_dp_fire": round(float(np.abs(a["p_fire"] - b["p_fire"])[em].max()), 5),
        "mean_abs_dp_fire": round(float(np.abs(a["p_fire"] - b["p_fire"])[em].mean()), 6),
    }
    per_order = []
    for oi, (t, item) in enumerate(__import__("agents.team_a.ppo.bcpath", fromlist=["F"]).F.ORDER_TYPES):
        n = int(em[:, oi].sum())
        if not n:
            continue
        per_order.append({"order": f"{t}:{item}", "n": n,
                          "fired_ref": int(fire_a[:, oi].sum()),
                          "fired_cand": int(fire_b[:, oi].sum()),
                          "flips": int((fire_a[:, oi] != fire_b[:, oi]).sum())})
    out["per_order"] = sorted(per_order, key=lambda r: -r["flips"])[:10]
    out["policy_moved"] = bool(out["order_flips"] or out["qty_changes_on_fired"]
                               or out["hire_changes"])
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if out["policy_moved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
