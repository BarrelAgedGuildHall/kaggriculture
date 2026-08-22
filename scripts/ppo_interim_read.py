"""Interim read on a live PPO run: the GREEDY checkpoint's paired money vs a reference.

This is **not** a Pit gate and is never quoted as one. It exists because the exp-013 tripwire
is a *money* statistic, and exp-011 step 0 established that this harness and the Pit produce
**byte-identical banks on 128/128 episodes** (mean_abs_diff 0.0) -- so a paired money number
measured here is the number the Pit would return. What the harness cannot measure is timing
(C2), because the training run is using most of the box; anything timing-related still has to
go through a quiesced Pit with the ladder paused.

The statistic is the one the addendum's Gate D uses: on the SAME boards, our money minus the
opponent's, as a percentage of the opponent's, with a bootstrap CI over (seed, seat) pairs.

    uv run python scripts/ppo_interim_read.py --checkpoint <ckpt.npz> --opponent base \\
        --seeds 1128-1191 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo import pool as POOL     # noqa: E402


def paired(rows, boot=20000, seed=0):
    d = np.array([r["us"] - r["them"] for r in rows], dtype=float)
    them = np.array([r["them"] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    ref = float(them.mean())
    return {"n_pairs": len(d), "mean_diff": round(float(d.mean()), 1),
            "paired_sd": round(float(d.std(ddof=1)), 1),
            "pct_change": round(100.0 * float(d.mean()) / ref, 3),
            "pct_ci95": [round(100.0 * float(lo) / ref, 3),
                         round(100.0 * float(hi) / ref, 3)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "our_money_mean": round(float(np.mean([r["us"] for r in rows])), 1),
            "opp_money_mean": round(ref, 1),
            "extraction_mean": round(float(np.mean([r["us"] + r["them"] for r in rows])), 1),
            "win_rate": round(float(np.mean([(r["us"] > r["them"]) + 0.5 * (r["us"] == r["them"])
                                             for r in rows])), 4)}


def play(actor, opponent, seeds, workers, steps=720):
    jobs = [{"seed": s, "seat": seat, "actor": actor, "opponent": opponent,
             "opponent_id": str(opponent), "steps": steps}
            for s in seeds for seat in (0, 1)]
    return POOL.run(jobs, workers, kind="eval")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", default=None, help="pi0 weights, for the improvement read")
    ap.add_argument("--opponent", default="base")
    ap.add_argument("--seeds", default="1128-1191")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    out = {"checkpoint": args.checkpoint, "opponent": args.opponent,
           "seeds": [seeds[0], seeds[-1]], "episodes": 2 * len(seeds),
           "note": "harness read, calibration-backed (exp-011 step 0); NOT a Pit gate"}

    cand = play([args.checkpoint], args.opponent, seeds, args.workers, args.steps)
    out["candidate_vs_opponent"] = paired(cand)
    if args.reference:
        ref = play([args.reference], args.opponent, seeds, args.workers, args.steps)
        out["reference_vs_opponent"] = paired(ref)
        c = {(r["seed"], r["seat"]): r["us"] for r in cand}
        p = {(r["seed"], r["seat"]): r["us"] for r in ref}
        keys = sorted(set(c) & set(p))
        d = np.array([c[k] - p[k] for k in keys], dtype=float)
        rng = np.random.default_rng(1)
        means = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        base = float(np.mean([p[k] for k in keys]))
        out["candidate_minus_reference"] = {
            "n_pairs": len(d), "mean_diff": round(float(d.mean()), 1),
            "pct_change": round(100.0 * float(d.mean()) / base, 3),
            "pct_ci95": [round(100.0 * float(lo) / base, 3),
                         round(100.0 * float(hi) / base, 3)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    # the pre-registered exp-013 tripwire (addendum 01 section 4.3)
    g = out["candidate_vs_opponent"]
    out["tripwire_parity_with_frozen_base"] = {
        "statistic_pct": g["pct_change"], "ci95": g["pct_ci95"],
        "reaches_parity": bool(g["pct_change"] >= 0.0 or g["pct_ci95"][1] >= 0.0)}
    print(json.dumps(out, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
