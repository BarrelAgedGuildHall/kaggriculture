"""exp-011 acceptance screen: a PPO checkpoint vs pi0, through the Pit.

The pre-registered bar (spec section 8) is a *money* bar with a *paired* interval:

    mean final money up by >= 3% on a 128-episode paired-seed held-out screen,
    the paired-difference CI excluding zero, 0 crashes.

Paired matters more than usual here. The unpaired bank SD across seeds is ~$20k; the SD of
the difference between two same-family agents on the same seed is $2-5k. Quoting the
unpaired interval would hide a real effect; quoting the paired one is the only way a 1-hour
pilot can answer the question at all.

The pairing is exact: the checkpoint and pi0 each play the same seed in the same seat against
the same reference opponent, so the difference is measured on identical boards.

Reported alongside, always (spec section 6.2, the metric triple): mirror extraction, money
and bank retention vs the reference opponents, and the win rate -- so a money gain that is
really the cooperative attractor (R1) is visible the day it appears.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena import ratings, registry              # noqa: E402
from arena.scheduler import run_match            # noqa: E402

from . import stage as S                         # noqa: E402
from .bcpath import WEIGHTS_DIR                  # noqa: E402


def _rows(res, me, opp):
    out = []
    for g in res["rows"]:
        if {g["p0"], g["p1"]} != {me, opp}:
            continue
        seat = 0 if g["p0"] == me else 1
        out.append({"seed": int(g["seed"]), "seat": seat,
                    "us": float(g["money0"] if seat == 0 else g["money1"]),
                    "them": float(g["money1"] if seat == 0 else g["money0"]),
                    "crash": int(g["crash0"] or g["crash1"]),
                    "timeout": int(g["timeout0"] or g["timeout1"]),
                    "p99_ms": float(g["p99_0"] if seat == 0 else g["p99_1"]),
                    "max_ms": float(g["max_0"] if seat == 0 else g["max_1"])})
    return out


def _summary(rows):
    us = [r["us"] for r in rows]
    wins = sum(1.0 if r["us"] > r["them"] else (0.5 if r["us"] == r["them"] else 0.0)
               for r in rows)
    lo, hi = ratings.wilson(wins, max(1, len(rows)))
    return {"n": len(rows), "money_mean": round(statistics.fmean(us), 1),
            "money_sd": round(statistics.pstdev(us), 1),
            "opp_money_mean": round(statistics.fmean([r["them"] for r in rows]), 1),
            "extraction_mean": round(statistics.fmean([r["us"] + r["them"] for r in rows]), 1),
            "win_rate": round(wins / max(1, len(rows)), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "crashes": sum(r["crash"] for r in rows), "timeouts": sum(r["timeout"] for r in rows),
            "p99_ms_max": round(max((r["p99_ms"] for r in rows), default=0), 2),
            "max_ms": round(max((r["max_ms"] for r in rows), default=0), 2)}


def paired_delta(cand_rows, ref_rows, boot=20000, seed=0):
    """Paired mean difference with a bootstrap CI over (seed, seat) pairs."""
    c = {(r["seed"], r["seat"]): r["us"] for r in cand_rows}
    b = {(r["seed"], r["seat"]): r["us"] for r in ref_rows}
    keys = sorted(set(c) & set(b))
    d = np.array([c[k] - b[k] for k in keys], dtype=float)
    if len(d) < 2:
        return {"n_pairs": len(d)}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    ref_mean = statistics.fmean([b[k] for k in keys])
    return {"n_pairs": len(d), "mean_diff": round(float(d.mean()), 1),
            "paired_sd": round(float(d.std(ddof=1)), 1),
            "ci95": [round(float(lo), 1), round(float(hi), 1)],
            "pct_change": round(100.0 * float(d.mean()) / ref_mean, 3),
            "pct_ci95": [round(100.0 * float(lo) / ref_mean, 3),
                         round(100.0 * float(hi) / ref_mean, 3)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "wins_pairwise": int((d > 0).sum()), "losses_pairwise": int((d < 0).sum())}


def _register_dir(path, name, team="B", notes=""):
    return registry.register_file(Path(path), name, team, notes=notes)


def run(checkpoint, n=128, seed_base=1000, workers=14, steps=720, name="TeamA-ppo-exp011",
        mirror=True, melonrace_dir=None):
    seeds = list(range(seed_base, seed_base + max(1, n // 2)))
    spec = f"{seeds[0]}-{seeds[-1]}"
    m0 = WEIGHTS_DIR / "rebuild-m0.npz"

    cand = registry.register_file(S.stage("teamA-ppo-cand", [checkpoint]), name, "A",
                                  notes=f"exp-011 P0 checkpoint {Path(checkpoint).name}")
    pi0 = registry.register_file(S.stage("teamA-pi0", [m0]), "TeamA-pi0-m0", "A",
                                 notes="exp-011 reference actor: BC ensemble member 0")
    base = registry.register_file(S.stage("teamA-bc-base", sorted(WEIGHTS_DIR.glob("*.npz"))),
                                  "TeamA-bc-base-staged", "A", notes="shipped 3-member champion")
    if cand["id"] == pi0["id"]:
        raise RuntimeError("the checkpoint is byte-identical to pi0 -- nothing was trained")

    out = {"checkpoint": str(checkpoint), "ids": {"cand": cand["id"], "pi0": pi0["id"],
                                                  "base": base["id"]},
           "seeds": [seeds[0], seeds[-1]], "episodes_per_matchup": 2 * len(seeds)}

    # head-to-head: the candidate against the reference actor it was trained from
    h2h = run_match([cand["id"], pi0["id"]], mode="confirm" if seed_base >= 1000 else "screen",
                    n=2 * len(seeds), seeds=spec, workers=workers, steps=steps, quiet=True)
    out["head_to_head_vs_pi0"] = _summary(_rows(h2h, cand["id"], pi0["id"]))

    # the paired money comparison: both agents vs the SAME reference opponent, same boards
    cand_vs_base = _rows(run_match([cand["id"], base["id"]], mode="confirm", n=2 * len(seeds),
                                   seeds=spec, workers=workers, steps=steps, quiet=True),
                         cand["id"], base["id"])
    pi0_vs_base = _rows(run_match([pi0["id"], base["id"]], mode="confirm", n=2 * len(seeds),
                                  seeds=spec, workers=workers, steps=steps, quiet=True),
                        pi0["id"], base["id"])
    out["cand_vs_base"] = _summary(cand_vs_base)
    out["pi0_vs_base"] = _summary(pi0_vs_base)
    out["paired_money_delta_vs_base"] = paired_delta(cand_vs_base, pi0_vs_base)

    # the same pairing read as a margin: on the boards where the two met directly, did we
    # out-earn pi0? (money vs a common opponent is the acceptance number; this corroborates.)
    h2h_rows = _rows(h2h, cand["id"], pi0["id"])
    out["paired_margin_vs_pi0"] = paired_delta(
        [{"seed": r["seed"], "seat": r["seat"], "us": r["us"]} for r in h2h_rows],
        [{"seed": r["seed"], "seat": r["seat"], "us": r["them"]} for r in h2h_rows])

    # Champion[B] melonrace: the contested-market stressor. Bank retention = our money
    # against it divided by our money against the A-lineage base (spec section 6.2); it is what
    # catches a policy that gains money by cooperating rather than by farming better.
    if melonrace_dir and Path(melonrace_dir).exists():
        mel = _register_dir(melonrace_dir, "TeamB-Millrace-melonrace", "B",
                            notes="Champion[B], staged for the exp-011 metric triple")
        cm = _rows(run_match([cand["id"], mel["id"]], mode="confirm", n=2 * len(seeds),
                             seeds=spec, workers=workers, steps=steps, quiet=True),
                   cand["id"], mel["id"])
        pm = _rows(run_match([pi0["id"], mel["id"]], mode="confirm", n=2 * len(seeds),
                             seeds=spec, workers=workers, steps=steps, quiet=True),
                   pi0["id"], mel["id"])
        out["cand_vs_melonrace"] = _summary(cm)
        out["pi0_vs_melonrace"] = _summary(pm)
        out["paired_money_delta_vs_melonrace"] = paired_delta(cm, pm)
        out["bank_retention"] = {
            "cand": round(out["cand_vs_melonrace"]["money_mean"]
                          / max(1e-9, out["cand_vs_base"]["money_mean"]), 4),
            "pi0": round(out["pi0_vs_melonrace"]["money_mean"]
                         / max(1e-9, out["pi0_vs_base"]["money_mean"]), 4)}

    if mirror:
        twin = registry.register_file(S.stage_mirror([checkpoint], "teamA-ppo-cand-mirror"),
                                      name + "-mirror", "A")
        mr = run_match([cand["id"], twin["id"]], mode="screen", n=16, seeds="0-7",
                       workers=min(workers, 8), steps=steps, quiet=True)
        rows = _rows(mr, cand["id"], twin["id"])
        score = sum(1.0 if r["us"] > r["them"] else (0.5 if r["us"] == r["them"] else 0.0)
                    for r in rows)
        by_seed = {}
        for g in mr["rows"]:
            by_seed.setdefault(int(g["seed"]), []).append(g)
        swap_ok = sum(1 for gs in by_seed.values() if len(gs) == 2
                      and abs(float(gs[0]["money0"]) - float(gs[1]["money0"])) < 1e-9
                      and abs(float(gs[0]["money1"]) - float(gs[1]["money1"])) < 1e-9)
        out["mirror"] = {"n": len(rows), "win_rate": round(score / max(1, len(rows)), 4),
                         "seat_swap_pairs": len(by_seed), "seat_swap_identical": swap_ok,
                         "extraction_mean": round(statistics.fmean(
                             [r["us"] + r["them"] for r in rows]), 1) if rows else None}

    d = out["paired_money_delta_vs_base"]
    out["acceptance"] = {
        "money_up_3pct": d.get("pct_change", 0.0) >= 3.0,
        "paired_ci_excludes_zero": bool(d.get("ci_excludes_zero", False)),
        "no_crashes": out["cand_vs_base"]["crashes"] == 0 and out["head_to_head_vs_pi0"]["crashes"] == 0,
        "no_timeouts": out["cand_vs_base"]["timeouts"] == 0,
    }
    out["pass"] = all(out["acceptance"].values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed-base", type=int, default=1000, help="1000 = held out from training")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--name", default="TeamA-ppo-exp011")
    ap.add_argument("--out", default=None)
    ap.add_argument("--melonrace-dir", default=None,
                    help="path to a TeamB-Millrace-melonrace agent snapshot (metric triple)")
    args = ap.parse_args(argv)
    res = run(args.checkpoint, n=args.n, seed_base=args.seed_base, workers=args.workers,
              steps=args.steps, name=args.name, melonrace_dir=args.melonrace_dir)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0 if res["pass"] else 1
