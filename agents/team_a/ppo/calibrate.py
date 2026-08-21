"""exp-011 step 0 -- does the PPO rollout harness agree with the Pit?

This is the riskiest unknown in the whole programme and the spec puts it first (section 3):
the harness runs `kaggle_environments` in-process with both policies as plain callables,
the Pit runs each agent in its own subprocess through a loader that mimics Kaggle's. If the
two disagree, every advantage PPO computes is measured against a game we do not actually
play.

Both sides play the SAME agents on the SAME seeds, side-swapped:
  harness : greedy pi0 vs the frozen 3-member base, in-process
  Pit     : the staged directories of exactly those two, `arena match`

Acceptance (spec section 3): mean bank agrees within the paired SD, win rates within 3pp.
It also answers O3 -- pi0's real ms/turn, which the whole compute plan is scaled from.

    python -m agents.team_a.ppo.calibrate --n 128 --seed-base 0 --workers 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena import ratings, registry            # noqa: E402
from arena.scheduler import run_match          # noqa: E402

from . import harness as H                     # noqa: E402
from . import pool as POOL                     # noqa: E402
from . import stage as S                       # noqa: E402


def harness_side(seeds, workers=1, steps=720):
    """Greedy pi0 vs the frozen base, seat-swapped, in-process. Returns per-episode rows."""
    jobs = [{"seed": s, "seat": seat, "actor": "pi0", "opponent": "base", "steps": steps}
            for s in seeds for seat in (0, 1)]
    return POOL.run(jobs, workers, kind="eval")


def _result(us, them):
    return 1.0 if us > them else (0.0 if us < them else 0.5)


def pit_side(seeds, workers, steps=720, quiet=True):
    pi0_dir, base_dir = S.stage_pi0(), S.stage_base()
    a = registry.register_file(pi0_dir, "TeamA-pi0-m0", "A",
                               notes="exp-011 stand-in actor: BC ensemble member 0 alone")
    b = registry.register_file(base_dir, "TeamA-bc-base-staged", "A",
                               notes="exp-011 reference: shipped 3-member champion, staged")
    seed_spec = f"{min(seeds)}-{max(seeds)}"
    res = run_match([a["id"], b["id"]], mode="screen", n=2 * len(seeds), seeds=seed_spec,
                    workers=workers, steps=steps, quiet=quiet)
    rows = []
    for g in res["rows"]:
        if {g["p0"], g["p1"]} != {a["id"], b["id"]}:
            continue
        seat = 0 if g["p0"] == a["id"] else 1
        us = float(g["money0"] if seat == 0 else g["money1"])
        them = float(g["money1"] if seat == 0 else g["money0"])
        rows.append({"seed": int(g["seed"]), "seat": seat, "us": us, "them": them,
                     "crash": int(g["crash0"] or g["crash1"]),
                     "timeout": int(g["timeout0"] or g["timeout1"]),
                     "p99_ms": float(g["p99_0"] if seat == 0 else g["p99_1"]),
                     "max_ms": float(g["max_0"] if seat == 0 else g["max_1"])})
    return rows, {"a": a["id"], "b": b["id"], "run_id": res["run_id"]}


def mirror_check(seeds, workers, steps=720):
    """Two identical pi0 copies must score exactly 50.0% with banks equal to the unit."""
    from .bcpath import WEIGHTS_DIR
    m = [WEIGHTS_DIR / "rebuild-m0.npz"]
    a = registry.register_file(S.stage("teamA-pi0-A", m), "TeamA-pi0-mirrorA", "A")
    b = registry.register_file(S.stage_mirror(m, "teamA-pi0-B"), "TeamA-pi0-mirrorB", "A")
    if a["id"] == b["id"]:
        raise RuntimeError("mirror twins collapsed to one arena id -- the check would be vacuous")
    res = run_match([a["id"], b["id"]], mode="screen", n=2 * len(seeds),
                    seeds=f"{min(seeds)}-{max(seeds)}", workers=workers, steps=steps, quiet=True)
    rows = [g for g in res["rows"] if {g["p0"], g["p1"]} == {a["id"], b["id"]}]
    score = sum(float(g["result0"]) if g["p0"] == a["id"] else 1.0 - float(g["result0"])
                for g in rows)
    # The sharp check is NOT money0 == money1 inside a game: the two seats are not
    # symmetric (weed spawn is keyed on farm index), so identical policies legitimately
    # bank different amounts in one game. What must hold is that swapping the seats
    # reproduces the same numbers -- seat 0's bank in (A,B) must equal seat 0's bank in
    # (B,A), for the same seed.
    by_seed = {}
    for g in rows:
        by_seed.setdefault(int(g["seed"]), []).append(g)
    swap_equal, swap_pairs = 0, 0
    for seed, gs in by_seed.items():
        if len(gs) != 2:
            continue
        swap_pairs += 1
        g0, g1 = gs
        if (abs(float(g0["money0"]) - float(g1["money0"])) < 1e-9
                and abs(float(g0["money1"]) - float(g1["money1"])) < 1e-9):
            swap_equal += 1
    return {"n": len(rows), "win_rate": round(score / max(1, len(rows)), 4),
            "seat_swap_pairs": swap_pairs, "seat_swap_identical": swap_equal,
            "exactly_50": abs(score / max(1, len(rows)) - 0.5) < 1e-9,
            "ids": [a["id"], b["id"]]}


def summarise(rows, label):
    us = [r["us"] for r in rows]
    them = [r["them"] for r in rows]
    wins = sum(_result(r["us"], r["them"]) for r in rows)
    lo, hi = ratings.wilson(wins, len(rows))
    return {"label": label, "n": len(rows),
            "money_us_mean": round(statistics.fmean(us), 1),
            "money_us_sd": round(statistics.pstdev(us), 1),
            "money_them_mean": round(statistics.fmean(them), 1),
            "extraction_mean": round(statistics.fmean([a + b for a, b in zip(us, them)]), 1),
            "win_rate": round(wins / len(rows), 4), "ci95": [round(lo, 4), round(hi, 4)]}


def compare(h_rows, p_rows):
    """Paired by (seed, seat) -- the only honest comparison, per the org's own SD note."""
    hp = {(r["seed"], r["seat"]): r for r in h_rows}
    pp = {(r["seed"], r["seat"]): r for r in p_rows}
    keys = sorted(set(hp) & set(pp))
    d = np.array([hp[k]["us"] - pp[k]["us"] for k in keys], dtype=float)
    paired_sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    sd_pit = float(np.std([pp[k]["us"] for k in keys], ddof=1))
    return {"n_paired": len(keys),
            "mean_abs_diff": round(float(np.abs(d).mean()), 2),
            "mean_diff": round(float(d.mean()), 2),
            "paired_sd": round(paired_sd, 2),
            "pit_sd": round(sd_pit, 2),
            "identical_episodes": int((np.abs(d) < 1e-6).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128, help="episodes (2 per seed, side-swapped)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mirror-n", type=int, default=16, help="episodes for the mirror self-check")
    args = ap.parse_args(argv)

    seeds = list(range(args.seed_base, args.seed_base + max(1, args.n // 2)))
    mirror = mirror_check(seeds[: max(1, args.mirror_n // 2)], args.workers, args.steps)
    t0 = time.perf_counter()
    p_rows, ids = pit_side(seeds, args.workers, args.steps)
    t_pit = time.perf_counter() - t0
    t0 = time.perf_counter()
    h_rows = harness_side(seeds, workers=args.workers, steps=args.steps)
    t_har = time.perf_counter() - t0

    hs, ps = summarise(h_rows, "harness"), summarise(p_rows, "pit")
    cmp = compare(h_rows, p_rows)
    verdict = {
        "money_within_paired_sd": abs(hs["money_us_mean"] - ps["money_us_mean"]) <= cmp["paired_sd"],
        "win_rate_within_3pp": abs(hs["win_rate"] - ps["win_rate"]) <= 0.03,
        "no_pit_crashes": sum(r.get("crash", 0) for r in p_rows) == 0,
        "no_pit_timeouts": sum(r.get("timeout", 0) for r in p_rows) == 0,
        "mirror_exactly_50": (mirror["exactly_50"]
                              and mirror["seat_swap_identical"] == mirror["seat_swap_pairs"]),
    }
    out = {"agents": ids, "mirror": mirror, "seeds": [seeds[0], seeds[-1]], "episodes": len(p_rows),
           "harness": hs, "pit": ps, "paired": cmp, "verdict": verdict,
           "pass": all(verdict.values()),
           "pit_wall_s": round(t_pit, 1), "harness_wall_s": round(t_har, 1),
           "harness_s_per_episode": round(t_har / max(1, len(h_rows)), 3),
           "pit_p99_ms_max": round(max((r["p99_ms"] for r in p_rows), default=0.0), 2),
           "pit_max_ms": round(max((r["max_ms"] for r in p_rows), default=0.0), 2)}
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
