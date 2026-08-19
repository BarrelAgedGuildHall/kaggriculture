"""Kaggriculture evaluation gauntlet: paired seeds, side swap, Wilson CI, timing gates.

This is the harness the Modeler's score comes from (strategy.md section 3). Rules it enforces:

* **Score = win rate**, never bank margin -- the ladder only counts win / loss / tie.
  Reported as wins/N with a **Wilson 95% CI** (ties count as half a win).
* **Paired seeds with side swap.** Every seed is played twice: once with our agent as
  player 0 and once as player 1. Seat matters a little (player 0 resolves HIRE/BUY_LAND
  first on a tied turn) and pairing removes most of the seed variance, which is dominated
  by the town's shop draw.
* **Hard gates.** Any ERROR/TIMEOUT/INVALID status, any agent exception, or a p99 turn time
  over --p99-ms (default 150) marks the run FAILED -- the score is not to be quoted.
* Seed hygiene: screen on --seed-base 0 (seeds 0..199), confirm on held-out 1000..1199.

Usage (from the repo root):
    uv run python scripts/evaluate.py --n 64 --tag exp-001-screen
    uv run python scripts/evaluate.py --n 400 --seed-base 1000 --tag exp-001-confirm
    uv run python scripts/evaluate.py --agent agent/main.py --opponents random starter \
        artisans/.../v0.py --n 64 --workers 8

`--opponents` takes built-in names (`random`, `starter`, `pass`) and/or paths to agent .py
files, so a past generation is added to the gauntlet by pointing at its frozen file.
Results land in data/results/<tag>/ as results.json (summary) and episodes.csv (one row
per episode). Nothing here submits anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import DATA_DIR, ENV_NAME, REPO_ROOT, engine_version, import_engine  # noqa: E402

BUILTINS = ("random", "starter", "pass")
BAD_STATUSES = ("ERROR", "TIMEOUT", "INVALID")

_ENGINE = None


def _init_worker():
    global _ENGINE
    _ENGINE = import_engine()


def _resolve(spec: str) -> str:
    if spec in BUILTINS:
        return spec
    p = Path(spec)
    if not p.is_absolute():
        p = REPO_ROOT / spec
    if not p.exists():
        raise SystemExit(f"agent not found: {spec}")
    return str(p.resolve())


def _play(job):
    """One episode. `job` = (agent_path, opponent_spec, seed, our_seat, steps)."""
    global _ENGINE
    if _ENGINE is None:
        _init_worker()
    agent, opp, seed, seat, steps = job
    order = [agent, opp] if seat == 0 else [opp, agent]
    t0 = time.perf_counter()
    err = ""
    try:
        env = _ENGINE.make(ENV_NAME, configuration={"episodeSteps": steps, "seed": seed}, debug=False)
        env.run(order)
        final = env.steps[-1]
        money = [float(s.observation.farms[i]["money"]) for i, s in enumerate(final)]
        statuses = [str(s.status) for s in final]
        durs = [lg[seat]["duration"] for lg in env.logs
                if len(lg) > seat and isinstance(lg[seat], dict) and "duration" in lg[seat]]
        # The framework records agent stderr/exceptions in the same log entries.
        traces = [lg[seat].get("stderr", "") for lg in env.logs
                  if len(lg) > seat and isinstance(lg[seat], dict)]
        err = next((t for t in traces if t and "Traceback" in t), "")
        n_steps = len(env.steps)
    except Exception as exc:  # harness-level failure -- still a void run
        return {
            "seed": seed, "seat": seat, "opponent": opp, "ok": False, "error": repr(exc)[:400],
            "us": 0.0, "them": 0.0, "result": 0.0, "steps": 0,
            "status_us": "HARNESS_ERROR", "status_them": "HARNESS_ERROR",
            "mean_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "wall_s": time.perf_counter() - t0,
        }

    us, them = money[seat], money[1 - seat]
    result = 1.0 if us > them else (0.0 if us < them else 0.5)
    ds = sorted(durs)
    return {
        "seed": seed, "seat": seat, "opponent": opp,
        "ok": (statuses[seat] not in BAD_STATUSES) and n_steps == steps and not err,
        "error": err[-400:],
        "us": us, "them": them, "result": result, "steps": n_steps,
        "status_us": statuses[seat], "status_them": statuses[1 - seat],
        "mean_ms": round(1000 * statistics.fmean(ds), 3) if ds else 0.0,
        "p99_ms": round(1000 * ds[min(len(ds) - 1, int(len(ds) * 0.99))], 3) if ds else 0.0,
        "max_ms": round(1000 * ds[-1], 3) if ds else 0.0,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def wilson(successes: float, n: int, z: float = 1.96):
    """Wilson score interval. Ties enter as half successes (a standard, slightly
    conservative approximation -- the interval is exact only for integer counts)."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        return {"opponent": label, "n": 0}
    wins = sum(1 for r in rows if r["result"] == 1.0)
    losses = sum(1 for r in rows if r["result"] == 0.0)
    ties = sum(1 for r in rows if r["result"] == 0.5)
    score = wins + 0.5 * ties
    lo, hi = wilson(score, n)
    us = [r["us"] for r in rows]
    them = [r["them"] for r in rows]
    bad = [r for r in rows if not r["ok"]]
    p99s = [r["p99_ms"] for r in rows]
    return {
        "opponent": label, "n": n,
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": round(score / n, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "money_us_mean": round(statistics.fmean(us), 1),
        "money_us_std": round(statistics.pstdev(us), 1) if n > 1 else 0.0,
        "money_them_mean": round(statistics.fmean(them), 1),
        "money_them_std": round(statistics.pstdev(them), 1) if n > 1 else 0.0,
        "money_us_min": round(min(us), 1),
        "failures": len(bad),
        "failure_examples": [f"seed{r['seed']}/seat{r['seat']}:{r['status_us']}:{r['error'][:160]}"
                             for r in bad[:3]],
        "turn_ms_p99_max": round(max(p99s), 3) if p99s else 0.0,
        "turn_ms_max": round(max(r["max_ms"] for r in rows), 3),
        "turn_ms_mean": round(statistics.fmean([r["mean_ms"] for r in rows]), 3),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="agent/main.py", help="the agent under test")
    p.add_argument("--opponents", nargs="+", default=["random", "starter"],
                   help="built-in names and/or paths to agent .py files")
    p.add_argument("--n", type=int, default=64, help="episodes PER OPPONENT (halved into seed pairs)")
    p.add_argument("--seed-base", type=int, default=0, help="0 = tuning seeds, 1000 = held-out")
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()-1")
    p.add_argument("--tag", default="adhoc", help="output goes to data/results/<tag>/")
    p.add_argument("--p99-ms", type=float, default=150.0, help="hard gate on p99 turn time")
    p.add_argument("--max-ms", type=float, default=400.0, help="hard gate on max turn time")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)

    import os
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    agent = _resolve(args.agent)
    opponents = [(spec, _resolve(spec)) for spec in args.opponents]

    pairs = max(1, args.n // 2)
    jobs = []
    for label, opp in opponents:
        for k in range(pairs):
            seed = args.seed_base + k
            jobs.append((agent, opp, seed, 0, args.steps))
            jobs.append((agent, opp, seed, 1, args.steps))

    print(f"engine     : kaggle-environments {engine_version()}")
    print(f"agent      : {agent}")
    print(f"opponents  : {[l for l, _ in opponents]}")
    print(f"episodes   : {len(jobs)} ({pairs} seed pairs x 2 seats x {len(opponents)} opponents), "
          f"seeds {args.seed_base}..{args.seed_base + pairs - 1}, {workers} workers")
    t0 = time.perf_counter()

    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(_play, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes  ({time.perf_counter() - t0:.0f}s)", flush=True)
    wall = time.perf_counter() - t0

    by_opp = {}
    for label, opp in opponents:
        by_opp[label] = [r for r in rows if r["opponent"] == opp]

    per_opp = [summarize(by_opp[label], label) for label, _ in opponents]
    overall = summarize(rows, "ALL")

    failures = sum(s["failures"] for s in per_opp)
    p99 = max((s["turn_ms_p99_max"] for s in per_opp), default=0.0)
    mx = max((s["turn_ms_max"] for s in per_opp), default=0.0)
    gates = {
        "no_failures": failures == 0,
        "p99_under_budget": p99 <= args.p99_ms,
        "max_under_budget": mx <= args.max_ms,
    }
    passed = all(gates.values())

    summary = {
        "tag": args.tag,
        "engine": engine_version(),
        "agent": agent,
        "opponents": [l for l, _ in opponents],
        "episodes_per_opponent": pairs * 2,
        "seed_base": args.seed_base,
        "seeds": [args.seed_base, args.seed_base + pairs - 1],
        "steps": args.steps,
        "wall_s": round(wall, 1),
        "per_opponent": per_opp,
        "overall": overall,
        "gates": gates,
        "gates_passed": passed,
        "p99_ms_budget": args.p99_ms,
        "max_ms_budget": args.max_ms,
    }

    print()
    hdr = f"{'opponent':<28} {'n':>4} {'win%':>7} {'95% CI':>15} {'our bank':>18} {'their bank':>18} {'fail':>5} {'p99ms':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in per_opp + [overall]:
        print(f"{s['opponent']:<28} {s['n']:>4} {100*s['win_rate']:>6.1f}% "
              f"{100*s['ci95'][0]:>6.1f}-{100*s['ci95'][1]:<6.1f}% "
              f"{s['money_us_mean']:>10.0f}+-{s['money_us_std']:<6.0f} "
              f"{s['money_them_mean']:>10.0f}+-{s['money_them_std']:<6.0f} "
              f"{s['failures']:>5} {s['turn_ms_p99_max']:>7.2f}")
    print()
    for k, v in gates.items():
        print(f"gate {k:<20} {'PASS' if v else 'FAIL'}")
    for s in per_opp:
        for ex_ in s["failure_examples"]:
            print(f"  ! {s['opponent']}: {ex_}")
    print(f"\nRESULT {'GATES_PASS' if passed else 'GATES_FAIL'}  "
          f"overall_win_rate={overall['win_rate']}  wall={wall:.0f}s")

    if not args.no_write:
        out = DATA_DIR / "results" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote      : {out / 'results.json'} and episodes.csv")

    print("SUMMARY " + json.dumps(summary["overall"]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
