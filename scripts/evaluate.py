"""Evaluate an agent against a gauntlet -- thin wrapper over the arena (`python -m arena match`).

Kept for the Modeler's workflow and flags; every game it plays goes through the arena scheduler
(paired seeds, side swap, process pool, crash/timeout flags, framework timing) and lands in the
arena store, so the leaderboard and the gates see it too. The agent file is registered ad hoc as
`<stem>@<sha>` (team --team, default A) unless it is already registered.

Usage (from the repo root):
    uv run python scripts/evaluate.py --n 64 --tag exp-001-screen
    uv run python scripts/evaluate.py --n 400 --seed-base 1000 --tag exp-001-confirm
    uv run python scripts/evaluate.py --agent agents/A/v1.py --opponents random starter v0 --n 64

Rules it enforces (strategy.md section 3): score = win rate with a Wilson 95% CI (ties count half);
paired seeds with side swap; screen on seeds 0..199, confirm on held-out 1000..1199; the run is
FAILED (exit 1, do not quote the score) on any crash, any timeout/turn over budget, or a p99 turn
time over --p99-ms. Writes data/results/<tag>/results.json (summary) as before.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arena import ratings, registry  # noqa: E402
from arena.paths import CONFIRM_SEED_BASE, DATA_DIR, SCREEN_SEED_BASE  # noqa: E402
from arena.scheduler import run_match  # noqa: E402


def _register(spec: str, team: str) -> dict:
    p = Path(spec)
    if p.suffix == ".py" or p.is_dir():
        if not p.is_absolute():
            p = REPO_ROOT / p
        return registry.register_file(p, p.stem if p.suffix else p.name, team)
    return registry.resolve(spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="agent/main.py", help="agent under test: .py path, dir, or registered name")
    ap.add_argument("--opponents", nargs="+", default=["random", "starter"],
                    help="built-in names, registered names, and/or agent .py paths")
    ap.add_argument("--n", type=int, default=64, help="episodes PER OPPONENT (halved into seed pairs)")
    ap.add_argument("--seed-base", type=int, default=SCREEN_SEED_BASE, help="0 = tuning seeds, 1000 = held-out")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()-2")
    ap.add_argument("--team", default="A", choices=("A", "B", "resident"), help="team to register the agent under")
    ap.add_argument("--tag", default="adhoc", help="summary goes to data/results/<tag>/results.json")
    ap.add_argument("--p99-ms", type=float, default=150.0)
    ap.add_argument("--max-ms", type=float, default=400.0)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    agent = _register(args.agent, args.team)
    opps = [_register(o, "resident") for o in args.opponents]
    pairs = max(1, args.n // 2)
    seeds = f"{args.seed_base}-{args.seed_base + pairs - 1}"
    mode = "confirm" if args.seed_base >= CONFIRM_SEED_BASE else "screen"
    t0 = time.perf_counter()
    res = run_match([agent["id"]] + [o["id"] for o in opps], mode=mode, n=args.n, seeds=seeds,
                    workers=args.workers or None, steps=args.steps, run_id=None)
    rows = res["rows"]
    wall = time.perf_counter() - t0

    per_opp = []
    for o in opps:
        sub = [g for g in rows if {g["p0"], g["p1"]} == {agent["id"], o["id"]}]
        st = ratings.per_agent_stats(sub)
        me, them = st.get(agent["id"], {}), st.get(o["id"], {})
        score = sum(float(g["result0"]) if g["p0"] == agent["id"] else 1.0 - float(g["result0"]) for g in sub)
        lo, hi = ratings.wilson(score, len(sub))
        per_opp.append({
            "opponent": o["id"], "n": len(sub), "win_rate": round(score / len(sub), 4) if sub else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)],
            "money_us_mean": round(me.get("money_mean", 0.0), 1), "money_us_std": round(me.get("money_std", 0.0), 1),
            "money_them_mean": round(them.get("money_mean", 0.0), 1), "money_them_std": round(them.get("money_std", 0.0), 1),
            "crashes": me.get("crashes", 0), "timeouts": me.get("timeouts", 0),
            "turn_ms_p99_max": round(me.get("p99_ms", 0.0), 3), "turn_ms_max": round(me.get("max_ms", 0.0), 3),
        })
    st_all = ratings.per_agent_stats(rows).get(agent["id"], {})
    score_all = sum(float(g["result0"]) if g["p0"] == agent["id"] else 1.0 - float(g["result0"]) for g in rows)
    lo, hi = ratings.wilson(score_all, len(rows))
    gates = {
        "no_crashes": st_all.get("crashes", 0) == 0,
        "no_timeouts": st_all.get("timeouts", 0) == 0,
        "p99_under_budget": st_all.get("p99_ms", 0.0) <= args.p99_ms,
        "max_under_budget": st_all.get("max_ms", 0.0) <= args.max_ms,
    }
    passed = all(gates.values())
    summary = {
        "tag": args.tag, "engine": rows[0]["engine"] if rows else None, "agent": agent["id"],
        "opponents": [o["id"] for o in opps], "episodes_per_opponent": pairs * 2, "seed_base": args.seed_base,
        "seeds": [args.seed_base, args.seed_base + pairs - 1], "steps": args.steps, "wall_s": round(wall, 1),
        "run_id": res["run_id"], "per_opponent": per_opp,
        "overall": {"n": len(rows), "win_rate": round(score_all / len(rows), 4) if rows else 0.0,
                    "ci95": [round(lo, 4), round(hi, 4)], **{k: v for k, v in st_all.items() if k != "score"}},
        "gates": gates, "gates_passed": passed, "p99_ms_budget": args.p99_ms, "max_ms_budget": args.max_ms,
    }
    print()
    hdr = f"{'opponent':<34} {'n':>4} {'win%':>7} {'95% CI':>15} {'our bank':>18} {'their bank':>18} {'crash':>5} {'t/o':>4} {'p99ms':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in per_opp:
        print(f"{s['opponent']:<34} {s['n']:>4} {100 * s['win_rate']:>6.1f}% {100 * s['ci95'][0]:>6.1f}-{100 * s['ci95'][1]:<6.1f}% "
              f"{s['money_us_mean']:>10.0f}+-{s['money_us_std']:<6.0f} {s['money_them_mean']:>10.0f}+-{s['money_them_std']:<6.0f} "
              f"{s['crashes']:>5} {s['timeouts']:>4} {s['turn_ms_p99_max']:>7.2f}")
    for k, v in gates.items():
        print(f"gate {k:<20} {'PASS' if v else 'FAIL'}")
    print(f"\nRESULT {'GATES_PASS' if passed else 'GATES_FAIL'}  overall_win_rate={summary['overall']['win_rate']}  "
          f"wall={wall:.0f}s  (arena run {res['run_id']}; `python -m arena leaderboard` to rank)")
    if not args.no_write:
        out = DATA_DIR / "results" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote      : {out / 'results.json'}")
    print("SUMMARY " + json.dumps(summary["overall"]))
    return 0 if passed else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
