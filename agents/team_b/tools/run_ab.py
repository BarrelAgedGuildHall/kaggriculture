"""Paired-seed side-swap A/B evaluator for Team B probes (sim-checklist).

Runs N seed-pairs of agent A vs agent B (each seed played twice with seats
swapped), reports A's win rate (ties = 0.5) with a Wilson 95% CI and money
mean +- std per side. This is the probe-grade counterpart of scripts/evaluate.py
for agents built from probe param dicts, which the arena registry cannot
snapshot as single files. Conventions match the arena: paired seeds, side swap,
screen band seeds 0.., engine version stamped.

Usage (from the repo root):
    uv run python agents/team_b/tools/run_ab.py --a probe:GooseRush --b probe:GradualRamp --pairs 32
    uv run python agents/team_b/tools/run_ab.py --a 'probe:{"hands_target":13}' --b 'probe:{"hands_target":10}' --pairs 32 --steps 720
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def wilson(score: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    phat = score / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def _init_worker():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b" / "probes"))
    sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))
    sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b" / "tools"))
    global _KE
    from _engine import import_engine
    _KE = import_engine()


def _load(spec: str):
    if spec.startswith("probe:"):
        from probe_core import make_agent, PRESETS
        arg = spec[6:]
        params = json.loads(arg) if arg.startswith("{") else dict(PRESETS[arg])
        return make_agent(params)
    if spec.startswith("g4control:"):
        # G4 hardcoded-price control (frozen 1.32.7 replica over routed_mixed)
        from g4_control import make_control_agent
        arg = spec[10:]
        if arg == "routed_mixed":
            return make_control_agent()
        if arg.startswith("{"):
            return make_control_agent(json.loads(arg))
        raise ValueError(f"unknown g4control spec: {spec}")
    if spec.startswith("millrace:"):
        # in-process factory with a package-unique module name: cannot collide
        # in sys.modules with probe_core or any directory-agent (arena bug of
        # 2026-08-19 — two directory-agents sharing top-level module names).
        from millrace.routed_probe import make_routed_agent
        arg = spec[9:]
        if arg == "routed_mixed":
            return make_routed_agent()
        if arg.startswith("{"):
            return make_routed_agent(json.loads(arg))
        raise ValueError(f"unknown millrace spec: {spec}")
    if spec in ("pass", "random", "starter"):
        return spec
    p = Path(spec)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def _episode(args):
    spec_a, spec_b, seed, swap, steps, market_perturb = args
    cfg = {"episodeSteps": steps, "seed": seed}
    if market_perturb:
        cfg["marketParams"] = market_perturb
    env = _KE.make("kaggriculture", configuration=cfg, debug=False)
    agents = [_load(spec_b), _load(spec_a)] if swap else [_load(spec_a), _load(spec_b)]
    env.run(agents)
    final = env.steps[-1]
    money = [final[i].observation.farms[i]["money"] for i in range(2)]
    statuses = [final[i].status for i in range(2)]
    a_idx = 1 if swap else 0
    return {"seed": seed, "swap": swap, "money_a": money[a_idx], "money_b": money[1 - a_idx],
            "status_a": statuses[a_idx], "status_b": statuses[1 - a_idx]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="agent A spec (probe:Preset, probe:{json}, builtin, .py path)")
    ap.add_argument("--b", required=True, help="agent B spec")
    ap.add_argument("--pairs", type=int, default=32, help="seed pairs (2 episodes each)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--seed-list", default=None,
                    help="comma-separated explicit seeds (overrides --pairs/--seed-base)")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--market-perturb", default=None,
                    help="JSON file with configuration.marketParams overrides (G4 assay)")
    args = ap.parse_args(argv)

    market_perturb = None
    if args.market_perturb:
        market_perturb = {k: v for k, v in json.loads(
            Path(args.market_perturb).read_text(encoding="utf-8")).items()
            if not k.startswith("_")}

    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",") if x.strip() != ""]
    else:
        seeds = list(range(args.seed_base, args.seed_base + args.pairs))
    jobs = []
    for s in seeds:
        jobs.append((args.a, args.b, s, False, args.steps, market_perturb))
        jobs.append((args.a, args.b, s, True, args.steps, market_perturb))

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.perf_counter()
    import multiprocessing as mp
    with mp.Pool(workers, initializer=_init_worker) as pool:
        rows = pool.map(_episode, jobs)
    wall = time.perf_counter() - t0

    score = 0.0
    crashes_a = sum(1 for r in rows if r["status_a"] != "DONE")
    crashes_b = sum(1 for r in rows if r["status_b"] != "DONE")
    for r in rows:
        if r["status_a"] != "DONE" and r["status_b"] == "DONE":
            continue  # loss for A
        elif r["status_b"] != "DONE" and r["status_a"] == "DONE":
            score += 1.0
        elif r["money_a"] > r["money_b"]:
            score += 1.0
        elif r["money_a"] == r["money_b"]:
            score += 0.5
    n = len(rows)
    lo, hi = wilson(score, n)
    ma = [r["money_a"] for r in rows]
    mb = [r["money_b"] for r in rows]
    from importlib.metadata import version as _v
    summary = {
        "engine": _v("kaggle-environments"),
        "tag": args.tag, "a": args.a, "b": args.b, "n_episodes": n, "pairs": len(seeds),
        "seeds": seeds if args.seed_list else [args.seed_base, args.seed_base + len(seeds) - 1],
        "steps": args.steps, "market_perturb": args.market_perturb,
        "win_rate_a": round(score / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "money_a_mean": round(statistics.fmean(ma), 1), "money_a_std": round(statistics.pstdev(ma), 1),
        "money_b_mean": round(statistics.fmean(mb), 1), "money_b_std": round(statistics.pstdev(mb), 1),
        "non_done_a": crashes_a, "non_done_b": crashes_b, "wall_s": round(wall, 1),
    }
    print(f"A={args.a}\nB={args.b}")
    print(f"win_rate(A)={summary['win_rate_a']:.3f}  CI95=[{lo:.3f},{hi:.3f}]  n={n}  "
          f"bankA={summary['money_a_mean']}+-{summary['money_a_std']}  bankB={summary['money_b_mean']}+-{summary['money_b_std']}  "
          f"nonDONE A/B={crashes_a}/{crashes_b}  wall={wall:.0f}s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
