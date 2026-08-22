"""Run one Kaggriculture episode locally and report the result.

Usage (from the repo root):
    uv run python scripts/run_episode.py --seed 0
    uv run python scripts/run_episode.py --agent-a agent/main.py --agent-b starter --seed 3
    uv run python scripts/run_episode.py --agent-a agent/main.py --agent-b random --no-save

Agents are built-in names (`pass`, `random`, `starter`), a path to a .py file exposing
`agent(obs)` (run through the official Agent wrapper, exactly as Kaggle does it), or
`module:function`. Prints final money for both players and per-turn wall-time stats for
agent A (our agent), then writes the replay JSON to data/replays/local/.

Exit code is 0 on a clean finish (both agents DONE), 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Team B's millrace chassis masks planner exceptions so a crash on the ladder is
# not `status ERROR` (a certain loss).  That mask must never be silent in a
# measurement harness: ruling-04 C-2 requires MILLRACE_STRICT in the test suite,
# in this script and in every Pit harness invocation, so a swallowed exception
# surfaces as a loud failure instead of a full episode of PASS.  Harmless for
# every other agent; override by exporting MILLRACE_STRICT= (empty) yourself.
os.environ.setdefault("MILLRACE_STRICT", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import ENV_NAME, LOCAL_REPLAYS, REPO_ROOT, engine_version, import_engine, load_agent_callable  # noqa: E402


def summarize_durations(durations: list[float]) -> dict:
    if not durations:
        return {"n": 0}
    ds = sorted(durations)
    return {
        "n": len(ds),
        "mean_ms": round(1000 * statistics.fmean(ds), 3),
        "p50_ms": round(1000 * ds[len(ds) // 2], 3),
        "p99_ms": round(1000 * ds[min(len(ds) - 1, int(len(ds) * 0.99))], 3),
        "max_ms": round(1000 * ds[-1], 3),
        "total_s": round(sum(ds), 3),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent-a", default=str(REPO_ROOT / "agent" / "main.py"), help="our agent (player 0)")
    p.add_argument("--agent-b", default="random", help="opponent (player 1)")
    p.add_argument("--seed", type=int, default=0, help="episode seed (engine RNG)")
    p.add_argument("--steps", type=int, default=720, help="episodeSteps (competition default 720)")
    p.add_argument("--no-save", action="store_true", help="do not write the replay JSON")
    p.add_argument("--out-dir", default=str(LOCAL_REPLAYS), help="replay output directory")
    p.add_argument("--quiet", action="store_true", help="only print the JSON summary line")
    args = p.parse_args(argv)

    ke = import_engine()
    env = ke.make(ENV_NAME, configuration={"episodeSteps": args.steps, "seed": args.seed}, debug=False)

    agents = [load_agent_callable(args.agent_a), load_agent_callable(args.agent_b)]
    t0 = time.perf_counter()
    env.run(agents)
    wall = time.perf_counter() - t0

    final = env.steps[-1]
    money = [s.observation.farms[i]["money"] for i, s in enumerate(final)]
    statuses = [s.status for s in final]
    rewards = [s.reward for s in final]

    # Per-turn durations measured by the framework's Agent wrapper (perf_counter around act()).
    dur_a = [lg[0]["duration"] for lg in env.logs if len(lg) > 0 and isinstance(lg[0], dict) and "duration" in lg[0]]
    dur_b = [lg[1]["duration"] for lg in env.logs if len(lg) > 1 and isinstance(lg[1], dict) and "duration" in lg[1]]

    summary = {
        "engine": f"kaggle-environments {engine_version()}",
        "seed": args.seed,
        "steps_recorded": len(env.steps),
        "agent_a": args.agent_a,
        "agent_b": args.agent_b,
        "final_money": money,
        "rewards": rewards,
        "statuses": statuses,
        "episode_wall_s": round(wall, 2),
        "agent_a_turn_time": summarize_durations(dur_a),
        "agent_b_turn_time": summarize_durations(dur_b),
    }

    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        a_name = Path(args.agent_a).stem if args.agent_a.endswith(".py") else args.agent_a.replace(":", "_")
        b_name = Path(args.agent_b).stem if args.agent_b.endswith(".py") else args.agent_b.replace(":", "_")
        out = out_dir / f"{a_name}_vs_{b_name}_seed{args.seed}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(env.toJSON(), f)
        summary["replay"] = str(out)

    if not args.quiet:
        print(f"engine      : {summary['engine']}  seed={args.seed}  steps={len(env.steps)}  wall={wall:.1f}s")
        print(f"player 0 (A): {args.agent_a}  money={money[0]:.0f}  reward={rewards[0]}  status={statuses[0]}")
        print(f"player 1 (B): {args.agent_b}  money={money[1]:.0f}  reward={rewards[1]}  status={statuses[1]}")
        print(f"A turn time : {summary['agent_a_turn_time']}")
        print(f"B turn time : {summary['agent_b_turn_time']}")
        if "replay" in summary:
            print(f"replay      : {summary['replay']}")
    print("SUMMARY " + json.dumps(summary))
    return 0 if all(s == "DONE" for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
