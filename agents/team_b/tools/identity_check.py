"""Behaviour-identity re-gate — ruling-04 section 1.5 / R-32.

A change confined to code paths that provably did not execute during the
original gate may be re-gated by proving ACTION-STREAM IDENTITY on the gate's
own seeds, instead of by re-running the gate.  It is cheaper, and it is stronger
evidence than a fresh win rate: a win rate at n=400 has a +/-4.9pp half-width,
whereas identity over 464 episodes has none.

This tool plays the SAME matchups (same opponent snapshot, same seeds, same
seats) with two builds of our agent and asserts, per seat-season:

  (a) both seats' final banks are bit-identical, and
  (b) the per-turn EMITTED-ACTION hash of our seat is identical.

(b) is the load-bearing one.  Bank identity alone can hide compensating
differences; the action stream is the agent's entire externally visible
behaviour, and `env.steps[t][i].action` is what the interpreter actually
consumed.

Determinism caveat, stated because it is the one way this proof can lie: the
OLD chassis carries a wall-clock branch (`if perf_counter() - t0 > 0.25: out =
fb`), so under host load its own action stream is not reproducible.  Every game
therefore records the baseline's worst turn, and any baseline game whose worst
turn exceeds 250 ms is reported as CONTAMINATED and must be re-run on a quiet
host before its comparison counts.  The new chassis has no wall-clock branch at
all, so its side is deterministic by construction.

Usage (from the repo root):
  uv run python agents/team_b/tools/identity_check.py \
      --base TeamB-Millrace-s32m6@hddf03d4580 \
      --arm  agents/team_b/millrace \
      --opponent TeamB-Millrace-pricer@hdd12354335 \
      --seeds 0-63 --out runs/expB-306c/identity-screen.json

  # mirror arm: our build on both seats
  ... --opponent SELF --seeds 0-31

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from arena import engine as eng              # noqa: E402
from arena import registry as areg           # noqa: E402
from arena.isolation import SubprocessAgent  # noqa: E402
from arena.paths import ENV_NAME             # noqa: E402

BASELINE_WALLCLOCK_S = 0.25   # the OLD chassis's discard threshold; see the caveat above


def resolve_entry(spec: str) -> Path:
    p = Path(spec)
    if p.is_dir():
        return p / "main.py"
    if p.suffix == ".py" and p.exists():
        return p
    entry = areg.resolve(spec)
    if entry.get("kind") != "file":
        raise SystemExit(f"agent {spec!r} is kind={entry.get('kind')!r}; this tool needs a file agent")
    return areg.entry_path(entry)


def _canon(action) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), default=str)


def play(entry0: Path, entry1: Path, seed: int, steps: int) -> dict:
    ke = eng.import_engine()
    env = ke.make(ENV_NAME, configuration={"episodeSteps": int(steps), "seed": int(seed)}, debug=False)
    agents = [SubprocessAgent(entry0, name="p0"), SubprocessAgent(entry1, name="p1")]
    try:
        env.run(agents)
    finally:
        for a in agents:
            a.close()
    final = env.steps[-1]
    streams, maxdur = [], [0.0, 0.0]
    for i in (0, 1):
        h = hashlib.sha256()
        for s in env.steps:
            h.update(_canon(s[i].action).encode())
            h.update(b"|")
        streams.append(h.hexdigest())
    for lg in env.logs:
        for i in (0, 1):
            if len(lg) > i and isinstance(lg[i], dict) and "duration" in lg[i]:
                maxdur[i] = max(maxdur[i], float(lg[i]["duration"]))
    return {
        "money": [float(final[i].observation.farms[i]["money"]) for i in (0, 1)],
        "status": [str(final[i].status) for i in (0, 1)],
        "action_sha": streams,
        "max_turn_s": maxdur,
        "n_steps": len(env.steps),
    }


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="the GATED build (arena id or path)")
    p.add_argument("--arm", required=True, help="the new build (arena id or path)")
    p.add_argument("--opponent", required=True, help="arena id / path, or SELF for the mirror")
    p.add_argument("--seeds", required=True)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--both-seats", action="store_true",
                   help="play every seed on BOTH seats (the gate's side-swap balance)")
    p.add_argument("--out")
    args = p.parse_args(argv)

    # A plan exception must never be masked inside an identity proof.
    os.environ.setdefault("MILLRACE_STRICT", "1")

    e_base = resolve_entry(args.base)
    e_arm = resolve_entry(args.arm)
    mirror = args.opponent.upper() == "SELF"
    e_opp = None if mirror else resolve_entry(args.opponent)

    seats = (0, 1) if args.both_seats else (0,)
    rows, mismatches, contaminated = [], [], []
    t_start = time.perf_counter()
    for seed in parse_seeds(args.seeds):
        for seat in seats:
            def sides(entry):
                if mirror:
                    return entry, entry
                return (entry, e_opp) if seat == 0 else (e_opp, entry)

            b0, b1 = sides(e_base)
            a0, a1 = sides(e_arm)
            rb = play(b0, b1, seed, args.steps)
            ra = play(a0, a1, seed, args.steps)

            same_money = rb["money"] == ra["money"]
            # mirror: both seats are ours, so both action streams must match
            check_seats = (0, 1) if mirror else (seat,)
            same_actions = all(rb["action_sha"][i] == ra["action_sha"][i] for i in check_seats)
            hot = max(rb["max_turn_s"]) > BASELINE_WALLCLOCK_S
            row = {
                "seed": seed, "seat": seat,
                "base_money": rb["money"], "arm_money": ra["money"],
                "base_sha": [rb["action_sha"][i][:16] for i in check_seats],
                "arm_sha": [ra["action_sha"][i][:16] for i in check_seats],
                "money_identical": same_money, "actions_identical": same_actions,
                "base_max_turn_ms": round(1000 * max(rb["max_turn_s"]), 1),
                "arm_max_turn_ms": round(1000 * max(ra["max_turn_s"]), 1),
                "baseline_contaminated": hot,
                "status": [rb["status"], ra["status"]],
            }
            rows.append(row)
            if hot:
                contaminated.append(row)
            if not (same_money and same_actions):
                mismatches.append(row)
            flag = "OK " if (same_money and same_actions) else "DIFF"
            if hot:
                flag += " HOT"
            print(f"  seed {seed} seat {seat}: {flag}  base {rb['money']} arm {ra['money']}  "
                  f"base_max {row['base_max_turn_ms']} ms", flush=True)

    report = {
        "base": args.base, "arm": args.arm, "opponent": args.opponent,
        "seeds": args.seeds, "both_seats": args.both_seats, "mirror": mirror,
        "engine": eng.engine_version(),
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "?"),
        "n_pairs": len(rows),
        "n_mismatch": len(mismatches),
        "n_baseline_contaminated": len(contaminated),
        "wall_s": round(time.perf_counter() - t_start, 1),
        "verdict": ("IDENTICAL" if not mismatches else "DIVERGENT"),
        "rows": rows,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"\nIDENTITY {report['verdict']}: {len(rows) - len(mismatches)}/{len(rows)} pairs identical "
          f"(bank AND per-turn action hash); baseline-contaminated {len(contaminated)}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
