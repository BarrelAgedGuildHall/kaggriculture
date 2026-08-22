"""expB-306d — did the shipped chassis's fallback fire on the real Kaggle ladder?

ruling-04 R-33. Ladder artefact 55683922 (`TeamB-Millrace-pricer@hdd12354335`) is
our calibration point, and it carries the OLD chassis: a per-turn wall-clock
branch at 250 ms that discards a completed plan and emits `fallback_action(obs)`
instead. The Kaggle sandbox is 1.6 vCPU against a Pit host of 16, so the question
"did that branch fire in the sandbox, silently, and drag the rating down" is not
answerable by any local throttle. It IS answerable from the replays, because
Kaggle publishes the full per-step observation — including `private` (the shed)
and `remainingOverageTime`.

THE DISCRIMINATOR IS EXACT, not heuristic. The old fallback is a pure function of
the observation:

    every unit -> ["DROP"] if it is carrying AND standing on a shed tile, else ["PASS"]
    market     -> ["SELL", item, n] for every non-animal item in the shed with n > 0,
                  in shed iteration order, capped at 10 orders

so this tool RECOMPUTES that function from each stored observation and compares it
for exact equality with the action the agent actually emitted. A turn matches iff
the agent played the fallback (or, rarely, iff the planner independently chose
exactly the same thing — which is what the control run measures).

CONTROL (the part that makes the number decidable): run the identical agent in the
Pit, where we know from the stored timing rows that no turn ever exceeded 250 ms,
and classify those replays with the same code. That fixes the discriminator's
false-positive rate against a build we know never fell back. A ladder rate above
the control rate is fallback; a ladder rate at the control rate is coincidence.

Also reported, because the replay hands it to us for free and it is the single
most direct measurement of ruling-04 C-6 available anywhere:
`remainingOverageTime` per turn — the engine's own overage bank, ON THE REAL
SANDBOX. `60.0 - min(bank)` is exactly how much of the 60 s budget the agent
consumed in that episode.

Usage:
  uv run python agents/team_b/tools/fallback_audit.py \
      --replays "D:/GuildhallData/kaggler/datasets/ladder-replays-55683922/*.json" \
      --team BarrelAged --out runs/expB-306d/ladder.json

  # control: replays produced by the same agent in the Pit
  uv run python agents/team_b/tools/fallback_audit.py \
      --replays "runs/expB-306d/control/*.json" --seat both \
      --out runs/expB-306d/control.json

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
ANIMAL_NAMES = ("GOOSE", "COW", "SHEEP")


def old_fallback(obs) -> dict | None:
    """`agents/team_b/millrace/chassis.py::fallback_action` as it shipped in
    `TeamB-Millrace-pricer@hdd12354335` (sha256 016f78b6...), reproduced exactly.
    Returns None if the observation lacks what the function reads."""
    try:
        me = obs["farms"][obs["player"]]
        priv = obs["private"]
        units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
        invs = priv["inventories"]
        acts = []
        for i, pos in enumerate(units):
            carry = sum((invs[i] if i < len(invs) else {}).values())
            acts.append(["DROP"] if (carry > 0 and pos in SHED_TILES) else ["PASS"])
        orders = []
        for item, n in priv["shed"].items():
            if n > 0 and item not in ANIMAL_NAMES and len(orders) < 10:
                orders.append(["SELL", item, int(n)])
        return {"farmer": acts[0], "hands": acts[1:], "market": orders}
    except Exception:
        return None


def _norm(a):
    """Compare structurally: JSON round-trips tuples to lists and ints to ints."""
    return json.loads(json.dumps(a, default=list))


def unit_actions(action):
    if not isinstance(action, dict):
        return []
    return [action.get("farmer")] + list(action.get("hands") or [])


def audit_episode(path: Path, seats: list[int]) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    steps = d.get("steps") or []
    info = d.get("info") or {}
    rows = []
    for seat in seats:
        n = fb = all_pass = pass_or_drop = 0
        banks = []
        fb_turns = []
        for t, st in enumerate(steps):
            if len(st) <= seat:
                continue
            s = st[seat]
            act = s.get("action")
            obs = s.get("observation") or {}
            if not isinstance(act, dict):
                continue          # step 0 / no action recorded
            n += 1
            b = obs.get("remainingOverageTime")
            if isinstance(b, (int, float)):
                banks.append(float(b))
            uas = unit_actions(act)
            if uas and all(isinstance(u, list) and u and u[0] == "PASS" for u in uas):
                all_pass += 1
            if uas and all(isinstance(u, list) and u and u[0] in ("PASS", "DROP") for u in uas):
                pass_or_drop += 1
            want = old_fallback(obs)
            if want is not None and _norm(want) == _norm(act):
                fb += 1
                fb_turns.append(t)
        rows.append({
            "episode": path.stem,
            "episode_id": info.get("EpisodeId"),
            "seat": seat,
            "team": (info.get("TeamNames") or [None, None])[seat] if info.get("TeamNames") else None,
            "turns": n,
            "fallback_turns": fb,
            "fallback_rate": (fb / n) if n else 0.0,
            "all_pass_turns": all_pass,
            "all_pass_rate": (all_pass / n) if n else 0.0,
            "pass_or_drop_turns": pass_or_drop,
            "pass_or_drop_rate": (pass_or_drop / n) if n else 0.0,
            "first_fallback_turns": fb_turns[:20],
            "overage_bank_start": banks[0] if banks else None,
            "overage_bank_min": min(banks) if banks else None,
            "overage_consumed_s": (round(banks[0] - min(banks), 6) if banks else None),
            "reward": (d.get("rewards") or [None, None])[seat] if d.get("rewards") else None,
            "status": (d.get("statuses") or [None, None])[seat] if d.get("statuses") else None,
        })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replays", required=True, help="glob of replay JSON files")
    p.add_argument("--team", help="only audit seats whose info.TeamNames entry equals this")
    p.add_argument("--seat", default="auto", choices=["auto", "0", "1", "both"])
    p.add_argument("--out")
    args = p.parse_args(argv)

    paths = sorted(Path(x) for x in glob.glob(args.replays))
    if not paths:
        raise SystemExit(f"no replays matched {args.replays!r}")

    rows = []
    for path in paths:
        if args.seat == "both":
            seats = [0, 1]
        elif args.seat in ("0", "1"):
            seats = [int(args.seat)]
        else:
            head = json.loads(path.read_text(encoding="utf-8")).get("info", {})
            names = head.get("TeamNames") or []
            seats = [i for i, nm in enumerate(names) if nm == args.team] if args.team else [0, 1]
            if not seats:
                print(f"  {path.name}: team {args.team!r} not in {names} — skipped", file=sys.stderr)
                continue
        for r in audit_episode(path, seats):
            rows.append(r)
            print(f"  {r['episode_id'] or r['episode']} seat {r['seat']} ({r['team']}): "
                  f"turns {r['turns']}  fallback {r['fallback_turns']} ({100*r['fallback_rate']:.3f}%)  "
                  f"all-PASS {100*r['all_pass_rate']:.2f}%  "
                  f"bank {r['overage_bank_start']}->{r['overage_bank_min']} "
                  f"(consumed {r['overage_consumed_s']} s)  reward {r['reward']}", flush=True)

    tot_turns = sum(r["turns"] for r in rows)
    tot_fb = sum(r["fallback_turns"] for r in rows)
    worst = max(rows, key=lambda r: r["fallback_rate"]) if rows else None
    consumed = [r["overage_consumed_s"] for r in rows if r["overage_consumed_s"] is not None]
    report = {
        "replays": args.replays, "n_seat_episodes": len(rows),
        "total_turns": tot_turns,
        "total_fallback_turns": tot_fb,
        "pooled_fallback_rate": (tot_fb / tot_turns) if tot_turns else 0.0,
        "worst_episode_fallback_rate": worst["fallback_rate"] if worst else 0.0,
        "worst_episode": (worst["episode_id"] or worst["episode"]) if worst else None,
        "pooled_all_pass_rate": (sum(r["all_pass_turns"] for r in rows) / tot_turns) if tot_turns else 0.0,
        "max_overage_consumed_s": max(consumed) if consumed else None,
        "min_bank_observed_s": min((r["overage_bank_min"] for r in rows
                                    if r["overage_bank_min"] is not None), default=None),
        "rows": rows,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"\nPOOLED fallback rate {100*report['pooled_fallback_rate']:.4f}% "
          f"over {tot_turns} turns / {len(rows)} seat-episodes; "
          f"worst episode {100*report['worst_episode_fallback_rate']:.4f}% "
          f"({report['worst_episode']}); all-PASS proxy {100*report['pooled_all_pass_rate']:.2f}%; "
          f"max overage consumed {report['max_overage_consumed_s']} s; "
          f"min bank {report['min_bank_observed_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
