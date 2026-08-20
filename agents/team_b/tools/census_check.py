"""Census assertion + arm-divergence check for Team B A/Bs (spec section 4 risk
table: "a chassis whose census contradicts its parameters voids every run on it";
CEO directive 2026-08-19: verify two A/B arms produce non-identical action
streams on one fixed seed before trusting any win rate).

Modes:
  census   — run one episode (agent vs starter), count what actually got built/
             placed/planted/fed, compare against --expect JSON, exit 1 on FAIL.
             Also reports planning-time stats if the agent exposes .timings
             (millrace chassis does).
  diverge  — run each of two agent specs vs the same opponent on the same seed
             and assert their per-turn action streams differ.

Usage (from the repo root):
  uv run python agents/team_b/tools/census_check.py census --agent millrace:routed_mixed \
      --expect '{"cows":6,"sheep":5,"geese_min":8,"melon_planted_min":4}' --seed 0
  uv run python agents/team_b/tools/census_check.py diverge --a millrace:routed_mixed \
      --b probe:Mixed --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _engine import import_engine, engine_version  # noqa: E402
from instrument import load_agent  # noqa: E402  (shares the spec loader)


def _percentile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def run_episode(spec_a, spec_b, seed, steps, record_actions_for=None):
    ke = import_engine()
    env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    agents = [load_agent(spec_a), load_agent(spec_b)]
    env.reset()
    stream = []
    fed_days = 0
    animal_days = 0
    while not env.done:
        obs = [env.state[i].observation for i in range(2)]
        acts = []
        for i, ag in enumerate(agents):
            try:
                a = ag(obs[i])
            except Exception:
                a = {"farmer": ["PASS"], "hands": [], "market": []}
            acts.append(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})
        if record_actions_for is not None:
            stream.append(json.dumps(acts[record_actions_for], sort_keys=True, default=str))
        # daily fed census at the last hour of each day
        if obs[0].hour == 23:
            for row in obs[0].farms[0]["tiles"]:
                for t in row:
                    if isinstance(t, dict) and "animal" in t:
                        animal_days += 1
                        fed_days += 1 if t["fed_today"] else 0
        env.step(acts)
    final = env.state[0].observation
    farm = final.farms[0]
    placed = {}
    planted = {}
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict):
                if "animal" in t:
                    placed[t["animal"]] = placed.get(t["animal"], 0) + 1
                elif t.get("kind") == "PLANT":
                    planted[t["crop"]] = planted.get(t["crop"], 0) + 1
    money = [env.steps[-1][i].observation.farms[i]["money"] for i in range(2)]
    return {
        "engine": engine_version(), "seed": seed, "steps": steps,
        "placed": placed, "planted_final": planted,
        "fed_fraction": round(fed_days / max(1, animal_days), 4),
        "final_money": money,
        "statuses": [s.status for s in env.steps[-1]],
        "timings": getattr(agents[0], "timings", None),
        "action_stream": stream,
    }


def cmd_census(args):
    r = run_episode(args.agent, args.opponent, args.seed, args.steps)
    exp = json.loads(args.expect) if args.expect else {}
    checks = []

    def chk(name, ok, detail):
        checks.append((name, ok, detail))

    p = r["placed"]
    if "cows" in exp:
        chk("cows placed", p.get("COW", 0) == exp["cows"], f"{p.get('COW', 0)} == {exp['cows']}")
    if "sheep" in exp:
        chk("sheep placed", p.get("SHEEP", 0) == exp["sheep"], f"{p.get('SHEEP', 0)} == {exp['sheep']}")
    if "geese_min" in exp:
        chk("geese placed >=", p.get("GOOSE", 0) >= exp["geese_min"], f"{p.get('GOOSE', 0)} >= {exp['geese_min']}")
    if "melon_planted_min" in exp:
        # final-board melon (cycles harvest out; require the program ran)
        chk("melon tiles >=", r["planted_final"].get("MELON", 0) >= exp["melon_planted_min"],
            f"{r['planted_final'].get('MELON', 0)} >= {exp['melon_planted_min']}")
    if "fed_fraction_min" in exp:
        chk("fed fraction >=", r["fed_fraction"] >= exp["fed_fraction_min"],
            f"{r['fed_fraction']} >= {exp['fed_fraction_min']}")
    chk("status DONE", r["statuses"][0] == "DONE", r["statuses"][0])

    timing = None
    if r["timings"]:
        timing = {"p50_ms": round(1000 * _percentile(r["timings"], 0.50), 2),
                  "p99_ms": round(1000 * _percentile(r["timings"], 0.99), 2),
                  "max_ms": round(1000 * max(r["timings"]), 2)}
        chk("p99 <= 150 ms", timing["p99_ms"] <= 150, f"{timing['p99_ms']} ms")
        chk("max <= 400 ms", timing["max_ms"] <= 400, f"{timing['max_ms']} ms")

    out = {"engine": r["engine"], "agent": args.agent, "opponent": args.opponent,
           "seed": r["seed"], "placed": r["placed"], "planted_final": r["planted_final"],
           "fed_fraction": r["fed_fraction"], "final_money": r["final_money"],
           "timing": timing,
           "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]}
    ok_all = all(ok for _, ok, _ in checks)
    out["verdict"] = "PASS" if ok_all else "FAIL"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if ok_all else 1


def cmd_diverge(args):
    ra = run_episode(args.a, args.opponent, args.seed, args.steps, record_actions_for=0)
    rb = run_episode(args.b, args.opponent, args.seed, args.steps, record_actions_for=0)
    sa, sb = ra["action_stream"], rb["action_stream"]
    n_diff = sum(1 for x, y in zip(sa, sb) if x != y)
    identical_banks = ra["final_money"] == rb["final_money"]
    out = {"engine": ra["engine"], "a": args.a, "b": args.b, "seed": args.seed,
           "turns_compared": min(len(sa), len(sb)), "turns_differing": n_diff,
           "final_money_a": ra["final_money"], "final_money_b": rb["final_money"],
           "identical_banks": identical_banks,
           "verdict": "PASS" if (n_diff > 0 and not identical_banks) else "FAIL"}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["verdict"] == "PASS" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("--agent", required=True)
    c.add_argument("--opponent", default="starter")
    c.add_argument("--expect", default=None, help="JSON expectations")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--steps", type=int, default=720)
    c.add_argument("--out", default=None)
    d = sub.add_parser("diverge")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--opponent", default="starter")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--steps", type=int, default=720)
    d.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return cmd_census(args) if args.cmd == "census" else cmd_diverge(args)


if __name__ == "__main__":
    raise SystemExit(main())
