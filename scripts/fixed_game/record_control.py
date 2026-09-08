"""Record an arbitrary seat-0 policy against pass on a pinned instance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fixed_game.pin import load_instance, make_env  # noqa: E402
from scripts.fixed_game.sim import Sim  # noqa: E402
from scripts.kpi import _analyze  # noqa: E402


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location(f"fixed_control_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--agent", type=Path)
    source.add_argument("--actions", type=Path)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-actions", action="store_true")
    args = parser.parse_args()

    instance = load_instance(args.instance)
    timings: list[float] = []
    faults = 0
    if args.agent:
        module = _load_agent(args.agent)

        def policy(obs, config=None):
            nonlocal faults
            started = time.perf_counter_ns()
            try:
                action = module.agent(obs, config)
            except TypeError:
                action = module.agent(obs)
            except Exception:
                faults += 1
                hands = len(obs["farms"][0].get("hands", []))
                action = {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(hands)], "market": []}
            timings.append((time.perf_counter_ns() - started) / 1e6)
            return action

        env = make_env(instance, debug=True)
        env.run([policy, "pass"])
        actions = [step[0].action or {"farmer": ["PASS"], "hands": [], "market": []}
                   for step in env.steps[1:]]
    else:
        actions = json.loads(args.actions.read_text(encoding="utf-8"))
        sim = Sim(instance)
        for action in actions:
            if sim.done():
                break
            sim.step(action)
        env = sim.env
    ledger = _analyze(env, 0, int(instance["seed"]))
    while len(actions) < 720:
        actions.append({"farmer": ["PASS"], "hands": [], "market": []})
    result = {
        "seed": int(instance["seed"]),
        "agent": str(args.agent) if args.agent else None,
        "source_actions": str(args.actions) if args.actions else None,
        "final_bank": ledger["final_bank"],
        "status": ledger["status"],
        "faults": faults,
        "sold_units": ledger["sold_units"],
        "realised_dollars": ledger["realised_dollars"],
        "town_demand": ledger["town_demand"],
        "timing_ms": {
            "p99": _percentile(timings, .99),
            "max": max(timings, default=0.0),
            "mean": statistics.fmean(timings) if timings else 0.0,
        },
    }
    if args.include_actions:
        result["actions"] = actions[:720]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "actions"}, indent=2))


if __name__ == "__main__":
    main()
