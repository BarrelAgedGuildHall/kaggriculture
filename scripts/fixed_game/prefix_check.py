"""Verify prefix-snapshot candidate banks against full fixed-game replays."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fixed_game.pin import load_instance
from scripts.fixed_game.search import evaluate, mutate, schedule_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3910)
    parser.add_argument("--mutations", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=943910)
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    instance_path = args.instance or ROOT / "data" / "results" / "fixed-game" / str(args.seed) / "instance.json"
    schedule_path = args.schedule or (ROOT / "data" / "results" / "fixed-game" / str(args.seed)
                                      / "search-round3-yardstick" / "best-schedule.json")
    output_path = args.output or (ROOT / "data" / "results" / "fixed-game" / str(args.seed)
                                  / "prefix-eval" / "exactness.json")
    instance = load_instance(instance_path)
    fertilizer_drain = sum(int(row.get("drain", {}).get("FERTILIZER", 0))
                           for row in instance.get("town", {}).get("daily_drain", []))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    parent = evaluate(schedule, instance_path, fertilizer_drain, prefix_eval=False)
    rng = random.Random(args.random_seed)
    failures = []
    kinds = Counter()
    started = time.time()
    for index in range(args.mutations):
        candidate, earliest, kind = mutate(schedule, rng)
        kinds[kind] += 1
        full = evaluate(candidate, instance_path, fertilizer_drain, prefix_eval=False)
        prefix = evaluate(candidate, instance_path, fertilizer_drain, parent, earliest,
                          prefix_eval=True)
        if full["bank"] != prefix["bank"]:
            failures.append({"index": index, "mutation": kind, "earliest_day": earliest,
                             "full_bank": full["bank"], "prefix_bank": prefix["bank"]})
    result = {
        "engine": "kaggle-environments 1.32.7",
        "seed": args.seed,
        "schedule": str(schedule_path),
        "schedule_hash": schedule_hash(schedule),
        "random_seed": args.random_seed,
        "exact": args.mutations - len(failures),
        "total": args.mutations,
        "passed": not failures,
        "mutation_kinds": dict(kinds),
        "failures": failures,
        "wall_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
