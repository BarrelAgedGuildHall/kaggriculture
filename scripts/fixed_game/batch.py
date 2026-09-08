"""Run fixed-game solves sequentially with resumable seed selection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results" / "fixed-game"


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"descending seed range: {part}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(part))
    return list(dict.fromkeys(seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, help="comma-separated seeds and inclusive ranges")
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bound", action="store_true")
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    for seed in parse_seeds(args.seeds):
        solved = RESULTS / str(seed) / "solved.json"
        if solved.exists():
            print(f"skip seed {seed}: {solved} exists", flush=True)
            continue
        command = [sys.executable, str(ROOT / "scripts" / "fixed_game" / "solve_instance.py"),
                   "--seed", str(seed), "--minutes", str(args.minutes), "--workers", str(args.workers)]
        if args.bound:
            command.append("--bound")
        if args.polish:
            command.append("--polish")
        print(f"solve seed {seed} with {args.workers} workers", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
