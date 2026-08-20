"""Scan seeds for shop-draw extremes (expB-110).

Runs pass-vs-pass episodes and records the final unlocked_shops per seed, then
classifies extremes: 2+ PET_CAFE (carrot spike) and 0 YARN_STORE (wool dead).

Usage: uv run python agents/team_b/tools/scan_shops.py --seeds 0 63 --out shops.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _engine import import_engine  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 63])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ke = import_engine()
    rows = {}
    for seed in range(args.seeds[0], args.seeds[1] + 1):
        env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
        env.run(["pass_agent" if False else (lambda obs: {"farmer": ["PASS"], "hands": [], "market": []}),
                 (lambda obs: {"farmer": ["PASS"], "hands": [], "market": []})])
        town = env.steps[-1][0].observation.town
        shops = Counter(town["unlocked_shops"])
        rows[seed] = dict(shops)
        print(f"seed {seed}: {dict(shops)}", flush=True)

    petcafe2 = [s for s, r in rows.items() if r.get("PET_CAFE", 0) >= 2]
    yarn0 = [s for s, r in rows.items() if r.get("YARN_STORE", 0) == 0]
    summary = {"seeds": rows, "petcafe_ge2": petcafe2, "yarn_eq0": yarn0}
    print("PET_CAFE>=2:", petcafe2)
    print("YARN_STORE==0:", yarn0)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
