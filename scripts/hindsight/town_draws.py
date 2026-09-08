"""Census deterministic Kaggriculture town-shop draws with no-op farms.

The draw is measured from a real pinned-engine episode because weed RNG is
consumed before each shop choice.  Drain is accumulated from the pre-step shop
list using the engine's exact town-consumption rule.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "hindsight-v3" / "town_draws.json"

import sys
sys.path.insert(0, str(REPO / "scripts"))
from _engine import engine_version, import_engine, to_plain  # noqa: E402


def _one(seed: int) -> dict[str, Any]:
    ke = import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as kg

    env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.reset(2)
    products = tuple(kg.PRODUCTS)
    drain = {p: 0 for p in products}
    unlocks: list[dict[str, Any]] = []
    previous: list[str] = []
    noop = {"farmer": "STAY", "hands": [], "market": []}
    while not env.done:
        obs = to_plain(env.state[0].observation)
        step = int(obs.get("step", len(env.steps) - 1))
        shops = list(obs["town"].get("unlocked_shops", []))
        for index in range(len(previous), len(shops)):
            name = shops[index]
            unlocks.append({"instance": index + 1, "shop": name,
                            "products": list(kg.SHOPS[name]), "unlock_day": step // 24})
        previous = shops
        if step % max(1, int(env.configuration.townShopSellInterval)) == 0:
            for name in shops:
                items = kg.SHOPS[name]
                multiplier = 2 if len(items) == 1 else 1
                for item in items:
                    drain[item] += multiplier
        if step % max(1, int(env.configuration.townCenterSellInterval)) == 0:
            for item in kg.TOWN_CENTER_PRODUCTS:
                drain[item] += 1
        env.step([noop, noop])
    # The last unlock occurs in the transition to a new day. Capture it even if
    # the episode ends immediately after that transition.
    shops = list(to_plain(env.state[0].observation)["town"].get("unlocked_shops", []))
    for index in range(len(previous), len(shops)):
        name = shops[index]
        unlocks.append({"instance": index + 1, "shop": name,
                        "products": list(kg.SHOPS[name]), "unlock_day": 3 * (index + 1)})
    return {"seed": seed, "unlocks": unlocks, "shops": shops, "season_drain": drain}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seed-end", type=int, default=5099, help="inclusive")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.seed_end < args.seed_start:
        parser.error("--seed-end must be >= --seed-start")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be in 1..8")
    seeds = list(range(args.seed_start, args.seed_end + 1))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        episodes = list(pool.map(_one, seeds))
    products = ("MILK", "WOOL", "STRAWBERRY", "EGG")
    census = {p: sum(any(p in u["products"] for u in row["unlocks"]) for row in episodes)
              for p in products}
    shop_instances = Counter(u["shop"] for row in episodes for u in row["unlocks"])
    payload = {"schema_version": 1, "engine": engine_version(),
               "seed_start": args.seed_start, "seed_end": args.seed_end,
               "seed_count": len(seeds), "method": "no-op vs no-op real episodes",
               "product_shop_seed_counts": census,
               "shop_instance_counts": dict(sorted(shop_instances.items())),
               "episodes": episodes}
    output = args.output if args.output.is_absolute() else REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "seed_count": len(seeds), "census": census,
                      "shop_instances": payload["shop_instance_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
