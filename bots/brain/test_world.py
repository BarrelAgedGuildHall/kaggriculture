"""Five-seed parity check for Brain's market and town world model."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as engine


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("brain_main", HERE / "main.py")
brain = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = brain
spec.loader.exec_module(brain)


def market_check(seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    max_price_error = 0
    max_inventory_error = 0
    for item in brain.PRODUCTS:
        start = 10000 + rng.randint(-250, 250)
        qty = rng.randint(1, 80)
        predicted_revenue, predicted_inventory = brain.predict_sale(item, start, qty)
        market = engine._new_market()
        market["inventory"][item] = start
        private = engine._new_private()
        private["shed"][item] = qty
        farm = engine._new_farm(10, 0)
        for _ in range(qty):
            quote = engine.market_price(item, market["inventory"][item])
            assert engine._commit_unit("SELL", item, quote, farm, private, market)
        engine._refresh_prices(market)
        max_price_error = max(max_price_error, abs(predicted_revenue - int(farm["money"])))
        max_inventory_error = max(max_inventory_error, abs(predicted_inventory - market["inventory"][item]))
        assert brain.market_price(item, market["inventory"][item]) == market["prices"][item]
    return max_price_error, max_inventory_error


def town_check(seed: int) -> tuple[int, int]:
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run(["pass", "pass"])
    max_drain_error = 0
    max_price_error = 0
    for index in range(1, len(env.steps)):
        previous = env.steps[index - 1][0].observation
        current = env.steps[index][0].observation
        step = int(previous.step)
        predicted = brain.town_drain(step, list(previous.town.unlocked_shops))
        for item in brain.PRODUCTS:
            actual = int(previous.market.inventory[item]) - int(current.market.inventory[item])
            max_drain_error = max(max_drain_error, abs(predicted[item] - actual))
            max_price_error = max(max_price_error, abs(brain.market_price(item, int(current.market.inventory[item])) - int(current.market.prices[item])))
    return max_drain_error, max_price_error


def main() -> None:
    results = []
    for seed in range(5):
        sale_revenue_error, sale_inventory_error = market_check(seed)
        town_drain_error, town_price_error = town_check(seed)
        row = {"seed": seed, "sale_revenue_error": sale_revenue_error,
               "sale_inventory_error": sale_inventory_error, "town_drain_error": town_drain_error,
               "town_price_error": town_price_error}
        assert all(value == 0 for key, value in row.items() if key != "seed")
        results.append(row)
    print(json.dumps({"seeds": 5, "max_sale_revenue_error": max(r["sale_revenue_error"] for r in results),
                      "max_sale_inventory_error": max(r["sale_inventory_error"] for r in results),
                      "max_town_drain_error": max(r["town_drain_error"] for r in results),
                      "max_town_price_error": max(r["town_price_error"] for r in results), "rows": results}, indent=2))


if __name__ == "__main__":
    main()
