"""Extract and document the fixed seed-3910 instance from engine 1.32.7."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._engine import engine_dir, engine_version, import_engine

OUT = ROOT / "data" / "results" / "fixed-game" / "3910"
SEED = 3910
SHOP_UNLOCKS = [
    (3, "YARN_STORE"),
    (6, "PET_CAFE"),
    (9, "YARN_STORE"),
    (12, "PIZZA_SHOP"),
    (15, "BAKERY"),
    (18, "PET_CAFE"),
    (21, "PIZZA_SHOP"),
    (24, "SMOOTHIE_SHOP"),
]


def _shape(func: str, x: float, T: float) -> float:
    if func == "linear": return x
    if func == "sq": return x * x
    if func == "sqrt": return math.sqrt(x)
    if func == "log": return math.log1p(x)
    if func == "log10": return math.log10(1 + x)
    if func == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    raise ValueError(func)


def _daily_drain(kg: Any) -> list[dict[str, Any]]:
    active: list[str] = []
    unlock = dict(SHOP_UNLOCKS)
    rows = []
    for day in range(30):
        if day in unlock:
            active.append(unlock[day])
        drain = {p: (0 if p == "FERTILIZER" else 1) for p in kg.PRODUCTS}
        for shop in active:
            products = kg.SHOPS[shop]
            multiplier = 2 if len(products) == 1 else 1
            for p in products:
                drain[p] += 6 * multiplier
        rows.append({"day": day, "active_shops": list(active), "drain": drain})
    return rows


def build() -> dict[str, Any]:
    import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    source = engine_dir() / "kaggriculture.py"
    cfg = {
        "episodeSteps": 720,
        "actTimeout": 1,
        "runTimeout": 1200,
        "boardSize": 10,
        "startingMoney": 3000,
        "maxMarketOrdersPerTurn": 10,
        "turnsPerDay": 24,
        "shedCapacity": 100,
        "weedSpawnChance": 0.005,
        "townShopUnlockInterval": 3,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 24,
        "farmHandCostMult": 1,
        "marketParams": {},
    }
    tiles = []
    for y in range(10):
        for x in range(10):
            q = kg._quadrant_of(x, y, 10)
            tiles.append({
                "x": x, "y": y, "quadrant": q,
                "initial_type": "EMPTY" if q == "NW" else "LOCKED",
                "initially_owned": q == "NW",
                "land_unlock_cost": {"NW": 0, "NE": 1000, "SW": 2000, "SE": 4000}[q],
            })

    prices = {}
    for product, p0 in kg.MARKET_PARAMS.items():
        p = dict(p0)
        below_amp = p["below_target"] * p["base"] / _shape(p["below_func"], p["T"], p["T"])
        above_amp = p["above_target"] * p["base"] / _shape(p["above_func"], p["T"], p["T"])
        p.update({
            "floor": kg.PRICE_FLOOR,
            "below_amplitude": below_amp,
            "above_amplitude": above_amp,
            "exact_formula": "max(1, round(base + below_amplitude*f(I0-inventory))) if inventory<I0; max(1, round(base - above_amplitude*f(inventory-I0))) otherwise",
            "shape_definitions": {"linear": "x", "sq": "x^2", "sqrt": "sqrt(x)", "log": "ln(1+x)", "log10": "log10(1+x)", "hinge": "u + 8*max(0,u-1)^2, u=x/T"},
        })
        prices[product] = p

    crops = {}
    for name, raw in kg.CROPS.items():
        c = dict(raw)
        c.update({
            "structure_cost": 0,
            "plant_action_cost_turns": 1,
            "water_action_cost_turns": 1,
            "harvest_action_cost_turns": 1,
            "fertilize_action_cost_turns": 1,
            "water_survival": "planting day starts unwatered; two consecutive unwatered end-of-days turns it into WEED",
            "fertilizer_effect": "one unit covers current day through day+2; on a watered bonus/production day adds one extra yield unit",
            "rot": "one-time: decay begins (planted_day+max_yield_day+1)*24; ongoing: one day after fourth scheduled production; then -1 held unit every 2 turns until WEED",
        })
        crops[name] = c

    animals = {}
    for name, raw in kg.ANIMALS.items():
        a = dict(raw)
        cared_first = min(int(raw["max_held"]), int(raw["first_yield_day"]))
        cared_steady = min(int(raw["max_held"]), 1 + int(raw["interval"]))
        a.update({
            "structure_cost": 0,
            "build_action_cost_turns": 1,
            "pickup_action_cost_turns": 1,
            "place_action_cost_turns": 1,
            "feed_per_day": {"WHEAT": 1},
            "feed_action_cost_turns": 1,
            "care_action_cost_turns": 1,
            "harvest_action_cost_turns": 1,
            "fertilizer_per_surviving_animal_per_day": 1,
            "fertilizer_storage_on_tile": 1,
            "care_bonus": "fed+cared end-of-day banks +1; all banked bonus is added to next fed scheduled yield then reset, subject to max_held",
            "fed_and_cared_yield_ramp_if_harvested_each_tick": {
                "first_production_units": cared_first,
                "later_production_units": cared_steady,
                "explanation": "production consumes care bank before the current day's care is banked",
            },
            "escape": "two consecutive unfed end-of-days removes animal and leaves its structure",
        })
        animals[name] = a

    drains = _daily_drain(kg)
    result = {
        "schema_version": 1,
        "seed": SEED,
        "seat": 0,
        "opponent": "pass",
        "engine": {
            "package": "kaggle-environments",
            "version": engine_version(),
            "environment": "kaggriculture",
            "module_version": "0.1.0",
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "ground_truth": True,
        },
        "configuration": cfg,
        "map": {
            "width": 10, "height": 10,
            "coordinate_system": "(x,y), origin NW, x east, y south",
            "quadrants": {"NW": {"owned": True, "price": 0}, "NE": {"owned": False, "price": 1000}, "SW": {"owned": False, "price": 2000}, "SE": {"owned": False, "price": 4000}},
            "land_unlock_order": ["NE", "SW", "SE"],
            "shed": {"type": "non-tile central facility", "access_tiles": [[4, 4], [5, 4], [4, 5], [5, 5]], "capacity": 100},
            "market": {"type": "non-spatial; orders require no unit position or unit action"},
            "water": {"type": "non-spatial; WATER requires only standing on the plant and consumes no inventory"},
            "tiles": tiles,
        },
        "calendar": {
            "days": 30, "turns_per_day": 24, "recorded_steps": 720, "action_transitions": 719,
            "day_boundary_after_hours": 24,
            "per_turn_order": ["seat unit actions", "market queues", "town consumption if cadence fires", "plant decay", "end-of-day refresh if hour 23", "advance day/hour"],
            "end_of_day_order": ["refresh plants", "refresh animals", "roll weeds on each empty unlocked tile", "drop inventories to shed (overflow discarded)", "despawn hands/reset farmer", "unlock shop on days 3,6,...,24"],
        },
        "town": {
            "shops": copy_shops(kg.SHOPS),
            "shop_unlocks": [{"day": d, "shop": s, "products": list(kg.SHOPS[s])} for d, s in SHOP_UNLOCKS],
            "draw_with_replacement": True,
            "shop_ticks_per_day": 6,
            "single_product_shop_multiplier": 2,
            "center_products": list(kg.TOWN_CENTER_PRODUCTS),
            "center_units_per_day": 1,
            "daily_drain": drains,
            "rng_coupling": "_end_of_day seeds random.Random((seed*1_000_003)^day), passes it through _spawn_weeds (one draw per empty unlocked tile), then rng.choice(sorted(SHOPS)); occupancy changes RNG position before choice",
            "pin_semantics": "run original end-of-day including weed RNG and original choice, then replace only a newly appended shop with this list's shop for that unlock day",
        },
        "market": {
            "initial_inventory_each_product": 10000,
            "starting_prices_are_base": True,
            "products": prices,
            "sales_at_floor_do_not_add_supply": True,
            "buyable_products": ["WHEAT", "FERTILIZER"],
            "order_slots_per_turn": 10,
            "quantity_per_order": "unbounded integer, processed one unit at a time until unaffordable/unavailable",
            "seat_resolution": "unit lockstep; both seats quote against same pre-commit inventory",
        },
        "crops": crops,
        "animals": animals,
        "labour": {
            "permanent_farmer": 1,
            "hands_ephemeral_daily": True,
            "hire_order_is_market_order": True,
            "hire_cost_by_zero_based_index": [kg._hire_cost(i) for i in range(20)],
            "hire_cost_formula": "fib(index), sequence 1,1,2,3,5,...; index resets daily",
            "turns_per_unit_per_day": 24,
            "movement_cost_per_orthogonal_tile_turns": 1,
            "non_movement_unit_action_cost_turns": 1,
            "market_order_unit_action_cost_turns": 0,
            "spawn_tiles_nwse": [[4, 4], [5, 4], [4, 5], [5, 5]],
        },
        "starting_money": 3000,
        "shed_capacity": 100,
    }
    return result


def copy_shops(shops: dict[str, list[str]]) -> dict[str, list[str]]:
    return {k: list(v) for k, v in shops.items()}


def markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Fixed Kaggriculture instance: seed 3910", "",
        f"Engine: kaggle-environments {data['engine']['version']} / Kaggriculture {data['engine']['module_version']}. Source SHA-256: `{data['engine']['source_sha256']}`.", "",
        "Seat 0, opponent `pass`, $3,000 starting bank, 30 days × 24 turns. The framework stores 720 observations (step 0 is initial), hence 719 action transitions.", "",
        "## Pinned town", "",
        "| Day | Shop | Products |", "|---:|---|---|",
    ]
    for row in data["town"]["shop_unlocks"]:
        lines.append(f"| {row['day']} | {row['shop']} | {', '.join(row['products'])} |")
    lines += ["", "Daily drain (town center plus all shops active that day):", "", "| Day | " + " | ".join(data["market"]["products"]) + " |", "|---:|" + "---:|" * len(data["market"]["products"])]
    for row in data["town"]["daily_drain"]:
        lines.append("| " + str(row["day"]) + " | " + " | ".join(str(row["drain"][p]) for p in data["market"]["products"]) + " |")
    lines += [
        "", "Fertilizer has no town drain. Every other product has center drain 1/day. Each active multi-product shop adds 6/day per listed product; each single-product shop adds 12/day.", "",
        "### RNG coupling and pin", "",
        data["town"]["rng_coupling"] + ".", "",
        data["town"]["pin_semantics"] + ". This preserves weeds and every other rule while making shops action-independent.", "",
        "## Map", "",
        "Coordinates use `(x,y)`, origin northwest. NW (x=0..4,y=0..4) is initially owned. NE costs $1,000, SW $2,000, SE $4,000 and must be bought in that order. Each is 25 tiles. The central shed is not a tile; access tiles are (4,4), (5,4), (4,5), (5,5). Market and water are non-spatial.", "",
        "```text", "      x=0 1 2 3 4 | 5 6 7 8 9", "y=0..4   NW owned | NE locked ($1,000)", "          ---------+------------------", "y=5..9  SW locked  | SE locked", "         ($2,000)  | ($4,000)", "```", "",
        "Every tile is enumerated in `instance.json` with coordinate, quadrant, initial type/ownership, and quadrant unlock price.", "",
        "## Calendar and action economy", "",
        "Within a turn the engine applies unit actions, processes up to 10 market-order slots (bulk quantities are allowed), applies town consumption on its cadence, decays plants, then performs the day refresh after hour 23. At refresh it updates crops and animals, rolls weeds, drops all carried inventory into the 100-unit shed (discarding overflow), removes hired hands, respawns the farmer, resets daily hire count, then unlocks a shop when due.", "",
        "One farmer is permanent. Each hand lasts one day and also gets 24 actions. Daily hires cost Fibonacci(index): " + ", ".join(str(x) for x in data["labour"]["hire_cost_by_zero_based_index"][:15]) + ". One orthogonal move or farm operation costs one unit turn; market orders cost no unit turn. Buildings are free but BUILD costs one turn.", "",
        "## Crops", "", "| Crop | Seed | First yield age | Max-yield age | Interval | Cap | Ongoing |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, c in data["crops"].items():
        lines.append(f"| {name} | {c['seed']} | {c['first_yield_day']} | {c['max_yield_day']} | {c['interval']} | {c['max_yield']} | {c['ongoing']} |")
    lines += [
        "", "Planting starts with one missed-water count; a second consecutive missed end-of-day turns it into a weed. One-time crops begin at 1 unit and gain on watered days in the latter half of the growth window (fertilizer makes the gain 2), capped at max_yield. Ongoing crops schedule up to four productions; base is 1 and a currently fertilized, watered production is 2. Fertilizer lasts the application day plus two. At lifespan, held yield decays by one every two turns until weed.", "",
        "## Animals", "", "| Animal | Buy | Structure | Structure $ | First yield age | Interval | Held cap | Product |", "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for name, a in data["animals"].items():
        lines.append(f"| {name} | {a['cost']} | {a['structure']} | {a['structure_cost']} | {a['first_yield_day']} | {a['interval']} | {a['max_held']} | {a['product']} |")
    lines += [
        "", "Each animal consumes one wheat on a FEED action per serviced day; two consecutive missed feeds makes it escape. Fed+CARE banks +1 for the next fed production, where the whole bank joins base yield subject to the held cap. If harvested each tick and fed+cared daily, first/later tick yields are goose 4/2, cow 6/3, sheep 6/4 (production consumes the old care bank before today's care is banked). Every surviving animal exposes one fertilizer after each day refresh; uncollected fertilizer does not stack.", "",
        "## Exact price functions", "",
        "For inventory `I`, `x=|I-10000|`. Below I0: `round(base + amp_below*f_below(x))`; at/above I0: `round(base - amp_above*f_above(x))`; then floor at $1. `amp = target*base/f(T)`. Hinge uses `u=x/T; f=u+8*max(0,u-1)^2`. Python's integer rounding is the engine rounding.", "",
        "| Product | Base | T | Scarcity shape/target/amp | Glut shape/target/amp |", "|---|---:|---:|---|---|",
    ]
    for name, p in data["market"]["products"].items():
        lines.append(f"| {name} | {p['base']} | {p['T']} | {p['below_func']} / {p['below_target']} / {p['below_amplitude']:.9g} | {p['above_func']} / {p['above_target']} / {p['above_amplitude']:.9g} |")
    lines += ["", "Sales at $1 pay $1 but do not add market inventory. Only wheat and fertilizer can be bought as products. Seeds cost the crop's seed price; animals cost the table price.", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build()
    (OUT / "instance.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (OUT / "instance.md").write_text(markdown(data), encoding="utf-8")
    print(OUT / "instance.json")


if __name__ == "__main__":
    main()
