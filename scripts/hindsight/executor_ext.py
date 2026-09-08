"""Backward-compatible plan-space extensions for the hindsight executor.

``load_agent`` delegates byte-for-byte v1 plans to solve.py's loader.  Extended
plans add daily labour, exact opening zones, capital-order priority, fertilizer
liquidation, and crop-specific maturity-day routing without changing the v1 keys.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from scripts.hindsight import solve

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PURCHASE_KINDS = ("animal", "land", "feed", "seed", "hire", "sell", "other")
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
TARGET_AGE = {"WHEAT": 4, "CARROT": 3, "TOMATO": 8,
              "STRAWBERRY": 10, "MELON": 10}
RATIO_PRODUCTS = ("MILK", "WOOL", "EGG", "STRAWBERRY", "MELON", "CARROT", "TOMATO")
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
# Cared output available at a steady-state collection.  These are deliberately the
# plan-space conversion rates requested by the experiment, rather than fitted sales.
ANIMAL_YIELD_PER_DAY = {"GOOSE": 4.0, "COW": 6.0, "SHEEP": 6.0}
# (occupied tile-days, units) for one fully serviced crop block.
CROP_CYCLE = {"STRAWBERRY": (17.0, 8.0), "MELON": (11.0, 6.0),
              "CARROT": (4.0, 4.0), "TOMATO": (12.0, 4.0)}
FINAL_STEP = 30 * 24 - 1


def _has_extensions(plan: dict[str, Any]) -> bool:
    return any(k in plan for k in ("hands_by_day", "purchase_order", "tile_blocks",
                                   "harvest_days", "fertilizer_mode", "routing_mode",
                                   "supply_ratio", "wheat_tiles"))


def _normalise_extensions(plan: dict[str, Any]) -> dict[str, Any]:
    p = copy.deepcopy(plan)
    if "hands_by_day" in p:
        days = [max(4, min(16, int(x))) for x in p["hands_by_day"]][:30]
        while len(days) < 30:
            days.append(days[-1] if days else 4)
        p["hands_by_day"] = days
    if "purchase_order" in p:
        order = [str(x).lower() for x in p["purchase_order"]]
        if order and order[0] == "legacy":
            p["purchase_order"] = ["legacy"]
        else:
            p["purchase_order"] = list(dict.fromkeys(x for x in order if x in PURCHASE_KINDS))
            p["purchase_order"] += [x for x in PURCHASE_KINDS if x not in p["purchase_order"]]
    if "tile_blocks" in p:
        p["tile_blocks"] = {c: str(v).lower() for c, v in p["tile_blocks"].items()
                            if c in CROPS and str(v).lower() in ("near", "middle", "far")}
    if "harvest_days" in p:
        modes = {}
        for c in CROPS:
            value = p["harvest_days"].get(c, False)
            modes[c] = value if value in (False, "opening", "all") else ("all" if value else False)
        p["harvest_days"] = modes
    if "fertilizer_mode" in p:
        mode = str(p["fertilizer_mode"]).lower()
        p["fertilizer_mode"] = mode if mode in ("legacy", "sell_all", "no_buy") else "legacy"
    if "routing_mode" in p:
        p["routing_mode"] = "nearest" if p["routing_mode"] == "nearest" else "legacy"
    if "supply_ratio" in p:
        supplied = p.get("supply_ratio", {}) or {}
        p["supply_ratio"] = {product: max(0.0, min(12.0, float(supplied.get(product, 0.0))))
                             for product in RATIO_PRODUCTS}
        p["wheat_tiles"] = max(0, min(100, int(p.get("wheat_tiles", 0))))
    return p


def _ticks(start: int, end: int, interval: int) -> int:
    """Number of inclusive interval boundaries in [start, end]."""
    if start > end:
        return 0
    return end // interval - (start - 1) // interval


def drain_forecast(obs: Any) -> dict[str, float]:
    """Forecast average town absorption per remaining day from ``obs.town``.

    At day 0 the unlock identities are unknowable: the observation contains an
    empty list and the seed is intentionally hidden.  Each unrevealed slot is
    therefore integrated over the engine's uniform eight-shop census.  Revealed
    instances (including duplicates) replace that expectation exactly.  Calling
    this every turn makes the forecast, and consequently capacity targets, react
    as soon as a shop appears.
    """
    products = tuple(x for x in solve.PRODUCTS if x != "FERTILIZER")
    step = int(obs.get("step", int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))))
    step = max(0, min(FINAL_STEP, step))
    shops = list((obs.get("town", {}) or {}).get("unlocked_shops", []) or [])
    remaining_days = max(1.0 / 24.0, (FINAL_STEP - step + 1) / 24.0)
    shop_ticks = _ticks(step, FINAL_STEP, 4)
    total = {product: float(_ticks(step, FINAL_STEP, 24)) for product in products}

    for shop in shops:
        items = _SHOP_PRODUCTS.get(str(shop), ())
        multiplier = 2 if len(items) == 1 else 1
        for product in items:
            total[product] += multiplier * shop_ticks

    # One shop unlocks at the end of days 2,5,...,23.  Its identity is not in the
    # observation until then, so use the exact uniform-with-replacement prior.
    expected_per_draw = {product: 0.0 for product in products}
    for items in _SHOP_PRODUCTS.values():
        multiplier = 2 if len(items) == 1 else 1
        for product in items:
            expected_per_draw[product] += multiplier / len(_SHOP_PRODUCTS)
    for index in range(len(shops), 8):
        unlock_step = 3 * (index + 1) * 24
        ticks = _ticks(max(step, unlock_step), FINAL_STEP, 4)
        for product in products:
            total[product] += ticks * expected_per_draw[product]
    return {product: total[product] / remaining_days for product in products}


def _revealed_drain_per_day(obs: Any) -> dict[str, float]:
    """Exact current run-rate: centre plus only shop identities already visible."""
    drain = {product: 1.0 for product in solve.PRODUCTS if product != "FERTILIZER"}
    for shop in list((obs.get("town", {}) or {}).get("unlocked_shops", []) or []):
        items = _SHOP_PRODUCTS.get(str(shop), ())
        multiplier = 2 if len(items) == 1 else 1
        for product in items:
            drain[product] += 6.0 * multiplier
    return drain


_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _ratio_capacity(plan: dict[str, Any], obs: Any) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    forecast = drain_forecast(obs)
    revealed = _revealed_drain_per_day(obs)
    ratios = plan["supply_ratio"]
    day = int(obs.get("day", 0))
    animal_targets = {}
    for animal, product in ANIMAL_PRODUCT.items():
        # Livestock cannot be sold, so unknown future shops are not capitalised.
        # A newly revealed shop raises this exact run-rate on the next call.
        required = int(math.ceil(revealed[product] * ratios[product] / ANIMAL_YIELD_PER_DAY[animal] - 1e-12))
        timing = plan["animals"][animal]
        if day < timing["start_day"] or timing["batch"] <= 0:
            ramp = 0
        else:
            ramp = (1 + (day - timing["start_day"]) // timing["interval"]) * timing["batch"]
        animal_targets[animal] = max(0, min(20, required, ramp))
    crop_targets = {"WHEAT": int(plan.get("wheat_tiles", 0))}
    for crop, (days, units) in CROP_CYCLE.items():
        crop_targets[crop] = max(0, min(100, int(math.ceil(
            forecast[crop] * ratios[crop] * days / units - 1e-12))))
    return animal_targets, crop_targets, forecast


def _xy_units(obs: Any) -> tuple[list[tuple[int, int]], list[dict[str, int]]]:
    player = int(obs["player"])
    farm = obs["farms"][player]
    positions = [tuple(farm["farmer"])] + [tuple(x) for x in farm.get("hands", [])]
    inventories = [dict(x or {}) for x in obs.get("private", {}).get("inventories", [])]
    inventories += [{} for _ in range(len(positions) - len(inventories))]
    return positions, inventories


def _step(pos: tuple[int, int], target: tuple[int, int]) -> list[str]:
    x, y = pos
    tx, ty = target
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return ["PASS"]


def _zone(pos: tuple[int, int]) -> str:
    # Geometry is explicit and stable across quadrants.  Distance 0/1/2... is from
    # the closest of the four shed-access squares.
    d = min(abs(pos[0] - x) + abs(pos[1] - y) for x, y in SHED_TILES)
    return "near" if d <= 2 else "middle" if d <= 5 else "far"


def _order_kind(order: list[Any]) -> str:
    op = order[0] if order else ""
    if op == "BUY_ANIMAL": return "animal"
    if op == "BUY_LAND": return "land"
    if op == "BUY_PRODUCT" and len(order) > 1 and order[1] == "WHEAT": return "feed"
    if op == "BUY_SEED": return "seed"
    if op == "HIRE": return "hire"
    if op == "SELL": return "sell"
    return "other"


def _prioritise_purchases(obs: Any, plan: dict[str, Any], orders: list[list[Any]],
                          ratio_targets: dict[str, int] | None = None) -> list[list[Any]]:
    if "purchase_order" not in plan or plan["purchase_order"] == ["legacy"]:
        return orders
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    augmented = list(orders)
    # The inherited coordinator can fill all ten slots with hires/sales before its
    # animal loop.  Reconstruct the due animal orders at hour zero so order priority
    # is genuine rather than merely a sort of the truncated list.
    if hour == 0 and day < 27:
        player = int(obs["player"])
        tiles = obs["farms"][player]["tiles"]
        shed = obs.get("private", {}).get("shed", {}) or {}
        invs = obs.get("private", {}).get("inventories", []) or []
        live = {a: 0 for a in solve.ANIMALS}
        for row in tiles:
            for tile in row:
                if isinstance(tile, dict) and tile.get("animal") in live:
                    live[tile["animal"]] += 1
        pending = {a: int(shed.get(a, 0)) + sum(int(inv.get(a, 0)) for inv in invs)
                   for a in solve.ANIMALS}
        if day == 0:
            existing = {o[1] for o in augmented if len(o) >= 2 and o[0] == "BUY_ANIMAL"}
            for animal in solve.ANIMALS:
                target = (ratio_targets or {}).get(animal, solve._animal_target(plan, animal, day))
                due = max(0, target - live[animal] - pending[animal])
                if due and animal not in existing:
                    augmented.append(["BUY_ANIMAL", animal, min(4, due)])
        if day == 0 and "supply_ratio" not in plan:
            # Opening weights are exact counts, but the inherited buyer requests up
            # to its global seed cap and skips an unaffordable crop batch wholesale.
            # Replace that request with the plan's 25 explicit opening assignments.
            seeds = obs.get("private", {}).get("seeds", {}) or {}
            augmented = [o for o in augmented if not (o and o[0] == "BUY_SEED")]
            for crop, want in plan["crop_mix"]["opening"].items():
                due = max(0, int(want) - int(seeds.get(crop, 0)))
                if due:
                    augmented.append(["BUY_SEED", crop, due])
    rank = {kind: i for i, kind in enumerate(plan["purchase_order"])}
    augmented.sort(key=lambda o: rank.get(_order_kind(o), len(rank)))
    # Avoid duplicate reconstructed/base animal orders and retain the engine cap.
    clean, animal_seen = [], set()
    for order in augmented:
        if order and order[0] == "BUY_ANIMAL":
            key = order[1]
            if key in animal_seen:
                continue
            animal_seen.add(key)
        clean.append(order)
    return clean[:10]


def _opening_blocks(obs: Any, plan: dict[str, Any], action: dict[str, Any]) -> None:
    blocks = plan.get("tile_blocks")
    if not blocks or int(obs.get("day", 0)) > 9:
        return
    positions, _ = _xy_units(obs)
    seeds = obs.get("private", {}).get("seeds", {}) or {}
    acts = [action.get("farmer", ["PASS"])] + list(action.get("hands", []))
    used: dict[str, int] = {}
    for i, (pos, act) in enumerate(zip(positions, acts)):
        if not act or act[0] != "PLANT":
            continue
        candidates = [c for c, zone in blocks.items() if zone == _zone(pos)]
        if not candidates:
            continue
        crop = max(candidates, key=lambda c: int(seeds.get(c, 0)) - used.get(c, 0))
        if int(seeds.get(crop, 0)) > used.get(crop, 0):
            acts[i] = ["PLANT", crop]
            used[crop] = used.get(crop, 0) + 1
    action["farmer"], action["hands"] = acts[0], acts[1:]


def _rush_harvest(obs: Any, plan: dict[str, Any], action: dict[str, Any]) -> None:
    enabled = [c for c, mode in plan.get("harvest_days", {}).items() if mode]
    if not enabled:
        return
    day = int(obs.get("day", 0))
    player = int(obs["player"])
    tiles = obs["farms"][player]["tiles"]
    crops_today = set()
    ready = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            crop = tile.get("crop")
            age = day - int(tile.get("planted_day", day))
            mode = plan["harvest_days"].get(crop, False)
            if crop in enabled and age >= TARGET_AGE[crop] and (mode != "opening" or int(tile.get("planted_day", -1)) == 0):
                crops_today.add(crop)
                if int(tile.get("yield_units", 0) or 0) > 0:
                    ready.append((x, y))
    if not crops_today:
        return
    positions, inventories = _xy_units(obs)
    acts = [action.get("farmer", ["PASS"])] + list(action.get("hands", []))
    available = set(range(len(positions)))
    dropping = {c: 0 for c in crops_today}
    # Harvested carriers have the first deadline: clear them into the shed, and add
    # the quantity to the same-turn sell order (unit actions precede market orders).
    for i in list(available):
        carried = sum(int(inventories[i].get(c, 0)) for c in crops_today)
        if carried <= 0:
            continue
        shed = min(SHED_TILES, key=lambda p: abs(p[0]-positions[i][0]) + abs(p[1]-positions[i][1]))
        if positions[i] == shed:
            acts[i] = ["DROP"]
            for c in crops_today:
                dropping[c] += int(inventories[i].get(c, 0))
        else:
            acts[i] = _step(positions[i], shed)
        available.remove(i)
    pending = set(ready)
    while available and pending:
        d, i, target = min((abs(positions[i][0]-p[0]) + abs(positions[i][1]-p[1]), i, p)
                           for i in available for p in pending)
        acts[i] = ["HARVEST"] if d == 0 else _step(positions[i], target)
        available.remove(i)
        pending.remove(target)
    action["farmer"], action["hands"] = acts[0], acts[1:]
    shed = obs.get("private", {}).get("shed", {}) or {}
    orders = [o for o in action.get("market", [])
              if not (len(o) >= 2 and o[0] == "SELL" and o[1] in crops_today)]
    for crop in crops_today:
        qty = int(shed.get(crop, 0)) + dropping[crop]
        if qty > 0:
            orders.insert(0, ["SELL", crop, qty])
    action["market"] = orders[:10]


def _cap_ratio_planting(obs: Any, targets: dict[str, int], action: dict[str, Any]) -> None:
    """Turn excess PLANT emissions into safe passes once a ratio block is full."""
    player = int(obs["player"])
    live = {crop: 0 for crop in targets}
    for row in obs["farms"][player]["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") in live:
                live[tile["crop"]] += 1
    acts = [action.get("farmer", ["PASS"])] + list(action.get("hands", []))
    for index, act in enumerate(acts):
        if act and act[0] == "PLANT":
            crop = act[1] if len(act) > 1 else ""
            if crop not in targets or live.get(crop, 0) >= targets.get(crop, 0):
                acts[index] = ["PASS"]
            else:
                live[crop] += 1
    action["farmer"], action["hands"] = acts[0], acts[1:]


def _ratio_structure_actions(obs: Any, animal_targets: dict[str, int], action: dict[str, Any]) -> None:
    """Ensure goose capacity is built as coops, not generic pastures."""
    farm = obs["farms"][int(obs["player"])]
    coops = 0
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "COOP":
                coops += 1
    need = max(0, animal_targets.get("GOOSE", 0) - coops)
    acts = [action.get("farmer", ["PASS"])] + list(action.get("hands", []))
    for i, act in enumerate(acts):
        if need and act and act[0] == "BUILD_PASTURE":
            acts[i] = ["BUILD_COOP"]
            need -= 1
    action["farmer"], action["hands"] = acts[0], acts[1:]


def _ratio_wheat_reserve(obs: Any, animal_targets: dict[str, int], action: dict[str, Any]) -> None:
    reserve = sum(animal_targets.values()) + 5
    shed = int(((obs.get("private", {}) or {}).get("shed", {}) or {}).get("WHEAT", 0))
    orders = []
    for order in action.get("market", []):
        if len(order) >= 3 and order[0] == "SELL" and order[1] == "WHEAT":
            qty = min(int(order[2]), max(0, shed - reserve))
            if qty:
                orders.append(["SELL", "WHEAT", qty])
        else:
            orders.append(order)
    action["market"] = orders


def load_agent(plan: dict[str, Any]):
    """Return a safe callable; v1-only plans use the original loader exactly."""
    if not _has_extensions(plan):
        return solve._load_parameterised_agent(plan)
    extended = _normalise_extensions(plan)
    legacy = {k: copy.deepcopy(v) for k, v in extended.items()
              if k not in ("hands_by_day", "purchase_order", "tile_blocks",
                           "harvest_days", "fertilizer_mode", "routing_mode",
                           "supply_ratio", "wheat_tiles")}
    if "hands_by_day" in extended:
        legacy["hands_by_5day"] = [max(extended["hands_by_day"][i:i+5]) for i in range(0, 30, 5)]
    base = solve._load_parameterised_agent(legacy)
    brain = next((cell.cell_contents for cell in (base.__closure__ or ())
                  if hasattr(cell.cell_contents, "DayPlan")), None)
    if brain is None:
        raise RuntimeError("could not locate inherited brain module")
    ratio_state: dict[str, Any] = {}
    if "supply_ratio" in extended:
        class RatioSeason:
            @staticmethod
            def plan_day(view):
                targets = ratio_state["animals"]
                crop_targets = ratio_state["crops"]
                desired = {crop: float(target) for crop, target in crop_targets.items()
                           if target > 0 and view.day <= extended["last_plant_day"].get(crop, -1)}
                total = sum(desired.values()) or 1.0
                weights = {crop: target / total for crop, target in desired.items()}
                return brain.DayPlan(
                    "opening" if view.day <= 9 else "days_10_16" if view.day <= 16 else
                    "days_17_22" if view.day <= 22 else "days_23_26",
                    extended["hands_by_day"][min(29, view.day)] if "hands_by_day" in extended
                    else extended["hands_by_5day"][min(5, view.day // 5)],
                    4, tuple(sorted(weights.items())), tuple(sorted(targets.items())),
                    tuple(sorted(brain.RESERVATION.items())), view.day >= brain.LIQUIDATE_DAY,
                    extended["cash_reserve"])

        brain.register_module("season", "ratio", RatioSeason)
        brain.config["modules"]["season"] = "ratio"

    if extended.get("routing_mode") == "nearest":

        def nearest_assign(positions, inventories, tasks):
            actions, available, pending = {}, set(range(len(positions))), list(range(len(tasks)))
            while available and pending:
                choices = []
                for ti in pending:
                    priority, target, _action, need, skip = tasks[ti]
                    for i in available:
                        if need is not None and inventories[i].get(need, 0) <= 0: continue
                        if skip is not None and inventories[i].get(skip, 0) > 0: continue
                        distance = abs(positions[i][0]-target[0]) + abs(positions[i][1]-target[1])
                        choices.append((priority, distance, i, ti))
                if not choices: break
                _priority, _distance, i, ti = min(choices)
                target = tasks[ti][1]
                actions[i] = list(tasks[ti][2]) if positions[i] == target else brain.LaborB1._move(positions[i], target)
                available.remove(i); pending.remove(ti)
            return actions

        brain.LaborB1._assign = staticmethod(nearest_assign)

    def candidate(obs: Any, config: Any = None):
        try:
            ratio_targets = None
            crop_targets = None
            if "supply_ratio" in extended:
                ratio_targets, crop_targets, forecast = _ratio_capacity(extended, obs)
                ratio_state.update(animals=ratio_targets, crops=crop_targets, forecast=forecast)
            action = base(obs, config)
            if ratio_targets is not None:
                _ratio_structure_actions(obs, ratio_targets, action)
                _ratio_wheat_reserve(obs, ratio_targets, action)
            if "hands_by_day" in extended:
                target = extended["hands_by_day"][min(29, int(obs.get("day", 0)))]
                have = len(obs["farms"][int(obs["player"])].get("hands", []))
                allowed = max(0, target - have)
                kept, hires = [], 0
                for order in action.get("market", []):
                    if order and order[0] == "HIRE":
                        if hires >= allowed:
                            continue
                        hires += 1
                    kept.append(order)
                action["market"] = kept
            _opening_blocks(obs, extended, action)
            if crop_targets is not None:
                _cap_ratio_planting(obs, crop_targets, action)
            _rush_harvest(obs, extended, action)
            if extended.get("fertilizer_mode") in ("sell_all", "no_buy"):
                action["market"] = [o for o in action.get("market", [])
                                    if not (len(o) >= 2 and o[0] == "BUY_PRODUCT" and o[1] == "FERTILIZER")]
            if extended.get("fertilizer_mode") == "sell_all":
                qty = int((obs.get("private", {}).get("shed", {}) or {}).get("FERTILIZER", 0))
                action["market"] = [o for o in action.get("market", [])
                                    if not (len(o) >= 2 and o[0] == "SELL" and o[1] == "FERTILIZER")]
                if qty:
                    action["market"].insert(0, ["SELL", "FERTILIZER", qty])
            action["market"] = _prioritise_purchases(obs, extended, action.get("market", []), ratio_targets)
            return action
        except Exception:
            try:
                n = len(obs["farms"][int(obs["player"])].get("hands", []))
            except Exception:
                n = 0
            return {"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": []}

    return candidate
