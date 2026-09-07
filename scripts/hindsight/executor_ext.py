"""Backward-compatible plan-space extensions for the hindsight executor.

``load_agent`` delegates byte-for-byte v1 plans to solve.py's loader.  Extended
plans add daily labour, exact opening zones, capital-order priority, fertilizer
liquidation, and crop-specific maturity-day routing without changing the v1 keys.
"""
from __future__ import annotations

import copy
from typing import Any

from scripts.hindsight import solve

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PURCHASE_KINDS = ("animal", "land", "feed", "seed", "hire", "sell", "other")
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
TARGET_AGE = {"WHEAT": 4, "CARROT": 3, "TOMATO": 8,
              "STRAWBERRY": 10, "MELON": 10}


def _has_extensions(plan: dict[str, Any]) -> bool:
    return any(k in plan for k in ("hands_by_day", "purchase_order", "tile_blocks",
                                   "harvest_days", "fertilizer_mode", "routing_mode"))


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
    return p


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


def _prioritise_purchases(obs: Any, plan: dict[str, Any], orders: list[list[Any]]) -> list[list[Any]]:
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
                target = solve._animal_target(plan, animal, day)
                due = max(0, target - live[animal] - pending[animal])
                if due and animal not in existing:
                    augmented.append(["BUY_ANIMAL", animal, min(4, due)])
        if day == 0:
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


def load_agent(plan: dict[str, Any]):
    """Return a safe callable; v1-only plans use the original loader exactly."""
    if not _has_extensions(plan):
        return solve._load_parameterised_agent(plan)
    extended = _normalise_extensions(plan)
    legacy = {k: copy.deepcopy(v) for k, v in extended.items()
              if k not in ("hands_by_day", "purchase_order", "tile_blocks",
                           "harvest_days", "fertilizer_mode", "routing_mode")}
    if "hands_by_day" in extended:
        legacy["hands_by_5day"] = [max(extended["hands_by_day"][i:i+5]) for i in range(0, 30, 5)]
    base = solve._load_parameterised_agent(legacy)
    if extended.get("routing_mode") == "nearest":
        brain = base.__closure__[0].cell_contents

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
            action = base(obs, config)
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
            action["market"] = _prioritise_purchases(obs, extended, action.get("market", []))
            return action
        except Exception:
            try:
                n = len(obs["farms"][int(obs["player"])].get("hands", []))
            except Exception:
                n = 0
            return {"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": []}

    return candidate
