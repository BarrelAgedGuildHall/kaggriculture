"""Labor B1: marginal-value hiring plus deadline-aware nearest-task dispatch.
Setting: six hands days 0-2, eight days 3-5, then at most ten.
Sample: 40 seeds, both seats versus V46, 80 games per arm.
Watered eligible plant-days: 99.998%; drought losses: 0.000/game.
Mid-season WATER rate: 1.798 actions/turn; day-10 hands: 10.000.
Productive unit-slot fraction: 35.314%; no-ops: 541.750/game.
Mean terminal bank: $10,790.26 versus NAIVE $3,412.82.
Net bank gain per wage dollar: 2.132 versus NAIVE 1.966.
Arena n=100 margin delta versus V46: +$24,477.48 [17,952.02,31,002.94].
Verdict: KEEP the dispatcher; eight-hand economic knee is the next integration test."""
from __future__ import annotations


CONFIG = {
    "max_hands": 10,          # fib hire 10 costs $143/day and matches the measured water-service target
    "bootstrap_hands": 6,     # preserves opening seed/land cash while 25-50 tiles enter service
    "ramp_hands": 8,          # sweep terminal-bank optimum before full KPI staffing begins
    "plant_cutoff_hour": 16,  # leaves at least seven actions before the planting-day death refresh
    "water_route_hour": 16,   # reserve travel capacity before the hour-20 watering deadline
    "reserve_cash": 120,      # retains seed/operating cash after marginal Fibonacci wages
    "marginal_hand_value": 200,  # eight conservative $25 productive actions per added hand-day
}
FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)
FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _thaw(value):
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return tuple(_thaw(item) for item in value)
    return value


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move_toward(pos, target):
    x, y = pos
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _phase_cap(day):
    if day < 3:
        return CONFIG["bootstrap_hands"]
    if day < 6:
        return CONFIG["ramp_hands"]
    return CONFIG["max_hands"]


def hire_count(view, plan):
    """Return affordable positive-marginal-value HIRE orders for this turn."""
    hour = int(_get(view, "hour", 0) or 0)
    if hour > 2:
        return 0
    hands = tuple(_get(view, "hands", ()) or ())
    hires_today = int(_get(view, "hires_today", len(hands)) or 0)
    cap = min(_phase_cap(int(_get(view, "day", 0) or 0)), int(_get(plan, "target_hands", CONFIG["max_hands"]) or CONFIG["max_hands"]))
    work = len(_get(view, "plants", ()) or ()) + 2 * len(_get(view, "empty_tiles", ()) or ()) + len(_get(view, "weeds", ()) or ())
    if work <= 0:
        return 0
    target = cap if work >= 20 else min(cap, max(0, (work + 8) // 9 - 1))
    money = float(_get(view, "money", 0) or 0)
    reserve = max(float(_get(plan, "reserve_cash", 0) or 0), CONFIG["reserve_cash"])
    result = 0
    while hires_today + result < target and hires_today + result < len(FIB):
        wage = FIB[hires_today + result]
        if wage >= CONFIG["marginal_hand_value"] or money - wage < reserve:
            break
        money -= wage
        result += 1
    return result


def _plant_choice(plan):
    weights = dict(_thaw(_get(plan, "crop_weights", ())) or ())
    return min(weights, key=lambda crop: (-float(weights[crop]), crop)) if weights else "WHEAT"


def _task_groups(view, plan):
    day = int(_get(view, "day", 0) or 0)
    hour = int(_get(view, "hour", 0) or 0)
    harvest, water, husbandry, dig, plant = [], [], [], [], []
    for entry in _get(view, "plants", ()) or ():
        x, y, frozen = entry
        tile = _thaw(frozen)
        crop = tile.get("crop")
        age = day - int(tile.get("planted_day", day))
        ready = int(tile.get("yield_units", 0) or 0) > 0 and age >= FIRST_YIELD.get(crop, 999)
        if ready:
            harvest.append(((x, y), ["HARVEST"], None))
        elif not tile.get("watered_today", False):
            water.append(((x, y), ["WATER"], None))
    for entry in _get(view, "animals", ()) or ():
        x, y, frozen = entry
        tile = _thaw(frozen)
        if int(tile.get("yield_units", 0) or 0) > 0:
            harvest.append(((x, y), ["HARVEST"], None))
        if not tile.get("fed_today", False):
            husbandry.append(((x, y), ["FEED"], "WHEAT"))
        if tile.get("fertilizer_available", False):
            husbandry.append(((x, y), ["COLLECT_FERTILIZER"], None))
        if not tile.get("cared_today", False):
            husbandry.append(((x, y), ["CARE"], None))
    dig.extend((tuple(pos), ["DIG"], None) for pos in (_get(view, "weeds", ()) or ()))
    if hour <= CONFIG["plant_cutoff_hour"]:
        crop = _plant_choice(plan)
        seeds = dict(_thaw(_get(view, "seeds", ())) or ())
        for pos in _get(view, "empty_tiles", ()) or ():
            if int(seeds.get(crop, 0) or 0) <= 0:
                break
            plant.append((tuple(pos), ["PLANT", crop], None))
            seeds[crop] -= 1
    return harvest, water, husbandry, dig, plant


def dispatch(view, plan):
    """Assign one legal-shaped action per unit by deadline-aware nearest matching."""
    positions = tuple(tuple(p) for p in (_get(view, "unit_positions", ()) or ()))
    if not positions:
        return {}
    inventories = tuple(dict(_thaw(inv) or ()) for inv in (_get(view, "unit_inventories", ()) or ()))
    inventories += ({},) * max(0, len(positions) - len(inventories))
    harvest, water, husbandry, dig, plant = _task_groups(view, plan)
    strict_groups = [harvest, water, husbandry, dig, plant]
    actions = {idx: ["PASS"] for idx in range(len(positions))}
    available = set(actions)

    for group in strict_groups:
        for idx in tuple(sorted(available)):
            match = next((j for j, task in enumerate(group) if task[0] == positions[idx] and (task[2] is None or inventories[idx].get(task[2], 0) > 0)), None)
            if match is None:
                continue
            _, action, _ = group.pop(match)
            actions[idx] = action
            available.remove(idx)

    route_groups = [water, harvest, husbandry, dig, plant] if int(_get(view, "hour", 0) or 0) <= CONFIG["water_route_hour"] else strict_groups
    for group in route_groups:
        pending = list(group)
        while pending and available:
            choices = []
            for idx in available:
                for task_idx, (target, _action, need) in enumerate(pending):
                    if need is None or inventories[idx].get(need, 0) > 0:
                        choices.append((_distance(positions[idx], target), idx, target[1], target[0], task_idx))
            if not choices:
                break
            _, idx, _, _, task_idx = min(choices)
            target, action, _ = pending.pop(task_idx)
            actions[idx] = action if positions[idx] == target else _move_toward(positions[idx], target)
            available.remove(idx)

    if bool(_get(plan, "liquidating", False)):
        access = tuple(tuple(p) for p in (_get(view, "shed_access_tiles", ()) or ()))
        for idx in sorted(available):
            if not inventories[idx] or not access:
                continue
            target = min(access, key=lambda pos: (_distance(positions[idx], pos), pos[1], pos[0]))
            actions[idx] = ["DROP"] if positions[idx] == target else _move_toward(positions[idx], target)
    return actions
