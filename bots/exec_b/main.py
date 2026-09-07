"""Kaggriculture Design B: per-turn geometric min-cost executor.

The economic plan is fixed.  Every observation is converted to deadline-bearing
jobs and the whole workforce is reassigned from exact current coordinates.  No
opponent action, replay, or persistent route tape is used.
"""

import math
import time


# The sole strategy input is the seed-3910 hindsight plan, embedded verbatim.
PLAN = {
    "crop_mix": {
        "opening": {"WHEAT": 25, "CARROT": 0, "MELON": 0, "STRAWBERRY": 0},
        "days_10_16": {"WHEAT": 55, "CARROT": 0, "MELON": 20, "STRAWBERRY": 25},
        "days_17_22": {"WHEAT": 50, "CARROT": 0, "MELON": 25, "STRAWBERRY": 25},
        "days_23_26": {"WHEAT": 74, "CARROT": 19, "MELON": 6, "STRAWBERRY": 1},
    },
    "land_days": {"NE": 30, "SE": 30, "SW": 30},
    "animals": {
        "COW": {"batch": 1, "interval": 1, "start_day": 0, "total": 14},
        "GOOSE": {"batch": 0, "interval": 7, "start_day": 30, "total": 0},
        "SHEEP": {"batch": 1, "interval": 1, "start_day": 0, "total": 11},
    },
    "cash_reserve": 190,
    "hands_by_5day": [5, 6, 9, 10, 10, 8],
    "fertilize": {"CARROT": True, "MELON": True, "STRAWBERRY": True, "TOMATO": True, "WHEAT": False},
    "last_plant_day": {"CARROT": 14, "MELON": 19, "STRAWBERRY": 15, "TOMATO": 14, "WHEAT": 29},
    "sell_policy": {"CARROT": "metered", "EGG": "metered", "FERTILIZER": "immediate", "MELON": "immediate", "MILK": "metered", "STRAWBERRY": "metered", "TOMATO": "immediate", "WHEAT": "immediate", "WOOL": "immediate"},
}

TURNS_PER_DAY = 24
FINAL_DAY = 29
MAX_ORDERS = 10
SHED_CAP = 100
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
SAFE = ["PASS"]
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROP_NAMES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_NAMES = ("GOOSE", "COW", "SHEEP")

CROPS = {
    "WHEAT": {"seed": 10, "first": 2, "last": 4, "ongoing": False},
    "CARROT": {"seed": 20, "first": 2, "last": 3, "ongoing": False},
    "TOMATO": {"seed": 50, "first": 8, "last": 10, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "last": 16, "ongoing": True},
    "MELON": {"seed": 80, "first": 10, "last": 10, "ongoing": False},
}
WATER_AGES = {
    "WHEAT": (0, 2, 3, 4), "CARROT": (0, 2, 3),
    "TOMATO": (0, 2, 4, 6, 7, 8, 9, 10),
    "STRAWBERRY": (0, 2, 4, 6, 8, 9, 11, 13, 15),
    "MELON": (0, 2, 4, 6, 7, 8, 9, 10),
}
FERT_AGES = {"WHEAT": (2,), "CARROT": (1,), "TOMATO": (7, 10), "STRAWBERRY": (9, 13), "MELON": (7,)}
ONGOING_HARVEST_AGES = {"TOMATO": (8, 9, 10, 11), "STRAWBERRY": (10, 12, 14, 16)}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "product": "EGG", "first": 4, "interval": 1, "max_held": 4},
    "COW": {"cost": 400, "structure": "PASTURE", "product": "MILK", "first": 8, "interval": 2, "max_held": 6},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL", "first": 6, "interval": 3, "max_held": 6},
}
SHOPS = {
    "BAKERY": ("EGG", "WHEAT"), "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"), "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"), "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

_FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)
_TIMES = []


def town_drain(step, shops):
    out = {p: 0 for p in PRODUCTS}
    if step % 4 == 0:
        for shop in shops:
            ps = SHOPS.get(shop, ())
            mul = 2 if len(ps) == 1 else 1
            for p in ps:
                out[p] += mul
    if step % 24 == 0:
        for p in PRODUCTS[:-1]:
            out[p] += 1
    return out


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest_shed(pos):
    return min(SHED_TILES, key=lambda p: (_dist(pos, p), p))


def _move(pos, target):
    x, y = pos
    tx, ty = target
    # Alternate the primary axis by parity to spread units without losing distance.
    if (x + y + tx + ty) & 1:
        if x < tx: return ["EAST"]
        if x > tx: return ["WEST"]
        if y < ty: return ["SOUTH"]
        if y > ty: return ["NORTH"]
    else:
        if y < ty: return ["SOUTH"]
        if y > ty: return ["NORTH"]
        if x < tx: return ["EAST"]
        if x > tx: return ["WEST"]
    return SAFE


def _animal_target(kind, day):
    cfg = PLAN["animals"][kind]
    if cfg["batch"] <= 0 or day < cfg["start_day"]:
        return 0
    return min(cfg["total"], (1 + (day - cfg["start_day"]) // cfg["interval"]) * cfg["batch"])


def _phase(day):
    if day <= 9: return "opening"
    if day <= 16: return "days_10_16"
    if day <= 22: return "days_17_22"
    return "days_23_26"


def _owned(tile):
    return tile != "LOCKED"


def _layout_rank(pos, crop=None):
    """Exact deterministic blocks: structures inward; crop blocks fan outward."""
    x, y = pos
    d = min(_dist(pos, s) for s in SHED_TILES)
    if crop == "WHEAT":
        return (d, x + y, y, x)
    if crop == "MELON":
        return (-x, y, d, x)
    if crop == "STRAWBERRY":
        return (-y, x, d, y)
    if crop == "CARROT":
        return (x, -y, d, x)
    return (d, y, x)


def _has(inv, item):
    return int(inv.get(item, 0) or 0) > 0


def _cargo_value(inv):
    values = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
    return sum(int(inv.get(k, 0) or 0) * v for k, v in values.items())


def _job(target, action, deadline, value, need=None, tag="", owner=None):
    return {"target": target, "action": action, "deadline": deadline,
            "value": value, "need": need, "tag": tag, "owner": owner}


def _eligible(inv, job):
    if job["tag"] == "pickup_feed" and _has(inv, "WHEAT"):
        return False
    need = job["need"]
    if need is None:
        return True
    if isinstance(need, tuple):
        return any(_has(inv, x) for x in need)
    return _has(inv, need)


def _assign(positions, inventories, jobs, hour):
    """Global greedy min-cost matching followed by all improving pair swaps."""
    n = len(positions)
    assignment = [None] * n
    free_u = set(range(n))
    free_j = set(range(len(jobs)))

    def cost(ui, ji):
        job = jobs[ji]
        if job["tag"] == "pickup_feed" and job["owner"] != ui:
            return 10**9
        if not _eligible(inventories[ui], job):
            return 10**9
        travel = _dist(positions[ui], job["target"])
        slack = job["deadline"] - hour - travel
        late = max(0, -slack)
        urgent = max(0, 2 - slack)
        # Travel is the base cost; value and deadline make hard work dominate.
        affinity = 0 if job["owner"] is None or job["owner"] == ui else 420
        return 18 * travel + 110 * late + 24 * urgent + affinity - job["value"]

    pairs = []
    for ui in range(n):
        for ji in range(len(jobs)):
            c = cost(ui, ji)
            if c < 10**9:
                pairs.append((c, ui, ji))
    pairs.sort()
    for _c, ui, ji in pairs:
        if ui in free_u and ji in free_j:
            assignment[ui] = ji
            free_u.remove(ui)
            free_j.remove(ji)

    # Greedy is nearly exact here; pair swaps remove its common crossing error.
    changed = True
    while changed:
        changed = False
        for a in range(n):
            ja = assignment[a]
            if ja is None: continue
            for b in range(a + 1, n):
                jb = assignment[b]
                if jb is None: continue
                before = cost(a, ja) + cost(b, jb)
                after = cost(a, jb) + cost(b, ja)
                if after < before:
                    assignment[a], assignment[b] = jb, ja
                    changed = True

    actions = []
    for ui, ji in enumerate(assignment):
        if ji is None:
            actions.append(SAFE)
            continue
        job = jobs[ji]
        actions.append(job["action"] if positions[ui] == job["target"] else _move(positions[ui], job["target"]))
    return actions


def _sanitize(actions, market, seeds, n_hands):
    valid = set(MOVES) | {"PASS", "DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST",
                          "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
                          "COLLECT_FERTILIZER", "CARE"}
    out = []
    used = {}
    for action in actions[:n_hands + 1]:
        a = action if isinstance(action, list) and action and action[0] in valid else SAFE
        if a[0] == "PLANT":
            crop = a[1] if len(a) > 1 else ""
            used[crop] = used.get(crop, 0) + 1
            if crop not in CROP_NAMES or used[crop] > int(seeds.get(crop, 0) or 0):
                a = SAFE
        out.append(a)
    while len(out) < n_hands + 1:
        out.append(SAFE)
    clean = []
    for order in market[:MAX_ORDERS]:
        if not isinstance(order, list) or not order: continue
        op = order[0]
        if op in ("HIRE", "BUY_LAND"):
            clean.append([op]); continue
        if len(order) != 3: continue
        try: qty = int(order[2])
        except (TypeError, ValueError): continue
        if qty <= 0: continue
        if op == "BUY_SEED" and order[1] in CROP_NAMES: clean.append([op, order[1], qty])
        elif op == "BUY_ANIMAL" and order[1] in ANIMAL_NAMES: clean.append([op, order[1], qty])
        elif op == "BUY_PRODUCT" and order[1] in ("WHEAT", "FERTILIZER"): clean.append([op, order[1], qty])
        elif op == "SELL" and order[1] in PRODUCTS: clean.append([op, order[1], qty])
    return {"farmer": out[0], "hands": out[1:], "market": clean}


def _policy(obs):
    player = int(obs.get("player", 0))
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    board = farm["tiles"]
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    step = day * 24 + hour
    last_day = day == FINAL_DAY
    positions = [tuple(farm["farmer"])] + [tuple(x) for x in farm.get("hands", [])]
    invs = [dict(x or {}) for x in private.get("inventories", [])]
    while len(invs) < len(positions): invs.append({})
    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    shops = tuple((obs.get("town", {}) or {}).get("unlocked_shops", []) or [])

    plants, animals, empty_structures, empty = [], [], [], []
    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            if tile is None: empty.append((x, y))
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT": plants.append((x, y, tile))
            elif isinstance(tile, dict) and tile.get("animal") in ANIMAL_NAMES: animals.append((x, y, tile))
            elif isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"): empty_structures.append((x, y, tile))

    live = {a: 0 for a in ANIMAL_NAMES}
    for _x, _y, tile in animals: live[tile["animal"]] += 1
    carried = {item: sum(int(inv.get(item, 0) or 0) for inv in invs) for item in PRODUCTS + ANIMAL_NAMES}
    pending = {a: int(shed.get(a, 0) or 0) + carried[a] for a in ANIMAL_NAMES}
    jobs = []

    # Final-day carriers are recalled early enough for the farthest exact geometry.
    if last_day and hour >= 18:
        for ui, inv in enumerate(invs):
            if inv:
                jobs.append(_job(_nearest_shed(positions[ui]), ["DROP"], 21, 20000,
                                 PRODUCTS + ANIMAL_NAMES, "liquidate"))
    else:
        # Animals: every open operation is its own job, so several hands can clear a
        # pasture simultaneously. Feed is the survival deadline, care creates yield,
        # harvest avoids the held cap, and fertilizer is daily expiring value.
        # Give each hand a contiguous segment of a nearest-neighbour tour.  This
        # prevents per-turn rematching from making equal-priority husbandry jobs
        # zig-zag forever, while the outer matcher still prices all other work.
        def fixed_route_rank(pos):
            x, y = pos
            if 0 <= x < 5 and 0 <= y < 5:
                row = 4 - y
                return row * 5 + ((4 - x) if row % 2 == 0 else x)
            return 25 + min(_dist(pos, s) for s in SHED_TILES) * 10 + y * 10 + x

        owner_by_pos = {
            (x, y): min(len(positions) - 1, fixed_route_rank((x, y)) * len(positions) // 25)
            for x, y, _tile in animals
        }

        for x, y, tile in animals:
            pos = (x, y)
            owner = owner_by_pos[pos]
            if not last_day and not tile.get("fed_today", False): jobs.append(_job(pos, ["FEED"], 22, 12000, "WHEAT", "feed", owner))
            if not last_day and not tile.get("cared_today", False): jobs.append(_job(pos, ["CARE"], 21, 6000, tag="care", owner=owner))
            if tile.get("fertilizer_available", False): jobs.append(_job(pos, ["COLLECT_FERTILIZER"], 22, 20000 if last_day else 7000, tag="fert", owner=owner))
            held = int(tile.get("yield_units", 0) or 0)
            if held > 0:
                jobs.append(_job(pos, ["HARVEST"], 18, 19000 if last_day else 9000,
                                 tag="harvest", owner=owner))

        # Exact crop calendars. Survival watering is inherent in these schedules.
        crop_counts = {c: 0 for c in CROP_NAMES}
        for x, y, tile in plants:
            crop = tile["crop"]; crop_counts[crop] += 1
            age = day - int(tile.get("planted_day", day))
            if age in WATER_AGES[crop] and not tile.get("watered_today", False):
                jobs.append(_job((x, y), ["WATER"], 21, 900 if age == 0 else 620, tag="water"))
            if PLAN["fertilize"].get(crop, False) and age in FERT_AGES.get(crop, ()) and int(tile.get("fertilized_until_day", -1)) < day:
                jobs.append(_job((x, y), ["FERTILIZE"], 19, 720, "FERTILIZER", "cropfert"))
            held = int(tile.get("yield_units", 0) or 0)
            if held > 0:
                if not CROPS[crop]["ongoing"] and age >= CROPS[crop]["last"]:
                    jobs.append(_job((x, y), ["HARVEST"], 20, 850 + 40 * held, tag="harvest"))
                elif CROPS[crop]["ongoing"] and age in ONGOING_HARVEST_AGES.get(crop, ()):
                    jobs.append(_job((x, y), ["HARVEST"], 18, 800 + 50 * held, tag="harvest"))
            if CROPS[crop]["ongoing"] and age > CROPS[crop]["last"] and held == 0:
                jobs.append(_job((x, y), ["DIG"], 22, 180, tag="dig"))

        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    jobs.append(_job((x, y), ["DIG"], 22, 240, tag="dig"))

        desired = {a: _animal_target(a, day) for a in ANIMAL_NAMES}
        # Build only the exact structures required by purchased/near-term animals.
        structures = {"COOP": 0, "PASTURE": 0}
        for _x, _y, tile in animals + empty_structures: structures[tile["kind"]] += 1
        needed = {
            "COOP": max(0, desired["GOOSE"] - structures["COOP"]),
            "PASTURE": max(0, desired["COW"] + desired["SHEEP"] - structures["PASTURE"]),
        }
        sites = sorted(empty, key=lambda p: (_layout_rank(p), p))
        used_sites = set()
        for structure in ("PASTURE", "COOP"):
            for pos in [p for p in sites if p not in used_sites][:needed[structure]]:
                used_sites.add(pos)
                jobs.append(_job(pos, ["BUILD_" + structure], 17, 10000, tag="build"))

        # Carried animals are tied to compatible exact structures; shed stock gets
        # pickup jobs. A hand may carry several animals but PLACE consumes one.
        empties_by_kind = {"COOP": [], "PASTURE": []}
        for x, y, tile in empty_structures: empties_by_kind[tile["kind"]].append((x, y))
        for a in ANIMAL_NAMES:
            for pos in sorted(empties_by_kind[ANIMALS[a]["structure"]], key=_layout_rank):
                jobs.append(_job(pos, ["PLACE", a], 19, 10500, a, "place"))
            if int(shed.get(a, 0) or 0) > 0 and carried[a] == 0:
                for s in SHED_TILES:
                    jobs.append(_job(s, ["PICKUP", a, min(4, int(shed.get(a, 0)))], 12, 11000, tag="pickup_animal"))

        # Batch feed/fertilizer pickup. Multiple shed jobs let several hands load in
        # parallel when an entire herd is waiting, while inventory need prevents waste.
        hungry_by_owner = [0] * len(positions)
        for x, y, tile in animals:
            if not tile.get("fed_today", False):
                hungry_by_owner[owner_by_pos[(x, y)]] += 1
        if not last_day and int(shed.get("WHEAT", 0) or 0) > 0:
            for ui, hungry in enumerate(hungry_by_owner):
                need = max(0, hungry - int(invs[ui].get("WHEAT", 0) or 0))
                if hour == 0:
                    planned_units = PLAN["hands_by_5day"][min(5, day // 5)] + 1
                    need = min(need, max(1, math.ceil(sum(hungry_by_owner) / planned_units)))
                if need:
                    s = min(SHED_TILES, key=lambda p: (_dist(positions[ui], p), p))
                    jobs.append(_job(s, ["PICKUP", "WHEAT", min(need, int(shed.get("WHEAT", 0)))],
                                     8, 13000, tag="pickup_feed", owner=ui))
        fert_due = sum(j["tag"] == "cropfert" for j in jobs)
        if fert_due > carried["FERTILIZER"] and int(shed.get("FERTILIZER", 0) or 0) > 0:
            jobs.append(_job(SHED_TILES[0], ["PICKUP", "FERTILIZER", fert_due], 12, 760, tag="pickup_fert"))

        # Plant exact tiles in crop blocks. The mix is interpreted as target counts
        # over currently crop-usable space after livestock reservations.
        phase_mix = PLAN["crop_mix"][_phase(day)]
        active = [c for c in phase_mix if phase_mix[c] > 0 and day <= PLAN["last_plant_day"].get(c, -1)]
        plantable = [p for p in empty if p not in used_sites]
        if active and hour <= 16:
            total_slots = len(plants) + len(plantable)
            target = {c: round(total_slots * phase_mix[c] / sum(phase_mix[k] for k in active)) for c in active}
            for pos in plantable:
                choices = [c for c in active if target.get(c, 0) > crop_counts.get(c, 0) and int(seeds.get(c, 0) or 0) > 0]
                if not choices: break
                crop = min(choices, key=lambda c: (_layout_rank(pos, c), -(target[c] - crop_counts[c]), c))
                crop_counts[crop] += 1
                jobs.append(_job(pos, ["PLANT", crop], 16, 460, tag="plant"))

        # Product carriers make deliberate deposit trips. Keep feed/animals/fertilizer
        # aboard when they still have a matching job; otherwise drop valuable cargo.
        for ui, inv in enumerate(invs):
            cargo = _cargo_value(inv)
            utility = sum(int(inv.get(x, 0) or 0) for x in ("WHEAT", "FERTILIZER") + ANIMAL_NAMES)
            if cargo >= 300 or (cargo > 0 and hour >= 18) or (sum(shed.values()) >= 82 and cargo > 0):
                jobs.append(_job(_nearest_shed(positions[ui]), ["DROP"], 21,
                                 650 + cargo - 80 * utility, PRODUCTS, "drop"))

    actions = _assign(positions, invs, jobs, hour)

    # Market queue: exact sequential cash accounting, hires first, then feed/seeds/
    # animals. Sales include same-turn DROP because unit actions precede the market.
    orders = []
    money = float(farm.get("money", 0))
    target_hands = PLAN["hands_by_5day"][min(5, day // 5)]
    hires_today = int(farm.get("hires_today", 0))

    # Sell everything immediate; metered goods match the exact preceding town tick.
    sale_qty = {p: int(shed.get(p, 0) or 0) for p in PRODUCTS}
    for ui, action in enumerate(actions):
        if action and action[0] == "DROP" and positions[ui] in SHED_TILES:
            room = max(0, SHED_CAP - sum(shed.values()))
            for p in PRODUCTS:
                n = min(room, int(invs[ui].get(p, 0) or 0)); sale_qty[p] += n; room -= n
    reserve_wheat = 0 if last_day else len(animals) + 3
    sale_qty["WHEAT"] = max(0, sale_qty["WHEAT"] - reserve_wheat)
    for p in ("FERTILIZER", "WOOL", "MILK", "MELON", "STRAWBERRY", "EGG", "TOMATO", "CARROT", "WHEAT"):
        if len(orders) >= MAX_ORDERS: break
        qty = sale_qty[p]
        if qty > 0:
            orders.append(["SELL", p, qty])

    if not last_day and day < 29:
        shed_used = sum(shed.values())
        live_pending = sum(live.values()) + sum(pending.values())
        wheat_have = int(shed.get("WHEAT", 0) or 0) + carried["WHEAT"]
        feed_want = live_pending + 4
        if len(orders) < MAX_ORDERS and wheat_have < feed_want and shed_used < 92:
            qty = min(feed_want - wheat_have, 92 - shed_used)
            # $25 is exact around I0 and conservative enough for sequential safety.
            qty = min(qty, max(0, int((money - PLAN["cash_reserve"]) // 25)))
            if qty > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", qty]); money -= 25 * qty; shed_used += qty

        # Fixed-plan capital purchases are placed at hour zero, in plan order,
        # before labor.  Feed can still be replenished intraday for survival.
        if hour >= 0:  # hour-zero priority; retry later only if liquidity delayed an order
            phase_mix = PLAN["crop_mix"][_phase(day)]
            open_count = len(empty)
            for crop in sorted(CROP_NAMES, key=lambda c: (-phase_mix.get(c, 0), c)):
                if len(orders) >= MAX_ORDERS: break
                if phase_mix.get(crop, 0) <= 0 or day > PLAN["last_plant_day"].get(crop, -1): continue
                phase_total = max(1, sum(phase_mix.values()))
                planted = sum(1 for _x, _y, tile in plants if tile["crop"] == crop)
                want = max(0, math.ceil((open_count + len(plants)) * phase_mix[crop] / phase_total)
                           - planted - int(seeds.get(crop, 0) or 0))
                qty = min(want, max(0, int((money - PLAN["cash_reserve"]) // CROPS[crop]["seed"])))
                if qty > 0:
                    orders.append(["BUY_SEED", crop, qty]); money -= CROPS[crop]["seed"] * qty

            for animal in PLAN["animals"]:
                if len(orders) >= MAX_ORDERS: break
                want = max(0, _animal_target(animal, day) - live[animal] - pending[animal])
                room = max(0, 96 - shed_used)
                qty = min(want, room, max(0, int((money - PLAN["cash_reserve"]) // ANIMALS[animal]["cost"])))
                if qty > 0:
                    orders.append(["BUY_ANIMAL", animal, qty]); money -= ANIMALS[animal]["cost"] * qty; shed_used += qty

        # Fertilizer is bought only when the exact calendar exposes uncovered jobs.
        fert_jobs = sum(j["tag"] == "cropfert" for j in jobs)
        fert_have = int(shed.get("FERTILIZER", 0) or 0) + carried["FERTILIZER"]
        if len(orders) < MAX_ORDERS and fert_jobs > fert_have and shed_used < 96:
            qty = min(fert_jobs - fert_have, 96 - shed_used, max(0, int((money - PLAN["cash_reserve"]) // 100)))
            if qty > 0: orders.append(["BUY_PRODUCT", "FERTILIZER", qty])

    if hour <= 2:
        for idx in range(hires_today, target_hands):
            if len(orders) >= MAX_ORDERS: break
            cost = _FIB[idx]
            if money - cost < PLAN["cash_reserve"]: break
            orders.append(["HIRE"]); money -= cost

    return _sanitize(actions, orders, seeds, len(positions) - 1)


def agent(obs, config=None):
    """Safe final entry point; malformed observations can never crash a game."""
    started = time.perf_counter()
    try:
        action = _policy(obs)
    except Exception:
        n = 0
        try: n = len(obs["farms"][int(obs.get("player", 0))].get("hands", []))
        except Exception: pass
        action = {"farmer": SAFE, "hands": [SAFE for _ in range(n)], "market": []}
    _TIMES.append((time.perf_counter() - started) * 1000.0)
    return action
