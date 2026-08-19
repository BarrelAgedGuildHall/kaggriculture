"""Kaggler's Kaggriculture agent -- exp-001 "Sensible farmer" v0.

Submission entry point: Kaggle imports this file and calls `agent(obs)` once per turn
(720 turns, 1 s each). Single file, stdlib only, no state carried between turns except a
tiny timing histogram (printed to the agent log for sandbox-speed calibration).

Policy (see workshop/deliverables/2026-08-19-kaggriculture-strategy/strategy.md, exp-001):
  * hire HIRES_PER_DAY hands at hour 0 every day (fib cost resets daily -> 10 hands = $143);
  * build coops near the shed, buy geese, run the FEED / CARE / COLLECT_FERTILIZER loop
    (a fed+cared goose lays 2 eggs/day and every animal yields 1 fertilizer/day);
  * wheat + carrot rotation on the remaining tiles with the *minimal* correct watering
    schedule (water on the planting day -- an unwatered seed weeds out that same night --
    then only on days that actually add yield);
  * sell out of the shed against per-product reservation prices, computed with an exact
    replica of the engine's `market_price` so we know how far an order walks the price down;
  * buy land when the bank can carry it; liquidate everything on days 28-29.

Everything is a greedy per-turn assignment of units to tile jobs -- no lookahead, no tuning.
This is the floor, not the ceiling: exp-004..007 replace the constants below with a demand
forecast and an evolutionary search.
"""

import math
import time

# --------------------------------------------------------------------------------------
# Engine replica (kaggle-environments 1.32.7, kaggriculture.py -- read line by line).
# Only the parts the policy needs to reason about; every constant is copied verbatim.
# --------------------------------------------------------------------------------------

TURNS_PER_DAY = 24
FINAL_DAY = 29           # day of the last acting turn (step 718 = day 29, hour 22)
PRICE_FLOOR = 1
MARKET_I0 = 10000
HINGE_GAIN = 8.0
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",  "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",  "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",  "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _shape(func, x, T=None):
    x = x if x > 0.0 else 0.0
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        over = u - 1.0
        return u + HINGE_GAIN * (over * over if over > 0.0 else 0.0)
    return x


# amp is a constant per item/side -- precompute it once at import.
_AMP = {}
for _it, _p in MARKET_PARAMS.items():
    _AMP[_it] = (
        _p["below_target"] * _p["base"] / _shape(_p["below_func"], _p["T"], _p["T"]),
        _p["above_target"] * _p["base"] / _shape(_p["above_func"], _p["T"], _p["T"]),
    )


def market_price(item, inventory):
    """Exact replica of kaggriculture.market_price (floored at 1, int(round(...)))."""
    p = MARKET_PARAMS[item]
    if inventory < MARKET_I0:
        price = p["base"] + _AMP[item][0] * _shape(p["below_func"], MARKET_I0 - inventory, p["T"])
    else:
        price = p["base"] - _AMP[item][1] * _shape(p["above_func"], inventory - MARKET_I0, p["T"])
    price = int(round(price))
    return price if price > PRICE_FLOOR else PRICE_FLOOR


# Watering bonus window start and the age at which a one-time crop reaches max yield.
WINDOW_START = {}
TARGET_AGE = {}
for _c, _cd in CROPS.items():
    if _cd["ongoing"]:
        WINDOW_START[_c] = 0
        TARGET_AGE[_c] = _cd["first_yield_day"]
    else:
        _ws = (_cd["max_yield_day"] + 1) // 2
        WINDOW_START[_c] = _ws
        # yield = 1 + (age - ws + 1) while watered in-window, capped by max_yield / max_yield_day
        TARGET_AGE[_c] = min(_cd["max_yield_day"],
                             max(_cd["first_yield_day"], _cd["max_yield"] + _ws - 2))

# --------------------------------------------------------------------------------------
# Tunables (exp-007 will search these; keep them in one block).
# --------------------------------------------------------------------------------------

HIRES_PER_DAY = 10           # fib cost resets daily: 10 hands = $143/day
MAX_ANIMALS = 12             # coops (geese) at steady state
ANIMAL = "GOOSE"
ANIMALS_RAMP = (4, 2)        # target coops on day d = min(MAX_ANIMALS, a + b*d)
LAST_ANIMAL_DAY = 26         # a goose bought later cannot pay back its $300
MONEY_RESERVE = 320          # never spend the bank below this on animals/seeds
LAND_TRIGGER = (900, 2600, 6000)    # bank needed before buying NE / SW / SE
LAND_FIRST_DAY = 0
LAND_LAST_DAY = 22
CROP_SHARE = {"WHEAT": 0.75, "CARROT": 0.25}
SEED_CAP = {"WHEAT": 40, "CARROT": 24}   # seeds are stocked against open tiles, not flat
ANIMAL_HARVEST_MIN = 2       # harvest a coop once it holds this many eggs
WHEAT_FEED_BUFFER = 4        # shed wheat kept back from the market, on top of one per animal
SHED_SOFT_CAP = 70           # above this, halve reservation prices
SHED_HARD_CAP = 90           # above this, dump
LIQUIDATE_DAY = 28           # reservation prices start collapsing here
DROP_HOUR = 13               # on the final day, everyone carrying goods walks to the shed
PLANNING_BUDGET_S = 0.30     # bail out of planning and emit what we have

RESERVATION = {
    "WHEAT": 14, "CARROT": 18, "TOMATO": 30, "STRAWBERRY": 50, "MELON": 90,
    "EGG": 26, "MILK": 70, "WOOL": 90, "FERTILIZER": 28,
}
SELL_ORDER = ["FERTILIZER", "EGG", "MELON", "MILK", "WOOL", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"]

DEBUG_TIMING = True
_TIMES = []

# --------------------------------------------------------------------------------------
# Action vocabulary (used by the sanitiser -- nothing illegal ever leaves this file).
# --------------------------------------------------------------------------------------

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
UNIT_OPS = set(MOVES) | {
    "PASS", "DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE",
    "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
}
MARKET_OPS = {"HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
SAFE_UNIT = ["PASS"]


def _get(o, key, default=None):
    """Observations arrive as a dict-like Struct; be liberal about both access styles."""
    try:
        v = o[key]
    except Exception:
        v = getattr(o, key, None)
    return default if v is None else v


_SITE_CACHE = {}
_RANK_CACHE = {}


def _shed_tiles(n):
    half = n // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _coop_sites(n):
    """Every non-shed-access tile, nearest-to-the-shed first: animals cost 4 actions/day and
    a wheat trip, so they get the short commute and crops take the outfield."""
    if n in _SITE_CACHE:
        return _SITE_CACHE[n]
    access = set(_shed_tiles(n))
    c = (n - 1) / 2.0
    pts = [(x, y) for y in range(n) for x in range(n) if (x, y) not in access]
    pts.sort(key=lambda p: ((p[0] - c) ** 2 + (p[1] - c) ** 2, p[1], p[0]))
    _SITE_CACHE[n] = pts
    _RANK_CACHE[n] = dict((p, i) for i, p in enumerate(pts))
    return pts


def _rank(n):
    """(x, y) -> distance rank from the shed. Keeping the worked farm compact around the
    centre is the single biggest saving available: ~60% of unit-turns go on walking."""
    if n not in _RANK_CACHE:
        _coop_sites(n)
    return _RANK_CACHE[n]


def _step_toward(ux, uy, tx, ty):
    if ux < tx:
        return ["EAST"]
    if ux > tx:
        return ["WEST"]
    if uy < ty:
        return ["SOUTH"]
    if uy > ty:
        return ["NORTH"]
    return None


# --------------------------------------------------------------------------------------
# Market: how many units can we sell before the price drops under our reservation?
# --------------------------------------------------------------------------------------

def _reservation(item, day, pressure):
    r = RESERVATION.get(item, 0)
    if day >= FINAL_DAY:
        return 0
    if pressure >= SHED_HARD_CAP:
        return 0
    if day >= LIQUIDATE_DAY:
        r *= 0.35
    elif day >= LIQUIDATE_DAY - 1:
        r *= 0.70
    if pressure >= SHED_SOFT_CAP:
        r *= 0.5
    return r


def _sell_orders(shed, mkt_inv, day, wheat_reserve, slots):
    """Walk each product's price down with the engine's own formula and stop at the
    reservation price. Selling at $1 does not add to market supply (engine detail)."""
    out = []
    pressure = 0
    for v in shed.values():
        pressure += v
    for item in SELL_ORDER:
        if len(out) >= slots:
            break
        qty = int(shed.get(item, 0))
        if item == "WHEAT":
            qty -= wheat_reserve
        if qty <= 0:
            continue
        res = _reservation(item, day, pressure)
        inv = mkt_inv.get(item, MARKET_I0)
        n = 0
        while n < qty:
            price = market_price(item, inv)
            if price < res:
                break
            n += 1
            if price > PRICE_FLOOR:
                inv += 1
        if n > 0:
            out.append(["SELL", item, n])
    return out


# --------------------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------------------

def _policy(obs, t0):
    me = int(_get(obs, "player", 0))
    farms = _get(obs, "farms", [])
    farm = farms[me]
    priv = _get(obs, "private", {})
    tiles = farm["tiles"]
    n = len(tiles)
    day = int(_get(obs, "day", 0))
    hour = int(_get(obs, "hour", 0))
    money = float(farm.get("money", 0))
    shed = dict(_get(priv, "shed", {}) or {})
    seeds = dict(_get(priv, "seeds", {}) or {})
    invs = list(_get(priv, "inventories", []) or [])
    market = _get(obs, "market", {}) or {}
    mkt_inv = dict(_get(market, "inventory", {}) or {})
    hands = list(farm.get("hands", []) or [])
    n_hands = len(hands)
    unlocked = list(farm.get("unlocked_quadrants", ["NW"]) or ["NW"])

    last_day = day >= FINAL_DAY
    hours_left = TURNS_PER_DAY - 1 - hour
    shed_used = sum(shed.values())
    shed_room = SHED_CAPACITY - shed_used

    # ---- units -------------------------------------------------------------------
    positions = [list(farm["farmer"])] + [list(h) for h in hands]
    units = []
    for i, pos in enumerate(positions):
        inv = invs[i] if i < len(invs) else {}
        units.append({"i": i, "x": int(pos[0]), "y": int(pos[1]), "inv": dict(inv or {})})
    carried = {}
    for u in units:
        for k, v in u["inv"].items():
            carried[k] = carried.get(k, 0) + int(v)

    # ---- board scan --------------------------------------------------------------
    plants, animals, empty_coops, empties, weeds = [], [], [], [], []
    for y in range(n):
        row = tiles[y]
        for x in range(n):
            t = row[x]
            if t is None:
                empties.append((x, y))
                continue
            if not isinstance(t, dict):
                continue                      # "LOCKED"
            kind = t.get("kind")
            if kind == "PLANT":
                plants.append((x, y, t))
            elif kind == "WEED":
                weeds.append((x, y))
            elif "animal" in t:
                animals.append((x, y, t))
            elif kind == "COOP":
                empty_coops.append((x, y))

    n_animals = len(animals)
    empty_set = set(empties)

    # ---- coop plan ---------------------------------------------------------------
    target_coops = min(MAX_ANIMALS, ANIMALS_RAMP[0] + ANIMALS_RAMP[1] * day)
    coop_pos = set(empty_coops) | set((x, y) for (x, y, _t) in animals)
    need_coops = target_coops - len(coop_pos)
    build_sites = []
    if need_coops > 0 and day <= LAST_ANIMAL_DAY:
        for p in _coop_sites(n):
            if len(build_sites) >= need_coops:
                break
            if p in coop_pos or p not in empty_set:
                continue
            build_sites.append(p)
    reserved = coop_pos | set(build_sites)

    tasks = []   # (prio, x, y, action, need_item, skip_if_has)

    def add(prio, x, y, action, need=None, skip=None):
        tasks.append((prio, x, y, action, need, skip))

    # ---- plants ------------------------------------------------------------------
    for (x, y, t) in plants:
        crop = t.get("crop")
        cd = CROPS.get(crop)
        if cd is None:
            continue
        age = day - int(t.get("planted_day", day))
        yu = int(t.get("yield_units", 0))
        watered = bool(t.get("watered_today", False))
        unwatered = int(t.get("consecutive_unwatered", 0))
        tgt = TARGET_AGE[crop]
        ready = yu > 0 and age >= cd["first_yield_day"]
        # A plant that cannot reach its first yield before the season ends is abandoned.
        viable = (day + max(0, cd["first_yield_day"] - age)) <= FINAL_DAY
        gains = (not cd["ongoing"]
                 and WINDOW_START[crop] <= age <= cd["max_yield_day"]
                 and yu < cd["max_yield"])
        if cd["ongoing"]:
            harvest = ready
        else:
            stalled = (day + max(0, tgt - age)) > FINAL_DAY   # cannot reach target age in time
            harvest = ready and (age >= tgt or stalled or age > cd["max_yield_day"])

        if not watered and viable and (unwatered >= 1 or gains):
            if last_day:
                # Watering only pays on the last day if we are going to harvest the tile.
                if gains and age >= cd["first_yield_day"]:
                    add(0, x, y, ["WATER"])
            else:
                urgent = unwatered >= 1 or age >= tgt
                add(0 if urgent else 5, x, y, ["WATER"])

        if harvest:
            # Harvest after the day's watering so the last bonus unit is not lost.
            if watered or not gains or hour >= 18 or (last_day and hour >= DROP_HOUR - 3):
                add(3, x, y, ["HARVEST"])

    # ---- animals -----------------------------------------------------------------
    for (x, y, t) in animals:
        a = ANIMALS.get(t.get("animal"), ANIMALS["GOOSE"])
        fed = bool(t.get("fed_today", False))
        cared = bool(t.get("cared_today", False))
        unfed = int(t.get("consecutive_unfed", 0))
        yu = int(t.get("yield_units", 0))
        # Day 29 never runs an end-of-day refresh, so feeding and caring are worthless then.
        if not last_day and not fed:
            add(1 if unfed >= 1 else 4, x, y, ["FEED"], need="WHEAT")
        if t.get("fertilizer_available", False):
            add(3, x, y, ["COLLECT_FERTILIZER"])
        if yu > 0 and (yu >= a["max_held"] or yu >= ANIMAL_HARVEST_MIN or day >= LIQUIDATE_DAY):
            add(1 if yu >= a["max_held"] else 4, x, y, ["HARVEST"])
        if not last_day and not cared:
            add(4, x, y, ["CARE"])

    # ---- animal placement / coop building ----------------------------------------
    if not last_day:
        goose_carried = carried.get(ANIMAL, 0)
        goose_shed = int(shed.get(ANIMAL, 0))
        slots = len(empty_coops)
        for k, (x, y) in enumerate(empty_coops):
            if k >= goose_carried:
                break
            add(3, x, y, ["PLACE", ANIMAL], need=ANIMAL)
        want_pick = min(slots - goose_carried, goose_shed)
        sheds = _shed_tiles(n)
        for k in range(max(0, min(want_pick, 3))):
            sx, sy = sheds[k % len(sheds)]
            add(2, sx, sy, ["PICKUP", ANIMAL, 1], skip=ANIMAL)
        for (x, y) in build_sites:
            add(5, x, y, ["BUILD_COOP"])

    # ---- wheat logistics ---------------------------------------------------------
    if not last_day:
        need_wheat = 0
        for (_x, _y, t) in animals:
            if not t.get("fed_today", False):
                need_wheat += 1
        have_wheat = carried.get("WHEAT", 0)
        deficit = min(need_wheat - have_wheat, int(shed.get("WHEAT", 0)))
        if deficit > 0:
            carriers = min(3, max(1, (deficit + 4) // 5))
            per = (deficit + carriers - 1) // carriers
            sheds = _shed_tiles(n)
            for k in range(carriers):
                sx, sy = sheds[k % len(sheds)]
                add(2, sx, sy, ["PICKUP", "WHEAT", per], skip="WHEAT")

    # ---- planting ----------------------------------------------------------------
    rank = _rank(n)
    plantable = sorted((p for p in empties if p not in reserved),
                       key=lambda p: rank.get(p, 999))
    if not last_day:
        live = {}
        for (_x, _y, t) in plants:
            c = t.get("crop")
            live[c] = live.get(c, 0) + 1
        budget = dict((c, int(seeds.get(c, 0))) for c in CROP_SHARE)
        planned = dict((c, live.get(c, 0)) for c in CROP_SHARE)
        for (x, y) in plantable[:60]:
            best, best_score = None, None
            for c in CROP_SHARE:
                if budget.get(c, 0) <= 0:
                    continue
                if day + TARGET_AGE[c] > FINAL_DAY:
                    continue
                score = planned[c] / CROP_SHARE[c]
                if best_score is None or score < best_score:
                    best, best_score = c, score
            if best is None:
                break
            budget[best] -= 1
            planned[best] += 1
            add(5, x, y, ["PLANT", best])

        for (x, y) in sorted(weeds, key=lambda p: rank.get(p, 999))[:24]:
            add(5, x, y, ["DIG"])

    # ---- assignment --------------------------------------------------------------
    actions = {}
    if last_day and hour >= DROP_HOUR:
        # Anything still in a unit's hands at the final bell is worth zero, and the day-29
        # end-of-day drop never runs. Walk it home and DROP so the market can take it.
        sheds = _shed_tiles(n)
        for u in units:
            if not u["inv"]:
                continue
            tx, ty = min(sheds, key=lambda p: abs(p[0] - u["x"]) + abs(p[1] - u["y"]))
            mv = _step_toward(u["x"], u["y"], tx, ty)
            actions[u["i"]] = mv if mv is not None else ["DROP"]
        free = [u for u in units if u["i"] not in actions]
    else:
        free = list(units)

    tasks.sort(key=lambda t: (t[0], t[1], t[2]))
    budget_hit = False
    idx = 0
    total = len(tasks)
    while idx < total and free and not budget_hit:
        prio = tasks[idx][0]
        bucket = []
        while idx < total and tasks[idx][0] == prio:
            bucket.append(tasks[idx])
            idx += 1
        # Greedy nearest-pair matching inside the priority bucket.
        while bucket and free:
            if time.perf_counter() - t0 > PLANNING_BUDGET_S:
                budget_hit = True
                break
            best = None
            for bi in range(len(bucket)):
                _p, tx, ty, act, need, skip = bucket[bi]
                for ui in range(len(free)):
                    u = free[ui]
                    if need is not None and u["inv"].get(need, 0) <= 0:
                        continue
                    if skip is not None and u["inv"].get(skip, 0) > 0:
                        continue
                    d = abs(u["x"] - tx) + abs(u["y"] - ty)
                    if d > hours_left:
                        continue
                    if best is None or d < best[0]:
                        best = (d, bi, ui, tx, ty, act)
                        if d == 0:
                            break
                if best is not None and best[0] == 0:
                    break
            if best is None:
                break
            _d, bi, ui, tx, ty, act = best
            u = free[ui]
            mv = _step_toward(u["x"], u["y"], tx, ty)
            actions[u["i"]] = act if mv is None else mv
            free.pop(ui)
            bucket.pop(bi)

    # ---- market orders -----------------------------------------------------------
    orders = []
    if hour == 0:
        for _ in range(min(HIRES_PER_DAY, MAX_MARKET_ORDERS)):
            orders.append(["HIRE"])

    slots = MAX_MARKET_ORDERS - len(orders)
    if slots > 0:
        wheat_reserve = 0 if day >= LIQUIDATE_DAY else (n_animals + WHEAT_FEED_BUFFER)
        orders.extend(_sell_orders(shed, mkt_inv, day, wheat_reserve, slots))

    # Buy order matters: orders execute in list order and each one spends the bank the
    # earlier ones left behind. Feed first -- an unfed animal escapes after two days and
    # takes its $300 and ~$150/day with it, which is what a starved opening costs us.
    buys = []
    if not last_day and day < LIQUIDATE_DAY:
        pending = int(shed.get(ANIMAL, 0)) + carried.get(ANIMAL, 0)
        mouths = n_animals + pending
        wheat_have = int(shed.get("WHEAT", 0)) + carried.get("WHEAT", 0)

        # 1. Feed safety net (no bank reserve: this is survival, not investment). We stock
        #    a few mouths ahead of the flock, otherwise the opening deadlocks: no wheat
        #    without animals, no animals without wheat.
        feed_target = mouths + WHEAT_FEED_BUFFER
        if day <= LAST_ANIMAL_DAY and target_coops > mouths:
            feed_target += min(4, target_coops - mouths)
        if feed_target > 0 and wheat_have < feed_target and shed_room > 2:
            k = min(feed_target - wheat_have, 14, shed_room)
            price = market_price("WHEAT", mkt_inv.get("WHEAT", MARKET_I0) - 1)
            if k > 0 and money >= price:
                k = min(k, int(money // price))
                if k > 0:
                    buys.append(["BUY_PRODUCT", "WHEAT", k])

        # 2. Seeds, stocked against the tiles actually open rather than a flat number:
        #    running out of seed leaves half the farm fallow.
        open_tiles = len(plantable)
        for c in ("WHEAT", "CARROT"):
            have = int(seeds.get(c, 0))
            if day + TARGET_AGE[c] > FINAL_DAY:
                continue
            want = min(SEED_CAP[c], int(open_tiles * CROP_SHARE[c]) + 4)
            if have >= max(4, want // 2):
                continue
            k = want - have
            if k > 0 and money - k * CROPS[c]["seed"] >= MONEY_RESERVE:
                buys.append(["BUY_SEED", c, k])

        # 3. Animals -- but never more mouths than we can already feed.
        want = target_coops - mouths
        if (want > 0 and day <= LAST_ANIMAL_DAY and shed_room > 0
                and wheat_have >= mouths + 1):
            afford = int((money - MONEY_RESERVE) // ANIMALS[ANIMAL]["cost"])
            k = min(want, afford, 4, shed_room, max(0, wheat_have - mouths))
            if k > 0:
                buys.append(["BUY_ANIMAL", ANIMAL, k])

        # 4. Land last: tiles are worthless without the seed and the hands to work them.
        n_extra = len(unlocked) - 1
        if (0 <= n_extra < 3 and LAND_FIRST_DAY <= day <= LAND_LAST_DAY
                and money >= LAND_TRIGGER[n_extra]):
            buys.append(["BUY_LAND"])

    for b in buys:
        if len(orders) >= MAX_MARKET_ORDERS:
            break
        orders.append(b)

    # ---- emit --------------------------------------------------------------------
    farmer_action = actions.get(0, SAFE_UNIT)
    hand_actions = [actions.get(i + 1, SAFE_UNIT) for i in range(n_hands)]
    return _sanitize(farmer_action, hand_actions, orders, seeds, n_hands)


# --------------------------------------------------------------------------------------
# Sanitiser -- nothing malformed leaves this file, and never more PLANTs of a crop than we
# hold seeds for (the engine drops *every* PLANT for a crop over-requested in one turn).
# --------------------------------------------------------------------------------------

def _ok_unit(a):
    return isinstance(a, list) and bool(a) and isinstance(a[0], str) and a[0] in UNIT_OPS


def _sanitize(farmer, hands, orders, seeds, n_hands):
    units = [farmer if _ok_unit(farmer) else SAFE_UNIT]
    for i in range(n_hands):
        a = hands[i] if i < len(hands) else SAFE_UNIT
        units.append(a if _ok_unit(a) else SAFE_UNIT)

    used = {}
    for i, a in enumerate(units):
        if a[0] != "PLANT":
            continue
        crop = a[1] if len(a) > 1 else None
        if crop not in CROPS:
            units[i] = SAFE_UNIT
            continue
        used[crop] = used.get(crop, 0) + 1
        if used[crop] > int(seeds.get(crop, 0)):
            units[i] = SAFE_UNIT

    clean = []
    for o in orders:
        if len(clean) >= MAX_MARKET_ORDERS:
            break
        if not isinstance(o, list) or not o or not isinstance(o[0], str) or o[0] not in MARKET_OPS:
            continue
        op = o[0]
        if op in ("HIRE", "BUY_LAND"):
            clean.append([op])
            continue
        if len(o) < 3:
            continue
        item = o[1]
        try:
            k = int(o[2])
        except (TypeError, ValueError):
            continue
        if k <= 0:
            continue
        if op == "SELL" and item not in PRODUCTS:
            continue
        if op == "BUY_PRODUCT" and item not in ("WHEAT", "FERTILIZER"):
            continue
        if op == "BUY_SEED" and item not in CROPS:
            continue
        if op == "BUY_ANIMAL" and item not in ANIMALS:
            continue
        clean.append([op, item, k])

    return {"farmer": units[0], "hands": units[1:], "market": clean}


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def agent(obs, config=None):
    t0 = time.perf_counter()
    try:
        action = _policy(obs, t0)
    except Exception:
        try:
            farms = _get(obs, "farms", [])
            n_hands = len(farms[int(_get(obs, "player", 0))].get("hands", []) or [])
        except Exception:
            n_hands = 0
        action = {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}
    if DEBUG_TIMING:
        _TIMES.append(time.perf_counter() - t0)
        n = len(_TIMES)
        if n % 240 == 0:
            s = sorted(_TIMES)
            print("TIMING n=%d mean=%.2fms p99=%.2fms max=%.2fms"
                  % (n, 1000 * sum(s) / n, 1000 * s[min(n - 1, int(n * 0.99))], 1000 * s[-1]),
                  flush=True)
    return action
