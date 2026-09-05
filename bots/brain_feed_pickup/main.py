"""Kaggler's Kaggriculture Brain v1 -- modular closed-loop policy.

Submission entry point: Kaggle imports this file and calls `agent(obs)` once per turn
(720 turns, 1 s each). Single file, stdlib only, no state carried between turns except a
tiny timing histogram (printed to the agent log for sandbox-speed calibration).

Policy:
  * hire against observable work and the next Fibonacci wage rather than a fixed plateau;
  * keep each unit on a deterministic spatial cluster and dispatch harvest, watering,
    planting, husbandry, and shed logistics by deadline and distance;
  * build compact pastures, split 18 animals across cows/sheep, and run the
    FEED / CARE / COLLECT_FERTILIZER loop;
  * bootstrap wheat/melon, then rotate wheat/strawberry/melon with survival- and
    yield-aware watering;
  * sell out of the shed against per-product reservation prices, computed with an exact
    replica of the engine's `market_price` so we know how far an order walks the price down;
  * buy up to three quadrants when the bank can carry them; liquidate on days 28-29.

All actions are derived from the current observation. There are no tapes, replayed actions,
opponent identities, or random choices. Fixed module contracts let experiments replace one
planner at a time without changing the coordinator.
"""

import json
import math
import os
import time
from dataclasses import dataclass, replace

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


SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def town_drain(step, unlocked_shops, shop_interval=4, center_interval=24):
    """Exact units removed by the town after actions at ``step``."""
    drain = {item: 0 for item in PRODUCTS}
    if step % shop_interval == 0:
        for shop in unlocked_shops:
            products = SHOPS.get(shop, ())
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                drain[item] += multiplier
    if step % center_interval == 0:
        for item in PRODUCTS:
            if item != "FERTILIZER":
                drain[item] += 1
    return drain


def predict_sale(item, inventory, quantity):
    """Return (revenue, ending inventory) for our uncontested SELL order."""
    revenue = 0
    inv = int(inventory)
    for _ in range(max(0, int(quantity))):
        quote = market_price(item, inv)
        revenue += quote
        if quote > PRICE_FLOOR:
            inv += 1
    return revenue, inv


def _fib(n):
    a, b = 1, 1
    for _ in range(max(0, int(n))):
        a, b = b, a + b
    return a


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

HIRES_PER_DAY = 12           # hard ceiling; the labor module normally stops at 8-11 as work saturates
HIRE_ON_LAST_DAY = True      # day 29 hands still harvest and carry goods to the shed
MAX_ANIMALS = 18             # mixed premium livestock remain serviceable at this measured edge
ANIMAL_TARGETS = {"COW": 7, "SHEEP": 11}
ANIMALS_RAMP = (2, 2)        # add mouths only as the daily labor budget grows into them
LAST_ANIMAL_DAY = 26         # a goose bought later cannot pay back its $300
MONEY_RESERVE = 240          # enough for a day's hires/feed while keeping early capital productive
LAND_TRIGGER = (1000, 2200, 5200)   # buy only with the land price plus a small operating cushion
LAND_FIRST_DAY = 0
LAND_LAST_DAY = 22
CROP_SHARE = {"WHEAT": 0.55, "STRAWBERRY": 0.25, "MELON": 0.20}
SEED_CAP = {"WHEAT": 50, "STRAWBERRY": 24, "MELON": 16}  # bounded by owned empty tiles
ANIMAL_HARVEST_MIN = 2       # harvest a coop once it holds this many eggs
WHEAT_FEED_BUFFER = 4        # shed wheat kept back from the market, on top of one per animal
SHED_SOFT_CAP = 70           # above this, halve reservation prices
SHED_HARD_CAP = 90           # above this, dump
LIQUIDATE_DAY = 28           # reservation prices start collapsing here
DROP_HOUR = 13               # on the final day, everyone carrying goods walks to the shed
PLANNING_BUDGET_S = 0.30     # bail out of planning and emit what we have

# Reservation prices are deliberately LOW in v0. Measured: holding stock back for a
# better price loses head-to-head. The market only degrades (the town drains a fixed
# amount and both players add supply), so the seller who moves first is paid more.
RESERVATION = {
    "WHEAT": 35, "CARROT": 2, "TOMATO": 4, "STRAWBERRY": 8, "MELON": 15,
    "EGG": 3, "MILK": 10, "WOOL": 12, "FERTILIZER": 4,
}
SELL_ORDER = ["FERTILIZER", "EGG", "MELON", "MILK", "WOOL", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"]

DEBUG_TIMING = False
_TIMES = []
_KPI = None

# Every policy threshold is centralised here for inspection and module swaps.  The
# comments are deliberately executable documentation for Team B's experiments.
config = {
    "turns_per_day": TURNS_PER_DAY,           # pinned engine calendar, used to derive step for either seat
    "final_day": FINAL_DAY,                   # no end-of-day value exists after day 29
    "max_market_orders": MAX_MARKET_ORDERS,   # engine silently truncates orders beyond ten
    "hires_per_day": HIRES_PER_DAY,          # fib(0..11) totals $376/day after the bootstrap ramp
    "hire_schedule": ((0, 5), (3, 8), (6, 10), (10, HIRES_PER_DAY)), # protect bootstrap cash, then fill slots
    "hire_on_last_day": HIRE_ON_LAST_DAY,     # final-day hands can still harvest, drop, and sell by step 718
    "max_animals": MAX_ANIMALS,              # paired screen: 20 saturated logistics; 18 cleared more money
    "animal_targets": ANIMAL_TARGETS,         # split pools: seven cows and eleven sheep avoid one-product collapse
    "animal_ramp": ANIMALS_RAMP,             # capital and feed capacity arrive over several days
    "last_animal_day": LAST_ANIMAL_DAY,       # later livestock cannot repay purchase/feed before turn 720
    "money_reserve": MONEY_RESERVE,           # preserves one operating cycle under a weak quote
    "land_trigger": LAND_TRIGGER,             # land is bought only when its nominal price is covered
    "land_first_day": LAND_FIRST_DAY,          # expansion may pay from the opening if cash survives purchases
    "land_last_day": LAND_LAST_DAY,            # under seven remaining days cannot repay travel and planting setup
    "crop_share": CROP_SHARE,                 # wheat feeds animals; strawberry/melon diversify premium pools
    "seed_cap": SEED_CAP,                     # prevents unlimited seed inventory from consuming cash
    "animal_harvest_min": ANIMAL_HARVEST_MIN, # avoids held-yield cap without harvesting every unit
    "wheat_feed_buffer": WHEAT_FEED_BUFFER,   # protects mouths from a one-turn market/order delay
    "shed_soft_cap": SHED_SOFT_CAP,           # lower floors before the 100-item overflow boundary
    "shed_hard_cap": SHED_HARD_CAP,           # force selling while at least ten slots remain
    "liquidate_day": LIQUIDATE_DAY,           # day 28 leaves two days to route carried inventory home
    "drop_hour": DROP_HOUR,                   # longest board-to-shed Manhattan trip is under 13 turns
    "sell_floors": RESERVATION,                # per-unit walk stops below economic value; wheat $35 won its screen
    "planning_budget_s": PLANNING_BUDGET_S,   # coordinator exits long before the one-second engine deadline
    "modules": {"season": "default", "labor": "labor_b1", "planting": "default", "market": "default"},
}


def _freeze_mapping(value):
    """A cheap recursive snapshot represented by ordinary immutable tuples."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_mapping(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_mapping(v) for v in value)
    return value


@dataclass(frozen=True)
class View:
    player: int
    step: int
    day: int
    hour: int
    turns_left: int
    money: float
    tiles: tuple
    farmer: tuple
    hands: tuple
    unit_positions: tuple
    unit_inventories: tuple
    shed: tuple
    seeds: tuple
    unlocked_quadrants: tuple
    hires_today: int
    market_inventory: tuple
    market_prices: tuple
    town_shops: tuple
    opponent: tuple
    plants: tuple
    animals: tuple
    empty_structures: tuple
    empty_tiles: tuple
    weeds: tuple
    shed_access_tiles: tuple
    jobs: tuple = ()

    @staticmethod
    def from_obs(obs):
        player = int(_get(obs, "player", 0))
        farms = _get(obs, "farms", [])
        me = farms[player]
        other = farms[1 - player]
        private = _get(obs, "private", {}) or {}
        day, hour = int(_get(obs, "day", 0)), int(_get(obs, "hour", 0))
        step = day * TURNS_PER_DAY + hour
        board = me["tiles"]
        plants, animals, empty_structures, empties, weeds = [], [], [], [], []
        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                if tile is None:
                    empties.append((x, y))
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    plants.append((x, y, _freeze_mapping(tile)))
                elif isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                    animals.append((x, y, _freeze_mapping(tile)))
                elif isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
                    empty_structures.append((x, y, _freeze_mapping(tile)))
                elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                    weeds.append((x, y))
        market = _get(obs, "market", {}) or {}
        positions = (tuple(me["farmer"]),) + tuple(tuple(p) for p in me.get("hands", []))
        return View(
            player, step, day, hour, max(0, 720 - step), float(me.get("money", 0)),
            _freeze_mapping(board), tuple(me["farmer"]), tuple(tuple(p) for p in me.get("hands", [])), positions,
            _freeze_mapping(_get(private, "inventories", []) or []), _freeze_mapping(_get(private, "shed", {}) or {}),
            _freeze_mapping(_get(private, "seeds", {}) or {}), tuple(me.get("unlocked_quadrants", [])),
            int(me.get("hires_today", 0)), _freeze_mapping(_get(market, "inventory", {}) or {}),
            _freeze_mapping(_get(market, "prices", {}) or {}), tuple(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or []),
            _freeze_mapping({"money": other.get("money", 0), "tiles": other.get("tiles", []),
                             "farmer": other.get("farmer", []), "hands": other.get("hands", []),
                             "unlocked_quadrants": other.get("unlocked_quadrants", []),
                             "hires_today": other.get("hires_today", 0)}),
            tuple(plants), tuple(animals), tuple(empty_structures), tuple(empties), tuple(weeds),
            tuple(_shed_tiles(len(board))))


@dataclass(frozen=True)
class DayPlan:
    phase: str
    target_hands: int
    target_quadrants: int
    crop_weights: tuple
    animal_targets: tuple
    sell_floors: tuple
    liquidating: bool
    reserve_cash: float


class DefaultSeason:
    @staticmethod
    def plan_day(view):
        phase = "bootstrap" if view.day < 5 else "scale" if view.day < 14 else "harvest" if view.day < 25 else "liquidate"
        shops = set(view.town_shops)
        weights = ({"WHEAT": 0.80, "MELON": 0.20} if view.day < 5 else dict(config["crop_share"]))
        if "PET_CAFE" in shops: weights["CARROT"] = weights.get("CARROT", 0) + 0.20
        if "FARMERS_MARKET" in shops: weights["CARROT"] = weights.get("CARROT", 0) + 0.10
        target_hands = max(hands for first_day, hands in config["hire_schedule"] if first_day <= view.day)
        animal_total = min(config["max_animals"], config["animal_ramp"][0] + config["animal_ramp"][1] * view.day)
        animal_targets = {kind: 0 for kind in ANIMAL_TARGETS}
        remaining = animal_total
        while remaining:
            advanced = False
            for kind, cap in ANIMAL_TARGETS.items():
                if animal_targets[kind] < cap and remaining:
                    animal_targets[kind] += 1
                    remaining -= 1
                    advanced = True
            if not advanced: break
        return DayPlan(phase, target_hands, min(3, 1 + view.day // 4), tuple(sorted(weights.items())),
                       tuple(sorted(animal_targets.items())),
                       tuple(sorted(RESERVATION.items())), view.day >= config["liquidate_day"], config["money_reserve"])


class DefaultLabor:
    @staticmethod
    def hire_count(view, plan):
        return max(0, min(10, plan.target_hands - len(view.hands))) if view.hour <= 1 else 0

    @staticmethod
    def dispatch(view, plan):
        return {}  # coordinator's deterministic task matcher is the default dispatcher


def _labor_dict(value):
    """Thaw the contract's immutable mapping representation."""
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                         for item in value):
            return {key: _labor_dict(item) for key, item in value}
        return tuple(_labor_dict(item) for item in value)
    return value


class LaborB1:
    """Work-priced hiring and deterministic persistent spatial-cluster dispatch.

    Unit ids are stable for a day, so mapping every board coordinate to the nearest
    fixed anchor gives each unit a persistent compact cluster without mutable state.
    All decisions remain a pure function of the contract view.
    """

    FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)
    PHASE_CAPS = ((0, 6), (3, 8), (6, 10), (10, 12))
    PRODUCTIVE_PER_UNIT_DAY = 10.5  # 44% of 24 slots, the V46/top-field throughput level
    TASK_VALUE = 24.0               # conservative blended marginal dollars/productive action
    PLANT_CUTOFF_HOUR = 16
    WATER_DEADLINE_HOUR = 20

    # Twelve compact homes, ordered to cover the starting NW field first and then
    # newly unlocked quadrants.  A task always belongs to the same active unit id.
    ANCHORS = ((4, 4), (2, 4), (4, 2), (2, 2), (0, 4), (4, 0),
               (7, 4), (4, 7), (7, 2), (2, 7), (7, 7), (9, 4), (4, 9))

    @staticmethod
    def _cap(day):
        return [cap for first, cap in LaborB1.PHASE_CAPS if first <= day][-1]

    @staticmethod
    def _work(view, plan):
        plants = len(view.plants)
        occupied = 0
        urgent = 0
        for _x, _y, frozen in view.animals:
            tile = _labor_dict(frozen)
            if tile.get("animal") in ANIMALS:
                occupied += 1
                urgent += (not tile.get("fed_today", False)) + (not tile.get("cared_today", False))
                urgent += bool(tile.get("fertilizer_available", False))
                urgent += int(tile.get("yield_units", 0) or 0) > 0
        seeds = dict(_labor_dict(view.seeds) or ())
        plantable = min(len(view.empty_tiles), sum(int(seeds.get(c, 0) or 0) for c in CROPS) + 8)
        target_animals = sum(dict(plan.animal_targets).values())
        setup = min(8, max(0, target_animals - occupied) * 2)
        # Water/harvest average just under one action per plant-day; occupied animals
        # create feed, care, fertilizer and periodic harvest work every day.
        return 0.90 * plants + 3.35 * occupied + plantable + setup + len(view.weeds) + 0.5 * urgent

    @staticmethod
    def hire_count(view, plan):
        if view.hour > 2 or (view.day >= FINAL_DAY and not HIRE_ON_LAST_DAY):
            return 0
        work = LaborB1._work(view, plan)
        if work <= 0:
            return 0
        cap = min(LaborB1._cap(view.day), plan.target_hands)
        # Four hands is the bootstrap floor once productive assets are planned; above
        # that, one unit is bought per 10.5 expected productive actions.
        target = min(cap, max(4, int(math.ceil(work / LaborB1.PRODUCTIVE_PER_UNIT_DAY)) - 1))
        hires = int(view.hires_today)
        money = float(view.money)
        reserve = max(float(plan.reserve_cash), 120.0)
        result = 0
        while hires + result < target and hires + result < len(LaborB1.FIB):
            index = hires + result
            wage = LaborB1.FIB[index]
            remaining_work = max(0.0, work - (index + 1) * LaborB1.PRODUCTIVE_PER_UNIT_DAY)
            marginal_actions = min(LaborB1.PRODUCTIVE_PER_UNIT_DAY,
                                   max(1.0, remaining_work))
            if wage >= marginal_actions * LaborB1.TASK_VALUE or money - wage < reserve:
                break
            money -= wage
            result += 1
        return result

    @staticmethod
    def _move(pos, target):
        return _step_toward(pos[0], pos[1], target[0], target[1]) or ["PASS"]

    @staticmethod
    def _owner(pos, n_units):
        n = min(n_units, len(LaborB1.ANCHORS))
        return min(range(n), key=lambda i: (abs(pos[0] - LaborB1.ANCHORS[i][0])
                                            + abs(pos[1] - LaborB1.ANCHORS[i][1]), i))

    @staticmethod
    def _assign(positions, inventories, tasks):
        """Assign the coordinator's complete immutable job graph by cluster and range."""
        actions = {}
        available = set(range(len(positions)))
        pending = list(range(len(tasks)))
        for idx in tuple(sorted(available)):
            exact = [ti for ti in pending if tasks[ti][1] == positions[idx]
                     and (tasks[ti][3] is None or inventories[idx].get(tasks[ti][3], 0) > 0)
                     and (tasks[ti][4] is None or inventories[idx].get(tasks[ti][4], 0) <= 0)]
            if exact:
                ti = min(exact, key=lambda j: (tasks[j][0], j))
                actions[idx] = list(tasks[ti][2])
                available.remove(idx)
                pending.remove(ti)
        while available and pending:
            choices = []
            for ti in pending:
                priority, target, _action, need, skip = tasks[ti]
                owner = LaborB1._owner(target, len(positions))
                for idx in available:
                    if need is not None and inventories[idx].get(need, 0) <= 0:
                        continue
                    if skip is not None and inventories[idx].get(skip, 0) > 0:
                        continue
                    distance = abs(positions[idx][0] - target[0]) + abs(positions[idx][1] - target[1])
                    cluster_cost = 0 if idx == owner else 1
                    choices.append((priority, cluster_cost, distance, idx, ti))
            if not choices:
                break
            _priority, _cluster, _distance, idx, ti = min(choices)
            target = tasks[ti][1]
            actions[idx] = LaborB1._move(positions[idx], target)
            available.remove(idx)
            pending.remove(ti)
        if tasks:
            targets = [task[1] for task in tasks]
            for idx in sorted(available):
                moving = [(abs(positions[idx][0] - p[0]) + abs(positions[idx][1] - p[1]), p)
                          for p in targets if p != positions[idx]]
                if moving:
                    actions[idx] = LaborB1._move(positions[idx], min(moving)[1])
        return actions

    @staticmethod
    def dispatch(view, plan):
        positions = tuple(tuple(p) for p in view.unit_positions)
        if not positions:
            return {}
        inventories = tuple(dict(_labor_dict(inv) or ()) for inv in view.unit_inventories)
        inventories += ({},) * max(0, len(positions) - len(inventories))
        if view.jobs:
            tasks = [(int(priority), tuple(target), tuple(action), need, skip)
                     for priority, target, action, need, skip in view.jobs]
            return LaborB1._assign(positions, inventories, tasks)
        shed = dict(_labor_dict(view.shed) or ())
        seeds = dict(_labor_dict(view.seeds) or ())
        day, hour = view.day, view.hour
        last_day = day >= FINAL_DAY
        tasks = []

        def add(priority, pos, action, need=None, skip=None):
            tasks.append((priority, tuple(pos), action, need, skip))

        # Required priority: harvest > water > plant > care/feed > deposit.  A crop
        # facing its second dry night, or any dry crop after h16, becomes deadline zero.
        for x, y, frozen in view.plants:
            tile = _labor_dict(frozen)
            crop = tile.get("crop")
            cd = CROPS.get(crop)
            if cd is None:
                continue
            age = day - int(tile.get("planted_day", day))
            yu = int(tile.get("yield_units", 0) or 0)
            watered = bool(tile.get("watered_today", False))
            dry = int(tile.get("consecutive_unwatered", 0) or 0)
            viable = day + max(0, cd["first_yield_day"] - age) <= FINAL_DAY
            gains = (not cd["ongoing"] and WINDOW_START[crop] <= age <= cd["max_yield_day"]
                     and yu < cd["max_yield"])
            if cd["ongoing"]:
                ready = yu > 0 and age >= cd["first_yield_day"]
            else:
                stalled = day + max(0, TARGET_AGE[crop] - age) > FINAL_DAY
                ready = (yu > 0 and age >= cd["first_yield_day"]
                         and (age >= TARGET_AGE[crop] or stalled or age > cd["max_yield_day"]))
            if ready and (watered or not gains or hour >= 18 or last_day):
                add(0, (x, y), ["HARVEST"])
            # One dry day is safe. Service yield windows and second-miss deadlines
            # immediately; the late sweep catches any remaining viable crop by h20.
            if not watered and viable and not last_day and (dry >= 1 or gains or hour >= 16):
                priority = -1 if dry >= 1 or hour >= LaborB1.PLANT_CUTOFF_HOUR else 1
                add(priority, (x, y), ["WATER"])

        occupied = []
        empty_structures = [(x, y, _labor_dict(frozen))
                            for x, y, frozen in view.empty_structures]
        for x, y, frozen in view.animals:
            tile = _labor_dict(frozen)
            animal = tile.get("animal")
            if animal not in ANIMALS:
                continue
            occupied.append((x, y, tile))
            spec = ANIMALS[animal]
            yu = int(tile.get("yield_units", 0) or 0)
            if yu > 0 and (yu >= spec["max_held"] or yu >= ANIMAL_HARVEST_MIN or plan.liquidating):
                add(0, (x, y), ["HARVEST"])
            if not last_day and not tile.get("fed_today", False):
                add(-2 if int(tile.get("consecutive_unfed", 0) or 0) >= 1 else 3,
                    (x, y), ["FEED"], "WHEAT")
            if not last_day and not tile.get("cared_today", False):
                add(3, (x, y), ["CARE"])
            if tile.get("fertilizer_available", False):
                add(3, (x, y), ["COLLECT_FERTILIZER"])

        # Reserve near-shed sites for missing animal structures; all other empty tiles
        # are plant candidates selected by the planting module's day-plan weights.
        target_animals = sum(dict(plan.animal_targets).values())
        need_structures = max(0, target_animals - len(occupied) - len(empty_structures))
        reserved = set()
        if not last_day and need_structures:
            for pos in _coop_sites(len(view.tiles)):
                if pos in view.empty_tiles:
                    reserved.add(pos)
                    add(2, pos, ["BUILD_PASTURE"])
                    if len(reserved) >= need_structures:
                        break

        if not last_day and hour <= LaborB1.PLANT_CUTOFF_HOUR:
            live = {}
            for _x, _y, frozen in view.plants:
                crop = _labor_dict(frozen).get("crop")
                live[crop] = live.get(crop, 0) + 1
            weights = dict(plan.crop_weights)
            remaining = {crop: int(seeds.get(crop, 0) or 0) for crop in weights}
            planned = {crop: live.get(crop, 0) for crop in weights}
            for pos in sorted((p for p in view.empty_tiles if p not in reserved),
                              key=lambda p: _rank(len(view.tiles)).get(p, 999)):
                choices = [c for c in weights if remaining.get(c, 0) > 0
                           and day + TARGET_AGE[c] <= FINAL_DAY]
                if not choices:
                    break
                crop = min(choices, key=lambda c: (planned[c] / max(0.001, weights[c]), c))
                add(2, pos, ["PLANT", crop])
                remaining[crop] -= 1
                planned[crop] += 1
        for pos in view.weeds:
            add(2, pos, ["DIG"])

        carried_animals = {kind: [] for kind in ANIMALS}
        for idx, inv in enumerate(inventories):
            for kind in ANIMALS:
                if int(inv.get(kind, 0) or 0) > 0:
                    carried_animals[kind].append(idx)
        open_slots = list(empty_structures)
        carried_total = 0
        for kind in ANIMAL_TARGETS:
            for idx in carried_animals[kind]:
                if open_slots:
                    x, y, _tile = open_slots.pop(0)
                    add(2, (x, y), ["PLACE", kind], kind)
                    carried_total += 1
        if open_slots:
            for kind in ANIMAL_TARGETS:
                qty = min(len(open_slots), int(shed.get(kind, 0) or 0), 3)
                for access in view.shed_access_tiles[:qty]:
                    add(1, access, ["PICKUP", kind, 1], None, kind)
                open_slots = open_slots[qty:]

        unfed = sum(1 for _x, _y, tile in occupied if not tile.get("fed_today", False))
        feed_emergency = any(not tile.get("fed_today", False)
                             and int(tile.get("consecutive_unfed", 0) or 0) >= 1
                             for _x, _y, tile in occupied)
        carriers = sum(int(inv.get("WHEAT", 0) or 0) for inv in inventories)
        if not last_day and unfed > carriers and int(shed.get("WHEAT", 0) or 0) > 0:
            qty = min(int(shed.get("WHEAT", 0)), unfed - carriers)
            for access in view.shed_access_tiles[:min(3, qty)]:
                add(-2 if feed_emergency else 3, access,
                    ["PICKUP", "WHEAT", max(1, (qty + 2) // 3)], None, "WHEAT")

        # Deposit after higher-value field work, except final liquidation where reaching
        # the shed is the only way carried goods can still be sold.
        for idx, inv in enumerate(inventories):
            carried = sum(int(v or 0) for k, v in inv.items() if k not in ANIMALS)
            shed_pressure = sum(int(v or 0) for v in shed.values())
            if carried and (plan.liquidating or (hour >= 20 and carried >= 12)
                            or shed_pressure >= SHED_HARD_CAP):
                target = min(view.shed_access_tiles,
                             key=lambda p: (abs(positions[idx][0] - p[0]) + abs(positions[idx][1] - p[1]), p))
                add(1 if last_day else 4, target, ["DROP"], None, None)

        return LaborB1._assign(positions, inventories, tasks)


class DefaultPlanting:
    @staticmethod
    def choose(view, plan):
        weights = dict(plan.crop_weights)
        crops = sorted(weights, key=lambda c: (-weights[c], c))
        return {tile: crops[i % len(crops)] for i, tile in enumerate(view.empty_tiles)} if crops else {}


class DefaultMarket:
    @staticmethod
    def orders(view, plan):
        return []  # coordinator owns the shared cash budget; market plugin orders are prepended


season = DefaultSeason
labor = DefaultLabor
planting = DefaultPlanting
market = DefaultMarket

MODULE_REGISTRY = {
    "season": {"default": DefaultSeason},
    "labor": {"default": DefaultLabor, "labor_b1": LaborB1},
    "planting": {"default": DefaultPlanting},
    "market": {"default": DefaultMarket},
}


def register_module(role, name, implementation):
    """Development hook: registration is explicit; submitted defaults remain fixed."""
    if role not in MODULE_REGISTRY or not isinstance(name, str):
        raise ValueError("unknown Brain module role")
    MODULE_REGISTRY[role][name] = implementation


def _selected(role):
    return MODULE_REGISTRY[role][config["modules"][role]]

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

def _animal_work_value(a, tile, day, market_inventory):
    product_quote = market_price(a["product"], market_inventory.get(a["product"], MARKET_I0) + 2)
    wheat_quote = market_price("WHEAT", market_inventory.get("WHEAT", MARKET_I0) - 1)
    first = int(tile.get("placed_day", day)) + a["first_yield_day"]
    next_bonus_day = first
    # Today's care is banked AFTER tonight's production, so cannot pay tonight.
    while next_bonus_day <= day + 1:
        next_bonus_day += a["interval"]
    pending = int(tile.get("pending_care_bonus", 0))
    due = day + 1 >= first and (day + 1 - first) % a["interval"] == 0
    pending_after_production = 0 if due else pending
    care_worth = (next_bonus_day <= FINAL_DAY and pending_after_production < a["max_held"] - 1
                  and product_quote > 0.5 * wheat_quote + 35)
    usable_bonus = min(pending, max(0, a["max_held"] - int(tile.get("yield_units", 0)) - 1))
    feed_worth = care_worth or (due and usable_bonus * product_quote > wheat_quote)
    return care_worth, feed_worth


def _policy(obs, t0):
    # The observation is normalised once. Swappable modules see only this snapshot.
    view = View.from_obs(obs)
    plan = _selected("season").plan_day(view)
    planting_module = _selected("planting")
    labor_module = _selected("labor")
    market_module = _selected("market")
    module_planting = planting_module.choose(view, plan)
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
            elif kind in {ANIMALS[a]["structure"] for a in ANIMAL_TARGETS}:
                empty_coops.append((x, y))

    n_animals = len(animals)
    empty_set = set(empties)

    # ---- coop plan ---------------------------------------------------------------
    target_animals = dict(plan.animal_targets)
    target_coops = sum(target_animals.values())
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

        if not watered and viable and (unwatered >= 1 or gains or hour >= 16):
            if last_day:
                # Watering only pays on the last day if we are going to harvest the tile.
                if gains and age >= cd["first_yield_day"]:
                    add(0, x, y, ["WATER"])
            else:
                urgent = unwatered >= 1 or hour >= 16 or age >= tgt
                add(-1 if urgent else 1, x, y, ["WATER"])

        if harvest:
            # Harvest after the day's watering so the last bonus unit is not lost.
            if watered or not gains or hour >= 18 or (last_day and hour >= DROP_HOUR - 3):
                add(0, x, y, ["HARVEST"])

    # ---- animals -----------------------------------------------------------------
    for (x, y, t) in animals:
        a = ANIMALS.get(t.get("animal"), ANIMALS["GOOSE"])
        fed = bool(t.get("fed_today", False))
        cared = bool(t.get("cared_today", False))
        unfed = int(t.get("consecutive_unfed", 0))
        yu = int(t.get("yield_units", 0))
        care_worth, feed_worth = _animal_work_value(a, t, day, mkt_inv)
        # Day 29 never runs an end-of-day refresh, so feeding and caring are worthless then.
        if not last_day and not fed and (unfed >= 1 or feed_worth):
            add(-2 if unfed >= 1 else (-1 if hour >= 12 else 3), x, y, ["FEED"], need="WHEAT")
        if t.get("fertilizer_available", False):
            add(3, x, y, ["COLLECT_FERTILIZER"])
        if yu > 0 and (yu >= a["max_held"] or yu >= ANIMAL_HARVEST_MIN or day >= LIQUIDATE_DAY):
            add(0, x, y, ["HARVEST"])
        if not last_day and not cared and care_worth:
            add(-1 if hour >= 16 else 3, x, y, ["CARE"])

    # ---- animal placement / coop building ----------------------------------------
    if not last_day:
        slots = len(empty_coops)
        slot_index = 0
        total_carried = 0
        for animal in ANIMAL_TARGETS:
            count = carried.get(animal, 0)
            total_carried += count
            for _ in range(count):
                if slot_index >= slots: break
                x, y = empty_coops[slot_index]
                add(1, x, y, ["PLACE", animal], need=animal)
                slot_index += 1
        sheds = _shed_tiles(n)
        pick_index = 0
        for animal in ANIMAL_TARGETS:
            want_pick = min(max(0, slots - total_carried - pick_index), int(shed.get(animal, 0)))
            for _ in range(max(0, min(want_pick, 3))):
                sx, sy = sheds[pick_index % len(sheds)]
                add(1, sx, sy, ["PICKUP", animal, 1], skip=animal)
                pick_index += 1
        for (x, y) in build_sites:
            add(1, x, y, ["BUILD_PASTURE"])

    # ---- wheat logistics ---------------------------------------------------------
    if not last_day:
        need_wheat = 0
        for (_x, _y, t) in animals:
            a = ANIMALS.get(t.get("animal"), ANIMALS["GOOSE"])
            _care_worth, feed_worth = _animal_work_value(a, t, day, mkt_inv)
            if not t.get("fed_today", False) and (int(t.get("consecutive_unfed", 0)) >= 1 or feed_worth):
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
        crop_share = dict(plan.crop_weights)
        budget = dict((c, int(seeds.get(c, 0))) for c in crop_share)
        planned = dict((c, live.get(c, 0)) for c in crop_share)
        for (x, y) in plantable[:60]:
            best, best_score = None, None
            preferred = module_planting.get((x, y)) if config["modules"]["planting"] != "default" else None
            candidates = [preferred] if preferred in crop_share else crop_share
            for c in candidates:
                if budget.get(c, 0) <= 0:
                    continue
                if day + TARGET_AGE[c] > FINAL_DAY:
                    continue
                score = planned[c] / max(0.001, crop_share[c])
                if best_score is None or score < best_score:
                    best, best_score = c, score
            if best is None:
                break
            budget[best] -= 1
            planned[best] += 1
            add(2, x, y, ["PLANT", best])

        for (x, y) in sorted(weeds, key=lambda p: rank.get(p, 999))[:24]:
            add(2, x, y, ["DIG"])

    # Fertilizer is an investment: use it only when the next three days of
    # extra harvest value repay the unit's market value and delivery effort.
    fertilizer_jobs = []
    fertilizer_quote = market_price('FERTILIZER', mkt_inv.get('FERTILIZER', MARKET_I0) - 1)
    for x, y, tile in plants:
        crop = tile['crop']
        spec = CROPS[crop]
        age = day - int(tile['planted_day'])
        if tile.get('fertilized_until_day', -1) >= day or day >= FINAL_DAY:
            continue
        extra = 0
        for offset in range(3):
            if day + offset >= FINAL_DAY:
                continue
            if spec['ongoing']:
                production_age = age + offset + 1 - spec['first_yield_day']
                if production_age >= 0 and production_age % spec['interval'] == 0:
                    if production_age // spec['interval'] < spec['max_yield']:
                        extra += 1
            elif (not (offset == 0 and tile.get('watered_today'))
                  and WINDOW_START[crop] <= age + offset <= TARGET_AGE[crop]):
                extra += 1
        if not spec['ongoing']:
            normal_remaining = max(0, TARGET_AGE[crop] - max(age, WINDOW_START[crop]) + 1)
            extra = min(extra, max(0, spec['max_yield'] - tile.get('yield_units', 0) - normal_remaining))
        price = market_price(crop, mkt_inv.get(crop, MARKET_I0) + 12)
        if extra * price > fertilizer_quote * 1.4 + 50:
            fertilizer_jobs.append((x, y))
            add(-1, x, y, ['FERTILIZE'], need='FERTILIZER')
    fertilizer_reserve = min(8, len(fertilizer_jobs))
    fertilizer_carried = carried.get('FERTILIZER', 0)
    if fertilizer_reserve > fertilizer_carried and shed.get('FERTILIZER', 0) > 0:
        amount = min(3, fertilizer_reserve - fertilizer_carried, shed['FERTILIZER'])
        for sx, sy in _shed_tiles(n)[:2]:
            add(1, sx, sy, ['PICKUP', 'FERTILIZER', amount], skip='FERTILIZER')

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
    while idx < total and free and not budget_hit and not (labor_module is LaborB1 and not (last_day and hour >= DROP_HOUR)):
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

    # B1 owns the complete labor graph (including feed, pickup/place and deposit), so
    # its returned indices replace the generic matcher without erasing coordinator jobs.
    # Other experimental dispatchers remain sparse overlays for contract compatibility.
    use_b1_jobs = labor_module is LaborB1 and not (last_day and hour >= DROP_HOUR)
    if use_b1_jobs:
        immutable_jobs = tuple((prio, (x, y), tuple(action), need, skip)
                               for prio, x, y, action, need, skip in tasks)
        module_actions = labor_module.dispatch(replace(view, jobs=immutable_jobs), plan)
    elif labor_module is LaborB1:
        module_actions = {}
    else:
        module_actions = labor_module.dispatch(view, plan)
    if isinstance(module_actions, dict):
        if use_b1_jobs:
            actions = {}
        for unit_id, action in module_actions.items():
            if isinstance(unit_id, int) and 0 <= unit_id <= n_hands:
                actions[unit_id] = action

    # ---- market orders -----------------------------------------------------------
    plugin_orders = market_module.orders(view, plan)
    orders = list(plugin_orders[:MAX_MARKET_ORDERS]) if isinstance(plugin_orders, list) else []
    # Hands vanish at end of day and the fib hire counter resets with them, so we re-hire
    # every morning. Only 10 market orders fit in a turn, so a bigger crew spills to hour 1
    # (those hands lose one turn, which is cheap next to what a hand does in a day).
    if not last_day or HIRE_ON_LAST_DAY:
        n_hire = labor_module.hire_count(view, plan)
        for _ in range(n_hire):
            if len(orders) < MAX_MARKET_ORDERS:
                orders.append(["HIRE"])

    slots = MAX_MARKET_ORDERS - len(orders)
    if slots > 0:
        wheat_reserve = 0 if day >= LIQUIDATE_DAY else (n_animals + WHEAT_FEED_BUFFER)
        sale_shed = dict(shed)
        sale_shed['FERTILIZER'] = max(0, sale_shed.get('FERTILIZER', 0) - fertilizer_reserve)
        orders.extend(_sell_orders(sale_shed, mkt_inv, day, wheat_reserve, slots))

    # Buy order matters: orders execute in list order and each one spends the bank the
    # earlier ones left behind. Feed first -- an unfed animal escapes after two days and
    # takes its $300 and ~$150/day with it, which is what a starved opening costs us.
    buys = []
    if not last_day and day < LIQUIDATE_DAY:
        live_counts = {a: sum(t.get("animal") == a for _x, _y, t in animals) for a in ANIMAL_TARGETS}
        pending_counts = {a: int(shed.get(a, 0)) + carried.get(a, 0) for a in ANIMAL_TARGETS}
        pending = sum(pending_counts.values())
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
        crop_share = dict(plan.crop_weights)
        for c in crop_share:
            have = int(seeds.get(c, 0))
            if day + TARGET_AGE[c] > FINAL_DAY:
                continue
            want = min(SEED_CAP.get(c, 12), int(open_tiles * crop_share[c]) + 4)
            if have >= max(4, want // 2):
                continue
            k = want - have
            if k > 0 and money - k * CROPS[c]["seed"] >= MONEY_RESERVE:
                buys.append(["BUY_SEED", c, k])

        # 3. Animals -- but never more mouths than we can already feed.
        if day <= LAST_ANIMAL_DAY and shed_room > 0 and wheat_have >= mouths + 1:
            remaining_cash = money - MONEY_RESERVE
            remaining_room = min(4, shed_room, max(0, wheat_have - mouths))
            for animal in ANIMAL_TARGETS:
                want = max(0, target_animals.get(animal, 0) - live_counts[animal] - pending_counts[animal])
                afford = int(remaining_cash // ANIMALS[animal]["cost"])
                k = min(want, afford, remaining_room)
                if k > 0:
                    buys.append(["BUY_ANIMAL", animal, k])
                    remaining_cash -= k * ANIMALS[animal]["cost"]
                    remaining_room -= k

        # 4. Land last: tiles are worthless without the seed and the hands to work them.
        n_extra = len(unlocked) - 1
        if (0 <= n_extra < 2 and len(unlocked) < plan.target_quadrants and LAND_FIRST_DAY <= day <= LAND_LAST_DAY
                and money >= LAND_TRIGGER[n_extra]):
            buys.append(["BUY_LAND"])

    fertilizer_deficit = fertilizer_reserve - fertilizer_carried - shed.get('FERTILIZER', 0)
    if fertilizer_deficit > 0 and money > MONEY_RESERVE + fertilizer_deficit * fertilizer_quote:
        buys.append(['BUY_PRODUCT', 'FERTILIZER', fertilizer_deficit])

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


def _kpi_record(obs, action):
    """Optional policy-neutral counters; never called unless BRAIN_KPI=1."""
    global _KPI
    player = int(_get(obs, "player", 0))
    step = int(_get(obs, "day", 0)) * TURNS_PER_DAY + int(_get(obs, "hour", 0))
    farm = _get(obs, "farms", [])[player]
    private = _get(obs, "private", {}) or {}
    if _KPI is None or step == 0:
        _KPI = {"player": player, "eligible_plant_days": 0, "water_actions": 0, "plants_lost": 0,
                "productive_actions": 0, "unit_slots": 0, "no_op_count": 0, "idle_unit_slots": 0,
                "sold_units": {p: 0 for p in PRODUCTS}, "realised_dollars": {p: 0 for p in PRODUCTS},
                "town_demand": {p: 0 for p in PRODUCTS}, "wages_paid": 0, "bank_turn_72": None,
                "last_plants": set(), "timing_ms": []}
    k = _KPI
    current_plants = set()
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                current_plants.add((x, y))
    if step % 24 == 0:
        k["eligible_plant_days"] += len(current_plants)
        if step > 0:
            for pos in k["last_plants"] - current_plants:
                x, y = pos
                tile = farm["tiles"][y][x]
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    k["plants_lost"] += 1
    k["last_plants"] = current_plants
    units = [action.get("farmer", ["PASS"])] + list(action.get("hands", []))
    productive = {"PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
                  "FEED", "COLLECT_FERTILIZER", "CARE", "PICKUP", "PLACE", "DROP"}
    for unit_action in units:
        op = unit_action[0] if isinstance(unit_action, list) and unit_action else "PASS"
        k["unit_slots"] += 1
        if op in productive: k["productive_actions"] += 1
        if op == "WATER": k["water_actions"] += 1
        if op == "PASS":
            k["no_op_count"] += 1
            k["idle_unit_slots"] += 1
    market_inv = dict(_get(_get(obs, "market", {}) or {}, "inventory", {}) or {})
    hire_index = int(farm.get("hires_today", 0))
    available_money = float(farm.get("money", 0))
    for order in action.get("market", []):
        if order and order[0] == "HIRE":
            wage = _fib(hire_index)
            if available_money < wage:
                continue
            k["wages_paid"] += wage
            available_money -= wage
            hire_index += 1
        elif len(order) >= 3 and order[0] == "SELL" and order[1] in PRODUCTS:
            item, qty = order[1], int(order[2])
            revenue, _ = predict_sale(item, market_inv.get(item, MARKET_I0), qty)
            k["sold_units"][item] += qty
            k["realised_dollars"][item] += revenue
    for item, amount in town_drain(step, _get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or []).items():
        k["town_demand"][item] += amount
    if step == 72: k["bank_turn_72"] = float(farm.get("money", 0))
    if step >= 718:  # engine requests actions for observation steps 0..718
        shed = dict(_get(private, "shed", {}) or {})
        pending_sell = {o[1]: int(o[2]) for o in action.get("market", []) if len(o) >= 3 and o[0] == "SELL"}
        unsold = 0
        prices = dict(_get(_get(obs, "market", {}) or {}, "prices", {}) or {})
        for item in PRODUCTS:
            carried = sum(int(inv.get(item, 0)) for inv in (_get(private, "inventories", []) or []))
            unsold += max(0, int(shed.get(item, 0)) - pending_sell.get(item, 0)) * int(prices.get(item, 0))
            unsold += carried * int(prices.get(item, 0))
        bases = {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS}
        summary = {key: value for key, value in k.items() if key not in ("last_plants", "timing_ms")}
        summary["watered_fraction_of_eligible_plant_days"] = round(k["water_actions"] / max(1, k["eligible_plant_days"]), 4)
        summary["productive_action_fraction"] = round(k["productive_actions"] / max(1, k["unit_slots"]), 4)
        summary["realised_over_base"] = {p: round(k["realised_dollars"][p] / max(1, k["sold_units"][p] * bases[p]), 4) for p in PRODUCTS}
        total_sales = sum(k["sold_units"].values())
        total_demand = sum(k["town_demand"].values())
        summary["town_demand_share"] = round(total_sales / max(1, total_demand), 4)
        summary["unsold_value_at_end"] = unsold
        summary["final_bank_before_last_orders"] = float(farm.get("money", 0))
        out = os.environ.get("BRAIN_KPI_PATH") or os.path.join(os.path.dirname(__file__), "kpi-%s-%s.json" % (os.getpid(), player))
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)


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
    if os.environ.get("BRAIN_KPI") == "1":
        try:
            _kpi_record(obs, action)
        except Exception:
            pass
    return action
