"""Execute a day-level plan through the real Kaggriculture interpreter.

The public entry point is ``execute(schedule, seed, workers)``.  ``workers`` is
accepted for orchestration compatibility; a fixed instance is deliberately one
game and never starts a worker pool.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fixed_game.pin import load_instance, make_env  # noqa: E402


REFERENCE_ROUTER_COMMIT = "7baa291"


def _load_reference_router():
    """Load the round-1 router verbatim, independent of the mutable bot path."""
    source = subprocess.check_output(
        ["git", "show", f"{REFERENCE_ROUTER_COMMIT}:bots/exec_b/main.py"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType("fixed_game_exec_b_7baa291")
    exec(compile(source, f"git:{REFERENCE_ROUTER_COMMIT}:bots/exec_b/main.py", "exec"), module.__dict__)
    return module


routing = _load_reference_router()

PRODUCTS = routing.PRODUCTS
CROPS = routing.CROP_NAMES
ANIMALS = routing.ANIMAL_NAMES
SHED = routing.SHED_TILES
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200,
              "FERTILIZER": 100}
ACTING = {"DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE",
          "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"}
MOVING = {"NORTH", "SOUTH", "EAST", "WEST"}


def _blank_day(day: int) -> dict:
    return {"day": day, "hires": 0, "purchases": [], "builds": [], "plantings": [], "sales": []}


def validate_schedule(schedule: dict) -> dict:
    if not isinstance(schedule, dict) or "days" not in schedule:
        raise ValueError("schedule must be an object containing days")
    by_day = {int(d["day"]): copy.deepcopy(d) for d in schedule["days"]}
    days = []
    for day in range(30):
        entry = _blank_day(day) | by_day.get(day, {})
        entry["day"] = day
        entry["hires"] = max(0, int(entry.get("hires", 0)))
        for key in ("purchases", "builds", "plantings", "sales"):
            entry[key] = list(entry.get(key, []) or [])
        days.append(entry)
    out = copy.deepcopy(schedule)
    out["days"] = days
    return out


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _inv_totals(obs):
    private = obs.get("private", {}) or {}
    total = Counter(private.get("shed", {}) or {})
    for inv in private.get("inventories", []) or []:
        total.update(inv or {})
    return total


class Executor:
    def __init__(self, schedule: dict):
        self.schedule = validate_schedule(schedule)
        self.days = self.schedule["days"]
        self.actions = []
        self.times = []
        self.faults = 0
        self.substitutions = []
        self.turn_counts = defaultdict(Counter)
        self.produced = defaultdict(Counter)
        self.sold = defaultdict(Counter)
        self.revenue = defaultdict(Counter)
        self.rotted = defaultdict(Counter)
        self.bank = {}
        self.snapshots = {}
        self._last_obs = None
        self._plant_done = defaultdict(set)
        self._build_done = defaultdict(set)
        self._issued_day = None

    def _observe_transition(self, obs):
        if self._last_obs is None:
            self._last_obs = copy.deepcopy(obs)
            return
        prev = self._last_obs
        pday = int(prev.get("day", 0))
        # A disappearing/reduced field yield without our preceding harvest is decay.
        last = self.actions[-1] if self.actions else {}
        acted = {}
        positions = [tuple(prev["farms"][0]["farmer"])] + [tuple(p) for p in prev["farms"][0].get("hands", [])]
        unit_actions = [last.get("farmer", ["PASS"])] + list(last.get("hands", []))
        for pos, action in zip(positions, unit_actions):
            if action and action[0] == "HARVEST":
                acted[pos] = True
        old_board, new_board = prev["farms"][0]["tiles"], obs["farms"][0]["tiles"]
        for y in range(10):
            for x in range(10):
                a, b = old_board[y][x], new_board[y][x]
                if not isinstance(a, dict) or (x, y) in acted:
                    continue
                old_yield = int(a.get("yield_units", 0) or 0)
                new_yield = int(b.get("yield_units", 0) or 0) if isinstance(b, dict) else 0
                if new_yield < old_yield:
                    product = a.get("crop") or routing.ANIMALS.get(a.get("animal"), {}).get("product")
                    if product:
                        self.rotted[pday][product] += old_yield - new_yield
        self._last_obs = copy.deepcopy(obs)

    def _capital_orders(self, obs, day):
        spec = self.days[day]
        orders = []
        # The format defines hires as the desired hands for that day.
        have_hands = len(obs["farms"][0].get("hands", []))
        for _ in range(max(0, spec["hires"] - have_hands)):
            orders.append(["HIRE"])
        for purchase in spec["purchases"]:
            item, qty = str(purchase["item"]), int(purchase["qty"])
            if qty <= 0:
                continue
            if item.endswith("_SEED"):
                orders.append(["BUY_SEED", item[:-5], qty])
            elif item in ANIMALS:
                orders.append(["BUY_ANIMAL", item, qty])
            elif item == "LAND":
                orders.extend([["BUY_LAND"] for _ in range(qty)])
            elif item == "FERTILIZER":
                orders.append(["BUY_PRODUCT", item, qty])
        # Capital orders are issued once, in exact schedule order.  Feed is the only
        # schema-external operational purchase and follows them.
        if self._issued_day != day:
            self._issued_day = day
            # Animal orders use the persistent, reserve-aware path below.
            result = [o for o in orders if o[0] != "BUY_ANIMAL"][:10]
        else:
            result = []
        farm, private = obs["farms"][0], obs.get("private", {}) or {}
        # Retry animal capital that an earlier order could only partially fill.
        # The engine processes quantity unit-by-unit, so this closes the silent
        # liquidity shortfall without duplicating filled purchases.
        live_or_stored = Counter(private.get("shed", {}) or {})
        for inv in private.get("inventories", []) or []:
            live_or_stored.update(inv or {})
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                    live_or_stored[tile["animal"]] += 1
        cumulative = Counter()
        for prior in self.days[:day + 1]:
            for purchase in prior["purchases"]:
                if purchase["item"] in ANIMALS:
                    cumulative[purchase["item"]] += int(purchase["qty"])
        available_money = float(farm.get("money", 0))
        feed_reserve = (sum(live_or_stored[a] for a in ANIMALS) + 4) * float(obs["market"]["prices"]["WHEAT"])
        animal_cost = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
        for animal in ANIMALS:
            deficit = cumulative[animal] - live_or_stored[animal]
            affordable = max(0, int((available_money - feed_reserve) // animal_cost[animal]))
            qty = min(deficit, affordable)
            if qty > 0 and len(result) < 10:
                result.append(["BUY_ANIMAL", animal, qty])
                available_money -= qty * animal_cost[animal]
        animals = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") in ANIMALS)
        wheat = int((private.get("shed", {}) or {}).get("WHEAT", 0) or 0)
        wheat += sum(int((i or {}).get("WHEAT", 0) or 0) for i in private.get("inventories", []) or [])
        if day < 29 and animals and wheat < animals + 3 and len(result) < 10:
            result.insert(0, ["BUY_PRODUCT", "WHEAT", animals + 3 - wheat])
            result = result[:10]
        return result

    def _jobs(self, obs, day, hour):
        farm, private = obs["farms"][0], obs.get("private", {}) or {}
        board = farm["tiles"]
        positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]
        invs = [dict(x or {}) for x in private.get("inventories", [])]
        while len(invs) < len(positions):
            invs.append({})
        jobs = []
        add = lambda target, action, deadline, value, need=None, tag="", owner=None: jobs.append(
            routing._job(target, action, deadline, value, need, tag, owner))

        # Schedule-exact structures and plantings.  Completed jobs are recognized
        # from the real board; an occupied target gets the nearest legal substitute.
        reserved = set()
        # A structure has no cash cost or productive downside.  Look ahead over
        # the season's declared build sites and build an empty named site as soon
        # as a hand can reach it; this is the capital pre-positioning pass.
        declared_builds = [(d["day"], i, b) for d in self.days if d["day"] <= day
                           for i, b in enumerate(d["builds"])]
        for build_day, idx, build in declared_builds:
            key = (build_day, idx)
            if key in self._build_done[build_day]:
                continue
            requested = tuple(build["tile"])
            target = requested
            if not (0 <= target[0] < 10 and 0 <= target[1] < 10) or board[target[1]][target[0]] is not None:
                candidates = [(x, y) for y, row in enumerate(board) for x, t in enumerate(row) if t is None and (x, y) not in reserved]
                if candidates:
                    target = min(candidates, key=lambda p: (_dist(p, requested), _dist(p, SHED[0]), p))
                    marker = {"day": day, "kind": "build", "requested": list(requested), "actual": list(target)}
                    if marker not in self.substitutions:
                        self.substitutions.append(marker)
            if target and board[target[1]][target[0]] is None:
                reserved.add(target)
                add(target, ["BUILD_" + build["kind"]], 8, 18000, tag=f"build:{build_day}:{idx}")
        for idx, planting in enumerate(self.days[day]["plantings"]):
            key = (day, idx)
            if key in self._plant_done[day]:
                continue
            if int(((private.get("seeds", {}) or {}).get(planting["crop"], 0) or 0)) <= 0:
                continue
            requested = tuple(planting["tile"])
            target = requested
            if (not (0 <= target[0] < 10 and 0 <= target[1] < 10)
                    or board[target[1]][target[0]] is not None or target in reserved):
                candidates = [(x, y) for y, row in enumerate(board) for x, t in enumerate(row) if t is None and (x, y) not in reserved]
                if candidates:
                    target = min(candidates, key=lambda p: (_dist(p, requested), _dist(p, SHED[0]), p))
                    marker = {"day": day, "kind": "plant", "crop": planting["crop"], "requested": list(requested), "actual": list(target)}
                    if marker not in self.substitutions:
                        self.substitutions.append(marker)
            if target and board[target[1]][target[0]] is None:
                reserved.add(target)
                add(target, ["PLANT", planting["crop"]], 12, 12000, tag=f"plant:{idx}")

        animals = []
        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                pos = (x, y)
                if tile.get("kind") == "PLANT":
                    crop = tile["crop"]
                    age = day - int(tile.get("planted_day", day))
                    if not tile.get("watered_today", False):
                        add(pos, ["WATER"], 20, 2800 if tile.get("consecutive_unwatered", 0) else 1000, tag="water")
                    if age in routing.FERT_AGES.get(crop, ()) and crop != "WHEAT" and int(tile.get("fertilized_until_day", -1)) < day:
                        add(pos, ["FERTILIZE"], 16, 900, "FERTILIZER", "fertilize")
                    held = int(tile.get("yield_units", 0) or 0)
                    if held and (age >= routing.CROPS[crop]["last"] or crop in ("TOMATO", "STRAWBERRY")):
                        deadline = 13 if crop in ("MELON", "STRAWBERRY") else 19
                        add(pos, ["HARVEST"], deadline, 7000 + held * BASE_PRICE[crop], tag="harvest")
                    if age > routing.CROPS[crop]["last"] and not held:
                        add(pos, ["DIG"], 22, 350, tag="dig")
                elif tile.get("animal") in ANIMALS:
                    animals.append((pos, tile))
                elif tile.get("kind") == "WEED":
                    add(pos, ["DIG"], 22, 500, tag="dig")

        # Stable strips are ownership-affined, avoiding per-turn route oscillation.
        animals.sort(key=lambda z: (z[0][1], z[0][0]))
        for rank, (pos, tile) in enumerate(animals):
            owner = min(len(positions) - 1, rank * len(positions) // max(1, len(animals)))
            if not tile.get("fed_today", False):
                add(pos, ["FEED"], 20, 15000, "WHEAT", "feed", owner)
            if not tile.get("cared_today", False):
                add(pos, ["CARE"], 19, 6500, tag="care", owner=owner)
            if tile.get("fertilizer_available", False):
                add(pos, ["COLLECT_FERTILIZER"], 21, 8000, tag="collect", owner=owner)
            held = int(tile.get("yield_units", 0) or 0)
            if held:
                add(pos, ["HARVEST"], 17, 9000 + held * BASE_PRICE[routing.ANIMALS[tile["animal"]]["product"]], tag="harvest", owner=owner)

        totals = _inv_totals(obs)
        shed = Counter(private.get("shed", {}) or {})
        # Place bought animals into every compatible empty structure.
        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get("animal"):
                    continue
                if tile.get("kind") not in ("COOP", "PASTURE"):
                    continue
                choices = ("GOOSE",) if tile["kind"] == "COOP" else ("COW", "SHEEP")
                for animal in choices:
                    if totals[animal] > 0:
                        add((x, y), ["PLACE", animal], 14, 17000, animal, "place")
                        totals[animal] -= 1
                        break
        for animal in ANIMALS:
            carried = sum(int(i.get(animal, 0) or 0) for i in invs)
            if shed[animal] > 0 and carried == 0:
                add(min(SHED, key=lambda p: min(_dist(q, p) for q in positions)), ["PICKUP", animal, min(4, shed[animal])], 9, 19000, tag="pickup_animal")

        # Batch utility pickup, one load per hand; harvested cargo remains aboard
        # until it is valuable or the maturity-day same-day-sale cutoff approaches.
        hungry = sum(not t.get("fed_today", False) for _, t in animals)
        carriers = sum(int(i.get("WHEAT", 0) or 0) for i in invs)
        if hungry > carriers and shed["WHEAT"]:
            for ui, inv in enumerate(invs):
                if int(inv.get("WHEAT", 0) or 0) == 0:
                    target = min(SHED, key=lambda p: (_dist(positions[ui], p), p))
                    add(target, ["PICKUP", "WHEAT", min(shed["WHEAT"], max(2, math.ceil(hungry / len(positions))))], 7, 16000, tag="pickup_feed", owner=ui)
        fert_jobs = sum(j["tag"] == "fertilize" for j in jobs)
        if fert_jobs and totals["FERTILIZER"] < fert_jobs and shed["FERTILIZER"]:
            add(SHED[0], ["PICKUP", "FERTILIZER", fert_jobs], 10, 5000, tag="pickup_fert")
        for ui, inv in enumerate(invs):
            cargo = sum(int(inv.get(p, 0) or 0) * BASE_PRICE[p] for p in PRODUCTS)
            urgent = any(int(inv.get(p, 0) or 0) for p in ("MELON", "FERTILIZER"))
            if cargo >= 1200 or (cargo and hour >= (14 if urgent else 19)) or day == 29:
                target = min(SHED, key=lambda p: (_dist(positions[ui], p), p))
                add(target, ["DROP"], 16 if urgent else 21, 6000 + cargo, PRODUCTS, "drop", ui)
        return positions, invs, jobs

    def policy(self, obs, config=None):
        del config
        started = time.perf_counter()
        try:
            self._observe_transition(obs)
            day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
            positions, invs, jobs = self._jobs(obs, day, hour)
            unit_actions = routing._assign(positions, invs, jobs, hour)
            market = self._capital_orders(obs, day)
            shed = Counter((obs.get("private", {}) or {}).get("shed", {}) or {})
            targets = {s["product"]: int(s["qty"]) for s in self.days[day]["sales"]}
            default = not self.days[day]["sales"]
            for product in PRODUCTS:
                if len(market) >= 10:
                    break
                qty = shed[product] if default else min(shed[product], targets.get(product, 0))
                # Include same-turn drops: unit actions precede market orders.
                for ui, action in enumerate(unit_actions):
                    if action and action[0] == "DROP" and positions[ui] in SHED:
                        qty += int(invs[ui].get(product, 0) or 0)
                if qty > 0:
                    market.append(["SELL", product, qty])
            action = routing._sanitize(unit_actions, market, (obs.get("private", {}) or {}).get("seeds", {}), len(positions) - 1)
            # Mark schedule jobs only when the requested tile action is emitted.
            for pos, act in zip(positions, [action["farmer"]] + action["hands"]):
                if not act:
                    continue
                for job in jobs:
                    if pos == job["target"] and act == job["action"]:
                        if job["tag"].startswith("plant:"):
                            self._plant_done[day].add((day, int(job["tag"].split(":")[1])))
                        elif job["tag"].startswith("build:"):
                            _, build_day, build_idx = job["tag"].split(":")
                            self._build_done[int(build_day)].add((int(build_day), int(build_idx)))
                        if act[0] == "HARVEST":
                            tile = obs["farms"][0]["tiles"][pos[1]][pos[0]]
                            product = tile.get("crop") or routing.ANIMALS.get(tile.get("animal"), {}).get("product")
                            self.produced[day][product] += int(tile.get("yield_units", 0) or 0)
            for act in [action["farmer"]] + action["hands"]:
                op = act[0] if act else "PASS"
                self.turn_counts[day]["moving" if op in MOVING else "acting" if op in ACTING else "idle"] += 1
            for order in action["market"]:
                if order[0] == "SELL":
                    product, qty = order[1], min(int(order[2]), shed[order[1]])
                    self.sold[day][product] += qty
                    self.revenue[day][product] += qty * int(obs["market"]["prices"][product])
            self.actions.append(action)
        except Exception:
            self.faults += 1
            n = len(obs.get("farms", [{}])[0].get("hands", []))
            action = {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(n)], "market": []}
            self.actions.append(action)
        self.times.append((time.perf_counter() - started) * 1000)
        return action


class ReferenceExecutor(Executor):
    """Schedule adapter around the exact 7baa291 work router."""

    def __init__(self, schedule: dict, *, idle_fix: bool = True,
                 fertilizer_drain: int = 0, fertilizer_threshold: int = 10,
                 animal_sweep: bool = False, layout_mode: str = "scheduled"):
        super().__init__(schedule)
        self.router = _load_reference_router()
        self.idle_fix = idle_fix
        self.fertilizer_drain = int(fertilizer_drain)
        self.fertilizer_threshold = int(fertilizer_threshold)
        self.animal_sweep = animal_sweep
        self.layout_mode = layout_mode
        self.sale_policies = {str(k): str(v) for k, v in
                              (self.schedule.get("sale_policies", {}) or {}).items()}
        self._current_fertilizer_price = BASE_PRICE["FERTILIZER"]
        self.positive_job_idle = defaultdict(int)
        self._sales_issued = defaultdict(Counter)
        self._scheduled_buys_issued = set()
        self._configure_plan()

    def _configure_plan(self):
        planned_crop_tiles = defaultdict(list)
        planned_build_tiles = []
        for spec in self.days:
            for planting in spec["plantings"]:
                planned_crop_tiles[planting["crop"]].append(tuple(planting["tile"]))
            planned_build_tiles.extend(tuple(build["tile"]) for build in spec["builds"])
        original_layout_rank = self.router._layout_rank
        if self.layout_mode == "swap_melon_wheat":
            planned_crop_tiles["MELON"], planned_crop_tiles["WHEAT"] = (
                planned_crop_tiles["WHEAT"], planned_crop_tiles["MELON"])

        def scheduled_layout_rank(pos, crop=None):
            if self.layout_mode == "reference":
                return original_layout_rank(pos, crop)
            targets = planned_crop_tiles.get(crop, ()) if crop else planned_build_tiles
            if targets:
                return (min(self.router._dist(pos, target) for target in targets),
                        original_layout_rank(pos, crop))
            return (99, original_layout_rank(pos, crop))

        self.router._layout_rank = scheduled_layout_rank
        phase_days = {
            "opening": range(0, 10),
            "days_10_16": range(10, 17),
            "days_17_22": range(17, 23),
            "days_23_26": range(23, 30),
        }
        for phase, days in phase_days.items():
            counts = Counter(
                planting["crop"]
                for day in days
                for planting in self.days[day]["plantings"]
            )
            if counts:
                self.router.PLAN["crop_mix"][phase] = {
                    crop: int(counts[crop]) for crop in self.router.CROP_NAMES
                }

        animal_targets = {animal: [] for animal in self.router.ANIMAL_NAMES}
        cumulative = Counter()
        for day in range(30):
            for purchase in self.days[day]["purchases"]:
                if purchase.get("item") in self.router.ANIMAL_NAMES:
                    cumulative[purchase["item"]] += max(0, int(purchase.get("qty", 0)))
            for animal in self.router.ANIMAL_NAMES:
                animal_targets[animal].append(cumulative[animal])

        def exact_animal_target(kind, day):
            return animal_targets[kind][max(0, min(29, int(day)))]

        self.router._animal_target = exact_animal_target
        original_assign = self.router._assign

        def assigned_with_preposition(positions, inventories, jobs, hour):
            # Fertilizer has no sink on many pinned instances.  Once its quote no
            # longer clears the configured floor, collecting it spends a field
            # action and a route visit to manufacture glut.  A schedule can make
            # this unconditional with the ``never_collect`` policy.
            fert_policy = self.sale_policies.get("FERTILIZER", "sell_on_sight")
            collect_fertilizer = not (
                fert_policy == "never_collect"
                or (self.fertilizer_drain == 0
                    and self._current_fertilizer_price < self.fertilizer_threshold)
            )
            if not collect_fertilizer:
                jobs[:] = [job for job in jobs
                           if job.get("action", [None])[0] != "COLLECT_FERTILIZER"]

            # Keep each hand on one contiguous animal strip for the full sweep.
            # The reference divided ranks by all 25 NW tiles, leaving skewed
            # ownership when the herd occupied only part of the quadrant.
            if self.animal_sweep and positions:
                animal_targets = sorted(
                    {job["target"] for job in jobs
                     if job.get("tag") in ("feed", "care", "fert")},
                    key=lambda p: ((4 - p[1]) * 5
                                   + ((4 - p[0]) if (4 - p[1]) % 2 == 0 else p[0]), p),
                )
                owner_by_target = {
                    target: min(len(positions) - 1,
                                rank * len(positions) // max(1, len(animal_targets)))
                    for rank, target in enumerate(animal_targets)
                }
                for job in jobs:
                    if job.get("target") in owner_by_target and job.get("tag") in (
                            "feed", "care", "fert", "harvest"):
                        job["owner"] = owner_by_target[job["target"]]
            actions = original_assign(positions, inventories, jobs, hour)
            if not self.idle_fix or not jobs:
                return actions
            for ui, action in enumerate(actions):
                if action != self.router.SAFE:
                    continue
                local_jobs = [job for job in jobs if job.get("owner") == ui]
                candidates = sorted(
                    local_jobs,
                    key=lambda job: (
                        self.router._dist(positions[ui], job["target"]),
                        -job["value"],
                        job["target"],
                    ),
                )
                target = next((job["target"] for job in candidates
                               if job["target"] != positions[ui]), None)
                if target is not None:
                    actions[ui] = self.router._move(positions[ui], target)
                self.positive_job_idle[int(hour)] += actions[ui] == self.router.SAFE
            return actions

        self.router._assign = assigned_with_preposition

    def policy(self, obs, config=None):
        del config
        started = time.perf_counter()
        try:
            self._observe_transition(obs)
            day = int(obs.get("day", 0))
            self._current_fertilizer_price = int(
                obs.get("market", {}).get("prices", {}).get("FERTILIZER", 0) or 0)
            phase = min(5, day // 5)
            self.router.PLAN["hands_by_5day"][phase] = self.days[day]["hires"]
            action = self.router.agent(obs)

            if int(obs.get("hour", 0)) == 0 and day not in self._scheduled_buys_issued:
                for purchase in self.days[day]["purchases"]:
                    if purchase.get("item") in ("WHEAT", "FERTILIZER") and len(action["market"]) < 10:
                        action["market"].append(["BUY_PRODUCT", purchase["item"],
                                                 max(0, int(purchase.get("qty", 0)))])
                self._scheduled_buys_issued.add(day)

            # Optional daily sale targets replace the router's sell-all default.
            targets = {row["product"]: max(0, int(row["qty"]))
                       for row in self.days[day]["sales"]}
            if targets:
                market = [order for order in action["market"] if order[0] != "SELL"]
                shed = Counter((obs.get("private", {}) or {}).get("shed", {}) or {})
                for product, target in targets.items():
                    remaining = target - self._sales_issued[day][product]
                    qty = min(remaining, shed[product])
                    if qty > 0 and len(market) < 10:
                        market.append(["SELL", product, qty])
                        self._sales_issued[day][product] += qty
                action["market"] = market

            # Product-level policies are orthogonal to daily quantity targets.
            # ``hold_to_drain`` waits for the final day, while ``never_collect``
            # also suppresses fertilizer sales already sitting in the shed.
            filtered = []
            for order in action["market"]:
                if order[0] != "SELL":
                    filtered.append(order)
                    continue
                policy = self.sale_policies.get(order[1], "sell_on_sight")
                if policy in ("never", "never_collect"):
                    continue
                if policy == "hold_to_drain" and day < 29:
                    continue
                filtered.append(order)
            action["market"] = filtered

            # The reference router had no land order path; honour scheduled land.
            land_target = sum(
                int(purchase.get("qty", 0))
                for spec in self.days[:day + 1]
                for purchase in spec["purchases"]
                if purchase.get("item") == "LAND"
            )
            unlocked = len(obs["farms"][0].get("unlocked_quadrants", []) or []) - 1
            if land_target > unlocked and len(action["market"]) < 10:
                action["market"].append(["BUY_LAND"])

            positions = [tuple(obs["farms"][0]["farmer"])] + [
                tuple(pos) for pos in obs["farms"][0].get("hands", [])
            ]
            invs = list((obs.get("private", {}) or {}).get("inventories", []) or [])
            shed = Counter((obs.get("private", {}) or {}).get("shed", {}) or {})
            for order in action["market"]:
                if order[0] == "SELL":
                    product, qty = order[1], min(int(order[2]), shed[order[1]])
                    self.sold[day][product] += qty
                    self.revenue[day][product] += qty * int(obs["market"]["prices"][product])
            for pos, act in zip(positions, [action["farmer"]] + action["hands"]):
                op = act[0] if act else "PASS"
                self.turn_counts[day]["moving" if op in MOVING else "acting" if op in ACTING else "idle"] += 1
                if op == "HARVEST":
                    tile = obs["farms"][0]["tiles"][pos[1]][pos[0]]
                    if isinstance(tile, dict):
                        product = tile.get("crop") or self.router.ANIMALS.get(tile.get("animal"), {}).get("product")
                        if product:
                            self.produced[day][product] += int(tile.get("yield_units", 0) or 0)
            self.actions.append(action)
        except Exception:
            self.faults += 1
            n = len(obs.get("farms", [{}])[0].get("hands", []))
            action = {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(n)], "market": []}
            self.actions.append(action)
        self.times.append((time.perf_counter() - started) * 1000)
        return action


def _result(executor: Executor, env, schedule: dict, seed: int) -> dict:
    final = env.steps[-1][0]
    observations = [step[0].observation for step in env.steps if step and step[0].observation]
    by_day = []
    prev_bank = 3000.0
    cumulative_revenue = 0.0
    for day in range(30):
        end = next((o for o in reversed(observations) if int(o.get("day", -1)) == day), None)
        bank = float(end["farms"][0]["money"]) if end else prev_bank
        produced, sold, rotted = executor.produced[day], executor.sold[day], executor.rotted[day]
        revenue = sum(executor.revenue[day].values())
        cumulative_revenue += revenue
        undelivered = Counter()
        if end:
            inv = _inv_totals(end)
            undelivered.update({p: inv[p] for p in PRODUCTS if inv[p]})
        implied = schedule.get("expected_revenue_by_day", [None] * 30)
        implied_day = implied[day] if day < len(implied) else None
        if implied_day is None:
            implied_day = sum(produced[p] * BASE_PRICE[p] for p in PRODUCTS)
        price_slip = max(0.0, sum(sold[p] * BASE_PRICE[p] for p in PRODUCTS) - revenue)
        rotted_dollars = sum(rotted[p] * BASE_PRICE[p] for p in PRODUCTS)
        undelivered_dollars = sum(undelivered[p] * BASE_PRICE[p] for p in PRODUCTS)
        counts = executor.turn_counts[day]
        # Travel/idle dollar attribution is opportunity cost, capped so categories
        # exactly reconcile to the observed implied-revenue shortfall.
        raw_gap = max(0.0, float(implied_day) - revenue)
        fixed = min(raw_gap, price_slip + rotted_dollars + undelivered_dollars)
        residual = raw_gap - fixed
        denom = counts["moving"] + counts["idle"] or 1
        travel = residual * counts["moving"] / denom
        idle = residual - travel
        loss = {"travel": travel, "idle": idle, "undelivered": min(undelivered_dollars, raw_gap - fixed + undelivered_dollars),
                "rotted": rotted_dollars, "price_slippage": price_slip}
        # Normalize any over-count caused by inventory carried across days.
        total_loss = sum(loss.values())
        if total_loss > raw_gap and total_loss:
            scale = raw_gap / total_loss
            loss = {k: v * scale for k, v in loss.items()}
        by_day.append({"day": day, "bank": bank, "bank_change": bank - prev_bank,
                       "implied_revenue": float(implied_day), "realised_revenue": revenue,
                       "produced": dict(produced), "sold": dict(sold), "undelivered": dict(undelivered),
                       "rotted": dict(rotted), "hand_turns": dict(counts), "execution_loss": loss})
        prev_bank = bank
    all_loss = {k: sum(d["execution_loss"].get(k, 0) for d in by_day)
                for k in ("travel", "idle", "undelivered", "rotted", "price_slippage")}
    ordered = sorted(executor.times)
    p99 = ordered[min(len(ordered) - 1, math.ceil(.99 * len(ordered)) - 1)] if ordered else 0
    return {"seed": seed, "engine": "kaggle-environments 1.32.7", "status": final.status,
            "final_bank": float(final.reward), "faults": executor.faults,
            "timing_ms": {"p99": p99, "max": max(executor.times, default=0), "mean": statistics.fmean(executor.times)},
            "execution_loss": all_loss, "days": by_day, "substitutions": executor.substitutions,
            "expectation_basis": "schedule.expected_revenue_by_day when supplied; otherwise realised production at engine base prices"}


def _write_outputs(result: dict, actions: list, schedule_path: Path | None, tag: str):
    out = ROOT / "data" / "results" / "fixed-game" / str(result["seed"]) / "execution" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "ledger.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    padded = list(actions)
    while len(padded) < 720:
        padded.append({"farmer": ["PASS"], "hands": [], "market": []})
    (out / "actions.json").write_text(json.dumps(padded[:720], indent=2) + "\n", encoding="utf-8")
    lines = [f"# Fixed-game execution: {tag}", "", f"Schedule: `{schedule_path}`" if schedule_path else "Schedule: in-memory",
             f"Final bank: **${result['final_bank']:,.0f}**", f"Faults: {result['faults']}",
             f"Policy timing p99: {result['timing_ms']['p99']:.3f} ms", "", "| day | bank | implied | realised | move | act | idle |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for d in result["days"]:
        h = d["hand_turns"]
        lines.append(f"| {d['day']} | ${d['bank']:,.0f} | ${d['implied_revenue']:,.0f} | ${d['realised_revenue']:,.0f} | {h.get('moving',0)} | {h.get('acting',0)} | {h.get('idle',0)} |")
    lines += ["", "## Execution loss", ""] + [f"- {k.replace('_',' ')}: ${v:,.0f}" for k, v in result["execution_loss"].items()]
    if result["substitutions"]:
        lines += ["", "## Tile substitutions", "", "```json", json.dumps(result["substitutions"], indent=2), "```"]
    (out / "ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def execute(schedule: dict, seed: int = 3910, workers: int = 1, *, tag: str = "run",
            schedule_path: Path | None = None, idle_fix: bool = True,
            instance_path: Path | None = None, fertilizer_threshold: int = 10,
            animal_sweep: bool = False, layout_mode: str = "scheduled") -> dict:
    del workers
    instance = load_instance(instance_path) if instance_path else load_instance()
    if int(instance["seed"]) != int(seed):
        raise ValueError(f"pinned instance seed is {instance['seed']}, requested {seed}")
    fertilizer_drain = sum(int(row.get("drain", {}).get("FERTILIZER", 0))
                           for row in instance.get("town", {}).get("daily_drain", []))
    controller = ReferenceExecutor(schedule, idle_fix=idle_fix,
                                   fertilizer_drain=fertilizer_drain,
                                   fertilizer_threshold=fertilizer_threshold,
                                   animal_sweep=animal_sweep, layout_mode=layout_mode)
    env = make_env(instance, debug=True)
    env.run([controller.policy, "pass"])
    result = _result(controller, env, controller.schedule, seed)
    result["router"] = f"bots/exec_b/main.py@{REFERENCE_ROUTER_COMMIT}"
    result["idle_fix"] = idle_fix
    result["fertilizer_threshold"] = fertilizer_threshold
    result["animal_sweep"] = animal_sweep
    result["layout_mode"] = layout_mode
    result["output_dir"] = str(_write_outputs(result, controller.actions, schedule_path, tag))
    return result


def main():
    parser = argparse.ArgumentParser(description="Execute a pinned day-level Kaggriculture schedule through the real engine")
    parser.add_argument("schedule", type=Path, help="schedule JSON in the fixed-game day format")
    parser.add_argument("--seed", type=int, default=3910)
    parser.add_argument("--workers", type=int, default=1, help="compatibility option; one fixed game is run")
    parser.add_argument("--tag", help="output directory tag (default: schedule stem)")
    parser.add_argument("--no-idle-fix", action="store_true", help="measure the verbatim router before pre-positioning")
    parser.add_argument("--instance", type=Path, help="pinned instance JSON (default: seed 3910)")
    parser.add_argument("--fertilizer-threshold", type=int, default=10,
                        help="minimum zero-drain fertilizer quote worth collecting")
    parser.add_argument("--animal-sweep", action="store_true",
                        help="normalize herd strips across the active animal footprint")
    parser.add_argument("--layout-mode", choices=("scheduled", "reference", "swap_melon_wheat"),
                        default="scheduled", help="crop block placement policy")
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    result = execute(schedule, args.seed, args.workers, tag=args.tag or args.schedule.stem,
                     schedule_path=args.schedule, idle_fix=not args.no_idle_fix,
                     instance_path=args.instance,
                     fertilizer_threshold=args.fertilizer_threshold,
                     animal_sweep=args.animal_sweep, layout_mode=args.layout_mode)
    print(json.dumps({k: result[k] for k in ("final_bank", "faults", "timing_ms", "execution_loss", "output_dir")}, indent=2))


if __name__ == "__main__":
    main()
