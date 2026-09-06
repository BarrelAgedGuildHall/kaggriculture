"""Compute the program KPI set from Kaggriculture episode records.

Runs ``N`` seeds with the candidate in both seats (2N games).  Every metric is
derived after the run from ``env.steps``; the agent's own telemetry is never an
input.  The optional ``--capture-self-report`` switch only preserves Brain KPI
files for a validation comparison.

Examples::

    uv run python scripts/kpi.py --agent bots/brain/main.py --n 20 --tag brain
    uv run python scripts/kpi.py --agent starter --opponent bots/v46/main.py --n 4

The action selected from observation t is read from ``steps[t + 1][seat].action``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from _engine import engine_version, import_engine, to_plain  # noqa: E402

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
NON_FERTILIZER = PRODUCTS[:-1]
PRODUCTIVE = {
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
    "FEED", "COLLECT_FERTILIZER", "CARE", "PICKUP", "PLACE", "DROP",
}
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
CHECKPOINTS = (72, 240, 480, 719)
INTERVAL_DAYS = 5
INTERVAL_COUNT = 6


def _resolve_agent(spec: str) -> str:
    if spec in {"random", "starter", "pass"}:
        return spec
    path = Path(spec)
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        raise SystemExit(f"agent does not exist: {path}")
    return str(path.resolve())


def _sha256(spec: str) -> str | None:
    path = Path(spec)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plant_id(tile: dict[str, Any], x: int, y: int) -> tuple[int, int, int, str]:
    return x, y, int(tile.get("planted_day", -1)), str(tile.get("crop", ""))


def _op(action: Any) -> str:
    return action[0] if isinstance(action, list) and action and isinstance(action[0], str) else "PASS"


def _unit_actions(action: dict[str, Any], expected_slots: int) -> list[list[Any]]:
    farmer = action.get("farmer", ["PASS"])
    hands = action.get("hands", [])
    if not isinstance(hands, list):
        hands = []
    supplied = [farmer, *hands]
    return [supplied[i] if i < len(supplied) else ["PASS"] for i in range(expected_slots)]


def _apply_units(kg: Any, farms: list[dict[str, Any]], privates: list[dict[str, Any]],
                 actions: list[dict[str, Any]], day: int, turns_per_day: int,
                 shed_capacity: int) -> list[list[dict[str, Any]]]:
    """Replay the engine's unit phase and return requested/effective outcomes."""
    outcomes: list[list[dict[str, Any]]] = []
    for seat in (0, 1):
        farm, private = farms[seat], privates[seat]
        action = actions[seat] if isinstance(actions[seat], dict) else {}
        expected_slots = 1 + len(farm.get("hands", []) or [])
        units = _unit_actions(action, expected_slots)

        # Atomic PLANT validation uses every supplied request in the engine.
        raw_hands = action.get("hands", []) if isinstance(action.get("hands", []), list) else []
        supplied = [action.get("farmer", ["PASS"]), *raw_hands]
        demand: dict[str, int] = {}
        for unit_action in supplied:
            if isinstance(unit_action, list) and len(unit_action) >= 2 and unit_action[0] == "PLANT":
                demand[str(unit_action[1])] = demand.get(str(unit_action[1]), 0) + 1
        seeds = private.get("seeds", {})
        blocked = {crop for crop, n in demand.items() if n > int(seeds.get(crop, 0) or 0)}

        seat_outcomes = []
        for idx, requested in enumerate(units):
            pos = tuple(kg._farmer_position(farm, idx) or (-1, -1))
            before_farm, before_private = copy.deepcopy(farm), copy.deepcopy(private)
            allowed = requested
            if (isinstance(requested, list) and len(requested) >= 2
                    and requested[0] == "PLANT" and requested[1] in blocked):
                allowed = ["PASS"]
            kg._apply_unit_action(
                farm, private, idx, allowed, len(farm["tiles"]), day, turns_per_day, shed_capacity
            )
            seat_outcomes.append({
                "op": _op(requested),
                "pos": pos,
                "changed": farm != before_farm or private != before_private,
            })
        outcomes.append(seat_outcomes)
    return outcomes


def _market_orders(kg: Any, actions: list[dict[str, Any]], farms: list[dict[str, Any]],
                   privates: list[dict[str, Any]], market: dict[str, Any], board_size: int,
                   max_orders: int, hire_mult: int, shed_capacity: int) -> list[dict[str, Any]]:
    """Replay 1.32.7's market phase exactly and ledger successful sells/wages."""
    ledger = [{
        "wages": 0.0,
        "sold_units": {p: 0 for p in PRODUCTS},
        "realised_dollars": {p: 0.0 for p in PRODUCTS},
    } for _ in (0, 1)]
    queues = []
    for action in actions:
        raw = action.get("market", []) if isinstance(action, dict) else []
        queues.append(list(raw)[:max_orders] if isinstance(raw, list) else [])

    for order_index in range(max((len(q) for q in queues), default=0)):
        states = [kg._parse_order(q[order_index]) if order_index < len(q) else None for q in queues]
        for seat, state in enumerate(states):
            if state is None:
                continue
            if state["type"] == "HIRE":
                before = float(farms[seat]["money"])
                kg._do_hire(farms[seat], privates[seat], board_size, hire_mult)
                ledger[seat]["wages"] += before - float(farms[seat]["money"])
                states[seat] = None
            elif state["type"] == "BUY_LAND":
                kg._do_buy_land(farms[seat], board_size)
                states[seat] = None

        guard = 0
        while True:
            guard += 1
            if guard >= 100_000:
                raise RuntimeError("market replay exceeded engine guard")
            quoted = [None, None]
            for seat, state in enumerate(states):
                if state is None or state["remaining"] <= 0:
                    continue
                op, item = state["type"], state["item"]
                if op == "SELL" and item in PRODUCTS:
                    price = kg.market_price(item, market["inventory"][item], market.get("params"))
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    price = kg.market_price(item, market["inventory"][item] - 1, market.get("params"))
                elif op == "BUY_SEED" and item in kg.CROPS:
                    price = kg.CROPS[item]["seed"]
                elif op == "BUY_ANIMAL" and item in kg.ANIMALS:
                    price = kg.ANIMALS[item]["cost"]
                else:
                    states[seat] = None
                    continue
                quoted[seat] = op, item, price, state
            if all(q is None for q in quoted):
                break

            committed = False
            for seat, quote in enumerate(quoted):
                if quote is None:
                    continue
                op, item, price, state = quote
                ok = kg._commit_unit(op, item, price, farms[seat], privates[seat], market, shed_capacity)
                if ok:
                    state["remaining"] -= 1
                    committed = True
                    if op == "SELL":
                        ledger[seat]["sold_units"][item] += 1
                        ledger[seat]["realised_dollars"][item] += float(price)
                else:
                    states[seat] = None
            if not committed:
                break
        kg._refresh_prices(market)
    return ledger


def _town_drain(kg: Any, step: int, shops: list[str], shop_interval: int,
                center_interval: int) -> dict[str, int]:
    drain = {p: 0 for p in PRODUCTS}
    if step % shop_interval == 0:
        for shop in shops:
            products = kg.SHOPS.get(shop, [])
            multiplier = 2 if len(products) == 1 else 1
            for product in products:
                drain[product] += multiplier
    if step % center_interval == 0:
        for product in NON_FERTILIZER:
            drain[product] += 1
    return drain


def _blank_game(seed: int, seat: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "seat": seat,
        "eligible_plant_days": 0,
        "watered_plant_days": 0,
        "water_actions": 0,
        "animal_days": 0,
        "fed_animal_days": 0,
        "fed_cared_animal_days": 0,
        "plants_lost": 0,
        "plants_lost_at_deadline": 0,
        "hands_per_day": [0] * 30,
        "wages_paid": 0.0,
        "productive_actions": 0,
        "successful_unit_actions": {op: 0 for op in sorted(PRODUCTIVE | MOVES)},
        "unit_slots": 0,
        "no_op_count": 0,
        "idle_unit_slots": 0,
        "moves": 0,
        "sold_units": {p: 0 for p in PRODUCTS},
        "realised_dollars": {p: 0.0 for p in PRODUCTS},
        "sold_units_by_5day_interval": {p: [0] * INTERVAL_COUNT for p in PRODUCTS},
        "town_demand": {p: 0 for p in PRODUCTS},
        "bank": {},
        "brain_compatible_unsold_value": None,
    }


def _analyze(env: Any, candidate_seat: int, seed: int) -> dict[str, Any]:
    import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as kg

    rows = [_blank_game(seed, 0), _blank_game(seed, 1)]
    deadline_eligible = [set(), set()]
    deadline_watered = [set(), set()]
    previous_plants = [set(), set()]
    cfg = env.configuration
    turns_per_day = max(1, int(getattr(cfg, "turnsPerDay", 24)))
    shed_capacity = int(getattr(cfg, "shedCapacity", 100))
    max_orders = max(1, int(getattr(cfg, "maxMarketOrdersPerTurn", 10)))
    hire_mult = int(getattr(cfg, "farmHandCostMult", 1))
    shop_interval = max(1, int(getattr(cfg, "townShopSellInterval", 4)))
    center_interval = max(1, int(getattr(cfg, "townCenterSellInterval", 24)))

    for j in range(1, len(env.steps)):
        pre = [to_plain(env.steps[j - 1][s].observation) for s in (0, 1)]
        post = [to_plain(env.steps[j][s].observation) for s in (0, 1)]
        actions = [to_plain(env.steps[j][s].action or {}) for s in (0, 1)]
        step = j - 1
        day = int(pre[0].get("day", step // turns_per_day))
        interval = min(INTERVAL_COUNT - 1, day // INTERVAL_DAYS)

        farms = copy.deepcopy(pre[0]["farms"])
        privates = [copy.deepcopy(pre[s]["private"]) for s in (0, 1)]
        market = copy.deepcopy(pre[0]["market"])
        outcomes = _apply_units(kg, farms, privates, actions, day, turns_per_day, shed_capacity)

        for seat in (0, 1):
            row = rows[seat]
            # Count after the final unit actions, before the daily refresh clears
            # husbandry flags. The terminal day has no production refresh.
            if int(post[0].get("day", day)) > day:
                animals = [tile for tile_row in farms[seat]["tiles"] for tile in tile_row
                           if isinstance(tile, dict) and tile.get("animal") in kg.ANIMALS]
                row["animal_days"] += len(animals)
                row["fed_animal_days"] += sum(bool(tile.get("fed_today")) for tile in animals)
                row["fed_cared_animal_days"] += sum(
                    bool(tile.get("fed_today")) and bool(tile.get("cared_today")) for tile in animals
                )
            pre_farm, post_farm = pre[seat]["farms"][seat], post[seat]["farms"][seat]
            current_plants = set()
            for y, tile_row in enumerate(pre_farm["tiles"]):
                for x, tile in enumerate(tile_row):
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        current_plants.add((x, y))
            if step % turns_per_day == 0:
                row["eligible_plant_days"] += len(current_plants)
                if step > 0:
                    for x, y in previous_plants[seat] - current_plants:
                        tile = pre_farm["tiles"][y][x]
                        if isinstance(tile, dict) and tile.get("kind") == "WEED":
                            row["plants_lost"] += 1
            previous_plants[seat] = current_plants
            if 0 <= day < 30:
                row["hands_per_day"][day] = max(
                    row["hands_per_day"][day], len(pre_farm.get("hands", [])), len(post_farm.get("hands", []))
                )
            row["unit_slots"] += len(outcomes[seat])
            for outcome in outcomes[seat]:
                op, changed = outcome["op"], outcome["changed"]
                if changed and op in row["successful_unit_actions"]:
                    row["successful_unit_actions"][op] += 1
                if changed and op in PRODUCTIVE:
                    row["productive_actions"] += 1
                if changed and op in MOVES:
                    row["moves"] += 1
                if not changed:
                    row["no_op_count"] += 1
                if op == "PASS":
                    row["idle_unit_slots"] += 1
                if op == "WATER":
                    row["water_actions"] += 1
                if changed and op == "WATER":
                    x, y = outcome["pos"]
                    if 0 <= y < len(pre_farm["tiles"]) and 0 <= x < len(pre_farm["tiles"][y]):
                        tile = pre_farm["tiles"][y][x]
                        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                            deadline_watered[seat].add((day, _plant_id(tile, x, y)))

            # At hour 23, simulated farms are after this action and immediately
            # before the engine's end-of-day refresh: the exact service deadline.
            if int(pre[seat].get("hour", 0)) == turns_per_day - 1:
                for y, tile_row in enumerate(farms[seat]["tiles"]):
                    for x, tile in enumerate(tile_row):
                        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                            pid = _plant_id(tile, x, y)
                            deadline_eligible[seat].add((day, pid))
                            if tile.get("watered_today"):
                                deadline_watered[seat].add((day, pid))
                for y, tile_row in enumerate(pre_farm["tiles"]):
                    for x, tile in enumerate(tile_row):
                        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                            continue
                        after = post_farm["tiles"][y][x]
                        key = (day, _plant_id(tile, x, y))
                        if (isinstance(after, dict) and after.get("kind") == "WEED"
                                and int(tile.get("consecutive_unwatered", 0)) >= 1
                                and key not in deadline_watered[seat]):
                            row["plants_lost_at_deadline"] += 1

        ledgers = _market_orders(
            kg, actions, farms, privates, market, len(farms[0]["tiles"]), max_orders,
            hire_mult, shed_capacity,
        )
        for seat in (0, 1):
            rows[seat]["wages_paid"] += ledgers[seat]["wages"]
            for product in PRODUCTS:
                units = ledgers[seat]["sold_units"][product]
                rows[seat]["sold_units"][product] += units
                rows[seat]["realised_dollars"][product] += ledgers[seat]["realised_dollars"][product]
                rows[seat]["sold_units_by_5day_interval"][product][interval] += units
            drain = _town_drain(
                kg, step, list(pre[seat].get("town", {}).get("unlocked_shops", [])),
                shop_interval, center_interval,
            )
            for product in PRODUCTS:
                rows[seat]["town_demand"][product] += drain[product]
            if step >= 718:
                shed = pre[seat].get("private", {}).get("shed", {})
                inventories = pre[seat].get("private", {}).get("inventories", []) or []
                pending = {
                    order[1]: int(order[2])
                    for order in actions[seat].get("market", [])
                    if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL"
                }
                prices = pre[seat].get("market", {}).get("prices", {})
                rows[seat]["brain_compatible_unsold_value"] = sum(
                    (
                        max(0, int(shed.get(product, 0) or 0) - pending.get(product, 0))
                        + sum(int(inv.get(product, 0) or 0) for inv in inventories)
                    ) * float(prices.get(product, 0) or 0)
                    for product in PRODUCTS
                )

    final = [to_plain(env.steps[-1][s].observation) for s in (0, 1)]
    bases = {p: float(kg.MARKET_PARAMS[p]["base"]) for p in PRODUCTS}
    for seat, row in enumerate(rows):
        obs = final[seat]
        farm, private = obs["farms"][seat], obs["private"]
        final_day = min(29, int(obs.get("day", 29)))
        for y, tile_row in enumerate(farm["tiles"]):
            for x, tile in enumerate(tile_row):
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    key = (final_day, _plant_id(tile, x, y))
                    deadline_eligible[seat].add(key)
                    if tile.get("watered_today"):
                        deadline_watered[seat].add(key)
        row["hands_per_day"][final_day] = max(row["hands_per_day"][final_day], len(farm.get("hands", [])))
        row["brain_compatible_eligible_plant_days"] = row["eligible_plant_days"]
        row["brain_compatible_watered_plant_days"] = row["water_actions"]
        row["brain_compatible_watered_fraction"] = (
            row["water_actions"] / row["eligible_plant_days"] if row["eligible_plant_days"] else 1.0
        )
        row["deadline_eligible_plant_days"] = len(deadline_eligible[seat])
        row["deadline_watered_plant_days"] = sum(
            key in deadline_watered[seat] for key in deadline_eligible[seat]
        )
        row["watered_fraction_at_daily_deadline"] = (
            row["deadline_watered_plant_days"] / row["deadline_eligible_plant_days"]
            if row["deadline_eligible_plant_days"] else 1.0
        )
        row["eligible_plant_days"] = row["deadline_eligible_plant_days"]
        row["watered_plant_days"] = row["deadline_watered_plant_days"]
        row["watered_fraction_of_eligible_plant_days"] = row["watered_fraction_at_daily_deadline"]
        row["hands_day_mean"] = statistics.fmean(row["hands_per_day"])
        row["productive_action_fraction"] = row["productive_actions"] / max(1, row["unit_slots"])
        row["moves_per_productive_action"] = row["moves"] / max(1, row["productive_actions"])
        row["realised_over_base"] = {
            p: (row["realised_dollars"][p] / (row["sold_units"][p] * bases[p])
                if row["sold_units"][p] else None) for p in PRODUCTS
        }
        row["town_demand_share_by_product"] = {
            p: row["sold_units"][p] / row["town_demand"][p] if row["town_demand"][p] else None
            for p in PRODUCTS
        }
        all_sales = sum(row["sold_units"][p] for p in PRODUCTS)
        sold_to_town = sum(row["sold_units"][p] for p in NON_FERTILIZER)
        total_demand = sum(row["town_demand"][p] for p in NON_FERTILIZER)
        row["brain_compatible_town_demand_share"] = all_sales / max(1, total_demand)
        row["town_demand_share"] = sold_to_town / max(1, total_demand)
        prices = obs.get("market", {}).get("prices", {})
        holdings = {p: int(private.get("shed", {}).get(p, 0) or 0) for p in PRODUCTS}
        for inventory in private.get("inventories", []) or []:
            for product in PRODUCTS:
                holdings[product] += int(inventory.get(product, 0) or 0)
        row["unsold_units_at_719"] = holdings
        row["unsold_value_at_719"] = sum(
            holdings[p] * float(prices.get(p, 0) or 0) for p in PRODUCTS
        )
        row["brain_compatible_unsold_value_at_719"] = (
            float(row["brain_compatible_unsold_value"])
            if row["brain_compatible_unsold_value"] is not None
            else row["unsold_value_at_719"]
        )
        row["fertilizer_dollars"] = row["realised_dollars"]["FERTILIZER"]
        row["bank"] = {
            str(k): float(to_plain(env.steps[min(k, len(env.steps) - 1)][seat].observation)["farms"][seat]["money"])
            for k in CHECKPOINTS
        }
        row["status"] = str(env.steps[-1][seat].status)
        row["final_bank"] = float(farm["money"])
        row["gross_sales_per_wage_dollar"] = (
            sum(row["realised_dollars"].values()) / row["wages_paid"] if row["wages_paid"] else None
        )
        row["net_bank_gain_per_wage_dollar"] = (
            (row["final_bank"] - 3000.0) / row["wages_paid"] if row["wages_paid"] else None
        )
    return rows[candidate_seat]


def _one_job(job: tuple[str, str, int, int, str | None]) -> dict[str, Any]:
    agent, opponent, seed, seat, self_report = job
    if self_report:
        os.environ["BRAIN_KPI"] = "1"
        os.environ["BRAIN_KPI_PATH"] = self_report
    else:
        os.environ.pop("BRAIN_KPI", None)
        os.environ.pop("BRAIN_KPI_PATH", None)
    ke = import_engine()
    env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    agents = [agent, opponent] if seat == 0 else [opponent, agent]
    env.run(agents)
    row = _analyze(env, seat, seed)
    if self_report and Path(self_report).is_file():
        row["self_report"] = self_report
    return row


def _aggregate(rows: list[dict[str, Any]], bases: dict[str, float]) -> dict[str, Any]:
    total = lambda key: sum(float(row[key]) for row in rows)
    mean = lambda key: statistics.fmean(float(row[key]) for row in rows)
    games = len(rows)
    sold = {p: sum(row["sold_units"][p] for row in rows) for p in PRODUCTS}
    dollars = {p: sum(row["realised_dollars"][p] for row in rows) for p in PRODUCTS}
    demand = {p: sum(row["town_demand"][p] for row in rows) for p in PRODUCTS}
    intervals = {
        p: [statistics.fmean(row["sold_units_by_5day_interval"][p][i] for row in rows)
            for i in range(INTERVAL_COUNT)] for p in PRODUCTS
    }
    return {
        "games": games,
        "successful_unit_actions": {
            op: statistics.fmean(row["successful_unit_actions"][op] for row in rows)
            for op in sorted(PRODUCTIVE | MOVES)
        },
        "faults": sum(row["status"] != "DONE" for row in rows),
        "watered_fraction_of_eligible_plant_days": total("watered_plant_days") / max(1, total("eligible_plant_days")),
        "brain_compatible_watered_fraction": (
            total("brain_compatible_watered_plant_days")
            / max(1, total("brain_compatible_eligible_plant_days"))
        ),
        "eligible_plant_days": mean("eligible_plant_days"),
        "animal_days": mean("animal_days"),
        "fed_animal_fraction": total("fed_animal_days") / max(1, total("animal_days")),
        "fed_cared_animal_fraction": total("fed_cared_animal_days") / max(1, total("animal_days")),
        "plants_lost": mean("plants_lost"),
        "plants_lost_at_deadline": mean("plants_lost_at_deadline"),
        "hands_per_day": [statistics.fmean(row["hands_per_day"][d] for row in rows) for d in range(30)],
        "hands_day_mean": mean("hands_day_mean"),
        "wages_paid": mean("wages_paid"),
        "gross_sales_per_wage_dollar": statistics.fmean(
            row["gross_sales_per_wage_dollar"] for row in rows if row["gross_sales_per_wage_dollar"] is not None
        ) if any(row["gross_sales_per_wage_dollar"] is not None for row in rows) else None,
        "net_bank_gain_per_wage_dollar": statistics.fmean(
            row["net_bank_gain_per_wage_dollar"] for row in rows if row["net_bank_gain_per_wage_dollar"] is not None
        ) if any(row["net_bank_gain_per_wage_dollar"] is not None for row in rows) else None,
        "productive_action_fraction": total("productive_actions") / max(1, total("unit_slots")),
        "productive_actions": mean("productive_actions"),
        "no_op_count": mean("no_op_count"),
        "idle_unit_slots": mean("idle_unit_slots"),
        "moves_per_productive_action": total("moves") / max(1, total("productive_actions")),
        "realised_over_base": {
            p: dollars[p] / (sold[p] * bases[p]) if sold[p] else None for p in PRODUCTS
        },
        "realised_dollars": {p: dollars[p] / games for p in PRODUCTS},
        "sold_units": {p: sold[p] / games for p in PRODUCTS},
        "sold_units_by_5day_interval": intervals,
        "brain_compatible_town_demand_share": (
            sum(sold.values()) / max(1, sum(demand[p] for p in NON_FERTILIZER))
        ),
        "town_demand_share": (
            sum(sold[p] for p in NON_FERTILIZER) / max(1, sum(demand[p] for p in NON_FERTILIZER))
        ),
        "town_demand_share_by_product": {
            p: sold[p] / demand[p] if demand[p] else None for p in PRODUCTS
        },
        "town_demand": {p: demand[p] / games for p in PRODUCTS},
        "unsold_value_at_719": mean("unsold_value_at_719"),
        "brain_compatible_unsold_value_at_719": mean("brain_compatible_unsold_value_at_719"),
        "fertilizer_dollars": mean("fertilizer_dollars"),
        "bank": {str(k): statistics.fmean(row["bank"][str(k)] for row in rows) for k in CHECKPOINTS},
        "final_bank": mean("final_bank"),
    }


def _self_report_validation(rows: list[dict[str, Any]], aggregate: dict[str, Any],
                            bases: dict[str, float]) -> dict[str, Any] | None:
    paths = [Path(row["self_report"]) for row in rows if row.get("self_report")]
    if len(paths) != len(rows) or not all(path.is_file() for path in paths):
        return None
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    total_self_units = {p: sum(float(report["sold_units"][p]) for report in reports) for p in PRODUCTS}
    total_self_dollars = {p: sum(float(report["realised_dollars"][p]) for report in reports) for p in PRODUCTS}
    total_self_demand = sum(
        float(report["town_demand"][p]) for report in reports for p in PRODUCTS
    )
    self_values: dict[str, float] = {
        "watered_fraction_of_eligible_plant_days": (
            sum(float(r["water_actions"]) for r in reports)
            / max(1.0, sum(float(r["eligible_plant_days"]) for r in reports))
        ),
        "plants_lost": statistics.fmean(float(r["plants_lost"]) for r in reports),
        "productive_action_fraction": (
            sum(float(r["productive_actions"]) for r in reports)
            / max(1.0, sum(float(r["unit_slots"]) for r in reports))
        ),
        "no_op_count": statistics.fmean(float(r["no_op_count"]) for r in reports),
        "idle_unit_slots": statistics.fmean(float(r["idle_unit_slots"]) for r in reports),
        "wages_paid": statistics.fmean(float(r["wages_paid"]) for r in reports),
        "town_demand_share": sum(total_self_units.values()) / max(1.0, total_self_demand),
        "unsold_value_at_719": statistics.fmean(float(r["unsold_value_at_end"]) for r in reports),
        "bank_72": statistics.fmean(float(r["bank_turn_72"]) for r in reports),
        "bank_719": statistics.fmean(float(r["final_bank_before_last_orders"]) for r in reports),
    }
    engine_values = {
        **{key: float(aggregate[key]) for key in (
            "plants_lost", "productive_action_fraction", "no_op_count", "idle_unit_slots", "wages_paid",
        )},
        "watered_fraction_of_eligible_plant_days": float(aggregate["brain_compatible_watered_fraction"]),
        "town_demand_share": float(aggregate["brain_compatible_town_demand_share"]),
        "unsold_value_at_719": float(aggregate["brain_compatible_unsold_value_at_719"]),
        "bank_72": float(aggregate["bank"]["72"]),
        "bank_719": float(aggregate["bank"]["719"]),
    }
    for product in PRODUCTS:
        if total_self_units[product]:
            key = f"realised_over_base.{product}"
            self_values[key] = total_self_dollars[product] / (total_self_units[product] * bases[product])
            engine_values[key] = float(aggregate["realised_over_base"][product])
            units_key = f"sold_units.{product}"
            self_values[units_key] = total_self_units[product] / len(reports)
            engine_values[units_key] = float(aggregate["sold_units"][product])

    comparisons = {}
    relative_deltas = []
    for key in self_values:
        actual, reference = engine_values[key], self_values[key]
        absolute = actual - reference
        relative = abs(absolute) / abs(reference) if reference else (0.0 if actual == 0 else None)
        comparisons[key] = {
            "engine_record": actual,
            "self_report": reference,
            "absolute_delta": absolute,
            "relative_delta_pct": None if relative is None else 100.0 * relative,
        }
        if relative is not None:
            relative_deltas.append(relative)
    maximum = max(relative_deltas, default=0.0)
    return {
        "compared_games": len(reports),
        "max_relative_delta_pct": 100.0 * maximum,
        "within_2pct": maximum <= 0.02,
        "metrics": comparisons,
    }


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{100 * float(value):.2f}%"
    return f"{float(value):,.2f}"


def _print_table(agg: dict[str, Any]) -> None:
    print("\nKPI                                      aggregate")
    print("-" * 58)
    lines = [
        ("games / faults", f"{agg['games']} / {agg['faults']}"),
        ("watered eligible plant-days", _fmt(agg["watered_fraction_of_eligible_plant_days"], True)),
        ("plants lost / game", _fmt(agg["plants_lost"])),
        ("hands / day", _fmt(agg["hands_day_mean"])),
        ("wages paid / game", _fmt(agg["wages_paid"])),
        ("productive-action fraction", _fmt(agg["productive_action_fraction"], True)),
        ("no-op count / game", _fmt(agg["no_op_count"])),
        ("idle unit-slots / game", _fmt(agg["idle_unit_slots"])),
        ("moves / productive action", _fmt(agg["moves_per_productive_action"])),
        ("realised/base by product", " ".join(f"{p[:3]}={_fmt(agg['realised_over_base'][p])}" for p in PRODUCTS)),
        ("town-demand share", _fmt(agg["town_demand_share"], True)),
        ("unsold value at 719", _fmt(agg["unsold_value_at_719"])),
        ("fertilizer dollars / game", _fmt(agg["fertilizer_dollars"])),
        ("bank 72/240/480/719", "/".join(_fmt(agg["bank"][str(k)]) for k in CHECKPOINTS)),
    ]
    for label, value in lines:
        print(f"{label:<40} {value}")
    print("units sold / game by 5-day interval (0-4 .. 25-29)")
    for product in PRODUCTS:
        values = " ".join(f"{v:.1f}" for v in agg["sold_units_by_5day_interval"][product])
        print(f"  {product:<12} {values}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="candidate .py path or built-in agent")
    parser.add_argument("--opponent", default="bots/v46/main.py", help="opponent .py path or built-in agent")
    parser.add_argument("--n", type=int, default=20, help="paired seeds; total games are 2N")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--tag", help="output directory name under data/results/kpi/")
    parser.add_argument("--capture-self-report", action="store_true", help="validation only: preserve Brain KPI files")
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")

    agent, opponent = _resolve_agent(args.agent), _resolve_agent(args.opponent)
    tag = args.tag or (Path(agent).parent.name if Path(agent).is_file() else str(agent))
    out_dir = REPO / "data" / "results" / "kpi" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    self_dir = out_dir / "self-reports"
    if args.capture_self_report:
        self_dir.mkdir(parents=True, exist_ok=True)

    before_hashes = {"agent": _sha256(agent), "opponent": _sha256(opponent)}
    jobs = []
    for seed in range(args.seed_base, args.seed_base + args.n):
        for seat in (0, 1):
            self_path = str(self_dir / f"seed-{seed}-seat-{seat}.json") if args.capture_self_report else None
            jobs.append((agent, opponent, seed, seat, self_path))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["seed"], row["seat"]))

    after_hashes = {"agent": _sha256(agent), "opponent": _sha256(opponent)}
    if before_hashes != after_hashes:
        raise RuntimeError(f"agent source changed during run: before={before_hashes}, after={after_hashes}")

    import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as kg
    bases = {p: float(kg.MARKET_PARAMS[p]["base"]) for p in PRODUCTS}
    aggregate = _aggregate(rows, bases)
    validation = _self_report_validation(rows, aggregate, bases) if args.capture_self_report else None
    payload = {
        "schema_version": 1,
        "engine": engine_version(),
        "agent": agent,
        "opponent": opponent,
        "source_sha256": before_hashes,
        "paired_seeds": list(range(args.seed_base, args.seed_base + args.n)),
        "replay_convention": "action from state t is steps[t+1][seat].action",
        "definitions": {
            "watered": "distinct standing plants with watered_today=true at the daily service deadline",
            "brain_compatible_watered": "recorded WATER actions / plants present at starts of days",
            "plants_lost": "plant coordinate present before refresh and observed as WEED at next day start",
            "productive": "state-changing unit action in the documented productive set",
            "no_op": "unit slot whose requested action caused no farm/private state change",
            "idle_unit_slot": "unit slot requesting PASS (including omitted hand actions)",
            "town_demand_share": "successful non-fertilizer units sold / scheduled non-fertilizer town drain",
            "brain_compatible_town_demand_share": "all successful units sold / scheduled town drain",
            "unsold_value": "actual shed plus carried product units in the turn-719 observation times its quote",
            "brain_compatible_unsold_value": "turn-718 holdings less pending final SELL orders, at the turn-718 quote",
        },
        "base_prices": bases,
        "games": rows,
        "aggregate": aggregate,
    }
    if validation is not None:
        payload["self_report_validation"] = validation
    output = out_dir / "kpi.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_table(aggregate)
    if validation is not None:
        verdict = "PASS" if validation["within_2pct"] else "FAIL"
        print(f"self-report validation: {verdict}; max relative delta "
              f"{validation['max_relative_delta_pct']:.3f}%")
    print(f"\nwrote {output}")
    return 1 if aggregate["faults"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
