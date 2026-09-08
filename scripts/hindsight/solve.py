"""HINDSIGHT SOLVER v3: many-draw contested optimisation of Strawberry-Brain.

Usage (from the repository root)::

    uv run python scripts/hindsight/solve.py --workers 8 --hours 2
    uv run python scripts/hindsight/solve.py --workers 8 --max-evals 16

The solver loads ``bots/brain_strawberry/main.py`` without copying it and registers
parameterised season/labour/market modules. Seat-symmetric panels use one alternating
seat per seed. One-seed racing and day-18/day-23 pruning make poor plans cheap. Search
is a resumable discrete elitist ES with adaptive mutation and group crossover.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
BOT = REPO / "bots" / "brain_strawberry" / "main.py"
OPPONENT = REPO / "bots" / "v46" / "main.py"
SOLO_RESULT_ROOT = REPO / "data" / "results" / "hindsight"
RESULT_ROOT = REPO / "data" / "results" / "hindsight-v3"
CONTESTED_V1 = REPO / "data" / "results" / "hindsight-contested" / "best.json"
CONTEST_SEEDS = (5005, 5033, 5043, 5072, 5026, 5038, 5024, 5064, 5063, 5090)
PRUNE_STEPS = (24 * 19, 24 * 24)
PRUNED_SCORE = -1_000_000_000.0
sys.path.insert(0, str(REPO / "scripts"))

from _engine import engine_version, import_engine, to_plain  # noqa: E402

CROPS = ("WHEAT", "CARROT", "MELON", "STRAWBERRY")
ALL_CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
PHASES = ("opening", "days_10_16", "days_17_22", "days_23_26")
LAND_COST = (1000, 2000, 4000)


def _plain_mix(values: list[int], total: int) -> dict[str, int]:
    values = [max(0, int(x)) for x in values]
    if not any(values):
        values[0] = total
    raw = [total * x / sum(values) for x in values]
    out = [int(x) for x in raw]
    for i in sorted(range(len(out)), key=lambda j: raw[j] - out[j], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return dict(zip(CROPS, out))


def _base_plan() -> dict[str, Any]:
    """Exact raw-bot anchor; mutations leave the anchor and enable plan knobs."""
    return {
        "raw_bot": True,
        "crop_mix": {
            "opening": {"WHEAT": 20, "CARROT": 0, "MELON": 5, "STRAWBERRY": 0},
            "days_10_16": {"WHEAT": 55, "CARROT": 0, "MELON": 20, "STRAWBERRY": 25},
            "days_17_22": {"WHEAT": 55, "CARROT": 0, "MELON": 20, "STRAWBERRY": 25},
            "days_23_26": {"WHEAT": 75, "CARROT": 25, "MELON": 0, "STRAWBERRY": 0},
        },
        "land_days": {"NE": 11, "SW": 11, "SE": 30},
        "animals": {
            "GOOSE": {"start_day": 30, "interval": 1, "batch": 0, "total": 0},
            "COW": {"start_day": 0, "interval": 1, "batch": 1, "total": 7},
            "SHEEP": {"start_day": 0, "interval": 1, "batch": 1, "total": 11},
        },
        "cash_reserve": 240,
        "hands_by_5day": [5, 8, 10, 12, 12, 12],
        "fertilize": {c: True for c in ALL_CROPS},
        "sell_policy": {p: ("metered" if p == "STRAWBERRY" else "immediate") for p in PRODUCTS},
        "last_plant_day": {"WHEAT": 27, "CARROT": 27, "TOMATO": 21, "STRAWBERRY": 17, "MELON": 19},
    }


def _normalise(plan: dict[str, Any]) -> dict[str, Any]:
    p = copy.deepcopy(plan)
    p["raw_bot"] = bool(p.get("raw_bot", False))
    p["crop_mix"]["opening"] = _plain_mix([p["crop_mix"]["opening"].get(c, 0) for c in CROPS], 25)
    for phase in PHASES[1:]:
        p["crop_mix"][phase] = _plain_mix([p["crop_mix"][phase].get(c, 0) for c in CROPS], 100)
    days = [max(0, min(30, int(p["land_days"][q]))) for q in ("NE", "SW", "SE")]
    for i in range(1, 3):
        days[i] = max(days[i], days[i - 1])
    p["land_days"] = dict(zip(("NE", "SW", "SE"), days))
    for animal in ANIMALS:
        a = p["animals"][animal]
        a["start_day"] = max(0, min(30, int(a["start_day"])))
        a["interval"] = max(1, min(8, int(a["interval"])))
        a["batch"] = max(0, min(4, int(a["batch"])))
        a["total"] = max(0, min(20, int(a["total"])))
        if not a["batch"] or not a["total"]:
            a.update(start_day=30, batch=0, total=0)
    p["cash_reserve"] = max(0, min(5000, int(p["cash_reserve"])))
    p["hands_by_5day"] = [max(4, min(16, int(x))) for x in p["hands_by_5day"]][:6]
    while len(p["hands_by_5day"]) < 6:
        p["hands_by_5day"].append(p["hands_by_5day"][-1])
    p["fertilize"] = {c: bool(p["fertilize"].get(c, False)) for c in ALL_CROPS}
    p["sell_policy"] = {x: ("metered" if p["sell_policy"].get(x) == "metered" else "immediate") for x in PRODUCTS}
    p["last_plant_day"] = {c: max(0, min(29, int(p["last_plant_day"].get(c, 29)))) for c in ALL_CROPS}
    return p


def _plan_key(plan: dict[str, Any]) -> str:
    raw = json.dumps(_normalise(plan), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _animal_target(plan: dict[str, Any], animal: str, day: int) -> int:
    a = plan["animals"][animal]
    if a["batch"] <= 0 or day < a["start_day"]:
        return 0
    purchase_days = 1 + (day - a["start_day"]) // a["interval"]
    return min(a["total"], purchase_days * a["batch"])


def _load_parameterised_agent(plan: dict[str, Any]):
    plan = _normalise(plan)
    name = "_hindsight_brain_" + _plan_key(plan) + "_" + str(os.getpid()) + "_" + str(time.time_ns())
    spec = importlib.util.spec_from_file_location(name, BOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BOT}")
    brain = importlib.util.module_from_spec(spec)
    sys.modules[name] = brain
    spec.loader.exec_module(brain)

    if plan["raw_bot"]:
        return brain.agent

    # Remove the coordinator's hard-coded two-land ceiling; our market module issues
    # the requested sequential NE/SW/SE purchases when their earliest day and reserve permit.
    brain.LAND_FIRST_DAY = 99
    brain.ANIMAL_TARGETS = {a: plan["animals"][a]["total"] for a in ANIMALS}
    brain.config["animal_targets"] = brain.ANIMAL_TARGETS
    brain.config["max_animals"] = sum(brain.ANIMAL_TARGETS.values())
    brain.SEED_CAP = {c: 100 for c in ALL_CROPS}
    brain.MONEY_RESERVE = plan["cash_reserve"]

    def phase_mix(day: int) -> dict[str, int]:
        phase = "opening" if day <= 9 else "days_10_16" if day <= 16 else "days_17_22" if day <= 22 else "days_23_26"
        mix = dict(plan["crop_mix"][phase])
        return {c: w for c, w in mix.items() if day <= plan["last_plant_day"].get(c, -1) and w > 0}

    class HindsightSeason:
        @staticmethod
        def plan_day(view):
            targets = {a: _animal_target(plan, a, view.day) for a in ANIMALS}
            return brain.DayPlan(
                PHASES[0] if view.day <= 9 else PHASES[1] if view.day <= 16 else PHASES[2] if view.day <= 22 else PHASES[3],
                plan["hands_by_5day"][min(5, view.day // 5)],
                4,
                tuple(sorted(phase_mix(view.day).items())),
                tuple(sorted(targets.items())),
                tuple(sorted(brain.RESERVATION.items())),
                view.day >= brain.LIQUIDATE_DAY,
                plan["cash_reserve"],
            )

    class HindsightLabor:
        @staticmethod
        def hire_count(view, day_plan):
            if view.hour > 2:
                return 0
            target = day_plan.target_hands
            count = max(0, target - len(view.hands))
            money = float(view.money)
            accepted = 0
            for hire_index in range(int(view.hires_today), int(view.hires_today) + count):
                wage = brain._fib(hire_index)
                if money - wage < day_plan.reserve_cash:
                    break
                money -= wage
                accepted += 1
            return accepted

        @staticmethod
        def dispatch(view, day_plan):
            jobs = []
            build_positions = []
            for job in view.jobs:
                action = tuple(job[2])
                if action and action[0] == "FERTILIZE":
                    crop_at = {(x, y): brain._labor_dict(tile).get("crop") for x, y, tile in view.plants}
                    if not plan["fertilize"].get(crop_at.get(tuple(job[1])), False):
                        continue
                if action and action[0] in ("BUILD_COOP", "BUILD_PASTURE"):
                    build_positions.append(tuple(job[1]))
                    continue
                if action and action[0] == "PLACE":
                    continue
                jobs.append(job)

            live = {a: 0 for a in ANIMALS}
            empty = {"COOP": [], "PASTURE": []}
            for _x, _y, tile in view.animals:
                animal = brain._labor_dict(tile).get("animal")
                if animal in live:
                    live[animal] += 1
            for x, y, tile in view.empty_structures:
                kind = brain._labor_dict(tile).get("kind")
                if kind in empty:
                    empty[kind].append((x, y))
            desired = dict(day_plan.animal_targets)
            need_coop = max(0, desired.get("GOOSE", 0) - live["GOOSE"] - len(empty["COOP"]))
            need_pasture = max(0, desired.get("COW", 0) + desired.get("SHEEP", 0) - live["COW"] - live["SHEEP"] - len(empty["PASTURE"]))
            kinds = ["COOP"] * need_coop + ["PASTURE"] * need_pasture
            for pos, kind in zip(build_positions, kinds):
                jobs.append((1, pos, ("BUILD_" + kind,), None, None))

            inventories = [dict(brain._labor_dict(x) or ()) for x in view.unit_inventories]
            carried = {a: sum(int(inv.get(a, 0) or 0) for inv in inventories) for a in ANIMALS}
            for animal, structure in (("GOOSE", "COOP"), ("COW", "PASTURE"), ("SHEEP", "PASTURE")):
                n = min(carried[animal], len(empty[structure]))
                for pos in empty[structure][:n]:
                    jobs.append((1, pos, ("PLACE", animal), animal, None))
                empty[structure] = empty[structure][n:]
            return brain.LaborB1.dispatch(view.__class__(**{**view.__dict__, "jobs": tuple(jobs)}), day_plan)

    class HindsightMarket:
        @staticmethod
        def orders(view, day_plan):
            extra = len(view.unlocked_quadrants) - 1
            land_days = [plan["land_days"][q] for q in ("NE", "SW", "SE")]
            if 0 <= extra < 3 and view.day >= land_days[extra] and land_days[extra] < 30:
                if view.money - LAND_COST[extra] >= day_plan.reserve_cash:
                    return [["BUY_LAND"]]
            return []

    def hindsight_sell(shed, mkt_inv, day, wheat_reserve, slots, step=None, town_shops=()):
        out = []
        for item in brain.SELL_ORDER:
            if len(out) >= slots:
                break
            qty = int(shed.get(item, 0)) - (wheat_reserve if item == "WHEAT" else 0)
            if qty <= 0:
                continue
            if day < brain.LIQUIDATE_DAY and plan["sell_policy"][item] == "metered":
                if step is None or step % 4 != 1:
                    continue
                qty = min(qty, brain.town_drain(step - 1, town_shops).get(item, 0))
            if qty > 0:
                out.append(["SELL", item, qty])
        return out

    brain.register_module("season", "hindsight", HindsightSeason)
    brain.register_module("labor", "hindsight", HindsightLabor)
    brain.register_module("market", "hindsight", HindsightMarket)
    brain.config["modules"].update(season="hindsight", labor="hindsight", market="hindsight")
    brain._sell_orders = hindsight_sell

    def candidate(obs, config=None):
        action = brain.agent(obs, config)
        # If every crop disables fertiliser, do not buy it merely because the base
        # coordinator computed a reserve before the labour adapter removed its jobs.
        if not any(plan["fertilize"].values()):
            action["market"] = [o for o in action.get("market", []) if not (len(o) >= 2 and o[0] == "BUY_PRODUCT" and o[1] == "FERTILIZER")]
        return action

    return candidate


def _bank_curve(env: Any, seat: int = 0) -> list[float]:
    curve = []
    for day in range(30):
        index = min((day + 1) * 24, len(env.steps) - 1)
        obs = to_plain(env.steps[index][seat].observation)
        curve.append(float(obs["farms"][seat]["money"]))
    return curve


def _money_now(env: Any, seat: int) -> float:
    obs = to_plain(env.state[seat].observation)
    return float(obs["farms"][seat]["money"])


def _run_episode(
    seed: int,
    plan: dict[str, Any] | None,
    seat: int,
    opponent: Path = OPPONENT,
    incumbent_curve: list[float] | None = None,
    prune_slack: float = 12_000,
) -> dict[str, Any]:
    """Run one contested game, stopping the engine loop when a prune gate fails."""
    ke = import_engine()
    ours: Any = str(BOT) if plan is None else _load_parameterised_agent(plan)
    agents = [ours, str(opponent)] if seat == 0 else [str(opponent), ours]
    env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.reset(len(agents))
    # Environment.run is this exact loop. Keeping it here lets us stop both agents,
    # rather than merely crashing the candidate while the opponent finishes day 29.
    runner = env._Environment__agent_runner(agents)  # noqa: SLF001 - pinned engine API
    pruned = False
    prune_step = None
    started = time.perf_counter()
    while not env.done:
        actions, logs = runner.act()
        env.step(actions, logs)
        step = int(to_plain(env.state[0].observation).get("step", len(env.steps) - 1))
        if incumbent_curve is not None and step in PRUNE_STEPS:
            day = step // 24 - 1
            if _money_now(env, seat) < float(incumbent_curve[day]) - prune_slack:
                pruned, prune_step = True, step
                break
    our_curve = _bank_curve(env, seat)
    their_curve = _bank_curve(env, 1 - seat)
    our_bank, their_bank = our_curve[-1], their_curve[-1]
    return {
        "seed": seed,
        "seat": seat,
        "status": str(env.steps[-1][seat].status),
        "opponent_status": str(env.steps[-1][1 - seat].status),
        "our_bank": our_bank,
        "their_bank": their_bank,
        "margin": PRUNED_SCORE if pruned else our_bank - their_bank,
        "our_bank_by_day": our_curve,
        "their_bank_by_day": their_curve,
        "pruned": pruned,
        "prune_step": prune_step,
        "steps_run": len(env.steps) - 1,
        "game_seconds": time.perf_counter() - started,
    }


def _panel_result(games: list[dict[str, Any]]) -> dict[str, Any]:
    pruned = any(g["pruned"] for g in games)
    return {
        "games": games,
        "mean_margin": PRUNED_SCORE if pruned else statistics.fmean(g["margin"] for g in games),
        "worst_margin": PRUNED_SCORE if pruned else min(g["margin"] for g in games),
        "mean_our_bank": statistics.fmean(g["our_bank"] for g in games),
        "our_banks": [g["our_bank"] for g in games],
        "their_banks": [g["their_bank"] for g in games],
        "mean_our_bank_by_day": [statistics.fmean(g["our_bank_by_day"][day] for g in games) for day in range(30)],
        "mean_their_bank_by_day": [statistics.fmean(g["their_bank_by_day"][day] for g in games) for day in range(30)],
        "pruned": pruned,
        "games_run": len(games),
        "steps_run": sum(g["steps_run"] for g in games),
    }


def _run_panel(
    plan: dict[str, Any] | None,
    seeds: tuple[int, ...] = CONTEST_SEEDS,
    opponent: Path = OPPONENT,
    incumbent_games: dict[int, dict[str, Any]] | None = None,
    prune_slack: float = 12_000,
) -> dict[str, Any]:
    games = []
    for seed in seeds:
        incumbent_curve = incumbent_games.get(seed, {}).get("our_bank_by_day") if incumbent_games else None
        game = _run_episode(seed, plan, seed % 2, opponent, incumbent_curve, prune_slack)
        games.append(game)
        if game["pruned"]:
            break
    return _panel_result(games)


def _worker(
    plan: dict[str, Any], seeds: tuple[int, ...], opponent: Path,
    incumbent_games: dict[int, dict[str, Any]] | None, prune_slack: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = _run_panel(plan, seeds, opponent, incumbent_games, prune_slack)
        result["error"] = None
    except Exception as exc:
        result = {"games": [], "mean_margin": -1e12, "our_banks": [], "their_banks": [], "mean_our_bank_by_day": [], "error": repr(exc)}
    result["eval_seconds"] = time.perf_counter() - started
    result["params"] = _normalise(plan)
    result["plan_id"] = _plan_key(plan)
    return result


def _transfer_mix(rng: random.Random, mix: dict[str, int], magnitude: int) -> None:
    donor, receiver = rng.sample(CROPS, 2)
    amount = min(mix[donor], rng.randint(1, max(1, magnitude)))
    mix[donor] -= amount
    mix[receiver] += amount


def _mutate(parent: dict[str, Any], rng: random.Random, heat: float = 1.0) -> dict[str, Any]:
    p = copy.deepcopy(parent)
    p["raw_bot"] = False
    # Independent per-group rates let heat act as a genuine mutation-rate scalar.
    # The generation-level 1/5 rule raises/lower this scalar from observed success.
    rates = {"mix": .42, "land": .10, "animal": .35, "reserve": .16,
             "hands": .32, "fert": .12, "sell": .18, "last": .16}
    groups = [g for g, rate in rates.items() if rng.random() < min(.95, rate * heat)]
    if not groups:
        groups = [rng.choice(tuple(rates))]
    for group in groups:
        if group == "mix":
            phase = rng.choice(PHASES)
            _transfer_mix(rng, p["crop_mix"][phase], 4 if phase == "opening" else int(15 * heat))
        elif group == "land":
            q = rng.choice(("NE", "SW", "SE"))
            p["land_days"][q] = rng.choice([30, max(0, min(29, p["land_days"][q] + rng.randint(-5, 5)))])
        elif group == "animal":
            a = p["animals"][rng.choice(ANIMALS)]
            field, lo, hi = rng.choice((("start_day", 0, 30), ("interval", 1, 8), ("batch", 0, 4), ("total", 0, 20)))
            a[field] = rng.randint(lo, hi) if rng.random() < 0.25 else max(lo, min(hi, a[field] + rng.choice((-2, -1, 1, 2))))
        elif group == "reserve":
            p["cash_reserve"] += rng.randint(-500, 500)
        elif group == "hands":
            i = rng.randrange(6)
            p["hands_by_5day"][i] += rng.choice((-2, -1, 1, 2))
        elif group == "fert":
            c = rng.choice(ALL_CROPS)
            p["fertilize"][c] = not p["fertilize"][c]
        elif group == "sell":
            x = rng.choice(PRODUCTS)
            p["sell_policy"][x] = "metered" if p["sell_policy"][x] == "immediate" else "immediate"
        else:
            c = rng.choice(ALL_CROPS)
            p["last_plant_day"][c] += rng.randint(-4, 4)
    return _normalise(p)


def _crossover(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Uniform crossover at the semantic top-level-group boundary."""
    child = copy.deepcopy(a)
    child["raw_bot"] = False
    for group in ("crop_mix", "land_days", "animals", "cash_reserve", "hands_by_5day",
                  "fertilize", "sell_policy", "last_plant_day"):
        if rng.random() < .5:
            child[group] = copy.deepcopy(b[group])
    return _normalise(child)


def _load_plan(path: Path) -> dict[str, Any]:
    item = json.loads(path.read_text(encoding="utf-8"))
    return _normalise(item.get("plan", item))


def _seed_plans(init_from: list[Path]) -> list[dict[str, Any]]:
    base = _base_plan()
    plans = [base]
    for path in init_from:
        if path.exists():
            plans.append(_load_plan(path))
    for crop in CROPS:
        p = copy.deepcopy(base)
        p["raw_bot"] = False
        p["crop_mix"]["opening"] = {c: (25 if c == crop else 0) for c in CROPS}
        plans.append(p)
    for hands in (8, 10, 12, 14, 16):
        p = copy.deepcopy(base)
        p["raw_bot"] = False
        p["hands_by_5day"] = [hands] * 6
        plans.append(p)
    p = copy.deepcopy(base)
    p["raw_bot"] = False
    for a in ANIMALS:
        p["animals"][a].update(start_day=30, batch=0, total=0)
    plans.append(p)
    p = copy.deepcopy(base)
    p["raw_bot"] = False
    p["animals"]["GOOSE"].update(start_day=1, interval=1, batch=2, total=16)
    p["animals"]["COW"].update(start_day=30, batch=0, total=0)
    p["animals"]["SHEEP"].update(start_day=30, batch=0, total=0)
    plans.append(p)
    p = copy.deepcopy(base)
    p["raw_bot"] = False
    p["land_days"] = {"NE": 10, "SW": 11, "SE": 12}
    plans.append(p)
    p = copy.deepcopy(base)
    p["raw_bot"] = False
    p["sell_policy"] = {x: "immediate" for x in PRODUCTS}
    plans.append(p)
    return [_normalise(x) for x in plans]


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def flatten(plan: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {"cash_reserve": plan["cash_reserve"]}
        for phase, mix in plan["crop_mix"].items():
            for crop, value in mix.items(): out[f"mix.{phase}.{crop}"] = value
        for q, value in plan["land_days"].items(): out[f"land.{q}"] = value
        for a, cfg in plan["animals"].items():
            for k, value in cfg.items(): out[f"animal.{a}.{k}"] = value
        for i, value in enumerate(plan["hands_by_5day"]): out[f"hands.d{i*5}_{i*5+4}"] = value
        for c, value in plan["fertilize"].items(): out[f"fertilize.{c}"] = float(value)
        for p, value in plan["sell_policy"].items(): out[f"sell.{p}"] = float(value == "metered")
        for c, value in plan["last_plant_day"].items(): out[f"last.{c}"] = value
        return out
    good = [r for r in rows if r.get("mean_margin", -1e12) > -1e11]
    if len(good) < 4: return []
    banks = [float(r["mean_margin"]) for r in good]
    mean_y = statistics.fmean(banks)
    sd_y = statistics.pstdev(banks) or 1.0
    features = [flatten(r["params"]) for r in good]
    result = []
    for key in features[0]:
        xs = [f[key] for f in features]
        sd_x = statistics.pstdev(xs)
        if not sd_x: continue
        mean_x = statistics.fmean(xs)
        corr = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,banks)) / (len(xs)*sd_x*sd_y)
        result.append({"parameter": key, "correlation": corr, "absolute_correlation": abs(corr)})
    return sorted(result, key=lambda x: x["absolute_correlation"], reverse=True)[:12]


def _plateau(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    banks = [float(r["mean_margin"]) for r in rows if r.get("mean_margin", -1e12) > -1e11]
    best = max(banks)
    result: dict[str, float | int] = {"best": best, "evaluations": len(banks)}
    for window in (100, 400):
        prior = max(banks[:-window]) if len(banks) > window else banks[0]
        result[f"gain_last_{window}"] = best - prior
    result["first_best_evaluation"] = 1 + banks.index(best)
    return result


def _derived_plan(plan: dict[str, Any]) -> dict[str, Any]:
    targets = [{a: _animal_target(plan, a, day) for a in ANIMALS} for day in range(30)]
    purchases = []
    previous = {a: 0 for a in ANIMALS}
    for target in targets:
        purchases.append({a: target[a] - previous[a] for a in ANIMALS})
        previous = target
    return {
        "target_hands_by_day": [plan["hands_by_5day"][min(5, day // 5)] for day in range(30)],
        "animal_target_by_day": targets,
        "animal_planned_purchases_by_day": purchases,
        "land_days_note": "earliest requested day; liquidity/reserve can delay execution",
    }


def solve_contested(
    workers: int, population: int, hours: float, max_evals: int | None,
    search_seed: int, seeds: tuple[int, ...], opponent: Path, prune_slack: float,
    race_slack: float, init_from: list[Path],
) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = RESULT_ROOT / "log.jsonl"
    rows = _read_log(log_path)
    seen = {r.get("plan_id") for r in rows}
    baseline = _run_panel(None, seeds, opponent)
    started = time.perf_counter()
    deadline = started + hours * 3600
    rng = random.Random(search_seed)
    candidates = [p for p in _seed_plans(init_from) if _plan_key(p) not in seen]
    generation = max([int(r.get("generation", -1)) for r in rows] + [-1]) + 1
    total_before = len(rows)

    heat = float(rows[-1].get("heat", 1.0)) if rows else 1.0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Establish a complete incumbent before racing or pruning anything.
        if not any(r.get("outcome") == "completed" for r in rows):
            anchor = _base_plan()
            row = _worker(anchor, seeds, opponent, None, prune_slack)
            row.update(generation=generation, outcome="completed", stage1_seed=None,
                       stages=[{"stage": "full-anchor", "games": row["games_run"]}], heat=heat,
                       completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            _append_log(log_path, row); rows.append(row); seen.add(row["plan_id"])
            candidates = [p for p in candidates if _plan_key(p) not in seen]
            generation += 1
        while time.perf_counter() < deadline and (max_evals is None or len(rows) - total_before < max_evals):
            ranked = sorted((r for r in rows if r.get("outcome") == "completed"),
                            key=lambda r: r.get("mean_margin", -1e12), reverse=True)
            elites = [r["params"] for r in ranked[:max(2, population // 4)]] or [_base_plan()]
            while len(candidates) < population:
                parent = rng.choice(elites)
                if len(elites) > 1 and rng.random() < .35:
                    parent = _crossover(parent, rng.choice(elites), rng)
                child = _mutate(parent, rng, heat=heat)
                if _plan_key(child) not in seen and all(_plan_key(child) != _plan_key(x) for x in candidates):
                    candidates.append(child)
            room = population
            if max_evals is not None:
                room = min(room, max_evals - (len(rows) - total_before))
            batch = candidates[:room]
            candidates = candidates[room:]
            if not batch: break
            incumbent = ranked[0]
            incumbent_games = {g["seed"]: g for g in incumbent["games"]}
            incumbent_margin = {g["seed"]: g["margin"] for g in incumbent["games"]}
            race_seeds = seeds[:2]
            staged: list[tuple[dict[str, Any], dict[str, Any]]] = []
            futures = {}
            for plan in batch:
                futures[pool.submit(_worker, plan, race_seeds, opponent, incumbent_games, prune_slack)] = plan
            for future in as_completed(futures):
                plan = futures[future]
                one = future.result()
                staged.append((plan, one))

            finalists = []
            for plan, one in staged:
                game = one["games"][0] if one["games"] else None
                incumbent_race = statistics.fmean(incumbent_margin[s] for s in race_seeds)
                qualifies = bool(len(one["games"]) == 2 and not one["pruned"] and not one["error"] and
                                 one["mean_margin"] >= incumbent_race - race_slack)
                if qualifies:
                    remaining = tuple(s for s in seeds if s not in race_seeds)
                    fut = pool.submit(_worker, plan, remaining, opponent, incumbent_games, prune_slack)
                    finalists.append((fut, plan, one))
                    continue
                row = one
                row["mean_margin"] = PRUNED_SCORE
                row["outcome"] = "pruned" if one.get("pruned") else "raced_out"
                row["stage1_margin"] = one.get("mean_margin", PRUNED_SCORE)
                row["stages"] = [{"stage": 1, "seeds": list(race_seeds), "qualified": False,
                                  "pruned": bool(game and game["pruned"])}]
                row.update(generation=generation, heat=heat,
                           completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                _append_log(log_path, row); rows.append(row); seen.add(row["plan_id"])

            successes = 0
            for fut, plan, one in finalists:
                rest = fut.result()
                games = one["games"] + rest["games"]
                row = _panel_result(games)
                row.update(params=_normalise(plan), plan_id=_plan_key(plan),
                           error=one.get("error") or rest.get("error"),
                           eval_seconds=one.get("eval_seconds", 0) + rest.get("eval_seconds", 0),
                           stage1_margin=one["mean_margin"], generation=generation, heat=heat,
                           outcome="pruned" if row["pruned"] else "completed",
                           stages=[{"stage": 1, "seeds": list(race_seeds), "qualified": True, "pruned": False},
                                   {"stage": 2, "seeds": [s for s in seeds if s not in race_seeds],
                                    "pruned": rest.get("pruned", False)}],
                           completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                if row["outcome"] == "completed" and row["mean_margin"] > incumbent["mean_margin"]:
                    successes += 1
                _append_log(log_path, row); rows.append(row); seen.add(row["plan_id"])
            success_rate = successes / max(1, len(batch))
            heat = max(.25, min(3.0, heat * math.exp(success_rate - .2)))
            best = max((r for r in rows if r.get("outcome") == "completed"), key=lambda r: r["mean_margin"])
            counts = {x: sum(r.get("outcome") == x for r in rows) for x in ("completed", "pruned", "raced_out")}
            print(f"gen={generation} eval={len(rows)} best={best['mean_margin']:.0f} heat={heat:.2f} counts={counts}", flush=True)
            generation += 1
            candidates = []

    completed = [r for r in rows if r.get("outcome") == "completed"]
    best_row = max(completed, key=lambda r: r["mean_margin"])
    measured = _run_panel(best_row["params"], seeds, opponent)
    wall_previous = 0.0
    old_best = RESULT_ROOT / "best.json"
    if old_best.exists():
        wall_previous = float(json.loads(old_best.read_text(encoding="utf-8")).get("wall_seconds", 0))
    payload = {
        "schema_version": 3, "engine": engine_version(), "seeds": list(seeds),
        "seats": [s % 2 for s in seeds],
        "opponent": str(opponent.relative_to(REPO)), "source_bot": str(BOT.relative_to(REPO)),
        "plan": best_row["params"], "plan_id": best_row["plan_id"], **measured,
        "evaluations": len(rows), "new_evaluations": len(rows)-total_before,
        "outcomes": {x: sum(r.get("outcome") == x for r in rows) for x in ("completed", "pruned", "raced_out")},
        "games_run": sum(int(r.get("games_run", 0)) for r in rows),
        "steps_run": sum(int(r.get("steps_run", 0)) for r in rows),
        "heat": heat, "prune_slack": prune_slack, "race_slack": race_slack,
        "wall_seconds": wall_previous + time.perf_counter()-started,
        "baseline": baseline, "sensitivity": _sensitivity(completed), "plateau": _plateau(completed),
        "derived_day_plan": _derived_plan(best_row["params"]),
    }
    old_best.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _plain_plan(p: dict[str, Any]) -> str:
    if p.get("raw_bot"):
        return "Exact raw bots/brain_strawberry/main.py behavior (plan-space anchor)."
    opening = ", ".join(f"{v} {k.lower()}" for k,v in p["crop_mix"]["opening"].items() if v)
    replants = "; ".join(f"{phase.replace('_', ' ')} " + "/".join(f"{c.lower()} {v}%" for c,v in p["crop_mix"][phase].items() if v) for phase in PHASES[1:])
    lands = ", ".join(f"{q} d{p['land_days'][q]}" if p["land_days"][q] < 30 else f"{q} never" for q in ("NE", "SW", "SE"))
    animals = []
    for a,c in p["animals"].items():
        animals.append(f"{a.lower()} {c['total']} from d{c['start_day']} every {c['interval']}d x{c['batch']}" if c["total"] else f"no {a.lower()}")
    hands = "/".join(map(str,p["hands_by_5day"]))
    metered = [x.lower() for x,v in p["sell_policy"].items() if v == "metered"]
    fertilized = [x.lower() for x,v in p["fertilize"].items() if v]
    return f"Open {opening}. Replant targets: {replants}. Land: {lands}. Animals: {', '.join(animals)}. Hands by five-day block: {hands}. Fertilise: {', '.join(fertilized) or 'none'}. Metered: {', '.join(metered) or 'none'}; all others immediate."


def _money(value: float) -> str:
    return f"${value:,.0f}"


def emit_bot(plan: dict[str, Any], destination: Path) -> Path:
    source = BOT.read_text(encoding="utf-8")
    normalised = _normalise(plan)
    if normalised.get("raw_bot"):
        adapter = ""
    else:
        template = Path(__file__).with_name("EMIT_ADAPTER.txt").read_text(encoding="utf-8")
        adapter = "\n" + template.replace("__PLAN_LITERAL__", repr(normalised))
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "main.py"
    path.write_text(source.rstrip() + "\n" + adapter, encoding="utf-8")
    return path


def verify_emitted(plan: dict[str, Any], emitted: Path, opponent: Path, seed: int = CONTEST_SEEDS[0]) -> dict[str, float]:
    expected = _run_episode(seed, plan, seed % 2, opponent)["our_bank"]
    ke = import_engine()
    seat = seed % 2
    agents = [str(emitted), str(opponent)] if seat == 0 else [str(opponent), str(emitted)]
    env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run(agents)
    actual = _bank_curve(env, seat)[-1]
    if actual != expected:
        raise RuntimeError(f"emitted bot mismatch on seed {seed}: {actual} != {expected}")
    return {"seed": seed, "in_process_bank": expected, "emitted_bank": actual}


def write_report(result: dict[str, Any]) -> Path:
    path = RESULT_ROOT / "REPORT.md"
    panels = [("brain_strawberry baseline", result["baseline"]), ("hindsight-v3 best", result)]
    lines = ["# Hindsight Solver v3 — Many-draw contested", "",
             f"Engine `{engine_version()}`; seeds {', '.join(map(str, result['seeds']))}; one alternating seat per seed; live opponent `{result['opponent']}`. Objective: mean paired margin.", "",
             "## Panel results", "", "| strategy | mean margin | worst margin | mean our bank | mean opponent bank |", "|---|---:|---:|---:|---:|"]
    for name, panel in panels:
        lines.append(f"| {name} | {_money(panel['mean_margin'])} | {_money(panel['worst_margin'])} | {_money(statistics.fmean(panel['our_banks']))} | {_money(statistics.fmean(panel['their_banks']))} |")
    lines += ["", "| strategy | seed | seat | our bank | opponent bank | margin |", "|---|---:|---:|---:|---:|---:|"]
    for name, panel in panels:
        for game in panel["games"]:
            lines.append(f"| {name} | {game['seed']} | {game['seat']} | {_money(game['our_bank'])} | {_money(game['their_bank'])} | {_money(game['margin'])} |")
    lines += ["", "## Best plan", "", _plain_plan(result["plan"]), "", "## Per-day bank (ours)", "",
              "| day | baseline | best |", "|---:|---:|---:|"]
    for day in range(30):
        lines.append(f"| {day} | {_money(result['baseline']['mean_our_bank_by_day'][day])} | {_money(result['mean_our_bank_by_day'][day])} |")
    lines += ["", "## Sensitivity", "", "| parameter | correlation with mean margin |", "|---|---:|"]
    for item in result.get("sensitivity", [])[:12]:
        lines.append(f"| `{item['parameter']}` | {item['correlation']:+.3f} |")
    lines += ["", "Associations are observational within the correlated evolutionary population.", "",
              "## Search efficiency", "",
              f"The resumable log contains {result['evaluations']} attempts: {result['outcomes']['completed']} completed panels, {result['outcomes']['pruned']} pruned, and {result['outcomes']['raced_out']} raced out. It ran for {result['wall_seconds']/60:.1f} cumulative minutes and executed {result['games_run']} games ({result['games_run']/max(1,result['wall_seconds']):.3f} games/s). The best first appeared at completed evaluation {result['plateau']['first_best_evaluation']}; gains over the final 100 and 400 completed panels were {_money(result['plateau']['gain_last_100'])} and {_money(result['plateau']['gain_last_400'])}. Final heat {result['heat']:.3f}; prune slack {_money(result['prune_slack'])}; race slack {_money(result['race_slack'])}; worker cap 8."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--population", type=int, default=8, help="candidates per generation; v0's 24 scaled for four-game evaluations")
    parser.add_argument("--hours", type=float, default=2.0, help="total wall-clock search budget")
    parser.add_argument("--max-evals", type=int, help="new four-game evaluations (useful for smoke tests)")
    parser.add_argument("--search-seed", type=int, default=20260907)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CONTEST_SEEDS), help="panel seeds; seat is seed %% 2")
    parser.add_argument("--opponent", type=Path, default=OPPONENT)
    parser.add_argument("--prune-slack", type=float, default=12_000)
    parser.add_argument("--race-slack", type=float, default=30_000)
    parser.add_argument("--init-from", nargs="*", type=Path, default=[CONTESTED_V1], help="plan JSON files; base is always included")
    parser.add_argument("--emit-bot", type=Path, help="directory receiving standalone main.py")
    args=parser.parse_args(argv)
    if not 1 <= args.workers <= 8: parser.error("--workers must be in 1..8")
    if args.population < args.workers: parser.error("--population must be >= --workers")
    opponent = args.opponent if args.opponent.is_absolute() else REPO / args.opponent
    init_from = [p if p.is_absolute() else REPO / p for p in args.init_from]
    result=solve_contested(args.workers,args.population,args.hours,args.max_evals,args.search_seed,
                           tuple(args.seeds),opponent,args.prune_slack,args.race_slack,init_from)
    if args.emit_bot:
        destination = args.emit_bot if args.emit_bot.is_absolute() else REPO / args.emit_bot
        emitted = emit_bot(result["plan"], destination)
        result["emitted_bot"] = str(emitted.relative_to(REPO))
        result["emitted_verification"] = verify_emitted(result["plan"], emitted, opponent)
        (RESULT_ROOT / "best.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"emitted {emitted}; verification={result['emitted_verification']}")
    report=write_report(result)
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
