"""Tight, auditable LP outer bound for the pinned seed-3910 game.

This layers the missing physical and market envelopes on the round-1 model.
It deliberately remains an LP: fractional jobs enlarge the integer game and
make the returned optimum a reproducible upper bound.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, linprog
from scipy.sparse import coo_matrix, vstack

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._engine import to_plain
from scripts.fixed_game import bound as v1
from scripts.fixed_game.pin import DEFAULT_INSTANCE, load_instance, make_env

OUT = ROOT / "data" / "results" / "fixed-game" / "3910"
DAYS = 30
CROPS = v1.CROPS
ANIMALS = v1.ANIMALS
PRODUCTS = v1.PRODUCTS
MOVE_OPS = v1.MOVE_OPS


def _add(row: dict[int, float], col: int, value: float) -> None:
    row[col] = row.get(col, 0.0) + value


def _quadrant_index(x: int, y: int) -> int | None:
    if x < 5 and y < 5:
        return None
    if y < 5:
        return 0
    if x < 5:
        return 1
    return 2


def _distance_to_shed(x: int, y: int) -> int:
    return min(abs(x - sx) + abs(y - sy) for sx, sy in ((4, 4), (5, 4), (4, 5), (5, 5)))


def _crop_last_day(kg: Any, crop: str, planted: int) -> int:
    cd = kg.CROPS[crop]
    if cd["ongoing"]:
        return min(DAYS - 1, planted + cd["first_yield_day"] + (cd["max_yield"] - 1) * cd["interval"])
    return min(DAYS - 1, planted + cd["max_yield_day"])


def _crop_min_water_visits(kg: Any, crop: str, planted: int) -> int:
    """Lower bound for a crop that realizes its lifetime cap.

    Surviving through age L needs at least ceil((L+1)/2) watered days because
    planting starts with one miss.  One-time crops also need enough bonus-window
    waters to build max_yield from the initial unit; granting free fertilizer
    means one water may add two units.
    """
    last = _crop_last_day(kg, crop, planted)
    age = last - planted
    survival = math.ceil((age + 1) / 2)
    cd = kg.CROPS[crop]
    production = math.ceil((cd["max_yield"] - (0 if cd["ongoing"] else 1)) / 2)
    return max(survival, production)


def _animal_envelope(kg: Any, animal: str, start: int, end: int = DAYS - 1) -> tuple[int, int]:
    """(max units, base units) with harvest-before-cap and perfect care."""
    ad = kg.ANIMALS[animal]
    ticks = [d for d in range(start + ad["first_yield_day"], end + 1, ad["interval"])]
    if not ticks:
        return 0, 0
    later = {"GOOSE": 2, "COW": 3, "SHEEP": 4}[animal]
    return ad["max_held"] + later * (len(ticks) - 1), len(ticks)


def _market_upper_segments(kg: Any, instance: dict[str, Any], product: str) -> list[tuple[int, int]]:
    """Block widths/prices after moving every town drain before every sale.

    This dominates every real sale ordering.  At the $1 floor inventory no
    longer rises, exactly matching _commit_unit.
    """
    drain = sum(int(r["drain"][product]) for r in instance["town"]["daily_drain"])
    inventory = 10000 - drain
    marginal: list[int] = []
    for _ in range(20000):
        price = int(kg.market_price(product, inventory))
        marginal.append(price)
        if price > 1:
            inventory += 1
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(marginal):
        width = 1 if i < 2000 else (10 if i < 10000 else 100)
        width = min(width, len(marginal) - i)
        out.append((width, marginal[i]))  # first marginal is block maximum
        i += width
    return out


def build(instance: dict[str, Any], travel_multiplier: float) -> tuple[v1.Model, dict[str, Any]]:
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    m, q = v1.build_model(instance, 0.0)
    # Round 1's deliberately huge speculative credit is replaced below by an
    # exact buy/sell fact and a separately computed carry ceiling.
    m.ub[q["trade_credit"]] = 0.0

    plant, buy, alive, sale, stock = (q[k] for k in ("plant", "buy", "alive", "sale", "stock"))
    hire, land, seg, segment_price = (q[k] for k in ("hire", "land", "seg", "segment_price"))

    # One global, cumulative market envelope.  Round 1 reset prior sales each
    # day; this constraint makes that credited revenue no larger than selling
    # the same quantity after all season drain, the best possible inventory.
    global_seg: dict[tuple[str, int], int] = {}
    global_price: dict[tuple[str, int], int] = {}
    market_caps: dict[str, list[tuple[int, int]]] = {}
    for p in PRODUCTS:
        blocks = _market_upper_segments(kg, instance, p)
        market_caps[p] = blocks
        for k, (width, price) in enumerate(blocks):
            global_seg[p, k] = m.var(f"v2_market_segment:{p}:{k}", ub=width)
            global_price[p, k] = price
        row = {sale[p, d]: 1.0 for d in range(DAYS)}
        for k in range(len(blocks)):
            _add(row, global_seg[p, k], -1.0)
        m.con(f"v2_market_quantity:{p}", row, lb=0, ub=0)

    old_revenue: dict[int, float] = {}
    new_revenue: dict[int, float] = {}
    for p in PRODUCTS:
        for d in range(DAYS):
            for k in range(len(v1.BREAKS) - 1):
                _add(old_revenue, seg[p, d, k], segment_price[p, d, k])
        for k in range(len(market_caps[p])):
            _add(new_revenue, global_seg[p, k], global_price[p, k])
    revenue_cap = dict(old_revenue)
    for col, coef in new_revenue.items():
        _add(revenue_cap, col, -coef)
    m.con("v2_cumulative_market_revenue", revenue_cap, ub=0)

    # Exact lifetime production envelope.  Each care action can add at most one
    # unit beyond base production; the perfect-care ramp gives a second cap.
    care = {a: m.var(f"v2_care_actions:{a}") for a in ANIMALS}
    harvest_actions = {a: m.var(f"v2_harvest_actions:{a}") for a in ANIMALS}
    for a in ANIMALS:
        p = kg.ANIMALS[a]["product"]
        perfect: dict[int, float] = {sale[p, d]: 1 for d in range(DAYS)}
        base_plus_care: dict[int, float] = dict(perfect)
        _add(base_plus_care, care[a], -1)
        care_cap: dict[int, float] = {care[a]: 1}
        for t in range(DAYS):
            ad = kg.ANIMALS[a]
            ticks = list(range(t + ad["first_yield_day"], DAYS, ad["interval"]))
            later = {"GOOSE": 2, "COW": 3, "SHEEP": 4}[a]
            for j, d in enumerate(ticks):
                _add(perfect, alive[a, t, d], -(ad["max_held"] if j == 0 else later))
                _add(base_plus_care, alive[a, t, d], -1)
            for d in range(t, DAYS): _add(care_cap, alive[a, t, d], -1)
        m.con(f"v2_animal_perfect_ramp:{a}", perfect, ub=0)
        m.con(f"v2_animal_care_units:{a}", base_plus_care, ub=0)
        m.con(f"v2_animal_care_cap:{a}", care_cap, ub=0)
        row = {harvest_actions[a]: -kg.ANIMALS[a]["max_held"]}
        for d in range(DAYS): _add(row, sale[p, d], 1)
        m.con(f"v2_animal_harvest_min:{a}", row, ub=0)
    total_feed = {q["feed_crop"][d]: -1 for d in range(DAYS)}
    for d in range(DAYS): _add(total_feed, q["feed_buy"][d], -1)
    for a in ANIMALS: _add(total_feed, care[a], 1)
    m.con("v2_care_requires_feed", total_feed, ub=0)

    # Crop watering lower bound proportional to realized crop output.  Free
    # fertilizer and fractional plants make it an outer relaxation.
    crop_water = {c: m.var(f"v2_water_actions:{c}") for c in CROPS}
    for c in CROPS:
        row = {crop_water[c]: 1}
        for t in range(DAYS):
            if _crop_last_day(kg, c, t) < DAYS:
                _add(row, plant[c, t], -_crop_min_water_visits(kg, c, t))
        m.con(f"v2_crop_water:{c}", row, lb=0)

    # Remove round-1's assumption that every live animal is harvested on every
    # scheduled tick.  Real play may let held product accumulate.  The aggregate
    # harvest lower bound above is valid instead.
    for d, name in enumerate(m.row_names):
        if not name.startswith("labour:"):
            continue
        day = int(name.split(":")[1])
        row = m.rows[d]
        for a in ANIMALS:
            ad = kg.ANIMALS[a]
            for t in range(day + 1):
                age = day - t
                if age >= ad["first_yield_day"] and (age - ad["first_yield_day"]) % ad["interval"] == 0:
                    row.pop(alive[a, t, day], None)

    # Season action capacity captures watering, care and the minimum animal
    # harvest count omitted from daily rows.  Pooling all 30 days is optimistic.
    season_actions: dict[int, float] = {}
    for c in CROPS: _add(season_actions, crop_water[c], 1)
    for a in ANIMALS:
        _add(season_actions, care[a], 1)
        _add(season_actions, harvest_actions[a], 1)
    # All existing daily required work also counts. Reconstruct the portion
    # that cannot be skipped in real play.
    for c in CROPS:
        for d in range(DAYS): _add(season_actions, plant[c, d], 2)  # plant + eventual harvest
    for a in ANIMALS:
        for d in range(DAYS): _add(season_actions, buy[a, d], 3)  # build, pickup, place
    for d in range(DAYS):
        _add(season_actions, q["collect"][d], 1)
        _add(season_actions, q["feed_crop"][d], 1)
        _add(season_actions, q["feed_buy"][d], 1)
        for k in range(40): _add(season_actions, hire[d, k], -24)
    m.con("v2_season_labour", season_actions, ub=24 * DAYS)

    # Tile-indexed assignment and one-structure/plant occupancy.  This is an LP
    # fractional coloring of every crop and surviving animal cohort.
    coords = [(x, y) for y in range(10) for x in range(10)]
    plant_at: dict[tuple[str, int, int], int] = {}
    animal_at: dict[tuple[str, int, int], int] = {}
    worked = {(tile, d): m.var(f"v2_worked:{tile}:{d}", ub=1) for tile in range(100) for d in range(DAYS)}
    travel = {d: m.var(f"v2_travel:{d}") for d in range(DAYS)}
    for c in CROPS:
        for t in range(DAYS):
            row = {plant[c, t]: -1}
            for tile in range(100):
                col = m.var(f"v2_plant_at:{c}:{tile}:{t}", ub=1)
                plant_at[c, tile, t] = col; _add(row, col, 1)
                x, y = coords[tile]; qi = _quadrant_index(x, y)
                if qi is not None:
                    unlock = {col: 1}
                    for dd in range(t + 1): _add(unlock, land[qi, dd], -1)
                    m.con(f"v2_crop_unlocked:{c}:{tile}:{t}", unlock, ub=0)
                m.con(f"v2_crop_visit_plant:{c}:{tile}:{t}", {col: 1, worked[tile, t]: -1}, ub=0)
                hd = _crop_last_day(kg, c, t)
                m.con(f"v2_crop_visit_harvest:{c}:{tile}:{t}", {col: 1, worked[tile, hd]: -1}, ub=0)
            m.con(f"v2_plant_assignment:{c}:{t}", row, lb=0, ub=0)
    for a in ANIMALS:
        for t in range(DAYS):
            row = {buy[a, t]: -1}
            for tile in range(100):
                col = m.var(f"v2_animal_at:{a}:{tile}:{t}", ub=1)
                animal_at[a, tile, t] = col; _add(row, col, 1)
                x, y = coords[tile]; qi = _quadrant_index(x, y)
                if qi is not None:
                    unlock = {col: 1}
                    for dd in range(t + 1): _add(unlock, land[qi, dd], -1)
                    m.con(f"v2_animal_unlocked:{a}:{tile}:{t}", unlock, ub=0)
                m.con(f"v2_animal_visit:{a}:{tile}:{t}", {col: 1, worked[tile, t]: -1}, ub=0)
            m.con(f"v2_animal_assignment:{a}:{t}", row, lb=0, ub=0)
    for tile, (x, y) in enumerate(coords):
        for d in range(DAYS):
            occ: dict[int, float] = {}
            for c in CROPS:
                for t in range(d + 1):
                    if d <= _crop_last_day(kg, c, t): _add(occ, plant_at[c, tile, t], 1)
            # Animal occupancy remains enforced exactly in aggregate by the
            # round-1 alive/tile-capacity rows. Placement is tile-indexed here;
            # omitting persistent per-tile identity is a relaxation that avoids
            # 139,500 redundant cohort-day coloring columns.
            for a in ANIMALS:
                _add(occ, animal_at[a, tile, d], 1)
            m.con(f"v2_tile_one_choice:{tile}:{d}", occ, ub=1)
            dist = _distance_to_shed(x, y)
            m.con(f"v2_tile_distance:{tile}:{d}", {travel[d]: 1, worked[tile, d]: -dist}, lb=0)

    # Daily travel: every worker starts at a shed-access tile.  Reaching the
    # furthest worked tile costs at least its map distance.  The realistic
    # variant additionally charges the measured 1.84 moves per non-move action.
    for ridx, name in enumerate(m.row_names):
        if name.startswith("labour:"):
            day = int(name.split(":")[1]); _add(m.rows[ridx], travel[day], 1)
            workers = {travel[day]: -1, q["feed_crop"][day]: 1, q["feed_buy"][day]: 1}
            for k in range(40): _add(workers, hire[day, k], -1)
            m.con(f"v2_feed_tile_travel:{day}", workers, ub=1)
            collectors = {travel[day]: -1, q["collect"][day]: 1}
            for k in range(40): _add(collectors, hire[day, k], -1)
            m.con(f"v2_collect_tile_travel:{day}", collectors, ub=1)
    if travel_multiplier:
        # Apply the empirical ratio only to the season work envelope; it is not
        # claimed as a proof constraint.
        realistic = dict(season_actions)
        for col, coef in list(realistic.items()):
            if coef > 0: realistic[col] = coef * (1 + travel_multiplier)
        m.con("v2_realistic_travel", realistic, ub=24 * DAYS)

    q.update(locals())
    return m, q


def solve(instance: dict[str, Any], travel_multiplier: float) -> tuple[Any, v1.Model, dict[str, Any]]:
    m, q = build(instance, travel_multiplier)
    c, _, bounds, cons, A = m.arrays()
    eq, ub, beq, bub, ubmeta = [], [], [], [], []
    for r, (lo, hi) in enumerate(zip(m.row_lb, m.row_ub)):
        if np.isfinite(lo) and np.isfinite(hi) and abs(lo-hi) < 1e-10:
            eq.append(r); beq.append(hi)
        else:
            if np.isfinite(hi): ub.append(A.getrow(r)); bub.append(hi); ubmeta.append((r, 1))
            if np.isfinite(lo): ub.append(-A.getrow(r)); bub.append(-lo); ubmeta.append((r, -1))
    res = linprog(c, A_ub=vstack(ub).tocsr(), b_ub=bub, A_eq=A[eq], b_eq=beq,
                  bounds=list(zip(m.lb, m.ub)), method="highs")
    if not res.success: raise RuntimeError(res.message)
    duals = {}
    for pos, r in enumerate(eq): duals[m.row_names[r]] = float(res.eqlin.marginals[pos])
    for pos, (r, sign) in enumerate(ubmeta): duals[m.row_names[r]] = float(res.ineqlin.marginals[pos]) * sign
    res["row_duals"] = duals
    return res, m, q


def _load_exec_b_7baa291() -> Any:
    # The requested known trajectory is specifically the 7baa291 executor;
    # the branch now contains a later round-2 bot, so load the committed source
    # directly without modifying the working tree.
    source = subprocess.check_output(
        ["git", "show", "7baa291:bots/exec_b/main.py"], cwd=ROOT, text=True, encoding="utf-8"
    )
    module = types.ModuleType("fixed_bound_v2_exec_b_7baa291")
    module.__file__ = str(ROOT / "bots" / "exec_b" / "main.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def run_exec_b(instance: dict[str, Any], *, pinned: bool) -> dict[str, Any]:
    module = _load_exec_b_7baa291()
    if pinned:
        env = make_env(instance)
    else:
        from scripts._engine import import_engine
        env = import_engine().make("kaggriculture", configuration={"seed": int(instance["seed"])}, debug=True)
    env.run([module.agent, "pass"])
    actions = [to_plain(s[0].action or {"farmer":["PASS"],"hands":[],"market":[]}) for s in env.steps[1:]]
    while len(actions) < 720: actions.append({"farmer":["PASS"],"hands":[],"market":[]})
    return {"bank": float(env.steps[-1][0].observation.farms[0].money), "actions": actions[:720], "env": env}


def routing_from_env(env: Any) -> dict[str, Any]:
    moves = actions = 0
    by_day = []
    for day in range(DAYS):
        dm = da = 0
        for j in range(day*24+1, min((day+1)*24+1, len(env.steps))):
            a = to_plain(env.steps[j][0].action or {})
            for unit in [a.get("farmer", ["PASS"]), *(a.get("hands", []) or [])]:
                op = unit[0] if isinstance(unit, list) and unit else "PASS"
                if op in MOVE_OPS: dm += 1
                elif op != "PASS": da += 1
        moves += dm; actions += da
        by_day.append({"day":day,"moves":dm,"nonmove_actions":da,"moves_per_nonmove_action":dm/da if da else 0})
    return {"engine_final_bank":float(env.steps[-1][0].observation.farms[0].money),
            "season_moves":moves,"season_nonmove_actions":actions,
            "season_moves_per_nonmove_action":moves/actions,"day_20":by_day[20],"per_day":by_day}


def model_value_of_run(run: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg
    from scripts import kpi
    ledger = kpi._analyze(run["env"], 0, int(instance["seed"]))
    realised_revenue = sum(float(x) for x in ledger["realised_dollars"].values())
    expenses = 3000.0 + realised_revenue - float(run["bank"])
    relaxed_revenue = 0.0
    by_product = {}
    for p in PRODUCTS:
        left = int(ledger["sold_units"][p]); value = 0
        for width, price in _market_upper_segments(kg, instance, p):
            take = min(left, width); value += take*price; left -= take
            if left <= 0: break
        relaxed_revenue += value
        by_product[p] = {"units":ledger["sold_units"][p],"realised":ledger["realised_dollars"][p],"model_revenue":value}
    return {"realised_bank":run["bank"],"realised_revenue":realised_revenue,"actual_expenses":expenses,
            "model_value":3000.0-expenses+relaxed_revenue,"model_revenue":relaxed_revenue,
            "looseness":3000.0-expenses+relaxed_revenue-run["bank"],"by_product":by_product,
            "kpi":ledger}


def trading_ceiling(instance: dict[str, Any]) -> dict[str, Any]:
    """Independent DP-style upper estimate with true 100-unit shed and slots.

    One bulk BUY and one bulk SELL order suffice, so slots do not bind this
    carry strategy.  Grant a fresh empty 100-unit shed before every town tick
    and the best full-season price rise to every unit: deliberately an upper
    ceiling, reported separately rather than injected into farm production.
    """
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg
    out = {}
    unlocks = {int(x["day"]): x["shop"] for x in instance["town"]["shop_unlocks"]}
    for p in ("WHEAT", "FERTILIZER"):
        shops: list[str] = []; ceiling = 0; events = []
        for step in range(719):
            day = step // 24
            if step % 24 == 0 and day in unlocks:
                # Duplicates are significant, so append by the exact unlock day.
                shops.append(unlocks[day])
            delta = (1 if p != "FERTILIZER" and step % 24 == 0 else 0)
            if step % 4 == 0:
                for shop in shops:
                    members = kg.SHOPS[shop]
                    if p in members: delta += 2 if len(members) == 1 else 1
            if not delta: continue
            # Wheat's sqrt scarcity side is concave and its glut side is log;
            # the largest finite-delta rise is at/near I0. This wide integer
            # scan contains the kink and both monotone tails.
            gain = max(kg.market_price(p, inv-delta)-kg.market_price(p, inv)
                       for inv in range(-10000, 30001))
            ceiling += 100 * max(0, gain)
            events.append({"step":step,"drain":delta,"max_unit_appreciation":gain})
        drain = sum(e["drain"] for e in events)
        out[p] = {"season_drain":drain,"drain_events":len(events),"events":events,
                  "coarse_100_unit_ceiling":ceiling}
    return out


def extract(res: Any, m: v1.Model, q: dict[str, Any], instance: dict[str, Any], ratio: float) -> dict[str, Any]:
    # v1's presentation-only greedy interval coloring rounds every LP value to
    # six decimals and can leave ~2e-5 residue. Nudge displayed occupancies
    # inward; objective and all reported economics remain from the solved x.
    shown = OptimizeResult(res); shown.x = np.array(res.x, copy=True)
    for col in list(q["plant"].values()) + list(q["alive"].values()):
        if shown.x[col] > 1e-4: shown.x[col] -= 5e-5
    base = v1.extract(shown, m, q, instance, 0.0)
    base["objective_final_bank"] = round(float(res.x[q["cash"][DAYS-1]]), 2)
    row_index = {name:i for i,name in enumerate(m.row_names)}
    def slack(name: str) -> float:
        i = row_index[name]
        value = sum(coef*res.x[col] for col,coef in m.rows[i].items())
        return float(m.row_ub[i]-value) if np.isfinite(m.row_ub[i]) else float(value-m.row_lb[i])
    for d, row in enumerate(base["binding_constraints"]):
        row["labour_slack_turns"] = round(slack(f"labour:{d}"), 6)
        row["labour_binding"] = row["labour_slack_turns"] < 1e-5
        row["order_slot_slack"] = round(slack(f"market_order_slots:{d}"), 6)
        row["order_slots_binding"] = row["order_slot_slack"] < 1e-5
        row["tile_slack"] = round(slack(f"tile_capacity:{d}"), 6)
        row["tile_binding"] = row["tile_slack"] < 1e-5
        row["cash_binding"] = float(res.x[q["cash"][d]]) < 1e-5
    base["season_labour_slack_turns"] = round(slack("v2_season_labour"), 6)
    base["season_labour_binding"] = base["season_labour_slack_turns"] < 1e-5
    base["model"] = "valid_v2" if ratio == 0 else "realistic_travel_v2"
    base["travel_moves_per_required_action"] = ratio
    base["speculative_trade_allowance"] = 0
    return base


def fixed_exec_b_model_value(run: dict[str, Any]) -> dict[str, Any]:
    """Score exec_b's actual tape under the model's sole remaining price relaxation."""
    env = run["env"]
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg
    sold = defaultdict(int); costs = 3000 - run["bank"]
    # Re-run market attribution exactly enough to obtain quantities; model value
    # then prices those quantities at the all-drain-first cumulative envelope.
    for s in env.steps[1:]:
        action = to_plain(s[0].action or {})
        for order in action.get("market", []) or []:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                sold[order[1]] += int(order[2])
    # Requested quantities can exceed shed stock. Use actual realized revenue
    # from bank/cost is unavailable here; cap requested sales by final physical
    # production is not reliable. A direct interpreter ledger is generated in
    # main and replaces this provisional field.
    return {"requested_sales": dict(sold), "note": "actual fixed-schedule value is computed by replay attribution in main"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE); ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args(); instance = load_instance(args.instance); args.output_dir.mkdir(parents=True, exist_ok=True)
    run = run_exec_b(instance, pinned=True)
    reference_run = run_exec_b(instance, pinned=False)
    routing = routing_from_env(reference_run["env"])
    valid_res, valid_m, valid_q = solve(instance, 0.0); valid = extract(valid_res, valid_m, valid_q, instance, 0.0)
    # The requested diagnostic uses the previously measured 1.842 moves/action.
    ratio = 1.8421821724305893
    realistic_res, realistic_m, realistic_q = solve(instance, ratio); realistic = extract(realistic_res, realistic_m, realistic_q, instance, ratio)
    trades = trading_ceiling(instance)
    # The cumulative-market farm model forbids bought-product resale. Add the
    # independently valid carry ceiling to the reported certified objective.
    carry = sum(v["coarse_100_unit_ceiling"] for v in trades.values())
    valid["farm_bound"] = valid["objective_final_bank"]; valid["objective_final_bank"] += carry
    realistic["farm_bound"] = realistic["objective_final_bank"]; realistic["objective_final_bank"] += carry
    calibration = model_value_of_run(reference_run, instance)
    pinned_calibration = model_value_of_run(run, instance)
    details = {"valid": valid, "realistic": realistic, "product_trading": trades,
               "exec_b_reference_calibration": calibration, "exec_b_pinned_calibration": pinned_calibration,
               "routing": routing,
               "instance_comparison":"The reference calibration uses this seed's endogenous action-coupled shop draw; the pinned calibration uses the pass/pass schedule."}
    (args.output_dir / "exec-b-7baa291-actions-v2.json").write_text(json.dumps(reference_run["actions"], indent=2)+"\n", encoding="utf-8")
    (args.output_dir / "exec-b-7baa291-pinned-actions-v2.json").write_text(json.dumps(run["actions"], indent=2)+"\n", encoding="utf-8")
    (args.output_dir / "bound-details-v2.json").write_text(json.dumps(details, indent=2)+"\n", encoding="utf-8")
    schedule = {"schema_version":1,"seed":int(instance["seed"]),"seat":0,"opponent":"pass","pinned":True,
                "warning":"Fractional LP witness; execute.py accepts the schema but cannot realize fractional relaxed work.",
                "valid_upper_bound":valid["objective_final_bank"],"realistic_bound":realistic["objective_final_bank"],
                "product_trading_credit":{"WHEAT":carry,"FERTILIZER":0,"kind":"per-drain 100-unit valid carry ceiling"},
                "days":valid["days"],"binding_constraints":valid["binding_constraints"],
                "realistic_days":realistic["days"],"realistic_binding_constraints":realistic["binding_constraints"]}
    (args.output_dir / "schedule-v2.json").write_text(json.dumps(schedule, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"valid":valid["objective_final_bank"],"realistic":realistic["objective_final_bank"],"farm_valid":valid["farm_bound"],"carry":carry,"exec_b_reference":reference_run["bank"],"exec_b_reference_model":calibration["model_value"],"exec_b_pinned":run["bank"],"exec_b_pinned_model":pinned_calibration["model_value"],"ratio":ratio,"variables":len(valid_m.names),"constraints":len(valid_m.rows)}, indent=2))


if __name__ == "__main__": main()
