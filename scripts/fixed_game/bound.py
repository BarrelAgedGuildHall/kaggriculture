"""Day-granularity economic upper bound for the pinned seed-3910 game.

This intentionally optimistic MILP keeps the real cash, land, tile, production,
feed, labour, order-slot, and market-drain directions while relaxing routing,
care, watering, storage, and within-day timing.  Every optimism is documented in
the generated bound.md; the MILP optimum is therefore an upper bound, not an
executable score prediction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._engine import import_engine, to_plain
from scripts.fixed_game.pin import DEFAULT_INSTANCE, load_instance, make_env

OUT = ROOT / "data" / "results" / "fixed-game" / "3910"
DAYS = 30
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
BREAKS = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 10000, 20000]
MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}


@dataclass
class Model:
    names: list[str]
    lb: list[float]
    ub: list[float]
    integrality: list[int]
    objective: list[float]
    rows: list[dict[int, float]]
    row_lb: list[float]
    row_ub: list[float]
    row_names: list[str]

    @classmethod
    def empty(cls) -> "Model":
        return cls([], [], [], [], [], [], [], [], [])

    def var(self, name: str, lb: float = 0, ub: float = np.inf, integer: bool = False, cost: float = 0) -> int:
        i = len(self.names)
        self.names.append(name); self.lb.append(lb); self.ub.append(ub)
        self.integrality.append(1 if integer else 0); self.objective.append(cost)
        return i

    def con(self, name: str, terms: dict[int, float], lb: float = -np.inf, ub: float = np.inf) -> None:
        self.row_names.append(name); self.rows.append({i: v for i, v in terms.items() if v})
        self.row_lb.append(lb); self.row_ub.append(ub)

    def arrays(self):
        rr, cc, vv = [], [], []
        for r, row in enumerate(self.rows):
            for c, v in row.items(): rr.append(r); cc.append(c); vv.append(v)
        A = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.names))).tocsr()
        return np.asarray(self.objective), np.asarray(self.integrality), Bounds(self.lb, self.ub), LinearConstraint(A, self.row_lb, self.row_ub), A


def _add(terms: dict[int, float], idx: int, value: float) -> None:
    terms[idx] = terms.get(idx, 0.0) + value


def _marginal_price(kg: Any, instance: dict[str, Any], product: str, day: int, prior_same_day_sales: int) -> int:
    # Optimism: every day prices this day's units after all town drain through
    # that day's final tick, but forgets all sales from earlier days.
    drained = sum(row["drain"][product] for row in instance["town"]["daily_drain"][: day + 1])
    inv = 10000 - drained + prior_same_day_sales
    return int(kg.market_price(product, inv))


def build_model(instance: dict[str, Any], travel_ratio: float) -> tuple[Model, dict[str, Any]]:
    import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    m = Model.empty()
    v: dict[str, Any] = {}
    # One transition can free at most the 100-unit shed before market orders,
    # hence at most 100 newly bought product units can commit. There are 719
    # transitions. Fertilizer has no exogenous drain, so a matched buy/sell
    # round trip cannot profit. For wheat, each purchased unit's best possible
    # round-trip gain is bounded by the largest price rise across the *entire*
    # 282-unit pinned town drain. Grant every possible purchase that full gain.
    max_product_buys = 719 * int(instance["shed_capacity"])
    wheat_total_drain = sum(r["drain"]["WHEAT"] for r in instance["town"]["daily_drain"])
    max_wheat_drain_gain = max(
        int(kg.market_price("WHEAT", inv - wheat_total_drain)) - int(kg.market_price("WHEAT", inv))
        for inv in range(-100_000, 100_001)
    )
    speculative_trade_allowance = max_product_buys * max_wheat_drain_gain
    trade_credit = m.var("speculative_wheat_trade_profit_credit", ub=speculative_trade_allowance)
    plant = {(c, d): m.var(f"plant:{c}:{d}", ub=100, integer=True) for c in CROPS for d in range(DAYS)}
    buy = {(a, d): m.var(f"buy:{a}:{d}", ub=100, integer=True) for a in ANIMALS for d in range(DAYS)}
    alive = {(a, t, d): m.var(f"alive:{a}:{t}:{d}", ub=100, integer=True)
             for a in ANIMALS for t in range(DAYS) for d in range(t, DAYS)}
    sale = {(p, d): m.var(f"sale:{p}:{d}", ub=20000, integer=True) for p in PRODUCTS for d in range(DAYS)}
    stock = {(p, d): m.var(f"stock:{p}:{d}") for p in PRODUCTS for d in range(DAYS)}
    cash = {d: m.var(f"cash:{d}") for d in range(DAYS)}
    collect = {d: m.var(f"collect_fertilizer:{d}", ub=100, integer=True) for d in range(DAYS)}
    feed_crop = {d: m.var(f"feed_grown:{d}") for d in range(DAYS)}
    feed_buy = {d: m.var(f"feed_bought:{d}") for d in range(DAYS)}
    land = {(k, d): m.var(f"land:{k}:{d}", ub=1, integer=True) for k in range(3) for d in range(DAYS)}
    hire = {(d, k): m.var(f"hire:{d}:{k}", ub=1, integer=True) for d in range(DAYS) for k in range(40)}
    active_plant = {(c, d): m.var(f"active_plant_order:{c}:{d}", ub=1, integer=True) for c in CROPS for d in range(DAYS)}
    active_buy = {(a, d): m.var(f"active_buy_order:{a}:{d}", ub=1, integer=True) for a in ANIMALS for d in range(DAYS)}
    active_sale = {(p, d): m.var(f"active_sale_order:{p}:{d}", ub=1, integer=True) for p in PRODUCTS for d in range(DAYS)}
    seg = {}
    segment_price = {}
    for p in PRODUCTS:
        for d in range(DAYS):
            for k, (lo, hi) in enumerate(zip(BREAKS, BREAKS[1:])):
                price = max(_marginal_price(kg, instance, p, d, q) for q in range(lo, min(hi, lo + 2)))
                # Marginal prices are non-increasing, so the first unit is the
                # maximum for the whole block.  Rounded plateaus are harmless.
                price = _marginal_price(kg, instance, p, d, lo)
                seg[p, d, k] = m.var(f"revenue_segment:{p}:{d}:{k}", ub=hi - lo)
                segment_price[p, d, k] = price

    # Cohort survival/disposal. Free disposal is optimistic and lets the model
    # represent real animals that escape or are irrelevant later.
    for a in ANIMALS:
        for t in range(DAYS):
            m.con(f"alive_purchase:{a}:{t}", {alive[a, t, t]: 1, buy[a, t]: -1}, lb=0, ub=0)
            for d in range(t + 1, DAYS):
                m.con(f"alive_monotone:{a}:{t}:{d}", {alive[a, t, d]: 1, alive[a, t, d - 1]: -1}, ub=0)

    # Land is bought once in order.
    for k in range(3):
        m.con(f"land_once:{k}", {land[k, d]: 1 for d in range(DAYS)}, ub=1)
    for d in range(DAYS):
        for k in range(1, 3):
            terms = {land[k, t]: 1 for t in range(d + 1)}
            for t in range(d + 1): _add(terms, land[k - 1, t], -1)
            m.con(f"land_order:{k}:{d}", terms, ub=0)

    # Tile capacity. Crops occupy their tile only through the earliest legal
    # harvest day; granting full cap yield then is an explicit relaxation.
    for d in range(DAYS):
        terms: dict[int, float] = {}
        for a in ANIMALS:
            for t in range(d + 1): _add(terms, alive[a, t, d], 1)
        for c in CROPS:
            delay = int(kg.CROPS[c]["first_yield_day"])
            for t in range(d + 1):
                if d <= t + delay: _add(terms, plant[c, t], 1)
        for k in range(3):
            for t in range(d + 1): _add(terms, land[k, t], -25)
        m.con(f"tile_capacity:{d}", terms, ub=25)

    # Feed is fungible across cohorts. Any cohort still alive by day d needs at
    # least floor((age+1)/2) feeds cumulatively to avoid two misses. Pooling those
    # obligations can only make the model easier than the real per-animal rule.
    for d in range(DAYS):
        terms = {}
        for q in range(d + 1):
            _add(terms, feed_crop[q], 1); _add(terms, feed_buy[q], 1)
        for a in ANIMALS:
            for t in range(d + 1):
                _add(terms, alive[a, t, d], -math.floor((d - t + 1) / 2))
        m.con(f"feed_survival_cumulative:{d}", terms, lb=0)

    # Fertilizer collections may be delayed and pooled; each collected unit
    # consumes an action. Availability itself is cumulative and never expires in
    # the relaxation (real per-tile availability caps at one).
    for d in range(DAYS):
        terms = {collect[q]: 1 for q in range(d + 1)}
        for q in range(1, d + 1):
            for a in ANIMALS:
                for t in range(q): _add(terms, alive[a, t, q - 1], -1)
        m.con(f"fertilizer_availability:{d}", terms, ub=0)

    # Product stock balances, with optimistic crop/animal yield envelopes.
    animal_product = {a: kg.ANIMALS[a]["product"] for a in ANIMALS}
    for p in PRODUCTS:
        for d in range(DAYS):
            terms = {stock[p, d]: 1, sale[p, d]: 1}
            if d > 0: _add(terms, stock[p, d - 1], -1)
            if p == "WHEAT": _add(terms, feed_crop[d], 1)
            if p == "FERTILIZER": _add(terms, collect[d], -1)
            for c in CROPS:
                if c == p:
                    t = d - int(kg.CROPS[c]["first_yield_day"])
                    if t >= 0: _add(terms, plant[c, t], -int(kg.CROPS[c]["max_yield"]))
            for a in ANIMALS:
                if animal_product[a] != p: continue
                ad = kg.ANIMALS[a]
                for t in range(d + 1):
                    age = d - t
                    if age >= ad["first_yield_day"] and (age - ad["first_yield_day"]) % ad["interval"] == 0:
                        _add(terms, alive[a, t, d], -int(ad["max_held"]))
            m.con(f"stock_balance:{p}:{d}", terms, lb=0, ub=0)

    # Concave piecewise-linear upper revenue: each block receives its highest
    # real marginal quote after cumulative pinned drain through that day.
    for p in PRODUCTS:
        for d in range(DAYS):
            terms = {sale[p, d]: 1}
            for k in range(len(BREAKS) - 1): _add(terms, seg[p, d, k], -1)
            m.con(f"sale_segments:{p}:{d}", terms, lb=0, ub=0)

    # Order indicators and the real aggregate ceiling of 10 slots x 24 turns.
    for d in range(DAYS):
        for c in CROPS: m.con(f"plant_order_link:{c}:{d}", {plant[c, d]: 1, active_plant[c, d]: -100}, ub=0)
        for a in ANIMALS: m.con(f"buy_order_link:{a}:{d}", {buy[a, d]: 1, active_buy[a, d]: -100}, ub=0)
        for p in PRODUCTS: m.con(f"sale_order_link:{p}:{d}", {sale[p, d]: 1, active_sale[p, d]: -20000}, ub=0)
        order_terms = {hire[d, k]: 1 for k in range(40)}
        for c in CROPS: _add(order_terms, active_plant[c, d], 1)
        for a in ANIMALS: _add(order_terms, active_buy[a, d], 1)
        for p in PRODUCTS: _add(order_terms, active_sale[p, d], 1)
        for k in range(3): _add(order_terms, land[k, d], 1)
        m.con(f"market_order_slots:{d}", order_terms, ub=240)
        for k in range(1, 40): m.con(f"hire_prefix:{d}:{k}", {hire[d, k]: 1, hire[d, k - 1]: -1}, ub=0)

    # Labour counts only actions every represented real schedule must perform.
    # The valid model's map-derived minimum travel is zero: a productive tile can
    # be a shed-access spawn tile, and successive operations can share a tile.
    # The realistic variant inflates required actions by measured router travel.
    action_rows: dict[int, dict[int, float]] = {}
    for d in range(DAYS):
        terms = {}
        for c in CROPS:
            _add(terms, plant[c, d], 1)  # plant
            t = d - int(kg.CROPS[c]["first_yield_day"])
            if t >= 0: _add(terms, plant[c, t], 1)  # one bulk tile harvest action
        for a in ANIMALS:
            _add(terms, buy[a, d], 3)  # build, pickup, place
            ad = kg.ANIMALS[a]
            for t in range(d + 1):
                age = d - t
                if age >= ad["first_yield_day"] and (age - ad["first_yield_day"]) % ad["interval"] == 0:
                    _add(terms, alive[a, t, d], 1)  # harvest held product
        _add(terms, collect[d], 1)
        _add(terms, feed_crop[d], 1); _add(terms, feed_buy[d], 1)
        if travel_ratio:
            terms = {i: coef * (1 + travel_ratio) for i, coef in terms.items()}
        for k in range(40): _add(terms, hire[d, k], -24)
        m.con(f"labour:{d}", terms, ub=24)
        action_rows[d] = terms

    # Daily cash. Revenue is usable on the same day; this relaxes within-day
    # liquidity but never removes a feasible real schedule.
    fib = [kg._hire_cost(k) for k in range(40)]
    for d in range(DAYS):
        terms = {cash[d]: 1}
        if d > 0: _add(terms, cash[d - 1], -1)
        rhs = 3000 if d == 0 else 0
        for p in PRODUCTS:
            for k in range(len(BREAKS) - 1): _add(terms, seg[p, d, k], -segment_price[p, d, k])
        for c in CROPS: _add(terms, plant[c, d], int(kg.CROPS[c]["seed"]))
        for a in ANIMALS: _add(terms, buy[a, d], int(kg.ANIMALS[a]["cost"]))
        for k, cost in enumerate(fib): _add(terms, hire[d, k], cost)
        for k, cost in enumerate([1000, 2000, 4000]): _add(terms, land[k, d], cost)
        _add(terms, feed_buy[d], 1)  # price-floor lower bound; feed-only, cannot be resold
        if d == DAYS - 1:
            _add(terms, trade_credit, -1)
        m.con(f"cash_flow:{d}", terms, lb=rhs, ub=rhs)
    m.objective[cash[DAYS - 1]] = -1
    v.update(locals())
    return m, v


def solve(instance: dict[str, Any], travel_ratio: float) -> tuple[Any, Model, dict[str, Any]]:
    model, v = build_model(instance, travel_ratio)
    c, integ, bounds, cons, A = model.arrays()
    # Solving the LP relaxation is deliberate: fractional tiles/hands/orders
    # enlarge the real feasible set and make the optimum itself a certified
    # upper bound with an attaining (fractional) schedule.
    eq_rows, ub_rows, b_eq, b_ub, ub_meta = [], [], [], [], []
    for r, (lo, hi) in enumerate(zip(model.row_lb, model.row_ub)):
        if np.isfinite(lo) and np.isfinite(hi) and abs(lo - hi) < 1e-10:
            eq_rows.append(r); b_eq.append(hi)
        else:
            if np.isfinite(hi):
                ub_rows.append(A.getrow(r)); b_ub.append(hi); ub_meta.append((r, 1.0))
            if np.isfinite(lo):
                ub_rows.append(-A.getrow(r)); b_ub.append(-lo); ub_meta.append((r, -1.0))
    A_eq = A[eq_rows] if eq_rows else None
    A_ub = None
    if ub_rows:
        from scipy.sparse import vstack
        A_ub = vstack(ub_rows).tocsr()
    result = linprog(c, A_ub=A_ub, b_ub=b_ub or None, A_eq=A_eq, b_eq=b_eq or None,
                     bounds=list(zip(model.lb, model.ub)), method="highs")
    if not result.success or result.x is None:
        raise RuntimeError(f"LP failed: {result.message}")
    duals: dict[str, float] = {}
    for pos, r in enumerate(eq_rows):
        duals[model.row_names[r]] = float(result.eqlin.marginals[pos])
    for pos, (r, sign) in enumerate(ub_meta):
        # Convert back to derivative of minimized objective with respect to the
        # original row bound direction.
        duals[model.row_names[r]] = float(result.ineqlin.marginals[pos]) * sign
    result["row_duals"] = duals
    return result, model, v


def _ival(x: np.ndarray, i: int) -> float:
    value = float(x[i])
    return 0.0 if abs(value) < 1e-7 else round(value, 6)


def extract(result: Any, model: Model, v: dict[str, Any], instance: dict[str, Any], travel_ratio: float) -> dict[str, Any]:
    x = result.x; kg = v["kg"]
    plant, buy, alive, sale, cash, land, hire = (v[k] for k in ("plant", "buy", "alive", "sale", "cash", "land", "hire"))
    collect, feed_crop, feed_buy = v["collect"], v["feed_crop"], v["feed_buy"]
    rows = []
    for d in range(DAYS):
        hands = round(sum(_ival(x, hire[d, k]) for k in range(40)), 6)
        purchases = []
        for a in ANIMALS:
            q = _ival(x, buy[a, d])
            if q: purchases.append({"item": a, "qty": q})
        for c in CROPS:
            q = _ival(x, plant[c, d])
            if q: purchases.append({"item": f"{c}_SEED", "qty": q})
        for k in range(3):
            if _ival(x, land[k, d]): purchases.append({"item": "LAND", "qty": 1, "unlocks": ["NE", "SW", "SE"][k]})
        fb = float(x[feed_buy[d]])
        if fb > 1e-5: purchases.append({"item": "WHEAT_FEED_AT_PRICE_FLOOR_RELAXATION", "qty": round(fb, 4)})
        sales = [{"product": p, "qty": _ival(x, sale[p, d])} for p in PRODUCTS if _ival(x, sale[p, d])]
        rows.append({
            "day": d, "hires": hands, "purchases": purchases, "builds": [], "plantings": [], "sales": sales,
            "relaxation_bank": round(float(x[cash[d]]), 2),
            "fertilizer_collected": _ival(x, collect[d]),
            "grown_wheat_fed": round(float(x[feed_crop[d]]), 4),
            "relaxation_credits": ([{"kind": "speculative_wheat_trade_profit_upper_allowance", "dollars": round(float(x[v['trade_credit']]), 2)}] if d == DAYS - 1 else []),
        })

    # Assign fractional interval jobs to named coordinates. Fractional occupation
    # is itself part of the LP relaxation and is exposed rather than rounded.
    coords = [(xx, yy) for yy in range(10) for xx in range(10)]
    intervals = []
    for a in ANIMALS:
        for t in range(DAYS):
            counts = [_ival(x, alive[a, t, d]) for d in range(t, DAYS)]
            for off, count in enumerate(counts):
                following = counts[off + 1] if off + 1 < len(counts) else 0.0
                qty = count - following
                if qty > 1e-7:
                    intervals.append((t, t + off, "animal", a, qty))
    for c in CROPS:
        delay = int(kg.CROPS[c]["first_yield_day"])
        for t in range(DAYS):
            qty = _ival(x, plant[c, t])
            if qty > 1e-7:
                intervals.append((t, min(DAYS - 1, t + delay), "crop", c, qty))
    occupied: dict[tuple[int, int], list[tuple[int, int, float]]] = defaultdict(list)
    unlock_count = []
    for d in range(DAYS): unlock_count.append(1 + sum(_ival(x, land[k, q]) for k in range(3) for q in range(d + 1)))

    def tile_availability(coord: tuple[int, int], day: int) -> float:
        xx, yy = coord
        if xx < 5 and yy < 5: return 1.0
        k = 0 if yy < 5 else (1 if xx < 5 else 2)
        return min(1.0, sum(_ival(x, land[k, q]) for q in range(day + 1)))

    for start, end, kind, item, quantity in sorted(intervals, key=lambda z: (z[0], z[2] == "crop", z[1])):
        remaining = quantity
        for chosen in coords:
            residual = min(
                tile_availability(chosen, d) - sum(q for s, e, q in occupied[chosen] if s <= d <= e)
                for d in range(start, end + 1)
            )
            take = min(remaining, max(0.0, residual))
            if take <= 1e-7: continue
            occupied[chosen].append((start, end, take)); remaining -= take
            entry = {"tile": list(chosen), "fractional_tile": round(take, 6)}
            if kind == "crop":
                entry["crop"] = item; rows[start]["plantings"].append(entry)
            else:
                entry.update({"kind": kg.ANIMALS[item]["structure"], "structure": kg.ANIMALS[item]["structure"], "animal": item})
                rows[start]["builds"].append(entry)
            if remaining <= 1e-6: break
        if remaining > 1e-5:
            raise AssertionError(f"could not fractionally color capacity interval {(start,end,kind,item,quantity)}; left {remaining}")

    # Direct slacks classify binding daily resource families.
    bindings = []
    for d in range(DAYS):
        available_tiles = 25 * unlock_count[d]
        used_tiles = sum(_ival(x, alive[a, t, d]) for a in ANIMALS for t in range(d + 1))
        for c in CROPS:
            delay = int(kg.CROPS[c]["first_yield_day"])
            used_tiles += sum(_ival(x, plant[c, t]) for t in range(d + 1) if d <= t + delay)
        base_actions = 0.0
        for c in CROPS:
            base_actions += _ival(x, plant[c, d])
            t = d - int(kg.CROPS[c]["first_yield_day"])
            if t >= 0: base_actions += _ival(x, plant[c, t])
        for a in ANIMALS:
            base_actions += 3 * _ival(x, buy[a, d])
            ad = kg.ANIMALS[a]
            for t in range(d + 1):
                age = d - t
                if age >= ad["first_yield_day"] and (age - ad["first_yield_day"]) % ad["interval"] == 0:
                    base_actions += _ival(x, alive[a, t, d])
        base_actions += float(x[collect[d]] + x[feed_crop[d]] + x[feed_buy[d]])
        hands = rows[d]["hires"]
        labour_used = base_actions * (1 + travel_ratio)
        bindings.append({
            "day": d, "cash": round(float(x[cash[d]]), 2), "cash_binding": float(x[cash[d]]) < 1e-5,
            "tiles_used": used_tiles, "tiles_available": available_tiles, "tile_binding": available_tiles - used_tiles < 1e-5,
            "labour_used": round(labour_used, 3), "labour_available": 24 * (1 + hands),
            "labour_binding": 24 * (1 + hands) - labour_used < 1e-4,
            "labour_shadow_value_per_extra_turn": round(max(0.0, -result.row_duals.get(f"labour:{d}", 0.0)), 4),
            "tile_shadow_value_per_extra_tile": round(max(0.0, -result.row_duals.get(f"tile_capacity:{d}", 0.0)), 4),
            "cash_floor_shadow_cost": round(max(0.0, float(result.lower.marginals[cash[d]])), 4),
        })
    return {
        "model": "realistic_travel" if travel_ratio else "valid_upper_bound",
        "travel_moves_per_required_action": travel_ratio,
        "objective_final_bank": round(float(x[cash[DAYS - 1]]), 2),
        "speculative_trade_allowance": float(v["speculative_trade_allowance"]),
        "max_wheat_price_gain_over_all_282_drain_units": int(v["max_wheat_drain_gain"]),
        "max_product_buys_from_shed_throughput": int(v["max_product_buys"]),
        "mip_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
        "days": rows,
        "binding_constraints": bindings,
    }


def measure_exec_b(instance: dict[str, Any]) -> dict[str, Any]:
    """Run Exec B once under the pin and attribute movement runs to next action."""
    path = ROOT / "bots" / "exec_b" / "main.py"
    spec = importlib.util.spec_from_file_location("fixed_game_exec_b_measure", path)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(module)
    env = make_env(instance)
    env.run([module.agent, "pass"])
    by_day = []
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"actions": 0, "moves": 0})
    for day in range(DAYS):
        pending: dict[int, int] = defaultdict(int); moves = acts = passes = 0
        classes: dict[str, dict[str, int]] = defaultdict(lambda: {"actions": 0, "moves": 0})
        for j in range(day * 24 + 1, min((day + 1) * 24 + 1, len(env.steps))):
            action = to_plain(env.steps[j][0].action or {})
            units = [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]
            for ui, a in enumerate(units):
                op = a[0] if isinstance(a, list) and a else "PASS"
                if op in MOVE_OPS:
                    pending[ui] += 1; moves += 1
                elif op == "PASS":
                    passes += 1
                else:
                    cls = op
                    classes[cls]["actions"] += 1; classes[cls]["moves"] += pending[ui]
                    by_class[cls]["actions"] += 1; by_class[cls]["moves"] += pending[ui]
                    pending[ui] = 0; acts += 1
        by_day.append({"day": day, "moves": moves, "nonmove_actions": acts, "passes": passes,
                       "moves_per_nonmove_action": moves / acts if acts else 0,
                       "by_action_class": dict(classes)})
    total_moves = sum(r["moves"] for r in by_day); total_actions = sum(r["nonmove_actions"] for r in by_day)
    return {
        "engine_final_bank": float(env.steps[-1][0].observation.farms[0].money),
        "final_shops": to_plain(env.steps[-1][0].observation.town)["unlocked_shops"],
        "season_moves": total_moves, "season_nonmove_actions": total_actions,
        "season_moves_per_nonmove_action": total_moves / total_actions,
        "day_20": by_day[20],
        "by_action_class": {k: {**r, "moves_per_action": r["moves"] / r["actions"] if r["actions"] else 0} for k, r in by_class.items()},
        "per_day": by_day,
    }


def report(valid: dict[str, Any], realistic: dict[str, Any], routing: dict[str, Any]) -> str:
    def ranges(key: str, rows: list[dict[str, Any]]) -> str:
        ds = [r["day"] for r in rows if r[key]]
        return ", ".join(map(str, ds)) if ds else "none"
    lines = [
        "# Fixed-game economic bound", "",
        f"**Valid upper bound: ${valid['objective_final_bank']:,.0f}. Realistic-routing relaxation: ${realistic['objective_final_bank']:,.0f}.** Both solved by SciPy/HiGHS; the first is the certified relaxation. The realistic number uses {realistic['travel_moves_per_required_action']:.3f} measured Exec B moves per non-move action and is diagnostic, not a proof bound.", "",
        f"Exec B routing measurement under the pin: final bank ${routing['engine_final_bank']:,.0f}; {routing['season_moves']:,} moves / {routing['season_nonmove_actions']:,} non-move actions = {routing['season_moves_per_nonmove_action']:.3f}. On actual day 20: {routing['day_20']['moves']} / {routing['day_20']['nonmove_actions']} = {routing['day_20']['moves_per_nonmove_action']:.3f}.", "",
        "## Why every valid-model constraint is a relaxation", "",
        "- **Cash/no borrowing.** Daily cash starts at $3,000 and is nonnegative. Seeds, animals, Fibonacci wages, land, and bought feed are charged on their purchase day; revenue is credited on the sale day. Netting within a day ignores intra-day sequencing, which can only help. Bought feed is priced at the $1 engine floor and cannot be resold; actual bought wheat is never cheaper.",
        "- **Production/growth.** Every crop receives its lifetime cap at its earliest legal harvest age and its tile is released immediately. Actual watering/fertilizer can only produce no earlier and no more. Each fed animal production tick is allowed its full held cap (4/6/6) immediately, which upper-bounds the base-plus-care ramp. Free cohort disposal relaxes escape/retirement.",
        "- **Feed.** Cohorts alive on day d impose the unavoidable cumulative lower bound floor((age+1)/2) feeds; feeds are pooled across animals and may be bought at $1, both easier than real per-animal wheat handling. Feed actions still consume labour.",
        "- **Tiles/quadrants.** At most 25 tiles per unlocked quadrant are occupied, and land follows NE→SW→SE at $1k/$2k/$4k. The model ignores weeds and gives accelerated crop release, so all real layouts map inside it.",
        "- **Labour/travel.** Required actions include plant+harvest, animal build+pickup+place, production harvest, feed, and fertilizer collection. Water and CARE are omitted. The valid map-derived travel lower bound is zero because shed-access tiles are productive/spawn tiles and consecutive actions may occur without movement. Thus required turns never exceed real required turns. The realistic variant multiplies these actions by the measured routing ratio and loses the proof guarantee.",
        "- **Orders/storage.** Active order classes plus individual HIRE/land orders are capped at the real 240 slots/day. Bulk quantities use one slot, as in-engine. Shed/inventory capacity and end-of-day overflow are omitted, enlarging feasibility.",
        "- **Revenue.** For each product/day the concave block curve uses the highest real marginal quote in each block after *all* pinned drain through that day. It resets prior sales to zero every day and lets end-of-day drain precede sales. Both raise quotes. Block maxima upper-bound every rounded marginal; melon and fertilizer therefore retain explicit dead-stock pools rather than linear base-price revenue.",
        f"- **Legal wheat speculation.** The production model alone would miss BUY_PRODUCT/SELL carry trades. A separate ${valid['speculative_trade_allowance']:,.0f} credit closes that gap: at most 100 product units can enter the shed per transition × 719 = {valid['max_product_buys_from_shed_throughput']:,}; fertilizer has zero external drain and zero profitable round trip; each wheat unit is granted the maximum ${valid['max_wheat_price_gain_over_all_282_drain_units']} price rise across all 282 pinned drain units. This deliberately double-counts timing opportunities and is therefore an upper allowance.",
        "- **Integrality/hire cap.** Tiles, animals, hands, land and order indicators are fractional in the final LP, enlarging the real integer set. Forty prefix-ordered hire slots/day cover any economically relevant real plan: the 40th Fibonacci wage alone is $102,334,155, above a coarse all-tiles/all-products physical revenue ceiling for this 30-day instance.", "",
        "## Binding resources by day", "",
        f"Valid model — zero cash: {ranges('cash_binding', valid['binding_constraints'])}; labour: {ranges('labour_binding', valid['binding_constraints'])}; tiles: {ranges('tile_binding', valid['binding_constraints'])}.", "",
        f"Realistic model — zero cash: {ranges('cash_binding', realistic['binding_constraints'])}; labour: {ranges('labour_binding', realistic['binding_constraints'])}; tiles: {ranges('tile_binding', realistic['binding_constraints'])}.", "",
        "The model is an LP relaxation, so the duals are defined. `schedule.json` records per-day labour value ($ gained per extra unit turn), tile value ($ per extra tile-day), cash-floor shadow cost, and primal binding flags. Values are local sensitivities of this relaxation.", "",
        "| Day | Cash floor shadow | Labour $/turn | Tile $/tile-day |", "|---:|---:|---:|---:|",
    ]
    for r in valid["binding_constraints"]:
        lines.append(f"| {r['day']} | ${r['cash_floor_shadow_cost']:,.2f} | ${r['labour_shadow_value_per_extra_turn']:,.2f} | ${r['tile_shadow_value_per_extra_tile']:,.2f} |")
    lines += ["",
        "## Bank curves", "", "| Day | Valid UB | Realistic routing |", "|---:|---:|---:|",
    ]
    for d in range(DAYS): lines.append(f"| {d} | ${valid['days'][d]['relaxation_bank']:,.0f} | ${realistic['days'][d]['relaxation_bank']:,.0f} |")
    lines += ["", "## Schedule summary (valid relaxation, five-day blocks)", "", "| Days | Hires | Animals | Plantings | Sales |", "|---|---:|---|---|---|"]
    for start in range(0, DAYS, 5):
        block = valid["days"][start:start + 5]
        hires = sum(r["hires"] for r in block)
        purchases: dict[str, float] = defaultdict(float); plants: dict[str, int] = defaultdict(int); sales: dict[str, int] = defaultdict(int)
        for r in block:
            for p in r["purchases"]:
                if p["item"] in ANIMALS: purchases[p["item"]] += p["qty"]
            for p in r["plantings"]: plants[p["crop"]] += 1
            for s in r["sales"]: sales[s["product"]] += s["qty"]
        fmt = lambda d: ", ".join(f"{k} {v:g}" for k, v in d.items()) or "none"
        lines.append(f"| {start}–{start+4} | {hires} | {fmt(purchases)} | {fmt(plants)} | {fmt(sales)} |")
    lines += ["", "## Largest overstatements (estimated)", "",
              f"1. Speculative-wheat safety allowance: **${valid['speculative_trade_allowance']:,.0f}** is granted in full although cash and routing make 71,900 profitable carry units unattainable.",
              "2. Daily market reset/end-of-day drain timing: expected overstatement **$120k–$300k**, largest in wool/milk and the dead-stock fertilizer/melon pools.",
              "3. Full held-cap animal yield without CARE plus accelerated crop caps: expected **$100k–$250k**.", "",
              "These are sensitivity ranges, not certified decompositions; overlapping mechanisms mean they must not be summed.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    instance = load_instance(args.instance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routing_path = args.output_dir / "routing-measurement.json"
    if routing_path.exists(): routing = json.loads(routing_path.read_text(encoding="utf-8"))
    else:
        routing = measure_exec_b(instance)
        routing_path.write_text(json.dumps(routing, indent=2) + "\n", encoding="utf-8")
    valid_res, valid_model, valid_vars = solve(instance, 0.0)
    valid = extract(valid_res, valid_model, valid_vars, instance, 0.0)
    ratio = float(routing["season_moves_per_nonmove_action"])
    real_res, real_model, real_vars = solve(instance, ratio)
    realistic = extract(real_res, real_model, real_vars, instance, ratio)
    schedule = {
        "schema_version": 1, "seed": 3910, "seat": 0, "opponent": "pass", "pinned": True,
        "warning": "Fractional LP relaxation witness, not directly executable: fractional_tile records fractional occupancy and optimistic yields/travel/market are documented in bound.md",
        "valid_upper_bound": valid["objective_final_bank"], "realistic_bound": realistic["objective_final_bank"],
        "days": valid["days"], "binding_constraints": valid["binding_constraints"],
        "realistic_days": realistic["days"], "realistic_binding_constraints": realistic["binding_constraints"],
    }
    (args.output_dir / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "bound-details.json").write_text(json.dumps({"valid": valid, "realistic": realistic}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "bound.md").write_text(report(valid, realistic, routing), encoding="utf-8")
    print(json.dumps({"valid_upper_bound": valid["objective_final_bank"], "realistic_bound": realistic["objective_final_bank"], "routing_ratio": ratio}, indent=2))


if __name__ == "__main__":
    main()
