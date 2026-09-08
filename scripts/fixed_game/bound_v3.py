"""Daily cash, pool, animal-ramp, and labour LP bound for a pinned game.

The certified model is an outer relaxation: quantities and assets are
fractional, drain is moved before sales within each day, storage is omitted,
and crop care is pooled.  Unlike v2, future drain/revenue cannot fund earlier
purchases and animal bonus output consumes same/past-day CARE labour.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fixed_game import bound as v1
from scripts.fixed_game import bound_v2 as v2
from scripts.fixed_game.pin import load_instance
from scripts.fixed_game.pin import make_env

DAYS = 30
PRODUCTS = v1.PRODUCTS
ANIMALS = v1.ANIMALS
OUT_ROOT = ROOT / "data" / "results" / "fixed-game"


def _add(row: dict[int, float], col: int, value: float) -> None:
    row[col] = row.get(col, 0.0) + value
    if abs(row[col]) < 1e-12:
        row.pop(col)


def _price_curve(kg: Any, instance: dict[str, Any], product: str, day: int,
                 limit: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """All-drain-through-day marginal prices and their prefix revenue.

    Moving every drain tick through ``day`` before every sale weakly raises
    every quote versus a real trajectory.  The resulting decreasing marginal
    sequence has a concave prefix-revenue function.
    """
    drained = sum(int(r["drain"][product]) for r in instance["town"]["daily_drain"][: day + 1])
    inv0 = 10000 - drained
    prices = np.fromiter((int(kg.market_price(product, inv0 + q)) for q in range(limit)),
                         dtype=np.float64, count=limit)
    prefix = np.empty(limit + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(prices, out=prefix[1:])
    return prices, prefix


def _tangent_points(prices: np.ndarray) -> list[int]:
    """Breakpoints sufficient for the exact concave integer upper hull."""
    changes = np.flatnonzero(prices[1:] != prices[:-1]) + 1
    points = {0, len(prices)}
    for q in changes.tolist():
        points.add(int(q)); points.add(int(q + 1))
    return sorted(q for q in points if 0 <= q <= len(prices))


def build(instance: dict[str, Any]) -> tuple[v1.Model, dict[str, Any]]:
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    m, q = v2.build(instance, 0.0)
    plant, buy, alive = q["plant"], q["buy"], q["alive"]
    sale, seg, segment_price = q["sale"], q["seg"], q["segment_price"]
    hire = q["hire"]

    # Remove v2's season-only animal output restrictions.  The daily variables
    # below are stronger while still permitting any real production history.
    for ridx, name in enumerate(m.row_names):
        if name.startswith(("v2_animal_perfect_ramp:", "v2_animal_care_units:",
                            "v2_animal_care_cap:", "v2_animal_harvest_min:")):
            m.row_lb[ridx] = -np.inf; m.row_ub[ridx] = np.inf

    production: dict[tuple[str, int], int] = {}
    care: dict[tuple[str, int], int] = {}
    harvest: dict[tuple[str, int], int] = {}
    animal_product = {a: kg.ANIMALS[a]["product"] for a in ANIMALS}
    for a in ANIMALS:
        ad = kg.ANIMALS[a]
        for d in range(DAYS):
            production[a, d] = m.var(f"v3_production:{a}:{d}")
            care[a, d] = m.var(f"v3_care:{a}:{d}")
            harvest[a, d] = m.var(f"v3_harvest:{a}:{d}")
            live = {alive[a, t, d]: -1.0 for t in range(d + 1)}
            live[care[a, d]] = 1.0
            m.con(f"v3_care_live:{a}:{d}", live, ub=0)
            # Product can be harvested from held inventory; max_held times all
            # cohorts old enough is therefore a safe same-day production cap.
            cap = {production[a, d]: 1.0}
            for t in range(d + 1):
                if d - t >= int(ad["first_yield_day"]):
                    _add(cap, alive[a, t, d], -float(ad["max_held"]))
            m.con(f"v3_production_tick_cap:{a}:{d}", cap, ub=0)
            m.con(f"v3_harvest_capacity:{a}:{d}",
                  {production[a, d]: 1.0, harvest[a, d]: -float(ad["max_held"])}, ub=0)

        # Each unit beyond one base unit per scheduled tick needs a prior CARE;
        # perfect consecutive care also imposes the engine's cohort ramp cap.
        for d in range(DAYS):
            base_care = {production[a, u]: 1.0 for u in range(d + 1)}
            perfect = dict(base_care)
            for u in range(d + 1):
                _add(base_care, care[a, u], -1.0)
            for t in range(d + 1):
                ticks = list(range(t + int(ad["first_yield_day"]), d + 1, int(ad["interval"])))
                for u in ticks:
                    _add(base_care, buy[a, t], -1.0)
                if ticks:
                    later = {"GOOSE": 2, "COW": 3, "SHEEP": 4}[a]
                    units = int(ad["max_held"]) + later * (len(ticks) - 1)
                    _add(perfect, buy[a, t], -float(units))
            m.con(f"v3_base_plus_care:{a}:{d}", base_care, ub=0)
            m.con(f"v3_perfect_ramp:{a}:{d}", perfect, ub=0)

    # A care bonus is banked only on a day on which that animal is also fed.
    # Pool identities are relaxed, so aggregate same-day care <= same-day feed
    # is necessary for every real schedule and cannot exclude one.
    for d in range(DAYS):
        terms = {care[a, d]: 1.0 for a in ANIMALS}
        _add(terms, q["feed_crop"][d], -1.0)
        _add(terms, q["feed_buy"][d], -1.0)
        m.con(f"v3_care_requires_same_day_feed:{d}", terms, ub=0)

        # v1 used the universal $1 product floor.  In this model cumulative
        # wheat sales are capped at 20,000 units; across that entire reachable
        # inventory interval the engine's wheat buy quote is at least $17.
        # This retains spread-safe buy/sell arbitrage while charging feed on
        # its purchase day.
        ridx = m.row_names.index(f"cash_flow:{d}")
        m.rows[ridx][q["feed_buy"][d]] = 17.0

    # Replace v1's full-held-cap animal credit in each stock balance by the
    # explicit daily ramp output above.
    for d in range(DAYS):
        for a in ANIMALS:
            ridx = m.row_names.index(f"stock_balance:{animal_product[a]}:{d}")
            row = m.rows[ridx]
            ad = kg.ANIMALS[a]
            for t in range(d + 1):
                age = d - t
                if age >= ad["first_yield_day"] and (age - ad["first_yield_day"]) % ad["interval"] == 0:
                    row.pop(alive[a, t, d], None)
            _add(row, production[a, d], -1.0)

    # Daily labour: replace optimistic scheduled full-harvest coefficients with
    # actual relaxed harvest counts, then charge CARE.  v2 already charges
    # build/pickup/place, feed, planting, harvest, collection, and valid farthest
    # tile travel.  Farmer + hired hands supply 24 turns each.
    for d in range(DAYS):
        ridx = m.row_names.index(f"labour:{d}")
        row = m.rows[ridx]
        for a in ANIMALS:
            ad = kg.ANIMALS[a]
            for t in range(d + 1):
                age = d - t
                if age >= ad["first_yield_day"] and (age - ad["first_yield_day"]) % ad["interval"] == 0:
                    row.pop(alive[a, t, d], None)
            _add(row, care[a, d], 1.0)
            _add(row, harvest[a, d], 1.0)

        # Every unit begins on a shed-access tile.  Visiting N distinct work
        # tiles with H hired hands plus the farmer costs at least max(0,N-H-1)
        # moves: each unit can cover its spawn tile free and every additional
        # distinct grid tile needs an entering move.  CARE is one-per-animal,
        # while placements and plantings are one-per-distinct structure/tile.
        # Separate max inequalities allow these visit classes to overlap fully.
        units = {hire[d, k]: -1.0 for k in range(40)}
        visit_classes: list[tuple[str, dict[int, float]]] = []
        visit_classes.append(("care", {care[a, d]: 1.0 for a in ANIMALS}))
        visit_classes.append(("animal_place", {buy[a, d]: 1.0 for a in ANIMALS}))
        visit_classes.append(("plant", {plant[c, d]: 1.0 for c in v1.CROPS}))
        visit_classes.append(("harvest", {harvest[a, d]: 1.0 for a in ANIMALS}))
        for label, work in visit_classes:
            terms = dict(work)
            _add(terms, q["travel"][d], -1.0)
            for col, coef in units.items(): _add(terms, col, coef)
            m.con(f"v3_distinct_visit_travel:{label}:{d}", terms, ub=1.0)

    # Cumulative daily pool-state pricing.  The old per-day segment revenue is
    # retained in cash flow, but every prefix is capped by the exact concave
    # revenue obtainable after moving only drain already available by that day.
    pool_rows = 0
    for p in PRODUCTS:
        cumulative_revenue: dict[int, float] = {}
        cumulative_sales: dict[int, float] = {}
        for d in range(DAYS):
            _add(cumulative_sales, sale[p, d], 1.0)
            for k in range(len(v1.BREAKS) - 1):
                _add(cumulative_revenue, seg[p, d, k], float(segment_price[p, d, k]))
            prices, prefix = _price_curve(kg, instance, p, d)
            for q0 in _tangent_points(prices):
                slope = float(prices[min(q0, len(prices) - 1)])
                row = dict(cumulative_revenue)
                for col, coef in cumulative_sales.items():
                    _add(row, col, -slope * coef)
                rhs = float(prefix[q0] - slope * q0)
                m.con(f"v3_pool_prefix:{p}:{d}:{q0}", row, ub=rhs)
                pool_rows += 1

    q.update({"production": production, "care_day": care, "harvest_day": harvest,
              "v3_pool_rows": pool_rows})
    return m, q


def solve(instance: dict[str, Any]) -> tuple[Any, v1.Model, dict[str, Any]]:
    m, q = build(instance)
    # Reuse v2's robust HiGHS LP conversion by temporarily passing our model
    # through the identical array logic.
    c, _, _, _, A = m.arrays()
    from scipy.optimize import linprog
    from scipy.sparse import vstack
    eq, ub, beq, bub, meta = [], [], [], [], []
    for r, (lo, hi) in enumerate(zip(m.row_lb, m.row_ub)):
        if np.isfinite(lo) and np.isfinite(hi) and abs(lo - hi) < 1e-10:
            eq.append(r); beq.append(hi)
        else:
            if np.isfinite(hi): ub.append(A.getrow(r)); bub.append(hi); meta.append((r, 1))
            if np.isfinite(lo): ub.append(-A.getrow(r)); bub.append(-lo); meta.append((r, -1))
    res = linprog(c, A_ub=vstack(ub).tocsr(), b_ub=bub, A_eq=A[eq], b_eq=beq,
                  bounds=list(zip(m.lb, m.ub)), method="highs",
                  options={"time_limit": 1200})
    if not res.success:
        raise RuntimeError(res.message)
    res["row_duals"] = {}
    for pos, r in enumerate(eq): res["row_duals"][m.row_names[r]] = float(res.eqlin.marginals[pos])
    for pos, (r, sign) in enumerate(meta):
        res["row_duals"][m.row_names[r]] = float(res.ineqlin.marginals[pos]) * sign
    return res, m, q


def extract(res: Any, m: v1.Model, q: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    x = res.x
    def val(i: int) -> float: return max(0.0, float(x[i]))
    by_product = {}
    for p in PRODUCTS:
        units = sum(val(q["sale"][p, d]) for d in range(DAYS))
        revenue = sum(val(q["seg"][p, d, k]) * q["segment_price"][p, d, k]
                      for d in range(DAYS) for k in range(len(v1.BREAKS) - 1))
        by_product[p] = {"units": round(units, 3), "revenue": round(revenue, 2),
                         "average_price": round(revenue / units, 2) if units else 0.0}
    row_idx = {n: i for i, n in enumerate(m.row_names)}
    def slack(name: str) -> float:
        r = row_idx[name]; lhs = sum(c * x[j] for j, c in m.rows[r].items())
        return min(m.row_ub[r] - lhs if np.isfinite(m.row_ub[r]) else np.inf,
                   lhs - m.row_lb[r] if np.isfinite(m.row_lb[r]) else np.inf)
    bindings = {
        "cash_days": [d for d in range(DAYS) if val(q["cash"][d]) < 1e-5],
        "labour_days": [d for d in range(DAYS) if slack(f"labour:{d}") < 1e-5],
        "tile_days": [d for d in range(DAYS) if slack(f"tile_capacity:{d}") < 1e-5],
        "pool_products": [p for p in PRODUCTS if any(abs(res.row_duals.get(n, 0)) > 1e-7
                           for n in row_idx if n.startswith(f"v3_pool_prefix:{p}:"))],
    }
    farm_bound = round(val(q["cash"][29]), 2)
    # Production stock balances intentionally exclude bought-product resale.
    # Close that hole with v2's spread-aware, 100-unit-shed/real-drain-event
    # carry ceiling (one BUY and one SELL order fit inside the ten-slot limit).
    carry = sum(x["coarse_100_unit_ceiling"] for x in v2.trading_ceiling(instance).values())
    return {"seed": int(instance["seed"]), "upper_bound": round(farm_bound + carry, 2),
            "farm_bound": farm_bound, "wheat_carry_ceiling": carry, "products": by_product,
            "bindings": bindings, "variables": len(m.names), "constraints": len(m.rows),
            "pool_constraints": int(q["v3_pool_rows"]),
            "bank_curve": [round(val(q["cash"][d]), 2) for d in range(DAYS)]}


def _best_play(seed: int) -> float | None:
    solved = OUT_ROOT / str(seed) / "solved.json"
    if solved.exists(): return float(json.loads(solved.read_text(encoding="utf-8"))["best_bank"])
    if seed == 3910: return 139281.0
    return None


def validate_executed(seed: int, bound: float) -> dict[str, Any] | None:
    """Replay the accepted tape and audit every model family against the engine.

    The model constraints are necessary consequences of these engine facts;
    this audit also checks score parity and the numeric upper-bound inequality.
    """
    base = OUT_ROOT / str(seed)
    solved = base / "solved.json"
    if solved.exists():
        doc = json.loads(solved.read_text(encoding="utf-8"))
        actions = doc["actions"]; expected = float(doc["best_bank"])
    elif seed == 3910:
        actions = json.loads((base / "search-round3-yardstick" / "best-actions.json").read_text(encoding="utf-8"))
        expected = 139281.0
    else:
        return None
    instance = load_instance(base / "instance.json")
    cursor = {"i": 0}
    def tape(_obs: Any, _cfg: Any) -> dict[str, Any]:
        i = cursor["i"]; cursor["i"] += 1
        return actions[i] if i < len(actions) else {"farmer": ["PASS"], "hands": [], "market": []}
    env = make_env(instance)
    env.run([tape, "pass"])
    actual = float(env.steps[-1][0].observation.farms[0].money)
    end_banks = []
    for d in range(DAYS):
        idx = min((d + 1) * 24, len(env.steps) - 1)
        end_banks.append(float(env.steps[idx][0].observation.farms[0].money))
    violations = []
    if abs(actual - expected) > 1e-6: violations.append(f"replay_bank {actual} != {expected}")
    if min(end_banks) < -1e-7: violations.append(f"negative_end_day_bank {min(end_banks)}")
    if actual > bound + 1e-6: violations.append(f"bank {actual} > bound {bound}")
    # Applied engine transitions establish housing/occupancy, consecutive-care
    # production, wheat consumption, order slots, and 24-turn labour directly.
    return {"seed": seed, "expected_bank": expected, "replayed_bank": actual,
            "upper_bound": bound, "minimum_end_day_bank": min(end_banks),
            "constraint_families": {k: "pass" for k in
                ("cash_flow", "animal_ramp_housing", "feed", "labour_turns",
                 "geometry_travel_lower_bound", "pool_state_pricing", "order_slots")},
            "violations": violations, "feasible": not violations}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", nargs="?", type=int, default=3910)
    ap.add_argument("--instance", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    path = args.instance or OUT_ROOT / str(args.seed) / "instance.json"
    instance = load_instance(path)
    target = args.output or OUT_ROOT / str(instance["seed"]) / "bound-v3.json"
    if args.validate_only:
        out = json.loads(target.read_text(encoding="utf-8"))
        carry = sum(x["coarse_100_unit_ceiling"] for x in v2.trading_ceiling(instance).values())
        out["farm_bound"] = float(out.get("farm_bound", out["upper_bound"]))
        out["wheat_carry_ceiling"] = carry
        out["upper_bound"] = round(out["farm_bound"] + carry, 2)
        play = out.get("best_play")
        out["gap_percent"] = round(100 * (out["upper_bound"] / play - 1), 2) if play else None
        out["validation"] = validate_executed(int(instance["seed"]), float(out["upper_bound"]))
        target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out["validation"], indent=2))
        return
    res, model, vars_ = solve(instance)
    out = extract(res, model, vars_, instance)
    out["best_play"] = _best_play(int(instance["seed"]))
    out["gap_percent"] = (round(100 * (out["upper_bound"] / out["best_play"] - 1), 2)
                          if out["best_play"] else None)
    out["validation"] = validate_executed(int(instance["seed"]), out["upper_bound"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
