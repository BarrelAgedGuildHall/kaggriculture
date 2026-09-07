"""Solo evolutionary search over v1 plus the hindsight executor extensions.

Run from the repository root with an absolute ``KAGGRI_ARENA_DIR``::

  uv run python scripts/hindsight/solve_ext.py --seed 3910 --workers 4 --max-evals 1500
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from _engine import engine_version, import_engine, to_plain  # noqa: E402
from scripts.hindsight import executor_ext, solve  # noqa: E402

RESULT_ROOT = REPO / "data" / "results" / "hindsight-ext"
SOURCE_BOT = REPO / "bots" / "brain_strawberry" / "main.py"
PRODUCTS = solve.PRODUCTS
BLOCKS = ("near", "middle", "far")


def _normalise(plan: dict[str, Any]) -> dict[str, Any]:
    return executor_ext._normalise_extensions(solve._normalise(plan))


def _key(plan: dict[str, Any]) -> str:
    raw = json.dumps(_normalise(plan), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _extend(plan: dict[str, Any]) -> dict[str, Any]:
    p = copy.deepcopy(plan)
    p.setdefault("hands_by_day", [p["hands_by_5day"][d // 5] for d in range(30)])
    p.setdefault("purchase_order", ["legacy"])
    p.setdefault("tile_blocks", {})
    p.setdefault("harvest_days", {c: False for c in executor_ext.CROPS})
    p.setdefault("fertilizer_mode", "legacy")
    p.setdefault("routing_mode", "legacy")
    return _normalise(p)


def _seed_plans(seed: int) -> list[dict[str, Any]]:
    base = solve._base_plan()
    v0 = json.loads((solve.SOLO_RESULT_ROOT / str(seed) / "best.json").read_text(encoding="utf-8"))["plan"]
    # Exact v1 seeds are intentionally retained as controls and prove that the new
    # search cannot regress merely because extension keys exist.
    seeds = [copy.deepcopy(base), copy.deepcopy(v0), _extend(base), _extend(v0)]
    # Known economically distinct starts make the first generation useful: retained
    # v0 dairy/wool, a melon cash tranche, and scalable goose/egg lines.
    for opening_melon in (0, 8, 12, 16, 25):
        p = _extend(v0)
        p["crop_mix"]["opening"] = {"WHEAT": 25-opening_melon, "CARROT": 0,
                                     "MELON": opening_melon, "STRAWBERRY": 0}
        if opening_melon:
            p["tile_blocks"] = {"MELON": "near", "WHEAT": "middle"}
            p["harvest_days"]["MELON"] = "opening"
        seeds.append(_normalise(p))
    for geese in (6, 10, 14, 18, 20):
        for retained in ("mixed", "eggs"):
            p = _extend(v0)
            p["animals"]["GOOSE"].update(start_day=0, interval=1, batch=2, total=geese)
            if retained == "eggs":
                p["animals"]["COW"].update(start_day=30, batch=0, total=0)
                p["animals"]["SHEEP"].update(start_day=30, batch=0, total=0)
            else:
                p["animals"]["COW"]["total"] = min(8, p["animals"]["COW"]["total"])
                p["animals"]["SHEEP"]["total"] = min(7, p["animals"]["SHEEP"]["total"])
            p["sell_policy"]["EGG"] = "immediate"
            p["hands_by_day"] = [8 if d < 5 else 10 if d < 15 else 12 for d in range(30)]
            seeds.append(_normalise(p))
    for mode in ("sell_all", "no_buy"):
        p = _extend(v0); p["fertilizer_mode"] = mode; seeds.append(_normalise(p))
    p = _extend(v0); p["purchase_order"] = ["animal", "feed", "seed", "hire", "sell", "land", "other"]; seeds.append(_normalise(p))
    p = _extend(v0); p["purchase_order"] = ["sell", "feed", "seed", "animal", "hire", "land", "other"]; seeds.append(_normalise(p))
    p = _extend(v0); p["routing_mode"] = "nearest"; seeds.append(_normalise(p))
    for day in range(30):
        for delta in (-2, -1, 1, 2):
            p = _extend(v0)
            p["hands_by_day"][day] += delta
            seeds.append(_normalise(p))
    return list({_key(p): p for p in seeds}.values())


def _mutate(parent: dict[str, Any], rng: random.Random, heat: float) -> dict[str, Any]:
    p = _extend(parent)
    for _ in range(rng.randint(1, 2 + int(2 * heat))):
        group = rng.choice(("daily_hands", "daily_hands", "daily_hands", "legacy", "legacy",
                            "order", "blocks", "rush", "fert", "goose", "routing"))
        if group == "legacy":
            # The v1 mutator canonicalises while preserving unknown extension keys.
            p = solve._mutate(p, rng, heat)
        elif group == "daily_hands":
            day = rng.randrange(30)
            radius = rng.choice((0, 1, 2))
            delta = rng.choice((-2, -1, 1, 2))
            for d in range(max(0, day-radius), min(30, day+radius+1)):
                p["hands_by_day"][d] += delta
        elif group == "order":
            if p["purchase_order"] == ["legacy"]:
                p["purchase_order"] = rng.choice((
                    ["sell", "feed", "animal", "seed", "hire", "land", "other"],
                    ["animal", "feed", "seed", "hire", "sell", "land", "other"]))
            elif rng.random() < .15:
                p["purchase_order"] = ["legacy"]
            else:
                i, j = rng.sample(range(len(p["purchase_order"])), 2)
                p["purchase_order"][i], p["purchase_order"][j] = p["purchase_order"][j], p["purchase_order"][i]
        elif group == "blocks":
            crop = rng.choice(executor_ext.CROPS)
            if rng.random() < .2:
                p["tile_blocks"].pop(crop, None)
            else:
                p["tile_blocks"][crop] = rng.choice(BLOCKS)
        elif group == "rush":
            crop = rng.choice(executor_ext.CROPS)
            p["harvest_days"][crop] = rng.choice((False, "opening", "all"))
        elif group == "fert":
            p["fertilizer_mode"] = rng.choice(("legacy", "sell_all", "no_buy"))
        elif group == "routing":
            p["routing_mode"] = "nearest" if p.get("routing_mode") != "nearest" else "legacy"
        else:
            a = p["animals"]["GOOSE"]
            if a["total"] == 0:
                a.update(start_day=rng.randint(0, 3), interval=1, batch=rng.choice((1, 2, 3)), total=rng.randint(5, 20))
            else:
                a["total"] += rng.choice((-3, -2, -1, 1, 2, 3))
                a["batch"] += rng.choice((-1, 0, 1))
            p["sell_policy"]["EGG"] = "immediate"
    return _normalise(p)


def _bank_curve(env: Any, seat: int = 0) -> list[float]:
    curve = []
    for day in range(30):
        index = min((day + 1) * 24, len(env.steps) - 1)
        obs = to_plain(env.steps[index][seat].observation)
        curve.append(float(obs["farms"][seat]["money"]))
    return curve


def _episode(seed: int, plan: dict[str, Any] | None, opponent: str = "pass",
             seat: int = 0, detailed: bool = False) -> dict[str, Any]:
    ke = import_engine()
    raw = str(SOURCE_BOT) if plan is None else executor_ext.load_agent(plan)
    timings: list[float] = []
    sales = {p: 0 for p in PRODUCTS}
    revenues = {p: 0.0 for p in PRODUCTS}
    if callable(raw):
        from bots.brain_strawberry import main as economics

        def ours(obs, config=None):
            start = time.perf_counter()
            action = raw(obs, config)
            timings.append(time.perf_counter() - start)
            if detailed:
                inv = dict(obs.get("market", {}).get("inventory", {}) or {})
                for order in action.get("market", []):
                    if len(order) >= 3 and order[0] == "SELL" and order[1] in sales:
                        item, qty = order[1], int(order[2])
                        revenue, end_inv = economics.predict_sale(item, int(inv.get(item, 10000)), qty)
                        sales[item] += qty
                        revenues[item] += revenue
                        inv[item] = end_inv
            return action
    else:
        ours = raw
    agents = [ours, opponent] if seat == 0 else [opponent, ours]
    env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run(agents)
    curve = _bank_curve(env, seat)
    result = {"seed": seed, "seat": seat, "status": str(env.steps[-1][seat].status),
              "opponent_status": str(env.steps[-1][1-seat].status),
              "final_bank": curve[-1], "bank_by_day": curve}
    if detailed:
        ordered = sorted(timings)
        backlog = []
        product_for_animal = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
        for day in range(30):
            index = min((day + 1) * 24 - 1, len(env.steps) - 1)
            state = to_plain(env.steps[index][seat].observation)
            farm, private = state["farms"][seat], state.get("private", {}) or {}
            unharvested = {p: 0 for p in PRODUCTS}
            for row in farm["tiles"]:
                for tile in row:
                    if not isinstance(tile, dict): continue
                    if tile.get("kind") == "PLANT" and tile.get("crop") in unharvested:
                        unharvested[tile["crop"]] += int(tile.get("yield_units", 0) or 0)
                    elif tile.get("animal") in product_for_animal:
                        unharvested[product_for_animal[tile["animal"]]] += int(tile.get("yield_units", 0) or 0)
            unsold = {p: int((private.get("shed", {}) or {}).get(p, 0)) for p in PRODUCTS}
            for inv in private.get("inventories", []) or []:
                for p in PRODUCTS: unsold[p] += int(inv.get(p, 0) or 0)
            backlog.append({"day": day, "hands": len(farm.get("hands", [])),
                            "unharvested": unharvested, "unsold": unsold})
        result.update(sold_units=sales, per_product_revenue=revenues,
                      timing_p99_ms=1000 * ordered[min(len(ordered)-1, int(.99*len(ordered)))] if ordered else 0,
                      timing_max_ms=1000 * max(ordered) if ordered else 0,
                      end_day_backlog=backlog)
    if opponent != "pass":
        other_curve = _bank_curve(env, 1-seat)
        result.update(their_bank=other_curve[-1], margin=curve[-1]-other_curve[-1])
    return result


def _worker(seed: int, plan: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        game = _episode(seed, plan)
        error = None
    except Exception as exc:
        game, error = {"final_bank": -1e12, "status": "ERROR"}, repr(exc)
    return {"plan_id": _key(plan), "params": _normalise(plan), "final_bank": game["final_bank"],
            "status": game.get("status"), "error": error, "eval_seconds": time.perf_counter()-start}


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out


def solve_seed(seed: int, workers: int, population: int, max_evals: int,
               plateau: int, search_seed: int) -> dict[str, Any]:
    root = RESULT_ROOT / str(seed)
    log = root / "log.jsonl"
    rows = _read(log)
    seen = {r["plan_id"] for r in rows}
    rng = random.Random(search_seed + seed)
    candidates = [p for p in _seed_plans(seed) if _key(p) not in seen]
    start_count = len(rows)
    generation = max([int(r.get("generation", -1)) for r in rows] + [-1]) + 1
    best_at = max([int(r.get("evaluation", 0)) for r in rows
                   if r.get("final_bank") == max([x.get("final_bank", -1e12) for x in rows] or [-1e12])] + [0])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(rows) - start_count < max_evals:
            ranked = sorted(rows, key=lambda r: r.get("final_bank", -1e12), reverse=True)
            elites = [r["params"] for r in ranked[:max(4, population // 4)]] or _seed_plans(seed)
            while len(candidates) < population:
                heat = max(.35, 1.4 / math.sqrt(max(1, generation + 1)))
                child = _mutate(rng.choice(elites), rng, heat)
                key = _key(child)
                if key not in seen and all(_key(x) != key for x in candidates):
                    candidates.append(child)
            room = min(population, max_evals - (len(rows)-start_count))
            batch, candidates = candidates[:room], candidates[room:]
            futures = [pool.submit(_worker, seed, p) for p in batch]
            for future in as_completed(futures):
                row = future.result()
                row["generation"] = generation
                row["evaluation"] = len(rows) + 1
                row["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _append(log, row)
                rows.append(row); seen.add(row["plan_id"])
                if row["final_bank"] >= max(x["final_bank"] for x in rows): best_at = len(rows)
                print(f"seed={seed} gen={generation} eval={len(rows)} bank={row['final_bank']:.0f} best={max(x['final_bank'] for x in rows):.0f}", flush=True)
            generation += 1
            candidates = []
            if len(rows)-best_at >= plateau and len(rows)-start_count >= plateau:
                break
    best = max(rows, key=lambda r: r.get("final_bank", -1e12))
    measured = _episode(seed, best["params"], detailed=True)
    baseline = _episode(seed, None, detailed=True)
    v0_artifact = json.loads((solve.SOLO_RESULT_ROOT / str(seed) / "best.json").read_text(encoding="utf-8"))
    v0 = _episode(seed, v0_artifact["plan"], detailed=True)
    payload = {"schema_version": 2, "engine": engine_version(), "seed": seed, "seat": 0,
               "opponent": "pass", "plan": best["params"], "plan_id": best["plan_id"],
               "evaluations": len(rows), "new_evaluations": len(rows)-start_count,
               "plateau": {"evaluations_since_best": len(rows)-best_at,
                           "threshold": plateau, "first_best_evaluation": best_at},
               "baseline": baseline, "v0_best": v0, **measured}
    root.mkdir(parents=True, exist_ok=True)
    (root / "best.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def verify_contested() -> dict[str, Any]:
    plans = {seed: json.loads((RESULT_ROOT / str(seed) / "best.json").read_text(encoding="utf-8"))["plan"]
             for seed in (3910, 3911)}
    cross = {}
    for source_seed, plan in plans.items():
        cross[str(source_seed)] = [_episode(seed, plan, detailed=True) for seed in (3910, 3911)]
    chosen_seed = max(plans, key=lambda s: statistics.fmean(g["final_bank"] for g in cross[str(s)]))
    opponent = str(REPO / "bots" / "v46" / "main.py")
    games, baseline = [], []
    for seed in range(3910, 3914):
        games.append(_episode(seed, plans[chosen_seed], opponent=opponent, seat=0, detailed=True))
        baseline.append(_episode(seed, None, opponent=opponent, seat=0, detailed=False))
    payload = {"engine": engine_version(), "chosen_plan_source_seed": chosen_seed,
               "cross_seed_solo": cross, "games": games, "baseline": baseline}
    path = RESULT_ROOT / "contested.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--max-evals", type=int, default=1500)
    parser.add_argument("--plateau", type=int, default=400)
    parser.add_argument("--search-seed", type=int, default=20260907)
    parser.add_argument("--verify-contested", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4: parser.error("--workers must be in 1..4")
    arena = os.environ.get("KAGGRI_ARENA_DIR", "")
    if not arena or not Path(arena).is_absolute(): parser.error("KAGGRI_ARENA_DIR must be absolute")
    if args.verify_contested:
        result = verify_contested()
        print(json.dumps({"chosen_plan_source_seed": result["chosen_plan_source_seed"],
                          "games": result["games"], "baseline": result["baseline"]}, indent=2))
        return 0
    if args.seed is None: parser.error("--seed is required unless --verify-contested is used")
    result = solve_seed(args.seed, args.workers, args.population, args.max_evals,
                        args.plateau, args.search_seed)
    print(json.dumps({"seed": args.seed, "bank": result["final_bank"],
                      "evaluations": result["evaluations"], "plateau": result["plateau"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
