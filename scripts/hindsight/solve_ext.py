"""Multi-seed solo search for the drain-matched extended executor.

Every candidate races on three fixed seeds. The strongest third then runs the
other nine seeds and is ranked by 12-seed mean bank (worst seed is retained).
Run only with an absolute arena directory::

  uv run python scripts/hindsight/solve_ext.py --workers 4 --minutes 90
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

RESULT_ROOT = REPO / "data" / "results" / "hindsight-ext2"
SOURCE_BOT = REPO / "bots" / "brain_strawberry" / "main.py"
EXEC_B = REPO / "bots" / "exec_b" / "main.py"
V46 = REPO / "bots" / "v46" / "main.py"
SOLO_SEEDS = tuple(range(5000, 5012))
RACE_SEEDS = (5000, 5005, 5010)
CONTEST_SEEDS = (3910, 3911, 3912, 3913, 5000, 5001, 5002, 5003)
PRODUCTS = solve.PRODUCTS


def _normalise(plan: dict[str, Any]) -> dict[str, Any]:
    return executor_ext._normalise_extensions(solve._normalise(plan))


def _key(plan: dict[str, Any]) -> str:
    raw = json.dumps(_normalise(plan), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _origin() -> dict[str, Any]:
    artifact = RESULT_ROOT.parent / "hindsight-ext" / "3910" / "best.json"
    p = copy.deepcopy(json.loads(artifact.read_text(encoding="utf-8"))["plan"])
    p["supply_ratio"] = {"MILK": .8, "WOOL": .8, "EGG": 2.0,
                         "STRAWBERRY": .55, "MELON": 1.0, "CARROT": .15, "TOMATO": 0.0}
    p["wheat_tiles"] = 28
    p["purchase_order"] = ["legacy"]
    for animal in solve.ANIMALS:
        p["animals"][animal].update(start_day=0, interval=1, batch=4, total=20)
    p["sell_policy"].update({product: "immediate" for product in PRODUCTS})
    return _normalise(p)


def _seed_plans() -> list[dict[str, Any]]:
    origin = _origin()
    plans = [origin]
    animal_sets = ((0, 0, 0), (.5, .5, 1), (.8, .8, 2), (1, 1, 4),
                   (.5, .5, 6), (0, 0, 8), (1.2, 1.2, 0),
                   (3, 3, 0), (4, 4, 0), (4, 4, 2), (3, 3, 4), (2, 2, 8))
    crop_sets = ((.4, 1, 0, 0), (.7, 1, .15, 0), (1, 1, .3, 0),
                 (.8, .5, .5, .15), (1.2, 0, 0, 0))
    for milk, wool, egg in animal_sets:
        for strawberry, melon, carrot, tomato in crop_sets:
            p = copy.deepcopy(origin)
            p["supply_ratio"].update(MILK=milk, WOOL=wool, EGG=egg,
                                      STRAWBERRY=strawberry, MELON=melon,
                                      CARROT=carrot, TOMATO=tomato)
            p["wheat_tiles"] = (32 if max(milk, wool) >= 3 else 20) + (0 if not egg else 8 if egg <= 2 else 16)
            plans.append(_normalise(p))
            if (milk, wool, egg) in ((3, 3, 0), (4, 4, 0), (3, 3, 4)):
                for wheat_tiles in (48, 64):
                    q = copy.deepcopy(p)
                    q["wheat_tiles"] = wheat_tiles
                    plans.append(_normalise(q))
    return list({_key(p): p for p in plans}.values())


def _mutate(parent: dict[str, Any], rng: random.Random, heat: float) -> dict[str, Any]:
    p = copy.deepcopy(parent)
    for _ in range(rng.randint(1, 2 + int(heat))):
        group = rng.choices(("ratio", "ratio", "wheat", "hands", "legacy", "timing"),
                            weights=(4, 4, 2, 2, 1, 1))[0]
        if group == "ratio":
            product = rng.choice(executor_ext.RATIO_PRODUCTS)
            step = rng.choice((.1, .2, .35, .5, 1.0)) * max(.6, heat)
            p["supply_ratio"][product] += rng.choice((-step, step))
        elif group == "wheat":
            p["wheat_tiles"] += rng.choice((-8, -5, -3, 3, 5, 8))
        elif group == "hands":
            day = rng.randrange(30)
            for d in range(max(0, day - 1), min(30, day + 2)):
                p["hands_by_day"][d] += rng.choice((-1, 1))
        elif group == "timing":
            animal = rng.choice(solve.ANIMALS)
            p["animals"][animal]["start_day"] += rng.choice((-1, 1))
            p["animals"][animal]["batch"] += rng.choice((-1, 1))
        else:
            p = solve._mutate(p, rng, min(1.0, heat))
    return _normalise(p)


def _bank_curve(env: Any, seat: int) -> list[float]:
    curve = []
    for day in range(30):
        index = min((day + 1) * 24, len(env.steps) - 1)
        obs = to_plain(env.steps[index][seat].observation)
        curve.append(float(obs["farms"][seat]["money"]))
    return curve


def _episode(seed: int, plan: dict[str, Any] | None, opponent: str | Path = "pass",
             seat: int = 0, detailed: bool = False, agent_path: Path | None = None) -> dict[str, Any]:
    ke = import_engine()
    raw: Any = str(agent_path) if agent_path is not None else (str(SOURCE_BOT) if plan is None else executor_ext.load_agent(plan))
    timings: list[float] = []
    sales = {p: 0 for p in PRODUCTS}
    revenues = {p: 0.0 for p in PRODUCTS}
    if callable(raw):
        from bots.brain_strawberry import main as economics

        def ours(obs, config=None):
            started = time.perf_counter()
            action = raw(obs, config)
            timings.append(time.perf_counter() - started)
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
    other = str(opponent) if isinstance(opponent, Path) else opponent
    agents = [ours, other] if seat == 0 else [other, ours]
    env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run(agents)
    curve = _bank_curve(env, seat)
    result = {"seed": seed, "seat": seat, "status": str(env.steps[-1][seat].status),
              "opponent_status": str(env.steps[-1][1-seat].status), "final_bank": curve[-1],
              "bank_by_day": curve}
    if detailed:
        ordered = sorted(timings)
        result.update(sold_units=sales, per_product_revenue=revenues,
                      timing_p99_ms=(1000 * ordered[min(len(ordered)-1, math.ceil(.99*len(ordered))-1)] if ordered else 0),
                      timing_max_ms=(1000 * max(ordered) if ordered else 0),
                      shops=list((to_plain(env.steps[-1][seat].observation).get("town", {}) or {}).get("unlocked_shops", [])))
    if other != "pass":
        their = _bank_curve(env, 1-seat)[-1]
        result.update(their_bank=their, margin=curve[-1] - their)
    return result


def _run_seeds(plan: dict[str, Any], seeds: tuple[int, ...]) -> dict[str, Any]:
    games = [_episode(seed, plan) for seed in seeds]
    banks = [game["final_bank"] for game in games]
    return {"games": games, "banks": banks, "mean_bank": statistics.fmean(banks),
            "worst_bank": min(banks)}


def _worker(plan: dict[str, Any], seeds: tuple[int, ...]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = _run_seeds(plan, seeds)
        result["error"] = None
    except Exception as exc:
        result = {"games": [], "banks": [], "mean_bank": -1e12, "worst_bank": -1e12,
                  "error": repr(exc)}
    result.update(plan_id=_key(plan), params=_normalise(plan), eval_seconds=time.perf_counter()-started)
    return result


def _fitness(row: dict[str, Any]) -> float:
    """Mean remains primary, with a tail term to reject shop-draw collapses."""
    return float(row.get("mean_bank", -1e12)) + 0.5 * float(row.get("worst_bank", -1e12))


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def search(workers: int, minutes: float, population: int, max_full: int,
           plateau: int, search_seed: int) -> dict[str, Any]:
    log = RESULT_ROOT / "search-v3.jsonl"
    rows = _read(log)
    full_rows = [r for r in rows if r.get("stage") == "full"]
    seen = {r["plan_id"] for r in rows}
    rng = random.Random(search_seed)
    queued = [p for p in _seed_plans() if _key(p) not in seen]
    deadline = time.monotonic() + minutes * 60
    generation = max([int(r.get("generation", -1)) for r in rows] + [-1]) + 1
    best_full_index = min((i for i, r in enumerate(full_rows)
                           if _fitness(r) == max(_fitness(x) for x in full_rows)), default=-1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while time.monotonic() < deadline and len(full_rows) < max_full:
            elites = [r["params"] for r in sorted(full_rows, key=_fitness, reverse=True)[:8]]
            if not elites:
                elites = _seed_plans()
            candidates = []
            while queued and len(candidates) < population:
                candidates.append(queued.pop(0))
            while len(candidates) < population:
                heat = max(.35, 1.2 / math.sqrt(max(1, generation + 1)))
                child = _mutate(rng.choice(elites), rng, heat)
                if _key(child) not in seen and all(_key(x) != _key(child) for x in candidates):
                    candidates.append(child)
            futures = [pool.submit(_worker, p, RACE_SEEDS) for p in candidates]
            raced = [f.result() for f in as_completed(futures)]
            for row in raced:
                row.update(stage="race", generation=generation,
                           completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                _append(log, row); rows.append(row); seen.add(row["plan_id"])
            survivors = sorted(raced, key=_fitness, reverse=True)[:max(1, population // 3)]
            remaining = tuple(s for s in SOLO_SEEDS if s not in RACE_SEEDS)
            futures = [pool.submit(_worker, row["params"], remaining) for row in survivors]
            for future in as_completed(futures):
                tail = future.result()
                head = next(x for x in survivors if x["plan_id"] == tail["plan_id"])
                games = sorted(head["games"] + tail["games"], key=lambda g: g["seed"])
                banks = [g["final_bank"] for g in games]
                row = {**tail, "games": games, "banks": banks,
                       "mean_bank": statistics.fmean(banks), "worst_bank": min(banks),
                       "race_mean": head["mean_bank"], "stage": "full", "generation": generation,
                       "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                _append(log, row); rows.append(row); full_rows.append(row)
                if _fitness(row) > max((_fitness(x) for x in full_rows[:-1]), default=-1e12):
                    best_full_index = len(full_rows) - 1
                print(f"gen={generation} full={len(full_rows)} mean={row['mean_bank']:.0f} "
                      f"worst={row['worst_bank']:.0f} best={max(x['mean_bank'] for x in full_rows):.0f}", flush=True)
            generation += 1
            if len(full_rows) - 1 - best_full_index >= plateau:
                break
    best = max(full_rows, key=_fitness)
    payload = {"schema_version": 3, "engine": engine_version(), "plan_id": best["plan_id"],
               "plan": best["params"], "mean_bank": best["mean_bank"], "worst_bank": best["worst_bank"],
               "games": best["games"], "attempts": len([r for r in rows if r.get("stage") == "race"]),
               "full_evaluations": len(full_rows), "evaluations_since_best": len(full_rows)-1-best_full_index,
               "plateau_threshold": plateau}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "best.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def verify(plan: dict[str, Any]) -> dict[str, Any]:
    solo = []
    for seed in SOLO_SEEDS:
        solo.append({"seed": seed, "raw": _episode(seed, None),
                     "exec_b": _episode(seed, None, agent_path=EXEC_B),
                     "ext2": _episode(seed, plan, detailed=True)})
    contested = []
    for i, seed in enumerate(CONTEST_SEEDS):
        seat = i % 2
        contested.append({"seed": seed, "seat": seat,
                          "raw": _episode(seed, None, opponent=V46, seat=seat),
                          "ext2": _episode(seed, plan, opponent=V46, seat=seat, detailed=True)})
    legacy_plan = json.loads((solve.SOLO_RESULT_ROOT / "3910" / "best.json").read_text(encoding="utf-8"))["plan"]
    legacy = _episode(3910, legacy_plan, detailed=True)
    payload = {"engine": engine_version(), "legacy_3910": legacy,
               "solo": solo, "contested": contested}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minutes", type=float, default=90)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--max-full", type=int, default=300)
    parser.add_argument("--plateau", type=int, default=80)
    parser.add_argument("--search-seed", type=int, default=20260908)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be in 1..4")
    arena = os.environ.get("KAGGRI_ARENA_DIR", "")
    if not arena or not Path(arena).is_absolute():
        parser.error("KAGGRI_ARENA_DIR must be absolute")
    if args.verify:
        artifact = json.loads((RESULT_ROOT / "best.json").read_text(encoding="utf-8"))
        result = verify(artifact["plan"])
        print(json.dumps({"legacy": result["legacy_3910"]["final_bank"],
                          "solo_mean": statistics.fmean(x["ext2"]["final_bank"] for x in result["solo"]),
                          "contested_mean": statistics.fmean(x["ext2"]["final_bank"] for x in result["contested"])}, indent=2))
        return 0
    result = search(args.workers, args.minutes, args.population, args.max_full,
                    args.plateau, args.search_seed)
    print(json.dumps({k: result[k] for k in ("plan_id", "mean_bank", "worst_bank", "attempts", "full_evaluations", "evaluations_since_best")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
