"""Parallel anytime search over pinned fixed-game schedules.

Round 3 mixes distinct economic families, block moves, herd timing moves,
hands-profile moves, sale policies, annealing, and top-k restarts.  Every winner
is replayed independently through the pinned real-engine simulator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import queue
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fixed_game.execute import ReferenceExecutor, validate_schedule
from scripts.fixed_game.pin import DEFAULT_INSTANCE, load_instance
from scripts.fixed_game.sim import PASS, Sim, replay

SCHEDULES = ROOT / "scripts" / "fixed_game" / "schedules"
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")


def schedule_hash(schedule):
    payload = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _phase_start(day):
    return 0 if day < 10 else 10 if day < 17 else 17 if day < 23 else 23


def _clear_plan(base, family):
    out = validate_schedule(base)
    out["family"] = family
    out["sale_policies"] = {}
    for day in out["days"]:
        day.update(hires=8, purchases=[], builds=[], plantings=[], sales=[])
    return out


def _tile_block(crop, count, *, near=True, offset=0):
    tiles = [(x, y) for y in range(10) for x in range(10)]
    shed = ((4, 4), (5, 4), (4, 5), (5, 5))
    tiles.sort(key=lambda p: (min(abs(p[0] - x) + abs(p[1] - y) for x, y in shed), p[1], p[0]),
               reverse=not near)
    return [{"tile": list(tile), "crop": crop} for tile in tiles[offset:offset + count]]


def _buy_each(schedule, animal, total, days):
    for index in range(total):
        schedule["days"][days[index % len(days)]]["purchases"].append({"item": animal, "qty": 1})


def _season_drain(instance):
    total = {product: 0 for product in PRODUCTS}
    for row in instance.get("town", {}).get("daily_drain", []):
        for product in PRODUCTS:
            total[product] += int(row.get("drain", {}).get(product, 0))
    return total


def _crop_markers(mix):
    """Encode target crop proportions as non-overlapping shed-relative blocks."""
    rows, offset = [], 0
    for crop in ("STRAWBERRY", "TOMATO", "WHEAT", "CARROT", "MELON"):
        count = max(0, int(mix.get(crop, 0)))
        rows.extend(_tile_block(crop, count, near=True, offset=offset))
        offset += count
    return rows


def _drain_crop_start(tempo, instance, *, strawberry_only=False):
    drain = _season_drain(instance)
    # A tile can complete two ongoing-crop cohorts in-season (the second is a
    # partial late cohort).  Translate season demand into comparable tile load,
    # then normalize to a 70-tile crop budget, leaving room for a small herd.
    capacity = {"STRAWBERRY": 12, "TOMATO": 16, "WHEAT": 42,
                "CARROT": 36, "MELON": 6}
    load = {crop: drain[crop] / capacity[crop] for crop in capacity}
    if strawberry_only:
        load["TOMATO"] = 0
    scale = 70 / max(1.0, sum(load.values()))
    mix = {crop: max(1 if drain[crop] else 0, round(load[crop] * scale))
           for crop in capacity}
    family = "strawberry_drain" if strawberry_only else "strawberry_tomato_drain"
    out = _clear_plan(tempo, family)
    out["crop_lines"] = {
        "source": "instance_town_drain",
        "season_drain": {crop: drain[crop] for crop in capacity},
        "target_tiles": mix,
        "last_plant_day": {"STRAWBERRY": 19, "TOMATO": 21},
        "priority_products": ["STRAWBERRY"] + ([] if strawberry_only else ["TOMATO"]),
    }
    for day in out["days"]:
        index = day["day"]
        day["hires"] = 8 if index < 8 else 12 if index < 24 else 10
    # Re-state the drain-sized blocks at each router phase boundary.  The router
    # interprets these markers as phase proportions and plants only legal tiles.
    for day in (0, 10, 17, 23):
        phase_mix = dict(mix)
        if day == 23:
            phase_mix["STRAWBERRY"] = phase_mix["TOMATO"] = 0
        out["days"][day]["plantings"] = _crop_markers(phase_mix)
    out["days"][0]["purchases"].append({"item": "LAND", "qty": 1})
    out["days"][8]["purchases"].append({"item": "LAND", "qty": 1})
    out["days"][14]["purchases"].append({"item": "LAND", "qty": 1})
    # Six animals retain fertilizer generation and non-crop revenue without
    # consuming the labour/tile majority needed by the high-drain crop line.
    _buy_each(out, "COW", 3, (2, 4, 6))
    _buy_each(out, "SHEEP", 3, (3, 5, 7))
    return validate_schedule(out)


def _brain_strawberry_start(tempo, instance):
    """Schedule-space analogue of the raw strawberry control's economic plan."""
    drain = _season_drain(instance)
    out = _clear_plan(tempo, "brain_strawberry_line")
    out["crop_lines"] = {
        "source": "instance_town_drain",
        "season_drain": {crop: drain[crop] for crop in CROPS},
        "target_tiles": {"WHEAT": 30, "STRAWBERRY": 65, "MELON": 5},
        "last_plant_day": {"STRAWBERRY": 16, "TOMATO": 14},
        "priority_products": ["STRAWBERRY", "TOMATO"],
    }
    profiles = {
        0: {"WHEAT": 18, "MELON": 5, "STRAWBERRY": 2},
        10: {"WHEAT": 30, "STRAWBERRY": 65, "MELON": 5},
        17: {"WHEAT": 72, "CARROT": 28},
        23: {"WHEAT": 72, "CARROT": 28},
    }
    for day, mix in profiles.items():
        out["days"][day]["plantings"] = _crop_markers(mix)
    for day in out["days"]:
        index = day["day"]
        day["hires"] = 4 if index < 7 else 5 if index < 10 else 6 if index == 10 else 10 if index == 11 else 12
    for day in (5, 9, 11):
        out["days"][day]["purchases"].append({"item": "LAND", "qty": 1})
    _buy_each(out, "COW", 7, (0, 1, 3, 5, 5, 5, 7))
    _buy_each(out, "SHEEP", 11, (0, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11))
    return validate_schedule(out)


def family_starts(prior, tempo, instance=None):
    incumbent = validate_schedule(prior)
    incumbent["family"] = "round2_incumbent"

    bound = _clear_plan(tempo, "goose_carrot_sheep")
    bound["days"][0]["hires"] = 7
    bound["days"][0]["purchases"] += [{"item": "WHEAT_SEED", "qty": 14},
                                        {"item": "CARROT_SEED", "qty": 18}]
    bound["days"][0]["plantings"] = (_tile_block("CARROT", 12, near=True)
                                         + _tile_block("WHEAT", 13, near=False))
    bound["days"][8]["purchases"].append({"item": "LAND", "qty": 1})
    # A feasible integer foothold in the LP's family.  The herd-count operator
    # can scale it toward the LP's 49/22 corner without starting bankrupt.
    _buy_each(bound, "GOOSE", 16, list(range(8)))
    _buy_each(bound, "SHEEP", 14, list(range(1, 11)))
    bound["days"][5]["plantings"] += _tile_block("MELON", 10, near=True, offset=20)

    crop = _clear_plan(tempo, "tomato_carrot_crop")
    crop["days"][0]["purchases"] += [{"item": "CARROT_SEED", "qty": 40},
                                       {"item": "TOMATO_SEED", "qty": 18}]
    crop["days"][0]["plantings"] = (_tile_block("CARROT", 17, near=True)
                                        + _tile_block("TOMATO", 8, near=False))
    crop["days"][6]["purchases"].append({"item": "LAND", "qty": 1})
    _buy_each(crop, "SHEEP", 8, list(range(2, 10)))

    mixed = _clear_plan(tempo, "mixed")
    mixed["days"][0]["purchases"] += [{"item": "WHEAT_SEED", "qty": 14},
                                        {"item": "CARROT_SEED", "qty": 14},
                                        {"item": "MELON_SEED", "qty": 10}]
    mixed["days"][0]["plantings"] = (_tile_block("CARROT", 8, near=True)
                                         + _tile_block("WHEAT", 9, near=False)
                                         + _tile_block("MELON", 8, near=True, offset=8))
    _buy_each(mixed, "GOOSE", 10, list(range(8)))
    _buy_each(mixed, "SHEEP", 12, list(range(1, 10)))
    _buy_each(mixed, "COW", 4, list(range(3, 10)))
    starts = [validate_schedule(x) for x in (incumbent, bound, crop, mixed)]
    if instance is not None:
        starts += [_drain_crop_start(tempo, instance),
                   _brain_strawberry_start(tempo, instance)]
    return starts


def _block(schedule, rng):
    candidates = []
    for day in schedule["days"]:
        for crop in CROPS:
            rows = [p for p in day["plantings"] if p["crop"] == crop]
            if len(rows) >= 2:
                candidates.append((day, crop, rows))
    return rng.choice(candidates) if candidates else None


def mutate(schedule, rng):
    out = copy.deepcopy(schedule)
    kind = rng.choices(
        ("recrop_block", "shift_block_day", "swap_block_positions", "herd_shift",
         "hands_profile", "sale_policy", "purchase", "animal_count"),
        weights=(18, 14, 10, 14, 18, 8, 8, 10), k=1)[0]
    earliest = 0
    if kind == "recrop_block":
        chosen = _block(out, rng)
        if chosen:
            day, old_crop, rows = chosen
            new_crop = rng.choice([c for c in CROPS if c != old_crop])
            for row in rng.sample(rows, rng.randint(2, len(rows))):
                row["crop"] = new_crop
            earliest = _phase_start(day["day"])
    elif kind == "shift_block_day":
        chosen = _block(out, rng)
        if chosen:
            source, _crop, rows = chosen
            new_day = max(0, min(27, source["day"] + rng.choice((-7, -3, -1, 1, 3, 7))))
            take = rng.sample(rows, rng.randint(2, len(rows)))
            for row in take:
                source["plantings"].remove(row)
                out["days"][new_day]["plantings"].append(row)
            earliest = min(_phase_start(source["day"]), _phase_start(new_day))
    elif kind == "swap_block_positions":
        blocks = []
        for day in out["days"]:
            for crop in CROPS:
                rows = [p for p in day["plantings"] if p["crop"] == crop]
                if rows:
                    blocks.append((day, rows))
        if len(blocks) >= 2:
            left, right = rng.sample(blocks, 2)
            n = min(len(left[1]), len(right[1]))
            for a, b in zip(rng.sample(left[1], n), rng.sample(right[1], n)):
                a["tile"], b["tile"] = b["tile"], a["tile"]
            earliest = min(_phase_start(left[0]["day"]), _phase_start(right[0]["day"]))
    elif kind == "herd_shift":
        animal = rng.choice(ANIMALS)
        rows = [(d, p) for d in out["days"] for p in d["purchases"] if p["item"] == animal]
        shift = rng.choice((-3, -1, 1, 3))
        for source, row in list(rows):
            source["purchases"].remove(row)
            out["days"][max(0, min(25, source["day"] + shift))]["purchases"].append(row)
        earliest = max(0, min((d["day"] for d, _ in rows), default=0) + min(0, shift))
    elif kind == "hands_profile":
        start = rng.randrange(30)
        end = min(30, start + rng.randint(2, 10))
        delta = rng.choice((-1, 1))
        for day in range(start, end):
            out["days"][day]["hires"] = max(1, min(15, out["days"][day]["hires"] + delta))
        earliest = start
    elif kind == "sale_policy":
        product = rng.choice(PRODUCTS)
        modes = ["sell_on_sight", "hold_to_drain",
                 "never_collect" if product == "FERTILIZER" else "never"]
        current = out.setdefault("sale_policies", {}).get(product, "sell_on_sight")
        out["sale_policies"][product] = rng.choice([x for x in modes if x != current])
    elif kind == "purchase":
        day = rng.randrange(20)
        item = rng.choice(("WHEAT_SEED", "CARROT_SEED", "TOMATO_SEED", "MELON_SEED",
                           "STRAWBERRY_SEED", "LAND"))
        out["days"][day]["purchases"].append(
            {"item": item, "qty": 1 if item == "LAND" else rng.randint(3, 30)})
        earliest = day
    else:
        animal = rng.choice(ANIMALS)
        rows = [(d, p) for d in out["days"] for p in d["purchases"] if p["item"] == animal]
        if rows and rng.random() < .5:
            day, row = rng.choice(rows)
            day["purchases"].remove(row)
            earliest = day["day"]
        else:
            day = rng.randrange(18)
            out["days"][day]["purchases"].append({"item": animal, "qty": 1})
            earliest = day
    return validate_schedule(out), earliest, kind


def _controller(schedule, fertilizer_drain):
    return ReferenceExecutor(schedule, idle_fix=True, fertilizer_drain=fertilizer_drain,
                             fertilizer_threshold=10, animal_sweep=False)


def _first_changed_day(controller, incumbent, earliest):
    """Find the first day whose routed action differs, without stepping the engine."""
    observations = incumbent["observations"]
    start = min(29, max(0, int(earliest)))
    first_turn = start * 24
    if first_turn >= len(observations):
        return None
    controller._last_obs = copy.deepcopy(observations[first_turn])
    for turn in range(first_turn, len(observations)):
        action = controller.policy(observations[turn])
        if action != incumbent["actions"][turn]:
            return turn // 24
    return None


def evaluate(schedule, instance_path, fertilizer_drain, incumbent=None, earliest=0,
             *, prefix_eval=True):
    controller = _controller(schedule, fertilizer_drain)
    changed_day = None
    if prefix_eval and incumbent is not None:
        # Positive mutation boundaries are already exact and avoid a redundant
        # router pass.  Day-zero is conservative for several global-looking
        # mutations, so discover their actual first changed action cheaply.
        changed_day = int(earliest) if earliest > 0 else _first_changed_day(
            controller, incumbent, earliest)
        if changed_day is None:
            return {"bank": incumbent["bank"], "curve": incumbent["curve"],
                    "actions": incumbent["actions"], "observations": incumbent["observations"],
                    "snapshots": incumbent["snapshots"], "sim": incumbent["sim"],
                    "faults": controller.faults}
        controller = _controller(schedule, fertilizer_drain)
    if not prefix_eval or incumbent is None:
        sim = Sim(instance_path)
        actions, observations, snapshots, curve = [], [], {}, {}
    else:
        sim = incumbent["sim"]
        start = changed_day
        sim.restore(incumbent["snapshots"][start])
        actions = list(incumbent["actions"][:start * 24])
        observations = list(incumbent["observations"][:start * 24])
        snapshots = {day: snap for day, snap in incumbent["snapshots"].items() if day <= start}
        curve = {day: value for day, value in incumbent["curve"].items() if day < start}
        controller._last_obs = copy.deepcopy(sim.observation())
    last_day = sim.day()
    while not sim.done():
        obs = sim.observation()
        day, hour = int(obs["day"]), int(obs["hour"])
        if hour == 0 and day not in snapshots:
            snapshots[day] = sim.snapshot()
        observations.append(obs)
        action = controller.policy(obs)
        actions.append(action)
        sim.step(action)
        if sim.done() or sim.day() != last_day:
            curve[last_day] = sim.bank()
            last_day = sim.day()
    padded = actions + [copy.deepcopy(PASS)] * (720 - len(actions))
    return {"bank": sim.bank(), "curve": curve, "actions": padded[:720],
            "observations": observations,
            "snapshots": snapshots, "sim": sim, "faults": controller.faults}


def worker_main(worker, start_schedule, eval_limit, deadline, started, seed, instance_path,
                fertilizer_drain, temperature, top_k, prefix_eval, result_queue):
    rng = random.Random(seed)
    family = start_schedule.get("family", f"worker_{worker}")
    incumbent_schedule = validate_schedule(start_schedule)
    incumbent = evaluate(incumbent_schedule, instance_path, fertilizer_drain)
    leaders = [(incumbent["bank"], copy.deepcopy(incumbent_schedule), incumbent)]
    accepted = stale = 0
    result_queue.put({"type": "best", "worker": worker, "family": family,
                      "bank": incumbent["bank"], "schedule": incumbent_schedule,
                      "actions": incumbent["actions"], "curve": incumbent["curve"]})
    for index in range(eval_limit):
        if time.time() >= deadline:
            break
        candidate_schedule, earliest, mutation = mutate(incumbent_schedule, rng)
        candidate = evaluate(candidate_schedule, instance_path, fertilizer_drain, incumbent, earliest,
                             prefix_eval=prefix_eval)
        delta = candidate["bank"] - incumbent["bank"]
        fraction = max(0.0, min(1.0, (time.time() - started) / max(1.0, deadline - started)))
        temp = max(40.0, temperature * (1.0 - .85 * fraction))
        take = candidate["faults"] == 0 and (delta > 0 or rng.random() < math.exp(delta / temp))
        if take:
            incumbent_schedule, incumbent = candidate_schedule, candidate
            accepted += 1
            stale = 0 if delta > 0 else stale + 1
            leaders.append((incumbent["bank"], copy.deepcopy(incumbent_schedule), incumbent))
            leaders = sorted(leaders, key=lambda row: row[0], reverse=True)[:top_k]
        else:
            stale += 1
        restarted = False
        if stale >= 60:
            _score, incumbent_schedule, incumbent = rng.choice(leaders)
            incumbent_schedule = copy.deepcopy(incumbent_schedule)
            stale = 0
            restarted = True
        result_queue.put({"type": "evaluation", "worker": worker, "family": family,
                          "index": index, "hash": schedule_hash(candidate_schedule),
                          "bank": candidate["bank"], "accepted": take,
                          "mutation": mutation, "earliest_day": earliest,
                          "faults": candidate["faults"], "restarted": restarted})
        if candidate["faults"] == 0 and candidate["bank"] >= leaders[0][0]:
            result_queue.put({"type": "best", "worker": worker, "family": family,
                              "bank": candidate["bank"], "schedule": candidate_schedule,
                              "actions": candidate["actions"], "curve": candidate["curve"]})
    result_queue.put({"type": "done", "worker": worker, "accepted_count": accepted})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=2400)
    parser.add_argument("--minutes", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=3910)
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--tag", help="write to search-<tag> instead of search")
    parser.add_argument("--temperature", type=float, default=900.0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--prefix-eval", action=argparse.BooleanOptionalAction, default=True,
                        help="restore the incumbent's exact day-start snapshot (default: enabled)")
    parser.add_argument("--start-schedule", type=Path,
                        help="optional incumbent schedule used as the primary start")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    instance_path = args.instance or ROOT / "data" / "results" / "fixed-game" / str(args.seed) / "instance.json"
    if not instance_path.exists() and args.seed == 3910:
        instance_path = DEFAULT_INSTANCE
    instance = load_instance(instance_path)
    if int(instance["seed"]) != args.seed:
        raise ValueError(f"instance seed {instance['seed']} does not match --seed {args.seed}")
    fertilizer_drain = sum(int(row.get("drain", {}).get("FERTILIZER", 0))
                           for row in instance.get("town", {}).get("daily_drain", []))
    base_out = ROOT / "data" / "results" / "fixed-game" / str(args.seed)
    out = base_out / (f"search-{args.tag}" if args.tag else "search")
    out.mkdir(parents=True, exist_ok=True)
    dairy = json.loads((SCHEDULES / "dairy-v0.json").read_text(encoding="utf-8"))
    tempo = json.loads((SCHEDULES / "tempo.json").read_text(encoding="utf-8"))
    seed_dir = ROOT / "data" / "results" / "fixed-game" / str(args.seed)
    seed_candidates = [seed_dir / "search-round3-generalization" / "best-schedule.json",
                       seed_dir / "search" / "best-schedule.json"]
    prior_path = args.start_schedule or next((path for path in seed_candidates if path.exists()),
                      ROOT / "data" / "results" / "fixed-game" / "3910" / "search" / "best-schedule.json")
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else dairy
    starts = family_starts(prior, tempo, instance)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    started = time.time()
    deadline = started + args.minutes * 60
    per_worker = (args.evaluations + args.workers - 1) // args.workers
    processes = [context.Process(target=worker_main,
                                 args=(worker, starts[worker % len(starts)], per_worker,
                                       deadline, started, args.seed + 1009 * worker,
                                       str(instance_path), fertilizer_drain,
                                       args.temperature, args.top_k, args.prefix_eval, result_queue))
                 for worker in range(args.workers)]
    for process in processes:
        process.start()
    done = evaluations = accepted = faults = 0
    best = None
    family_best = {}
    best_events = []
    log_path = out / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        while done < len(processes):
            try:
                message = result_queue.get(timeout=5)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue
            elapsed = time.time() - started
            if message["type"] == "done":
                done += 1
                accepted += message["accepted_count"]
            elif message["type"] == "evaluation":
                evaluations += 1
                faults += message["faults"]
                message["elapsed_seconds"] = elapsed
                log.write(json.dumps(message, separators=(",", ":")) + "\n")
            else:
                family = message["family"]
                if family not in family_best or message["bank"] > family_best[family]["bank"]:
                    family_best[family] = message
                if best is None or message["bank"] > best["bank"]:
                    best = message
                    best_events.append({"elapsed_seconds": elapsed, "bank": best["bank"],
                                        "family": family, "evaluations": evaluations})
                    (out / "best-schedule.json").write_text(json.dumps(best["schedule"], indent=2) + "\n", encoding="utf-8")
                    (out / "best-actions.json").write_text(json.dumps(best["actions"], indent=2) + "\n", encoding="utf-8")
                    print(f"best ${best['bank']:,.0f} family={family} evaluations={evaluations}", flush=True)
            log.flush()
    for process in processes:
        process.join(timeout=10)
    if best is None:
        raise RuntimeError("search produced no schedule")
    verified = replay(best["actions"], instance_path)
    if verified != best["bank"]:
        raise AssertionError({"search": best["bank"], "replay": verified})
    wall = time.time() - started
    milestones = {}
    for minute in (5, 10, 20, 30):
        eligible = [event for event in best_events if event["elapsed_seconds"] <= minute * 60]
        milestones[str(minute)] = max((event["bank"] for event in eligible), default=None)
    family_summary = {name: {"bank": row["bank"], "worker": row["worker"]}
                      for name, row in sorted(family_best.items())}
    summary = {"engine": "kaggle-environments 1.32.7", "seed": args.seed,
               "instance": str(instance_path), "evaluations": evaluations,
               "accepted_moves": accepted, "wall_seconds": wall, "workers": args.workers,
               "prefix_evaluation": args.prefix_eval,
               "evaluations_per_minute": evaluations / max(wall / 60, 1e-9),
               "best_bank": best["bank"], "replay_bank": verified,
               "winning_family": best["family"], "family_best": family_summary,
               "best_so_far_minutes": milestones, "best_events": best_events,
               "best_curve": best["curve"], "faults": faults}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
