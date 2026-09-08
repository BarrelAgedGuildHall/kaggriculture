"""Eight-process local search over pinned fixed-game day schedules.

Each worker keeps day-boundary engine snapshots for its accepted incumbent.  A
mutation first affecting day d restores that snapshot and executes only d..29;
the final winner is independently replayed by ``sim.replay``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from scripts.fixed_game.sim import PASS, Sim, replay

OUT = ROOT / "data" / "results" / "fixed-game" / "3910" / "search"
SCHEDULES = ROOT / "scripts" / "fixed_game" / "schedules"
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")


def schedule_hash(schedule):
    payload = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _phase_start(day):
    return 0 if day < 10 else 10 if day < 17 else 17 if day < 23 else 23


def random_start(base, rng):
    schedule = copy.deepcopy(base)
    # A demand-shaped starting point: carrot/wool, one land purchase, and a
    # smaller milk line.  Planting rows are target block markers for the router.
    for day in schedule["days"]:
        day["plantings"] = []
        day["builds"] = []
        day["purchases"] = [p for p in day["purchases"] if p["item"] not in ANIMALS]
        day["hires"] = rng.randint(7, 12)
    schedule["days"][rng.randint(5, 10)]["purchases"].append({"item": "LAND", "qty": 1})
    tiles = [(x, y) for y in range(10) for x in range(10)]
    rng.shuffle(tiles)
    for index, tile in enumerate(tiles[:rng.randint(35, 70)]):
        day = rng.choice((0, 10, 17, 23))
        crop = rng.choices(CROPS, weights=(25, 50, 7, 5, 8))[0]
        schedule["days"][day]["plantings"].append({"tile": list(tile), "crop": crop})
    for animal, total in (("SHEEP", rng.randint(12, 22)), ("COW", rng.randint(1, 8)), ("GOOSE", rng.randint(0, 6))):
        for index in range(total):
            day = min(20, index + rng.randint(0, 4))
            schedule["days"][day]["purchases"].append({"item": animal, "qty": 1})
    return validate_schedule(schedule)


def mutate(schedule, rng):
    out = copy.deepcopy(schedule)
    kind = rng.choice(("move_plant", "purchase", "hands", "sale", "swap_crop", "animal"))
    earliest = 0
    if kind == "move_plant":
        source_days = [d for d in out["days"] if d["plantings"]]
        if source_days:
            source = rng.choice(source_days)
            row = source["plantings"].pop(rng.randrange(len(source["plantings"])))
        else:
            source = out["days"][0]
            row = {"tile": [rng.randrange(10), rng.randrange(10)], "crop": rng.choice(CROPS)}
        old_day = source["day"]
        new_day = max(0, min(26, old_day + rng.choice((-7, -3, 3, 7))))
        row["tile"] = [rng.randrange(10), rng.randrange(10)]
        if rng.random() < .45:
            row["crop"] = rng.choice(CROPS)
        out["days"][new_day]["plantings"].append(row)
        earliest = min(_phase_start(old_day), _phase_start(new_day))
    elif kind == "purchase":
        choices = [(d, i, p) for d in out["days"] for i, p in enumerate(d["purchases"])]
        if choices and rng.random() < .75:
            spec, index, row = rng.choice(choices)
            old_day = spec["day"]
            spec["purchases"].pop(index)
            row["qty"] = max(1, int(row.get("qty", 1)) + rng.choice((-2, -1, 1, 2)))
            new_day = max(0, min(28, old_day + rng.choice((-3, -1, 1, 3))))
            out["days"][new_day]["purchases"].append(row)
            earliest = min(old_day, new_day)
        else:
            day = rng.randrange(0, 20)
            item = rng.choice(("WHEAT_SEED", "CARROT_SEED", "MELON_SEED", "STRAWBERRY_SEED", "WHEAT", "LAND"))
            out["days"][day]["purchases"].append({"item": item, "qty": 1 if item == "LAND" else rng.randint(3, 60)})
            earliest = day
    elif kind == "hands":
        day = rng.randrange(30)
        out["days"][day]["hires"] = max(1, min(15, out["days"][day]["hires"] + rng.choice((-2, -1, 1, 2))))
        earliest = day
    elif kind == "sale":
        day = rng.randrange(30)
        rows = out["days"][day]["sales"]
        if rows and rng.random() < .6:
            row = rng.choice(rows)
            row["qty"] = max(1, row["qty"] + rng.choice((-12, -6, 6, 12)))
        else:
            rows.append({"product": rng.choice(PRODUCTS), "qty": rng.randint(6, 60)})
        earliest = day
    elif kind == "swap_crop":
        rows = [(d, p) for d in out["days"] for p in d["plantings"]]
        if len(rows) >= 2:
            (a, p), (b, q) = rng.sample(rows, 2)
            p["crop"], q["crop"] = q["crop"], p["crop"]
            earliest = min(_phase_start(a["day"]), _phase_start(b["day"]))
    else:
        rows = [(d, i, p) for d in out["days"] for i, p in enumerate(d["purchases"]) if p["item"] in ANIMALS]
        if rows and rng.random() < .55:
            spec, index, _row = rng.choice(rows)
            spec["purchases"].pop(index)
            earliest = spec["day"]
        else:
            day = rng.randrange(0, 18)
            out["days"][day]["purchases"].append({"item": rng.choice(ANIMALS), "qty": 1})
            earliest = day
    return validate_schedule(out), earliest, kind


def evaluate(schedule, incumbent=None, earliest=0):
    controller = ReferenceExecutor(schedule, idle_fix=False)
    if incumbent is None or earliest <= 0:
        sim = Sim()
        actions = []
        snapshots = {}
        curve = {}
    else:
        sim = incumbent["sim"]
        start = min(29, max(0, int(earliest)))
        sim.restore(incumbent["snapshots"][start])
        actions = list(incumbent["actions"][:start * 24])
        snapshots = {day: snap for day, snap in incumbent["snapshots"].items() if day <= start}
        curve = {day: value for day, value in incumbent["curve"].items() if day < start}
        controller._last_obs = copy.deepcopy(sim.observation())
    last_day = sim.day()
    while not sim.done():
        obs = sim.observation()
        day, hour = int(obs["day"]), int(obs["hour"])
        if hour == 0 and day not in snapshots:
            snapshots[day] = sim.snapshot()
        action = controller.policy(obs)
        actions.append(action)
        sim.step(action)
        if sim.done() or sim.day() != last_day:
            curve[last_day] = sim.bank()
            last_day = sim.day()
    padded = actions + [copy.deepcopy(PASS)] * (720 - len(actions))
    return {"bank": sim.bank(), "curve": curve, "actions": padded[:720],
            "snapshots": snapshots, "sim": sim, "faults": controller.faults}


def worker_main(worker, start_schedule, eval_limit, deadline, seed, result_queue):
    rng = random.Random(seed)
    incumbent_schedule = validate_schedule(start_schedule)
    incumbent = evaluate(incumbent_schedule)
    champion_schedule, champion = incumbent_schedule, incumbent
    accepted = 0
    stale = 0
    result_queue.put({"type": "best", "worker": worker, "bank": incumbent["bank"],
                      "schedule": incumbent_schedule, "actions": incumbent["actions"],
                      "curve": incumbent["curve"], "accepted_count": accepted})
    for index in range(eval_limit):
        if time.time() >= deadline:
            break
        candidate_schedule, earliest, mutation = mutate(incumbent_schedule, rng)
        candidate = evaluate(candidate_schedule, incumbent, earliest)
        better = candidate["faults"] == 0 and candidate["bank"] > incumbent["bank"]
        if better:
            incumbent_schedule, incumbent = candidate_schedule, candidate
            if incumbent["bank"] > champion["bank"]:
                champion_schedule, champion = incumbent_schedule, incumbent
            accepted += 1
            stale = 0
        else:
            stale += 1
        restarted = False
        if stale >= 80:
            if worker % 4 == 0:
                incumbent_schedule, incumbent = champion_schedule, champion
            else:
                incumbent_schedule = random_start(start_schedule, rng)
                incumbent = evaluate(incumbent_schedule)
            stale = 0
            restarted = True
        result_queue.put({
            "type": "evaluation", "worker": worker, "index": index,
            "hash": schedule_hash(candidate_schedule), "bank": candidate["bank"],
            "day10_bank": candidate["curve"].get(10), "day20_bank": candidate["curve"].get(20),
            "accepted": better, "mutation": mutation, "earliest_day": earliest,
            "faults": candidate["faults"], "restarted": restarted,
        })
        if better:
            result_queue.put({"type": "best", "worker": worker, "bank": incumbent["bank"],
                              "schedule": incumbent_schedule, "actions": incumbent["actions"],
                              "curve": incumbent["curve"], "accepted_count": accepted})
    result_queue.put({"type": "done", "worker": worker, "accepted_count": accepted})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=2400)
    parser.add_argument("--minutes", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=3910)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    OUT.mkdir(parents=True, exist_ok=True)
    dairy = json.loads((SCHEDULES / "dairy-v0.json").read_text(encoding="utf-8"))
    tempo = json.loads((SCHEDULES / "tempo.json").read_text(encoding="utf-8"))
    prior_path = OUT / "best-schedule.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else dairy
    random_a = random_start(dairy, random.Random(args.seed + 101))
    random_b = random_start(tempo, random.Random(args.seed + 202))
    starts = (prior, tempo, random_a, random_b)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    started = time.time()
    deadline = started + args.minutes * 60
    per_worker = (args.evaluations + args.workers - 1) // args.workers
    processes = [context.Process(target=worker_main,
                                 args=(worker, starts[worker % 4], per_worker, deadline,
                                       args.seed + 1009 * worker, result_queue))
                 for worker in range(args.workers)]
    for process in processes:
        process.start()
    done = evaluations = accepted = 0
    best = None
    log_path = OUT / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        while done < len(processes):
            try:
                message = result_queue.get(timeout=5)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue
            if message["type"] == "done":
                done += 1
                accepted += message["accepted_count"]
            elif message["type"] == "evaluation":
                evaluations += 1
                message["elapsed_seconds"] = time.time() - started
                log.write(json.dumps(message, separators=(",", ":")) + "\n")
                log.flush()
            elif best is None or message["bank"] > best["bank"]:
                best = message
                (OUT / "best-schedule.json").write_text(json.dumps(best["schedule"], indent=2) + "\n", encoding="utf-8")
                (OUT / "best-actions.json").write_text(json.dumps(best["actions"], indent=2) + "\n", encoding="utf-8")
                print(f"best ${best['bank']:,.0f} worker={best['worker']} evaluations={evaluations}", flush=True)
    for process in processes:
        process.join(timeout=10)
    if best is None:
        raise RuntimeError("search produced no accepted schedule")
    verified = replay(best["actions"])
    summary = {"evaluations": evaluations, "accepted_moves": accepted,
               "wall_seconds": time.time() - started, "workers": args.workers,
               "best_bank": best["bank"], "replay_bank": verified,
               "best_curve": best["curve"], "faults": 0}
    if verified != best["bank"]:
        raise AssertionError(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
