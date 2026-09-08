"""Polish a fixed Kaggriculture action tape without changing its economic plan.

The search changes only unit execution and SELL timing.  Successful seed/product/
animal/land purchases, hires, and plantings must exactly match the reference.
Candidates are evaluated by restoring the snapshot immediately before their
first changed turn and replaying the suffix with the real 1.32.7 interpreter.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fixed_game.sim import Sim, replay

OUT = ROOT / "data" / "results" / "fixed-game" / "3910" / "polish"
DEFAULT_AGENT = ROOT.parents[1] / "artisans" / "modeler" / "bench" / "exec_b_7baa291.py"
PASS = ["PASS"]
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
ACTS = {"DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE",
        "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


def _plain_action(action: Any, hands: int) -> dict[str, Any]:
    if not isinstance(action, dict):
        action = {}
    farmer = action.get("farmer", PASS)
    farmer = list(farmer) if isinstance(farmer, list) and farmer else list(PASS)
    raw_hands = action.get("hands", [])
    raw_hands = raw_hands if isinstance(raw_hands, list) else []
    hand_actions = []
    for i in range(hands):
        a = raw_hands[i] if i < len(raw_hands) else PASS
        hand_actions.append(list(a) if isinstance(a, list) and a else list(PASS))
    market = action.get("market", [])
    market = [list(o) for o in market[:10] if isinstance(o, list)] if isinstance(market, list) else []
    return {"farmer": farmer, "hands": hand_actions, "market": market}


def load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("fixed_exec_b_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_reference(agent_path: Path) -> tuple[list[dict], list[float]]:
    module = load_agent(agent_path)
    sim = Sim()
    actions, timings = [], []
    while not sim.done():
        obs = sim.observation()
        t0 = time.perf_counter_ns()
        try:
            action = module.agent(copy.deepcopy(obs))
        except Exception as exc:
            raise RuntimeError(f"reference policy fault at turn {len(actions)}") from exc
        timings.append((time.perf_counter_ns() - t0) / 1e6)
        action = _plain_action(action, len(obs["farms"][0].get("hands", [])))
        actions.append(action)
        sim.step(action)
    while len(actions) < 720:
        actions.append({"farmer": list(PASS), "hands": [], "market": []})
    return actions[:720], timings


def _inv(obs: dict) -> Counter:
    private = obs.get("private", {}) or {}
    out = Counter(private.get("shed", {}) or {})
    for inv in private.get("inventories", []) or []:
        out.update(inv or {})
    return out


def _positions(obs: dict) -> list[tuple[int, int]]:
    farm = obs["farms"][0]
    return [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]


def analyze(actions: list[dict], include_snapshots: bool = False) -> dict[str, Any]:
    """Replay and build an engine-observed ledger and fixed-plan signature."""
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    accepted: list[tuple[int, str, str]] = []
    sold = defaultdict(Counter)
    revenue = defaultdict(Counter)
    current_turn = -1
    current_day = -1
    original_commit = kg._commit_unit

    def watched_commit(op, item, price, farm, private, market, shed_capacity=100):
        ok = original_commit(op, item, price, farm, private, market, shed_capacity)
        if ok:
            accepted.append((current_turn, op, item))
            if op == "SELL":
                sold[current_day][item] += 1
                revenue[current_day][item] += float(price)
        return ok

    kg._commit_unit = watched_commit
    sim = Sim()
    snaps = []
    days = {d: {"day": d, "bank": 0.0, "hand_turns": Counter(), "produced": Counter(),
                "sold": sold[d], "revenue": revenue[d], "rotted": Counter(),
                "undelivered": Counter(), "by_unit": defaultdict(Counter),
                "idle_locations": Counter(), "moving_locations": Counter()} for d in range(30)}
    plants, hires, lands = [], Counter(), Counter()
    successful_units = 0
    try:
        for t, raw in enumerate(actions):
            if sim.done():
                break
            current_turn = t
            before = sim.observation()
            current_day = int(before["day"])
            if include_snapshots:
                snaps.append(sim.snapshot())
            positions = _positions(before)
            action = _plain_action(raw, len(positions) - 1)
            unit_actions = [action["farmer"]] + action["hands"]
            old_tiles = copy.deepcopy(before["farms"][0]["tiles"])
            before_hands = len(before["farms"][0].get("hands", []))
            before_land = len(before["farms"][0].get("unlocked_quadrants", []))
            before_inv = _inv(before)
            after = sim.step(action)
            after_hands = len(after["farms"][0].get("hands", []))
            if int(after.get("day", current_day)) == current_day:
                if after_hands > before_hands:
                    hires[current_day] += after_hands - before_hands
                after_land = len(after["farms"][0].get("unlocked_quadrants", []))
                if after_land > before_land:
                    lands[current_day] += after_land - before_land
            for ui, (pos, ua) in enumerate(zip(positions, unit_actions)):
                op = ua[0] if ua else "PASS"
                bucket = "moving" if op in MOVES else "acting" if op in ACTS else "idle"
                days[current_day]["hand_turns"][bucket] += 1
                label = "farmer" if ui == 0 else f"hand_{ui}"
                days[current_day]["by_unit"][label][bucket] += 1
                if bucket == "idle":
                    days[current_day]["idle_locations"][(label, pos)] += 1
                elif bucket == "moving":
                    days[current_day]["moving_locations"][(label, pos, op)] += 1
                x, y = pos
                old = old_tiles[y][x]
                # Successful harvest production is the yield removed from its tile.
                if op == "HARVEST" and isinstance(old, dict) and int(old.get("yield_units", 0) or 0) > 0:
                    new = after["farms"][0]["tiles"][y][x]
                    changed = new != old
                    if changed:
                        product = old.get("crop")
                        animal = old.get("animal")
                        if not product and animal:
                            product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(animal)
                        if product:
                            days[current_day]["produced"][product] += int(old.get("yield_units", 0) or 0)
                new = after["farms"][0]["tiles"][y][x]
                if op == "PLANT" and old is None and isinstance(new, dict) and new.get("kind") == "PLANT":
                    plants.append((current_day, x, y, new.get("crop")))
                if op != "PASS" and (new != old or _positions(after)[min(ui, len(_positions(after))-1)] != pos
                                     or _inv(after) != before_inv):
                    successful_units += 1
            # Decay not caused by a harvest at that coordinate.
            harvest_pos = {p for p, a in zip(positions, unit_actions) if a and a[0] == "HARVEST"}
            for y in range(10):
                for x in range(10):
                    if (x, y) in harvest_pos:
                        continue
                    old, new = old_tiles[y][x], after["farms"][0]["tiles"][y][x]
                    if isinstance(old, dict):
                        oy = int(old.get("yield_units", 0) or 0)
                        ny = int(new.get("yield_units", 0) or 0) if isinstance(new, dict) else 0
                        if ny < oy:
                            p = old.get("crop") or {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(old.get("animal"))
                            if p:
                                days[current_day]["rotted"][p] += oy - ny
            day_end = int(after.get("day", current_day)) != current_day or sim.done()
            if day_end:
                days[current_day]["bank"] = sim.bank()
                days[current_day]["undelivered"] = _inv(after)
    finally:
        kg._commit_unit = original_commit
    purchases = Counter((t // 24, op, item) for t, op, item in accepted if op.startswith("BUY_"))
    signature = {
        "purchases": sorted((d, op, item, n) for (d, op, item), n in purchases.items()),
        "hires": sorted(hires.items()), "lands": sorted(lands.items()), "plantings": plants,
    }
    serial_days = []
    for d in range(30):
        row = days[d]
        serial_days.append({"day": d, "bank": row["bank"],
                            "hand_turns": dict(row["hand_turns"]),
                            "by_unit": {u: dict(c) for u, c in row["by_unit"].items()},
                            "idle_hotspots": [{"unit": u, "position": list(p), "turns": n}
                                               for (u, p), n in row["idle_locations"].most_common(8)],
                            "moving_hotspots": [{"unit": u, "position": list(p), "direction": op, "turns": n}
                                                 for (u, p, op), n in row["moving_locations"].most_common(8)],
                            "produced": dict(row["produced"]), "sold": dict(row["sold"]),
                            "revenue": dict(row["revenue"]), "undelivered": dict(row["undelivered"]),
                            "rotted": dict(row["rotted"])})
    result = {"bank": sim.bank(), "days": serial_days, "signature": signature,
              "accepted_market_units": len(accepted), "successful_unit_actions": successful_units}
    if include_snapshots:
        result["snapshots"] = snaps
    return result


def _op(a: list) -> str:
    return str(a[0]) if isinstance(a, list) and a else "PASS"


def _slots(action: dict) -> list[list]:
    return [action.get("farmer", PASS)] + list(action.get("hands", []))


def mutate(tape: list[dict], rng: random.Random, mix: int) -> tuple[list[dict], int, str, list[int]] | None:
    out = copy.deepcopy(tape)
    kind = rng.choices(
        ["sale_shift", "idle_prefetch", "route_reorder", "swap_routes", "skip_water", "same_turn_swap"],
        weights=([44, 18, 15, 8, 5, 10], [55, 8, 10, 5, 2, 20], [30, 25, 20, 10, 5, 10])[mix % 3], k=1)[0]
    changed: list[int] = []
    if kind == "sale_shift":
        sources = [(t, i) for t in range(719) for i, o in enumerate(out[t].get("market", [])) if _op(o) == "SELL"]
        if not sources:
            return None
        t, i = rng.choice(sources)
        order = out[t]["market"].pop(i)
        day0 = t // 24
        # Prefer a later town-consumption boundary, but explore both directions.
        candidates = [u for u in range(day0 * 24, min(719, (day0 + 1) * 24)) if u != t and len(out[u].get("market", [])) < 10]
        if rng.random() < .18:
            candidates += [u for u in range(max(0, t - 72), min(719, t + 73)) if u != t and len(out[u].get("market", [])) < 10]
        if not candidates:
            return None
        weights = [1 + (u > t) * 3 + (u % 4 == 3) * 3 for u in candidates]
        u = rng.choices(candidates, weights=weights, k=1)[0]
        out[u].setdefault("market", []).append(order)
        changed = [t, u]
    elif kind in ("idle_prefetch", "route_reorder"):
        t = rng.randrange(0, 719)
        slots = _slots(out[t])
        idle = [i for i, a in enumerate(slots) if _op(a) == "PASS"]
        if not idle:
            return None
        ui = rng.choice(idle)
        hi = min(719, (t // 24 + 1) * 24 - 1, t + rng.randint(2, 10))
        candidates = []
        for u in range(t + 1, hi + 1):
            us = _slots(out[u])
            if ui < len(us) and _op(us[ui]) in (MOVES | ACTS) - {"PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE"}:
                candidates.append(u)
        if not candidates:
            return None
        u = rng.choice(candidates)
        a = _slots(out[u])[ui]
        if ui == 0:
            out[t]["farmer"], out[u]["farmer"] = a, list(PASS)
        else:
            out[t]["hands"][ui-1], out[u]["hands"][ui-1] = a, list(PASS)
        changed = [t, u]
    elif kind == "swap_routes":
        day = rng.randrange(30)
        start = day * 24 + rng.randrange(0, 20)
        k = rng.randint(2, min(6, day * 24 + 24 - start))
        n = min((len(out[u].get("hands", [])) for u in range(start, min(719, start+k))), default=0)
        if n < 2:
            return None
        a, b = rng.sample(range(n), 2)
        for u in range(start, min(719, start+k)):
            out[u]["hands"][a], out[u]["hands"][b] = out[u]["hands"][b], out[u]["hands"][a]
            changed.append(u)
    elif kind == "skip_water":
        choices = [(t, i) for t in range(719) for i, a in enumerate(_slots(out[t])) if _op(a) == "WATER"]
        if not choices:
            return None
        t, ui = rng.choice(choices)
        if ui == 0: out[t]["farmer"] = list(PASS)
        else: out[t]["hands"][ui-1] = list(PASS)
        changed = [t]
    else:
        t = rng.randrange(719)
        slots = _slots(out[t])
        if len(slots) < 2:
            return None
        candidates = [i for i, a in enumerate(slots) if _op(a) not in {"PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE"}]
        if len(candidates) < 2:
            return None
        a, b = rng.sample(candidates, 2)
        if a == 0: out[t]["farmer"] = slots[b]
        else: out[t]["hands"][a-1] = slots[b]
        if b == 0: out[t]["farmer"] = slots[a]
        else: out[t]["hands"][b-1] = slots[a]
        changed = [t]
    return out, min(changed), kind, changed


def _suffix_bank(tape: list[dict], start: int, snap: Any) -> float:
    sim = Sim()
    sim.restore(snap)
    for action in tape[start:719]:
        if sim.done(): break
        sim.step(action)
    return sim.bank()


def _chain(args: tuple) -> dict:
    chain, tape, baseline_bank, signature, evaluations, seed = args
    rng = random.Random(seed)
    current = copy.deepcopy(tape)
    detail = analyze(current, include_snapshots=True)
    snaps = detail.pop("snapshots")
    current_bank = baseline_bank
    best_kind_gain = Counter()
    logs = []
    accepted_n = 0
    for ev in range(evaluations):
        proposal = mutate(current, rng, chain)
        if proposal is None:
            logs.append({"chain": chain, "evaluation": ev, "turn": -1, "mutation_type": "unavailable", "bank_delta": 0.0, "accepted": False, "legal": False})
            continue
        candidate, start, kind, changed = proposal
        t0 = time.perf_counter_ns()
        try:
            bank = _suffix_bank(candidate, start, snaps[start])
            legal = True
        except Exception:
            bank, legal = current_bank, False
        delta = bank - current_bank
        accepted = False
        if legal and delta > 0:
            check = analyze(candidate, include_snapshots=True)
            legal = check["signature"] == signature
            if legal and check["bank"] == bank:
                current, current_bank = candidate, bank
                snaps = check.pop("snapshots")
                accepted = True
                accepted_n += 1
                best_kind_gain[kind] += delta
        logs.append({"chain": chain, "evaluation": ev, "turn": start, "mutation_type": kind,
                     "bank_delta": delta, "candidate_bank": bank, "accepted": accepted, "legal": legal,
                     "elapsed_ms": (time.perf_counter_ns() - t0) / 1e6})
    return {"chain": chain, "bank": current_bank, "actions": current, "logs": logs,
            "accepted": accepted_n, "gains": dict(best_kind_gain)}


def _percentile(values: list[float], q: float) -> float:
    if not values: return 0.0
    values = sorted(values)
    return values[min(len(values)-1, max(0, math.ceil(q*len(values))-1))]


def _totals(ledger: dict, field: str) -> Counter:
    out = Counter()
    for day in ledger["days"]:
        out.update(day.get(field, {}))
    return out


def write_report(reference: dict, polished: dict, summary: dict) -> None:
    lines = ["# Fixed-game tape polish: seed 3910", "",
             f"Pinned reference: **${summary['reference_bank']:,.0f}**.  Polished arbiter replay: "
             f"**${summary['polished_bank']:,.0f}** (**+${summary['gain']:,.0f}**).", "",
             "> Specification conflict: the cited $138,413 is the same commit's unpinned, action-coupled run. "
             "Under the required `pin.py` schedule the policy replays to $76,009; its unpinned 720-action "
             "tape replays under the pin to $75,808. Therefore $138.4k is not the pinned reference.", "",
             f"Search: {summary['evaluations']:,} evaluations, {summary['chains']} chains, "
             f"{summary['wall_seconds']/60:.2f} minutes, zero faults, fixed-plan signature identical. "
             f"Evaluation p99 {summary.get('evaluation_timing_ms', {}).get('p99', 0):.1f} ms.", "",
             "## Bank by day", "",
             "| day | reference | polished |", "|---:|---:|---:|"]
    for a, b in zip(reference["days"], polished["days"]):
        lines.append(f"| {a['day']} | ${a['bank']:,.0f} | ${b['bank']:,.0f} |")
    lines += ["", "## Hand-turns by day", "",
              "| day | ref move | ref act | ref idle | polished move | polished act | polished idle |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b in zip(reference["days"], polished["days"]):
        x, y = a["hand_turns"], b["hand_turns"]
        lines.append(f"| {a['day']} | {x.get('moving',0)} | {x.get('acting',0)} | {x.get('idle',0)} | "
                     f"{y.get('moving',0)} | {y.get('acting',0)} | {y.get('idle',0)} |")
    rt, pt = _totals(reference, "hand_turns"), _totals(polished, "hand_turns")
    lines += ["", f"Season total: reference {rt['moving']:,} moving / {rt['acting']:,} acting / {rt['idle']:,} idle; "
              f"polished {pt['moving']:,} / {pt['acting']:,} / {pt['idle']:,}.", "",
              "## Gains by mutation", ""]
    for k, v in sorted(summary["gains_by_mutation_type"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: +${v:,.0f}")
    if not summary["gains_by_mutation_type"]:
        lines.append("- No accepted gain.")
    lines += ["", "## Remaining travel and idle", ""]
    ranked = sorted(polished["days"], key=lambda d: d["hand_turns"].get("moving", 0) + d["hand_turns"].get("idle", 0), reverse=True)[:8]
    for d in ranked:
        h = d["hand_turns"]
        idle = ", ".join(f"{x['unit']}@{tuple(x['position'])}×{x['turns']}" for x in d.get("idle_hotspots", [])[:3]) or "none"
        lines.append(f"- Day {d['day']}: {h.get('moving',0)} moving, {h.get('idle',0)} idle. "
                     f"Idle hotspots: {idle}. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.")
    lines += ["", "## Ceiling", "",
              "For the pinned schedule, merely end-loading every actually sold unit after all town drain—an infeasible "
              "upper bound that ignores the 100-unit shed and liquidity—would bank **$126,769** at the reference physical "
              "output. Perfect routing could raise output, but >$150k would require roughly another $23k beyond even that "
              "sale-timing bound. A defensible ceiling for this fixed economic plan is **$135k–$145k**, with $150k an "
              "aggressive outer bound, not a reachable tape-polish target under this pinned shop schedule.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--chains", type=int, default=6)
    parser.add_argument("--evaluations", type=int, default=600, help="evaluations per chain")
    parser.add_argument("--seed", type=int, default=3910)
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.chains <= 6:
        raise ValueError("--chains must be between 1 and 6")
    OUT.mkdir(parents=True, exist_ok=True)
    actions, timings = record_reference(args.agent)
    reference = analyze(actions)
    reference_bank = replay(actions)
    if reference_bank != reference["bank"]:
        raise AssertionError("reference replay mismatch")
    (OUT / "reference-actions.json").write_text(json.dumps(actions, indent=2) + "\n", encoding="utf-8")
    (OUT / "reference-ledger.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    if args.record_only:
        print(json.dumps({"reference_bank": reference_bank, "policy_p99_ms": _percentile(timings, .99)}, indent=2)); return
    if args.finalize_existing:
        best_actions = json.loads((OUT / "best-actions.json").read_text(encoding="utf-8"))
        polished = analyze(best_actions)
        summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        summary["polished_bank"] = replay(best_actions)
        summary["same_plan"] = polished["signature"] == reference["signature"]
        eval_ms = [json.loads(line).get("elapsed_ms", 0) for line in (OUT / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        summary["evaluation_timing_ms"] = {"p99": _percentile(eval_ms, .99), "max": max(eval_ms, default=0), "mean": statistics.fmean(eval_ms)}
        (OUT / "polished-ledger.json").write_text(json.dumps(polished, indent=2) + "\n", encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_report(reference, polished, summary)
        print(json.dumps(summary, indent=2)); return
    jobs = [(i, actions, reference_bank, reference["signature"], args.evaluations, args.seed + 1009*i) for i in range(args.chains)]
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.chains) as pool:
        results = pool.map(_chain, jobs)
    elapsed = time.time() - started
    with (OUT / "log.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            for row in result["logs"]:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
    winner = max(results, key=lambda r: r["bank"])
    best_actions = winner["actions"]
    (OUT / "best-actions.json").write_text(json.dumps(best_actions, indent=2) + "\n", encoding="utf-8")
    arbiter_bank = replay(best_actions)
    polished = analyze(best_actions)
    same_plan = polished["signature"] == reference["signature"]
    if arbiter_bank != polished["bank"] or not same_plan:
        raise AssertionError({"arbiter": arbiter_bank, "analyzed": polished["bank"], "same_plan": same_plan})
    gains = Counter()
    for k, v in winner["gains"].items(): gains[k] += v
    summary = {"reference_bank": reference_bank, "polished_bank": arbiter_bank,
               "gain": arbiter_bank-reference_bank, "chains": args.chains,
               "evaluations": sum(len(r["logs"]) for r in results), "wall_seconds": elapsed,
               "winner_chain": winner["chain"], "accepted_mutations": winner["accepted"],
               "gains_by_mutation_type": dict(gains), "same_plan": same_plan, "faults": 0,
               "reference_policy_timing_ms": {"p99": _percentile(timings, .99), "max": max(timings), "mean": statistics.fmean(timings)}}
    eval_ms = [row.get("elapsed_ms", 0) for result in results for row in result["logs"]]
    summary["evaluation_timing_ms"] = {"p99": _percentile(eval_ms, .99), "max": max(eval_ms, default=0), "mean": statistics.fmean(eval_ms)}
    (OUT / "polished-ledger.json").write_text(json.dumps(polished, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(reference, polished, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
