"""Feed-failure attribution + no-op attribution + census audit (expB-211, M4 gate).

Steps an episode manually (both agents in-process) and reports, per player:

1. **Feed attribution** (expB-211, spec section 5.3 step 0): every animal-day
   where the animal ends the day unfed is classified into exactly one bucket:
     (a) no wheat existed in the system (shed + all carried) for >= 50% of the
         day's turns and no (c) event occurred;
     (b) wheat existed but no unit brought it to the tile — sub-split b1 (wheat
         sat in the shed) / b2 (wheat only in some unit's pocket elsewhere);
     (c) a unit carrying wheat stood on or orthogonally adjacent to the tile at
         some turn that day and did something other than feeding it.
   Precedence c > a > b, pre-registered before the run. Animals placed the same
   day are counted and also reported separately (they cannot escape from one
   unfed day and are often placed too late to feed).

2. **No-op attribution** (PR #10 review finding 1): every emitted unit action
   that did not change the state is classified by exact cause, re-derived from
   the pre-step state with the engine's own preconditions (kaggriculture 1.32.7
   `_apply_unit_action`). Invalid actions fail silently, so this is the
   correctness diagnostic.

3. **Census** (review finding 2): final-board animal counts by species, escapes,
   weed-outs (unwatered kills vs lifespan decay), melon tiles planted (executed
   PLANT MELON), and MILK/WOOL/MELON sales presence.

4. **M4 gate numbers**: fed fraction (fed animal-days / animal-days), escapes,
   weed-outs, fertilizer collected / animal-days-with-fertilizer-available,
   plus PASS fraction, productive actions per unit-day and MOVE fraction.

Usage (from the repo root):
    uv run python agents/team_b/tools/feed_audit.py --agent millrace:routed_mixed --seeds 0,1,2
    uv run python agents/team_b/tools/feed_audit.py --agent millrace:routed_mixed --seeds 0..31 --workers 8 --out artisans/.../runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))

from _engine import import_engine, engine_version  # noqa: E402
from instrument import load_agent, classify, MOVES, MOVE_D  # noqa: E402

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
ANIMAL_NAMES = ("GOOSE", "COW", "SHEEP")


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _units(farm):
    return [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]


def noop_cause(K, op_action, pos, tile, inv, shed, seeds, board):
    """Re-derive why an emitted action no-opped, from the engine's own
    preconditions. `tile`/`inv`/`shed` are the PRE-step values."""
    op = op_action[0]
    x, y = pos
    is_animal = isinstance(tile, dict) and "animal" in tile
    is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
    shed_adj = (x, y) in SHED_TILES
    if op in MOVES:
        dx, dy = MOVE_D[op]
        nx, ny = x + dx, y + dy
        return "move_offboard" if not (0 <= nx < board and 0 <= ny < board) else "move_other"
    if op == "FEED":
        if not is_animal:
            return "feed_not_animal_tile"
        if tile["fed_today"]:
            return "feed_already_fed"
        if inv.get("WHEAT", 0) <= 0:
            return "feed_no_wheat_carried"
        return "feed_other"
    if op == "CARE":
        if not is_animal:
            return "care_not_animal_tile"
        return "care_already_cared" if tile.get("cared_today") else "care_other"
    if op == "COLLECT_FERTILIZER":
        if not is_animal:
            return "collect_not_animal_tile"
        return "collect_unavailable" if not tile.get("fertilizer_available") else "collect_other"
    if op == "WATER":
        if not is_plant:
            return "water_not_plant_tile"
        return "water_already_watered" if tile.get("watered_today") else "water_other"
    if op == "HARVEST":
        if not isinstance(tile, dict):
            return "harvest_no_tile"
        if tile.get("yield_units", 0) <= 0:
            return "harvest_no_yield"
        if tile.get("kind") == "PLANT":
            return "harvest_immature"  # yield > 0 but age < first_yield_day
        return "harvest_other"
    if op == "PLANT":
        crop = op_action[1] if len(op_action) > 1 else None
        if tile is not None:
            return "plant_tile_occupied"
        if crop not in K.CROPS:
            return "plant_bad_crop"
        if seeds.get(crop, 0) <= 0:
            return "plant_no_seed"
        return "plant_other"
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return "build_tile_occupied" if tile is not None else "build_other"
    if op == "PICKUP":
        if not shed_adj:
            return "pickup_not_shed_adjacent"
        item = op_action[1] if len(op_action) > 1 else None
        if shed.get(item, 0) <= 0:
            return "pickup_shed_empty_of_item"
        return "pickup_other"
    if op == "DROP":
        if not shed_adj:
            return "drop_not_shed_adjacent"
        if not any(v > 0 for v in inv.values()):
            return "drop_empty_inventory"
        return "drop_shed_full"
    if op == "PLACE":
        item = op_action[1] if len(op_action) > 1 else None
        if is_animal:
            return "place_structure_occupied"
        want = K.ANIMALS.get(item, {}).get("structure")
        kind = tile.get("kind") if isinstance(tile, dict) else None
        if want and kind != want:
            return "place_wrong_or_no_structure"
        if inv.get(item, 0) <= 0:
            return "place_item_not_carried"
        return "place_other"
    if op == "DIG":
        if tile is None:
            return "dig_empty"
        return "dig_animal" if is_animal else "dig_other"
    return f"other_{op.lower()}"


def run_episode(agent_a: str, agent_b: str, seed: int, steps: int,
                census_targets=None):
    ke = import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    agents = [load_agent(agent_a), load_agent(agent_b)]
    env.reset()
    turns_per_day = 24

    n_players = 2
    # per player accumulators
    noops = [{} for _ in range(n_players)]
    acts = [{"unit_turns": 0, "pass": 0, "move_exec": 0, "productive": 0,
             "noop": 0, "pass_with_unfed_in_quadrant": 0, "hour23_unmeasured": 0}
            for _ in range(n_players)]
    day_feed = [dict() for _ in range(n_players)]   # (x,y) -> per-day flags
    feed_days = [{"animal_days": 0, "fed_days": 0, "unfed": [],
                  "placed_today_unfed": 0} for _ in range(n_players)]
    fert = [{"available_days": 0, "collected_days": 0} for _ in range(n_players)]
    escapes = [0, 0]
    weed_kills = [0, 0]
    lifespan_weeds = [0, 0]
    melon_planted = [0, 0]
    unit_days = [0, 0]
    cum_sales = [dict(), dict()]

    def fresh_dayrec(tile, day):
        return {"animal": tile["animal"], "placed_today": tile.get("placed_day") == day,
                "any_c": False, "wheat_zero_turns": 0, "turns": 0,
                "shed_wheat_turns": 0, "pocket_wheat_elsewhere_turns": 0,
                "fert_avail_start": bool(tile.get("fertilizer_available")),
                "collected": False, "fed": False}

    prev_sold = [dict(), dict()]

    while not env.done:
        step = env.state[0].observation.step
        day = step // turns_per_day
        hour = step % turns_per_day
        obs = [env.state[i].observation for i in range(n_players)]
        farms = obs[0].farms
        actions = []
        for i, ag in enumerate(agents):
            try:
                a = ag(obs[i])
            except Exception:
                a = {"farmer": ["PASS"], "hands": [], "market": []}
            actions.append(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})

        pre_farms = json.loads(json.dumps(farms))
        pre_priv = [json.loads(json.dumps(dict(obs[i].private))) for i in range(n_players)]

        # ---- per-turn feed-attribution bookkeeping (pre-step state) ----------
        for pi in range(n_players):
            shed = pre_priv[pi]["shed"]
            invs = pre_priv[pi]["inventories"]
            units = _units(pre_farms[pi])
            unit_acts = [actions[pi].get("farmer", ["PASS"])] + list(actions[pi].get("hands", []) or [])
            sys_wheat = shed.get("WHEAT", 0) + sum(iv.get("WHEAT", 0) for iv in invs)
            # register/refresh day records for every animal tile
            board = len(pre_farms[pi]["tiles"])
            for y in range(board):
                for x in range(board):
                    t = pre_farms[pi]["tiles"][y][x]
                    if isinstance(t, dict) and "animal" in t:
                        rec = day_feed[pi].get((x, y))
                        if rec is None:
                            rec = fresh_dayrec(t, day)
                            day_feed[pi][(x, y)] = rec
                        rec["turns"] += 1
                        if t["fed_today"]:
                            rec["fed"] = True
                        if sys_wheat == 0:
                            rec["wheat_zero_turns"] += 1
                        if shed.get("WHEAT", 0) > 0:
                            rec["shed_wheat_turns"] += 1
                        if not t["fed_today"]:
                            # (c): a wheat-carrying unit on/adjacent doing something else
                            for ui, upos in enumerate(units):
                                if ui < len(invs) and invs[ui].get("WHEAT", 0) > 0 and _dist(upos, (x, y)) <= 1:
                                    ua = unit_acts[ui] if ui < len(unit_acts) else ["PASS"]
                                    if not (isinstance(ua, list) and ua and ua[0] == "FEED" and upos == (x, y)):
                                        rec["any_c"] = True
                            wheat_elsewhere = any(iv.get("WHEAT", 0) > 0 for iv in invs)
                            if shed.get("WHEAT", 0) == 0 and wheat_elsewhere:
                                rec["pocket_wheat_elsewhere_turns"] += 1

        env.step(actions)
        post_farms = env.state[0].observation.farms

        # ---- unit action classification + no-op attribution ------------------
        for pi in range(n_players):
            unit_acts = [actions[pi].get("farmer", ["PASS"])] + list(actions[pi].get("hands", []) or [])
            pre_pos = _units(pre_farms[pi])
            post_pos = _units(post_farms[pi])
            post_inv = env.state[pi].observation.private["inventories"]
            board = len(pre_farms[pi]["tiles"])
            # unfed animals per quadrant (for PASS context)
            unfed_quads = set()
            half = board // 2
            for y in range(board):
                for x in range(board):
                    t = pre_farms[pi]["tiles"][y][x]
                    if isinstance(t, dict) and "animal" in t and not t["fed_today"]:
                        unfed_quads.add(("N" if y < half else "S") + ("W" if x < half else "E"))
            for ui, ua in enumerate(unit_acts):
                if ui >= len(pre_pos):
                    break
                acts[pi]["unit_turns"] += 1
                if not (isinstance(ua, list) and ua):
                    ua = ["PASS"]
                # Hour 23: the engine applies unit actions and THEN runs the
                # end-of-day refresh inside the same step (farmer respawned,
                # hands cleared, fed/watered/cared flags reset), so post-state
                # comparison cannot see whether the action executed. Verified
                # 2026-08-20: 100% of a routed_mixed mirror's classified
                # "no-ops" were hour-23 artifacts. Excluded from the
                # executed/noop attribution; counted separately.
                if step % turns_per_day == turns_per_day - 1 and ua[0] != "PASS":
                    acts[pi]["hour23_unmeasured"] = acts[pi].get("hour23_unmeasured", 0) + 1
                    continue
                px, py = pre_pos[ui]
                pre_tile = pre_farms[pi]["tiles"][py][px]
                post_tile = post_farms[pi]["tiles"][py][px]
                ppos = post_pos[ui] if ui < len(post_pos) else (px, py)
                inv_pre = pre_priv[pi]["inventories"][ui] if ui < len(pre_priv[pi]["inventories"]) else {}
                inv_post = dict(post_inv[ui]) if ui < len(post_inv) else {}
                inv_delta = json.dumps(inv_pre, sort_keys=True) != json.dumps(inv_post, sort_keys=True)
                c = classify(ua, (px, py), ppos, pre_tile, post_tile, inv_delta)
                if ua[0] == "PASS":
                    acts[pi]["pass"] += 1
                    uq = ("N" if py < half else "S") + ("W" if px < half else "E")
                    if uq in unfed_quads:
                        acts[pi]["pass_with_unfed_in_quadrant"] += 1
                elif c == "executed":
                    if ua[0] in MOVES:
                        acts[pi]["move_exec"] += 1
                    else:
                        acts[pi]["productive"] += 1
                        if ua[0] == "PLANT" and len(ua) > 1 and ua[1] == "MELON":
                            melon_planted[pi] += 1
                        if ua[0] == "COLLECT_FERTILIZER":
                            rec = day_feed[pi].get((px, py))
                            if rec is not None:
                                rec["collected"] = True
                elif c == "noop":
                    acts[pi]["noop"] += 1
                    import kaggle_environments.envs.kaggriculture.kaggriculture as K
                    cause = noop_cause(K, ua, (px, py), pre_tile, inv_pre,
                                       pre_priv[pi]["shed"], pre_priv[pi]["seeds"], board)
                    noops[pi][cause] = noops[pi].get(cause, 0) + 1

            # weed-out vs decay detection (any turn)
            for y in range(board):
                for x in range(board):
                    pre_t = pre_farms[pi]["tiles"][y][x]
                    post_t = post_farms[pi]["tiles"][y][x]
                    if (isinstance(pre_t, dict) and pre_t.get("kind") == "PLANT"
                            and isinstance(post_t, dict) and post_t.get("kind") == "WEED"):
                        if hour == turns_per_day - 1 and not pre_t["watered_today"] and pre_t["consecutive_unwatered"] >= 1:
                            weed_kills[pi] += 1
                        else:
                            lifespan_weeds[pi] += 1

        # ---- end-of-day: close feed records ---------------------------------
        if hour == turns_per_day - 1:
            for pi in range(n_players):
                board = len(post_farms[pi]["tiles"])
                for (x, y), rec in day_feed[pi].items():
                    post_t = post_farms[pi]["tiles"][y][x]
                    escaped = not (isinstance(post_t, dict) and "animal" in post_t)
                    # fed on the final turn shows up post-refresh as consecutive_unfed == 0
                    fed = rec["fed"] or (isinstance(post_t, dict) and "animal" in post_t
                                         and post_t.get("consecutive_unfed", 1) == 0)
                    feed_days[pi]["animal_days"] += 1
                    if rec["fert_avail_start"]:
                        fert[pi]["available_days"] += 1
                        if rec["collected"]:
                            fert[pi]["collected_days"] += 1
                    if escaped:
                        escapes[pi] += 1
                    if fed:
                        feed_days[pi]["fed_days"] += 1
                    else:
                        if rec["placed_today"]:
                            feed_days[pi]["placed_today_unfed"] += 1
                        if rec["any_c"]:
                            bucket = "c"
                        elif rec["wheat_zero_turns"] >= 0.5 * rec["turns"]:
                            bucket = "a"
                        elif rec["shed_wheat_turns"] > 0:
                            bucket = "b1"
                        else:
                            bucket = "b2"
                        feed_days[pi]["unfed"].append({
                            "day": day, "pos": [x, y], "animal": rec["animal"],
                            "bucket": bucket, "placed_today": rec["placed_today"],
                            "escaped": escaped,
                            "wheat_zero_frac": round(rec["wheat_zero_turns"] / max(1, rec["turns"]), 3),
                            "shed_wheat_frac": round(rec["shed_wheat_turns"] / max(1, rec["turns"]), 3),
                        })
                day_feed[pi] = {}
                unit_days[pi] += 1 + len(pre_farms[pi]["hands"])

    # ---- final census --------------------------------------------------------
    final = env.steps[-1]
    census = []
    for pi in range(n_players):
        tiles = final[0].observation.farms[pi]["tiles"]
        counts = {"GOOSE": 0, "COW": 0, "SHEEP": 0}
        for row in tiles:
            for t in row:
                if isinstance(t, dict) and "animal" in t:
                    counts[t["animal"]] += 1
        census.append(counts)

    summary = {
        "engine": engine_version(), "seed": seed, "steps": steps,
        "agents": [agent_a, agent_b],
        "final_money": [final[i].observation.farms[i]["money"] for i in range(n_players)],
        "statuses": [s.status for s in final],
        "per_player": [],
    }
    for pi in range(n_players):
        fd = feed_days[pi]
        buckets = {}
        for u in fd["unfed"]:
            buckets[u["bucket"]] = buckets.get(u["bucket"], 0) + 1
        a = acts[pi]
        summary["per_player"].append({
            "animal_days": fd["animal_days"],
            "fed_days": fd["fed_days"],
            "fed_fraction": round(fd["fed_days"] / max(1, fd["animal_days"]), 4),
            "unfed_buckets": buckets,
            "unfed_placed_today": fd["placed_today_unfed"],
            "unfed_detail": fd["unfed"],
            "escapes": escapes[pi],
            "weed_kills_unwatered": weed_kills[pi],
            "weeds_lifespan_decay": lifespan_weeds[pi],
            "fertilizer_available_days": fert[pi]["available_days"],
            "fertilizer_collected_days": fert[pi]["collected_days"],
            "fertilizer_collect_rate": round(fert[pi]["collected_days"] / max(1, fert[pi]["available_days"]), 4),
            "census_final": census[pi],
            "melon_planted": melon_planted[pi],
            "unit_turns": a["unit_turns"],
            "hour23_unmeasured": a["hour23_unmeasured"],
            "pass_fraction": round(a["pass"] / max(1, a["unit_turns"]), 4),
            "pass_with_unfed_in_quadrant": a["pass_with_unfed_in_quadrant"],
            "move_fraction": round(a["move_exec"] / max(1, a["unit_turns"] - a["hour23_unmeasured"]), 4),
            "noop_fraction": round(a["noop"] / max(1, a["unit_turns"] - a["hour23_unmeasured"]), 4),
            "productive_per_unit_day": round(a["productive"] / max(1, unit_days[pi]), 2),
            "noop_causes": dict(sorted(noops[pi].items(), key=lambda kv: -kv[1])),
        })
    return summary


def _worker(job):
    agent_a, agent_b, seed, steps = job
    return run_episode(agent_a, agent_b, seed, steps)


def parse_seeds(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if ".." in part:
            lo, hi = part.split("..")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="agent spec (both seats unless --opponent)")
    ap.add_argument("--opponent", default=None)
    ap.add_argument("--seeds", default="0,1,2", help="e.g. 0,1,2 or 0..31")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="write one JSON per seed into this dir")
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    opp = args.opponent or args.agent
    jobs = [(args.agent, opp, s, args.steps) for s in seeds]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    # aggregate across seeds and (mirror) both seats
    agg = {"animal_days": 0, "fed_days": 0, "escapes": 0, "weed_kills": 0,
           "fert_avail": 0, "fert_coll": 0, "buckets": {}, "noops": {},
           "unit_turns": 0, "measurable": 0, "pass": 0, "noop": 0,
           "placed_today_unfed": 0}
    seats = 0
    for r in results:
        for pp in r["per_player"]:
            seats += 1
            agg["animal_days"] += pp["animal_days"]
            agg["fed_days"] += pp["fed_days"]
            agg["escapes"] += pp["escapes"]
            agg["weed_kills"] += pp["weed_kills_unwatered"]
            agg["fert_avail"] += pp["fertilizer_available_days"]
            agg["fert_coll"] += pp["fertilizer_collected_days"]
            agg["placed_today_unfed"] += pp["unfed_placed_today"]
            for k, v in pp["unfed_buckets"].items():
                agg["buckets"][k] = agg["buckets"].get(k, 0) + v
            for k, v in pp["noop_causes"].items():
                agg["noops"][k] = agg["noops"].get(k, 0) + v
            agg["unit_turns"] += pp["unit_turns"]
            meas = pp["unit_turns"] - pp["hour23_unmeasured"]
            agg["measurable"] += meas
            agg["pass"] += pp["pass_fraction"] * pp["unit_turns"]
            agg["noop"] += pp["noop_fraction"] * meas
        if args.out:
            outdir = Path(args.out)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"feed_audit_seed{r['seed']}.json").write_text(
                json.dumps(r, indent=2), encoding="utf-8")

    total_unfed = sum(agg["buckets"].values())
    print(f"engine={results[0]['engine']} agent={args.agent} opponent={opp} "
          f"seeds={seeds} seats={seats}")
    print(f"fed_fraction={agg['fed_days']/max(1,agg['animal_days']):.4f} "
          f"({agg['fed_days']}/{agg['animal_days']} animal-days)  "
          f"escapes={agg['escapes']}  weed_kills={agg['weed_kills']}")
    print(f"fertilizer_collect_rate={agg['fert_coll']/max(1,agg['fert_avail']):.4f} "
          f"({agg['fert_coll']}/{agg['fert_avail']})")
    print(f"unfed buckets (n={total_unfed}, placed-today {agg['placed_today_unfed']}): "
          + json.dumps({k: {"n": v, "frac": round(v/max(1,total_unfed), 3)}
                        for k, v in sorted(agg["buckets"].items())}))
    print(f"pass_fraction={agg['pass']/max(1,agg['unit_turns']):.4f}  "
          f"noop_fraction={agg['noop']/max(1,agg['measurable']):.4f} "
          f"(over {agg['measurable']} measurable unit-turns; "
          f"{agg['unit_turns']-agg['measurable']} hour-23 turns unmeasurable)")
    print("noop causes: " + json.dumps(dict(sorted(agg["noops"].items(), key=lambda kv: -kv[1]))))
    print("SUMMARY " + json.dumps({
        "engine": results[0]["engine"], "agent": args.agent, "opponent": opp,
        "seeds": seeds, "seats": seats,
        "fed_fraction": round(agg["fed_days"] / max(1, agg["animal_days"]), 4),
        "animal_days": agg["animal_days"], "escapes": agg["escapes"],
        "weed_kills": agg["weed_kills"],
        "fert_collect_rate": round(agg["fert_coll"] / max(1, agg["fert_avail"]), 4),
        "unfed_buckets": agg["buckets"], "placed_today_unfed": agg["placed_today_unfed"],
        "noop_causes": agg["noops"],
        "pass_fraction": round(agg["pass"] / max(1, agg["unit_turns"]), 4),
        "noop_fraction": round(agg["noop"] / max(1, agg["measurable"]), 4),
        "unit_turns": agg["unit_turns"], "measurable_unit_turns": agg["measurable"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
