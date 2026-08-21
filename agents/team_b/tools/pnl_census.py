"""expB-301 — the P&L census: a per-line income statement and terminal balance
sheet for a Kaggriculture episode.

Design: `workshop/deliverables/2026-08-21-team-b-business-lens/business-lens.md`
sections 2.4 and 5 (expB-301). This is a *diagnostic*: it changes no agent code
and gates the rest of the business-lens slate.

What it measures, per player, per season:

  revenue        exact SELL commits (units + realised dollars) per product
  purchased      exact BUY_PRODUCT / BUY_SEED / BUY_ANIMAL commits (units +
                 realised dollars), plus HIRE and BUY_LAND money deltas
  production     units actually issued by HARVEST (read off the tile *before*
                 the harvest lands, so it is exact, not inferred from sales)
  tile-days      every owned tile sampled at hour 0 of every day, keyed by the
                 line occupying it (crop / animal / empty structure / weed /
                 fallow / locked)
  unit-turns     every farmer/hand action classified into
                 produce | maintain | invest | logistics | idle,
                 attributed to the line whose tile it touched, with an exact
                 executed/no-op verdict
  terminal       cash, shed + carried inventory walked down the sell curve,
                 unharvested pipeline, unused seeds at cost, and the capital
                 (land + animals) that day 29 writes to zero

Two engine facts drive the instrumentation choices:

  1. Hour-23 unit actions execute and are then followed by the end-of-day
     refresh inside the same `env.step`, so a pre/post *state diff* taken
     around `env.step` misclassifies every hour-23 action (memory:
     "NEVER classify hour-23 actions by state diff"). This tool therefore
     patches `_apply_unit_action` itself and diffs the tile across that single
     call — correct at every hour, hour 23 included.
  2. Unit actions run BEFORE `_process_market` in `interpreter`, so the
     farm->player identity map cannot be built from a `_process_market` patch
     (which is what tools/instrument.py does). We wrap `env.interpreter`
     instead and rebuild the map at the top of every step.

Usage (from the repo root):

    uv run python agents/team_b/tools/pnl_census.py \
        --a millrace:melonrace --b millrace:melonrace \
        --seeds 0-7 --out-dir ../bench/runs/expB-301/mirror

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _engine import import_engine, engine_version  # noqa: E402
from instrument import load_agent, arity_wrap, curve_value  # noqa: E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}

BUCKET = {
    "HARVEST": "produce", "COLLECT_FERTILIZER": "produce",
    "FEED": "maintain", "WATER": "maintain", "CARE": "maintain", "FERTILIZE": "maintain",
    "PLANT": "invest", "BUILD_COOP": "invest", "BUILD_PASTURE": "invest",
    "PLACE": "invest", "DIG": "invest",
    "PICKUP": "logistics", "DROP": "logistics",
    "NORTH": "logistics", "SOUTH": "logistics", "EAST": "logistics", "WEST": "logistics",
    "PASS": "idle",
}
PRODUCTIVE = ("produce", "maintain", "invest")

# line -> the product it sells
LINE_PRODUCT = {
    "GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL",
    "WHEAT": "WHEAT", "CARROT": "CARROT", "TOMATO": "TOMATO",
    "STRAWBERRY": "STRAWBERRY", "MELON": "MELON",
    "FERTILIZER": "FERTILIZER",
}
ANIMAL_LINES = ("GOOSE", "COW", "SHEEP")
CROP_LINES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")


def _blank_line():
    return {
        "revenue": 0.0, "units_sold": 0, "units_produced": 0,
        "seed_spend": 0.0, "seed_units": 0, "capex": 0.0, "capex_units": 0,
        "tile_days": 0,
        "acts": {b: 0 for b in ("produce", "maintain", "invest")},
        "acts_noop": 0,
    }


def run_pnl(spec_a: str, spec_b: str, seed: int, steps: int = 720):
    ke = import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    turns_per_day = 24
    farm_ids: dict[int, int] = {}

    P = [{
        "sales": {p: {"units": 0, "dollars": 0.0} for p in K.PRODUCTS},
        "buy_product": {p: {"units": 0, "dollars": 0.0} for p in ("WHEAT", "FERTILIZER")},
        "buy_product_prices": {"WHEAT": [], "FERTILIZER": []},
        "buy_seed": {c: {"units": 0, "dollars": 0.0} for c in K.CROPS},
        "buy_animal": {a: {"units": 0, "dollars": 0.0} for a in K.ANIMALS},
        "hire": {"count": 0, "dollars": 0.0},
        "land": {"count": 0, "dollars": 0.0},
        "refused": {"buy_animal_shed_full": 0, "buy_product_shed_full": 0, "cash": 0},
        "harvest_units": {p: 0 for p in K.PRODUCTS},
        "feed_events": {a: 0 for a in K.ANIMALS},
        "fert_collected": 0,
        "fertilize_on": {c: 0 for c in K.CROPS},
        "acts": {b: {"executed": 0, "noop": 0} for b in
                 ("produce", "maintain", "invest", "logistics", "idle", "unknown")},
        "acts_by_op": {},
        "unit_turns": 0,
        "lines": {},
        "tile_days_other": {"EMPTY_STRUCT": 0, "WEED": 0, "FALLOW": 0, "LOCKED": 0},
        "daily_tiles": [],
        "hands_by_day": [],
    } for _ in range(2)]
    for pl in P:
        for ln in list(LINE_PRODUCT) :
            pl["lines"][ln] = _blank_line()

    orig_commit = K._commit_unit
    orig_hire = K._do_hire
    orig_land = K._do_buy_land
    orig_apply = K._apply_unit_action

    def patched_commit(op, item, price, farm, private, market, shed_capacity=100):
        pid = farm_ids.get(id(farm), -1)
        shed_full = sum(private["shed"].values()) >= shed_capacity
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if pid < 0:
            return ok
        if not ok:
            if op == "BUY_ANIMAL" and shed_full:
                P[pid]["refused"]["buy_animal_shed_full"] += 1
            elif op == "BUY_PRODUCT" and shed_full:
                P[pid]["refused"]["buy_product_shed_full"] += 1
            elif op in ("BUY_ANIMAL", "BUY_PRODUCT", "BUY_SEED"):
                P[pid]["refused"]["cash"] += 1
            return ok
        if op == "SELL":
            P[pid]["sales"][item]["units"] += 1
            P[pid]["sales"][item]["dollars"] += price
        elif op == "BUY_PRODUCT":
            P[pid]["buy_product"][item]["units"] += 1
            P[pid]["buy_product"][item]["dollars"] += price
            P[pid]["buy_product_prices"][item].append(price)
        elif op == "BUY_SEED":
            P[pid]["buy_seed"][item]["units"] += 1
            P[pid]["buy_seed"][item]["dollars"] += price
        elif op == "BUY_ANIMAL":
            P[pid]["buy_animal"][item]["units"] += 1
            P[pid]["buy_animal"][item]["dollars"] += price
        return ok

    def patched_hire(farm, private, board_size, mult=1):
        pid = farm_ids.get(id(farm), -1)
        before = farm["money"]
        r = orig_hire(farm, private, board_size, mult)
        if pid >= 0 and farm["money"] < before:
            P[pid]["hire"]["count"] += 1
            P[pid]["hire"]["dollars"] += before - farm["money"]
        return r

    def patched_land(farm, board_size):
        pid = farm_ids.get(id(farm), -1)
        before = farm["money"]
        r = orig_land(farm, board_size)
        if pid >= 0 and farm["money"] < before:
            P[pid]["land"]["count"] += 1
            P[pid]["land"]["dollars"] += before - farm["money"]
        return r

    def _line_of_tile(tile):
        if not isinstance(tile, dict):
            return None
        if "animal" in tile:
            return tile["animal"]
        if tile.get("kind") == "PLANT":
            return tile.get("crop")
        return None

    def patched_apply(farm, private, idx, action, board_size, day, turns_per_day_, shed_capacity=100):
        pid = farm_ids.get(id(farm), -1)
        if pid < 0:
            return orig_apply(farm, private, idx, action, board_size, day, turns_per_day_, shed_capacity)
        st = P[pid]
        st["unit_turns"] += 1
        op = action[0] if isinstance(action, list) and action else "PASS"
        bucket = BUCKET.get(op, "unknown")
        pos = K._farmer_position(farm, idx)
        px, py = (pos[0], pos[1]) if pos else (-1, -1)
        live_tile = farm["tiles"][py][px] if pos else None
        # NB: tiles are mutated IN PLACE by _apply_unit_action, so the "pre"
        # snapshot must be a copy or every in-place op reads back as a no-op
        # (this bug zeroed FEED/WATER/CARE on the first cut of this tool).
        # Tile values are scalars, so a shallow dict() is a full snapshot.
        pre_tile = dict(live_tile) if isinstance(live_tile, dict) else live_tile
        invs = private.get("inventories", [])
        pre_inv = dict(invs[idx]) if idx < len(invs) else {}
        pre_seeds = dict(private.get("seeds", {}))
        pre_shed = dict(private.get("shed", {}))
        pre_yield = pre_tile.get("yield_units", 0) if isinstance(pre_tile, dict) else 0

        r = orig_apply(farm, private, idx, action, board_size, day, turns_per_day_, shed_capacity)

        post_pos = K._farmer_position(farm, idx)
        post_tile = farm["tiles"][py][px] if pos else None
        invs = private.get("inventories", [])
        post_inv = dict(invs[idx]) if idx < len(invs) else {}
        if op == "PASS":
            executed = True
        elif op in MOVES:
            executed = bool(post_pos) and (post_pos[0], post_pos[1]) != (px, py)
        elif op in ("PICKUP", "DROP"):
            executed = (post_inv != pre_inv) or (dict(private.get("shed", {})) != pre_shed)
        elif op == "PLACE":
            executed = (post_inv != pre_inv) or (dict(private.get("shed", {})) != pre_shed)
            if isinstance(post_tile, dict) and "animal" in post_tile \
                    and not (isinstance(pre_tile, dict) and "animal" in pre_tile):
                executed = True
        elif op == "PLANT":
            executed = dict(private.get("seeds", {})) != pre_seeds
        elif op == "HARVEST":
            executed = post_inv != pre_inv
        else:
            a = pre_tile if not isinstance(pre_tile, dict) else dict(pre_tile)
            b = post_tile if not isinstance(post_tile, dict) else dict(post_tile)
            executed = a != b

        st["acts"][bucket]["executed" if executed else "noop"] += 1
        key = f"{op}:{'ok' if executed else 'noop'}"
        st["acts_by_op"][key] = st["acts_by_op"].get(key, 0) + 1

        if not executed:
            return r

        # ---- line attribution (executed actions only) -----------------------
        line = None
        if op == "COLLECT_FERTILIZER":
            line = "FERTILIZER"
            st["fert_collected"] += 1
        elif op == "FERTILIZE":
            crop = _line_of_tile(pre_tile)
            if crop in K.CROPS:
                st["fertilize_on"][crop] += 1
            line = crop
        elif op == "PLANT":
            line = action[1] if len(action) >= 2 else None
        elif op == "PLACE":
            line = action[1] if len(action) >= 2 and action[1] in K.ANIMALS else None
        elif op in ("BUILD_COOP", "BUILD_PASTURE"):
            line = None  # structure: charged to the animal that lands on it
        elif op == "FEED":
            line = _line_of_tile(pre_tile)
            if line in K.ANIMALS:
                st["feed_events"][line] += 1
        elif op == "HARVEST":
            line = _line_of_tile(pre_tile)
            if isinstance(pre_tile, dict) and "animal" in pre_tile:
                prod = K.ANIMALS[pre_tile["animal"]]["product"]
            else:
                prod = pre_tile.get("crop") if isinstance(pre_tile, dict) else None
            if prod in st["harvest_units"]:
                st["harvest_units"][prod] += pre_yield
        else:
            line = _line_of_tile(pre_tile)

        if line in st["lines"] and bucket in PRODUCTIVE:
            st["lines"][line]["acts"][bucket] += 1
        return r

    K._commit_unit = patched_commit
    K._do_hire = patched_hire
    K._do_buy_land = patched_land
    K._apply_unit_action = patched_apply
    try:
        env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
        orig_interpreter = env.interpreter

        def wrapped_interpreter(state, env_):
            farm_ids.clear()
            farms = getattr(state[0].observation, "farms", None)
            if farms:
                for i, f in enumerate(farms):
                    farm_ids[id(f)] = i
            return orig_interpreter(state, env_)

        env.interpreter = wrapped_interpreter
        agents = [arity_wrap(load_agent(spec_a), env.configuration),
                  arity_wrap(load_agent(spec_b), env.configuration)]
        env.reset()

        while not env.done:
            step = env.state[0].observation.step
            obs = [env.state[i].observation for i in range(2)]
            # tile-day census: sample every owned tile at hour 0 of each day
            if step % turns_per_day == 12:
                # hands are cleared every night and re-hired during the day, so
                # sample the crew at mid-day, not at hour 0 (which reads zero).
                for pi in range(2):
                    P[pi]["hands_by_day"].append(len(env.state[0].observation.farms[pi]["hands"]))
            if step % turns_per_day == 0:
                farms = env.state[0].observation.farms
                for pi in range(2):
                    st = P[pi]
                    counts = {}
                    for row in farms[pi]["tiles"]:
                        for t in row:
                            if t == "LOCKED":
                                st["tile_days_other"]["LOCKED"] += 1
                                continue
                            if t is None:
                                st["tile_days_other"]["FALLOW"] += 1
                                continue
                            if isinstance(t, dict):
                                if "animal" in t:
                                    st["lines"][t["animal"]]["tile_days"] += 1
                                    counts[t["animal"]] = counts.get(t["animal"], 0) + 1
                                elif t.get("kind") == "PLANT":
                                    c = t.get("crop")
                                    if c in st["lines"]:
                                        st["lines"][c]["tile_days"] += 1
                                    counts[c] = counts.get(c, 0) + 1
                                elif t.get("kind") == "WEED":
                                    st["tile_days_other"]["WEED"] += 1
                                else:
                                    st["tile_days_other"]["EMPTY_STRUCT"] += 1
                    st["daily_tiles"].append({"day": step // turns_per_day, "counts": counts})

            actions = []
            for i, ag in enumerate(agents):
                try:
                    a = ag(obs[i])
                except Exception:
                    a = {"farmer": ["PASS"], "hands": [], "market": []}
                actions.append(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})
            env.step(actions)

        # ---- terminal balance sheet -----------------------------------------
        final_farms = env.state[0].observation.farms
        minv = dict(env.state[0].observation.market["inventory"])
        terminal = []
        for pi in range(2):
            priv = env.state[pi].observation.private
            carried = {}
            for iv in priv["inventories"]:
                for it, n in iv.items():
                    carried[it] = carried.get(it, 0) + n
            shed = {k: v for k, v in priv["shed"].items() if v > 0}
            unharv = {}
            for row in final_farms[pi]["tiles"]:
                for t in row:
                    if isinstance(t, dict) and t.get("yield_units", 0) > 0:
                        prod = (K.ANIMALS[t["animal"]]["product"] if "animal" in t else t.get("crop"))
                        if prod:
                            unharv[prod] = unharv.get(prod, 0) + t["yield_units"]
            liq_inv = dict(minv)
            val = lambda d: round(sum(curve_value(K, liq_inv, it, n)
                                      for it, n in d.items() if it in K.PRODUCTS), 0)
            seeds_left = {k: v for k, v in priv["seeds"].items() if v > 0}
            terminal.append({
                "cash": final_farms[pi]["money"],
                "shed": shed, "shed_value": val(shed),
                "carried": carried, "carried_value": val(carried),
                "unharvested": unharv, "unharvested_value": val(unharv),
                "seeds_left": seeds_left,
                "seeds_left_cost": sum(K.CROPS[c]["seed"] * n for c, n in seeds_left.items()),
                "capital_written_off": round(
                    P[pi]["land"]["dollars"] + sum(v["dollars"] for v in P[pi]["buy_animal"].values()), 0),
                "animals_alive": sum(1 for row in final_farms[pi]["tiles"] for t in row
                                     if isinstance(t, dict) and "animal" in t),
            })

        final = env.steps[-1]
        money = [final[i].observation.farms[i]["money"] for i in range(2)]
        out = {"engine": engine_version(), "seed": seed, "steps": steps,
               "agents": [spec_a, spec_b], "final_money": money,
               "statuses": [s.status for s in final],
               "players": []}
        for pi in range(2):
            st = P[pi]
            st.pop("daily_tiles")
            st["terminal"] = terminal[pi]
            out["players"].append(st)
        return out
    finally:
        K._commit_unit = orig_commit
        K._do_hire = orig_hire
        K._do_buy_land = orig_land
        K._apply_unit_action = orig_apply


# =========================== income statement ================================

def income_statement(st: dict) -> dict:
    """Turn one player's raw census into a per-line income statement.

    Feed is charged at the season's realised average *purchased* wheat price,
    allocated across FEED events (the marginal source of feed is the market:
    we grow every wheat tile we can and buy the rest). Two views are produced:
      charged_bill   — the actual purchased-wheat bill, split pro-rata by FEED
      charged_margin — every FEED event valued at the average purchase price
                       (i.e. grown wheat charged at its opportunity cost)
    """
    lines = {ln: dict(v) for ln, v in st["lines"].items()}
    gross_revenue = sum(v["dollars"] for v in st["sales"].values())

    # revenue / units sold onto lines
    for ln, prod in LINE_PRODUCT.items():
        lines[ln]["revenue"] = st["sales"][prod]["dollars"]
        lines[ln]["units_sold"] = st["sales"][prod]["units"]
        lines[ln]["units_produced"] = st["harvest_units"][prod]
    lines["FERTILIZER"]["units_produced"] = st["fert_collected"]

    # direct purchased inputs
    for c in CROP_LINES:
        lines[c]["seed_spend"] = st["buy_seed"][c]["dollars"]
        lines[c]["seed_units"] = st["buy_seed"][c]["units"]
    for a in ANIMAL_LINES:
        lines[a]["capex"] = st["buy_animal"][a]["dollars"]
        lines[a]["capex_units"] = st["buy_animal"][a]["units"]

    wheat_bill = st["buy_product"]["WHEAT"]["dollars"]
    wheat_units_bought = st["buy_product"]["WHEAT"]["units"]
    avg_buy = (wheat_bill / wheat_units_bought) if wheat_units_bought else 0.0
    feed_total = sum(st["feed_events"].values())
    for a in ANIMAL_LINES:
        share = (st["feed_events"][a] / feed_total) if feed_total else 0.0
        lines[a]["feed_charged_bill"] = round(wheat_bill * share, 0)
        lines[a]["feed_charged_margin"] = round(avg_buy * st["feed_events"][a], 0)
        lines[a]["feed_events"] = st["feed_events"][a]

    # the wheat line is internal: it "sells" its harvest to the animals at the
    # market buy price it displaces. Reported as memo revenue, never banked.
    wheat_grown = st["harvest_units"]["WHEAT"]
    lines["WHEAT"]["internal_transfer_revenue"] = round(
        avg_buy * max(0, wheat_grown - st["sales"]["WHEAT"]["units"]), 0)

    for ln, v in lines.items():
        acts = v["acts"]
        v["productive_actions"] = sum(acts.values())
        direct = v.get("seed_spend", 0) + v.get("feed_charged_bill", 0)
        v["contribution"] = round(v["revenue"] - direct, 0)
        v["contribution_after_capex"] = round(v["contribution"] - v.get("capex", 0), 0)
        v["contrib_per_tile_day"] = round(v["contribution"] / v["tile_days"], 1) if v["tile_days"] else None
        v["contrib_per_action"] = round(v["contribution"] / v["productive_actions"], 1) \
            if v["productive_actions"] else None
        v["rev_per_tile_day"] = round(v["revenue"] / v["tile_days"], 1) if v["tile_days"] else None
        v["rev_per_action"] = round(v["revenue"] / v["productive_actions"], 1) \
            if v["productive_actions"] else None

    total_spend = (wheat_bill + st["buy_product"]["FERTILIZER"]["dollars"]
                   + sum(v["dollars"] for v in st["buy_seed"].values())
                   + sum(v["dollars"] for v in st["buy_animal"].values())
                   + st["hire"]["dollars"] + st["land"]["dollars"])
    acts = st["acts"]
    measured = sum(v["executed"] + v["noop"] for v in acts.values())
    prod_acts = sum(acts[b]["executed"] for b in PRODUCTIVE)
    return {
        "lines": lines,
        "gross_revenue": round(gross_revenue, 0),
        "purchased_wheat_dollars": round(wheat_bill, 0),
        "purchased_wheat_units": wheat_units_bought,
        "purchased_wheat_avg_price": round(avg_buy, 2),
        "purchased_wheat_pct_of_gross": round(100.0 * wheat_bill / gross_revenue, 2) if gross_revenue else None,
        "total_spend": round(total_spend, 0),
        "net_margin_pct": round(100.0 * (gross_revenue - total_spend) / gross_revenue, 1) if gross_revenue else None,
        "hire_dollars": round(st["hire"]["dollars"], 0),
        "hire_count": st["hire"]["count"],
        "land_dollars": round(st["land"]["dollars"], 0),
        "seed_dollars": round(sum(v["dollars"] for v in st["buy_seed"].values()), 0),
        "animal_dollars": round(sum(v["dollars"] for v in st["buy_animal"].values()), 0),
        "fertilizer_bought_dollars": round(st["buy_product"]["FERTILIZER"]["dollars"], 0),
        "unit_turns": st["unit_turns"],
        "acts_by_op": st["acts_by_op"],
        "activity": {b: acts[b]["executed"] for b in acts},
        "activity_noop": {b: acts[b]["noop"] for b in acts},
        "activity_frac": {b: round(acts[b]["executed"] / measured, 4) for b in acts} if measured else {},
        "productive_actions": prod_acts,
        "rev_per_unit_turn": round(gross_revenue / st["unit_turns"], 2) if st["unit_turns"] else None,
        "rev_per_productive_action": round(gross_revenue / prod_acts, 2) if prod_acts else None,
        "refused": st["refused"],
        "terminal": st["terminal"],
        "hands_mean": round(statistics.fmean(st["hands_by_day"]), 2) if st["hands_by_day"] else None,
        "tile_days_other": st["tile_days_other"],
    }


# ================================ battery ====================================

def parse_seeds(txt):
    out = []
    for part in txt.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _one(job):
    spec_a, spec_b, seed, steps, out_dir = job
    import pnl_census
    raw = pnl_census.run_pnl(spec_a, spec_b, seed, steps)
    stmts = [pnl_census.income_statement(raw["players"][i]) for i in range(2)]
    rec = {"seed": seed, "agents": raw["agents"], "engine": raw["engine"],
           "final_money": raw["final_money"], "statuses": raw["statuses"],
           "statements": stmts}
    Path(out_dir, f"pnl_seed{seed}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def _agg(vals, nd=1):
    if not vals:
        return None
    return {"mean": round(statistics.fmean(vals), nd),
            "std": round(statistics.pstdev(vals), nd) if len(vals) > 1 else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--swap-half", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, s in enumerate(seeds):
        swap = args.swap_half and i >= len(seeds) // 2
        a, b = (args.b, args.a) if swap else (args.a, args.b)
        jobs.append((a, b, s, args.steps, str(out_dir)))

    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            rows = pool.map(_one, jobs)
    else:
        rows = [_one(j) for j in jobs]

    # aggregate by agent spec
    by_spec: dict[str, list] = {}
    for r in rows:
        for seat, sp in enumerate(r["agents"]):
            by_spec.setdefault(sp, []).append(r["statements"][seat])

    report = {"engine": rows[0]["engine"], "seeds": seeds, "n_episodes": len(rows),
              "per_agent": {}}
    for sp, stmts in by_spec.items():
        top = {}
        for k in ("gross_revenue", "purchased_wheat_dollars", "purchased_wheat_units",
                  "purchased_wheat_avg_price", "purchased_wheat_pct_of_gross", "total_spend",
                  "net_margin_pct", "hire_dollars", "hire_count", "land_dollars", "seed_dollars",
                  "animal_dollars", "fertilizer_bought_dollars", "unit_turns",
                  "productive_actions", "rev_per_unit_turn", "rev_per_productive_action",
                  "hands_mean"):
            top[k] = _agg([s[k] for s in stmts if s.get(k) is not None])
        top["bank"] = _agg([s["terminal"]["cash"] for s in stmts])
        top["activity"] = {b: _agg([s["activity"][b] for s in stmts]) for b in stmts[0]["activity"]}
        top["activity_noop"] = {b: _agg([s["activity_noop"][b] for s in stmts]) for b in stmts[0]["activity_noop"]}
        top["activity_frac"] = {b: _agg([s["activity_frac"][b] for s in stmts], 4)
                                for b in stmts[0]["activity_frac"]}
        top["acts_by_op"] = {}
        for k in sorted({k for s in stmts for k in s["acts_by_op"]}):
            top["acts_by_op"][k] = _agg([s["acts_by_op"].get(k, 0) for s in stmts])
        top["wheat_internal_transfer"] = _agg(
            [s["lines"]["WHEAT"]["internal_transfer_revenue"] for s in stmts])
        top["wheat_resold_from_stock"] = _agg(
            [max(0, s["lines"]["WHEAT"]["units_sold"] - s["lines"]["WHEAT"]["units_produced"])
             for s in stmts])
        top["refused"] = {k: _agg([s["refused"][k] for s in stmts]) for k in stmts[0]["refused"]}
        top["tile_days_other"] = {k: _agg([s["tile_days_other"][k] for s in stmts])
                                  for k in stmts[0]["tile_days_other"]}
        top["terminal"] = {k: _agg([s["terminal"][k] for s in stmts])
                           for k in ("cash", "shed_value", "carried_value", "unharvested_value",
                                     "seeds_left_cost", "capital_written_off", "animals_alive")}
        ln_out = {}
        for ln in LINE_PRODUCT:
            keys = ["revenue", "units_sold", "units_produced", "tile_days", "productive_actions",
                    "contribution", "contribution_after_capex", "seed_spend", "capex",
                    "contrib_per_tile_day", "contrib_per_action", "rev_per_tile_day", "rev_per_action"]
            if ln in ANIMAL_LINES:
                keys += ["feed_charged_bill", "feed_charged_margin", "feed_events"]
            d = {}
            for k in keys:
                vals = [s["lines"][ln].get(k) for s in stmts]
                vals = [v for v in vals if v is not None]
                d[k] = _agg(vals)
            d["acts"] = {b: _agg([s["lines"][ln]["acts"][b] for s in stmts])
                         for b in ("produce", "maintain", "invest")}
            tot_u = sum(s["lines"][ln]["units_sold"] for s in stmts)
            tot_r = sum(s["lines"][ln]["revenue"] for s in stmts)
            d["dollars_per_unit"] = round(tot_r / tot_u, 2) if tot_u else None
            ln_out[ln] = d
        top["lines"] = ln_out
        report["per_agent"][sp] = top

    Path(out_dir, "pnl_aggregate.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print_report(report)
    return 0


def print_report(report):
    print(f"engine {report['engine']}  n={report['n_episodes']} episodes  seeds "
          f"{report['seeds'][0]}..{report['seeds'][-1]}")
    for sp, t in report["per_agent"].items():
        print(f"\n================ {sp}")
        print(f"  bank {t['bank']['mean']:>9,.0f} +- {t['bank']['std']:,.0f}   "
              f"gross rev {t['gross_revenue']['mean']:>9,.0f}   spend {t['total_spend']['mean']:>9,.0f}   "
              f"net margin {t['net_margin_pct']['mean']}%")
        pw = t["purchased_wheat_dollars"]["mean"]
        print(f"  purchased WHEAT ${pw:,.0f} = {t['purchased_wheat_pct_of_gross']['mean']}% of gross "
              f"({t['purchased_wheat_units']['mean']:.0f} units @ ${t['purchased_wheat_avg_price']['mean']:.2f})")
        print(f"  spend split: wheat {pw:,.0f} | animals {t['animal_dollars']['mean']:,.0f} | "
              f"land {t['land_dollars']['mean']:,.0f} | seeds {t['seed_dollars']['mean']:,.0f} | "
              f"hire {t['hire_dollars']['mean']:,.0f} | fert-buy {t['fertilizer_bought_dollars']['mean']:,.0f}")
        af = t["activity_frac"]
        print(f"  unit-turns {t['unit_turns']['mean']:,.0f}  productive {t['productive_actions']['mean']:,.0f}"
              f"  | produce {af['produce']['mean']:.1%} maintain {af['maintain']['mean']:.1%}"
              f" invest {af['invest']['mean']:.1%} logistics {af['logistics']['mean']:.1%}"
              f" idle {af['idle']['mean']:.1%}")
        print(f"  $/unit-turn {t['rev_per_unit_turn']['mean']}   $/productive action "
              f"{t['rev_per_productive_action']['mean']}   hands/day {t['hands_mean']['mean']}")
        tm = t["terminal"]
        print(f"  terminal: cash {tm['cash']['mean']:,.0f}  shed {tm['shed_value']['mean']:,.0f}  "
              f"carried {tm['carried_value']['mean']:,.0f}  unharvested {tm['unharvested_value']['mean']:,.0f}  "
              f"seeds@cost {tm['seeds_left_cost']['mean']:,.0f}  capital->0 {tm['capital_written_off']['mean']:,.0f}")
        rf = t["refused"]
        print(f"  refused: animal(shed full) {rf['buy_animal_shed_full']['mean']}  "
              f"product(shed full) {rf['buy_product_shed_full']['mean']}  cash {rf['cash']['mean']}")
        print(f"  wheat internal transfer (memo, never banked) "
              f"${t['wheat_internal_transfer']['mean']:,.0f}   "
              f"wheat resold from bought stock {t['wheat_resold_from_stock']['mean']:.0f} units")
        tot_td = sum((d["tile_days"]["mean"] if d.get("tile_days") else 0) for d in t["lines"].values())
        tot_ac = sum((d["productive_actions"]["mean"] if d.get("productive_actions") else 0)
                     for d in t["lines"].values())
        hdr = (f"  {'line':<11}{'rev':>9}{'units':>7}{'$/u':>8}{'made':>7}{'tileD':>7}{'%td':>6}"
               f"{'acts':>7}{'%act':>6}{'seed$':>7}{'feed$':>8}{'capex$':>8}{'contrib':>9}"
               f"{'c/tileD':>9}{'c/act':>8}")
        print(hdr)
        for ln, d in t["lines"].items():
            g = lambda k: (d[k]["mean"] if d.get(k) else 0)
            dpu = d["dollars_per_unit"]
            s_dpu = ("%.1f" % dpu) if dpu else "-"
            s_ctd = ("%.1f" % d["contrib_per_tile_day"]["mean"]) if d.get("contrib_per_tile_day") else "-"
            s_cac = ("%.1f" % d["contrib_per_action"]["mean"]) if d.get("contrib_per_action") else "-"
            p_td = (100.0 * g("tile_days") / tot_td) if tot_td else 0
            p_ac = (100.0 * g("productive_actions") / tot_ac) if tot_ac else 0
            print(f"  {ln:<11}{g('revenue'):>9,.0f}{g('units_sold'):>7.0f}"
                  f"{s_dpu:>8}{g('units_produced'):>7.0f}"
                  f"{g('tile_days'):>7.0f}{p_td:>5.1f}%{g('productive_actions'):>7.0f}{p_ac:>5.1f}%"
                  f"{g('seed_spend'):>7,.0f}{g('feed_charged_bill'):>8,.0f}{g('capex'):>8,.0f}"
                  f"{g('contribution'):>9,.0f}{s_ctd:>9}{s_cac:>8}")


if __name__ == "__main__":
    raise SystemExit(main())
