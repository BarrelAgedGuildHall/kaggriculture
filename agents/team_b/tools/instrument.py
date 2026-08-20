"""Instrumented Kaggriculture episode runner (Team B sim-checklist).

Steps an episode manually and logs, per turn: market prices + inventory,
per-player money, EXACT per-player sales/buys per product (via a _commit_unit
patch inside the local engine -- attribution by farm identity), carried
inventory, per-unit action ops with executed/no-op classification, and MOVE
fraction. Also produces per-day aggregates and a last-day (liquidation stub)
accounting: items carried at the final step, shed contents, unharvested yield,
and FEED/WATER/CARE actions taken on the last day, all valued by walking the
sell curve at final market state.

Usage (from the repo root):
    uv run python agents/team_b/tools/instrument.py --agent-a agents/team_a/v0.py --agent-b agents/team_a/v0.py --seed 0
    uv run python agents/team_b/tools/instrument.py --agent-a probe:GooseRush --agent-b probe:GooseRush --seed 0 --trace out.jsonl
    uv run python agents/team_b/tools/instrument.py --agent-a 'probe:{"geese_day0":9}' --agent-b starter

Agent specs: builtin name (pass/random/starter), path to .py exposing agent(obs),
or probe:<PresetName> / probe:{json params} for the Team B probe core.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))

from _engine import import_engine, engine_version  # noqa: E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
MOVE_D = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def load_agent(spec: str):
    if spec.startswith("probe:"):
        from probe_core import make_agent, PRESETS
        arg = spec[6:]
        params = json.loads(arg) if arg.startswith("{") else PRESETS[arg]
        return make_agent(params)
    if spec.startswith("millrace:"):
        sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))
        from millrace.routed_probe import make_routed_agent
        arg = spec[9:]
        if arg == "routed_mixed":
            return make_routed_agent()
        if arg.startswith("{"):
            return make_routed_agent(json.loads(arg))
        raise ValueError(f"unknown millrace spec: {spec}")
    if spec in ("pass", "random", "starter"):
        import kaggle_environments.envs.kaggriculture.kaggriculture as K
        return {"pass": K.pass_agent, "random": K.random_agent, "starter": K.starter_agent}[spec]
    p = Path(spec)
    if not p.is_absolute():
        p = REPO_ROOT / p
    import importlib.util
    sys.path.insert(0, str(p.parent))
    mod_spec = importlib.util.spec_from_file_location(f"agent_{p.stem}_{id(p)}", p)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod.agent


def curve_value(K, market_inv, item, n):
    """Marginal revenue of selling n units of item into the current market."""
    inv = market_inv.get(item, 10000)
    total = 0
    for _ in range(int(n)):
        p = K.market_price(item, inv)
        total += p
        if p > 1:
            inv += 1
    return total


def unit_positions(farm):
    return [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]


def classify(op_action, pre_pos, post_pos, pre_tile, post_tile, inv_delta):
    """Best-effort executed / no-op classification of one unit action."""
    if not isinstance(op_action, list) or not op_action:
        return "none"
    op = op_action[0]
    if op == "PASS":
        return "pass"
    if op in MOVES:
        dx, dy = MOVE_D[op]
        return "executed" if post_pos == (pre_pos[0] + dx, pre_pos[1] + dy) else "noop"
    g = lambda t, k, d=None: (t.get(k, d) if isinstance(t, dict) else d)
    if op == "WATER":
        return "executed" if (g(pre_tile, "watered_today") is False and g(post_tile, "watered_today") is True) else "noop"
    if op == "FEED":
        return "executed" if (g(pre_tile, "fed_today") is False and g(post_tile, "fed_today") is True) else "noop"
    if op == "CARE":
        return "executed" if (g(pre_tile, "cared_today") is False and g(post_tile, "cared_today") is True) else "noop"
    if op == "COLLECT_FERTILIZER":
        return "executed" if (g(pre_tile, "fertilizer_available") is True and g(post_tile, "fertilizer_available") is False) else "noop"
    if op == "HARVEST":
        pre_y, post_y = g(pre_tile, "yield_units", 0) or 0, g(post_tile, "yield_units", 0) or 0
        return "executed" if (post_y < pre_y or (isinstance(pre_tile, dict) and post_tile is None)) else "noop"
    if op == "PLANT":
        return "executed" if (pre_tile is None and g(post_tile, "kind") == "PLANT") else "noop"
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return "executed" if (pre_tile is None and g(post_tile, "kind") in ("COOP", "PASTURE")) else "noop"
    if op == "FERTILIZE":
        return "executed" if (g(post_tile, "fertilized_until_day", -1) or -1) > (g(pre_tile, "fertilized_until_day", -1) or -1) else "noop"
    if op == "DIG":
        return "executed" if (pre_tile is not None and post_tile is None) else "noop"
    if op in ("DROP", "PICKUP", "PLACE"):
        return "executed" if inv_delta else "noop"
    return "unknown"


def run(agent_a: str, agent_b: str, seed: int, steps: int, trace_path: str | None):
    ke = import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    trades = []  # (player, op, item, price)
    farm_ids = {}
    orig_commit = K._commit_unit
    orig_process = K._process_market

    def patched_commit(op, item, price, farm, private, market, shed_capacity=100):
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if ok:
            trades.append((farm_ids.get(id(farm), -1), op, item, price))
        return ok

    def patched_process(state, env_):
        # env.step re-structifies state each turn, so farm object identity is
        # only stable WITHIN a step: rebuild the id->player map from the live
        # objects the interpreter is about to trade with.
        farm_ids.clear()
        for i, f in enumerate(state[0].observation.farms):
            farm_ids[id(f)] = i
        return orig_process(state, env_)

    K._commit_unit = patched_commit
    K._process_market = patched_process
    try:
        env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
        agents = [load_agent(agent_a), load_agent(agent_b)]
        env.reset()

        turns_per_day = 24
        last_day = (steps - 2) // turns_per_day
        trace = open(trace_path, "w", encoding="utf-8") if trace_path else None

        cum_sales = [{p: 0 for p in K.PRODUCTS} for _ in range(2)]
        cum_rev = [{p: 0 for p in K.PRODUCTS} for _ in range(2)]
        cum_buys = [{}, {}]
        act_counts = [{"moves": 0, "pass": 0, "noop": 0, "executed": 0, "unknown": 0, "none": 0, "unit_turns": 0, "hour23_unmeasured": 0} for _ in range(2)]
        lastday_actions = [{"FEED": 0, "WATER": 0, "CARE": 0} for _ in range(2)]
        fert_floor_step = None
        daily = []  # per-day snapshot rows
        final_acct = None

        while not env.done:
            step = env.state[0].observation.step
            day = step // turns_per_day
            obs = [env.state[i].observation for i in range(2)]
            farms = env.state[0].observation.farms
            actions = []
            for i, ag in enumerate(agents):
                try:
                    a = ag(obs[i])
                except Exception:
                    a = {"farmer": ["PASS"], "hands": [], "market": []}
                actions.append(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})

            pre_farms = json.loads(json.dumps(farms))
            pre_invs = [json.loads(json.dumps(env.state[i].observation.private["inventories"])) for i in range(2)]
            n_trades_before = len(trades)

            env.step(actions)

            post_farms = env.state[0].observation.farms
            for pi in range(2):
                unit_acts = [actions[pi].get("farmer", ["PASS"])] + list(actions[pi].get("hands", []) or [])
                pre_pos = unit_positions(pre_farms[pi])
                post_pos = unit_positions(post_farms[pi])
                post_inv = env.state[pi].observation.private["inventories"]
                for ui, ua in enumerate(unit_acts):
                    if ui >= len(pre_pos):
                        break
                    act_counts[pi]["unit_turns"] += 1
                    # Hour 23: unit actions execute and THEN the end-of-day
                    # refresh runs inside the same env.step (farmer respawned,
                    # hands cleared, daily tile flags reset), so post-state
                    # comparison misclassifies every hour-23 action as "noop".
                    # Verified 2026-08-20 (feed_audit.py): 100% of a
                    # routed_mixed mirror's classified no-ops were hour-23
                    # artifacts. Excluded from classification; counted apart.
                    if day * turns_per_day + 23 == step and isinstance(ua, list) and ua and ua[0] != "PASS":
                        act_counts[pi]["hour23_unmeasured"] += 1
                        continue
                    px, py = pre_pos[ui]
                    pre_tile = pre_farms[pi]["tiles"][py][px]
                    post_tile = post_farms[pi]["tiles"][py][px]
                    ppos = post_pos[ui] if ui < len(post_pos) else pre_pos[ui]
                    inv_delta = (json.dumps(pre_invs[pi][ui] if ui < len(pre_invs[pi]) else {}, sort_keys=True)
                                 != json.dumps(dict(post_inv[ui]) if ui < len(post_inv) else {}, sort_keys=True))
                    c = classify(ua, (px, py), ppos, pre_tile, post_tile, inv_delta)
                    if isinstance(ua, list) and ua and ua[0] in MOVES and c == "executed":
                        act_counts[pi]["moves"] += 1
                    act_counts[pi][c if c in act_counts[pi] else "unknown"] += 1
                    if day == last_day and isinstance(ua, list) and ua and ua[0] in ("FEED", "WATER", "CARE") and c == "executed":
                        lastday_actions[pi][ua[0]] += 1

            for (pl, op, item, price) in trades[n_trades_before:]:
                if pl < 0:
                    continue
                if op == "SELL":
                    cum_sales[pl][item] += 1
                    cum_rev[pl][item] += price
                elif op in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"):
                    key = f"{op}:{item}"
                    cum_buys[pl][key] = cum_buys[pl].get(key, 0) + 1

            market = env.state[0].observation.market
            if fert_floor_step is None and market["prices"].get("FERTILIZER", 100) <= 1:
                fert_floor_step = step

            if trace:
                trace.write(json.dumps({
                    "step": step, "day": day,
                    "prices": dict(market["prices"]), "inventory": dict(market["inventory"]),
                    "money": [post_farms[i]["money"] for i in range(2)],
                }) + "\n")

            if (step + 1) % turns_per_day == 0 or step >= steps - 2:
                daily.append({
                    "day": day,
                    "prices": dict(market["prices"]),
                    "money": [post_farms[i]["money"] for i in range(2)],
                    "cum_sales": [dict(cum_sales[i]) for i in range(2)],
                    "cum_rev": [dict(cum_rev[i]) for i in range(2)],
                    "animals": [sum(1 for row in post_farms[i]["tiles"] for t in row
                                    if isinstance(t, dict) and "animal" in t) for i in range(2)],
                    "hands": [len(post_farms[i]["hands"]) for i in range(2)],
                })

            if step >= steps - 2 and final_acct is None:
                minv = dict(market["inventory"])
                acct = []
                for pi in range(2):
                    priv = env.state[pi].observation.private
                    carried = {}
                    for iv in priv["inventories"]:
                        for it, n in iv.items():
                            carried[it] = carried.get(it, 0) + n
                    shed = {k: v for k, v in priv["shed"].items() if v > 0}
                    unharv = {}
                    for row in post_farms[pi]["tiles"]:
                        for t in row:
                            if isinstance(t, dict) and t.get("yield_units", 0) > 0:
                                prod = (K.ANIMALS[t["animal"]]["product"] if "animal" in t
                                        else t.get("crop"))
                                if prod:
                                    unharv[prod] = unharv.get(prod, 0) + t["yield_units"]
                    val = lambda d: round(sum(curve_value(K, minv, it, n) for it, n in d.items() if it in K.PRODUCTS), 0)
                    acct.append({
                        "carried": carried, "carried_value": val(carried),
                        "shed": shed, "shed_value": val(shed),
                        "unharvested": unharv, "unharvested_value": val(unharv),
                        "lastday_wasted_actions": lastday_actions[pi],
                    })
                final_acct = acct

        final = env.steps[-1]
        money = [final[i].observation.farms[i]["money"] for i in range(2)]
        summary = {
            "engine": engine_version(), "seed": seed, "steps": steps,
            "agents": [agent_a, agent_b],
            "final_money": money, "statuses": [s.status for s in final],
            "fertilizer_floor_step": fert_floor_step,
            "fertilizer_floor_day": (fert_floor_step // 24) if fert_floor_step is not None else None,
            "fert_units_sold": [cum_sales[i]["FERTILIZER"] for i in range(2)],
            "fert_revenue": [cum_rev[i]["FERTILIZER"] for i in range(2)],
            "cum_sales": [dict(cum_sales[i]) for i in range(2)],
            "cum_rev": [dict(cum_rev[i]) for i in range(2)],
            "cum_buys": cum_buys,
            "action_stats": act_counts,
            "move_fraction": [round(act_counts[i]["moves"] / max(1, act_counts[i]["unit_turns"] - act_counts[i]["hour23_unmeasured"]), 4) for i in range(2)],
            "noop_fraction": [round(act_counts[i]["noop"] / max(1, act_counts[i]["unit_turns"] - act_counts[i]["hour23_unmeasured"]), 4) for i in range(2)],
            "pass_fraction": [round(act_counts[i]["pass"] / max(1, act_counts[i]["unit_turns"]), 4) for i in range(2)],
            "final_accounting": final_acct,
            "daily": daily,
        }
        if trace:
            trace.close()
        return summary
    finally:
        K._commit_unit = orig_commit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-a", required=True)
    ap.add_argument("--agent-b", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--trace", default=None, help="write per-turn JSONL here")
    ap.add_argument("--out", default=None, help="write summary JSON here")
    ap.add_argument("--daily", action="store_true", help="print the per-day table")
    args = ap.parse_args(argv)

    s = run(args.agent_a, args.agent_b, args.seed, args.steps, args.trace)
    if args.daily:
        for d in s["daily"]:
            print(f"day {d['day']:>2}  fertP={d['prices']['FERTILIZER']:>3}  "
                  f"money={d['money']}  animals={d['animals']}  "
                  f"fert_sold={[d['cum_sales'][i]['FERTILIZER'] for i in range(2)]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(s, indent=2), encoding="utf-8")
    keep = {k: v for k, v in s.items() if k != "daily"}
    print("SUMMARY " + json.dumps(keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
