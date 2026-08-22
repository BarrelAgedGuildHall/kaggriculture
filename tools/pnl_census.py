"""The P&L census: a per-line income statement and terminal balance sheet for a Kaggriculture
episode.

Shared measurement tooling (plan `2026-08-21-balance-sheet-program/plan.md` §6a.5: *teams compete
on agents, never on books*). Originally `agents/team_b/tools/pnl_census.py` (expB-301, design in
`2026-08-21-team-b-business-lens/business-lens.md` §§2.4, 5); promoted here with the corrections
Architect B ruled in `ruling-02-m4-and-sprint2-order.md` §3.1 and §3.3. It is a *diagnostic*: it
changes no agent code.

What it measures, per player, per season:

  revenue        exact SELL commits (units + realised dollars) per product
  purchased      exact BUY_PRODUCT / BUY_SEED / BUY_ANIMAL commits (units + realised dollars),
                 plus HIRE and BUY_LAND money deltas
  production     units actually issued by HARVEST (read off the tile *before* the harvest lands,
                 so it is exact, not inferred from sales)
  tile-days      every owned tile sampled at hour 0 of every day, keyed by the line occupying it
  unit-turns     every farmer/hand action classified into
                 produce | maintain | invest | logistics | idle, attributed to the line whose
                 tile it touched, with an exact executed/no-op verdict
  terminal       cash, shed + carried inventory walked down the sell curve, unharvested pipeline,
                 unused seeds at cost, and the capital (land + animals) that day 29 writes to zero

Two corrections to the expB-301 version, both from ruling-02:

  §3.3 FERTILIZER IS A JOINT PRODUCT, NOT A LINE. The engine emits fertilizer from an animal tile
  unconditionally (`_daily_refresh_animals` sets `fertilizer_available`); `COLLECT_FERTILIZER`
  turns one available tile into exactly one unit. Crediting that revenue to a standalone
  FERTILIZER line while charging the feed to the animals understates every animal line, goose
  most. So: every executed collection is attributed to **the animal standing on the collected
  tile** (read before the action, so it is exact, not apportioned), fertilizer sales revenue is
  allocated to the animal lines pro rata by units collected, and the collection actions are
  counted in those lines' action budgets -- which is the "net of its own collection actions" term:
  it rides in the denominator, so no opportunity-cost constant has to be assumed to state it.
  FERTILIZER survives as a **memo line** (`joint_product: true`) carrying the gross figures.

  §3.1 GROSS AND CONTRIBUTION ARE DIFFERENT DENOMINATORS OF VALUE and must be labelled as such:
  every line reports `gross_per_action` *and* `contribution_per_action` (likewise per tile-day),
  and the statement carries a `lambda_opp` block -- the portfolio contribution per productive
  action (the reference a marginal task must beat; this is the definition the agent's own live
  ledger uses), the per-line budget ladder sorted by contribution/action, and the marginal
  executing line at the budget cut.

Two engine facts drive the instrumentation choices:

  1. Hour-23 unit actions execute and are then followed by the end-of-day refresh inside the same
     `env.step`, so a pre/post *state diff* taken around `env.step` misclassifies every hour-23
     action (memory: "NEVER classify hour-23 actions by state diff"). This tool therefore patches
     `_apply_unit_action` itself and diffs the tile across that single call -- correct at every
     hour, hour 23 included.
  2. Unit actions run BEFORE `_process_market` in `interpreter`, so the farm->player identity map
     cannot be built from a `_process_market` patch. We wrap `env.interpreter` instead and rebuild
     the map at the top of every step.

Usage (from the repo root):

    uv run python tools/pnl_census.py --a millrace:feedchain --b millrace:feedchain \
        --seeds 0-7 --out-dir artisans/<you>/bench/runs/<exp>/mirror

Agent specs: see `tools/agent_spec.py` (built-ins, .py path, agent dir, probe:, millrace:).
Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _engine import engine_version, import_engine  # noqa: E402
from agent_spec import arity_wrap, curve_value, load_agent  # noqa: E402

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

# line -> the product it sells. FERTILIZER is kept as a memo line (§3.3): it has no tiles of its
# own and, after allocation, no revenue of its own either.
LINE_PRODUCT = {
    "GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL",
    "WHEAT": "WHEAT", "CARROT": "CARROT", "TOMATO": "TOMATO",
    "STRAWBERRY": "STRAWBERRY", "MELON": "MELON",
    "FERTILIZER": "FERTILIZER",
}
ANIMAL_LINES = ("GOOSE", "COW", "SHEEP")
CROP_LINES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
JOINT_PRODUCT_LINE = "FERTILIZER"

# A line consuming less than this share of the productive-action budget is not treated as the
# marginal line at the budget cut (§3.1): a rounding-error line is not what a scheduler trades off.
MARGINAL_MIN_ACTION_SHARE = 0.01


def _blank_line():
    return {
        "sales_revenue": 0.0, "units_sold": 0, "units_produced": 0,
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
        # §3.3: one executed COLLECT_FERTILIZER == one unit, and the engine refuses the action
        # unless the tile carries an animal -- so this attribution is exact, not apportioned.
        "fert_collected_by_line": {a: 0 for a in K.ANIMALS},
        "fert_collected_unattributed": 0,
        "fertilize_on": {c: 0 for c in K.CROPS},
        "acts": {b: {"executed": 0, "noop": 0} for b in
                 ("produce", "maintain", "invest", "logistics", "idle", "unknown")},
        "acts_by_op": {},
        "unit_turns": 0,
        "lines": {},
        "tile_days_other": {"EMPTY_STRUCT": 0, "WEED": 0, "FALLOW": 0, "LOCKED": 0},
        "daily_tiles": [],
        "hands_by_day": [],
        # expB-304c: the shed destroys overflow. `DROP` and `_drop_inventories_to_shed`
        # both compute `room = max(0, capacity - sum(shed))`, `take = min(n, room)`,
        # and then `del inv[item]` UNCONDITIONALLY -- so anything that did not fit is
        # gone, silently, with no refusal and no market event. It is invisible to every
        # revenue-side counter, which is why the census's conservation identity did not
        # close. Measured by diffing `sum(inventories) + sum(shed)` ACROSS the call, not
        # by re-deriving the room arithmetic (which reports zero and is wrong).
        "destroyed": {},                              # item -> units evaporated
        "destroyed_by_path": {"drop": 0, "end_of_day": 0},
        "shed_days": [],                              # shed occupancy sampled hourly
        "shed_full_turns": 0,
        "placed_animals": {a: 0 for a in K.ANIMALS},
    } for _ in range(2)]
    for pl in P:
        for ln in list(LINE_PRODUCT):
            pl["lines"][ln] = _blank_line()

    orig_commit = K._commit_unit
    orig_hire = K._do_hire
    orig_land = K._do_buy_land
    orig_apply = K._apply_unit_action
    orig_drop_eod = K._drop_inventories_to_shed
    priv_ids: dict[int, int] = {}

    def _held(private):
        """Every unit the shed accounting can see: shed + all carried inventories."""
        out: dict[str, int] = {}
        for k, v in private["shed"].items():
            out[k] = out.get(k, 0) + v
        for iv in private.get("inventories", []):
            for k, v in iv.items():
                out[k] = out.get(k, 0) + v
        return out

    def _charge_destroyed(pid, before, after, path):
        for item, n in before.items():
            lost = n - after.get(item, 0)
            if lost > 0:
                P[pid]["destroyed"][item] = P[pid]["destroyed"].get(item, 0) + lost
                P[pid]["destroyed_by_path"][path] += lost

    def patched_drop_eod(private, capacity):
        pid = priv_ids.get(id(private), -1)
        if pid < 0:
            return orig_drop_eod(private, capacity)
        before = _held(private)
        r = orig_drop_eod(private, capacity)
        _charge_destroyed(pid, before, _held(private), "end_of_day")
        return r

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
        # expB-304c: DROP is the other place the shed evaporates overflow. Same
        # measurement as the end-of-day path -- diff what the player holds across
        # the engine's own call.
        held_pre = _held(private) if op == "DROP" else None
        bucket = BUCKET.get(op, "unknown")
        pos = K._farmer_position(farm, idx)
        px, py = (pos[0], pos[1]) if pos else (-1, -1)
        live_tile = farm["tiles"][py][px] if pos else None
        # NB: tiles are mutated IN PLACE by _apply_unit_action, so the "pre" snapshot must be a
        # copy or every in-place op reads back as a no-op (this bug zeroed FEED/WATER/CARE on the
        # first cut of this tool). Tile values are scalars, so a shallow dict() is a full snapshot.
        pre_tile = dict(live_tile) if isinstance(live_tile, dict) else live_tile
        invs = private.get("inventories", [])
        pre_inv = dict(invs[idx]) if idx < len(invs) else {}
        pre_seeds = dict(private.get("seeds", {}))
        pre_shed = dict(private.get("shed", {}))
        pre_yield = pre_tile.get("yield_units", 0) if isinstance(pre_tile, dict) else 0

        r = orig_apply(farm, private, idx, action, board_size, day, turns_per_day_, shed_capacity)

        if held_pre is not None:
            _charge_destroyed(pid, held_pre, _held(private), "drop")
        if sum(private["shed"].values()) >= shed_capacity:
            st["shed_full_turns"] += 1
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
            # §3.3: the collected unit is a joint product of THIS animal-day. The engine has
            # already refused the action unless `pre_tile` carries an animal, so the emitting
            # line is known exactly; the action lands on that line's budget, not on a
            # standalone FERTILIZER line.
            line = _line_of_tile(pre_tile)
            st["fert_collected"] += 1
            if line in st["fert_collected_by_line"]:
                st["fert_collected_by_line"][line] += 1
            else:
                st["fert_collected_unattributed"] += 1
        elif op == "FERTILIZE":
            crop = _line_of_tile(pre_tile)
            if crop in K.CROPS:
                st["fertilize_on"][crop] += 1
            line = crop
        elif op == "PLANT":
            line = action[1] if len(action) >= 2 else None
        elif op == "PLACE":
            line = action[1] if len(action) >= 2 and action[1] in K.ANIMALS else None
            if line:
                st["placed_animals"][line] += 1
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
    K._drop_inventories_to_shed = patched_drop_eod
    try:
        env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
        orig_interpreter = env.interpreter

        def wrapped_interpreter(state, env_):
            farm_ids.clear()
            farms = getattr(state[0].observation, "farms", None)
            if farms:
                for i, f in enumerate(farms):
                    farm_ids[id(f)] = i
            # `_drop_inventories_to_shed` is handed `private`, not `farm`, so it needs
            # its own identity map -- rebuilt every step for the same reason farm_ids is.
            priv_ids.clear()
            for i in range(len(state)):
                pv = getattr(state[i].observation, "private", None)
                if pv is not None:
                    priv_ids[id(pv)] = i
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
                # hands are cleared every night and re-hired during the day, so sample the crew
                # at mid-day, not at hour 0 (which reads zero).
                for pi in range(2):
                    P[pi]["hands_by_day"].append(len(env.state[0].observation.farms[pi]["hands"]))
            # expB-304c: shed occupancy against the 100-unit cap, sampled hourly --
            # the backpressure policy needs the distribution, not the daily mean.
            for pi in range(2):
                P[pi]["shed_days"].append(sum(env.state[pi].observation.private["shed"].values()))
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

def allocate_fertilizer(st: dict) -> dict:
    """Ruling-02 §3.3: split fertilizer sales revenue back onto the animal lines that emitted it.

    Base = units collected per animal line (exact: one executed COLLECT_FERTILIZER is one unit and
    the engine only allows it on an animal tile). If fertilizer was also *bought*, only the
    collected share of supply is a joint product; the remainder is resale of purchased stock and
    stays on the memo line, unallocated, rather than flattering the herd.

    Returns a block that is reported verbatim in the statement, so the allocation can always be
    audited back to its base.
    """
    fert_revenue = st["sales"][JOINT_PRODUCT_LINE]["dollars"]
    fert_units_sold = st["sales"][JOINT_PRODUCT_LINE]["units"]
    collected = {a: n for a, n in st["fert_collected_by_line"].items() if n}
    total_collected = sum(collected.values()) + st.get("fert_collected_unattributed", 0)
    bought_units = st["buy_product"][JOINT_PRODUCT_LINE]["units"]

    supply = total_collected + bought_units
    joint_share = (total_collected / supply) if supply else 0.0
    allocatable = fert_revenue * joint_share
    base = sum(collected.values())
    allocated = {a: round(allocatable * n / base, 2) for a, n in collected.items()} if base else {}
    allocated_units = int(round(fert_units_sold * joint_share)) if base else 0
    return {
        "gross_revenue": round(fert_revenue, 2),
        "units_sold": fert_units_sold,
        "units_collected": total_collected,
        "units_bought": bought_units,
        "collected_by_line": dict(collected),
        "collect_actions_by_line": dict(collected),   # 1 executed action == 1 unit (engine)
        "joint_share_of_supply": round(joint_share, 4),
        "allocated_by_line": allocated,
        "allocated_units": allocated_units,
        "residual_units": fert_units_sold - allocated_units,
        "unallocated_revenue": round(fert_revenue - sum(allocated.values()), 2),
        "unattributed_units": st.get("fert_collected_unattributed", 0),
    }


def _lambda_block(lines: dict, total_contribution: float, gross_revenue: float,
                  productive_actions: int) -> dict:
    """§3.1: the λ_opp reference on the contribution denominator, plus the budget ladder.

    `reference` is total contribution / productive actions -- the same definition the agent's live
    ledger uses ((gross - cogs) / productive actions), and the number a marginal task must beat when
    the action budget is fully spent. `marginal_line` is the lowest contribution/action line still
    consuming a material share of the budget: the task a pricer would cut first.
    """
    ladder = []
    for ln, v in lines.items():
        acts = v["productive_actions"]
        if not acts:
            continue
        ladder.append({"line": ln, "contribution_per_action": v["contribution_per_action"],
                       "gross_per_action": v["gross_per_action"], "actions": acts,
                       "action_share": round(acts / productive_actions, 4) if productive_actions else None})
    ladder.sort(key=lambda d: d["contribution_per_action"], reverse=True)
    cum = 0
    for row in ladder:
        cum += row["actions"]
        row["cum_action_share"] = round(cum / productive_actions, 4) if productive_actions else None
    material = [r for r in ladder if (r["action_share"] or 0) >= MARGINAL_MIN_ACTION_SHARE]
    marginal = material[-1] if material else (ladder[-1] if ladder else None)
    return {
        "reference": round(total_contribution / productive_actions, 2) if productive_actions else None,
        "gross_per_action": round(gross_revenue / productive_actions, 2) if productive_actions else None,
        "marginal_line": marginal["line"] if marginal else None,
        "marginal_contribution_per_action": marginal["contribution_per_action"] if marginal else None,
        "ladder": ladder,
    }


def unit_conservation(st: dict) -> dict:
    """expB-304c: close the census's unit-conservation identity, per item.

        produced + bought - sold - consumed - terminal_stock - destroyed == 0

    Every term is an exact engine event: `produced` is the units a HARVEST issued
    (or a COLLECT_FERTILIZER, or a BUY_ANIMAL for a live animal), `consumed` is
    FERTILIZE for FERTILIZER and PLACE for animals, `terminal_stock` is the shed
    plus everything still in a pocket at the end, and `destroyed` is the overflow
    the shed evaporated. Before `destroyed` existed the identity did not close,
    and the residual was silently distributed over every line's realised revenue
    -- which is precisely the input expB-304b sorts on (ruling-03 section 3.4
    reason 4). FEED is not a term: it consumes wheat from a unit's inventory but
    the engine deletes it there, and that deletion is picked up by the same
    inventory walk, so it is folded into `consumed` via the wheat residual.

    The identity is reported per item with its residual; `closed` is the count of
    items whose residual is exactly zero. A non-zero residual is a measurement
    bug, not an economic finding, and must be read as one.
    """
    term = st["terminal"]
    stock: dict[str, int] = {}
    for k, v in term["shed"].items():
        stock[k] = stock.get(k, 0) + v
    for k, v in term["carried"].items():
        stock[k] = stock.get(k, 0) + v

    rows = {}
    items = set(st["harvest_units"]) | set(st["destroyed"]) | set(stock) \
        | set(st["buy_product"]) | set(st["buy_animal"]) | {"FERTILIZER"}
    for item in sorted(items):
        produced = st["harvest_units"].get(item, 0)
        if item == "FERTILIZER":
            produced += st["fert_collected"]
        bought = st["buy_product"].get(item, {}).get("units", 0) \
            + st["buy_animal"].get(item, {}).get("units", 0)
        sold = st["sales"].get(item, {}).get("units", 0)
        consumed = 0
        if item == "FERTILIZER":
            consumed = sum(st["fertilize_on"].values())
        elif item in st["placed_animals"]:
            consumed = st["placed_animals"][item]
        elif item == "WHEAT":
            consumed = sum(st["feed_events"].values())
        destroyed = st["destroyed"].get(item, 0)
        end = stock.get(item, 0)
        residual = produced + bought - sold - consumed - end - destroyed
        rows[item] = {"produced": produced, "bought": bought, "sold": sold,
                      "consumed": consumed, "terminal_stock": end,
                      "destroyed": destroyed, "residual": residual}
    return {
        "per_item": rows,
        "destroyed_units": sum(st["destroyed"].values()),
        "destroyed_by_path": dict(st["destroyed_by_path"]),
        "residual_abs": sum(abs(r["residual"]) for r in rows.values()),
        "items_closed": sum(1 for r in rows.values() if r["residual"] == 0),
        "items": len(rows),
        "shed_mean": round(statistics.fmean(st["shed_days"]), 2) if st["shed_days"] else None,
        "shed_max": max(st["shed_days"]) if st["shed_days"] else None,
        "shed_full_turns": st["shed_full_turns"],
    }


def income_statement(st: dict) -> dict:
    """Turn one player's raw census into a per-line income statement.

    Feed is charged at the season's realised average *purchased* wheat price, allocated across FEED
    events (the marginal source of feed is the market: we grow every wheat tile we can and buy the
    rest). Two views are produced:
      feed_charged_bill   -- the actual purchased-wheat bill, split pro-rata by FEED event
      feed_charged_margin -- every FEED event valued at the average purchase price (i.e. grown
                             wheat charged at its opportunity cost)
    Purchased fertilizer is charged the same way, pro-rata by FERTILIZE event, onto the crop lines
    that received it (zero for an agent that issues no FERTILIZE -- it then shows up as
    `unallocated_input_spend`, never as a silent zero).

    Column vocabulary (§3.1), identical for every line:
      sales_revenue                own-product SELL dollars
      fertilizer_revenue_allocated joint-product share (animal lines only, §3.3)
      gross_revenue                sales_revenue + fertilizer_revenue_allocated
      direct_cost                  seed + charged feed + charged fertilizer (cash inputs only)
      contribution                 gross_revenue - direct_cost
      gross_per_action / contribution_per_action        (and the same pair per tile-day)
    """
    lines = {ln: dict(v) for ln, v in st["lines"].items()}
    gross_revenue = sum(v["dollars"] for v in st["sales"].values())

    # own-product revenue / units onto lines
    for ln, prod in LINE_PRODUCT.items():
        lines[ln]["sales_revenue"] = st["sales"][prod]["dollars"]
        lines[ln]["units_sold"] = st["sales"][prod]["units"]
        lines[ln]["units_produced"] = st["harvest_units"][prod]
    lines[JOINT_PRODUCT_LINE]["units_produced"] = st["fert_collected"]

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
        lines[a]["feed_charged_bill"] = round(wheat_bill * share, 2)
        lines[a]["feed_charged_margin"] = round(avg_buy * st["feed_events"][a], 2)
        lines[a]["feed_events"] = st["feed_events"][a]

    fert_bill = st["buy_product"][JOINT_PRODUCT_LINE]["dollars"]
    fertilize_total = sum(st["fertilize_on"].values())
    for c in CROP_LINES:
        share = (st["fertilize_on"][c] / fertilize_total) if fertilize_total else 0.0
        lines[c]["fert_charged_bill"] = round(fert_bill * share, 2)
        lines[c]["fertilize_events"] = st["fertilize_on"][c]
    unallocated_input_spend = round(fert_bill if not fertilize_total else 0.0, 2)

    # §3.3 joint-product allocation: fertilizer revenue back onto the emitting animal-days
    fert_alloc = allocate_fertilizer(st)
    for ln in lines:
        lines[ln]["fertilizer_revenue_allocated"] = 0.0
    for a, dollars in fert_alloc["allocated_by_line"].items():
        if a in lines:
            lines[a]["fertilizer_revenue_allocated"] = dollars
    # the memo line keeps the gross figures but no revenue of its own: its dollars and its
    # collection actions now live on the animal lines that emitted them.
    memo = lines[JOINT_PRODUCT_LINE]
    memo["joint_product"] = True
    memo["memo_gross_revenue"] = fert_alloc["gross_revenue"]
    memo["memo_units_sold"] = fert_alloc["units_sold"]
    memo["sales_revenue"] = fert_alloc["unallocated_revenue"]
    memo["units_sold"] = fert_alloc["residual_units"]

    # the wheat line is internal: it "sells" its harvest to the animals at the market buy price it
    # displaces. Reported as memo revenue, never banked.
    wheat_grown = st["harvest_units"]["WHEAT"]
    lines["WHEAT"]["internal_transfer_revenue"] = round(
        avg_buy * max(0, wheat_grown - st["sales"]["WHEAT"]["units"]), 2)

    for ln, v in lines.items():
        acts = v["acts"]
        v["productive_actions"] = sum(acts.values())
        v["direct_cost"] = round(v.get("seed_spend", 0.0) + v.get("feed_charged_bill", 0.0)
                                 + v.get("fert_charged_bill", 0.0), 2)
        v["gross_revenue"] = round(v["sales_revenue"] + v["fertilizer_revenue_allocated"], 2)
        v["contribution"] = round(v["gross_revenue"] - v["direct_cost"], 2)
        v["contribution_after_capex"] = round(v["contribution"] - v.get("capex", 0.0), 2)
        td, pa = v["tile_days"], v["productive_actions"]
        v["gross_per_tile_day"] = round(v["gross_revenue"] / td, 2) if td else None
        v["contribution_per_tile_day"] = round(v["contribution"] / td, 2) if td else None
        v["gross_per_action"] = round(v["gross_revenue"] / pa, 2) if pa else None
        v["contribution_per_action"] = round(v["contribution"] / pa, 2) if pa else None

    total_spend = (wheat_bill + fert_bill
                   + sum(v["dollars"] for v in st["buy_seed"].values())
                   + sum(v["dollars"] for v in st["buy_animal"].values())
                   + st["hire"]["dollars"] + st["land"]["dollars"])
    acts = st["acts"]
    measured = sum(v["executed"] + v["noop"] for v in acts.values())
    prod_acts = sum(acts[b]["executed"] for b in PRODUCTIVE)
    total_contribution = sum(v["contribution"] for v in lines.values())
    return {
        "lines": lines,
        "conservation": unit_conservation(st),
        "gross_revenue": round(gross_revenue, 2),
        "total_contribution": round(total_contribution, 2),
        "fertilizer_allocation": fert_alloc,
        "unallocated_input_spend": unallocated_input_spend,
        "purchased_wheat_dollars": round(wheat_bill, 2),
        "purchased_wheat_units": wheat_units_bought,
        "purchased_wheat_avg_price": round(avg_buy, 2),
        "purchased_wheat_pct_of_gross": round(100.0 * wheat_bill / gross_revenue, 2) if gross_revenue else None,
        "total_spend": round(total_spend, 2),
        "net_margin_pct": round(100.0 * (gross_revenue - total_spend) / gross_revenue, 1) if gross_revenue else None,
        "hire_dollars": round(st["hire"]["dollars"], 2),
        "hire_count": st["hire"]["count"],
        "land_dollars": round(st["land"]["dollars"], 2),
        "seed_dollars": round(sum(v["dollars"] for v in st["buy_seed"].values()), 2),
        "animal_dollars": round(sum(v["dollars"] for v in st["buy_animal"].values()), 2),
        "fertilizer_bought_dollars": round(fert_bill, 2),
        "unit_turns": st["unit_turns"],
        "acts_by_op": st["acts_by_op"],
        "activity": {b: acts[b]["executed"] for b in acts},
        "activity_noop": {b: acts[b]["noop"] for b in acts},
        "activity_frac": {b: round(acts[b]["executed"] / measured, 4) for b in acts} if measured else {},
        "productive_actions": prod_acts,
        "gross_per_unit_turn": round(gross_revenue / st["unit_turns"], 2) if st["unit_turns"] else None,
        "gross_per_productive_action": round(gross_revenue / prod_acts, 2) if prod_acts else None,
        "contribution_per_productive_action": round(total_contribution / prod_acts, 2) if prod_acts else None,
        "lambda_opp": _lambda_block(lines, total_contribution, gross_revenue, prod_acts),
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
    raw = run_pnl(spec_a, spec_b, seed, steps)
    stmts = [income_statement(raw["players"][i]) for i in range(2)]
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


TOP_KEYS = ("gross_revenue", "total_contribution", "purchased_wheat_dollars", "purchased_wheat_units",
            "purchased_wheat_avg_price", "purchased_wheat_pct_of_gross", "total_spend",
            "net_margin_pct", "hire_dollars", "hire_count", "land_dollars", "seed_dollars",
            "animal_dollars", "fertilizer_bought_dollars", "unallocated_input_spend", "unit_turns",
            "productive_actions", "gross_per_unit_turn", "gross_per_productive_action",
            "contribution_per_productive_action", "hands_mean")

LINE_KEYS = ("sales_revenue", "fertilizer_revenue_allocated", "gross_revenue", "units_sold",
             "units_produced", "tile_days", "productive_actions", "direct_cost", "contribution",
             "contribution_after_capex", "seed_spend", "capex", "gross_per_tile_day",
             "contribution_per_tile_day", "gross_per_action", "contribution_per_action")


def aggregate(rows: list[dict]) -> dict:
    by_spec: dict[str, list] = {}
    for r in rows:
        for seat, sp in enumerate(r["agents"]):
            by_spec.setdefault(sp, []).append(r["statements"][seat])

    report = {"engine": rows[0]["engine"], "seeds": [r["seed"] for r in rows], "n_episodes": len(rows),
              "per_agent": {}}
    for sp, stmts in by_spec.items():
        top = {k: _agg([s[k] for s in stmts if s.get(k) is not None]) for k in TOP_KEYS}
        top["bank"] = _agg([s["terminal"]["cash"] for s in stmts])
        top["lambda_opp_reference"] = _agg([s["lambda_opp"]["reference"] for s in stmts
                                            if s["lambda_opp"]["reference"] is not None], 2)
        top["lambda_opp_marginal"] = _agg([s["lambda_opp"]["marginal_contribution_per_action"] for s in stmts
                                           if s["lambda_opp"]["marginal_contribution_per_action"] is not None], 2)
        marg = [s["lambda_opp"]["marginal_line"] for s in stmts if s["lambda_opp"]["marginal_line"]]
        top["lambda_opp_marginal_line"] = statistics.mode(marg) if marg else None
        top["fertilizer_allocation"] = {
            "gross_revenue": _agg([s["fertilizer_allocation"]["gross_revenue"] for s in stmts]),
            "units_collected": _agg([s["fertilizer_allocation"]["units_collected"] for s in stmts]),
            "unallocated_revenue": _agg([s["fertilizer_allocation"]["unallocated_revenue"] for s in stmts]),
            "allocated_by_line": {a: _agg([s["fertilizer_allocation"]["allocated_by_line"].get(a, 0.0)
                                           for s in stmts]) for a in ANIMAL_LINES},
            "collected_by_line": {a: _agg([s["fertilizer_allocation"]["collected_by_line"].get(a, 0)
                                           for s in stmts]) for a in ANIMAL_LINES},
        }
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
            keys = list(LINE_KEYS)
            if ln in ANIMAL_LINES:
                keys += ["feed_charged_bill", "feed_charged_margin", "feed_events"]
            if ln in CROP_LINES:
                keys += ["fert_charged_bill", "fertilize_events"]
            d = {}
            for k in keys:
                vals = [s["lines"][ln].get(k) for s in stmts]
                vals = [v for v in vals if v is not None]
                d[k] = _agg(vals)
            d["acts"] = {b: _agg([s["lines"][ln]["acts"][b] for s in stmts])
                         for b in ("produce", "maintain", "invest")}
            tot_u = sum(s["lines"][ln]["units_sold"] for s in stmts)
            tot_r = sum(s["lines"][ln]["sales_revenue"] for s in stmts)
            d["dollars_per_unit"] = round(tot_r / tot_u, 2) if tot_u else None
            ln_out[ln] = d
        top["lines"] = ln_out
        report["per_agent"][sp] = top
    return report


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

    report = aggregate(rows)
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
        # §3.1: the two denominators of value, never quoted as one number.
        gpa, cpa = t["gross_per_productive_action"], t["contribution_per_productive_action"]
        print(f"  $/productive action:  GROSS {gpa['mean'] if gpa else '-'}   "
              f"CONTRIBUTION {cpa['mean'] if cpa else '-'}  <- lambda_opp reference")
        lm, ll = t["lambda_opp_marginal"], t["lambda_opp_marginal_line"]
        print(f"  marginal line at the budget cut: {ll} @ ${lm['mean'] if lm else '-'}/action "
              f"(contribution)   hands/day {t['hands_mean']['mean']}")
        fa = t["fertilizer_allocation"]
        alloc = "  ".join(f"{a} {fa['allocated_by_line'][a]['mean']:,.0f}"
                          f"({fa['collected_by_line'][a]['mean']:.0f}u)" for a in ANIMAL_LINES)
        print(f"  fertilizer (joint product): gross ${fa['gross_revenue']['mean']:,.0f} on "
              f"{fa['units_collected']['mean']:.0f} collected units -> allocated {alloc}"
              f"   unallocated {fa['unallocated_revenue']['mean']:,.0f}")
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
        hdr = (f"  {'line':<11}{'sales$':>9}{'fert$':>8}{'gross$':>9}{'units':>7}{'$/u':>7}{'made':>7}"
               f"{'tileD':>7}{'%td':>6}{'acts':>7}{'%act':>6}{'seed$':>7}{'feed$':>8}{'capex$':>8}"
               f"{'contrib':>9}{'$g/act':>8}{'$c/act':>8}{'c/tileD':>9}")
        print(hdr)
        for ln, d in t["lines"].items():
            g = lambda k: (d[k]["mean"] if d.get(k) else 0)
            dpu = d["dollars_per_unit"]
            s_dpu = ("%.1f" % dpu) if dpu else "-"
            s_gac = ("%.1f" % d["gross_per_action"]["mean"]) if d.get("gross_per_action") else "-"
            s_cac = ("%.1f" % d["contribution_per_action"]["mean"]) if d.get("contribution_per_action") else "-"
            s_ctd = ("%.1f" % d["contribution_per_tile_day"]["mean"]) if d.get("contribution_per_tile_day") else "-"
            p_td = (100.0 * g("tile_days") / tot_td) if tot_td else 0
            p_ac = (100.0 * g("productive_actions") / tot_ac) if tot_ac else 0
            tag = "*" if ln == JOINT_PRODUCT_LINE else ""
            print(f"  {ln + tag:<11}{g('sales_revenue'):>9,.0f}{g('fertilizer_revenue_allocated'):>8,.0f}"
                  f"{g('gross_revenue'):>9,.0f}{g('units_sold'):>7.0f}{s_dpu:>7}{g('units_produced'):>7.0f}"
                  f"{g('tile_days'):>7.0f}{p_td:>5.1f}%{g('productive_actions'):>7.0f}{p_ac:>5.1f}%"
                  f"{g('seed_spend'):>7,.0f}{g('feed_charged_bill'):>8,.0f}{g('capex'):>8,.0f}"
                  f"{g('contribution'):>9,.0f}{s_gac:>8}{s_cac:>8}{s_ctd:>9}")
        print("  * FERTILIZER is a memo line: a joint product of an animal-day (ruling-02 §3.3). Its"
              " revenue sits in the\n    animal lines' fert$ column and its collection actions in"
              " their acts, so only resale of *bought* stock is left here.")


if __name__ == "__main__":
    raise SystemExit(main())
