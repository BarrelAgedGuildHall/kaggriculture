"""expB-304b-b0 — the offline refutation step (ruling-04 §2.1 / R-36, T-3).

**No Pit slot, no arm, no lease.**  This runs the unmodified Champion[B] locally
and asks three pre-registered questions of the drain-aware projection *before*
it is allowed to cost b1's single attempt at R-19.

  G-b0.1  Median absolute relative error of the projected landing quote < 25%,
          and the SIGN of (projected change from spot) matches realised in
          >= 80% of pairs, over >= 1,000 (product, day) pairs.
          FAILURE REFUTES THE RULE HERE and spends R-19's attempt.
  G-b0.2  The binding-constraint histogram.  If `price` binds on < 40% of
          STRAWBERRY's season turns, b1's arm is VOID-eligible (§2.2) — the rule
          cannot be tested where price does not bind.
  G-b0.3  The projection reproduces the two measured marginal-revenue points,
          MELON $47.4/unit and STRAWBERRY $147.3/unit, within +-30%, and emits a
          melon MR curve over 30-80 units/season.

Method, in one pass per episode
-------------------------------
The projection is computed FORWARD at every step, for every product, at that
product's structural lead: at step `t` we record what we think the quote will be
at `t + L_p`.  Harvest events are logged exactly (the engine's own
`_apply_unit_action` is patched, so hour-23 harvests are classified correctly --
a pre/post state diff around `env.step` is wrong at hour 23 and always will be).
At the end of the episode, every (product, day) at which we harvested is joined
against the projection recorded `L_p` steps earlier and against the REALISED
quote at that step.  Nothing is retained across episodes but the joined rows.

Two things this deliberately does not do:

  * it does not re-derive the town drain (every drain term is a
    `nav.DemandBook` call through `drain_project`), and
  * it does not let the projection see the future: `Nav` is fed observations in
    order and its curves only ever contain points up to the step at which the
    projection was taken.  That is what makes this a post-diction test rather
    than a fit.

Usage (from the repo root):
    uv run python agents/team_b/tools/b0_backtest.py --seeds 0-11 --workers 12 \
        --out-dir ../runs/expB-304b-b0

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(REPO_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_spec import arity_wrap, load_agent            # noqa: E402
from nav.api import Nav, NavConfig                       # noqa: E402
from nav.rules import CROPS, TURNS_PER_DAY               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drain_project as dp                               # noqa: E402

# The gated Champion[B] configuration, s32m6.  Kept here as the ONE literal in
# this file, and asserted against the shipped entry by `--assert-champion`.
CHAMPION_SPEC = ('millrace:feedchain{"melon_replant": true, "melon_rule": "fixed", '
                 '"melon_tiles": 6, "strawberry_rule": "fixed", "strawberry_tiles": 32, '
                 '"m5b_hire": true, "smart_endgame": true, "dispatch": "priced", '
                 '"dedupe_ontile": true, "dig": true, "ongoing_water_span": true}')

# ruling-04 §3.1's two measured marginal-revenue points, the calibration targets.
MR_TARGETS = {"MELON": 47.4, "STRAWBERRY": 147.3}
MR_TOL = 0.30


# --------------------------------------------------------------- binding constraint
def binding_label(product, want, standing, plantable_now, seeds, free_actions,
                  window_open, affordable):
    """Which constraint would have to be relaxed FIRST for standing acreage of
    `product` to rise on this turn.

    Two orderings are reported by the caller, because they answer different
    questions and G-b0.2 turns on the difference:

      economics-first (this function): if the rule itself says "no more", the
        turn is labelled `price` even if tiles were also short.  This is the
        reading G-b0.2's 40% threshold is written against — "does the price
        mechanism bind?" — and it is the GENEROUS one for the rule.
      physical-first (`binding_label_physical`): a hard resource that is
        actually exhausted wins over the rule's opinion.

    If the generous reading cannot clear 40%, the strict one certainly cannot,
    so a VOID verdict taken on this function is a verdict the arm cannot argue
    with.
    """
    if standing >= want:
        return "price"
    if not window_open:
        return "window"
    if plantable_now <= 0:
        return "tiles"
    if seeds <= 0 and not affordable:
        return "seeds"
    if free_actions <= 0:
        return "actions"
    return "open"


def binding_label_physical(product, want, standing, plantable_now, seeds, free_actions,
                           window_open, affordable):
    if not window_open:
        return "window"
    if plantable_now <= 0:
        return "tiles"
    if seeds <= 0 and not affordable:
        return "seeds"
    if free_actions <= 0:
        return "actions"
    if standing >= want:
        return "price"
    return "open"


# --------------------------------------------------------------- one episode
def run_episode(seed: int, steps: int, spec_a: str, spec_b: str) -> dict:
    os.environ.setdefault("MILLRACE_STRICT", "1")
    from _engine import import_engine, engine_version
    ke = import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    orig_apply = K._apply_unit_action
    orig_commit = K._commit_unit
    farm_ids: dict[int, int] = {}
    # per player: harvest events and our own executed market flow per step
    harvests = [[], []]
    own_flow = [{}, {}]          # step -> {product: net units into the pool}
    cur_step = [0]

    def patched_apply(farm, private, idx, action, board_size, day, turns_per_day_, shed_capacity):
        pid = farm_ids.get(id(farm))
        op = action[0] if isinstance(action, (list, tuple)) and action else "PASS"
        pre_tile = None
        pre_yield = 0
        pos = K._farmer_position(farm, idx)
        if op == "HARVEST" and pos:
            live = farm["tiles"][pos[1]][pos[0]]
            pre_tile = dict(live) if isinstance(live, dict) else live
            pre_yield = pre_tile.get("yield_units", 0) if isinstance(pre_tile, dict) else 0
        invs = private.get("inventories", [])
        pre_inv = dict(invs[idx]) if idx < len(invs) else {}
        r = orig_apply(farm, private, idx, action, board_size, day, turns_per_day_, shed_capacity)
        if op == "HARVEST" and pid is not None and isinstance(pre_tile, dict):
            invs = private.get("inventories", [])
            post_inv = dict(invs[idx]) if idx < len(invs) else {}
            if post_inv != pre_inv:                       # executed, not a no-op
                if "animal" in pre_tile:
                    prod = K.ANIMALS[pre_tile["animal"]]["product"]
                else:
                    prod = pre_tile.get("crop")
                if prod:
                    harvests[pid].append({"step": cur_step[0], "product": prod,
                                          "units": int(pre_yield)})
        return r

    def patched_commit(op, item, price, farm, private, market, shed_capacity=100):
        """One unit, one commit.  `own_flow` is what `nav.OpponentBook` needs to
        subtract before attributing the pool's move to the opponent — and it is
        the one term nav cannot read for itself."""
        pid = farm_ids.get(id(farm), -1)
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if ok and pid >= 0 and op in ("SELL", "BUY_PRODUCT"):
            d = own_flow[pid].setdefault(cur_step[0], {})
            d[item] = d.get(item, 0.0) + (1.0 if op == "SELL" else -1.0)
        return ok

    K._apply_unit_action = patched_apply
    K._commit_unit = patched_commit

    rows = {"projections": [], "binding": [], "realised_quote": {}, "realised_inv": {},
            "drain_path": {}, "harvests": None}
    try:
        env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed},
                      debug=False)
        orig_interpreter = env.interpreter

        def wrapped_interpreter(state, env_):
            farm_ids.clear()
            farms = getattr(state[0].observation, "farms", None)
            if farms:
                for i, f in enumerate(farms):
                    farm_ids[id(f)] = i
            return orig_interpreter(state, env_)

        env.interpreter = wrapped_interpreter
        raw_a, raw_b = load_agent(spec_a), load_agent(spec_b)
        agents = [arity_wrap(raw_a, env.configuration), arity_wrap(raw_b, env.configuration)]
        env.reset()

        # We instrument SEAT 0 only; the mirror gives seat 1 its own episode when
        # the caller swaps.  One Nav per instrumented seat, fed in order.
        nav = Nav(NavConfig())
        agent_state = getattr(raw_a, "state", None)

        while not env.done:
            obs0 = env.state[0].observation
            step = int(obs0.step)
            cur_step[0] = step
            flow = own_flow[0].get(step - 1)
            nav.observe(obs0, own_flow=flow)

            market = obs0.market
            prices = market["prices"] if isinstance(market, dict) else market.prices
            rows["realised_quote"][step] = {p: int(v) for p, v in prices.items()}

            farm0 = obs0.farms[0]
            farm1 = obs0.farms[1] if len(obs0.farms) > 1 else None
            priv0 = obs0.private
            inv_map = market["inventory"] if isinstance(market, dict) else market.inventory
            rows["realised_inv"][step] = {p: int(v) for p, v in inv_map.items()}
            # The REALISED drain path.  The unlock draw is the dominant source of
            # season-to-season variance in a drain-limited product (measured:
            # STRAWBERRY ends between 13 and 31 units/day across four seeds), so a
            # season model that uses the final unlock set for the whole season is
            # wrong at both ends.  Sampled once a day; the set only moves on a draw.
            if step % TURNS_PER_DAY == 0:
                rows["drain_path"][step] = {p: nav.book.drain_per_day(p)
                                            for p in dp.LEAD_STEPS}
            for product, lead in dp.LEAD_STEPS.items():
                pr = dp.project(nav, product, lead=lead, farm=farm0, private=priv0,
                                their_farm=farm1)
                rows["projections"].append(pr.as_row())

            # The emitted action is what makes `actions` a MEASUREMENT rather than
            # a guess: a unit that emitted PASS was available to plant and did
            # not, so on that turn the action budget did not bind.
            act0 = agents[0](env.state[0].observation)
            act1 = agents[1](env.state[1].observation)
            if agent_state is not None:
                rows["binding"].append(_binding_row(agent_state, obs0, step, act0, nav,
                                                    farm0, priv0, farm1))
            env.step([act0, act1])

        rows["harvests"] = harvests[0]
        _town = env.state[0].observation.town
        rows["final_shops"] = list((_town.get("unlocked_shops", []) if isinstance(_town, dict)
                                    else getattr(_town, "unlocked_shops", [])) or [])
        rows["money"] = [float(env.state[0].observation.farms[i]["money"]) for i in (0, 1)]
        rows["engine"] = engine_version()
        rows["seed"] = seed
        rows["plan_exceptions"] = int(getattr(raw_a, "plan_exceptions", 0) or 0)
    finally:
        K._apply_unit_action = orig_apply
        K._commit_unit = orig_commit
    return rows


def shadow_want(nav, product, standing, hurdle, y_est, acts_per_cycle, farm, private,
                their_farm):
    """What the DRAIN-AWARE rule would want this turn, given the same hurdle the
    champion's own ledger is running on.

    This is what makes G-b0.2 interpretable.  The champion's `want` is the §0
    constant 32, so on the shipped build the price mechanism cannot bind *by
    construction* — measuring `price` against the constant would answer a
    question nobody asked.  The shadow rule is b1's rule, evaluated offline on
    the champion's own board, so "does price bind?" becomes a statement about the
    mechanism b1 would actually ship.

    Structure is expB-220's melon machinery verbatim (grow while the next tile's
    cycle clears the hurdle per productive action, then shrink), with the
    marginal revenue read at the PROJECTED landing pool rather than the spot one.
    """
    curve = nav.curves[product]
    lead = dp.LEAD_STEPS.get(product, TURNS_PER_DAY)
    pr = dp.project(nav, product, lead=lead, farm=farm, private=private,
                    their_farm=their_farm)
    inv = int(round(pr.inv_land))

    def tile_mr(k):
        return (curve.marginal_revenue(inv, int(round((k + 1) * y_est)))
                - curve.marginal_revenue(inv, int(round(k * y_est))))

    want = standing
    while want < standing + 10:
        if tile_mr(want) / acts_per_cycle <= hurdle:
            break
        want += 1
    while want > 0:
        if tile_mr(want - 1) / acts_per_cycle > hurdle:
            break
        want -= 1
    return want, pr


def _binding_row(agent_state, obs, step, action, nav, farm0, priv0, farm1):
    """`want` / `standing` / `plantable_now` / `seeds` / `free_unit_actions` and
    the binding labels, for the two products the ruling cares about."""
    st = agent_state.get("st", {})
    farm = farm0
    priv = priv0
    day = step // TURNS_PER_DAY
    last_day = 29
    counts, empties, locked = {}, 0, 0
    for row in farm["tiles"]:
        for t in row:
            if t == "LOCKED":
                locked += 1
            elif t is None:
                empties += 1
            elif isinstance(t, dict) and "crop" in t:
                counts[t["crop"]] = counts.get(t["crop"], 0) + 1
    n_units = 1 + len(farm.get("hands", []) or [])
    # MEASURED idleness: units that actually emitted PASS this turn were free to
    # plant and did not.  `n_units` alone is not an action budget -- most units
    # are already committed to the service queue -- and using it was what made
    # 42% of strawberry turns read as an uninformative `open`.
    acts = [action.get("farmer")] + list(action.get("hands") or []) if isinstance(action, dict) else []
    idle_units = sum(1 for a in acts if isinstance(a, list) and a and a[0] == "PASS")
    money = float(farm.get("money", 0))
    seeds = priv.get("seeds", {}) or {}
    # the hurdle the agent's own ledger is running on, as logged per day
    lam_log = st.get("lam_log") or []
    last_lam = lam_log[-1] if lam_log else {}
    hurdle = float(last_lam.get("lam_cut") or last_lam.get("lam_opp") or 0.0)
    out = {"step": step, "day": day, "empties": empties, "locked": locked,
           "units": n_units, "idle_units": idle_units, "money": money,
           "hurdle": hurdle, "products": {}}
    for product, want_key in (("STRAWBERRY", "want_straw"), ("MELON", "want_melon")):
        want = int(st.get(want_key, -1))
        standing = int(counts.get(product, 0))
        held = int(seeds.get(product, 0))
        spec = CROPS[product]
        window_open = (day + int(spec["first_yield_day"])) <= last_day
        affordable = money >= spec["seed"]
        seen = st.get("straw_harvest_units" if product == "STRAWBERRY" else "melon_harvest_units") or []
        y_est = max(1.0, (sum(seen) / len(seen)) if seen else float(spec["max_yield"] * 2))
        acts_per_cycle = (2.0 + spec["first_yield_day"]) if not spec["ongoing"] else \
            (1.0 + spec["first_yield_day"] + spec["max_yield"]
             + spec["max_yield"] // max(1, spec["interval"]))
        want_sh, pr = (want, None)
        if hurdle > 0:
            try:
                want_sh, pr = shadow_want(nav, product, standing, hurdle, y_est,
                                          acts_per_cycle, farm, priv, farm1)
            except Exception:
                want_sh, pr = want, None
        args = (product, want, standing, empties, held, idle_units, window_open, affordable)
        args_sh = (product, want_sh, standing, empties, held, idle_units, window_open, affordable)
        out["products"][product] = {
            "want": want, "want_shadow": int(want_sh), "standing": standing, "seeds": held,
            "plantable_now": empties, "free_unit_actions": idle_units,
            "window_open": window_open, "affordable": affordable,
            "inv_land": (round(pr.inv_land, 2) if pr else None),
            "binding_shadow": binding_label(*args_sh),
            "binding_shadow_physical": binding_label_physical(*args_sh),
            "binding": binding_label(*args),
            "binding_physical": binding_label_physical(*args),
        }
    return out


# --------------------------------------------------------------- the join + gates
def join_pairs(ep: dict) -> list[dict]:
    """One row per (product, day) at which we harvested: the projection taken
    `L_p` steps earlier vs the realised quote at the landing step."""
    by_key = {}
    for h in ep["harvests"] or []:
        by_key.setdefault((h["product"], h["step"] // TURNS_PER_DAY), h["step"])
    proj_index = {}
    for r in ep["projections"]:
        proj_index[(r["product"], r["land_step"])] = r
    quotes = {int(k): v for k, v in ep["realised_quote"].items()}

    out = []
    for (product, day), land_step in sorted(by_key.items()):
        pr = proj_index.get((product, land_step))
        if pr is None:
            continue                       # projection window started before step 0
        realised = quotes.get(land_step, {}).get(product)
        if realised is None:
            continue
        projected = pr["quote_land"]
        # The sign test compares the projected MOVE against the realised move, so
        # both ends must come from the same model.  `pr["spot"]` is
        # `price_at(inv_now)`, which returns the EXACT observed point when there
        # is one, while `quote_land` is always the fit -- differencing them mixes
        # fit error into the sign and, on a product whose quote moves by a dollar
        # a day (WHEAT), that error is larger than the signal being tested.
        spot = (dp.price_from_state(pr["curve"], pr["inv_now"]) if pr.get("curve")
                else pr["spot"])
        are = abs(projected - realised) / max(1.0, abs(realised))
        # --- the decomposition (see `drain_project.curve_state`) -------------
        # `inv_real` is the pool the units ACTUALLY landed into.  Feeding it back
        # through the SAME frozen curve isolates how much of the quote error is
        # the inventory model and how much is the curve extrapolating past the
        # range it has evidence for.
        inv_real = (ep["realised_inv"].get(land_step)
                    or ep["realised_inv"].get(str(land_step)) or {}).get(product)
        are_curve = inv_err = None
        if inv_real is not None and pr.get("curve"):
            q_at_truth = dp.price_from_state(pr["curve"], inv_real)
            are_curve = abs(q_at_truth - realised) / max(1.0, abs(realised))
            inv_err = abs(pr["inv_land"] - inv_real) / max(1.0, abs(inv_real))
        d_proj, d_real = projected - spot, realised - spot
        # A realised move inside the engine's own integer rounding is not a sign
        # the projection can be asked to call.  `flat` is 1% of the price or half
        # a dollar, whichever is larger; both the strict (nothing is flat) and
        # the relative reading are reported so the verdict's sensitivity to this
        # choice is visible rather than assumed.
        flat_tol = max(0.5, 0.01 * abs(realised))
        flat = abs(d_real) < flat_tol
        sign_ok = (abs(d_proj) < flat_tol) if flat else (d_proj * d_real > 0)
        sign_ok_strict = d_proj * d_real > 0
        out.append({
            "seed": ep["seed"], "product": product, "day": day, "land_step": land_step,
            "from_step": pr["step"], "spot": spot, "projected": projected,
            "realised": float(realised), "are": are, "sign_ok": bool(sign_ok),
            "sign_ok_strict": bool(sign_ok_strict),
            "flat": bool(flat), "provisional": pr["provisional"],
            "exact_drain": pr["exact_drain"], "posterior_drain": pr["posterior_drain"],
            "own_pipeline": pr["own_pipeline"], "opponent_est": pr["opponent_est"],
            "inv_land": pr["inv_land"], "inv_real": inv_real,
            "inv_rel_err": inv_err, "are_curve": are_curve,
            "obs_max": pr["curve"].get("obs_max") if pr.get("curve") else None,
        })
    return out


def _drain_census(episodes):
    """Realised town drain per product per season, and its spread across seeds.

    The single most decision-relevant table b0 produces: a product whose drain is
    a constant needs no projection, and a product whose drain swings 2.4x on the
    unlock draw cannot be sized by a constant.
    """
    out = {}
    for ep in episodes:
        book = _book_for(ep)
        for product in dp.LEAD_STEPS:
            out.setdefault(product, []).append(book.drain_over_steps(product, 0, 719))
    rows = {}
    for product, vals in sorted(out.items()):
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
        rows[product] = {"n": n, "season_drain_mean": round(mean, 1),
                         "season_drain_std": round(var ** 0.5, 1),
                         "min": round(min(vals), 1), "max": round(max(vals), 1),
                         "spread_ratio": round(max(vals) / max(1e-9, min(vals)), 2)}
    return rows


def _median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


SIGN_BANDS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 1e9))
MATERIAL = 0.05        # a realised move worth calling, as a fraction of the price


def sign_by_materiality(pairs):
    """Sign-match as a function of how big the realised move was.

    A sign test on a projection whose typical LEVEL error is a few percent cannot
    be read as a single number: on every pair where the realised move is smaller
    than the projection's own error bar the answer is a coin flip, and those
    pairs dominate the count.  Reporting the breakdown is what separates "the
    projection does not know which way the pool is going" from "the projection is
    being asked to call moves smaller than its resolution".
    """
    out = []
    for lo, hi in SIGN_BANDS:
        sel = [p for p in pairs
               if lo <= abs(p["realised"] - p["spot"]) / max(1.0, p["realised"]) < hi]
        if not sel:
            continue
        out.append({
            "band": f"{lo:.0%}-{'inf' if hi > 1 else format(hi, '.0%')}",
            "n": len(sel),
            "sign_match": sum(1 for p in sel if p["sign_ok_strict"]) / len(sel),
            "med_are": _median([p["are"] for p in sel]),
        })
    return out


def gate_b0_1(pairs: list[dict]) -> dict:
    ares = [p["are"] for p in pairs]
    med = _median(ares)
    sign = sum(1 for p in pairs if p["sign_ok"]) / len(pairs) if pairs else float("nan")
    n = len(pairs)
    ok = (n >= 1000) and (med < 0.25) and (sign >= 0.80)
    per_product = {}
    for p in pairs:
        per_product.setdefault(p["product"], []).append(p)
    inv_errs = [p["inv_rel_err"] for p in pairs if p.get("inv_rel_err") is not None]
    curve_errs = [p["are_curve"] for p in pairs if p.get("are_curve") is not None]
    return {
        "n_pairs": n, "med_are": med, "sign_match": sign,
        # THE DECOMPOSITION: `med_are_curve` is the error left when the true
        # landing inventory is fed through the same frozen curve, i.e. pure
        # extrapolation error; `med_inv_rel_err` is the inventory model's own.
        "med_are_curve": _median(curve_errs), "med_inv_rel_err": _median(inv_errs),
        "sign_match_strict": (sum(1 for p in pairs if p["sign_ok_strict"]) / len(pairs)
                              if pairs else float("nan")),
        "sign_by_materiality": sign_by_materiality(pairs),
        "sign_match_material": (
            sum(1 for p in pairs
                if abs(p["realised"] - p["spot"]) / max(1.0, p["realised"]) >= MATERIAL
                and p["sign_ok_strict"])
            / max(1, sum(1 for p in pairs
                         if abs(p["realised"] - p["spot"]) / max(1.0, p["realised"]) >= MATERIAL))),
        "n_material": sum(1 for p in pairs
                          if abs(p["realised"] - p["spot"]) / max(1.0, p["realised"]) >= MATERIAL),
        "n_flat": sum(1 for p in pairs if p["flat"]),
        "verdict": "PASS" if ok else "FAIL",
        "criteria": {"n>=1000": n >= 1000, "medARE<0.25": med < 0.25, "sign>=0.80": sign >= 0.80},
        "per_product": {
            k: {"n": len(v), "med_are": _median([x["are"] for x in v]),
                "med_are_curve": _median([x["are_curve"] for x in v
                                          if x.get("are_curve") is not None]),
                "med_inv_rel_err": _median([x["inv_rel_err"] for x in v
                                            if x.get("inv_rel_err") is not None]),
                "sign_match": sum(1 for x in v if x["sign_ok"]) / len(v),
                "n_material": sum(1 for x in v if abs(x["realised"] - x["spot"])
                                  / max(1.0, x["realised"]) >= MATERIAL),
                "sign_match_material": (
                    sum(1 for x in v if abs(x["realised"] - x["spot"])
                        / max(1.0, x["realised"]) >= MATERIAL and x["sign_ok_strict"])
                    / max(1, sum(1 for x in v if abs(x["realised"] - x["spot"])
                                 / max(1.0, x["realised"]) >= MATERIAL)))}
            for k, v in sorted(per_product.items())
        },
    }


def gate_b0_2(binding_rows: list[dict]) -> dict:
    """The binding-constraint histogram, reported FOUR ways.

    `binding` is measured against the champion's own `want`, which on the shipped
    build is the §0 constant 32 -- so `price` there is a statement about the
    constant.  `binding_shadow` is measured against what the drain-aware rule
    would want on the same board, which is the mechanism b1 would ship, and it is
    the one G-b0.2's 40% threshold is adjudicated on.  Both are reported under
    both label orderings so a reader can see that the verdict does not turn on
    the choice.
    """
    out = {}
    for product in ("STRAWBERRY", "MELON"):
        hists = {k: {} for k in ("binding", "binding_physical",
                                 "binding_shadow", "binding_shadow_physical")}
        wants, wants_sh, standings = [], [], []
        for r in binding_rows:
            e = r["products"].get(product)
            if not e:
                continue
            for k in hists:
                v = e.get(k)
                if v is not None:
                    hists[k][v] = hists[k].get(v, 0) + 1
            wants.append(e["want"])
            wants_sh.append(e.get("want_shadow", e["want"]))
            standings.append(e["standing"])
        n = sum(hists["binding"].values()) or 1
        out[product] = {
            "n_turns": n,
            "mean_want_constant": sum(wants) / max(1, len(wants)),
            "mean_want_shadow": sum(wants_sh) / max(1, len(wants_sh)),
            "mean_standing": sum(standings) / max(1, len(standings)),
        }
        for k, h in hists.items():
            out[product][k + "_share"] = {kk: round(vv / n, 4)
                                          for kk, vv in sorted(h.items(), key=lambda kv: -kv[1])}
            out[product][k + "_price_share"] = h.get("price", 0) / n

    ss = out.get("STRAWBERRY", {}).get("binding_shadow_price_share", 0.0)
    out["strawberry_price_share_shadow"] = ss
    out["strawberry_price_share_constant"] = out.get("STRAWBERRY", {}).get("binding_price_share", 0.0)
    out["verdict"] = "PRICE-BINDS" if ss >= 0.40 else "VOID-ELIGIBLE"
    return out


def gate_b0_3(episodes: list[dict]) -> dict:
    """Reproduce ruling-04 §3.1's two marginal-revenue points, and emit the melon
    MR curve over 30-80 units/season.

    The ruling's numbers are SEGMENT AVERAGES between two measured arms:

        MELON       (15,524 - 9,644) / (163.6 - 39.6) = $47.4/unit
        STRAWBERRY  (34,026 - 15,579) / (190.4 - 65.2) = $147.3/unit

    So the faithful reproduction is not a tangent on the curve at some
    inventory -- it is the same difference of two SEASON revenues, computed from
    our fitted curve with the town draining underneath, which is what
    `drain_project.season_revenue` models.  Computing it as a block sale instead
    (the obvious reading) puts a 190-unit strawberry season at the price floor,
    because over thirty days the town removes a large multiple of the standing
    pool -- the very effect the drain-aware rule exists to price.
    """
    SEGMENTS = {
        "MELON": (39.6, 163.6, 47.4),
        "STRAWBERRY": (65.2, 190.4, 147.3),
    }
    SELL_FIRST, SELL_LAST = 240, 719      # the selling season for a 10-day-lead crop

    out = {"targets": MR_TARGETS, "tolerance": MR_TOL, "measured": {}, "melon_curve": [],
           "method": "segment average of two drain-aware season revenues (ruling-04 section 3.1)"}
    for product, (lo_u, hi_u, target) in SEGMENTS.items():
        vals = []
        for ep in episodes:
            curve = _curve_for(ep, product)
            book = _book_for(ep)
            if curve is None or curve.base is None:
                continue
            start_inv = ep["realised_inv"].get(str(SELL_FIRST)) or ep["realised_inv"].get(SELL_FIRST)
            if not start_inv:
                continue
            i0 = int(start_inv[product])
            r_hi = dp.season_revenue(curve, book, product, hi_u, i0, SELL_FIRST, SELL_LAST)
            r_lo = dp.season_revenue(curve, book, product, lo_u, i0, SELL_FIRST, SELL_LAST)
            vals.append((r_hi - r_lo) / (hi_u - lo_u))
        if vals:
            m = sum(vals) / len(vals)
            rel = abs(m - target) / target
            out["measured"][product] = {"n_seasons": len(vals), "mr_per_unit": m,
                                        "target": target, "rel_err": rel,
                                        "verdict": "PASS" if rel <= MR_TOL else "FAIL"}
        else:
            out["measured"][product] = {"n_seasons": 0, "verdict": "NO-DATA"}

    # R-41: the melon MR curve over 30-80 units/season, which is the interval the
    # ruling says is two-thirds as wide as the price itself.
    curve_pts = {}
    for ep in episodes:
        curve = _curve_for(ep, "MELON")
        book = _book_for(ep)
        start_inv = ep["realised_inv"].get(str(SELL_FIRST)) or ep["realised_inv"].get(SELL_FIRST)
        if curve is None or curve.base is None or not start_inv:
            continue
        i0 = int(start_inv["MELON"])
        for u in range(30, 81, 5):
            r1 = dp.season_revenue(curve, book, "MELON", u + 5, i0, SELL_FIRST, SELL_LAST)
            r0 = dp.season_revenue(curve, book, "MELON", u, i0, SELL_FIRST, SELL_LAST)
            curve_pts.setdefault(u, []).append((r1 - r0) / 5.0)
    out["melon_curve"] = [{"units_per_season": u, "mr_per_unit": round(sum(v) / len(v), 2),
                           "n": len(v)} for u, v in sorted(curve_pts.items())]
    verdicts = [v.get("verdict") for v in out["measured"].values()]
    out["verdict"] = "PASS" if all(v == "PASS" for v in verdicts) else "FAIL"
    return out


_BOOK_CACHE: dict = {}


class _RealisedBook:
    """A `DemandBook`-shaped object that answers `drain_over_steps` from the
    episode's OWN daily drain path instead of from one unlock set.

    Same interface, so `drain_project.season_revenue` is unchanged; it just stops
    pretending the town had its day-29 appetite on day 10.
    """

    def __init__(self, path: dict):
        self.path = {int(k): v for k, v in path.items()}
        self.days = sorted(self.path)
        self.shop_interval = 4

    def _rate(self, product, step):
        d = None
        for s in self.days:
            if s <= step:
                d = s
            else:
                break
        return float(self.path.get(d, {}).get(product, 0.0)) if d is not None else 0.0

    def drain_over_steps(self, product, first_step, last_step):
        if last_step < first_step:
            return 0.0
        total = 0.0
        for s in range(int(first_step), int(last_step) + 1):
            total += self._rate(product, s) / TURNS_PER_DAY
        return total


def _book_for(ep):
    """The episode's realised drain path (see `_RealisedBook`)."""
    key = ep["seed"]
    if key not in _BOOK_CACHE:
        _BOOK_CACHE[key] = _RealisedBook(ep.get("drain_path") or {})
    return _BOOK_CACHE[key]


_CURVE_CACHE: dict = {}


def _curve_for(ep, product):
    """Rebuild the product's fitted curve from the episode's recorded quote
    stream — the same online object, replayed."""
    key = (ep["seed"], product)
    if key in _CURVE_CACHE:
        return _CURVE_CACHE[key]
    from nav.curves import PriceCurve  # noqa: PLC0415
    inv_by_step = {}
    for r in ep["projections"]:
        if r["product"] == product:
            inv_by_step[r["step"]] = r["inv_now"]
    quotes = {int(k): v for k, v in ep["realised_quote"].items()}
    c = PriceCurve(product)
    for step in sorted(inv_by_step):
        q = quotes.get(step, {}).get(product)
        if q is None:
            continue
        c.observe(inv_by_step[step], int(q), step=step)
    _CURVE_CACHE[key] = c
    return c


# --------------------------------------------------------------- driver
def _worker(args):
    seed, steps, spec_a, spec_b = args
    t0 = time.perf_counter()
    ep = run_episode(seed, steps, spec_a, spec_b)
    ep["wall_s"] = round(time.perf_counter() - t0, 1)
    return ep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--a", default=CHAMPION_SPEC, help="instrumented seat (seat 0)")
    ap.add_argument("--b", default=CHAMPION_SPEC, help="opponent seat (mirror by default)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    seeds = []
    for part in a.seeds.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, a.steps, a.a, a.b) for s in seeds]
    episodes = []
    t0 = time.perf_counter()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for ep in ex.map(_worker, jobs):
                episodes.append(ep)
                print(f"  seed {ep['seed']}: money {ep['money']} "
                      f"harvest_events {len(ep['harvests'])} "
                      f"plan_exceptions {ep['plan_exceptions']} ({ep['wall_s']}s)", flush=True)
    else:
        for j in jobs:
            ep = _worker(j)
            episodes.append(ep)
            print(f"  seed {ep['seed']}: money {ep['money']} "
                  f"harvest_events {len(ep['harvests'])}", flush=True)

    exc = sum(ep["plan_exceptions"] for ep in episodes)
    pairs = [row for ep in episodes for row in join_pairs(ep)]
    binding = [r for ep in episodes for r in ep["binding"]]

    report = {
        "experiment": "expB-304b-b0",
        "engine": episodes[0]["engine"] if episodes else None,
        "agent": a.a, "opponent": a.b, "seeds": a.seeds,
        "n_episodes": len(episodes), "n_seat_seasons": len(episodes),
        "plan_exceptions_total": exc,
        "wall_s": round(time.perf_counter() - t0, 1),
        "drain_census": _drain_census(episodes),
        "G_b0_1": gate_b0_1(pairs),
        "G_b0_2": gate_b0_2(binding),
        "G_b0_3": gate_b0_3(episodes),
    }
    if exc:
        report["HALT"] = "plan_exceptions > 0 (ruling-04 C-2): this run FAILS its gate"

    (out / "pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    g1, g2, g3 = report["G_b0_1"], report["G_b0_2"], report["G_b0_3"]
    print("\n=== expB-304b-b0 ===")
    print(f"G-b0.1  n={g1['n_pairs']}  medARE={g1['med_are']:.4f}  "
          f"sign={g1['sign_match']:.4f}  -> {g1['verdict']}")
    print(f"          decomposition: curve-only medARE={g1['med_are_curve']:.4f}  "
          f"inventory med rel-err={g1['med_inv_rel_err']:.4f}")
    print(f"          sign on MATERIAL moves (>={MATERIAL:.0%}): "
          f"{g1['sign_match_material']:.4f} (n={g1['n_material']})")
    for b in g1["sign_by_materiality"]:
        print(f"            move {b['band']:<9} n={b['n']:<5} sign={b['sign_match']:.3f} "
              f"medARE={b['med_are']:.4f}")
    for k, v in g1["per_product"].items():
        print(f"          {k:<11} n={v['n']:<5} medARE={v['med_are']:.4f} "
              f"(curve {v['med_are_curve']:.4f} / inv {v['med_inv_rel_err']:.4f}) "
              f"sign={v['sign_match']:.3f} material({v['n_material']})={v['sign_match_material']:.3f}")
    print(f"G-b0.2  STRAWBERRY price-binds: shadow rule {g2['strawberry_price_share_shadow']:.4f} "
          f"/ shipped constant {g2['strawberry_price_share_constant']:.4f} -> {g2['verdict']}")
    for p in ("STRAWBERRY", "MELON"):
        e = g2[p]
        print(f"          {p}: want const {e['mean_want_constant']:.1f} / shadow "
              f"{e['mean_want_shadow']:.1f} / standing {e['mean_standing']:.2f}")
        print(f"            shadow    {e['binding_shadow_share']}")
        print(f"            constant  {e['binding_share']}")
    print(f"G-b0.3  {g3['verdict']}")
    for k, v in g3["measured"].items():
        if v.get("n_seasons"):
            print(f"          {k:<11} MR={v['mr_per_unit']:.2f} target={v['target']} "
                  f"rel={v['rel_err']:.3f} {v['verdict']}")
    print("          melon MR curve (units/season -> $/unit): "
          + ", ".join(f"{r['units_per_season']}:{r['mr_per_unit']:.0f}" for r in g3["melon_curve"]))
    print(f"\nwrote {out/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
