"""expB-304b-seeds / R-51 — why does STRAWBERRY run out of seed?

`seeds` binds 14.1% of STRAWBERRY turns (expB-304b-b0's binder histogram) and the
seed price is the engine's one FIXED money constant, so a stock-out on it is a
leak rather than an optimisation.  R-51 forbids assuming the cause: v3.1 already
cost 45pp by assuming price where the binding constraint was cash.  This tool
counts, it does not model.

A turn is a STOCK-OUT for a crop when the acreage rule wants more standing
acreage than it has, the replant window is open, and we hold ZERO seed.  Each
one is attributed to exactly one cause, taken from the agent's own seed-buy row
for that turn (`MILLRACE_SEEDLOG=1`) plus the engine's commit outcome:

  cash_short_planner   the row asked, and `budget // seed_price == 0`
  cash_short_engine    the order was placed and `_commit_unit` refused it
                       (`farm["money"] < price` at the moment it was quoted)
  crowded_out          `budget_open >= price` but an earlier row (feed, animals,
                       land, hire) had already spent it down below one seed.
                       This is a SEQUENCING cause, not a shortfall.
  slot_full            the order queue was already at the 10-order engine cap
  capex_stop           `capex_ok` was false (endgame / capex_stop_day)
  cap_satisfied        `need == 0`: the §0 tile constant says we hold enough
  ordered              seed was ordered on this turn (a same-turn stock-out that
                       resolves inside the same market phase)

The engine fact the `crowded_out` bucket rests on: `_process_market` walks the
order list IN ORDER and `_commit_unit` mutates `farm["money"]` immediately, so a
SELL at index 0 funds a BUY_SEED at index 4 within the same turn.  Millrace
appends SELLs first and prices its purse at `money - cash_reserve + 0.8 x
expected_rev`, so unlike the failure Team A's census found, our buys are NOT
quoted against pre-sale cash.  What we can still lose to is order-of-spending
inside our own turn, and that is what `crowded_out` measures.

Usage (repo root):
  uv run python agents/team_b/tools/seed_stockout.py --seeds 0-7 --both-seats \\
      --workers 8 --out bench/runs/expB-304b-seeds/report.json

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))

from cap_bind import CHAMPION, PRICER, TURNS_PER_DAY, _counts  # noqa: E402

CROPS_WATCHED = ("STRAWBERRY", "MELON", "WHEAT", "CARROT")


def run_episode(seed: int, seat: int, steps: int) -> dict:
    os.environ["MILLRACE_SEEDLOG"] = "1"
    os.environ.setdefault("MILLRACE_STRICT", "1")
    from _engine import import_engine, engine_version
    from millrace.feedchain import make_feedchain_agent
    from millrace.probe_core import CROPS

    ke = import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    orig_commit = K._commit_unit
    farm_ids: dict[int, int] = {}
    cur_step = [0]
    commits = []          # every BUY_SEED quote our seat saw, and its outcome

    def patched_commit(op, item, price, farm, private, market, shed_capacity=100):
        pid = farm_ids.get(id(farm), -1)
        money_before = float(farm.get("money", 0))
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if pid == seat and op == "BUY_SEED":
            commits.append({"step": cur_step[0], "item": item, "price": float(price),
                            "money_before": money_before, "ok": bool(ok)})
        if pid == seat and op == "SELL" and ok:
            # post-sale cash inside the same market phase: the number the
            # planner's 0.8 x expected_rev is an estimate OF.
            peak[cur_step[0]] = max(peak.get(cur_step[0], 0.0), float(farm.get("money", 0)))
        return ok

    peak: dict[int, float] = {}
    K._commit_unit = patched_commit
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
        champ = make_feedchain_agent(CHAMPION)
        opp = make_feedchain_agent(PRICER)
        env.reset()
        cfg = env.configuration
        turns = []
        while not env.done:
            obs_us = env.state[seat].observation
            obs_them = env.state[1 - seat].observation
            # `step` is a SHARED observation field, present on seat 0 only
            # (measured); day/hour are per-seat, so derive it.
            step = int(obs_us["day"]) * TURNS_PER_DAY + int(obs_us["hour"])
            cur_step[0] = step
            n_log = len(champ.state["st"].get("seedlog", []))
            act_us = champ(obs_us, cfg)
            act_them = opp(obs_them, cfg)
            farm = obs_us.farms[seat]
            counts, empties = _counts(farm)
            held = obs_us.private.get("seeds", {}) or {}
            st = champ.state["st"]
            turns.append({
                "step": step, "day": step // TURNS_PER_DAY, "hour": step % TURNS_PER_DAY,
                "money": float(farm.get("money", 0)), "empties": empties,
                "standing": {c: int(counts.get(c, 0)) for c in CROPS_WATCHED},
                "held": {c: int(held.get(c, 0)) for c in CROPS_WATCHED},
                "want": {"STRAWBERRY": int(st.get("want_straw", -1)),
                         "MELON": int(st.get("want_melon", -1))},
                # structural replant window: a tile planted on day d first
                # yields at d + first_yield_day, so seed has value while that
                # still lands in season. WHEAT/CARROT carry no window gate.
                "window": {c: (bool(step // TURNS_PER_DAY + CROPS[c]["first"] <= 29)
                               if c in ("STRAWBERRY", "MELON") else True)
                           for c in CROPS_WATCHED},
                "rows": list(st.get("seedlog", [])[n_log:]),
                "orders": [o for o in (act_us.get("market") or [])],
            })
            env.step([act_us, act_them] if seat == 0 else [act_them, act_us])
    finally:
        K._commit_unit = orig_commit
        os.environ.pop("MILLRACE_SEEDLOG", None)

    by_step = {}
    for c in commits:
        by_step.setdefault(c["step"], []).append(c)
    for t in turns:
        t["commits"] = by_step.get(t["step"], [])
        t["post_sale_cash"] = peak.get(t["step"])
    return {"seed": seed, "seat": seat, "engine": engine_version(),
            "money": [float(env.state[0].observation.farms[i]["money"]) for i in (0, 1)],
            "plan_exceptions": int(getattr(champ, "plan_exceptions", 0) or 0),
            "turns": turns}


def classify(turn: dict, crop: str) -> tuple[str, int] | None:
    """One cause for one stock-out turn, or None if this turn is not one.

    A stock-out is `held == 0` on a turn where the seed still has value — i.e.
    the replant window is open.  A closed window is not a stock-out: seed bought
    then could never be harvested.
    """
    if turn["held"][crop] > 0 or not turn["window"].get(crop, True):
        return None
    rows = [r for r in turn["rows"] if r.get("crop") == crop]
    if not rows:
        return ("capex_stop", 0) if any(r.get("gate") for r in turn["rows"]) else None
    r = rows[-1]
    if r["need"] <= 0:
        # the rule says we hold enough standing acreage: the §0 tile constant,
        # not a resource, is what stops the order.
        return ("cap_satisfied", 0)
    short = int(r["need"] - r["k"])
    if r["k"] <= 0:
        if r["budget"] < r["px"]:
            if r["budget_open"] >= r["px"]:
                return ("crowded_out", short)
            return ("cash_short_planner", short)
        return ("unclassified", short)                 # need>0, budget ok, k==0
    if not r["placed"]:
        return ("slot_full", short)
    if not any(o[0] == "BUY_SEED" and o[1] == crop for o in turn["orders"]):
        # `chassis.sanitise` truncates the market list at 10 with SELLs kept
        # preferentially, so a BUY_SEED the planner queued can still be dropped
        # after `_plan` returns.
        return ("dropped_by_sanitiser", short)
    if any(c["item"] == crop and not c["ok"] for c in turn["commits"]):
        return ("cash_short_engine", short)
    return ("ordered", short)


def summarise(eps: list[dict]) -> dict:
    out = {}
    for crop in CROPS_WATCHED:
        cause = Counter()
        units = Counter()
        by_day: dict[str, Counter] = {}
        turns_by_day = Counter()
        n_turns = 0
        recoverable = 0        # crowded_out / slot_full turns where the cash existed
        money_on_cash_short = []
        for ep in eps:
            for t in ep["turns"]:
                n_turns += 1
                turns_by_day[t["day"]] += 1
                got = classify(t, crop)
                if got is None:
                    continue
                c, short = got
                cause[c] += 1
                units[c] += short
                by_day.setdefault(c, Counter())[t["day"]] += 1
                if c.startswith("cash_short"):
                    money_on_cash_short.append(t["money"])
                if c in ("crowded_out", "slot_full"):
                    px = next((r["px"] for r in t["rows"] if r.get("crop") == crop), 0)
                    if max(t["post_sale_cash"] or 0.0, t["money"]) >= px:
                        recoverable += 1
        stock_outs = sum(v for k, v in cause.items() if k != "cap_satisfied")
        ms = sorted(money_on_cash_short)
        out[crop] = {
            "turns": n_turns,
            "stock_out_turns": stock_outs,
            "stock_out_share": round(stock_outs / max(1, n_turns), 4),
            "cause_turns": dict(cause.most_common()),
            "cause_share_of_stockouts": {k: round(v / max(1, stock_outs), 4)
                                         for k, v in cause.most_common()
                                         if k != "cap_satisfied"},
            "turn_weighted_units_short": dict(units.most_common()),
            "sequencing_recoverable_turns": recoverable,
            "median_money_on_cash_short": (ms[len(ms) // 2] if ms else None),
            "cause_by_day": {c: {str(d): n for d, n in sorted(h.items())}
                             for c, h in by_day.items() if c != "cap_satisfied"},
        }
    return out


def _job(args):
    return run_episode(*args)


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--both-seats", action="store_true")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    seats = (0, 1) if a.both_seats else (0,)
    jobs = [(s, seat, a.steps) for s in parse_seeds(a.seeds) for seat in seats]
    eps = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for ep in ex.map(_job, jobs):
            eps.append(ep)
            print(f"  seed {ep['seed']} seat {ep['seat']}: money {ep['money']} "
                  f"plan_exceptions {ep['plan_exceptions']}", flush=True)

    rep = {"experiment": "expB-304b-seeds", "ruling": "R-51",
           "engine": eps[0]["engine"], "seat_seasons": len(eps),
           "champion": CHAMPION, "opponent": PRICER,
           "summary": summarise(eps)}
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        (p.parent / "turns.json").write_text(json.dumps(eps), encoding="utf-8")
    print(json.dumps(rep["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
