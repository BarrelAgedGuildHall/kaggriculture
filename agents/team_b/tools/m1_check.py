"""M1 acceptance check + G4 function-level assay (millrace/market.py).

Steps a mirror episode and, alongside it, feeds `millrace.market.MarketModel`
with player 0's observations and emitted orders — exactly the inputs the live
agent will give it. At each day boundary it scores:

1. **Drain acceptance** (spec M1: within ±15% of truth by day 6): the model's
   LIVE estimate (mean of no-trade inventory decrements, >=8 samples; the
   shop-recipe prior fallback is bypassed so the test cannot be vacuous)
   against the exact engine drain — 6 shop ticks/day (x2 for single-product
   shops) + 1 town-centre unit/day per product, from the currently unlocked
   shops.

2. **marginal_revenue acceptance** (±5% of realised proceeds of a 20-unit
   sale): model.marginal_revenue(p, 20) against the engine's own sell walk
   (market_price at incrementing inventory, +1 inventory only while price > 1)
   at the same instant, using the LIVE market params (perturbed-aware).
   For comparison the same error is computed for a HARDCODED REPLICA of the
   1.32.7 MARKET_PARAMS table — under a perturbed engine the replica's error
   is the G4 function-level control.

Usage (from the repo root):
    uv run python agents/team_b/tools/m1_check.py --seeds 0,1,2
    uv run python agents/team_b/tools/m1_check.py --seeds 0,1,2 --perturb agents/team_b/tools/g4_perturb.json
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
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))

from _engine import import_engine, engine_version  # noqa: E402
from instrument import load_agent  # noqa: E402
from millrace.market import MarketModel, PRODUCTS  # noqa: E402

CHECK_DAYS = (6, 10, 14, 18, 22)
MR_HOUR = 12
MR_N = 20


def true_drain_per_day(K, town, product):
    """Exact engine drain (kaggriculture 1.32.7 _town_consume, default config:
    shop tick every 4 steps => 6/day, centre every 24 steps => 1/day)."""
    if product == "FERTILIZER":
        return 0.0
    per_day = 1.0
    for shop in town.get("unlocked_shops", []):
        prods = K.SHOPS[shop]
        if product in prods:
            per_day += (2 if len(prods) == 1 else 1) * 6
    return per_day


def sell_walk(K, params, item, inventory, n):
    """Realised proceeds of selling n units now (engine _commit_unit walk)."""
    inv = inventory
    total = 0.0
    for _ in range(n):
        p = K.market_price(item, inv, params)
        total += p
        if p > 1:
            inv += 1
    return total


def run_episode(agent_spec, seed, steps, perturb):
    ke = import_engine()
    import kaggle_environments.envs.kaggriculture.kaggriculture as K

    cfg = {"episodeSteps": steps, "seed": seed}
    if perturb:
        cfg["marketParams"] = perturb
    env = ke.make("kaggriculture", configuration=cfg, debug=False)
    agents = [load_agent(agent_spec), load_agent(agent_spec)]
    env.reset()

    model = MarketModel()
    last_orders = []
    drain_rows = []   # (day, product, live_est, n_samples, truth)
    mr_rows = []      # (day, product, model_mr, replica_mr, truth, inv, quote)
    replica_params = {p: dict(v) for p, v in K.MARKET_PARAMS.items()}  # frozen copy

    while not env.done:
        obs0 = env.state[0].observation
        step = obs0.step
        day, hour = step // 24, step % 24
        model.observe(obs0, last_orders)

        if hour == 0 and day in CHECK_DAYS:
            for p in PRODUCTS:
                live, n_days = model.drain_live(p)
                truth = true_drain_per_day(K, obs0.town, p)
                drain_rows.append({"day": day, "product": p,
                                   "live_est": None if live is None else round(live, 2),
                                   "n_samples": n_days, "tick_ok": model.tick_ok,
                                   "truth": truth})
        if hour == MR_HOUR and day in CHECK_DAYS:
            live_params = obs0.market.get("params") or K.MARKET_PARAMS
            for p in PRODUCTS:
                inv = obs0.market["inventory"][p]
                truth = sell_walk(K, live_params, p, inv, MR_N)
                model_mr = model.marginal_revenue(p, MR_N)
                replica_mr = sell_walk(K, replica_params, p, inv, MR_N)
                mr_rows.append({"day": day, "product": p, "inv": inv,
                                "quote": obs0.market["prices"][p],
                                "model_mr": round(model_mr, 1),
                                "replica_mr": round(replica_mr, 1),
                                "truth": round(truth, 1)})

        actions = []
        for i, ag in enumerate(agents):
            o = env.state[i].observation
            try:
                a = ag(o)
            except Exception:
                a = {"farmer": ["PASS"], "hands": [], "market": []}
            actions.append(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})
        last_orders = actions[0].get("market", [])
        env.step(actions)

    return {"engine": engine_version(), "seed": seed, "perturbed": bool(perturb),
            "agent": agent_spec, "drain": drain_rows, "marginal_revenue": mr_rows}


def relerr(est, truth):
    if truth == 0:
        return None
    return (est - truth) / truth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="millrace:routed_mixed")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--perturb", default=None, help="JSON file with marketParams overrides")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    perturb = json.loads(Path(args.perturb).read_text(encoding="utf-8")) if args.perturb else None
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = [run_episode(args.agent, s, args.steps, perturb) for s in seeds]

    # ---- aggregate: drain at day 6 (the acceptance day) + later days --------
    print(f"engine={results[0]['engine']} agent={args.agent} seeds={seeds} "
          f"perturbed={bool(perturb)}")
    for check_day in CHECK_DAYS:
        errs, misses, unmeasured = [], 0, 0
        for r in results:
            for row in r["drain"]:
                if row["day"] != check_day or row["product"] == "FERTILIZER":
                    continue
                if row["live_est"] is None:
                    unmeasured += 1
                    continue
                e = relerr(row["live_est"], row["truth"])
                errs.append(abs(e))
                if abs(e) > 0.15:
                    misses += 1
        if errs:
            print(f"drain day {check_day:>2}: n={len(errs)} unmeasured={unmeasured} "
                  f"median|err|={sorted(errs)[len(errs)//2]:.3f} max|err|={max(errs):.3f} "
                  f">15%: {misses}")
        else:
            print(f"drain day {check_day:>2}: no live estimates yet (unmeasured={unmeasured})")

    for check_day in CHECK_DAYS:
        m_errs, r_errs, m_miss = [], [], 0
        for r in results:
            for row in r["marginal_revenue"]:
                if row["day"] != check_day or row["truth"] < MR_N:  # floored/dead product
                    continue
                em = relerr(row["model_mr"], row["truth"])
                er = relerr(row["replica_mr"], row["truth"])
                m_errs.append(abs(em)); r_errs.append(abs(er))
                if abs(em) > 0.05:
                    m_miss += 1
        if m_errs:
            print(f"mr20 day {check_day:>2}: n={len(m_errs)} "
                  f"model median|err|={sorted(m_errs)[len(m_errs)//2]:.3f} max={max(m_errs):.3f} >5%: {m_miss} | "
                  f"replica median|err|={sorted(r_errs)[len(r_errs)//2]:.3f} max={max(r_errs):.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"raw -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
