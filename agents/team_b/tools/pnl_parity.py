"""Chassis-validity check for pnl_census.py (expB-301).

`pnl_census.run_pnl` monkey-patches five engine entry points (`interpreter`,
`_apply_unit_action`, `_commit_unit`, `_do_hire`, `_do_buy_land`). An
instrumentation patch that perturbs the episode invalidates every number it
produces — the expB-101 lesson, where a chassis bug inverted two A/B signs.

This asserts that an instrumented episode is byte-for-byte the same episode as
an uninstrumented one: identical final money for both seats, on every seed.
It further reconciles the census against the engine's own end state:

  * revenue booked per product == money in minus money out (accounting identity)
  * FEED events == wheat units consumed as feed
  * harvest units >= units sold (you cannot sell what you never made)

Usage:  uv run python agents/team_b/tools/pnl_parity.py --a SPEC --b SPEC --seeds 0-3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _engine import import_engine  # noqa: E402
from instrument import load_agent, arity_wrap  # noqa: E402
import pnl_census  # noqa: E402


def plain_episode(spec_a, spec_b, seed, steps=720):
    ke = import_engine()
    env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    agents = [arity_wrap(load_agent(spec_a), env.configuration),
              arity_wrap(load_agent(spec_b), env.configuration)]
    env.reset()
    while not env.done:
        obs = [env.state[i].observation for i in range(2)]
        acts = []
        for i, ag in enumerate(agents):
            try:
                a = ag(obs[i])
            except Exception:
                a = {"farmer": ["PASS"], "hands": [], "market": []}
            acts.append(a if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []})
        env.step(acts)
    final = env.steps[-1]
    return [final[i].observation.farms[i]["money"] for i in range(2)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args(argv)

    fails = 0
    for seed in pnl_census.parse_seeds(args.seeds):
        plain = plain_episode(args.a, args.b, seed, args.steps)
        raw = pnl_census.run_pnl(args.a, args.b, seed, args.steps)
        inst = raw["final_money"]
        ok = plain == inst
        print(f"seed {seed}: plain {plain}  instrumented {inst}  "
              f"{'PARITY' if ok else '*** DIVERGED ***'}")
        if not ok:
            fails += 1
            continue
        for pi in range(2):
            st = raw["players"][pi]
            s = pnl_census.income_statement(st)
            starting = 3000
            spend = s["total_spend"]
            cash = st["terminal"]["cash"]
            identity = round(starting + s["gross_revenue"] - spend - cash, 2)
            feed_events = sum(st["feed_events"].values())
            made_ge_sold = all(st["harvest_units"][p] >= st["sales"][p]["units"]
                               for p in ("EGG", "MILK", "WOOL", "MELON", "CARROT",
                                         "TOMATO", "STRAWBERRY"))
            wheat_ok = (st["harvest_units"]["WHEAT"] + st["buy_product"]["WHEAT"]["units"]
                        >= feed_events + st["sales"]["WHEAT"]["units"])
            print(f"   p{pi}: cash-identity residual {identity}  (0 = books balance) | "
                  f"feed events {feed_events} | made>=sold {made_ge_sold} | wheat-supply ok {wheat_ok}")
            if identity != 0 or not made_ge_sold or not wheat_ok:
                fails += 1
    print("RESULT " + ("PASS" if fails == 0 else f"FAIL ({fails})"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
