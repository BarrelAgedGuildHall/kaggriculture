"""The engine-bump tripwire for `nav`, as one command.

    uv run python scripts/nav_tripwire.py
    uv run python scripts/nav_tripwire.py --seeds 3,7,11 --agents starter,starter --json out.json

What it answers: **do `nav`'s runtime-fitted price curves still describe the market the engine
actually runs?** `nav` is never given `marketParams`; it pins `base` and `I0` from step 0 and
fits everything else from observed `(inventory, price)` pairs. If a `kaggle-environments`
release re-shapes a curve, nav should re-derive the new one silently -- and if a release
introduces a curve nav *cannot* represent, that must be loud, immediate and attributable
rather than a balance sheet that is quietly 20% wrong.

Three checks, each of which fails the command (exit 1):

1. **Rulebook.** `nav.rules` versus the installed engine's tables. A changed crop or shop is
   an engine bump by definition.
2. **Fit accuracy.** Over each product's observed inventory range -- the region where nav
   has evidence -- `|nav.quote - engine price|`: median <= 2 units, p95 <= 5. Run both on the
   shipped params and on the g4 perturbation (amplitudes +-30%, MELON's glut shape swapped)
   with nav given no params, which is what makes the check curve-agnostic instead of a
   re-statement of the constants. Those limits are the measured envelope on the pinned engine
   (worst median 2, worst p95 5 across seeds 3/7/11, both param sets) with no margin to
   spare, so they are a coarse smoke check; the sharp instruments are (1) and (3).

   Outside the observed range nav does not extrapolate a curve it has no evidence for: the
   side falls back to `base` and every mark that used it is flagged `provisional`. The
   evidence line in the output says which sides of which pools were ever observed, because
   a future sale marked into an unobserved glut side is marked AT BASE, i.e. optimistically.
3. **Runtime tripwire.** `CurveSet.tripwire()` must stay empty throughout (no crying wolf),
   and the built-in negative control -- a market whose curve is outside the five shape
   families -- must make it fire (a tripwire that cannot be seen firing is decoration).

Run it after every engine bump, before re-running any experiment on the new engine.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _engine import ENV_NAME, engine_version, import_engine  # noqa: E402

import nav  # noqa: E402

PERTURB = Path(__file__).resolve().parents[1] / "agents" / "team_b" / "tools" / "g4_perturb.json"
MEDIAN_LIMIT = 2.0
P95_LIMIT = 5.0


def perturb_params() -> dict:
    return {k: v for k, v in json.loads(PERTURB.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}


def play(agents, seed, params=None, steps=720):
    ke = import_engine()
    cfg = {"episodeSteps": steps, "seed": seed}
    if params:
        cfg["marketParams"] = params
    env = ke.make(ENV_NAME, configuration=cfg, debug=False)
    env.run(list(agents))
    return env


def check_rules(K) -> list[str]:
    problems = []
    if dict(nav.rules.CROPS) != K.CROPS:
        problems.append("CROPS differ from the engine")
    if dict(nav.rules.ANIMALS) != K.ANIMALS:
        problems.append("ANIMALS differ from the engine")
    if {k: list(v) for k, v in nav.rules.SHOPS.items()} != K.SHOPS:
        problems.append("SHOPS differ from the engine")
    if list(nav.rules.PRODUCTS) != K.PRODUCTS:
        problems.append("PRODUCTS differ from the engine")
    if nav.rules.PRICE_FLOOR != K.PRICE_FLOOR or nav.rules.HINGE_GAIN != K.HINGE_GAIN:
        problems.append("market mechanics constants differ from the engine")
    if nav.rules.ENGINE_VERSION != engine_version():
        problems.append(f"nav.rules was transcribed from {nav.rules.ENGINE_VERSION}, "
                        f"engine is {engine_version()} -- re-verify and update the record")
    return problems


def fit_episode(env, seat=0):
    navobj = nav.Nav()
    for i in range(len(env.steps)):
        navobj.observe(env.steps[i][seat].observation)
    return navobj


def accuracy(navobj, K, params=None) -> dict:
    resolved = K._resolve_market_params(params) if params else None
    rows = {}
    for product, curve in navobj.curves.curves.items():
        seen = curve._observed
        if len(seen) < 5:
            continue
        lo, hi = min(seen), max(seen)
        errs = [abs(curve.quote(inv) - K.market_price(product, inv, resolved))
                for inv in range(lo, hi + 1) if inv not in seen]
        if not errs:
            continue
        errs.sort()
        rows[product] = {
            "n_points": curve.n_points, "range": [lo, hi],
            "median": statistics.median(errs), "p95": errs[int(0.95 * (len(errs) - 1))],
            "max": errs[-1], "shape": list(curve.fitted_shape),
        }
    return rows


def negative_control() -> tuple[bool, int | None]:
    """A curve outside the five families. The tripwire must fire on it."""
    base, cs, fired = 25, nav.CurveSet(), None
    for k in range(200):
        inv = 10000 - 3 * k
        d = 10000 - inv
        price = max(1, int(round(base + 14 * math.sin(d / 37.0) + 0.02 * d)))
        market = {"inventory": {p: (inv if p == "WHEAT" else 10000) for p in nav.rules.PRODUCTS},
                  "prices": {p: (price if p == "WHEAT" else 25) for p in nav.rules.PRODUCTS}}
        cs.observe(market, step=k)
        if fired is None and "WHEAT" in cs.tripwire():
            fired = k
    return fired is not None, fired


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="3,7,11")
    ap.add_argument("--agents", default="starter,starter")
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args(argv)

    K = import_engine().envs.kaggriculture.kaggriculture
    agents = args.agents.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    report = {"engine": engine_version(), "nav_schema": nav.SCHEMA_VERSION,
              "nav_source": nav.source_hash(), "agents": agents, "seeds": seeds}
    failures = []

    print(f"engine   : kaggle-environments {engine_version()}")
    print(f"nav      : schema {nav.SCHEMA_VERSION}, source {nav.source_hash()}")

    rule_problems = check_rules(K)
    report["rules"] = rule_problems or "ok"
    print(f"rulebook : {'OK' if not rule_problems else 'CHANGED -- ' + '; '.join(rule_problems)}")
    failures += rule_problems

    params_sets = [("shipped", None), ("g4-perturbed", perturb_params())]
    report["accuracy"] = {}
    for label, params in params_sets:
        worst = {"median": 0.0, "p95": 0.0}
        fired = {}
        for seed in seeds:
            env = play(agents, seed, params)
            navobj = fit_episode(env)
            rows = accuracy(navobj, K, params)
            report["accuracy"].setdefault(label, {})[seed] = rows
            for product, r in rows.items():
                worst["median"] = max(worst["median"], r["median"])
                worst["p95"] = max(worst["p95"], r["p95"])
                if r["median"] > MEDIAN_LIMIT or r["p95"] > P95_LIMIT:
                    failures.append(f"{label} seed {seed}: {product} median {r['median']} p95 {r['p95']}")
            t = navobj.curves.tripwire()
            if t:
                fired[seed] = t
                failures.append(f"{label} seed {seed}: tripwire fired on {t} -- nav could not "
                                f"re-derive a curve it should have")
        print(f"curves   : {label:13s} worst median {worst['median']:.1f}, worst p95 {worst['p95']:.1f} "
              f"(limits {MEDIAN_LIMIT}/{P95_LIMIT}) tripwire {fired or 'quiet'}")
        evidence = {p: [s for s, sh in zip(("below", "above"), navobj.curves[p].fitted_shape) if sh]
                    for p in nav.rules.PRODUCTS}
        blind = [p for p, sides in evidence.items() if "above" not in sides]
        print(f"           evidence: no glut-side points for {', '.join(blind) or 'none'} "
              f"-- a future sale there marks at base and is flagged provisional")
        report.setdefault("evidence", {})[label] = evidence

    ok, at = negative_control()
    report["negative_control"] = {"fires": ok, "after_points": at}
    print(f"control  : unrepresentable curve trips the wire after {at} points"
          if ok else "control  : FAILED -- the tripwire never fires, it is not measuring anything")
    if not ok:
        failures.append("negative control: the tripwire did not fire on an unfittable curve")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"report   : {args.json}")

    if failures:
        print("\nTRIPWIRE: FAIL")
        for f in failures:
            print(f"  ! {f}")
        print("\nThe engine's price curves no longer match what nav fits. Do NOT trust any "
              "balance-sheet number until this is resolved: re-check docs/engine/VERSION.md, "
              "then either extend nav.curves' shape families or re-transcribe nav.rules.")
        return 1
    print("\nTRIPWIRE: PASS -- nav re-derives the engine's curves from observation alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
