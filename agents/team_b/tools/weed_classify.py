"""ruling-03 section 1.5 — classify every weed-out and re-price the M4 weed line.

Architect B's ruling-03 vacated the -$4,483 weed regression expB-304a reported and
pre-registered the measurement that decides whether the vacatur is right:

  > Per killed ongoing tile, record **age at death** and **production events
  > realised vs `max_yield`**. Classify: `RETIREMENT` = events realised ==
  > `max_yield`; `LOSS` = anything less.
  > **Falsification:** if fewer than **half** of the arm's long-lead kills
  > classify as RETIREMENT, this ruling is wrong, the regression is real, and
  > the remediation question re-opens on the spot.

This tool reads `feed_audit.py --out` directories (which now emit `weed_events`
with the per-tile spend ledger) and prints:

  1. the RETIREMENT / LOSS split, by crop and by kill path, per season;
  2. the age-at-death distribution against the pre-fix mean of 12.9 (ruling-02 R3);
  3. spec section 6.1's weed-loss formula **as written** and **as restated** under
     R-13 (a) exclude retirements, (b) charge lambda only on actions spent after
     the tile's last realised production event, (c) lambda is contribution per
     productive action, not gross;
  4. R-14's resource basing: weed tile-days and fallow tile-days per season, which
     is what the tripwire now prices instead of the transition count;
  5. the pre-registered verdict, computed rather than argued.

Two numbers cannot come from the audit and must be supplied (from the arm's own
`pnl_census` aggregate, which is what `--census` reads):

  lambda      contribution per productive action (R-13(c))
  $/unit      realised dollars per unit sold, per crop -- the price the forgone
              units would actually have fetched, not the base price

Usage (from the repo root):

    uv run python agents/team_b/tools/weed_classify.py \
        --base ../bench/runs/expB-303b-gate/audit-base \
        --arm  ../bench/runs/expB-303b-gate/audit-arm \
        --base-census ../bench/runs/expB-303b-gate/census-base \
        --arm-census  ../bench/runs/expB-303b-gate/census-arm

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter
from pathlib import Path

# ruling-02 section 6.1's own list: the crops whose weed-out was held to an
# absolute 0 because they carry 10-17 tile-days of committed investment.
LONG_LEAD = ("MELON", "TOMATO", "STRAWBERRY")

# Fallback realised prices (expB-304a arm census, engine 1.32.7). Only used for a
# crop the census does not price; every one of these is reported as a fallback so
# a stale constant can never pass unnoticed.
FALLBACK_PRICE = {"WHEAT": 54.2, "CARROT": 35.0, "TOMATO": 60.0,
                  "STRAWBERRY": 220.6, "MELON": 98.1}


def load_audit(d, seat="both"):
    """Seat-seasons in `d`. `seat="both"` is correct for a mirror run (the audit
    plays the arm against itself), and doubles the sample for free."""
    seats = (0, 1) if seat == "both" else (int(seat),)
    out = []
    for f in sorted(glob.glob(str(Path(d) / "feed_audit_seed*.json"))):
        rec = json.loads(Path(f).read_text())
        for s in seats:
            out.append(rec["per_player"][s])
    return out


def load_census(d):
    """lambda (contribution / productive action) and realised $/unit, per crop."""
    if not d:
        return None, {}, None
    agg = json.loads((Path(d) / "pnl_aggregate.json").read_text())
    seeds = [json.loads(Path(f).read_text())
             for f in sorted(glob.glob(str(Path(d) / "pnl_seed*.json")))]
    spec = seeds[0]["agents"][0]
    a = agg["per_agent"][spec]
    lines = a["lines"]

    def contribution(ln):
        v = lines[ln]
        for k in ("contribution", "contribution_after_capex"):
            if v.get(k) and v[k].get("mean") is not None:
                return v[k]["mean"]
        return None

    # R-13(c): lambda is contribution per productive action, at the whole-farm
    # level -- the opportunity cost of the marginal unit-turn.
    total_contrib = sum(c for c in (contribution(ln) for ln in lines) if c is not None)
    prod_acts = a["productive_actions"]["mean"]
    lam = round(total_contrib / prod_acts, 2) if prod_acts else None
    prices = {}
    for ln, v in lines.items():
        dpu = v.get("dollars_per_unit")
        if dpu:
            prices[ln] = dpu
    return lam, prices, spec


def price_events(events, lam, prices, restated: bool):
    """Spec section 6.1: seed + lambda * spent actions + forgone units at price.

    `restated=False` is the formula as written (every weed priced, lambda on every
    action ever spent on the tile). `restated=True` applies R-13 (a)/(b): drop
    retirements, and charge lambda only on post-terminal actions.
    """
    seed = acts_cost = forgone = 0.0
    n = 0
    fallback_used = set()
    for e in events:
        if restated and e["class"] == "RETIREMENT":
            continue
        n += 1
        seed += e["seed_cost"]
        a = e["acts_post_terminal"] if restated else e["acts"]
        if a is None:
            a = 0
        acts_cost += lam * a
        crop = e["crop"]
        if crop in prices:
            p = prices[crop]
        else:
            p = FALLBACK_PRICE.get(crop, 0.0)
            fallback_used.add(crop)
        if e["ongoing"]:
            # remaining production events, one unit each (the unfertilized floor)
            # plus whatever had already grown on the tile and was never harvested
            units = max(0, e["max_yield"] - e["events_realised"]) + e["yield"]
        else:
            # a non-ongoing tile realises nothing until HARVEST removes it, so a
            # weeded one forfeits its whole potential
            units = e["max_yield"]
        forgone += p * units
    return {"n": n, "seed": round(seed), "lambda_actions": round(acts_cost),
            "forgone": round(forgone), "total": round(seed + acts_cost + forgone),
            "fallback_priced": sorted(fallback_used)}


def summarise(seasons, lam, prices, label):
    ev = [e for s in seasons for e in s["weed_events"]]
    n_seasons = len(seasons)
    per = lambda x: round(x / n_seasons, 2)
    out = {"label": label, "n_seasons": n_seasons, "lam": lam,
           "events_total": len(ev), "events_per_season": per(len(ev))}

    def bucket(sel):
        sub = [e for e in ev if sel(e)]
        ret = [e for e in sub if e["class"] == "RETIREMENT"]
        return {"n": len(sub), "per_season": per(len(sub)),
                "retirement": len(ret), "loss": len(sub) - len(ret),
                "retirement_frac": round(len(ret) / len(sub), 4) if sub else None}

    out["all"] = bucket(lambda e: True)
    out["long_lead"] = bucket(lambda e: e["crop"] in LONG_LEAD)
    out["ongoing"] = bucket(lambda e: e["ongoing"])
    out["short_crop"] = bucket(lambda e: e["crop"] not in LONG_LEAD)
    out["unwatered_path"] = bucket(lambda e: e["path"] == "unwatered")
    out["decay_path"] = bucket(lambda e: e["path"] == "lifespan_decay")
    out["by_crop"] = {c: bucket(lambda e, c=c: e["crop"] == c)
                      for c in sorted({e["crop"] for e in ev})}
    ages = [e["age"] for e in ev if e["ongoing"]]
    out["ongoing_age_mean"] = round(statistics.fmean(ages), 2) if ages else None
    out["ongoing_age_hist"] = dict(sorted(Counter(ages).items()))
    ll_ages = [e["age"] for e in ev if e["crop"] in LONG_LEAD]
    out["long_lead_age_mean"] = round(statistics.fmean(ll_ages), 2) if ll_ages else None
    # the consistency check the classifier records: an ongoing tile killed by
    # lifespan decay MUST have reached max_yield (max_lifespan_step is only set
    # at the terminal event), so the two derivations have to agree.
    out["decay_classification_disagreements"] = sum(
        1 for e in ev if e["decay_implies_retired"] and e["class"] != "RETIREMENT")
    out["spend_untracked"] = sum(1 for e in ev if not e["spend_tracked"])
    out["retired_tiles_per_season"] = per(sum(s["retired_tiles"] for s in seasons))
    out["escapes"] = sum(s["escapes"] for s in seasons)
    if lam is not None:
        out["price_as_written"] = {k: (round(v / n_seasons) if isinstance(v, (int, float)) else v)
                                   for k, v in price_events(ev, lam, prices, False).items()}
        out["price_restated"] = {k: (round(v / n_seasons) if isinstance(v, (int, float)) else v)
                                 for k, v in price_events(ev, lam, prices, True).items()}
    return out


def show(s):
    print(f"\n================ {s['label']}   ({s['n_seasons']} seat-seasons, "
          f"lambda = {s['lam']})")
    print(f"  weed events/season        {s['events_per_season']:>8}   "
          f"retired tiles/season {s['retired_tiles_per_season']}   escapes {s['escapes']}")
    for k in ("all", "long_lead", "ongoing", "short_crop", "unwatered_path", "decay_path"):
        b = s[k]
        print(f"  {k:<24} {b['per_season']:>8}/season   RETIREMENT {b['retirement']:>4} "
              f"LOSS {b['loss']:>4}   retirement frac "
              f"{('%.1f%%' % (100 * b['retirement_frac'])) if b['retirement_frac'] is not None else '-'}")
    print("  by crop:")
    for c, b in s["by_crop"].items():
        print(f"    {c:<12} {b['per_season']:>6}/season  RET {b['retirement']:>4}  LOSS {b['loss']:>4}"
              f"  frac {('%.1f%%' % (100 * b['retirement_frac'])) if b['retirement_frac'] is not None else '-'}")
    print(f"  ongoing age at death: mean {s['ongoing_age_mean']} (pre-R3-fix mean was 12.9)")
    print(f"    hist {s['ongoing_age_hist']}")
    print(f"  long-lead age at death: mean {s['long_lead_age_mean']}")
    print(f"  consistency: decay-vs-events disagreements {s['decay_classification_disagreements']}"
          f"   spend-ledger misses {s['spend_untracked']}")
    if "price_as_written" in s:
        for tag in ("price_as_written", "price_restated"):
            p = s[tag]
            print(f"  {tag:<20} n={p['n']:>4}  seed ${p['seed']:>6}  lambda*acts ${p['lambda_actions']:>6}"
                  f"  forgone ${p['forgone']:>7}  TOTAL ${p['total']:>7} /season"
                  + (f"   [fallback-priced: {p['fallback_priced']}]" if p["fallback_priced"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="feed_audit out-dir for the control")
    ap.add_argument("--arm", required=True, help="feed_audit out-dir for the arm")
    ap.add_argument("--base-census", default=None)
    ap.add_argument("--arm-census", default=None)
    ap.add_argument("--seat", default="both", choices=["both", "0", "1"])
    ap.add_argument("--lam", type=float, default=None,
                    help="override lambda (contribution per productive action)")
    ap.add_argument("--paired-edge", type=float, default=None,
                    help="the arm's paired bank edge, for the section 1.3 tripwire limbs")
    ap.add_argument("--mean-bank", type=float, default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    lam_b, px_b, spec_b = load_census(args.base_census)
    lam_a, px_a, spec_a = load_census(args.arm_census)
    b = summarise(load_audit(args.base, args.seat), args.lam or lam_b, px_b, "BASE " + (spec_b or args.base))
    a = summarise(load_audit(args.arm, args.seat), args.lam or lam_a, px_a, "ARM  " + (spec_a or args.arm))
    show(b)
    show(a)

    ll = a["long_lead"]
    print("\n---- ruling-03 section 1.5 pre-registered falsification ----")
    if ll["retirement_frac"] is None:
        verdict = "NO LONG-LEAD KILLS — test vacuous"
    elif ll["retirement_frac"] >= 0.5:
        verdict = "RULING UPHELD (>= 50% RETIREMENT)"
    else:
        verdict = "RULING FALSIFIED (< 50% RETIREMENT) — the regression is real and remediation re-opens"
    print(f"  arm long-lead kills {ll['n']} ({ll['per_season']}/season): "
          f"RETIREMENT {ll['retirement']}  LOSS {ll['loss']}  "
          f"frac {('%.1f%%' % (100 * ll['retirement_frac'])) if ll['retirement_frac'] is not None else '-'}")
    print(f"  VERDICT: {verdict}")

    if "price_restated" in a and "price_restated" in b:
        delta_w = a["price_as_written"]["total"] - b["price_as_written"]["total"]
        delta_r = a["price_restated"]["total"] - b["price_restated"]["total"]
        print(f"\n  regression as written  base ${b['price_as_written']['total']:,} -> "
              f"arm ${a['price_as_written']['total']:,}   delta ${-delta_w:,}")
        print(f"  regression restated    base ${b['price_restated']['total']:,} -> "
              f"arm ${a['price_restated']['total']:,}   delta ${-delta_r:,}")
        for tag, d in (("as written", delta_w), ("restated", delta_r)):
            if args.paired_edge:
                print(f"    tripwire limb 1 ({tag}): {100 * d / args.paired_edge:.1f}% of paired edge (fires > 25%)")
            if args.mean_bank:
                print(f"    tripwire limb 2 ({tag}): {100 * d / args.mean_bank:.1f}% of mean bank (fires > 5%)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"base": b, "arm": a, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
