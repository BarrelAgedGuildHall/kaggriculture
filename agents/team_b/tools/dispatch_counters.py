"""expB-304a — read the pre-registered dispatch counters out of a P&L census.

ruling-02 section 2.6 pre-registers six falsifiable counters for the dispatch
pricer. This tool reads them off a `pnl_census.py` output directory so the
branch outcome is computed, not eyeballed:

  no-op unit-turns/season        559 -> < 150      (census per-ACTION diff)
  standing STRAWBERRY tiles      3.2 -> >= 8       (tile-days / season_days)
  realised STRAWBERRY units       18 -> >= 45      (units produced)
  fallow (WEED) tile-days       30.1 -> <= 7
  GOOSE CARE actions                 -> fall >= 50%
  dispatch band constants            -> gone       (checked in the source)

Usage (from the repo root):

    uv run python agents/team_b/tools/dispatch_counters.py \
        --base ../runs/expB-304a/census-base --arm ../runs/expB-304a/census-arm

Both directories must have been produced with the SAME opponent, or the
counters are not comparable. Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ANIMAL_LINES = ("GOOSE", "COW", "SHEEP")


def read(d, season_days=30):
    d = Path(d)
    agg = json.loads((d / "pnl_aggregate.json").read_text())
    seeds = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(d / "pnl_seed*.json")))]
    spec = seeds[0]["agents"][0]        # the arm always sits in seat 0
    a = agg["per_agent"][spec]

    def m(node, key="mean"):
        return (node or {}).get(key)

    lines = a["lines"]
    out = {
        "spec": spec,
        "n_seasons": sum(1 for s in seeds for sp in s["agents"] if sp == spec),
        "noop_unit_turns": round(sum(m(a["activity_noop"][b]) for b in a["activity_noop"]), 1),
        "noop_by_op": {k: round(m(v), 1) for k, v in a["acts_by_op"].items() if k.endswith(":noop")},
        "pass": round(m(a["acts_by_op"].get("PASS:ok", {})) or 0, 1),
        "unit_turns": round(m(a["unit_turns"]), 1),
        "productive_actions": round(m(a["productive_actions"]), 1),
        "straw_tile_days": round(m(lines["STRAWBERRY"]["tile_days"]), 1),
        "straw_standing": round(m(lines["STRAWBERRY"]["tile_days"]) / season_days, 2),
        "straw_units": round(m(lines["STRAWBERRY"]["units_produced"]), 1),
        "weed_tile_days": round(m(a["tile_days_other"]["WEED"]), 1),
        "fallow_tile_days": round(m(a["tile_days_other"]["FALLOW"]), 1),
        "bank": round(m(a["bank"])),
        "gross": round(m(a["gross_revenue"])),
        "wheat_units": round(m(a["purchased_wheat_units"]), 1),
        "wheat_price": round(m(a["purchased_wheat_avg_price"]), 1),
        "rev_per_productive_action": round(m(a["rev_per_productive_action"]), 1),
        "acts": {k: round(m(v), 1) for k, v in a["acts_by_op"].items() if k.endswith(":ok")},
        # ruling-02 addendum-02 A-7: realised $/unit per commodity. The whole
        # point of pricing off `obs.market` rather than a constant is that the
        # realised price is whatever the live pool pays; report it.
        "lines": {ln: {"tile_days": round(m(lines[ln]["tile_days"]), 1),
                       "acts": round(m(lines[ln]["productive_actions"]), 1),
                       "contrib_per_action": m(lines[ln].get("contrib_per_action")),
                       "units": round(m(lines[ln]["units_produced"]), 1),
                       "units_sold": round(m(lines[ln]["units_sold"]), 1),
                       "revenue": round(m(lines[ln]["revenue"])),
                       "realised_per_unit": (round(m(lines[ln]["revenue"])
                                                   / m(lines[ln]["units_sold"]), 1)
                                             if m(lines[ln]["units_sold"]) else None)}
                  for ln in lines},
    }
    # CARE per animal line: an animal line's `maintain` bucket is FEED + CARE
    # only (WATER/FERTILIZE cannot land on an animal tile), and feed events are
    # counted exactly, so CARE = maintain - feed_events.
    care, feed = {}, {}
    for ln in ANIMAL_LINES:
        tm = tf = n = 0
        for s in seeds:
            for si, sp in enumerate(s["agents"]):
                if sp != spec:
                    continue
                st = s["statements"][si]
                tm += st["lines"][ln]["acts"]["maintain"]
                tf += st["lines"][ln]["feed_events"]
                n += 1
        care[ln] = round((tm - tf) / max(1, n), 1)
        feed[ln] = round(tf / max(1, n), 1)
    out["care_actions"] = care
    out["feed_actions"] = feed
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--season-days", type=int, default=30)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    b = read(args.base, args.season_days)
    a = read(args.arm, args.season_days)

    def row(label, key, target, better="down", fmt="%.1f"):
        vb, va = b[key], a[key]
        ok = (va <= target) if better == "down" else (va >= target)
        print("  %-30s %10s -> %10s   target %-10s %s"
              % (label, fmt % vb, fmt % va, ("<= " if better == "down" else ">= ") + str(target),
                 "PASS" if ok else "MISS"))
        return ok

    print("expB-304a pre-registered counters (base -> arm), %d / %d seat-seasons"
          % (b["n_seasons"], a["n_seasons"]))
    res = {}
    res["noop"] = row("no-op unit-turns/season", "noop_unit_turns", 150)
    res["straw_standing"] = row("standing STRAWBERRY tiles", "straw_standing", 8, "up", "%.2f")
    res["straw_units"] = row("realised STRAWBERRY units", "straw_units", 45, "up")
    res["weed"] = row("fallow (WEED) tile-days", "weed_tile_days", 7)
    gb, ga = b["care_actions"]["GOOSE"], a["care_actions"]["GOOSE"]
    ok = ga <= 0.5 * gb
    res["goose_care"] = ok
    print("  %-30s %10.1f -> %10.1f   target %-10s %s"
          % ("GOOSE CARE actions", gb, ga, "<= -50%", "PASS" if ok else "MISS"))
    print()
    for k in ("bank", "gross", "wheat_units", "wheat_price", "pass", "productive_actions",
              "rev_per_productive_action", "fallow_tile_days"):
        print("  %-30s %10s -> %10s" % (k, b[k], a[k]))
    print("  %-30s %10s -> %10s" % ("CARE by line", b["care_actions"], a["care_actions"]))
    print("  %-30s %10s -> %10s" % ("FEED by line", b["feed_actions"], a["feed_actions"]))
    print("  base acts:", b["acts"])
    print("  arm  acts:", a["acts"])
    print("  base noop:", b["noop_by_op"])
    print("  arm  noop:", a["noop_by_op"])
    for ln in ("GOOSE", "COW", "SHEEP", "WHEAT", "MELON", "STRAWBERRY", "FERTILIZER"):
        print("  line %-11s base td=%7.1f acts=%7.1f c/act=%-7s | arm td=%7.1f acts=%7.1f c/act=%s"
              % (ln, b["lines"][ln]["tile_days"], b["lines"][ln]["acts"],
                 b["lines"][ln]["contrib_per_action"], a["lines"][ln]["tile_days"],
                 a["lines"][ln]["acts"], a["lines"][ln]["contrib_per_action"]))
    print("\n  realised $/unit (revenue / units sold), base -> arm:")
    for ln in b["lines"]:
        if not (b["lines"][ln]["units_sold"] or a["lines"][ln]["units_sold"]):
            continue
        print("    %-11s %8s -> %-8s  (units %6.1f -> %6.1f)"
              % (ln, b["lines"][ln]["realised_per_unit"], a["lines"][ln]["realised_per_unit"],
                 b["lines"][ln]["units_sold"], a["lines"][ln]["units_sold"]))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"base": b, "arm": a, "verdict": res}, indent=1))
    print("\nVERDICT", json.dumps(res))


if __name__ == "__main__":
    main()
