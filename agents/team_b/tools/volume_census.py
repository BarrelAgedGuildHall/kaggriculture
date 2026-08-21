"""Per-player, per-product volume census over a battery of instrumented episodes.

Runs `instrument.run` for a list of seeds (optionally swapping seats on the
second half so both agents see both seats), then aggregates BY AGENT SPEC:
units sold per season, realised $/unit, buys, final money — the direct
measurement of which pools each agent contests and how much of each pool is
left open (Architect B's EV proposal, Option 1 step 0 / step 1 census).

Usage (from the repo root):
    uv run python agents/team_b/tools/volume_census.py \
        --a millrace:feedchain --b data/arena/agents/X@sha/main.py \
        --seeds 0-7 --swap-half --out-dir ../bench/runs/expB-218/vsbc

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def _one(job):
    spec_a, spec_b, seed, steps, out_dir = job
    import instrument
    s = instrument.run(spec_a, spec_b, seed, steps, None)
    p = Path(out_dir) / f"census_seed{seed}.json"
    p.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return {"seed": seed, "agents": [spec_a, spec_b],
            "final_money": s["final_money"], "statuses": s["statuses"],
            "cum_sales": s["cum_sales"], "cum_rev": s["cum_rev"],
            "cum_buys": s["cum_buys"],
            "fertilizer_floor_day": s["fertilizer_floor_day"],
            "move_fraction": s["move_fraction"], "pass_fraction": s["pass_fraction"]}


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="agent spec (instrument.load_agent grammar)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", default="0-7", help="e.g. 0-7 or 0,1,5")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--swap-half", action="store_true", help="swap seats on the 2nd half of the seed list")
    ap.add_argument("--workers", type=int, default=4)
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

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            rows = pool.map(_one, jobs)
    else:
        rows = [_one(j) for j in jobs]

    # ---- aggregate by agent spec -------------------------------------------
    specs = sorted({sp for r in rows for sp in r["agents"]})
    agg = {sp: {"episodes": 0, "money": [],
                "units": {p: [] for p in PRODUCTS}, "rev": {p: [] for p in PRODUCTS},
                "buys": {}} for sp in specs}
    combined = {p: [] for p in PRODUCTS}  # both players together, per episode
    for r in rows:
        for seat, sp in enumerate(r["agents"]):
            a = agg[sp]
            a["episodes"] += 1
            a["money"].append(r["final_money"][seat])
            for p in PRODUCTS:
                a["units"][p].append(r["cum_sales"][seat].get(p, 0))
                a["rev"][p].append(r["cum_rev"][seat].get(p, 0))
            for k, v in r["cum_buys"][seat].items():
                a["buys"][k] = a["buys"].get(k, 0) + v
        for p in PRODUCTS:
            combined[p].append(sum(r["cum_sales"][seat].get(p, 0) for seat in range(2)))

    report = {"engine": rows and json.loads(Path(out_dir, f"census_seed{seeds[0]}.json").read_text())["engine"],
              "seeds": seeds, "n_episodes": len(rows), "specs": specs, "per_agent": {}, "combined_units_mean": {}}
    for sp in specs:
        a = agg[sp]
        per = {"episodes": a["episodes"],
               "money_mean": round(statistics.fmean(a["money"]), 0),
               "money_std": round(statistics.pstdev(a["money"]), 0),
               "products": {}}
        for p in PRODUCTS:
            u = a["units"][p]
            rv = a["rev"][p]
            tot_u, tot_r = sum(u), sum(rv)
            per["products"][p] = {
                "units_mean": round(statistics.fmean(u), 1),
                "units_std": round(statistics.pstdev(u), 1),
                "rev_mean": round(statistics.fmean(rv), 0),
                "dollars_per_unit": round(tot_r / tot_u, 2) if tot_u else None,
            }
        per["buys_total"] = dict(sorted(a["buys"].items()))
        report["per_agent"][sp] = per
    for p in PRODUCTS:
        report["combined_units_mean"][p] = round(statistics.fmean(combined[p]), 1)

    Path(out_dir, "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"n={len(rows)} episodes, seeds {seeds[0]}..{seeds[-1]}")
    for sp in specs:
        per = report["per_agent"][sp]
        print(f"\n== {sp}  (n={per['episodes']}, bank {per['money_mean']}+-{per['money_std']})")
        print(f"{'product':<11} {'units':>7} {'+-':>6} {'$/unit':>8} {'rev':>8}")
        for p in PRODUCTS:
            d = per["products"][p]
            dpu = f"{d['dollars_per_unit']:.2f}" if d["dollars_per_unit"] is not None else "-"
            print(f"{p:<11} {d['units_mean']:>7} {d['units_std']:>6} {dpu:>8} {d['rev_mean']:>8}")
    print("\ncombined units/season (both players):",
          json.dumps(report["combined_units_mean"]))
    print("AGGREGATE " + json.dumps({k: v for k, v in report.items() if k != "per_agent"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
