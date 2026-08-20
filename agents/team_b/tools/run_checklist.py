"""Driver for the sim-checklist A/B sweeps (expB-101..112, minus instrumentation).

Runs each A/B via run_ab.main() with fixed seeds, paired + side-swapped,
writing one JSON per A/B to the runs dir. Idempotent: skips an A/B whose
output file already exists. Two seed bands: --band screen (seeds 0..pairs-1,
default) and --band heldout (seeds 1000..1000+pairs-1, output files suffixed
`_heldout`) — any experiment quoting a held-out replication must be
reproducible with `--only expB-NNN --band heldout`.

Usage: uv run python agents/team_b/tools/run_checklist.py [--only expB-103] [--band heldout] [--runs-dir ../../runs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ab  # noqa: E402

J = json.dumps


def p(d):  # probe spec from param dict
    return "probe:" + J(d)


SWEEP = []

# expB-101: day-0 goose maximum vs gradual ramp (predict: max by >15 pp)
SWEEP.append(("expB-101", "rush_vs_ramp", "probe:GooseRush", "probe:GradualRamp", 32, 720))

# expB-106: class-aware metering vs sell-on-sight, production frozen (same Mixed portfolio)
SWEEP.append(("expB-106", "meter_vs_onsight", "probe:MeterSell", "probe:Mixed", 32, 720))

# expB-103: goose/wheat split sweep vs the 60/40 proxy anchor (57/38 at ~95 tiles)
_anchor = {"goose_target": 57, "wheat_tiles": 38, "hands_target": 13}
for gt, wt in [(95, 0), (76, 19), (66, 28), (48, 47), (38, 57)]:
    SWEEP.append(("expB-103", f"split_{gt}_{wt}_vs_57_38",
                  p({"goose_target": gt, "wheat_tiles": wt, "hands_target": 13}),
                  p(_anchor), 32, 720))

# expB-105: cows+sheep displace 12 geese, tiles held constant (predict mixed +10 pp)
SWEEP.append(("expB-105", "cowsheep_vs_allgoose",
              p({"cows": 7, "sheep": 5, "goose_target": 43}), p({"goose_target": 55}), 32, 720))

# expB-104: fertilizing wheat tiles on vs off (predict +8 pp)
SWEEP.append(("expB-104", "fertwheat_on_vs_off",
              p({}), p({"fertilize_wheat": False}), 32, 720))

# expB-109: hire depth sweep vs 10 (predict peak 13-14)
for h in [8, 12, 13, 14, 15, 17]:
    SWEEP.append(("expB-109", f"hands_{h}_vs_10",
                  p({"hands_target": h}), p({"hands_target": 10}), 24, 720))

# expB-112: 5-day game (steps=120): goose-max vs carrot-max vs mixed (predict goose-max wins)
_g5 = {"season_days": 5, "geese_day0": 9, "wheat_buy_open": 8, "goose_target": 9,
       "wheat_tiles": 0, "carrots_day0": 0, "fertilize_wheat": False, "buy_land": False,
       "hands_target": 4, "capex_stop_day": 4}
_c5 = {"season_days": 5, "geese_day0": 0, "wheat_buy_open": 0, "goose_target": 0,
       "wheat_tiles": 0, "carrots_day0": 20, "fertilize_wheat": False, "buy_land": False,
       "hands_target": 4, "capex_stop_day": 4}
_m5 = {"season_days": 5, "geese_day0": 5, "wheat_buy_open": 5, "goose_target": 5,
       "wheat_tiles": 0, "carrots_day0": 8, "fertilize_wheat": False, "buy_land": False,
       "hands_target": 4, "capex_stop_day": 4}
SWEEP.append(("expB-112", "goose5_vs_carrot5", p(_g5), p(_c5), 24, 120))
SWEEP.append(("expB-112", "goose5_vs_mixed5", p(_g5), p(_m5), 24, 120))
SWEEP.append(("expB-112", "mixed5_vs_carrot5", p(_m5), p(_c5), 24, 120))

# expB-110: shop-draw adaptivity — daily carrot re-pricing vs fixed 6-carrot opening plan
# (extreme-band variant runs the same arms via run_ab --seed-list over the
#  PET_CAFE>=2 seeds from scan_shops.py)
SWEEP.append(("expB-110", "adaptive_vs_fixed6",
              p({"adaptive_carrots": True}), p({"carrots_day0": 6}), 32, 720))

BANDS = {"screen": (0, ""), "heldout": (1000, "_heldout")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only this expB id")
    ap.add_argument("--band", choices=sorted(BANDS), default="screen",
                    help="seed band: screen = seeds 0.., heldout = seeds 1000.. (suffix _heldout)")
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[3].parent / "runs"))
    args = ap.parse_args(argv)
    runs = Path(args.runs_dir)
    seed_base, suffix = BANDS[args.band]
    for exp, name, a, b, pairs, steps in SWEEP:
        if args.only and exp != args.only:
            continue
        out = runs / exp / f"{name}{suffix}.json"
        if out.exists():
            print(f"skip {exp}/{name}{suffix} (exists)")
            continue
        print(f"=== {exp}/{name}{suffix}  pairs={pairs} steps={steps} seeds={seed_base}..", flush=True)
        run_ab.main(["--a", a, "--b", b, "--pairs", str(pairs), "--steps", str(steps),
                     "--seed-base", str(seed_base),
                     "--tag", f"{exp}:{name}{suffix}", "--out", str(out)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
