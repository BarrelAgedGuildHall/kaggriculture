"""exp-012 acceptance screen: the distilled `pi0` against the 3-member ensemble it came from.

The offline agreement number is a proxy; THIS is the question the proxy stands in for -- does
the single net play the same game as the mixture? Pre-registered gate (spec section 2.2):
win rate in [45%, 55%] over 128 episodes (64 paired seeds, side-swapped), mirror exactly
50.0%, no crashes. Timing is reported per seat from the same games, so the rollout speed-up
is measured against the ensemble on identical boards rather than inherited from the spec's
assumption.

Top-level script on purpose (spawn workers re-import __main__; see scripts/ppo_calibrate.py).

    python scripts/distil_screen.py --student ~/ppo/out/pi0-exp012.npz
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo import stage as S            # noqa: E402
from agents.team_a.ppo.bcpath import WEIGHTS_DIR    # noqa: E402
from agents.team_a.ppo.screen import _rows, _summary  # noqa: E402
from arena import ratings, registry                 # noqa: E402
from arena.scheduler import run_match               # noqa: E402


def _timing(res, agent_id):
    """Median per-turn ms for `agent_id`, pooled over every game it played."""
    p50, p99, mx = [], [], []
    for g in res["rows"]:
        for seat in (0, 1):
            if g[f"p{seat}"] == agent_id:
                p50.append(float(g[f"p50_{seat}"]))
                p99.append(float(g[f"p99_{seat}"]))
                mx.append(float(g[f"max_{seat}"]))
    if not p50:
        return {}
    return {"p50_ms_median": round(statistics.median(p50), 3),
            "p50_ms_mean": round(statistics.fmean(p50), 3),
            "p99_ms_median": round(statistics.median(p99), 3),
            "max_ms": round(max(mx), 1), "games": len(p50)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True, help="distilled pi0 .npz")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--name", default="TeamA-pi0-distil")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    seeds = list(range(args.seed_base, args.seed_base + max(1, args.n // 2)))
    spec = f"{seeds[0]}-{seeds[-1]}"

    student = registry.register_file(S.stage("teamA-pi0-distil", [args.student]),
                                     args.name, "A", notes="exp-012 distilled single net")
    base = registry.register_file(S.stage("teamA-bc-base", sorted(WEIGHTS_DIR.glob("*.npz"))),
                                  "TeamA-bc-base-staged", "A", notes="shipped 3-member champion")
    m0 = registry.register_file(S.stage("teamA-pi0", [WEIGHTS_DIR / "rebuild-m0.npz"]),
                                "TeamA-pi0-m0", "A", notes="member 0 alone (exp-011 actor)")
    out = {"student": str(args.student), "seeds": [seeds[0], seeds[-1]],
           "episodes": 2 * len(seeds),
           "ids": {"student": student["id"], "base": base["id"], "m0": m0["id"]}}

    res = run_match([student["id"], base["id"]], mode="confirm", n=2 * len(seeds),
                    seeds=spec, workers=args.workers, steps=args.steps, quiet=True)
    out["vs_base"] = _summary(_rows(res, student["id"], base["id"]))
    out["timing_student"] = _timing(res, student["id"])
    out["timing_base_3member"] = _timing(res, base["id"])
    ps, pb = out["timing_student"].get("p50_ms_median"), out["timing_base_3member"].get("p50_ms_median")
    out["speedup_vs_3member"] = round(pb / ps, 3) if ps else None

    # member 0 alone, on the same boards: the honest "did distillation buy anything over
    # just taking one member" comparison, which is what exp-011 actually ran on.
    res_m0 = run_match([m0["id"], base["id"]], mode="confirm", n=2 * len(seeds),
                       seeds=spec, workers=args.workers, steps=args.steps, quiet=True)
    out["m0_vs_base"] = _summary(_rows(res_m0, m0["id"], base["id"]))
    out["timing_m0_1member"] = _timing(res_m0, m0["id"])

    twin = registry.register_file(S.stage_mirror([args.student], "teamA-pi0-distil-mirror"),
                                  args.name + "-mirror", "A")
    mr = run_match([student["id"], twin["id"]], mode="screen", n=16, seeds="0-7",
                   workers=min(args.workers, 8), steps=args.steps, quiet=True)
    rows = _rows(mr, student["id"], twin["id"])
    score = sum(1.0 if r["us"] > r["them"] else (0.5 if r["us"] == r["them"] else 0.0)
                for r in rows)
    by_seed = {}
    for g in mr["rows"]:
        by_seed.setdefault(int(g["seed"]), []).append(g)
    swap_ok = sum(1 for gs in by_seed.values() if len(gs) == 2
                  and abs(float(gs[0]["money0"]) - float(gs[1]["money0"])) < 1e-9
                  and abs(float(gs[0]["money1"]) - float(gs[1]["money1"])) < 1e-9)
    out["mirror"] = {"n": len(rows), "win_rate": round(score / max(1, len(rows)), 4),
                     "seat_swap_pairs": len(by_seed), "seat_swap_identical": swap_ok,
                     "extraction_mean": round(statistics.fmean([r["us"] + r["them"]
                                                               for r in rows]), 1) if rows else None}

    w = out["vs_base"]["win_rate"]
    lo, _ = ratings.wilson(w * out["vs_base"]["n"], out["vs_base"]["n"])
    out["acceptance"] = {
        "win_rate_in_45_55": 0.45 <= w <= 0.55,
        "wilson_lb_ge_40": lo >= 0.40,
        "mirror_exactly_50": out["mirror"]["win_rate"] == 0.5,
        "no_crashes": out["vs_base"]["crashes"] == 0,
    }
    out["pass"] = all(out["acceptance"].values())
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
