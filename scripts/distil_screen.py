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
from agents.team_a.ppo.screen import _rows, _summary, paired_delta  # noqa: E402
from arena import ratings, registry                 # noqa: E402
from arena.scheduler import run_match               # noqa: E402


def hub_paused(url="http://localhost:8765/api/status"):
    """Precondition of addendum-01 section 7.2: the pause must be verified at the START and the
    END of every measured arm, and an arm across which it flipped is DISCARDED. exp-011's
    screen was contaminated when the hub un-paused itself mid-run."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return bool(json.loads(r.read())["hub"]["paused"])
    except Exception as exc:                      # noqa: BLE001
        return f"unknown: {exc}"


def h2h_delta(rows, boot=20000, seed=0):
    """Gate D statistic: the PAIRED per-episode bank difference between the two seats of the
    same game (identical board, both agents present), expressed as a % of the opponent's
    mean bank. Paired by construction -- same-family paired SD is $2-4.8k against $9.3-10k
    unpaired, which is the whole reason this is the primary statistic."""
    import numpy as np
    d = np.array([r["us"] - r["them"] for r in rows], dtype=float)
    them = statistics.fmean([r["them"] for r in rows])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n_pairs": len(d), "mean_diff": round(float(d.mean()), 1),
            "paired_sd": round(float(d.std(ddof=1)), 1),
            "ci95": [round(float(lo), 1), round(float(hi), 1)],
            "opp_mean": round(them, 1),
            "pct_change": round(100.0 * float(d.mean()) / them, 3),
            "pct_ci95": [round(100.0 * float(lo) / them, 3),
                         round(100.0 * float(hi) / them, 3)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


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
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--name", default="TeamA-pi0-distil")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    seeds = list(range(args.seed_base, args.seed_base + max(1, args.n // 2)))
    spec = f"{seeds[0]}-{seeds[-1]}"
    paused_start = hub_paused()

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
    rows_vb = _rows(res, student["id"], base["id"])
    out["vs_base"] = _summary(rows_vb)
    out["timing_student"] = _timing(res, student["id"])
    out["timing_base_3member"] = _timing(res, base["id"])
    ps, pb = out["timing_student"].get("p50_ms_median"), out["timing_base_3member"].get("p50_ms_median")
    out["speedup_vs_3member"] = round(pb / ps, 3) if ps else None

    # member 0 alone, on the same boards: the honest "did distillation buy anything over
    # just taking one member" comparison, which is what exp-011 actually ran on.
    res_m0 = run_match([m0["id"], base["id"]], mode="confirm", n=2 * len(seeds),
                       seeds=spec, workers=args.workers, steps=args.steps, quiet=True)
    rows_m0 = _rows(res_m0, m0["id"], base["id"])
    out["m0_vs_base"] = _summary(rows_m0)
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

    out["gate_D_paired_bank_vs_base"] = h2h_delta(rows_vb)
    out["gate_R_paired_vs_member0"] = paired_delta(rows_vb, rows_m0)

    paused_end = hub_paused()
    out["pause"] = {"start": paused_start, "end": paused_end,
                    "held": paused_start is True and paused_end is True}

    w = out["vs_base"]["win_rate"]
    lo, _ = ratings.wilson(w * out["vs_base"]["n"], out["vs_base"]["n"])
    t = out["timing_student"]
    D = out["gate_D_paired_bank_vs_base"]["pct_change"]
    R = out["gate_R_paired_vs_member0"]
    # addendum-01 section 2.3. Gate B (pnl_census) is run separately by scripts/pi0_accept.py
    # and merged into the row; C3 (train/serve bit-parity) and C4 (legality) likewise.
    out["gates"] = {
        "C1_mirror_exactly_50": out["mirror"]["win_rate"] == 0.5
                                and out["mirror"]["seat_swap_identical"] == out["mirror"]["seat_swap_pairs"],
        "C2_no_crash_timeout": out["vs_base"]["crashes"] == 0 and out["vs_base"]["timeouts"] == 0,
        "C2_p50_le_25ms": t.get("p50_ms_median", 999) <= 25.0,
        "C2_p99_le_150ms": t.get("p99_ms_median", 999) <= 150.0,
        "D_paired_bank_ge_minus8pct": D >= -8.0,
        "R_beats_member0_ci_excludes_zero": bool(R.get("mean_diff", 0) > 0
                                                 and R.get("ci_excludes_zero", False)),
        "pause_held": out["pause"]["held"],
    }
    out["pass_pit_gates"] = all(out["gates"].values())
    # the superseded instrument, reported alongside so the row can label both verdicts
    out["old_gate_verdict"] = {"win_rate_in_45_55": 0.45 <= w <= 0.55,
                               "wilson_lb": round(lo, 4), "win_rate": w}
    out["delta_vs_exp012_floor_pct"] = round(D - (-5.1), 3)
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if out["pass_pit_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
