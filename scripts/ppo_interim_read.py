"""Interim read on a live PPO run: the GREEDY checkpoint's paired money vs a reference.

This is **not** a Pit gate and is never quoted as one. It exists because the exp-013 tripwire is
a *money* statistic, and exp-011 step 0 established that this harness and the Pit produce
**byte-identical banks on 128/128 episodes** (mean_abs_diff 0.0) -- so a paired money number
measured here is the number the Pit would return. What the harness cannot measure is timing (C2),
because the training run is using most of the box; anything timing-related still has to go through
a quiesced Pit with the ladder paused.

Three instruments, all pre-registered:

* **Gate D** (addendum 01) -- our money minus the opponent's as a percentage of the opponent's,
  bootstrap CI over (seed, seat) pairs. Unchanged.
* **`K7'-b`** (halt ruling R-6.2 as amended by addendum-01 A-3.3) -- paired margin `d` and
  `d/bank` for the current greedy artefact **and for the run's iteration-0 policy on the same
  fixed probe boards**, with the trigger sized in **SE of the probe's own paired differences**,
  not in per-episode SD. That distinction is the whole content of A-3.3: at Team A's own
  `sigma_pair ~ $16k` the exp-013 decline of $5,588 is 0.35 per-episode SD (misses) but 2.79 SE at
  n = 64 (fires). **>= 64 paired seat-pairs**, raised from 32.
* **`K8'`** (R-5.1 as amended by A-1.4) -- the sampled policy read against the greedy artefact on
  the same boards. Two forms: an **admission check** before an arm's first gradient step
  (`--admission`), and the in-run monitor. Both report the money gap **and the `d/bank` gap with
  its sign**, because D-3's sharper failure was not the money gap but the two series moving in
  opposite directions on the deciding statistic.

    # admission check, before launching an arm
    uv run python scripts/ppo_interim_read.py --checkpoint pi0.npz --admission \\
        --temperature 1.0 --opponent base --seeds 3096-3127 --workers 6

    # interim read on a live run
    uv run python scripts/ppo_interim_read.py --checkpoint ckpt/iter005.npz --iter0 pi0.npz \\
        --opponent base --seeds 3096-3159 --temperature 1.0 --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo import pool as POOL     # noqa: E402

# addendum-01 A-3.2: Team A's own same-family paired sigma, for the MDE table only. It is NEVER
# used as the trigger's denominator -- A-3.2 makes that a standing rule ("a gate expressed in SDs
# computes its SD from the matchup it is evaluating, on that arm's own paired differences, never
# from a stored constant"). B-1's $2.0-5.8k is a Millrace-lineage number and must not appear here.
SIGMA_PAIR_REFERENCE = 16_000.0
K7_Z = 1.645                    # one-sided 95%
K8_GAP_LIMIT = 0.25             # sampled below greedy by > 25% of greedy money


def _boot_ci(d, boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(boot, len(d)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def paired(rows, boot=20000, seed=0):
    d = np.array([r["us"] - r["them"] for r in rows], dtype=float)
    them = np.array([r["them"] for r in rows], dtype=float)
    ours = np.array([r["us"] for r in rows], dtype=float)
    lo, hi = _boot_ci(d, boot, seed)
    ref = float(them.mean())
    own = float(ours.mean())
    return {"n_pairs": len(d), "mean_diff": round(float(d.mean()), 1),
            "paired_sd": round(float(d.std(ddof=1)), 1),
            "paired_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 1),
            "pct_change": round(100.0 * float(d.mean()) / ref, 3),
            "pct_ci95": [round(100.0 * float(lo) / ref, 3),
                         round(100.0 * float(hi) / ref, 3)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            # THE decision variable (R-2.3): paired margin as a fraction of OWN bank. Note the
            # denominator -- `pct_change` above divides by the opponent's bank because that is
            # Gate D's definition, and the two differ by a few percent.
            "d_over_bank_pct": round(100.0 * float(d.mean()) / own, 3),
            "our_money_mean": round(own, 1),
            "opp_money_mean": round(ref, 1),
            "extraction_mean": round(float(np.mean(ours + them)), 1),
            "timeouts": int(sum(1 for r in rows if r.get("timeout"))),
            "win_rate": round(float(np.mean((ours > them) + 0.5 * (ours == them))), 4)}


def play(actor, opponent, seeds, workers, steps=720, temperature=None, rng_seed=0):
    jobs = [{"seed": s, "seat": seat, "actor": actor, "opponent": opponent,
             "opponent_id": str(opponent), "steps": steps,
             "temperature": temperature, "rng_seed": rng_seed + 977 * s + seat}
            for s in seeds for seat in (0, 1)]
    return POOL.run(jobs, workers, kind="eval")


def _by_board(rows):
    return {(r["seed"], r["seat"]): r for r in rows}


def k7_prime(cand_rows, ref_rows):
    """K7'-b. `ref_rows` is the run's ITERATION-0 policy on the SAME boards (R-6.2)."""
    c, p = _by_board(cand_rows), _by_board(ref_rows)
    keys = sorted(set(c) & set(p))
    dc = np.array([c[k]["us"] - c[k]["them"] for k in keys], dtype=float)
    dp = np.array([p[k]["us"] - p[k]["them"] for k in keys], dtype=float)
    ours_c = np.array([c[k]["us"] for k in keys], dtype=float)
    ours_p = np.array([p[k]["us"] for k in keys], dtype=float)
    them_c = np.array([c[k]["them"] for k in keys], dtype=float)
    them_p = np.array([p[k]["them"] for k in keys], dtype=float)
    n = len(keys)

    # The SE the addendum's MDE table is built from: the SD of the probe's own paired margins,
    # over root-n. Pooled across the two arms because both are measurements of the same matchup
    # on the same boards, and A-3.2 requires it be computed here rather than looked up.
    sigma = float(np.std(np.concatenate([dc, dp]), ddof=1))
    se = sigma / np.sqrt(n)
    decline = float(dp.mean() - dc.mean())          # positive == margin got worse
    # The doubly-paired form is strictly more powerful (the board-common component of the margin
    # cancels), so it is reported as a secondary read. It is NOT the trigger: the pre-registered
    # unit is the one the MDE table was written against, and swapping in a tighter statistic after
    # seeing the data is exactly the drift A-2.4 forbids.
    dd = dp - dc
    se_dd = float(np.std(dd, ddof=1) / np.sqrt(n))
    lo, hi = _boot_ci(-dd, seed=3)

    money_up = bool(ours_c.mean() > ours_p.mean())
    return {
        "n_pairs": n,
        "sigma_pair": round(sigma, 1), "se_paired_mean": round(se, 1),
        "sigma_pair_reference": SIGMA_PAIR_REFERENCE,
        "iter0": {"our_bank": round(float(ours_p.mean()), 1),
                  "opp_bank": round(float(them_p.mean()), 1),
                  "margin": round(float(dp.mean()), 1),
                  "d_over_bank_pct": round(100.0 * float(dp.mean()) / float(ours_p.mean()), 3)},
        "candidate": {"our_bank": round(float(ours_c.mean()), 1),
                      "opp_bank": round(float(them_c.mean()), 1),
                      "margin": round(float(dc.mean()), 1),
                      "d_over_bank_pct": round(100.0 * float(dc.mean())
                                               / float(ours_c.mean()), 3)},
        # R-6.2's required decomposition, reported whether or not the trigger fires.
        "decomposition": {
            "d_ours": round(float(ours_c.mean() - ours_p.mean()), 1),
            "d_theirs": round(float(them_c.mean() - them_p.mean()), 1),
            "d_extraction": round(float((ours_c + them_c).mean() - (ours_p + them_p).mean()), 1)},
        "margin_decline": round(decline, 1),
        "decline_in_se": round(decline / max(1e-9, se), 3),
        "threshold_se": K7_Z,
        "mde_at_1645se": round(K7_Z * se, 1),
        "doubly_paired": {"mean_decline": round(float(dd.mean()), 1),
                          "se": round(se_dd, 1),
                          "decline_in_se": round(float(dd.mean()) / max(1e-9, se_dd), 3),
                          "ci95_on_change": [round(float(lo), 1), round(float(hi), 1)]},
        "money_up": money_up,
        # ESCALATE AND PROCEED (never a halt -- R-6.3: when a priced tripwire and a hard gate
        # fire on the same evidence, the halt governs; this only makes the escalation earlier).
        "fires": bool(money_up and decline > K7_Z * se),
    }


def k8_prime(greedy_rows, sampled_rows, temperature):
    """K8'. Same boards, greedy artefact vs the sampled policy at the arm's temperature."""
    g, s = _by_board(greedy_rows), _by_board(sampled_rows)
    keys = sorted(set(g) & set(s))
    gm = np.array([g[k]["us"] for k in keys], dtype=float)
    sm = np.array([s[k]["us"] for k in keys], dtype=float)
    gd = np.array([g[k]["us"] - g[k]["them"] for k in keys], dtype=float)
    sd = np.array([s[k]["us"] - s[k]["them"] for k in keys], dtype=float)
    gap = float(sm.mean() - gm.mean()) / max(1e-9, float(gm.mean()))
    g_db = float(gd.mean()) / max(1e-9, float(gm.mean()))
    s_db = float(sd.mean()) / max(1e-9, float(sm.mean()))
    return {
        "n_pairs": len(keys), "temperature": temperature,
        "greedy_money": round(float(gm.mean()), 1), "sampled_money": round(float(sm.mean()), 1),
        "money_gap_pct": round(100.0 * gap, 2),
        "limit_pct": -100.0 * K8_GAP_LIMIT,
        # A-1.4's addition: D-3's sharper failure was the two series moving in OPPOSITE directions
        # on the deciding statistic, which a money gap alone does not show.
        "greedy_d_over_bank_pct": round(100.0 * g_db, 3),
        "sampled_d_over_bank_pct": round(100.0 * s_db, 3),
        "d_over_bank_gap_pp": round(100.0 * (s_db - g_db), 3),
        "same_sign": bool(np.sign(g_db) == np.sign(s_db)),
        "admission_pass": bool(gap >= -K8_GAP_LIMIT),
        "trails_by_more_than_limit": bool(gap < -K8_GAP_LIMIT),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--iter0", default=None,
                    help="the run's ITERATION-0 policy -- K7'-b's reference (R-6.2)")
    ap.add_argument("--reference", default=None, help="pi0 weights, for the improvement read")
    ap.add_argument("--opponent", default="base")
    ap.add_argument("--seeds", default="3096-3159",
                    help="RESERVED PROBE band. Never a decision band: 3032-3095 is the alpha "
                         "pilot's decision band and 3000-3031 is spent as the halt probe.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="the arm's launch temperature, for the K8' sampled read")
    ap.add_argument("--admission", action="store_true",
                    help="K8' admission check only: greedy vs sampled on --checkpoint")
    ap.add_argument("--no-k8", action="store_true", help="skip the sampled read")
    ap.add_argument("--label", default=None, help="arm name recorded in the output")
    ap.add_argument("--look", default=None, help="which look this is (25%%/50%%/100%%)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    out = {"checkpoint": args.checkpoint, "opponent": args.opponent, "label": args.label,
           "look": args.look, "seeds": [seeds[0], seeds[-1]], "episodes": 2 * len(seeds),
           "temperature": args.temperature,
           "note": "harness read, calibration-backed (exp-011 step 0); NOT a Pit gate"}

    cand = play([args.checkpoint], args.opponent, seeds, args.workers, args.steps)
    out["candidate_vs_opponent"] = paired(cand)

    if not args.no_k8:
        samp = play([args.checkpoint], args.opponent, seeds, args.workers, args.steps,
                    temperature=args.temperature)
        out["k8_prime"] = k8_prime(cand, samp, args.temperature)
        if args.admission:
            out["k8_prime"]["role"] = "iteration-0 admission check (A-1.4)"
            out["admission_verdict"] = ("PASS -- arm may launch at this temperature"
                                        if out["k8_prime"]["admission_pass"]
                                        else "FAIL -- lower T and re-check; arm does NOT launch")

    if args.iter0 and not args.admission:
        ref0 = play([args.iter0], args.opponent, seeds, args.workers, args.steps)
        out["iter0_vs_opponent"] = paired(ref0)
        out["k7_prime"] = k7_prime(cand, ref0)

    if args.reference:
        ref = play([args.reference], args.opponent, seeds, args.workers, args.steps)
        out["reference_vs_opponent"] = paired(ref)
        c, p = _by_board(cand), _by_board(ref)
        keys = sorted(set(c) & set(p))
        d = np.array([c[k]["us"] - p[k]["us"] for k in keys], dtype=float)
        lo, hi = _boot_ci(d, seed=1)
        base = float(np.mean([p[k]["us"] for k in keys]))
        out["candidate_minus_reference"] = {
            "n_pairs": len(d), "mean_diff": round(float(d.mean()), 1),
            "pct_change": round(100.0 * float(d.mean()) / base, 3),
            "pct_ci95": [round(100.0 * float(lo) / base, 3),
                         round(100.0 * float(hi) / base, 3)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}

    # the pre-registered exp-013 tripwire (addendum 01 section 4.3)
    g = out["candidate_vs_opponent"]
    out["tripwire_parity_with_frozen_base"] = {
        "statistic_pct": g["pct_change"], "ci95": g["pct_ci95"],
        "reaches_parity": bool(g["pct_change"] >= 0.0 or g["pct_ci95"][1] >= 0.0)}

    # A-2.3(3), standing: the arena scores an over-budget turn as a loss regardless of bank, so a
    # contended read is not a read. Discarded, never corrected.
    n_to = sum(v.get("timeouts", 0) for k, v in out.items()
               if isinstance(v, dict) and "timeouts" in v)
    out["contention"] = {"timeouts": int(n_to),
                         "verdict": "DISCARD THIS READ" if n_to else "clean"}
    print(json.dumps(out, indent=1), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
