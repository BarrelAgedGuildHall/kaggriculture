"""Calibrate the sampling temperature of the UNIT head before a P1 run.

Why this exists. `exp-011` measured the market order head at **0.040 nats/factor at T = 1**
-- the sampled policy *was* the greedy policy -- and raised the temperature to 2.0 so that
PPO had a ratio to work with. Phase P1 adds an **18-way op head**, which is nowhere near that
peaked, and re-using T = 2 there was measured (this script, first run) to turn the actor into
a different and far worse agent: PPO would then spend a day-long run learning to recover from
self-inflicted noise, and every rollout statistic would describe a policy we never ship.

What it reports, per candidate unit temperature: op-head entropy, the sampled actor's mean
bank against a fixed opponent on fixed seeds, and the paired difference against the SAME
actor run greedily on the same boards. The temperature we want is the highest one whose
paired money cost is small -- exploration is worth paying for, but not the farm.

Seeds come from the training band (5900+) on purpose: this is a training-configuration
measurement, not a result, and it must not consume screen (0-199) or confirm (1000-1199) seeds.

    uv run python scripts/ppo_temp_calib.py --base <pi0.npz> --temps 1.0 1.15 1.3 1.6 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo import pool as POOL          # noqa: E402
from agents.team_a.ppo import unit_head as UH       # noqa: E402
from agents.team_a.ppo.bcpath import P              # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="actor weights (pi0)")
    ap.add_argument("--opponent", default="base")
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.15, 1.3, 1.6, 2.0])
    ap.add_argument("--mkt-temperature", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=5900)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    seeds = [args.seed_base + i for i in range(args.seeds)]
    out = {"base": args.base, "opponent": args.opponent, "seeds": seeds,
           "mkt_temperature": args.mkt_temperature, "arms": []}

    for T in args.temps:
        jobs = [{"seed": s, "seat": seat, "actor": [args.base],
                 "opponent": args.opponent, "opponent_id": args.opponent,
                 "rng_seed": 7 * s + seat, "temperature": args.mkt_temperature,
                 "unit_temperature": T, "record_units": True, "unit_subsample": 0.05,
                 "steps": args.steps}
                for s in seeds for seat in (0, 1)]
        res = POOL.run(jobs, args.workers, kind="rollout")
        R = np.array([r["R"] for r in res], dtype=float)
        Ro = np.array([r["R_opp"] for r in res], dtype=float)

        rows = [u for r in res for u in r["unit_rows"]]
        m = P.Member(args.base)
        b_x = np.stack([u["x"] for u in rows])
        b_m = np.stack([u["op_mask"] for u in rows]).astype(np.float32)
        d = UH.dists(m.unit, m.umu, m.usd, b_x, b_m, None, None, T)
        h_op = float(UH.entropy(d)[0].mean())
        d1 = UH.dists(m.unit, m.umu, m.usd, b_x, b_m, None, None, 1.0)
        h_op1 = float(UH.entropy(d1)[0].mean())

        arm = {"unit_temperature": T, "entropy_op": round(h_op, 4),
               "entropy_op_at_T1": round(h_op1, 4),
               "money_mean": float(R.mean()), "money_sd": float(R.std(ddof=1)),
               "opp_money_mean": float(Ro.mean()),
               "win_rate": float(np.mean((R > Ro) + 0.5 * (R == Ro))),
               "episodes": len(res)}
        out["arms"].append(arm)
        print(json.dumps(arm), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
