"""exp-011 phase P0 driver. Top-level script on purpose (see scripts/ppo_calibrate.py).

    uv run python scripts/ppo_train.py --iters 20 --workers 14 --out data/results/exp-011/run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo import auxlabels as AUX  # noqa: E402
from agents.team_a.ppo.train import AUX_DIMS, Run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop after this wall time")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seeds-per-iter", type=int, default=32)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--opponents", nargs="+", default=["pi0", "base"])
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--entropy", type=float, default=1e-3)
    ap.add_argument("--kl-target", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp-final", type=float, default=0.5)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--phase", default="P0", choices=["P0", "P1"],
                    help="P0 = market heads (exp-011); P1 = + market trunk + unit heads")
    ap.add_argument("--base-npz", default=None, help="starting weights (pi0 for exp-013)")
    ap.add_argument("--unit-subsample", type=float, default=0.25)
    ap.add_argument("--alpha-aux", type=float, default=0.0,
                    help="BC-auxiliary weight; the target is the FROZEN MIXTURE, never pi0")
    ap.add_argument("--aux-rows", nargs="*", default=[],
                    help="corpus row .npz chunks for the BC-auxiliary states")
    ap.add_argument("--unit-temperature", type=float, default=1.0,
                    help="FIXED sampling temperature of the unit head (never annealed)")
    ap.add_argument("--pfsp", action="store_true", help="PFSP opponent sampling + snapshots")
    # --- the alpha pilot / exp-014 knobs (halt ruling R-2.2, R-4.4; addendum-01 A-4) ---------
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="training return is R_us - ALPHA * R_them. 0 = the founder's standing "
                         "money directive (the pilot's control arm); 1 = pure margin. Logged on "
                         "every row like the engine pin -- numbers do not carry across a reward.")
    ap.add_argument("--critic", default="scalar", choices=["scalar", "factored"],
                    help="scalar = exp-013b's control; factored = exp-014's ONE variable")
    ap.add_argument("--critic-aux", nargs="*", default=[],
                    choices=sorted(AUX_DIMS), metavar="HEAD",
                    help="auxiliary supervised heads on the critic trunk, rollout-computed "
                         "(addendum-01 A-4.1). Excluded from value() by construction; "
                         "requires --critic factored")
    ap.add_argument("--critic-alpha-aux", type=float, default=0.1,
                    help="weight on the critic's auxiliary regression loss")
    args = ap.parse_args(argv)

    if args.critic_aux and args.critic != "factored":
        ap.error("--critic-aux requires --critic factored (a scalar critic has no aux heads)")
    missing = [k for k in args.critic_aux if k not in AUX.available()]
    if missing:
        ap.error(f"auxiliary head(s) {missing} have no label implementation in this process "
                 "(nav needs S-1's nav.py -- see agents/team_a/ppo/auxlabels.py:set_nav_fn)")

    run = Run(args.out, base_npz=args.base_npz, opponents=args.opponents,
              replicates=args.replicates,
              seeds_per_iter=args.seeds_per_iter, workers=args.workers, lr=args.lr,
              clip=args.clip, epochs=args.epochs, minibatch=args.minibatch,
              entropy=args.entropy, kl_target=args.kl_target, temperature=args.temperature,
              temp_final=args.temp_final, seed_base=args.seed_base, steps=args.steps,
              phase=args.phase, unit_subsample=args.unit_subsample,
              alpha_aux=args.alpha_aux, aux_rows=args.aux_rows, pfsp=args.pfsp,
              unit_temperature=args.unit_temperature,
              alpha=args.alpha, critic_kind=args.critic,
              critic_aux=tuple(args.critic_aux), critic_alpha_aux=args.critic_alpha_aux,
              snapshot_every=args.snapshot_every)
    run.total_iters = args.iters
    Path(args.out, "config.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    t0 = time.perf_counter()
    for it in range(args.iters):
        row = run.iteration(it)
        print(json.dumps(row), flush=True)
        if (it + 1) % args.snapshot_every == 0 or it == args.iters - 1:
            run.snapshot(it)
        if args.minutes and (time.perf_counter() - t0) / 60.0 >= args.minutes:
            run.snapshot(it)
            print(json.dumps({"stopped": "wall-clock budget", "iters_done": it + 1}), flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
