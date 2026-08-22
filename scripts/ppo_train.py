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

from agents.team_a.ppo.train import Run  # noqa: E402


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
    ap.add_argument("--pfsp", action="store_true", help="PFSP opponent sampling + snapshots")
    args = ap.parse_args(argv)

    run = Run(args.out, base_npz=args.base_npz, opponents=args.opponents,
              replicates=args.replicates,
              seeds_per_iter=args.seeds_per_iter, workers=args.workers, lr=args.lr,
              clip=args.clip, epochs=args.epochs, minibatch=args.minibatch,
              entropy=args.entropy, kl_target=args.kl_target, temperature=args.temperature,
              temp_final=args.temp_final, seed_base=args.seed_base, steps=args.steps,
              phase=args.phase, unit_subsample=args.unit_subsample,
              alpha_aux=args.alpha_aux, aux_rows=args.aux_rows, pfsp=args.pfsp,
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
