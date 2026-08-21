"""exp-011 phase P0: PPO on the market net's orders/qty/hire heads. Everything else frozen.

One iteration = rollouts -> advantages -> a few epochs of the clipped surrogate -> a new
weights file the next batch of workers loads. Nothing about the policy code changes; the
`.npz` is both the trainer's checkpoint and the submission artefact (spec section 2.1).

Baselines (spec section 1.2), both action-independent, so the objective is untouched:

* **leave-one-out over (seed, seat, opponent).** Every cell is played `--replicates` times
  with different sampling rng, and an episode's baseline is the mean return of the *others*
  in its cell. This is not polish. Money is strongly opponent- and seed-dependent (bank SD
  is ~$20k across seeds, ~$3k within a matched pair), so without it the policy learns which
  seeds are rich. The seat is part of the cell because the two seats are not symmetric --
  weed spawn is keyed on farm index.
* **critic.** GAE lambda = 1.0, i.e. `A_t = (R - b_LOO)/S - V(s_t)`: the unbiased control arm
  the spec names, and the right default for a pilot whose only question is whether money
  moves at all.

Training seeds live in their own band (default 5000+) so the screen band (0..199) and the
confirm band (1000..1199) stay held out.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import harness as H
from . import pool as POOL
from . import ppo as PPO
from .bcpath import F, P, WEIGHTS_DIR

REWARD_SCALE = 50_000.0
TRAIN_SEED_BASE = 5000


def save_member(src_npz, mkt, out_path):
    """Write a new ensemble member: the source weights with the market head replaced."""
    d = dict(np.load(src_npz, allow_pickle=True))
    for k, v in mkt.state().items():
        d["m_" + k] = np.asarray(v, dtype=np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)
    return str(out_path)


def stack_batch(results):
    """Flatten recorded turns into column arrays + a per-row episode index."""
    turns, ep_of_row = [], []
    for i, r in enumerate(results):
        turns.extend(r["turns"])
        ep_of_row.extend([i] * len(r["turns"]))
    b = {
        "x": np.stack([t["x"] for t in turns]),
        "order_mask": np.stack([t["order_mask"] for t in turns]),
        "fire": np.stack([t["fire"] for t in turns]).astype(np.int32),
        "qty_idx": np.stack([t["qty_idx"] for t in turns]).astype(np.int32),
        "hire_idx": np.array([int(t["hire_idx"]) for t in turns]),
        "exec_fire": np.stack([t["exec_fire"] for t in turns]),
        "exec_qty": np.stack([t["exec_qty"] for t in turns]),
        "exec_hire": np.array([int(t["exec_hire"]) for t in turns]),
        "logp_fire": np.stack([t["logp_fire"] for t in turns]),
        "logp_qty": np.stack([t["logp_qty"] for t in turns]),
        "logp_hire": np.array([float(t["logp_hire"]) for t in turns]),
    }
    return b, np.asarray(ep_of_row)


def loo_baseline(results):
    """Leave-one-out mean return within each (seed, seat, opponent) cell."""
    cells = {}
    for i, r in enumerate(results):
        cells.setdefault((r["seed"], r["seat"], r["opponent_id"]), []).append(i)
    base = np.zeros(len(results))
    singles = 0
    for idx in cells.values():
        rs = np.array([results[i]["R"] for i in idx], dtype=float)
        if len(idx) == 1:
            base[idx[0]] = rs[0]        # degenerate cell: no signal, advantage 0
            singles += 1
            continue
        tot = rs.sum()
        for k, i in enumerate(idx):
            base[i] = (tot - rs[k]) / (len(idx) - 1)
    return base, singles


def slice_batch(b, idx):
    return {k: v[idx] for k, v in b.items()}


def chunked_old_logprobs(trainer, b, frozen, chunk=8192):
    """pi_old over the whole batch, in chunks -- p_qty is (B, 20, 18) float64 and a full
    184k-row batch would allocate ~5 GB."""
    outs = [[], [], []]
    for s in range(0, len(b["x"]), chunk):
        sl = slice(s, s + chunk)
        part = {k: v[sl] for k, v in b.items()}
        lf, lq, lh = trainer.old_logprobs(part, frozen)
        outs[0].append(lf)
        outs[1].append(lq)
        outs[2].append(lh)
    return [np.concatenate(o) for o in outs]


class Run:
    def __init__(self, out_dir, base_npz=None, opponents=("pi0", "base"), replicates=2,
                 seeds_per_iter=32, workers=14, lr=3e-5, clip=0.1, epochs=4, minibatch=4096,
                 entropy=1e-3, kl_target=0.02, temperature=1.0, temp_final=0.5,
                 seed_base=TRAIN_SEED_BASE, steps=720, critic_hidden=256):
        self.out = Path(out_dir)
        (self.out / "ckpt").mkdir(parents=True, exist_ok=True)
        self.base_npz = str(base_npz or (WEIGHTS_DIR / "rebuild-m0.npz"))
        self.member = P.Member(self.base_npz)
        th = (self.member.ens_thresholds if self.member.ens_thresholds is not None
              else self.member.thresholds)
        from . import market_head as MH
        self.trainer = PPO.MarketPPO(self.member.mkt, self.member.mmu, self.member.msd,
                                     MH.logit(np.asarray(th, dtype=np.float64)),
                                     lr=lr, clip=clip, entropy_coef=entropy,
                                     kl_target=kl_target, temperature=temperature)
        self.critic = PPO.Critic(F.MARKET_TURN_DIM, hidden=critic_hidden)
        self.critic_ready = False
        self.opponents = list(opponents)
        self.replicates = replicates
        self.seeds_per_iter = seeds_per_iter
        self.workers = workers
        self.epochs = epochs
        self.minibatch = minibatch
        self.seed_base = seed_base
        self.steps = steps
        self.temp0, self.temp1 = temperature, temp_final
        self.log = []
        self.cursor = 0
        self.total_iters = 1
        self.actor_npz = str(self.out / "ckpt" / "actor.npz")
        save_member(self.base_npz, self.member.mkt, self.actor_npz)

    # -- one iteration -------------------------------------------------------
    def jobs(self, it):
        seeds = [self.seed_base + (self.cursor + i) for i in range(self.seeds_per_iter)]
        self.cursor += self.seeds_per_iter
        js = []
        for s in seeds:
            for seat in (0, 1):
                for opp in self.opponents:
                    for rep in range(self.replicates):
                        js.append({"seed": s, "seat": seat,
                                   "actor": [self.actor_npz],
                                   "opponent": opp, "opponent_id": opp,
                                   "rng_seed": (it * 1_000_003 + s * 97 + seat * 7 + rep),
                                   "temperature": self.temperature(it),
                                   "steps": self.steps})
        return js

    def temperature(self, it):
        if self.total_iters <= 1:
            return self.temp0
        f = min(1.0, it / max(1, self.total_iters - 1))
        return self.temp0 + f * (self.temp1 - self.temp0)

    def iteration(self, it):
        t0 = time.perf_counter()
        js = self.jobs(it)
        results = POOL.run(js, self.workers, kind="rollout")
        t_roll = time.perf_counter() - t0

        R = np.array([r["R"] for r in results], dtype=float)
        R_opp = np.array([r["R_opp"] for r in results], dtype=float)
        base, singles = loo_baseline(results)
        target_ep = (R - base) / REWARD_SCALE

        t1 = time.perf_counter()
        b, ep_of_row = stack_batch(results)
        self.trainer.temperature = self.temperature(it)
        if not self.critic_ready:
            self.critic.fit_norm(b["x"])
            self.critic_ready = True
        target_row = target_ep[ep_of_row]
        v = self.critic.value(b["x"])
        adv = target_row - v
        adv = (adv - adv.mean()) / max(1e-6, adv.std())

        frozen = PPO.clone_mlp(self.trainer.mkt)
        logp_old = chunked_old_logprobs(self.trainer, b, frozen)

        n = len(b["x"])
        rng = np.random.default_rng(1000 + it)
        agg = {}
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.minibatch):
                idx = order[s:s + self.minibatch]
                mb = slice_batch(b, idx)
                anchor_d = self.trainer.dists(self.trainer.anchor, mb["x"], mb["order_mask"])
                st = self.trainer.step(mb, adv[idx], [lp[idx] for lp in logp_old],
                                       anchor_d=anchor_d)
                for k, val in st.items():
                    agg.setdefault(k, []).append(val)
                self.critic.update(mb["x"], target_row[idx])
        t_update = time.perf_counter() - t1

        kl = float(np.mean(agg.get("kl", [0.0])))
        beta = self.trainer.adapt_beta(kl)
        save_member(self.base_npz, self.trainer.mkt, self.actor_npz)

        v2 = self.critic.value(b["x"])
        ev = 1.0 - float(np.var(target_row - v2) / max(1e-12, np.var(target_row)))
        row = {
            "iter": it, "episodes": len(results),
            "money_mean": float(R.mean()), "money_sd": float(R.std(ddof=1)),
            "opp_money_mean": float(R_opp.mean()),
            "extraction_mean": float((R + R_opp).mean()),
            "win_rate": float(np.mean((R > R_opp) + 0.5 * (R == R_opp))),
            "kl": kl, "beta": beta, "temperature": self.trainer.temperature,
            "clipfrac": float(np.mean(agg.get("clipfrac", [0.0]))),
            "surrogate": float(np.mean(agg.get("surrogate", [0.0]))),
            "entropy_orders": float(np.mean(agg.get("entropy_orders", [0.0]))),
            "entropy_hire": float(np.mean(agg.get("entropy_hire", [0.0]))),
            "explained_var": ev, "loo_singleton_cells": singles,
            "rows": int(n), "rollout_s": round(t_roll, 1), "update_s": round(t_update, 1),
            "update_frac": round(t_update / max(1e-6, t_roll + t_update), 3),
        }
        self.log.append(row)
        (self.out / "train-log.json").write_text(json.dumps(self.log, indent=1), encoding="utf-8")
        return row

    def run(self, iters, snapshot_every=5):
        self.total_iters = iters
        for it in range(iters):
            row = self.iteration(it)
            print(json.dumps(row), flush=True)
            if (it + 1) % snapshot_every == 0 or it == iters - 1:
                save_member(self.base_npz, self.trainer.mkt,
                            self.out / "ckpt" / f"iter{it + 1:03d}.npz")
        return self.log
