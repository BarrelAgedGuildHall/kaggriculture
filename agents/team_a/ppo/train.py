"""The PPO training loop. Phase P0 (`exp-011`) and phase P1 (`exp-013`).

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

Phase P1 adds three things, each of which is a *binding* requirement of the exp-013
pre-registration rather than a convenience:

1. **The unit heads train** alongside the market trunk (`UnitPPO`). The unit trunk stays
   frozen -- that is P2/`exp-014`.
2. **The KL anchor and the BC-auxiliary loss reference the FROZEN 3-MEMBER MIXTURE, never
   `pi0`** (spec addendum 01 section 2.1). `pi0` is a distillation of the mixture with a
   measured -1.59% money handicap; anchoring to it would nail the run to that handicap.
3. **PFSP opponent sampling** with even-matchup priority `w ~ x(1-x) + eps` over the
   self-play body, and hard floors on the anchors (spec section 4.1/4.2).

A unit row's advantage is the advantage of *its own turn*: unit and market rows are joined on
`(episode, day, hour)`, so the critic baseline a unit factor sees is the same state value its
turn's market factors see. Joining on the episode mean instead would silently give every unit
row a different (and wrong) baseline from the market row it shares a state with.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import harness as H
from . import market_head as MH
from . import pool as POOL
from . import ppo as PPO
from . import unit_head as UH
from .bcpath import F, P, WEIGHTS_DIR

REWARD_SCALE = 50_000.0
TRAIN_SEED_BASE = 5000
EPS0 = 0.05                    # PFSP floor on the even-matchup weight


def save_member(src_npz, mkt, out_path, unit=None):
    """Write a new actor: the source weights with the trained tensors replaced."""
    d = dict(np.load(src_npz, allow_pickle=True))
    for k, v in mkt.state().items():
        d["m_" + k] = np.asarray(v, dtype=np.float32)
    if unit is not None:
        for k, v in unit.state().items():
            d["u_" + k] = np.asarray(v, dtype=np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)
    return str(out_path)


def stack_batch(results):
    """Flatten recorded market turns into column arrays + a per-row episode index."""
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
    turn_key = [(e, int(t["day"]), int(t["hour"])) for e, t in zip(ep_of_row, turns)]
    return b, np.asarray(ep_of_row), {k: i for i, k in enumerate(turn_key)}


def stack_units(results, turn_ix):
    """Flatten recorded unit decisions; `row_of_turn` maps each unit row to its market row."""
    rows, ep_of_row, mkt_row = [], [], []
    for i, r in enumerate(results):
        for u in r.get("unit_rows", ()):
            j = turn_ix.get((i, int(u["day"]), int(u["hour"])))
            if j is None:          # a unit decision with no market row: cannot be baselined
                continue
            rows.append(u)
            ep_of_row.append(i)
            mkt_row.append(j)
    if not rows:
        return None, None, None
    b = {
        "x": np.stack([r["x"] for r in rows]),
        "op_mask": np.stack([r["op_mask"] for r in rows]).astype(np.float32),
        "crop_mask": np.stack([r["crop_mask"] for r in rows]).astype(np.float32),
        "item_mask": np.stack([r["item_mask"] for r in rows]).astype(np.float32),
        "op_idx": np.array([int(r["op_idx"]) for r in rows], dtype=np.int32),
        "crop_idx": np.array([int(r["crop_idx"]) for r in rows], dtype=np.int32),
        "item_idx": np.array([int(r["item_idx"]) for r in rows], dtype=np.int32),
        "qty_idx": np.array([int(r["qty_idx"]) for r in rows], dtype=np.int32),
        "logp_op": np.array([float(r["logp_op"]) for r in rows]),
        "logp_crop": np.array([float(r["logp_crop"]) for r in rows]),
        "logp_item": np.array([float(r["logp_item"]) for r in rows]),
        "logp_qty": np.array([float(r["logp_qty"]) for r in rows]),
        "exec_op": np.array([int(r["exec_op"]) for r in rows]),
        "exec_crop": np.array([int(r["exec_crop"]) for r in rows]),
        "exec_item": np.array([int(r["exec_item"]) for r in rows]),
        "exec_qty": np.array([int(r["exec_qty"]) for r in rows]),
    }
    return b, np.asarray(ep_of_row), np.asarray(mkt_row)


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
    outs = None
    for s in range(0, len(b["x"]), chunk):
        sl = slice(s, s + chunk)
        part = {k: v[sl] for k, v in b.items()}
        lps = trainer.old_logprobs(part, frozen)
        if outs is None:
            outs = [[] for _ in lps]
        for o, lp in zip(outs, lps):
            o.append(lp)
    return [np.concatenate(o) for o in outs]


class AuxCorpus:
    """A fixed pool of corpus states with the FROZEN MIXTURE's own distributions as targets.

    The targets are the mixture's probabilities, not the corpus's hard labels: exp-012
    measured that those are different targets (the hard label is what the teacher episodes
    did; the mixture's distribution is what our anchor believes), and addendum 01 section 2.1
    binds the auxiliary loss to the mixture.
    """

    def __init__(self, row_paths, members, n_unit=48_000, n_mkt=24_000, seed=17):
        # One chunk at a time, subsampled on the way in. The union of the three rebuild
        # chunks is ~3.2 M unit rows; materialising it whole costs ~2 GB for a pool we then
        # throw 98% of away, and this trainer shares the box with the Pit.
        rng = np.random.default_rng(seed)
        per = max(1, n_unit // max(1, len(row_paths)))
        per_m = max(1, n_mkt // max(1, len(row_paths)))
        ux, um, mx = [], [], []
        for p in row_paths:
            d = np.load(str(p), allow_pickle=True)
            a, b_, c = d["train_unit_x"], d["train_unit_mask"], d["train_mkt_x"]
            i = rng.choice(len(a), size=min(per, len(a)), replace=False)
            j = rng.choice(len(c), size=min(per_m, len(c)), replace=False)
            ux.append(np.asarray(a[i], dtype=np.float32))
            um.append(np.asarray(b_[i], dtype=np.float32))
            mx.append(np.asarray(c[j], dtype=np.float32))
            del a, b_, c, d
        tr_u, tr_um, tr_m = np.concatenate(ux), np.concatenate(um), np.concatenate(mx)
        iu = np.arange(len(tr_u))
        im = np.arange(len(tr_m))
        self.unit = {"x": tr_u[iu], "op_mask": tr_um[iu]}
        self.unit["teacher"] = UH.mixture_dists(members, self.unit["x"],
                                                self.unit["op_mask"], None, None)
        self.mkt = {"x": tr_m[im],
                    "order_mask": np.ones((len(im), F.N_ORDERS), np.float32)}
        th = members[0].ens_thresholds
        th = th if th is not None else members[0].thresholds
        self.mkt["teacher"] = MH.mixture_dists(members, self.mkt["x"],
                                               self.mkt["order_mask"],
                                               MH.logit(np.asarray(th, dtype=np.float64)))
        self.rng = np.random.default_rng(seed + 1)

    def _draw(self, d, n):
        i = self.rng.choice(len(d["x"]), size=min(n, len(d["x"])), replace=False)
        out = {k: v[i] for k, v in d.items() if k != "teacher"}
        out["teacher"] = {k: v[i] for k, v in d["teacher"].items()}
        return out

    def unit_batch(self, n=2048):
        return self._draw(self.unit, n)

    def mkt_batch(self, n=1024):
        return self._draw(self.mkt, n)


class Run:
    def __init__(self, out_dir, base_npz=None, opponents=("pi0", "base"), replicates=2,
                 seeds_per_iter=32, workers=14, lr=3e-5, clip=0.1, epochs=4, minibatch=4096,
                 entropy=1e-3, kl_target=0.02, temperature=1.0, temp_final=0.5,
                 seed_base=TRAIN_SEED_BASE, steps=720, critic_hidden=256,
                 phase="P0", unit_subsample=0.25, alpha_aux=0.0, aux_rows=(),
                 pfsp=False, snapshot_every=5, max_ckpt=10, unit_temperature=1.0):
        self.out = Path(out_dir)
        (self.out / "ckpt").mkdir(parents=True, exist_ok=True)
        self.base_npz = str(base_npz or (WEIGHTS_DIR / "rebuild-m0.npz"))
        self.member = P.Member(self.base_npz)
        self.phase = phase
        th = (self.member.ens_thresholds if self.member.ens_thresholds is not None
              else self.member.thresholds)
        self.th_logit = MH.logit(np.asarray(th, dtype=np.float64))
        self.trainer = PPO.MarketPPO(self.member.mkt, self.member.mmu, self.member.msd,
                                     self.th_logit, lr=lr, clip=clip, entropy_coef=entropy,
                                     kl_target=kl_target, temperature=temperature,
                                     train_trunk=(phase != "P0"))
        self.unit_trainer = None
        # The unit head keeps its OWN, FIXED temperature. Measured (scripts/ppo_temp_calib.py,
        # 16 ep/arm vs the frozen base): op-head entropy at T = 1 is already 0.60 nats -- 15x
        # the market order head's 0.040 -- and raising it costs money fast (bank 28.1k at
        # T = 1.0, 32.8k at 1.3, 16.4k at 1.6, 23.4k at 2.0 against a ~68k opponent). So
        # T_unit = 1.0, flat, identical across every exp-013 arm; only the market head anneals.
        self.unit_temperature = float(unit_temperature)
        if phase != "P0":
            self.unit_trainer = PPO.UnitPPO(self.member.unit, self.member.umu, self.member.usd,
                                            lr=lr, clip=clip, entropy_coef=entropy,
                                            temperature=self.unit_temperature)
        self.critic = PPO.Critic(F.MARKET_TURN_DIM, hidden=critic_hidden)
        self.critic_ready = False

        # the frozen 3-member mixture: KL anchor + BC-auxiliary reference (addendum 01 2.1)
        self.mixture = [P.Member(p) for p in H.BASE_MEMBERS]
        self.aux = None
        self.alpha_aux = float(alpha_aux)
        if self.alpha_aux and aux_rows:
            self.aux = AuxCorpus(aux_rows, self.mixture)

        self.opponents = list(opponents)
        self.pfsp = bool(pfsp)
        self.win_est = {o: 0.5 for o in self.opponents}     # decayed win rate per opponent
        self.win_seen = {}
        self.ckpts = []
        self.max_ckpt = max_ckpt
        self.snapshot_every = snapshot_every
        self.replicates = replicates
        self.seeds_per_iter = seeds_per_iter
        self.workers = workers
        self.epochs = epochs
        self.minibatch = minibatch
        self.seed_base = seed_base
        self.steps = steps
        self.unit_subsample = float(unit_subsample)
        self.kl_target = kl_target
        self.kl0 = None                 # KL(pi0 || mixture) measured at iteration 0
        self.kl_target_unit = kl_target
        self.temp0, self.temp1 = temperature, temp_final
        self.log = []
        self.cursor = 0
        self.total_iters = 1
        self.rng = np.random.default_rng(4242)
        self.actor_npz = str(self.out / "ckpt" / "actor.npz")
        self.save_actor(self.actor_npz)

    def save_actor(self, path):
        return save_member(self.base_npz, self.trainer.mkt, path,
                           unit=(self.unit_trainer.unit if self.unit_trainer else None))

    # -- opponent sampling ----------------------------------------------------
    FLOORS = {"base": 0.15, "pi0": 0.05}

    def draw_opponents(self, k=None):
        """PFSP with even-matchup priority (spec 4.2). Fixed anchors always present.

        The draw is per ITERATION, not per episode, and it is without replacement: the
        leave-one-out baseline's cell is `(seed, seat, opponent)` and needs `--replicates`
        episodes inside each cell, so an opponent drawn per-episode would shatter every cell
        into singletons and hand the whole batch a zero advantage. Sampling *which* opponents
        an iteration faces, and playing the full seed x seat x replicate block against each,
        is the same PFSP distribution at the granularity the baseline can support.
        """
        names = list(self.opponents) + [c[0] for c in self.ckpts]
        # One slot is reserved for the self-play body whenever a checkpoint exists. Without
        # it the fixed floors fill every slot and the PFSP pool is decorative: the run is
        # then plain two-opponent training with a snapshot directory nobody plays.
        k = k or (len(self.opponents) + (1 if self.ckpts else 0))
        if not self.pfsp or len(names) <= k:
            return names
        fixed = [nm for nm in names if nm in self.FLOORS]
        rest = [nm for nm in names if nm not in self.FLOORS]
        pick = list(fixed)[:k]
        room = k - len(pick)
        if room > 0 and rest:
            w = np.array([self.win_est.get(nm, 0.5) * (1 - self.win_est.get(nm, 0.5)) + EPS0
                          for nm in rest], dtype=float)
            w /= w.sum()
            pick += list(self.rng.choice(rest, size=min(room, len(rest)),
                                         replace=False, p=w))
        return pick

    def spec_of(self, name):
        if name in ("pi0", "base"):
            return name
        for nm, path in self.ckpts:
            if nm == name:
                return [path]
        return name

    # -- one iteration -------------------------------------------------------
    def jobs(self, it):
        seeds = [self.seed_base + (self.cursor + i) for i in range(self.seeds_per_iter)]
        self.cursor += self.seeds_per_iter
        opps = sorted(set(self.draw_opponents())) or list(self.opponents)
        js = []
        for s in seeds:
            for seat in (0, 1):
                for opp in opps:
                    for rep in range(self.replicates):
                        js.append({"seed": s, "seat": seat,
                                   "actor": [self.actor_npz],
                                   "opponent": self.spec_of(opp), "opponent_id": opp,
                                   "rng_seed": (it * 1_000_003 + s * 97 + seat * 7 + rep),
                                   "temperature": self.temperature(it),
                                   "unit_temperature": self.unit_temperature,
                                   "record_units": self.phase != "P0",
                                   "unit_subsample": self.unit_subsample,
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

        # decayed per-opponent win rate (half-life 10 iterations), for PFSP
        for nm in set(r["opponent_id"] for r in results):
            sel = [i for i, r in enumerate(results) if r["opponent_id"] == nm]
            x = float(np.mean((R[sel] > R_opp[sel]) + 0.5 * (R[sel] == R_opp[sel])))
            # half-life 10 iterations; a name's first reading seeds the estimate outright
            if nm in self.win_est and self.win_seen.get(nm):
                self.win_est[nm] = 0.933 * self.win_est[nm] + 0.067 * x
            else:
                self.win_est[nm] = x
            self.win_seen[nm] = True

        t1 = time.perf_counter()
        b, ep_of_row, turn_ix = stack_batch(results)
        ub, u_ep, u_mkt_row = (None, None, None)
        if self.phase != "P0":
            ub, u_ep, u_mkt_row = stack_units(results, turn_ix)
        self.trainer.temperature = self.temperature(it)
        if not self.critic_ready:
            self.critic.fit_norm(b["x"])
            self.critic_ready = True
        target_row = target_ep[ep_of_row]
        v = self.critic.value(b["x"])
        adv = target_row - v
        adv_mu, adv_sd = adv.mean(), max(1e-6, adv.std())
        adv = (adv - adv_mu) / adv_sd
        u_adv = adv[u_mkt_row] if ub is not None else None

        frozen = PPO.clone_mlp(self.trainer.mkt)
        logp_old = chunked_old_logprobs(self.trainer, b, frozen)
        u_logp_old = None
        if ub is not None:
            u_frozen = PPO.clone_mlp(self.unit_trainer.unit)
            u_logp_old = chunked_old_logprobs(self.unit_trainer, ub, u_frozen)

        n = len(b["x"])
        rng = np.random.default_rng(1000 + it)
        agg, uagg = {}, {}
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.minibatch):
                idx = order[s:s + self.minibatch]
                mb = slice_batch(b, idx)
                anchor_d = MH.mixture_dists(self.mixture, mb["x"], mb["order_mask"],
                                            self.th_logit)
                st = self.trainer.step(mb, adv[idx], [lp[idx] for lp in logp_old],
                                       anchor_d=anchor_d,
                                       aux=(self.aux.mkt_batch() if self.aux else None),
                                       alpha=self.alpha_aux)
                for k, val in st.items():
                    agg.setdefault(k, []).append(val)
                self.critic.update(mb["x"], target_row[idx])
            if ub is not None:
                un = len(ub["x"])
                uorder = rng.permutation(un)
                for s in range(0, un, self.minibatch):
                    idx = uorder[s:s + self.minibatch]
                    mb = slice_batch(ub, idx)
                    anchor_d = UH.mixture_dists(self.mixture, mb["x"], mb["op_mask"],
                                                mb["crop_mask"], mb["item_mask"])
                    st = self.unit_trainer.step(
                        mb, u_adv[idx], [lp[idx] for lp in u_logp_old], anchor_d=anchor_d,
                        aux=(self.aux.unit_batch() if self.aux else None),
                        alpha=self.alpha_aux)
                    for k, val in st.items():
                        uagg.setdefault(k, []).append(val)
        t_update = time.perf_counter() - t1

        kl = float(np.mean(agg.get("kl", [0.0])))
        u_kl = float(np.mean(uagg.get("kl", [0.0]))) if uagg else 0.0
        # kappa is a DRIFT budget, and pi0 does not start at the anchor: measured at iteration
        # 0, KL(pi0 || mixture) is ~0.07 nats/factor on the market head and ~0.54 on the unit
        # heads (pi0 is a distillation of the mixture, not the mixture). Comparing those to a
        # flat kappa = 0.02 would peg beta at its 64 ceiling from the first update and turn the
        # run into "re-distil pi0 into the mixture" with the surrogate as a rounding error.
        # The target is therefore KL_0 + kappa: do not drift more than kappa beyond where this
        # policy started. Fixed after iteration 0 and logged, so it is a pre-registered number
        # and not a knob that moves with the result.
        if self.kl0 is None:
            self.kl0 = {"mkt": kl, "unit": u_kl}
            self.trainer.kl_target = kl + self.kl_target
            self.kl_target_unit = u_kl + self.kl_target
        beta = self.trainer.adapt_beta(kl)
        u_beta = (self.unit_trainer.adapt_beta(u_kl, self.kl_target_unit)
                  if self.unit_trainer else 0.0)
        self.save_actor(self.actor_npz)

        v2 = self.critic.value(b["x"])
        ev = 1.0 - float(np.var(target_row - v2) / max(1e-12, np.var(target_row)))
        row = {
            "iter": it, "episodes": len(results), "phase": self.phase,
            "money_mean": float(R.mean()), "money_sd": float(R.std(ddof=1)),
            "opp_money_mean": float(R_opp.mean()),
            "extraction_mean": float((R + R_opp).mean()),
            "win_rate": float(np.mean((R > R_opp) + 0.5 * (R == R_opp))),
            "win_by_opp": {nm: round(self.win_est[nm], 3) for nm in sorted(self.win_est)},
            "kl": kl, "beta": beta, "kl_unit": u_kl, "beta_unit": u_beta,
            "kl0": self.kl0, "kl_target_mkt": self.trainer.kl_target,
            "kl_target_unit": self.kl_target_unit,
            "unit_temperature": self.unit_temperature,
            "temperature": self.trainer.temperature,
            "clipfrac": float(np.mean(agg.get("clipfrac", [0.0]))),
            "clipfrac_unit": float(np.mean(uagg.get("clipfrac", [0.0]))) if uagg else 0.0,
            "surrogate": float(np.mean(agg.get("surrogate", [0.0]))),
            "entropy_orders": float(np.mean(agg.get("entropy_orders", [0.0]))),
            "entropy_hire": float(np.mean(agg.get("entropy_hire", [0.0]))),
            "entropy_op": float(np.mean(uagg.get("entropy_op", [0.0]))) if uagg else 0.0,
            "aux_mkt": float(np.mean(agg.get("aux_mkt", [0.0]))),
            "aux_unit": float(np.mean(uagg.get("aux_unit", [0.0]))) if uagg else 0.0,
            "explained_var": ev, "loo_singleton_cells": singles,
            "rows": int(n), "unit_rows": int(len(ub["x"])) if ub is not None else 0,
            "rollout_s": round(t_roll, 1), "update_s": round(t_update, 1),
            "update_frac": round(t_update / max(1e-6, t_roll + t_update), 3),
        }
        self.log.append(row)
        (self.out / "train-log.json").write_text(json.dumps(self.log, indent=1), encoding="utf-8")
        return row

    def snapshot(self, it):
        path = self.out / "ckpt" / f"iter{it + 1:03d}.npz"
        self.save_actor(path)
        if self.pfsp:
            nm = f"ck{it + 1:03d}"
            self.ckpts.append((nm, str(path)))
            self.win_est.setdefault(nm, 0.5)
            while len(self.ckpts) > self.max_ckpt:
                dead, _ = self.ckpts.pop(0)
                self.win_est.pop(dead, None)
        return str(path)

    def run(self, iters, snapshot_every=None):
        self.total_iters = iters
        every = snapshot_every or self.snapshot_every
        for it in range(iters):
            row = self.iteration(it)
            print(json.dumps(row), flush=True)
            if (it + 1) % every == 0 or it == iters - 1:
                self.snapshot(it)
        return self.log
