"""The rollout actor: `policy.BCPolicy` with a *sampled, recorded* market emission.

Only `_market` is overridden. Unit ops, the safety layer and the sanitiser come from the
shipped `policy.py` untouched, which is the whole point: PPO optimises the network's
pre-wrapper emission and treats `engine . wrapper` as the MDP (spec section 3.2).

What the recording is for
-------------------------
Every turn stores the market features, the order mask, the sampled factor indices and their
log-probs, and -- after the wrapper has run -- an `executed == emitted` flag per factor. The
last one is not bookkeeping: a factor the wrapper overrode did not influence the outcome, so
leaving it in the surrogate loss feeds it advantage x grad-log-pi on an action that never
happened, i.e. pure noise, aimed at exactly the heads the wrapper is protecting
(spec section 3.3).

At safety 3 the SELL bits are *structurally* overridden -- `_liquidate` replaces the whole
SELL block every turn -- so they are masked out unconditionally rather than by the diff,
which would otherwise score a coincidental agreement as "executed". Day 29 is the same story
for every order factor (`_endgame` rewrites the list).
"""
from __future__ import annotations

import numpy as np

from . import market_head as MH
from .bcpath import F, P


class RecordingPolicy(P.BCPolicy):
    """`greedy=True` reproduces `BCPolicy` decisions exactly (see market_head docstring)."""

    def __init__(self, members, safety=3, greedy=False, temperature=1.0, seed=0, record=True):
        super().__init__(members, safety=safety)
        if len(self.members) != 1:
            # Folding the calibrated threshold into a logit offset is exact for one member
            # (p == sigmoid(z)); for a mixture of sigmoids it is not, and the ratio would be
            # computed against a distribution we cannot differentiate cleanly. exp-011 runs
            # on member 0 alone, which spec section 8 explicitly allows.
            raise ValueError("RecordingPolicy is single-member; distil to pi0 first (exp-012)")
        m = self.members[0]
        th = m.ens_thresholds if m.ens_thresholds is not None else m.thresholds
        self.th_logit = MH.logit(np.asarray(th, dtype=np.float64))
        self.greedy = bool(greedy)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.record = bool(record)
        self.turns = []          # one dict per turn
        self._pending = None

    # -- the sampled emission ------------------------------------------------
    def _market(self, ctx):
        m = self.members[0]
        x = ctx.market_turn_features()
        mask = ctx.order_mask()
        d = MH.dists(m.mkt, m.mmu, m.msd, x[None, :], mask[None, :],
                     self.th_logit, self.temperature)
        p_fire = d["p_fire"][0]
        p_qty = d["p_qty"][0]
        p_hire = d["p_hire"][0]

        if self.greedy:
            fire = (p_fire >= 0.5).astype(np.int8)
            qty_pick = np.argmax(p_qty, axis=1).astype(np.int16)
            hire_idx = int(np.argmax(p_hire))
        else:
            fire = (self.rng.random(MH.N_ORDERS) < p_fire).astype(np.int8)
            u = self.rng.random((MH.N_ORDERS, 1))
            qty_pick = (p_qty.cumsum(1) < u).sum(1).clip(0, MH.N_QTY - 1).astype(np.int16)
            hire_idx = int((p_hire.cumsum() < self.rng.random()).sum().clip(0, MH.MAX_HIRES - 1))
        fire = np.where(mask > 0, fire, 0).astype(np.int8)

        slots = m.slot_order
        fired, emitted = [], {}
        qty_idx = np.full(MH.N_ORDERS, -1, dtype=np.int16)
        for oi in range(MH.N_ORDERS):
            if not fire[oi]:
                continue
            tname, item = F.ORDER_TYPES[oi]
            n = F.QTY_BUCKETS[int(qty_pick[oi])]
            if tname == "SELL":
                # BCPolicy's own emission-side clamp. When it swallows the order we record
                # the factor as not-fired; that misattributes the sampled bit, but SELL
                # factors are excluded from the loss at safety 3 anyway (see act()).
                n = min(n, int(ctx.shed.get(item, 0)))
                if n <= 0:
                    fire[oi] = 0
                    continue
            qty_idx[oi] = qty_pick[oi]
            order = ["BUY_LAND"] if tname == "BUY_LAND" else [tname, item, int(n)]
            emitted[(tname, item)] = int(n)
            fired.append((float(slots[oi]), order))

        hire_slot = float(slots[MH.N_ORDERS]) if len(slots) > MH.N_ORDERS else 99.0
        for i in range(hire_idx):
            fired.append((hire_slot + i * 1e-3, ["HIRE"]))
        fired.sort(key=lambda t: t[0])
        orders = [o for _, o in fired[:F.MAX_MARKET_ORDERS]]

        if self.record:
            lf, lq, lh = MH.logprobs(d, fire[None, :], qty_idx[None, :].astype(np.int32),
                                     np.array([hire_idx]))
            self._pending = {
                "day": int(ctx.day), "hour": int(ctx.hour),
                "x": np.asarray(x, dtype=np.float32),
                "order_mask": np.asarray(mask, dtype=np.int8),
                "fire": fire.copy(), "qty_idx": qty_idx.copy(), "hire_idx": np.int16(hire_idx),
                # float64: log(1-p) reaches ~-30, where float32 resolution is 2e-6 and the
                # spec's <=1e-6 recompute invariant could not be met by storage alone.
                "logp_fire": lf[0].astype(np.float64), "logp_qty": lq[0].astype(np.float64),
                "logp_hire": np.float64(lh[0]),
                "emitted": emitted, "emitted_hires": hire_idx,
            }
        return orders

    # -- executed-vs-emitted, after the wrapper has had its say ---------------
    def act(self, obs):
        action = super().act(obs)          # unchanged BCPolicy.act -> _market -> safety -> sanitise
        if not self.record or self._pending is None:
            return action
        rec = self._pending
        self._pending = None
        final, hires_final = {}, 0
        for o in action["market"]:
            if o[0] == "HIRE":
                hires_final += 1
            elif o[0] == "BUY_LAND":
                final[("BUY_LAND", None)] = 1
            else:
                final.setdefault((o[0], o[1]), int(o[2]))

        emitted, mask = rec.pop("emitted"), rec["order_mask"]
        ex_fire = np.zeros(MH.N_ORDERS, dtype=np.int8)
        ex_qty = np.zeros(MH.N_ORDERS, dtype=np.int8)
        endgame = self.safety >= 1 and rec["day"] >= F.DAYS - 1
        for oi in range(MH.N_ORDERS):
            if not mask[oi]:
                continue
            if endgame or (self.safety >= 3 and MH.SELL_ORDERS[oi]):
                continue                    # structurally rewritten by the wrapper
            tname, item = F.ORDER_TYPES[oi]
            key = (tname, item)
            present = key in final
            if bool(rec["fire"][oi]) == present:
                ex_fire[oi] = 1
            if rec["fire"][oi] and present and final[key] == emitted.get(key):
                ex_qty[oi] = 1
        rec["exec_fire"] = ex_fire
        rec["exec_qty"] = ex_qty
        rec["exec_hire"] = np.int8(0 if endgame else int(hires_final == rec["emitted_hires"]))
        rec.pop("emitted_hires")
        self.turns.append(rec)
        return action
