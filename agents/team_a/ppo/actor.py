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
from . import unit_head as UH
from .bcpath import F, P


class RecordingPolicy(P.BCPolicy):
    """`greedy=True` reproduces `BCPolicy` decisions exactly (see market_head docstring).

    Phase P1 (`record_units=True`) adds the unit path: the op head and its conditional
    argument heads are sampled and recorded the same way the market factors are, with the
    same executed-vs-emitted discipline. Unit rows outnumber market rows by ~7x, so they are
    subsampled at `unit_subsample` (spec section 2.4: the advantage is shared across a turn's
    factors, so uniformly dropping factors is an unbiased, rescaled gradient estimator).
    The subsampling draw uses its own rng, so turning it up or down does not change a single
    action the policy takes on a given seed.
    """

    def __init__(self, members, safety=3, greedy=False, temperature=1.0, seed=0, record=True,
                 record_units=False, unit_subsample=1.0, unit_temperature=None):
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
        # The unit head gets its own temperature. The market order head measured 0.040
        # nats/factor at T = 1 (exp-011), which is why T = 2 was needed there to make PPO's
        # ratio mean anything; the 18-way op head is nowhere near that peaked, and inflating
        # it to T = 2 makes the sampled actor a different, far worse agent (measured, see the
        # exp-013 temperature calibration).
        self.unit_temperature = float(temperature if unit_temperature is None
                                      else unit_temperature)
        self.rng = np.random.default_rng(seed)
        self.record = bool(record)
        self.record_units = bool(record_units)
        self.unit_subsample = float(unit_subsample)
        self.sub_rng = np.random.default_rng(seed + 991)
        self.turns = []          # one dict per turn (market factors)
        self.unit_rows = []      # one dict per recorded unit decision
        self._pending = None
        self._pending_units = []

    # -- the sampled unit emission (phase P1) ---------------------------------
    def _sample(self, p):
        """One draw from a 1-D categorical, greedy-or-sampled, sharing the action rng."""
        if self.greedy:
            return int(np.argmax(p))
        return int(min((p.cumsum() < self.rng.random()).sum(), len(p) - 1))

    def _unit_op(self, ctx, u, claimed):
        if not self.record_units:
            return super()._unit_op(ctx, u, claimed)
        m = self.members[0]
        x = ctx.unit_features(u, claimed)
        op_mask = ctx.unit_mask(u)
        crop_mask = ctx.crop_mask()
        d = UH.dists(m.unit, m.umu, m.usd, x[None, :], op_mask[None, :],
                     crop_mask[None, :], np.ones((1, F.N_ITEMS), np.float32),
                     self.unit_temperature)
        oi = self._sample(d["p_op"][0])
        name = F.OPS[oi]
        crop_idx = item_idx = qty_idx = -1
        item_mask = np.ones(F.N_ITEMS, np.float32)

        if name == "PLANT":
            crop_idx = self._sample(d["p_crop"][0])
            op = ["PLANT", F.CROPS[crop_idx]]
        elif name in ("PICKUP", "PLACE"):
            item_mask = (ctx.item_mask_pickup() if name == "PICKUP"
                         else ctx.item_mask_place(u))
            # re-form the item distribution over the op's own legal set
            d["p_item"] = UH._masked_softmax(
                np.where(item_mask[None, :] > 0, d["z_item"], UH.NEG), None,
                self.unit_temperature)
            d["z_item"] = np.where(item_mask[None, :] > 0, d["z_item"], UH.NEG)
            item_idx = self._sample(d["p_item"][0])
            item = F.ITEMS[item_idx]
            if name == "PICKUP":
                qty_idx = self._sample(d["p_qty"][0])
                n = F.QTY_BUCKETS[qty_idx]
                op = ["PICKUP", item, int(max(1, min(n, ctx.shed.get(item, 0))))]
            else:
                op = ["PLACE", item]
        else:
            op = [name]

        if self.record:
            lo, lc, li, lq = UH.logprobs(d, np.array([oi]), np.array([crop_idx]),
                                         np.array([item_idx]), np.array([qty_idx]))
            self._pending_units.append({
                "u": int(u), "day": int(ctx.day), "hour": int(ctx.hour),
                "x": np.asarray(x, dtype=np.float32),
                "op_mask": np.asarray(op_mask, dtype=np.int8),
                "crop_mask": np.asarray(crop_mask, dtype=np.int8),
                "item_mask": np.asarray(item_mask, dtype=np.int8),
                "op_idx": np.int16(oi), "crop_idx": np.int16(crop_idx),
                "item_idx": np.int16(item_idx), "qty_idx": np.int16(qty_idx),
                "logp_op": np.float64(lo[0]), "logp_crop": np.float64(lc[0]),
                "logp_item": np.float64(li[0]), "logp_qty": np.float64(lq[0]),
                "emitted": list(op),
            })
        return op, name not in ("PASS", "NORTH", "SOUTH", "EAST", "WEST")

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

    def _finish_units(self, action):
        """executed == emitted, per unit factor, after the wrapper and the sanitiser."""
        pend, self._pending_units = self._pending_units, []
        if not pend:
            return
        final = [action["farmer"]] + list(action["hands"])
        keep = self.unit_subsample
        for rec in pend:
            u = rec.pop("u")
            emitted = rec.pop("emitted")
            got = final[u] if u < len(final) else ["PASS"]
            same_op = bool(got and got[0] == emitted[0])
            rec["exec_op"] = np.int8(same_op)
            arg_ok = same_op and len(got) > 1 and len(emitted) > 1 and got[1] == emitted[1]
            rec["exec_crop"] = np.int8(arg_ok and emitted[0] == "PLANT")
            rec["exec_item"] = np.int8(arg_ok and emitted[0] in ("PICKUP", "PLACE"))
            rec["exec_qty"] = np.int8(arg_ok and emitted[0] == "PICKUP"
                                      and len(got) > 2 and got[2] == emitted[2])
            if keep >= 1.0 or self.sub_rng.random() < keep:
                self.unit_rows.append(rec)

    # -- executed-vs-emitted, after the wrapper has had its say ---------------
    def act(self, obs):
        action = super().act(obs)          # unchanged BCPolicy.act -> _market -> safety -> sanitise
        if self.record and self.record_units:
            self._finish_units(action)
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
