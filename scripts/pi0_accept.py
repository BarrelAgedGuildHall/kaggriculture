"""Offline acceptance gates for a distilled `π0` — addendum-01 §2.3 Gates C3, C4 and the
recommended margin-weighted agreement diagnostic.

The Pit half of the instrument (C1, C2, D, R) lives in `scripts/distil_screen.py`; Gate B
(the P&L census) is driven by `--census` here because it must run one BC agent per process
(see the note on that flag).

  C3  TRAIN/SERVE BIT-PARITY. The greedy argmax emitted by the SERVING path (`policy.Member`,
      one row at a time, exactly as `main.py` runs it) must equal the TRAINER's batched
      forward pass on a fixed 10,000-row sample. This is the gate that protects against the
      exp-003b bug class -- 0.906 offline accuracy and a 0% win rate, because the evaluated
      model and the served model were not the same program. It is a completely different
      measurement from agreement-with-the-teacher: a student can be a perfect clone and still
      fail C3, or a poor clone and pass it.

  C4  LEGALITY. `π0` may trigger no more safety-layer overrides per turn than the mixture
      does on the same boards (±5% relative). A student that leans harder on the wrapper has
      offloaded its policy onto the rule layer and will not survive PPO, which trains the
      pre-wrapper emission.

  Margin-weighted agreement (reported, never gated). Plain argmax agreement charges full
      price for disagreements where the mixture itself was near-tied -- routing ties among
      the four movement ops are worth exactly zero dollars. Weighting each row by the
      teacher's own confidence margin (p_top1 - p_top2) turns a misleading diagnostic into an
      interpretable one.

Standing measurement rule (addendum-01 §3.1): every offline number here is computed on the
FULL held-out split or a RANDOMLY sampled subset -- never a prefix slice.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_a" / "bc"))

import features as F                                   # noqa: E402
import policy as P                                     # noqa: E402
from net import MLP, masked_softmax, sigmoid           # noqa: E402

UNIT_HEADS = {"op": F.N_OPS, "crop": F.N_CROPS, "item": F.N_ITEMS, "qty": F.N_UQTY}
MKT_HEADS = {"orders": F.N_ORDERS, "qty": F.N_ORDERS * F.N_QTY, "hire": F.MAX_HIRES}


# ------------------------------------------------------------------- C3 -----
def gate_c3(student_npz, rows_npz, n=10000, seed=12345):
    """Serving path (batch 1, policy.Member) vs trainer path (batched MLP), argmax-exact."""
    d = np.load(rows_npz, allow_pickle=True)
    rng = np.random.default_rng(seed)
    ux, umask = d["val_unit_x"], d["val_unit_mask"].astype(np.float32)
    iu = rng.choice(len(ux), size=min(n, len(ux)), replace=False)
    mx = d["val_mkt_x"]
    im = rng.choice(len(mx), size=min(n, len(mx)), replace=False)

    member = P.Member(student_npz)
    z = np.load(student_npz, allow_pickle=True)
    unet = MLP.from_state({k[2:]: z[k] for k in z.files
                           if k.startswith("u_") and k not in ("u_mu", "u_sd")}, UNIT_HEADS)
    mnet = MLP.from_state({k[2:]: z[k] for k in z.files
                           if k.startswith("m_") and k not in ("m_mu", "m_sd",
                                                               "m_thresholds",
                                                               "m_ens_thresholds")}, MKT_HEADS)
    umu, usd, mmu, msd = z["u_mu"], z["u_sd"], z["m_mu"], z["m_sd"]

    # trainer path, batched
    tu = unet.forward((ux[iu] - umu) / usd)
    t_op = masked_softmax(tu["op"], umask[iu]).argmax(1)
    t_arg = {k: masked_softmax(tu[k]).argmax(1) for k in ("crop", "item", "qty")}
    tm = mnet.forward((mx[im] - mmu) / msd)
    t_fire = sigmoid(tm["orders"])
    t_mq = masked_softmax(tm["qty"].reshape(-1, F.N_QTY)).reshape(len(im), F.N_ORDERS,
                                                                  F.N_QTY).argmax(2)
    t_hire = masked_softmax(tm["hire"]).argmax(1)

    # serving path, one row at a time, exactly as main.py calls it
    s_op = np.empty(len(iu), np.int64)
    s_arg = {k: np.empty(len(iu), np.int64) for k in ("crop", "item", "qty")}
    for j, i in enumerate(iu):
        po, pc, pi_, pq = member.unit_probs(ux[i], umask[i])
        s_op[j] = int(po.argmax())
        s_arg["crop"][j], s_arg["item"][j], s_arg["qty"][j] = (int(pc.argmax()),
                                                              int(pi_.argmax()),
                                                              int(pq.argmax()))
    s_fire = np.empty((len(im), F.N_ORDERS), np.float64)
    s_mq = np.empty((len(im), F.N_ORDERS), np.int64)
    s_hire = np.empty(len(im), np.int64)
    for j, i in enumerate(im):
        po, pq, ph = member.mkt_probs(mx[i])
        s_fire[j], s_mq[j], s_hire[j] = po, pq.argmax(1), int(ph.argmax())

    th = member.ens_thresholds if member.ens_thresholds is not None else member.thresholds
    res = {"n_unit": int(len(iu)), "n_mkt": int(len(im)), "seed": seed,
           "unit_op": float((t_op == s_op).mean())}
    for k in ("crop", "item", "qty"):
        res[f"unit_{k}"] = float((t_arg[k] == s_arg[k]).mean())
    # the order head's emitted decision is the threshold crossing, so parity is on the BIT
    res["mkt_orders_fired"] = float(((t_fire >= th) == (s_fire >= th)).mean())
    res["mkt_qty"] = float((t_mq == s_mq).mean())
    res["mkt_hire"] = float((t_hire == s_hire).mean())
    res["max_abs_prob_diff_orders"] = float(np.abs(t_fire - s_fire).max())
    heads = [k for k in res if k.startswith(("unit_", "mkt_"))]
    res["pass"] = all(res[k] == 1.0 for k in heads)
    res["worst_head"] = min((res[k], k) for k in heads)[1]
    return res


# ------------------------------------------------------------------- C4 -----
class CountingPolicy(P.BCPolicy):
    """Counts how often the safety layer rewrote what the net emitted, per turn.

    Works for any member count (unlike ppo/actor.RecordingPolicy, which is single-member by
    construction), because it only diffs the wrapper's input against its output.
    """

    def __init__(self, members, safety=3):
        super().__init__(members, safety=safety)
        self.turns = 0
        self.op_overrides = 0
        self.order_overrides = 0
        # per-stage, and BUY vs SELL: at safety 3 `_liquidate` REPLACES the learned SELL head
        # wholesale BY DESIGN, for every policy alike, so a positional diff there measures how
        # much inventory the farm happens to be carrying -- not whether the student leans on
        # the wrapper. The interpretable legality signals are op rescues and BUY-side edits.
        self.by_stage = {"endgame": 0, "keep_alive": 0, "liquidate": 0}
        self.buy_overrides = 0
        self.sell_overrides = 0
        self._stage = None

    def _endgame(self, ctx, ops, orders):
        self._stage = "endgame"
        o2, r2 = super()._endgame(ctx, ops, orders)
        self._diff(ops, orders, o2, r2)
        return o2, r2

    def _keep_alive(self, ctx, ops, orders):
        self._stage = "keep_alive"
        o2, r2 = super()._keep_alive(ctx, ops, orders)
        self._diff(ops, orders, o2, r2)
        return o2, r2

    def _liquidate(self, ctx, orders):
        self._stage = "liquidate"
        r2 = super()._liquidate(ctx, orders)
        self._diff(None, orders, None, r2)
        return r2

    @staticmethod
    def _multiset(orders):
        return sorted(tuple(o) for o in (orders or []))

    def _diff(self, ops, orders, o2, r2):
        if ops is not None and o2 is not None:
            n = sum(1 for a, b in zip(ops, o2) if a != b) + abs(len(o2) - len(ops))
            self.op_overrides += n
        if orders is not None and r2 is not None:
            # multiset symmetric difference: order-of-emission is a slot ordering, not a
            # decision, so a reordering is not an override
            a, b = self._multiset(orders), self._multiset(r2)
            from collections import Counter
            ca, cb = Counter(a), Counter(b)
            n = sum((ca - cb).values()) + sum((cb - ca).values())
            self.order_overrides += n
            self.by_stage[self._stage] = self.by_stage.get(self._stage, 0) + n
            for o, k in list((ca - cb).items()) + list((cb - ca).items()):
                if o and str(o[0]).startswith("SELL"):
                    self.sell_overrides += k
                else:
                    self.buy_overrides += k

    def act(self, obs):
        self.turns += 1
        return super().act(obs)


def gate_c4(student_npz, member_npzs, seeds, steps=720):
    """Override rate per turn for pi0 and for the mixture, same seeds, same opponent (v0)."""
    import kaggle_environments as ke
    sys.path.insert(0, str(REPO_ROOT / "agents" / "team_a"))
    import v0 as V0                                              # noqa: N812

    def run(members, seed):
        pol = CountingPolicy(members, safety=3)
        env = ke.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed},
                      debug=False)
        env.run([lambda obs, cfg=None: pol.act(obs), V0.agent])
        return {"turns": pol.turns,
                "ops": pol.op_overrides, "orders": pol.order_overrides,
                "by_stage": dict(pol.by_stage),
                "buy": pol.buy_overrides, "sell": pol.sell_overrides,
                "per_turn": (pol.op_overrides + pol.order_overrides) / max(1, pol.turns),
                "buy_per_turn": (pol.op_overrides + pol.buy_overrides) / max(1, pol.turns)}

    out = {"seeds": list(seeds), "student": [], "mixture": []}
    for s in seeds:
        out["student"].append(run([student_npz], s))
        out["mixture"].append(run(list(member_npzs), s))
    a = statistics.fmean(r["per_turn"] for r in out["student"])
    b = statistics.fmean(r["per_turn"] for r in out["mixture"])
    ab = statistics.fmean(r["buy_per_turn"] for r in out["student"])
    bb = statistics.fmean(r["buy_per_turn"] for r in out["mixture"])
    out["student_per_turn"] = round(a, 4)
    out["mixture_per_turn"] = round(b, 4)
    out["relative"] = round((a - b) / b, 4) if b else None
    # the interpretable statistic: op rescues + BUY-side edits, excluding the by-design
    # wholesale SELL replacement that safety 3 applies identically to every policy
    out["student_buy_per_turn"] = round(ab, 4)
    out["mixture_buy_per_turn"] = round(bb, 4)
    out["relative_buy"] = round((ab - bb) / bb, 4) if bb else None
    out["pass_all_orders"] = bool(b == 0 and a == 0) or (b > 0 and (a - b) / b <= 0.05)
    out["pass"] = bool(bb == 0 and ab == 0) or (bb > 0 and (ab - bb) / bb <= 0.05)
    return out


# ------------------------------------------- margin-weighted agreement ------
def margin_agreement(student_npz, member_npzs, rows_npz, batch=8192):
    """Agreement weighted by the teacher's own confidence margin p_top1 - p_top2, on the FULL
    held-out split. Near-ties (routing ties among the movement ops) get near-zero weight."""
    d = np.load(rows_npz, allow_pickle=True)
    ux, umask = d["val_unit_x"], d["val_unit_mask"].astype(np.float32)
    members = [P.Member(p) for p in member_npzs]
    st = P.Member(student_npz)
    n = len(ux)
    agree = np.empty(n, bool)
    margin = np.empty(n, np.float64)
    for i in range(0, n, batch):
        xb, mb = ux[i:i + batch], umask[i:i + batch]
        acc = None
        for m in members:
            o = m.unit.forward((xb - m.umu) / m.usd)
            p = masked_softmax(o["op"], mb)
            acc = p if acc is None else acc + p
        t = acc / len(members)
        o = st.unit.forward((xb - st.umu) / st.usd)
        s = masked_softmax(o["op"], mb)
        agree[i:i + batch] = t.argmax(1) == s.argmax(1)
        part = np.sort(t, axis=1)
        margin[i:i + batch] = part[:, -1] - part[:, -2]
    w = margin / max(1e-12, margin.sum())
    return {"n": int(n), "unweighted": round(float(agree.mean()), 4),
            "margin_weighted": round(float((agree * w).sum()), 4),
            "mean_margin": round(float(margin.mean()), 4),
            "disagree_mean_margin": round(float(margin[~agree].mean()), 4) if (~agree).any() else None,
            "agree_mean_margin": round(float(margin[agree].mean()), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--rows", required=True, help="a rebuild chunk (carries the frozen val split)")
    ap.add_argument("--c4-seeds", default="1000-1003")
    ap.add_argument("--skip-c4", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    lo, hi = (args.c4_seeds.split("-") + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]

    out = {"student": args.student}
    out["C3_train_serve_parity"] = gate_c3(args.student, args.rows)
    out["margin_agreement_op"] = margin_agreement(args.student, args.members, args.rows)
    if not args.skip_c4:
        out["C4_legality"] = gate_c4(args.student, args.members, seeds)
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    ok = out["C3_train_serve_parity"]["pass"] and out.get("C4_legality", {"pass": True})["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
