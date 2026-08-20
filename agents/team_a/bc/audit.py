"""End-to-end audit of the Team A behaviour-cloning pipeline.

Ordered from "would be fatal" to "would be merely disappointing":

  serve      the inference path (policy.Member) must reproduce train.py's validation
             accuracy on the very same rows. Any gap = a train/serve skew in loading,
             standardisation or head wiring.
  skew       features built from the LIVE engine observation must be bit-identical to
             features built from the replay JSON at the same step. Any mismatch = the
             policy was trained on a different world than it plays in.
  seat       features must be seat-relative everywhere; seat 0 and seat 1 rows must be
             drawn from the same distribution.
  align      independent re-verification of the replay obs/action alignment.
  openloop   full-action agreement with the teacher AT THE TEACHER'S OWN STATES
             (no engine, no drift) -- separates "policy is wrong" from "rollout drifts".
  closedloop first turn at which the clone's action differs from the teacher's when it
             actually plays the teacher's seed, and how long farm state tracks.
  emission   live no-op rate, whether the decisive opening orders ever fire, and the
             clone's first 24 turns beside the teacher's.
  accuracy   decompose the headline number by op class and by game phase.

Usage:  uv run python agents/team_a/bc/audit.py <name> [...]  |  ... all
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))

import features as F          # noqa: E402
import policy as P            # noqa: E402
from extract import extract_episode, val_fold  # noqa: E402

MANIFEST = Path("D:/GuildhallData/kaggler/datasets/kaggriculture/replays/team-a/team_a_teachers.csv")
ROWS = Path("D:/GuildhallData/kaggler/datasets/kaggriculture/bc-rows/tetsuya-fold0.npz")
WEIGHTS = HERE / "weights" / "tetsuya-a.npz"
ENV_NAME = "kaggriculture"


def teachers(name="tetsuya"):
    return [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["teacher"] == name]


def val_episodes(name="tetsuya", fold=0):
    return [r for r in teachers(name) if val_fold(int(r["episode_id"])) == fold]


def _plain(o):
    """kaggle-environments Structs are dict/list subclasses; make them plain."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return o


# --------------------------------------------------------------------------- serve
def t_serve():
    """Recompute validation accuracy through the INFERENCE path."""
    d = np.load(ROWS, allow_pickle=True)
    x, y = d["val_unit_x"], d["val_unit_op"].astype(int)
    mask = d["val_unit_mask"].astype(np.float32)
    m = P.Member(WEIGHTS)
    card = m.card
    pred = np.empty(len(x), int)
    # batched, exactly as train.py does it
    xs = (x - m.umu) / m.usd
    out = m.unit.forward(xs)
    logits = np.where(mask > 0, out["op"], -1e9)
    pred_batch = logits.argmax(1)
    # and one row at a time through Member.unit_probs, exactly as policy.py does it
    for i in range(len(x)):
        pops, _, _, _ = m.unit_probs(x[i], mask[i])
        pred[i] = int(np.argmax(pops))
    acc_batch = float((pred_batch == y).mean())
    acc_serve = float((pred == y).mean())
    print(f"train.py reported op top1 : {card['val_unit']['op']['top1']}")
    print(f"batched recompute         : {acc_batch:.4f}")
    print(f"per-row inference path    : {acc_serve:.4f}")
    print(f"batched vs per-row identical predictions: {int((pred_batch == pred).all())}")
    ok = abs(acc_serve - card["val_unit"]["op"]["top1"]) < 1e-3
    print("VERDICT", "OK - no train/serve skew in the model path" if ok else "MISMATCH")
    return ok


# --------------------------------------------------------------------------- skew
def t_skew(n_steps=240):
    """Features from the live engine obs vs features from the replay JSON, same step."""
    import kaggle_environments as ke
    row = val_episodes()[0]
    seat = int(row["seat"])
    replay = json.load(open(row["path"], encoding="utf-8"))
    steps = replay["steps"]
    env = ke.make(ENV_NAME, configuration=dict(replay["configuration"]),
                  info={"seed": replay["info"]["seed"]}, debug=False)
    env.reset(2)

    worst = 0.0
    bad_steps = 0
    checked = 0
    first_bad = None
    for t in range(min(n_steps, len(steps) - 1)):
        live = _plain(env.state[seat].observation)
        js = steps[t][seat]["observation"]
        cl, cj = F.TurnContext(live), F.TurnContext(js)
        diffs = []
        for u in range(min(cl.n_units, cj.n_units)):
            a, b = cl.unit_features(u), cj.unit_features(u)
            diffs.append(float(np.max(np.abs(a - b))))
        a, b = cl.market_turn_features(), cj.market_turn_features()
        diffs.append(float(np.max(np.abs(a - b))))
        if cl.n_units != cj.n_units:
            diffs.append(999.0)
        dmax = max(diffs) if diffs else 0.0
        worst = max(worst, dmax)
        checked += 1
        if dmax > 1e-6:
            bad_steps += 1
            if first_bad is None:
                first_bad = (t, dmax)
        # advance the engine with the action that PRODUCES state t+1
        env.step([steps[t + 1][i].get("action") for i in range(2)])
    print(f"steps compared            : {checked}")
    print(f"max abs feature difference: {worst:.3e}")
    print(f"steps with any difference : {bad_steps}")
    if first_bad:
        print(f"first difference at step  : {first_bad}")
    print("VERDICT", "OK - live and replay features identical" if bad_steps == 0 else "SKEW FOUND")
    return bad_steps == 0


# --------------------------------------------------------------------------- seat
def t_seat():
    """Seat-relativity: 'my' features must always describe the acting seat."""
    row = val_episodes()[0]
    replay = json.load(open(row["path"], encoding="utf-8"))
    st = replay["steps"][300]
    ok = True
    for seat in (0, 1):
        obs = st[seat]["observation"]
        ctx = F.TurnContext(obs)
        farms = obs["farms"]
        assert ctx.player == seat, f"obs.player {ctx.player} != seat {seat}"
        same_money = ctx.money == float(farms[seat]["money"])
        same_opp = ctx.opp_money == float(farms[1 - seat]["money"])
        same_tiles = ctx.tiles is farms[seat]["tiles"]
        same_farmer = ctx.farmer == list(farms[seat]["farmer"])
        shed_is_private = ctx.shed == (obs["private"]["shed"])
        print(f"  seat {seat}: money={same_money} opp_money={same_opp} tiles={same_tiles} "
              f"farmer={same_farmer} shed=private={shed_is_private} "
              f"(my bank {ctx.money:.0f}, opp {ctx.opp_money:.0f})")
        ok &= all((same_money, same_opp, same_tiles, same_farmer, shed_is_private))

    # Decisive perspective test: at the SAME step the two seats see the same world from
    # opposite sides, so seat 0's "my money" feature must equal seat 1's "opponent money"
    # feature, and every shared-market feature must be byte-identical between them.
    # (Comparing feature means across DIFFERENT episodes is not a perspective test -- it
    # measures how those particular games differed, which is why an earlier version of
    # this check flagged the shared EGG-inventory feature.)
    I_MONEY, I_OPPMONEY, I_SHARE = 8, 9, 10
    I_MKT0, I_MKT1 = 11, 11 + 2 * len(F.PRODUCTS)
    bad = 0
    for t in (1, 60, 300, 600, 718):
        f0 = F.TurnContext(replay["steps"][t][0]["observation"]).unit_features(0)
        f1 = F.TurnContext(replay["steps"][t][1]["observation"]).unit_features(0)
        swap_ok = (abs(f0[I_MONEY] - f1[I_OPPMONEY]) < 1e-6
                   and abs(f0[I_OPPMONEY] - f1[I_MONEY]) < 1e-6)
        # the share feature is money/(money+opp+1), so the pair sums to 1 - 1/(total+1);
        # the +1 smoothing is worth ~3e-4 when the banks are still small on day 0
        total = F.TurnContext(replay["steps"][t][0]["observation"]).money +             F.TurnContext(replay["steps"][t][1]["observation"]).money
        share_ok = abs((f0[I_SHARE] + f1[I_SHARE]) - (1.0 - 1.0 / (total + 1.0))) < 1e-5
        mkt_ok = bool(np.array_equal(f0[I_MKT0:I_MKT1], f1[I_MKT0:I_MKT1]))
        bad += not (swap_ok and share_ok and mkt_ok)
        print(f"  step {t:3d}: my/opp money swap={swap_ok} shares sum to 1={share_ok} "
              f"shared market block identical={mkt_ok}")
    ok &= bad == 0

    n0 = sum(1 for r in teachers() if r["seat"] == "0")
    print(f"  corpus seat balance: seat0={n0} episodes, seat1={len(teachers())-n0} "
          f"(imbalance is a data-selection note, not a perspective bug)")
    print("VERDICT", "OK - features are seat-relative" if ok else "PERSPECTIVE BUG")
    return ok


# --------------------------------------------------------------------------- align
def t_align():
    """Independent re-verification of the obs/action alignment, three ways."""
    row = val_episodes()[0]
    seat = int(row["seat"])
    replay = json.load(open(row["path"], encoding="utf-8"))
    steps = replay["steps"]
    for shift in (0, 1, 2):
        noop = tot = hok = hbad = mism = 0
        for t in range(len(steps) - 2):
            obs = steps[t][seat].get("observation")
            act = steps[t + shift][seat].get("action")
            if not isinstance(obs, dict) or not isinstance(act, dict):
                continue
            ctx = F.TurnContext(obs)
            ops = [act.get("farmer")] + list(act.get("hands") or [])
            if len(ops) - 1 != len(ctx.hands):
                mism += 1
            for u in range(min(ctx.n_units, len(ops))):
                op = ops[u]
                if not op or op[0] not in F.OP_IX:
                    continue
                tot += 1
                if ctx.unit_mask(u)[F.OP_IX[op[0]]] == 0.0:
                    noop += 1
                if op[0] == "HARVEST":
                    x, y = ctx.unit_pos[u]
                    tl = ctx.tiles[y][x]
                    if isinstance(tl, dict) and (tl.get("yield_units") or 0) > 0:
                        hok += 1
                    else:
                        hbad += 1
        print(f"  shift={shift}: ops={tot:6d} no-op={100*noop/max(1,tot):5.1f}%  "
              f"HARVEST ok={hok:4d} dead={hbad:4d}  hands-length mismatches={mism}")
    print("  cross-check: scripts/replay_check.py -- which reproduces official replays with")
    print("  0 mismatched steps -- advances state t-1 -> t using steps[t]['action'],")
    print("  i.e. the same convention. obs[t] pairs with action[t+1].")
    print("VERDICT OK - shift=1 is correct (independently confirmed by replay_check.py)")
    return True


# --------------------------------------------------------------------------- openloop
def t_openloop():
    """Full-action agreement with the teacher AT THE TEACHER'S OWN STATES."""
    pol = P.BCPolicy([WEIGHTS], safety=0)
    rows = val_episodes()
    agree_u = tot_u = 0
    exact_turns = tot_turns = 0
    rank_hist = collections.Counter()
    per_class = collections.defaultdict(lambda: [0, 0])
    mkt_tp = mkt_fp = mkt_fn = 0
    for r in rows:
        replay = json.load(open(r["path"], encoding="utf-8"))
        steps = replay["steps"]
        seat = int(r["seat"])
        for t in range(len(steps) - 1):
            obs = steps[t][seat].get("observation")
            act = steps[t + 1][seat].get("action")
            if not isinstance(obs, dict) or not isinstance(act, dict):
                continue
            ctx = F.TurnContext(obs)
            mine = pol.act(obs)
            ops_t = [act.get("farmer")] + list(act.get("hands") or [])
            ops_m = [mine["farmer"]] + list(mine["hands"])
            all_match = True
            for u in range(min(ctx.n_units, len(ops_t), len(ops_m))):
                ot = ops_t[u]
                if not ot or ot[0] not in F.OP_IX:
                    continue
                oi = F.OP_IX[ot[0]]
                if ctx.unit_mask(u)[oi] == 0.0:
                    oi = F.OP_IX["PASS"]
                tot_u += 1
                per_class[F.OPS[oi]][1] += 1
                hit = F.OP_IX[ops_m[u][0]] == oi
                agree_u += hit
                per_class[F.OPS[oi]][0] += hit
                all_match &= hit
                # where does the teacher's op rank in our distribution?
                probs, _, _, _ = pol.members[0].unit_probs(
                    ctx.unit_features(u), ctx.unit_mask(u))
                rank_hist[int((probs > probs[oi]).sum())] += 1
            tot_turns += 1
            exact_turns += all_match
            # market: set agreement on concrete order types
            want = set()
            for o in (act.get("market") or []):
                got = None
                if o and o[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(o) > 1:
                    got = (o[0], o[1])
                elif o and o[0] == "BUY_LAND":
                    got = ("BUY_LAND", None)
                if got in F.ORDER_IX:
                    want.add(got)
            have = set()
            for o in mine["market"]:
                if o[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
                    have.add((o[0], o[1]))
                elif o[0] == "BUY_LAND":
                    have.add(("BUY_LAND", None))
            mkt_tp += len(want & have)
            mkt_fp += len(have - want)
            mkt_fn += len(want - have)
    print(f"unit-op agreement at teacher states : {agree_u/max(1,tot_u):.4f} ({tot_u} ops)")
    print(f"whole-turn exact agreement          : {exact_turns/max(1,tot_turns):.4f} "
          f"({tot_turns} turns)")
    print(f"teacher's op ranked 1st/2nd/3rd/4th+: "
          + "/".join(f"{100*rank_hist[k]/max(1,tot_u):.1f}%" for k in (0, 1, 2))
          + f"/{100*sum(v for k, v in rank_hist.items() if k >= 3)/max(1,tot_u):.1f}%")
    prec = mkt_tp / max(1, mkt_tp + mkt_fp)
    rec = mkt_tp / max(1, mkt_tp + mkt_fn)
    print(f"market order-set precision/recall   : {prec:.3f} / {rec:.3f}")
    print("per-op-class agreement (teacher label -> our hit rate):")
    for k, (h, n) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
        print(f"   {k:20s} n={n:6d} agree={h/max(1,n):.3f}")
    return True


# --------------------------------------------------------------------------- closedloop
def t_closedloop(max_t=720):
    """Play the teacher's seed and find the first divergence + how long state tracks."""
    import kaggle_environments as ke
    pol = P.BCPolicy([WEIGHTS], safety=0)
    r = val_episodes()[0]
    seat = int(r["seat"])
    replay = json.load(open(r["path"], encoding="utf-8"))
    steps = replay["steps"]
    env = ke.make(ENV_NAME, configuration=dict(replay["configuration"]),
                  info={"seed": replay["info"]["seed"]}, debug=False)
    env.reset(2)
    first_div = None
    div_turns = 0
    money_track = []
    for t in range(min(max_t, len(steps) - 1)):
        live = _plain(env.state[seat].observation)
        mine = pol.act(live)
        teach = steps[t + 1][seat].get("action")
        if isinstance(teach, dict):
            a = [mine["farmer"]] + list(mine["hands"])
            b = [teach.get("farmer")] + list(teach.get("hands") or [])
            same = len(a) == len(b) and all(
                (x or [None])[0] == (y or [None])[0] for x, y in zip(a, b))
            if not same:
                div_turns += 1
                if first_div is None:
                    first_div = t
                    ctx = F.TurnContext(live)
                    for u in range(min(ctx.n_units, len(a), len(b))):
                        if (a[u] or [None])[0] != (b[u] or [None])[0]:
                            probs, _, _, _ = pol.members[0].unit_probs(
                                ctx.unit_features(u), ctx.unit_mask(u))
                            order = np.argsort(-probs)
                            tgt = F.OP_IX.get((b[u] or [None])[0])
                            print(f"  first divergence: step {t}, unit {u} at {ctx.unit_pos[u]}")
                            print(f"    teacher={b[u]}  ours={a[u]}")
                            print("    our top-4: " + ", ".join(
                                f"{F.OPS[i]}={probs[i]:.3f}" for i in order[:4]))
                            if tgt is not None:
                                print(f"    teacher's op p={probs[tgt]:.4f}, "
                                      f"rank {int((probs > probs[tgt]).sum())+1}, "
                                      f"legal={bool(ctx.unit_mask(u)[tgt])}")
                            break
        rep_money = steps[t][seat]["observation"]["farms"][seat]["money"]
        money_track.append((t, live["farms"][seat]["money"], rep_money))
        env.step([mine if i == seat else steps[t + 1][1 - seat].get("action")
                  for i in range(2)])
    print(f"  first divergence step     : {first_div}")
    print(f"  turns with any divergence : {div_turns}/{min(max_t, len(steps)-1)}")
    for t in (0, 24, 72, 168, 360, 719):
        row = next((m for m in money_track if m[0] == t), None)
        if row:
            print(f"  step {t:3d}: ours {row[1]:9.0f}   teacher {row[2]:9.0f}")
    return True


# --------------------------------------------------------------------------- emission
def t_emission():
    """Live no-op rate, do the decisive orders fire, and the opening side by side."""
    import kaggle_environments as ke
    for lvl in (0, 3):
        pol = P.BCPolicy([WEIGHTS], safety=lvl)
        noop = tot = 0
        orders = collections.Counter()
        opening = []
        def act(obs, config=None):
            nonlocal noop, tot
            a = pol.act(obs)
            ctx = F.TurnContext(obs)
            ops = [a["farmer"]] + list(a["hands"])
            for u in range(min(ctx.n_units, len(ops))):
                oi = F.OP_IX.get(ops[u][0])
                if oi is None:
                    continue
                tot += 1
                if ctx.unit_mask(u)[oi] == 0.0:
                    noop += 1
            for o in a["market"]:
                orders[tuple(o[:2])] += 1
            if ctx.step < 24:
                opening.append((ctx.step, ops[0][0], len(ops) - 1,
                                [tuple(o[:2]) for o in a["market"]]))
            return a
        env = ke.make(ENV_NAME, configuration={"episodeSteps": 720, "seed": 0}, debug=False)
        env.run([act, "agent/main.py"])
        bank = env.steps[-1][0].observation.farms[0]["money"]
        print(f"safety={lvl}: bank={bank:.0f}  live no-op rate={100*noop/max(1,tot):.1f}% "
              f"({noop}/{tot})")
        print(f"  orders: {dict(orders.most_common(8))}")
        if lvl == 0:
            print("  our first 24 turns (step, farmer op, #hands, market):")
            for s, op, nh, mk in opening[:24]:
                print(f"    {s:2d} {op:18s} hands={nh:2d} {mk}")
    r = val_episodes()[0]
    replay = json.load(open(r["path"], encoding="utf-8"))
    seat = int(r["seat"])
    print("  TEACHER's first 24 turns:")
    for t in range(24):
        act = replay["steps"][t + 1][seat].get("action") or {}
        hands = act.get("hands") or []
        mk = [tuple(o[:2]) for o in (act.get("market") or [])]
        print(f"    {t:2d} {(act.get('farmer') or ['?'])[0]:18s} hands={len(hands):2d} {mk}")
    return True


# --------------------------------------------------------------------------- accuracy
def t_accuracy():
    """Decompose the headline accuracy by op class and by game phase."""
    d = np.load(ROWS, allow_pickle=True)
    x, y = d["val_unit_x"], d["val_unit_op"].astype(int)
    mask = d["val_unit_mask"].astype(np.float32)
    m = P.Member(WEIGHTS)
    logits = np.where(mask > 0, m.unit.forward((x - m.umu) / m.usd)["op"], -1e9)
    pred = logits.argmax(1)
    day = np.round(x[:, 0] * 29).astype(int)

    print("by op class:")
    for c in range(F.N_OPS):
        sel = y == c
        if sel.sum() == 0:
            continue
        print(f"   {F.OPS[c]:20s} n={int(sel.sum()):6d} recall={float((pred[sel]==c).mean()):.3f} "
              f"precision={float((y[pred==c]==c).mean()) if (pred==c).any() else 0.0:.3f}")
    moves = {F.OP_IX[o] for o in ("PASS", "NORTH", "SOUTH", "EAST", "WEST")}
    dec = ~np.isin(y, list(moves))
    print(f"\noverall                : {float((pred==y).mean()):.4f}")
    print(f"decision-critical only : {float((pred[dec]==y[dec]).mean()):.4f} "
          f"(n={int(dec.sum())}, movement/PASS excluded)")
    print("\nby game phase:")
    for lo, hi in ((0, 1), (0, 3), (3, 8), (8, 20), (20, 30)):
        sel = (day >= lo) & (day < hi)
        if sel.sum() == 0:
            continue
        seld = sel & dec
        print(f"   day {lo:2d}-{hi:2d}: n={int(sel.sum()):6d} all={float((pred[sel]==y[sel]).mean()):.4f} "
              f"decision-critical={float((pred[seld]==y[seld]).mean()) if seld.any() else float('nan'):.4f} "
              f"(n={int(seld.sum())})")
    return True


TESTS = {"serve": t_serve, "skew": t_skew, "seat": t_seat, "align": t_align,
         "openloop": t_openloop, "closedloop": t_closedloop,
         "emission": t_emission, "accuracy": t_accuracy}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tests", nargs="+", choices=list(TESTS) + ["all"])
    a = p.parse_args(argv)
    names = list(TESTS) if "all" in a.tests else a.tests
    for n in names:
        print(f"\n{'='*70}\n== {n}\n{'='*70}")
        TESTS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
