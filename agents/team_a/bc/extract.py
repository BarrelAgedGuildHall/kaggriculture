"""Replay -> behaviour-cloning training rows for one teacher.

One row per *decision*, never per turn: a turn produces one row per acting unit (farmer +
each hired hand) plus one row per market slot including the terminating END. Rows come only
from the teacher's own seat of episodes the teacher WON, which is the Pokemon challenge's
"winning seat of an exact-configuration replay" rule.

Split is by episode id, never by row, so no seat of one episode can straddle train/val.

Usage (from the repo root):
    uv run python agents/team_a/bc/extract.py --teacher tetsuya
    uv run python agents/team_a/bc/extract.py --teacher tetsuya --max-episodes 8 --out /tmp/x.npz
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

MANIFEST = Path("D:/GuildhallData/kaggler/datasets/kaggriculture/replays/team-a/team_a_teachers.csv")
OUT_DIR = Path("D:/GuildhallData/kaggler/datasets/kaggriculture/bc-rows")
ENGINE = "1.32.7"


def val_fold(episode_id: int, folds: int = 5) -> int:
    """Stable per-episode fold from a hash (not id % k -- ids are time-ordered)."""
    h = hashlib.sha256(str(episode_id).encode()).hexdigest()
    return int(h[:8], 16) % folds


def _norm_op(op):
    """Recorded op -> (op_index, crop_index or -1, item_index or -1). None if unmappable."""
    if not op:
        return None
    name = op[0]
    if name not in F.OP_IX:
        return None
    oi = F.OP_IX[name]
    ci = ii = -1
    if name == "PLANT":
        ci = F.CROP_IX.get(op[1] if len(op) > 1 else None, -1)
        if ci < 0:
            return None
    elif name in ("PICKUP", "PLACE"):
        ii = F.ITEM_IX.get(op[1] if len(op) > 1 else None, -1)
        if ii < 0:
            return None
    return oi, ci, ii


def _norm_market(order):
    if not order:
        return None
    name = order[0]
    if name not in F.MKT_IX:
        return None
    item, n = None, 1
    if name in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
        if len(order) < 2:
            return None
        item = order[1]
        if item not in F.ITEM_IX:
            return None
        n = int(order[2]) if len(order) > 2 else 1
    return name, item, max(1, n)


def extract_episode(path: Path, seat: int, stats: collections.Counter):
    """-> dict of row arrays for one demonstration (one seat of one episode)."""
    replay = json.load(open(path, encoding="utf-8"))
    if replay.get("module_version") != ENGINE:
        stats["skipped_engine"] += 1
        return None
    ux, uop, ucrop, uitem, umask = [], [], [], [], []
    mx, mtype, mitem, mqty, mmask = [], [], [], [], []

    steps = replay["steps"]
    # OFF-BY-ONE, verified 2026-08-19: a kaggle-environments replay stores at index t the
    # action that *produced* state t, not the action chosen from state t. Pairing
    # obs[t] with action[t] makes every HARVEST land on an already-harvested tile and
    # every PLANT on an occupied one (22.3% of ops look like silent no-ops, and HARVEST
    # never once succeeds). Pairing obs[t] with action[t+1] gives 1.6% no-ops, 403/403
    # valid harvests, and a hands-list length that always matches the farm. Do not "fix"
    # this back.
    for t in range(len(steps) - 1):
        entry = steps[t][seat]
        obs = entry.get("observation")
        act = steps[t + 1][seat].get("action")
        if not isinstance(obs, dict) or not isinstance(act, dict):
            continue
        if entry.get("status") not in (None, "ACTIVE", "DONE"):
            continue
        try:
            ctx = F.TurnContext(obs)
        except Exception:
            stats["ctx_error"] += 1
            continue

        ops = [act.get("farmer")] + list(act.get("hands") or [])
        claimed = set()
        for u in range(min(ctx.n_units, len(ops))):
            lab = _norm_op(ops[u])
            if lab is None:
                stats["unit_unmapped"] += 1
                continue
            oi, ci, ii = lab
            mask = ctx.unit_mask(u)
            if mask[oi] == 0.0:
                # The teacher emitted an op that is a silent no-op in this state -- e.g.
                # HARVEST on bare ground, walking into a wall, FEED with no wheat carried.
                # Verified by inspection: every violated category is a genuine no-op, not a
                # gap in the mask. Top ladder agents broadcast one op to all of their units
                # each turn, so ~20% of their emitted ops do nothing. We clone the
                # *effective* action (identical state transition), which is PASS.
                stats["noop_relabelled"] += 1
                stats["noop_" + F.OPS[oi]] += 1
                oi, ci, ii = F.OP_IX["PASS"], -1, -1
            ux.append(ctx.unit_features(u, claimed))
            uop.append(oi)
            ucrop.append(ci)
            uitem.append(ii)
            umask.append(mask)
            if F.OPS[oi] not in ("PASS", "NORTH", "SOUTH", "EAST", "WEST"):
                claimed.add(tuple(ctx.unit_pos[u]))
            stats["unit_rows"] += 1

        # One row per TURN: which concrete orders did the teacher place, in what quantity,
        # and how many hands did it hire. (The autoregressive slot form is kept in
        # features.py but is not trained -- see the note on ORDER_TYPES.)
        orders = [o for o in (act.get("market") or []) if o][:F.MAX_MARKET_ORDERS]
        multi = np.zeros(F.N_ORDERS, dtype=np.uint8)
        qty = np.full(F.N_ORDERS, -1, dtype=np.int16)
        slot_of = {}
        hires = 0
        for slot, raw in enumerate(orders):
            got = _norm_market(raw)
            if got is None:
                stats["market_unmapped"] += 1
                continue
            tname, item, n = got
            if tname == "HIRE":
                hires += 1
                stats["slotsum_hire"] += slot
                stats["slotn_hire"] += 1
                continue
            key = (tname, item) if tname != "BUY_LAND" else ("BUY_LAND", None)
            oi = F.ORDER_IX.get(key)
            if oi is None:
                stats["order_unmapped_" + tname] += 1
                continue
            multi[oi] = 1
            qty[oi] = F.qty_bucket(n)
            slot_of.setdefault(oi, slot)
        mx.append(ctx.market_turn_features())
        mtype.append(multi)
        mitem.append(qty)
        mqty.append(min(hires, F.MAX_HIRES - 1))
        mmask.append(ctx.order_mask())
        for oi, slot in slot_of.items():
            stats["slotsum_%d" % oi] += slot
            stats["slotn_%d" % oi] += 1
        stats["market_rows"] += 1

    if not ux:
        return None
    return {
        "unit_x": np.asarray(ux, dtype=np.float32),
        "unit_op": np.asarray(uop, dtype=np.int16),
        "unit_crop": np.asarray(ucrop, dtype=np.int16),
        "unit_item": np.asarray(uitem, dtype=np.int16),
        "unit_mask": np.asarray(umask, dtype=np.uint8),
        "mkt_x": np.asarray(mx, dtype=np.float32),
        "mkt_multi": np.asarray(mtype, dtype=np.uint8),      # (n, N_ORDERS) 0/1
        "mkt_qty": np.asarray(mitem, dtype=np.int16),        # (n, N_ORDERS) bucket or -1
        "mkt_hire": np.asarray(mqty, dtype=np.int16),        # (n,) hires this turn
        "mkt_mask": np.asarray(mmask, dtype=np.uint8),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--out", default="")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--val-fold", type=int, default=0, help="which of 5 episode folds is validation")
    p.add_argument("--min-opp-rating", type=float, default=0.0,
                   help="drop episodes whose opponent was below this ladder rating")
    args = p.parse_args(argv)

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8"))
            if r["teacher"] == args.teacher]
    if args.min_opp_rating:
        rows = [r for r in rows if float(r["opp_rating_before"]) >= args.min_opp_rating]
    if args.max_episodes:
        rows = rows[: args.max_episodes]
    if not rows:
        raise SystemExit(f"no manifest rows for teacher {args.teacher!r}")

    stats = collections.Counter()
    parts = {"train": [], "val": []}
    episodes = {"train": [], "val": []}
    t0 = time.perf_counter()
    for i, r in enumerate(rows, 1):
        eid = int(r["episode_id"])
        split = "val" if val_fold(eid) == args.val_fold else "train"
        got = extract_episode(Path(r["path"]), int(r["seat"]), stats)
        if got is None:
            continue
        parts[split].append(got)
        episodes[split].append(eid)
        print(f"  [{i}/{len(rows)}] {eid} seat{r['seat']} -> {split} "
              f"({len(got['unit_op'])} unit rows, {time.perf_counter()-t0:.0f}s)", flush=True)

    if not parts["train"]:
        raise SystemExit("no training rows extracted")
    out = {}
    for split, chunks in parts.items():
        if not chunks:
            continue
        for k in chunks[0]:
            out[f"{split}_{k}"] = np.concatenate([c[k] for c in chunks])
    # Mean slot position the teacher used for each order type -- the emission order is a
    # real decision (each order spends what the earlier ones left), so the policy replays
    # the teacher's ordering instead of inventing one.
    slot_order = []
    for oi in range(F.N_ORDERS):
        n = stats.get("slotn_%d" % oi, 0)
        slot_order.append(stats.get("slotsum_%d" % oi, 0) / n if n else 99.0)
    nh = stats.get("slotn_hire", 0)
    hire_slot = stats.get("slotsum_hire", 0) / nh if nh else 99.0
    clean = {k: v for k, v in stats.items() if not k.startswith(("slotsum_", "slotn_"))}
    out["meta"] = np.asarray([json.dumps({
        "teacher": args.teacher, "engine": ENGINE, "unit_dim": F.UNIT_DIM,
        "market_turn_dim": F.MARKET_TURN_DIM, "val_fold": args.val_fold,
        "episodes_train": episodes["train"], "episodes_val": episodes["val"],
        "min_opp_rating": args.min_opp_rating,
        "slot_order": slot_order, "hire_slot": hire_slot,
        "stats": clean,
    })])

    path = Path(args.out) if args.out else OUT_DIR / f"{args.teacher}-fold{args.val_fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"\nunit rows  train={len(out.get('train_unit_op', []))} val={len(out.get('val_unit_op', []))}")
    print(f"market rows train={len(out.get('train_mkt_type', []))} val={len(out.get('val_mkt_type', []))}")
    print("stats:", dict(stats))
    print(f"wrote {path} ({path.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
