"""S-2 — supervised label pipeline: per-turn opponent flow and pool-headroom labels.

Extracts, for every (episode, transition, product) cell, the two label families the
balance-sheet program's `exp-014` auxiliary heads need:

  (a) **opponent net-sales flow** — units the rival actually pushed into (or pulled
      out of) each shared market pool on that turn;
  (b) **pool headroom** — how many more units that pool can still absorb above the
      $1 floor between now and the horizon, given the live price curve and the town
      drain that is still to come.

Why a re-step and not accounting off the recorded observations
--------------------------------------------------------------
The engine applies unit actions (PICKUP / DROP / harvest deposits) to a player's shed
*before* `_process_market` runs in the same step, so shed deltas between two recorded
observations do not isolate market flow. The only exact attribution is to re-step the
replay through the pinned engine with `_commit_unit` instrumented — each successful
commit is one unit, with its player, item and executed price. Replay parity for engine
1.32.7 is already established (`scripts/replay_check.py`), and this script re-proves the
market slice of it per episode: local market inventory is diffed against the recorded
inventory at every step, and any episode that does not reproduce is dropped.

The market inventory identity
-----------------------------
Only two code paths in `kaggriculture.py` write `market["inventory"]`: `_commit_unit`
(SELL +1, BUY_PRODUCT -1) and `_town_consume` (shop and town-centre draws). Therefore,
for the transition out of step `t`:

    inv[t+1] - inv[t] == sum_players (sold_above_floor - bought) - drain(t)

which is exact, with one caveat that is the whole point of the floor bookkeeping: a SELL
executed at the $1 floor pays the seller but **does not add market supply**
(`_commit_unit`: "Sales at $1 do not increase market supply"). So the flow that is
*visible in inventory* is `net_visible = sold_above_floor - bought`, while the flow that
actually happened is `net_true = sold_total - bought`. In the floor regime the two
diverge and no observer can recover the difference from inventory alone. Both are
emitted; `net_true` is the honest supervised target, `net_visible` is the ceiling on
what a runtime estimator could ever infer.

Town drain is deterministic and needs no estimation here: it is a pure function of
`obs.town.unlocked_shops` and the step counter (`_town_consume`, shop interval 4,
town-centre interval 24, single-product shops consume at 2x).

Outputs (numpy `.npz`, one row per (episode, transition), 2-D per product)
-------------------------------------------------------------------------
`.npz` rather than parquet on purpose: the repo is numpy-only (no pyarrow in the lock
file), Team A's BC/PPO pipeline already loads `.npz`, and adding an Arrow dependency to
ship a label table is not worth it. The layout is column-oriented and memory-mappable
enough for the sizes involved (tens of MB).

Usage
-----
    uv run python scripts/extract_s2_labels.py \
        --replays "$GUILDHALL_WAREHOUSE/datasets/kaggriculture/replays/1.32.7/*.json" \
        --sample 24 \
        --out "$GUILDHALL_WAREHOUSE/datasets/s2-labels" \
        --tag ladder-1.32.7

Idempotent: an episode already present in the output shard is skipped unless --force.
Resumable: episodes are appended to the shard one at a time and the shard is rewritten
after each, so a killed run loses at most the episode in flight.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import ENV_NAME, engine_version, import_engine  # noqa: E402
from replay_check import _action_for_step  # noqa: E402

PIN = "1.32.7"


# --------------------------------------------------------------------------- engine


def _kg():
    """The kaggriculture engine module itself (not the wrapper)."""
    import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as kg

    return kg


NO_FLOOR = -1  # sentinel: this pool's $1 floor is unreachable (see FLOOR_SEARCH_CAP)

# Every pool starts the episode at 10,000 units and a whole 720-turn game moves it by
# tens to a few hundred (measured over 24 ladder episodes: max observed inventory
# 10,051-10,493 across all nine products). A floor sitting beyond 100,000 units --
# i.e. needing ~90,000 units of net supply -- is therefore not reachable by any play,
# and calling it "the floor" would put a fictional number in a training label. The cap
# is the honest boundary between "a mark" and "unbounded", not a numerical shortcut.
FLOOR_SEARCH_CAP = 100_000


def floor_inventory(kg, item, params, cap=FLOOR_SEARCH_CAP):
    """Smallest market inventory at which `item` quotes the $1 floor, or `NO_FLOOR`.

    `market_price` is non-increasing in inventory over the whole range (below I0 the
    premium shrinks as inventory rises; above I0 the discount grows), so a bisection
    is exact. Verified per run by `_check_monotone`.

    NOT EVERY POOL HAS A REACHABLE FLOOR. Under 1.32.7's defaults WHEAT is still at
    $15 and EGG at $30 with 100,000 units on the shelf, while CARROT/TOMATO/MELON/
    FERTILIZER/MILK/WOOL/STRAWBERRY floor between 10,059 and 10,842 -- that is 59 to
    842 units of headroom over the starting pool, which agents do reach. So the two
    grain/egg pools are effectively bottomless and their headroom is emitted as the
    `NO_FLOOR` sentinel rather than a fake number; consumers must mask those columns
    (`drain_remaining_*` stays valid for every product and is the usable substitute).
    Measured per episode from the episode's own params, never hard-coded, so an engine
    rebalance re-derives it instead of silently invalidating the labels.
    """
    if kg.market_price(item, cap, params) > kg.PRICE_FLOOR:
        return NO_FLOOR
    lo, hi = 0, cap
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if kg.market_price(item, mid, params) > kg.PRICE_FLOOR:
            lo = mid
        else:
            hi = mid
    return hi


def price_at_cap(kg, item, params, cap=FLOOR_SEARCH_CAP):
    """The quote at the reachability cap -- the practical floor of a bottomless pool."""
    return int(kg.market_price(item, cap, params))


def _check_monotone(kg, params, products, floors, cap=FLOOR_SEARCH_CAP):
    """Sample the price curve and report any inventory where price rises."""
    bad = []
    for p in products:
        top = max(4, floors[p] if floors[p] != NO_FLOOR else cap)
        step = max(1, top // 512)
        prev = None
        for inv in range(0, top + step, step):
            price = kg.market_price(p, inv, params)
            if prev is not None and price > prev:
                bad.append((p, inv, prev, price))
                break
            prev = price
    return bad


def town_drain(kg, cfg, shops, step, products):
    """Units of each product the town removes in the transition out of `step`.

    Mirrors `_town_consume` exactly. `shops` is the unlocked-shop list as of `step`
    (shops are drawn in `_end_of_day`, which runs *after* `_town_consume`, so the
    pre-step list is the one that fires).
    """
    shop_interval = max(1, int(cfg.get("townShopSellInterval", 4)))
    center_interval = max(1, int(cfg.get("townCenterSellInterval", 24)))
    drain = {p: 0 for p in products}
    if step % shop_interval == 0:
        for shop_name in shops:
            items = kg.SHOPS[shop_name]
            multiplier = 2 if len(items) == 1 else 1
            for item in items:
                if item in drain:
                    drain[item] += multiplier
    if step % center_interval == 0:
        for item in kg.TOWN_CENTER_PRODUCTS:
            if item in drain:
                drain[item] += 1
    return drain


# --------------------------------------------------------------------------- extract

# Column blocks written per (episode, transition) row, each shaped (n_rows, n_products).
PRODUCT_COLUMNS = [
    "inv_pre", "price_pre", "inv_post", "price_post", "drain",
    "p0_sold", "p0_sold_floored", "p0_bought",
    "p1_sold", "p1_sold_floored", "p1_bought",
    "p0_net_visible", "p0_net_true", "p1_net_visible", "p1_net_true",
    "p0_ordered_sell", "p0_ordered_buy", "p1_ordered_sell", "p1_ordered_buy",
    "identity_residual", "order_identity_residual",
    "inv_floor", "headroom_static",
    "drain_remaining_actual", "drain_remaining_known",
    "headroom_actual", "headroom_known",
    "p0_future_net_visible", "p1_future_net_visible",
    "p0_future_net_true", "p1_future_net_true",
]
SCALAR_COLUMNS = ["episode_id", "step", "day", "hour", "seat0_status", "seat1_status"]

_STATUS = {"ACTIVE": 0, "DONE": 1, "INACTIVE": 2, "TIMEOUT": 3, "ERROR": 4, "INVALID": 5}


def extract_episode(path: Path, episode_id: int | None = None, verbose: bool = False,
                    raw: bytes | None = None) -> dict:
    """Re-step one replay and build its label block. Returns a dict of arrays + meta."""
    kg = _kg()
    ke = import_engine()

    replay = json.loads(raw if raw is not None else path.read_bytes())
    steps = replay["steps"]
    cfg = dict(replay["configuration"])
    info = replay.get("info") or {}
    seed = info.get("seed")
    module_version = replay.get("module_version")
    ep_id = episode_id if episode_id is not None else info.get("EpisodeId")

    meta = {
        "file": str(path),
        "episode_id": ep_id,
        "engine_version_replay": module_version,
        "engine_version_local": engine_version(),
        "seed": seed,
        "recorded_steps": len(steps),
        "recorded_rewards": replay.get("rewards"),
    }
    if module_version != PIN:
        meta.update(ok=False, error=f"engine {module_version!r} != pin {PIN}")
        return meta
    if seed is None:
        meta.update(ok=False, error="replay has no info.seed -- cannot re-step deterministically")
        return meta

    products = list(kg.PRODUCTS)
    n_prod = len(products)
    p_index = {p: i for i, p in enumerate(products)}
    max_orders = max(1, int(cfg.get("maxMarketOrdersPerTurn", 10)))
    turns_per_day = max(1, int(cfg.get("turnsPerDay", 24)))

    env = ke.make(ENV_NAME, configuration=cfg, info={"seed": seed}, debug=False)
    env.reset(len(steps[0]))

    params = env.state[0].observation.market.get("params")
    floors = {p: floor_inventory(kg, p, params) for p in products}
    asymptotes = {p: price_at_cap(kg, p, params) for p in products}
    monotone_violations = _check_monotone(kg, params, products, floors)

    # ---- instrument the engine's single unit-commit point -------------------
    # `_process_market` resolves `_commit_unit` as a module global, so replacing the
    # module attribute intercepts every executed unit without touching the engine's
    # logic. Player identity comes from the `farm` dict passed in (object identity
    # against `obs.farms`), because `_commit_unit` itself is player-agnostic.
    #
    # The id map has to be built INSIDE the interpreter call: `env.step` hands the
    # interpreter a fresh deep copy of the state each step, so farm dicts captured
    # before the step are different objects and every commit would go unattributed
    # (silently -- the identity residual is the only thing that would notice).
    real_commit = kg._commit_unit
    real_process_market = kg._process_market
    farm_ids: dict[int, int] = {}
    tally = {}

    def _reset_tally():
        tally.clear()
        for pl in (0, 1):
            for key in ("sold", "sold_floored", "bought"):
                tally[(pl, key)] = [0] * n_prod

    def spy(op, item, price, farm, private, market, shed_capacity=100):
        ok = real_commit(op, item, price, farm, private, market, shed_capacity)
        if ok and item in p_index:
            pl = farm_ids.get(id(farm))
            if pl is not None:
                j = p_index[item]
                if op == "SELL":
                    tally[(pl, "sold")][j] += 1
                    if price <= kg.PRICE_FLOOR:
                        tally[(pl, "sold_floored")][j] += 1
                elif op == "BUY_PRODUCT":
                    tally[(pl, "bought")][j] += 1
        return ok

    def process_market_spy(state, env_):
        farm_ids.clear()
        for i, f in enumerate(state[0].observation.farms):
            farm_ids[id(f)] = i
        return real_process_market(state, env_)

    n_rows = len(steps) - 1
    cols = {c: np.zeros((n_rows, n_prod), dtype=np.int32) for c in PRODUCT_COLUMNS}
    scal = {c: np.zeros(n_rows, dtype=np.int64) for c in SCALAR_COLUMNS}

    market_mismatched_steps = 0
    first_market_mismatch = None
    rows = 0
    t0 = time.time()

    kg._commit_unit = spy
    kg._process_market = process_market_spy
    try:
        for t in range(n_rows):
            if env.done:
                meta.update(ok=False, error=f"local env finished early at step {t}")
                break
            obs0 = env.state[0].observation

            step_counter = int(kg.get(obs0, "step", t))
            inv_pre = [int(obs0.market["inventory"][p]) for p in products]
            price_pre = [int(obs0.market["prices"][p]) for p in products]
            shops_pre = list(obs0.town.get("unlocked_shops", []))
            drain = town_drain(kg, cfg, shops_pre, step_counter, products)

            # what each seat *asked* for (pre-fill), truncated the way the engine does
            for pl in (0, 1):
                act = steps[t + 1][pl].get("action")
                orders = []
                if isinstance(act, dict):
                    m = act.get("market", [])
                    if isinstance(m, list):
                        orders = m[:max_orders]
                for o in orders:
                    if not (isinstance(o, list) and len(o) >= 3 and o[1] in p_index):
                        continue
                    try:
                        n = int(o[2])
                    except (TypeError, ValueError):
                        continue
                    if n <= 0:
                        continue
                    j = p_index[o[1]]
                    if o[0] == "SELL":
                        cols[f"p{pl}_ordered_sell"][rows, j] += n
                    elif o[0] == "BUY_PRODUCT":
                        cols[f"p{pl}_ordered_buy"][rows, j] += n

            _reset_tally()
            actions = [_action_for_step(steps[t + 1][i], ke) for i in range(len(steps[t + 1]))]
            env.step(actions)

            obs0 = env.state[0].observation
            inv_post = [int(obs0.market["inventory"][p]) for p in products]
            price_post = [int(obs0.market["prices"][p]) for p in products]

            # engine parity, market slice: the local pool must equal the recorded pool
            rec_market = steps[t + 1][0]["observation"].get("market") or {}
            rec_inv = rec_market.get("inventory") or {}
            if rec_inv:
                if any(int(rec_inv.get(p, inv_post[j])) != inv_post[j] for j, p in enumerate(products)):
                    market_mismatched_steps += 1
                    if first_market_mismatch is None:
                        first_market_mismatch = t

            cols["inv_pre"][rows] = inv_pre
            cols["price_pre"][rows] = price_pre
            cols["inv_post"][rows] = inv_post
            cols["price_post"][rows] = price_post
            cols["drain"][rows] = [drain[p] for p in products]
            for pl in (0, 1):
                cols[f"p{pl}_sold"][rows] = tally[(pl, "sold")]
                cols[f"p{pl}_sold_floored"][rows] = tally[(pl, "sold_floored")]
                cols[f"p{pl}_bought"][rows] = tally[(pl, "bought")]

            scal["episode_id"][rows] = ep_id if ep_id is not None else -1
            scal["step"][rows] = step_counter
            scal["day"][rows] = step_counter // turns_per_day
            scal["hour"][rows] = step_counter % turns_per_day
            for pl in (0, 1):
                scal[f"seat{pl}_status"][rows] = _STATUS.get(steps[t + 1][pl].get("status"), 9)

            cols["inv_floor"][rows] = [floors[p] for p in products]
            rows += 1
            if verbose and t % 240 == 0:
                print(f"    step {t}/{n_rows}", file=sys.stderr)
    finally:
        kg._commit_unit = real_commit
        kg._process_market = real_process_market

    if rows and int(np.abs(cols["p0_sold"][:rows]).sum() + np.abs(cols["p1_sold"][:rows]).sum()) == 0:
        # nobody sold a single unit in 720 turns: that is an instrumentation failure,
        # not a game. Fail loudly rather than shipping a shard of zeros.
        meta.update(ok=False, error="instrumentation recorded no market commits at all")
        return meta

    if rows == 0:
        meta.setdefault("error", "no steps extracted")
        meta["ok"] = False
        return meta

    for c in PRODUCT_COLUMNS:
        cols[c] = cols[c][:rows]
    for c in SCALAR_COLUMNS:
        scal[c] = scal[c][:rows]

    # ---- derived label columns ---------------------------------------------
    for pl in (0, 1):
        sold = cols[f"p{pl}_sold"]
        floored = cols[f"p{pl}_sold_floored"]
        bought = cols[f"p{pl}_bought"]
        cols[f"p{pl}_net_visible"] = sold - floored - bought
        cols[f"p{pl}_net_true"] = sold - bought

    # identity, checked against the *recorded* pool where available
    dI = cols["inv_post"] - cols["inv_pre"]
    cols["identity_residual"] = (
        dI - (cols["p0_net_visible"] + cols["p1_net_visible"]) + cols["drain"]
    )
    # the same identity as a live agent could run it: own *orders* instead of own fills
    own_order_net = cols["p0_ordered_sell"] - cols["p0_ordered_buy"]
    cols["order_identity_residual"] = (
        dI - (own_order_net + cols["p1_net_visible"]) + cols["drain"]
    )

    # headroom: units still absorbable above the $1 floor. Pools whose curve never
    # reaches $1 (WHEAT, EGG under 1.32.7 defaults) carry NO_FLOOR throughout.
    unbounded = cols["inv_floor"] == NO_FLOOR
    cols["headroom_static"] = np.where(
        unbounded, NO_FLOOR, np.maximum(0, cols["inv_floor"] - cols["inv_pre"])
    )
    # remaining town drain, inclusive of this transition, walked backwards
    drain_rem = np.cumsum(cols["drain"][::-1], axis=0)[::-1]
    cols["drain_remaining_actual"] = drain_rem
    # ...and the same assuming the shop set never changes again (runtime-knowable)
    known = np.zeros_like(drain_rem)
    shop_interval = max(1, int(cfg.get("townShopSellInterval", 4)))
    center_interval = max(1, int(cfg.get("townCenterSellInterval", 24)))
    steps_arr = scal["step"]
    last_step = int(steps_arr[-1])
    # per-tick drain under the shop set observed at each row: reconstruct from `drain`
    # on tick rows is unreliable (a row can be both a shop tick and a centre tick), so
    # recompute from the recorded shop list.
    for i in range(rows):
        s = int(steps_arr[i])
        per_shop = {p: 0 for p in products}
        for shop_name in _shops_at(steps, i):
            items = kg.SHOPS[shop_name]
            multiplier = 2 if len(items) == 1 else 1
            for item in items:
                if item in per_shop:
                    per_shop[item] += multiplier
        n_shop_ticks = _count_multiples(s, last_step, shop_interval)
        n_center_ticks = _count_multiples(s, last_step, center_interval)
        for j, p in enumerate(products):
            centre = n_center_ticks if p in kg.TOWN_CENTER_PRODUCTS else 0
            known[i, j] = per_shop[p] * n_shop_ticks + centre
    cols["drain_remaining_known"] = known
    cols["headroom_actual"] = np.where(
        unbounded, NO_FLOOR, cols["headroom_static"] + cols["drain_remaining_actual"])
    cols["headroom_known"] = np.where(
        unbounded, NO_FLOOR, cols["headroom_static"] + cols["drain_remaining_known"])

    for pl in (0, 1):
        for kind in ("visible", "true"):
            src = cols[f"p{pl}_net_{kind}"]
            cols[f"p{pl}_future_net_{kind}"] = np.cumsum(src[::-1], axis=0)[::-1]

    local_rewards = [s.reward for s in env.state]
    meta.update(
        ok=("error" not in meta) and market_mismatched_steps == 0,
        rows=rows,
        products=products,
        local_rewards=local_rewards,
        rewards_match=(local_rewards == replay.get("rewards")),
        market_mismatched_steps=market_mismatched_steps,
        first_market_mismatch=first_market_mismatch,
        monotone_violations=monotone_violations,
        inv_floor={p: floors[p] for p in products},
        price_at_cap={p: asymptotes[p] for p in products},
        seconds=round(time.time() - t0, 1),
    )
    if market_mismatched_steps:
        meta["error"] = f"market inventory diverged from the replay at step {first_market_mismatch}"
    return {**meta, "cols": cols, "scal": scal}


def _shops_at(steps, t):
    obs = steps[t][0]["observation"]
    town = obs.get("town") or {}
    return list(town.get("unlocked_shops", []))


def _count_multiples(lo, hi, k):
    """How many u in [lo, hi] satisfy u % k == 0."""
    if hi < lo:
        return 0
    return hi // k - (lo - 1) // k if lo > 0 else hi // k + 1


# --------------------------------------------------------------------------- shard io


def load_shard(npz_path: Path):
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_shard(npz_path: Path, blocks: list[dict], products, compress=True):
    out = {}
    for c in PRODUCT_COLUMNS:
        out[c] = np.concatenate([b["cols"][c] for b in blocks], axis=0).astype(np.int32)
    for c in SCALAR_COLUMNS:
        out[c] = np.concatenate([b["scal"][c] for b in blocks], axis=0).astype(np.int64)
    out["products"] = np.array(products, dtype="<U12")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_suffix(".npz.tmp")
    saver = np.savez_compressed if compress else np.savez
    with open(tmp, "wb") as fh:
        saver(fh, **out)
    tmp.replace(npz_path)
    return out


# --------------------------------------------------------------------------- report


def validation_report(out: dict, episodes: list[dict]) -> dict:
    products = [str(p) for p in out["products"]]
    resid = out["identity_residual"]
    order_resid = out["order_identity_residual"]
    floored = out["p0_sold_floored"] + out["p1_sold_floored"]
    at_floor = out["price_pre"] <= 1
    n_cells = int(resid.size)

    kg = _kg()
    per_product = {}
    for j, p in enumerate(products):
        inv_floor = int(out["inv_floor"][0, j])
        per_product[p] = {
            "cells": int(resid.shape[0]),
            "identity_max_abs_residual": int(np.abs(resid[:, j]).max()),
            "identity_nonzero_cells": int((resid[:, j] != 0).sum()),
            "floor_regime_cells": int(at_floor[:, j].sum()),
            "floor_regime_frac": round(float(at_floor[:, j].mean()), 6),
            "units_sold_at_floor": int(floored[:, j].sum()),
            "units_sold_total": int((out["p0_sold"] + out["p1_sold"])[:, j].sum()),
            "opp_label_true_minus_visible_units": int(
                (out["p1_net_true"] - out["p1_net_visible"])[:, j].sum()
            ),
            "order_identity_max_abs_residual": int(np.abs(order_resid[:, j]).max()),
            "order_identity_nonzero_cells": int((order_resid[:, j] != 0).sum()),
            "inv_floor": inv_floor,
            "has_floor": inv_floor != NO_FLOOR,
            "price_at_cap": price_at_cap(kg, p, None),
            "headroom_actual_median": (int(np.median(out["headroom_actual"][:, j]))
                                       if inv_floor != NO_FLOOR else None),
            "headroom_known_median": (int(np.median(out["headroom_known"][:, j]))
                                      if inv_floor != NO_FLOOR else None),
            "drain_remaining_actual_day0": int(out["drain_remaining_actual"][0, j]),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": PIN,
        "episodes": len(episodes),
        "rows": int(resid.shape[0]),
        "cells": n_cells,
        "identity": {
            "max_abs_residual": int(np.abs(resid).max()),
            "nonzero_cells": int((resid != 0).sum()),
            "nonzero_frac": round(float((resid != 0).mean()), 9),
        },
        "order_identity": {
            "comment": "same identity run with our *orders* in place of our executed fills "
                       "-- the residual is exactly our unfilled order volume, i.e. the extra "
                       "noise a live in-agent estimator carries over these offline labels",
            "max_abs_residual": int(np.abs(order_resid).max()),
            "nonzero_cells": int((order_resid != 0).sum()),
            "nonzero_frac": round(float((order_resid != 0).mean()), 6),
        },
        "floor_regime": {
            "comment": "a SELL at the $1 floor pays the seller but adds no market supply, "
                       "so net_true - net_visible is the part of the flow no inventory-based "
                       "observer can ever see",
            "cells_at_floor": int(at_floor.sum()),
            "frac_cells_at_floor": round(float(at_floor.mean()), 6),
            "units_sold_at_floor": int(floored.sum()),
            "units_sold_total": int((out["p0_sold"] + out["p1_sold"]).sum()),
        },
        "per_product": per_product,
        "episode_meta": [
            {k: e[k] for k in ("episode_id", "seed", "rows", "market_mismatched_steps",
                               "rewards_match", "seconds") if k in e}
            for e in episodes
        ],
    }


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replays", nargs="+", required=True, help="replay JSON file(s) or globs")
    p.add_argument("--out", required=True, help="output directory (warehouse tier)")
    p.add_argument("--tag", default="ladder-1.32.7", help="shard name suffix")
    p.add_argument("--sample", type=int, default=0, help="take a deterministic sample of N replays (0 = all)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument("--limit", type=int, default=0, help="stop after N successful episodes (0 = no limit)")
    p.add_argument("--force", action="store_true", help="re-extract episodes already in the shard")
    p.add_argument("--workers", type=int, default=4, help="parallel replay readers (IO-bound prefetch)")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    files: list[Path] = []
    for pat in args.replays:
        matches = sorted(glob.glob(pat))
        files += [Path(m) for m in matches] if matches else [Path(pat)]
    files = [f for f in files if f.exists()]
    if not files:
        print("no replay files matched", file=sys.stderr)
        return 1
    if args.sample and args.sample < len(files):
        rng = random.Random(args.seed)
        files = sorted(rng.sample(files, args.sample))

    out_dir = Path(args.out)
    npz_path = out_dir / f"s2_labels_{args.tag}.npz"
    meta_path = out_dir / f"s2_episodes_{args.tag}.csv"
    report_path = out_dir / f"s2_validation_{args.tag}.json"

    existing = load_shard(npz_path)
    done_ids = set()
    blocks: list[dict] = []
    episodes: list[dict] = []
    if existing is not None and not args.force:
        done_ids = set(int(x) for x in np.unique(existing["episode_id"]))
        print(f"shard {npz_path.name}: {len(done_ids)} episode(s) already extracted")
        # rehydrate as one pseudo-block so appends keep everything
        blocks.append({
            "cols": {c: existing[c] for c in PRODUCT_COLUMNS},
            "scal": {c: existing[c] for c in SCALAR_COLUMNS},
        })
        if meta_path.exists():
            import csv

            with open(meta_path, newline="", encoding="utf-8") as fh:
                episodes = [dict(r) for r in csv.DictReader(fh)]

    todo = [f for f in files
            if (_episode_id_from_name(f) is None or _episode_id_from_name(f) not in done_ids)]
    if args.limit:
        todo = todo[: args.limit + 4]  # a little slack for replays that fail to qualify
    print(f"{len(todo)} replay(s) to extract (workers={args.workers})")

    products = None
    n_new = 0
    for f, raw in prefetch(todo, args.workers):
        ep_id = _episode_id_from_name(f)
        if args.limit and n_new >= args.limit:
            break
        print(f"extracting {f.name} ({len(raw) / 1e6:.1f} MB) ...", flush=True)
        try:
            res = extract_episode(f, episode_id=ep_id, verbose=args.verbose, raw=raw)
        except Exception as exc:  # noqa: BLE001 - one bad replay must not kill the run
            print(f"  FAILED {f.name}: {exc!r}")
            episodes.append({"episode_id": ep_id, "file": str(f), "ok": False, "error": repr(exc)})
            continue
        if not res.get("ok"):
            print(f"  SKIP {f.name}: {res.get('error')}")
            episodes.append({k: v for k, v in res.items() if k not in ("cols", "scal")})
            continue
        products = res["products"]
        blocks.append({"cols": res["cols"], "scal": res["scal"]})
        episodes.append({
            "episode_id": res["episode_id"],
            "file": str(f),
            "seed": res["seed"],
            "engine_version": res["engine_version_replay"],
            "rows": res["rows"],
            "market_mismatched_steps": res["market_mismatched_steps"],
            "rewards_match": res["rewards_match"],
            "recorded_rewards": json.dumps(res["recorded_rewards"]),
            "seconds": res["seconds"],
            "ok": True,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        n_new += 1
        print(f"  ok  rows={res['rows']}  parity_mismatches={res['market_mismatched_steps']}  "
              f"{res['seconds']}s", flush=True)
        # checkpoint after every episode: the run is resumable
        out = write_shard(npz_path, blocks, products or list(_kg().PRODUCTS),
                          compress=not args.no_compress)
        _write_meta(meta_path, episodes)

    if not blocks:
        print("nothing extracted")
        return 1
    out = load_shard(npz_path)
    report = validation_report(out, [e for e in episodes if str(e.get("ok")).lower() == "true"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"shard      {npz_path}  rows={report['rows']}  episodes={report['episodes']}")
    print(f"identity   max|residual|={report['identity']['max_abs_residual']}  "
          f"nonzero cells={report['identity']['nonzero_cells']}/{report['cells']}")
    print(f"order-form max|residual|={report['order_identity']['max_abs_residual']}  "
          f"nonzero cells={report['order_identity']['nonzero_cells']}")
    print(f"floor      cells at $1 = {report['floor_regime']['cells_at_floor']} "
          f"({report['floor_regime']['frac_cells_at_floor']:.3%})  "
          f"units sold at floor = {report['floor_regime']['units_sold_at_floor']} / "
          f"{report['floor_regime']['units_sold_total']}")
    print(f"report     {report_path}")
    return 0 if report["identity"]["nonzero_cells"] == 0 else 1


def _write_meta(path: Path, episodes: list[dict]):
    import csv

    if not episodes:
        return
    fields = []
    for e in episodes:
        for k in e:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for e in episodes:
            w.writerow(e)


def prefetch(paths, workers: int):
    """Yield (path, raw_bytes), reading `workers` files ahead of the consumer.

    Replays live on the bulk tier (a shared spinning disk). A single sequential
    reader gets whatever slice of the IO queue it can win against the other
    processes on that disk; a handful of concurrent readers wins proportionally
    more, and the re-step itself is ~2 s of CPU against ~100 s of read, so the
    pipeline is entirely IO-bound without this. Backpressure keeps at most
    `workers` replays (~30 MB each) resident.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = list(paths)
    if workers <= 1:
        for p in paths:
            yield p, p.read_bytes()
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        in_flight = []
        nxt = 0
        while nxt < len(paths) and len(in_flight) < workers:
            in_flight.append((paths[nxt], ex.submit(paths[nxt].read_bytes)))
            nxt += 1
        while in_flight:
            path, fut = in_flight.pop(0)
            if nxt < len(paths):
                in_flight.append((paths[nxt], ex.submit(paths[nxt].read_bytes)))
                nxt += 1
            yield path, fut.result()


def _episode_id_from_name(path: Path):
    stem = path.stem
    for token in (stem, stem.replace("episode-", "").replace("-replay", "")):
        if token.isdigit():
            return int(token)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
