"""Per-episode, per-commodity market census over the ghost corpus (G-1 companion).

Produces the table the **uncontested-pool pricing** item needs (Architect A ruling R-4b
INSERT 2, re-scoped by Architect B's ruling-02-addendum-02 §1): for every corpus episode
and every product, how many units each seat actually sold and bought, **at what realised
price**, and what the pool's quote path and town drain looked like while they did it.

Why realised price and not just units.  The addendum's binding requirement is $/season
*capturable at our own price impact*, not gross dollars left on the table.  That needs the
demand curve as it was actually walked -- units, revenue, and the quote at entry and exit --
so a consumer can integrate marginal revenue rather than multiply units by a sticker quote.
The S-2 label pipeline (`scripts/extract_s2_labels.py`) counts units but not money; this
adds the money and aggregates to one row per (episode, product) instead of per turn.

Why a re-step and not accounting off the recorded observations.  Identical reason to S-2,
and it is not optional: the engine applies unit actions (PICKUP / DROP / harvest deposits)
to a player's shed **before** `_process_market` runs in the same step, so shed deltas
between two recorded observations do not isolate market flow.  The only exact attribution
is to re-step the replay through the pinned engine with `_commit_unit` instrumented -- each
successful commit is one unit, with its player, item and executed price.

Two traps inherited from S-2, both of which fail *silently*:

 1. `env.step` hands the interpreter a **fresh deep copy** of the state each step, so a
    player id map built before the step matches nothing during it and every commit goes
    unattributed.  The map is therefore rebuilt inside a patched `_process_market`.  An
    episode that records zero commits is reported as an error, never as a quiet zero row.
 2. Actions are off by one: `steps[t+1][seat]["action"]` is the action chosen **from** the
    observation at `t`.  Pairing at the same index looks plausible and is wrong.

Floor caveat, carried from S-2: a SELL executed at the $1 floor pays the seller but does
not add market supply, so `units_sold_floored` is broken out and `net_visible` differs from
`net_true` exactly there.

Output: `census.csv` beside the corpus index, one row per (episode, product).  Small
enough to stay a csv (360 episodes x 9 products = ~3,240 rows).

Usage:
    uv run python scripts/census_ghost_corpus.py --limit 4          # pilot
    uv run python scripts/census_ghost_corpus.py                    # whole corpus
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import ENV_NAME, REPO_ROOT, engine_version, import_engine  # noqa: E402
from replay_check import _action_for_step  # noqa: E402

WAREHOUSE = Path(os.environ.get("GUILDHALL_WAREHOUSE") or (REPO_ROOT.parents[3] / "warehouse"))
CORPUS = WAREHOUSE / "datasets" / "ghost-corpus"
PIN = "1.32.7"

FIELDS = [
    "episode_id", "tier", "team_id", "lb_rank", "corpus_seat", "product",
    # what each seat did in this pool over the whole episode
    "p0_units_sold", "p0_units_sold_floored", "p0_units_bought", "p0_sell_revenue", "p0_buy_cost",
    "p1_units_sold", "p1_units_sold_floored", "p1_units_bought", "p1_sell_revenue", "p1_buy_cost",
    "p0_realised_price", "p1_realised_price",
    # the pool itself
    "town_drain_units", "inv_start", "inv_end", "inv_min", "inv_max",
    "price_start", "price_end", "price_min", "price_max", "price_mean",
    "turns_at_floor", "net_true_total", "net_visible_total",
    "identity_residual_abs",       # 0 == the re-step reproduced the recorded pool exactly
    "market_parity_mismatches",
    "engine_version",
]


def _kg():
    import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as kg
    return kg


def town_drain(kg, cfg, shops, step, products):
    """Units of each product the town removes in the transition out of `step`.

    Mirrors `_town_consume`: shops are drawn in `_end_of_day`, which runs *after*
    `_town_consume`, so the pre-step shop list is the one that fires.
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


def census_episode(path: Path) -> dict:
    """Re-step one packed replay and aggregate its market flow per product."""
    kg = _kg()
    ke = import_engine()

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        replay = json.load(f)

    steps = replay["steps"]
    cfg = dict(replay["configuration"])
    info = replay.get("info") or {}
    seed = info.get("seed")
    module_version = replay.get("module_version")

    if module_version != PIN:
        return {"ok": False, "error": f"engine {module_version!r} != pin {PIN}"}
    if seed is None:
        return {"ok": False, "error": "replay has no info.seed -- cannot re-step deterministically"}

    products = list(kg.PRODUCTS)
    p_index = {p: i for i, p in enumerate(products)}
    n = len(products)

    env = ke.make(ENV_NAME, configuration=cfg, info={"seed": seed}, debug=False)
    env.reset(len(steps[0]))

    # ---- instrument the engine's single unit-commit point --------------------
    real_commit = kg._commit_unit
    real_process_market = kg._process_market
    farm_ids: dict[int, int] = {}
    agg = {(pl, k): [0] * n for pl in (0, 1)
           for k in ("sold", "sold_floored", "bought", "sell_rev", "buy_cost")}

    def spy(op, item, price, farm, private, market, shed_capacity=100):
        ok = real_commit(op, item, price, farm, private, market, shed_capacity)
        if ok and item in p_index:
            pl = farm_ids.get(id(farm))
            if pl is not None:
                j = p_index[item]
                if op == "SELL":
                    agg[(pl, "sold")][j] += 1
                    agg[(pl, "sell_rev")][j] += int(price)
                    if price <= kg.PRICE_FLOOR:
                        agg[(pl, "sold_floored")][j] += 1
                elif op == "BUY_PRODUCT":
                    agg[(pl, "bought")][j] += 1
                    agg[(pl, "buy_cost")][j] += int(price)
        return ok

    def process_market_spy(state, env_):
        farm_ids.clear()
        for i, f in enumerate(state[0].observation.farms):
            farm_ids[id(f)] = i
        return real_process_market(state, env_)

    drain_tot = [0] * n
    inv_min = [None] * n
    inv_max = [None] * n
    price_min = [None] * n
    price_max = [None] * n
    price_sum = [0] * n
    at_floor = [0] * n
    inv_start = price_start = None
    inv_end = price_end = None
    parity_mismatch = 0
    rows = 0
    commits_seen_before = 0

    kg._commit_unit = spy
    kg._process_market = process_market_spy
    try:
        for t in range(len(steps) - 1):
            if env.done:
                break
            obs0 = env.state[0].observation
            step_counter = int(kg.get(obs0, "step", t))
            inv = [int(obs0.market["inventory"][p]) for p in products]
            price = [int(obs0.market["prices"][p]) for p in products]
            if inv_start is None:
                inv_start, price_start = list(inv), list(price)
            for j in range(n):
                inv_min[j] = inv[j] if inv_min[j] is None else min(inv_min[j], inv[j])
                inv_max[j] = inv[j] if inv_max[j] is None else max(inv_max[j], inv[j])
                price_min[j] = price[j] if price_min[j] is None else min(price_min[j], price[j])
                price_max[j] = price[j] if price_max[j] is None else max(price_max[j], price[j])
                price_sum[j] += price[j]
                at_floor[j] += 1 if price[j] <= kg.PRICE_FLOOR else 0

            d = town_drain(kg, cfg, list(obs0.town.get("unlocked_shops", [])),
                           step_counter, products)
            for j, p in enumerate(products):
                drain_tot[j] += d[p]

            actions = [_action_for_step(steps[t + 1][i], ke) for i in range(len(steps[t + 1]))]
            env.step(actions)

            obs0 = env.state[0].observation
            inv_end = [int(obs0.market["inventory"][p]) for p in products]
            price_end = [int(obs0.market["prices"][p]) for p in products]

            # engine parity, market slice: the local pool must equal the recorded pool
            rec = (steps[t + 1][0]["observation"].get("market") or {}).get("inventory") or {}
            if rec and any(int(rec.get(p, inv_end[j])) != inv_end[j]
                           for j, p in enumerate(products)):
                parity_mismatch += 1
            rows += 1
    finally:
        kg._commit_unit = real_commit
        kg._process_market = real_process_market

    total_commits = sum(sum(agg[(pl, k)]) for pl in (0, 1) for k in ("sold", "bought"))
    if rows == 0:
        return {"ok": False, "error": "no steps re-stepped"}
    if total_commits == commits_seen_before:
        # nobody traded a single unit in 720 turns: an instrumentation failure, not a game
        return {"ok": False, "error": "instrumentation recorded no market commits at all"}

    out = []
    for j, p in enumerate(products):
        net_true = ((agg[(0, "sold")][j] + agg[(1, "sold")][j])
                    - (agg[(0, "bought")][j] + agg[(1, "bought")][j]))
        net_visible = net_true - (agg[(0, "sold_floored")][j] + agg[(1, "sold_floored")][j])
        # inv_end - inv_start == net_visible - drain, exactly (S-2's identity, integrated)
        residual = (inv_end[j] - inv_start[j]) - net_visible + drain_tot[j]
        row = {
            "product": p,
            "town_drain_units": drain_tot[j],
            "inv_start": inv_start[j], "inv_end": inv_end[j],
            "inv_min": inv_min[j], "inv_max": inv_max[j],
            "price_start": price_start[j], "price_end": price_end[j],
            "price_min": price_min[j], "price_max": price_max[j],
            "price_mean": round(price_sum[j] / rows, 2),
            "turns_at_floor": at_floor[j],
            "net_true_total": net_true, "net_visible_total": net_visible,
            "identity_residual_abs": abs(residual),
            "engine_version": module_version,
        }
        for pl in (0, 1):
            sold = agg[(pl, "sold")][j]
            bought = agg[(pl, "bought")][j]
            rev = agg[(pl, "sell_rev")][j]
            cost = agg[(pl, "buy_cost")][j]
            row[f"p{pl}_units_sold"] = sold
            row[f"p{pl}_units_sold_floored"] = agg[(pl, "sold_floored")][j]
            row[f"p{pl}_units_bought"] = bought
            row[f"p{pl}_sell_revenue"] = rev
            row[f"p{pl}_buy_cost"] = cost
            row[f"p{pl}_realised_price"] = round(rev / sold, 2) if sold else ""
        out.append(row)
    return {"ok": True, "rows": out, "steps": rows,
            "market_parity_mismatches": parity_mismatch,
            "engine_version_local": engine_version()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--limit", type=int, default=0, help="stop after N episodes (pilot runs)")
    p.add_argument("--tier", help="restrict to one tier")
    p.add_argument("--out", default="", help="default <corpus>/census.csv")
    p.add_argument("--force", action="store_true", help="re-census episodes already present")
    args = p.parse_args(argv)

    corpus = Path(args.corpus)
    index_path = corpus / "index.csv"
    if not index_path.exists():
        raise SystemExit(f"no corpus index at {index_path} -- run fetch_ghost_corpus.py first")
    with open(index_path, newline="", encoding="utf-8") as f:
        index = list(csv.DictReader(f))

    out_path = Path(args.out) if args.out else corpus / "census.csv"
    done: set[int] = set()
    if out_path.exists() and not args.force:
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {int(r["episode_id"]) for r in csv.DictReader(f)}

    todo = [r for r in index
            if (not args.tier or r["tier"] == args.tier) and int(r["episode_id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(index)} indexed, {len(done)} already censused, {len(todo)} to do",
          file=sys.stderr)

    new = not out_path.exists()
    ok = failed = 0
    residual_bad = parity_bad = 0
    t_start = time.time()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, meta in enumerate(todo, 1):
            path = corpus / meta["replay_path"]
            t0 = time.time()
            try:
                res = census_episode(path)
            except Exception as e:                       # one bad replay must not kill a run
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if not res.get("ok"):
                failed += 1
                print(f"  [{i}/{len(todo)}] {meta['episode_id']} FAILED {res.get('error')}",
                      file=sys.stderr)
                continue
            bad = sum(1 for r in res["rows"] if r["identity_residual_abs"] != 0)
            residual_bad += bad
            parity_bad += 1 if res["market_parity_mismatches"] else 0
            for r in res["rows"]:
                r.update({
                    "episode_id": meta["episode_id"], "tier": meta["tier"],
                    "team_id": meta["team_id"], "lb_rank": meta["lb_rank"],
                    "corpus_seat": meta["seat"],
                    "market_parity_mismatches": res["market_parity_mismatches"],
                })
                w.writerow({k: r.get(k, "") for k in FIELDS})
            f.flush()                                    # resumable: never lose a done episode
            ok += 1
            print(f"  [{i}/{len(todo)}] {meta['episode_id']} ok  {res['steps']} steps, "
                  f"{time.time()-t0:.1f}s, residual-nonzero {bad}/9, "
                  f"parity mismatches {res['market_parity_mismatches']}", file=sys.stderr)

    print(f"\ncensused {ok}, failed {failed} in {time.time()-t_start:.0f}s -> {out_path}",
          file=sys.stderr)
    print(f"identity residual non-zero in {residual_bad} (episode, product) cells "
          f"-- must be 0; a non-zero residual means the drain model or the "
          f"instrumentation is wrong, it is not a tolerance", file=sys.stderr)
    print(f"episodes with any market-parity mismatch: {parity_bad}", file=sys.stderr)
    json.dump({"censused": ok, "failed": failed,
               "identity_residual_nonzero_cells": residual_bad,
               "episodes_with_parity_mismatch": parity_bad,
               "engine_pin": PIN, "engine_local": engine_version(),
               "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
              open(corpus / "meta" / "census_validation.json", "w"), indent=2)
    return 1 if (residual_bad or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
