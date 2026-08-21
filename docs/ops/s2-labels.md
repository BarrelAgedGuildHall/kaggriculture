# Runbook — S-2 supervised label pipeline

Owner: Ops. Consumer: Team A's `exp-014` auxiliary heads (opponent net sales, pool headroom).
Programme item **S-2** in `workshop/deliverables/2026-08-21-balance-sheet-program/plan.md`.
Engine pin **kaggle-environments 1.32.7** — labels are *not* comparable across an engine bump.

## What it produces

`scripts/extract_s2_labels.py` turns replay JSONs into a per-turn, per-product label table:

| family | columns | meaning |
|---|---|---|
| (a) opponent flow | `p{0,1}_net_true`, `p{0,1}_net_visible`, `p{0,1}_sold`, `p{0,1}_sold_floored`, `p{0,1}_bought` | units each seat actually pushed into / pulled out of each shared pool that turn |
| (b) pool headroom | `headroom_static`, `drain_remaining_actual`, `drain_remaining_known`, `headroom_actual`, `headroom_known`, `inv_floor` | units the pool can still absorb above the $1 floor between now and the horizon |
| context | `inv_pre/post`, `price_pre/post`, `drain`, `p{0,1}_ordered_{sell,buy}` | market state and what each seat *asked* for (vs. what filled) |
| lookahead | `p{0,1}_future_net_{true,visible}` | suffix sums — the rival's remaining supply, for `headroom - opponent` marks |
| diagnostics | `identity_residual`, `order_identity_residual` | see Validation below |

The table is **seat-symmetric**: one row per `(episode, transition)`, with explicit `p0_*` / `p1_*`
columns rather than a duplicated per-seat row. A consumer training seat `s` reads its own flow from
`p{s}_*` and the label from `p{1-s}_*`.

## Why it re-steps instead of differencing observations

The engine applies unit actions (PICKUP / DROP / harvest deposits) to a player's shed **before**
`_process_market` runs in the same step, so shed deltas between two recorded observations do not
isolate market flow. The only exact attribution is to re-step the replay through the pinned engine
with `_commit_unit` instrumented — every successful commit is one unit, with its player, item and
executed price.

Two traps live here, both of which fail *silently*:

1. `env.step` hands the interpreter a **fresh deep copy** of the state each step, so a player id map
   built from `env.state[...].farms` before the step matches nothing during it and every commit goes
   unattributed. The map is therefore built inside a patched `_process_market`. The extractor aborts
   the episode if it records zero commits, and `test_instrumentation_actually_saw_trades` guards it.
2. Actions are off by one: `steps[t+1][seat]["action"]` is the action chosen **from** the observation
   at `t`. Pairing at the same index looks plausible and is wrong.

## The identity, and where it degrades

Only two code paths write `market["inventory"]` — `_commit_unit` and `_town_consume` — so for every
product and every transition out of step `t`:

    inv[t+1] - inv[t] == sum_players (sold_above_floor - bought) - drain(t)

Town drain needs no estimation offline: it is a pure function of `obs.town.unlocked_shops` and the
step counter (shop interval 4, single-product shops 2x, town centre every 24, FERTILIZER never).

**The floor caveat is the whole point of the bookkeeping.** A SELL executed at the $1 floor pays the
seller but does not add market supply. So `net_visible = sold_above_floor - bought` is what inventory
can ever reveal, while `net_true = sold_total - bought` is what happened. Both ship: `net_true` is
the supervised target, `net_visible` is the ceiling on what any inventory-based runtime estimator
could infer. `net_true - net_visible == sold_floored` by construction.

**Not every pool has a reachable floor.** All nine start at 10,000 units. Seven floor between 10,059
and 10,842 (WOOL +59, STRAWBERRY +62, MILK +76, MELON +158, FERTILIZER +493, TOMATO +529,
CARROT +842 above the start) — thin, and agents do reach them. WHEAT and EGG still quote $15 and $30
at 100,000 units: bottomless in practice. Their headroom columns carry the `NO_FLOOR = -1` sentinel
and **must be masked by consumers**; `drain_remaining_*` stays valid for all nine and is the
substitute. All of this is measured per episode from the episode's own market params, so an engine
rebalance re-derives it rather than silently invalidating the labels.

## Validation, per run

The script writes `s2_validation_<tag>.json` and exits non-zero if the identity does not close.

- **Replay parity, market slice** — local market inventory is diffed against the recorded inventory
  at every step; any episode that diverges is dropped, not repaired.
- **Identity residual** — must be exactly 0 in every cell. A non-zero residual means the drain model
  or the instrumentation is wrong; it is not a tolerance.
- **Order-form residual** — the same identity run with our *orders* in place of our executed fills.
  This is the extra noise a live in-agent estimator carries over these offline labels (unfilled
  orders + our own floored sales). Reported, not gated.
- **Floor regime** — fraction of cells quoting $1 and units sold there, per product. This is the
  honest bound on label quality for the thin pools.

## Running it

```bash
cd artisans/ops/bench/repo
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/extract_s2_labels.py \
  --replays "$GUILDHALL_WAREHOUSE/datasets/kaggriculture/replays/1.32.7/*.json" \
  --sample 24 --seed 0 --workers 6 \
  --out <build-dir> --tag ladder-1.32.7
```

Idempotent (episodes already in the shard are skipped) and resumable (the shard is rewritten after
every episode, so a kill loses at most the episode in flight). `--force` re-extracts everything.

**Build on the fast tier, publish to the warehouse.** The run checkpoints the shard after each
episode; two dozen rewrites of a ~20 MB file onto the shared spinning disk is exactly the pattern
the storage doctrine forbids. Copy the three finished artefacts across once at the end.

### Reading the shard

```python
import numpy as np
z = np.load(".../s2_labels_ladder-1.32.7.npz")
products = [str(p) for p in z["products"]]        # column order for every 2-D array
label    = z["p1_net_true"]                        # (rows, 9) — seat 0's opponent flow
headroom = np.where(z["headroom_actual"] < 0, np.nan, z["headroom_actual"])  # mask WHEAT/EGG
episode  = z["episode_id"]; step = z["step"]       # (rows,) keys — group/split by episode
```

Split **by episode**, never by row: consecutive turns inside one episode are strongly correlated and
the lookahead columns are suffix sums over the whole episode.

## Sources it accepts

Any kaggle-environments replay JSON carrying `module_version == 1.32.7` and `info.seed`. That covers
the warehouse ladder library and the Pit's own captures. Note that the Pit currently persists only
**crash** replays (`data/arena/replays/crash_*.json` on workflows-box, 40 files as of 2026-08-21) —
normal gate episodes are not written to disk, so the Pit is not yet a usable corpus for S-2. If Team
A wants self-play labels, the Pit needs an opt-in replay-retention flag first (Engineer owns the
hub); the extractor itself needs no change.

## When the engine bumps

Labels do not average across an engine version. On a bump: re-run `scripts/replay_check.py`, then
re-extract into a **new tag** (`--tag ladder-<version>`) and leave the old shard in place.
`test_wheat_and_egg_pools_cannot_be_crashed_to_the_floor` and `test_floor_inventory_*` are the
tripwires — they pin the measured floor inventories, so a rebalance that moves a pool's floor fails
CI rather than quietly changing what "headroom" means.
