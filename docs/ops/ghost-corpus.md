# Runbook — ghost corpus (G-1) and its per-commodity census

Owner: Ops. Consumers: `engineer` (G-2 BC ghosts → `ghostpool-v1`), Ops (the
uncontested-pool pricing table → Team B's `expB-304b`).
Engine pin **kaggle-environments 1.32.7** — a corpus is not comparable across an engine bump.

Governing documents:
- `workshop/deliverables/2026-08-22-pit-ladder-calibration-ruling/ruling.md` §R-2 Stage 1 (G-1)
  and §R-4b INSERT 2 (uncontested-pool pricing).
- `workshop/deliverables/2026-08-21-team-b-business-lens/ruling-02-addendum-02.md` §1 — the
  pricing table's scope change and its re-addressed consumer.

## What it is

Three tiers of public-leaderboard teams, **winning episodes only**, most recent first:

| tier | public LB rank | teams | episodes/team | why |
|---|---|---|---|---|
| `frontier` | 1–10 | 4 | 30 | what beats the frontier |
| `upper` | 100–300 | 4 | 30 | the middle of the field |
| `field` | 2,800–3,300 | 4 | 30 | **our own neighbourhood — who we actually meet** |

360 replays, packed, in `$GUILDHALL_WAREHOUSE/datasets/ghost-corpus/`.

## Running it

```bash
# 1. select (metadata only -- no replay bytes). Writes meta/selection.json.
uv run python scripts/fetch_ghost_corpus.py --plan

# 2. fetch. ALWAYS --use-selection once a selection exists (see "Freeze the selection").
uv run python scripts/fetch_ghost_corpus.py --use-selection --delay-s 0.5

# 3. per-commodity census (re-steps every replay; ~2.5 s each, ~15 min for 360)
uv run python scripts/census_ghost_corpus.py
```

Both are idempotent and resumable: `index.csv` / `census.csv` are appended one episode at a
time and flushed, so a killed run continues where it stopped. `index.csv` is authoritative —
a `.json.gz` with no index row is incomplete and gets re-fetched.

## Freeze the selection

`--plan` writes `meta/selection.json` and the fetch replays it with `--use-selection`. Do not
let a fetch re-select. Two reasons:

1. **The corpus would silently drift.** "30 most recent wins" is evaluated against a live
   ladder; a re-run days later is a different corpus under the same name, which is exactly
   the failure mode `ghostpool-v1@<sha>` versioning exists to prevent.
2. **Kaggle rate-limits `ListEpisodes`.** Selection makes ~30–70 unauthenticated API calls;
   run it three times in half an hour and the fourth run stalls in backoff for minutes with
   no output. That is what it looks like when it happens — the process is alive, burning no
   CPU, and no scratch directory has appeared yet.

## Storage and the cap

A raw replay is ~30 MB and gzips to ~0.55 MB (**~55×** — the observation dumps are enormously
repetitive), so replays are stored `.json.gz` and the raw file is deleted as soon as it is
packed. Peak scratch is one file; the full 360 lands around **200 MB against the ruling's
5 GB hard cap** (~4%). The cap is decimal (5e9, the conservative reading), enforced on packed
bytes, and the run aborts and reports rather than half-filling the disk.

Read a replay with `gzip.open(path, "rt", encoding="utf-8")`.

## Engine pin

Every replay's own top-level `module_version` is checked against the pin **after download and
before packing**; a mismatch is deleted, counted, and reported per version at the end. The pin
is never assumed from the fetch date. As of 2026-08-22 the entire corpus is 1.32.7 (1.32.7
shipped 2026-08-15 and Kaggle adopts within ~a day, so recent episodes are uniform).

## The census, and why it re-steps

`census_ghost_corpus.py` emits one row per (episode, product): each seat's units sold /
sold-at-the-$1-floor / bought, **sell revenue and realised $/unit**, plus the pool's town
drain, inventory path and quote path, and turns spent at the floor.

It re-steps each replay through the pinned engine with `_commit_unit` instrumented rather
than differencing recorded observations. This is not a preference:

- The engine applies unit actions (PICKUP / DROP / harvest deposits) to a player's shed
  **before** `_process_market` runs in the same step, so shed deltas between two recorded
  observations do not isolate market flow.
- `env.step` hands the interpreter a **fresh deep copy** of the state each step, so a
  player-id map built before the step matches nothing during it and every commit goes
  unattributed — *silently*. The map is rebuilt inside a patched `_process_market`, and an
  episode with zero commits is reported as an error, never written as a quiet zero row.
- Actions are off by one: `steps[t+1][seat]["action"]` is the action chosen **from** the
  observation at `t`.

All three traps are inherited from `scripts/extract_s2_labels.py`; see `docs/ops/s2-labels.md`
for the longer derivation.

### Validation, per run

`meta/census_validation.json`, and a non-zero exit if anything failed:

- **Market-inventory identity**, per product per episode:
  `inv_end − inv_start == net_visible − town_drain`. Must be **exactly 0**. It is not a
  tolerance — a non-zero residual means the drain model or the instrumentation is wrong.
- **Replay parity, market slice** — the local pool is diffed against the recorded pool at
  every step; mismatched episodes are counted.
- **Floor bookkeeping** — a SELL executed at the $1 floor pays the seller but does not add
  market supply, so `units_sold_floored` is broken out and `net_true` exceeds `net_visible`
  exactly there. Any consumer integrating a demand curve must use the floored count.

## Reading it for the pricing table

The addendum's scope is **$/season capturable at our own price impact, per tile-day and per
action** — *not* gross dollars left on the table. So the census deliberately ships the quote
path (`price_start/end/min/max/mean`, `inv_*`) and `town_drain_units` next to the units: the
binding constraint on an uncontested pool is the shop drain rate and the slope of the demand
curve under **our** selling, not the field's absence. Multiplying field units by a sticker
quote would make an allocator over-buy.

`EGG` is the table's first row: it doubles as a falsification test of Architect B's R-8
goose-CARE ruling.
