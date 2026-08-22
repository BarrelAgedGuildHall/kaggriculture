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

Plus **review tiers**, which are not behavioural bands and are added on request beside the
three above without disturbing them (own frozen selection file, own directory, rows tagged
with their own `tier`):

| tier | public LB rank | teams | episodes | why |
|---|---|---|---|---|
| `rank1` | 1 | 1 | 40 W + 10 L | granular review of the single best team |

**A review tier carries losses on purpose** — a behavioural review of one specific team has
to see what beats it. The index has no `outcome` column (adding one would rewrite the header
of a file live consumers are reading), so **`bank_margin` is the outcome**: strictly positive
on a win, strictly negative on a loss, never zero. A consumer that wants the old wins-only
invariant filters `bank_margin > 0`, or filters to `RULING_TIERS`.

## Running it

```bash
# 1. select (metadata only -- no replay bytes). Writes meta/selection.json.
uv run python scripts/fetch_ghost_corpus.py --plan

# 2. fetch. ALWAYS --use-selection once a selection exists (see "Freeze the selection").
uv run python scripts/fetch_ghost_corpus.py --use-selection --delay-s 0.5

# 3. per-commodity census (re-steps every replay; ~3 s each, ~18 min for 352)
uv run python scripts/census_ghost_corpus.py

# top up a short tier with teams it does not already have, writing a SEPARATE selection
uv run python scripts/fetch_ghost_corpus.py --plan --top-up --tier field --target 30 --min-teams 1
uv run python scripts/fetch_ghost_corpus.py --use-selection <corpus>/meta/selection-topup.json

# add a review tier: the rank-1 team's 40 most recent wins + 10 most recent losses.
# --submissions is not optional up here -- see "The mirror cannot see the top of the ladder".
uv run python scripts/fetch_ghost_corpus.py --plan --tier rank1 \
    --per-team 40 --losses 10 --target 50 --min-teams 1 \
    --submissions <active-submission-id> --selection-out selection-rank1.json
uv run python scripts/fetch_ghost_corpus.py --use-selection <corpus>/meta/selection-rank1.json
uv run python scripts/census_ghost_corpus.py --tier rank1
```

`--selection-out` is what keeps the freeze honest when a tier is added later: the new tier
gets its own selection file and the existing `selection.json` is never rewritten. The
leaderboard download is likewise archived under its own timestamped name before it is
replaced, so a frozen selection's `snapshot` field still names a file that exists.

**Do not run the census while a fetch is in flight.** Both hammer the bulk disk; measured
2026-08-22, the fetch dropped from **21.7 to 1.3 episodes/min** with a census running
alongside, and recovered as soon as the census was stopped. Serialise them.

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

## The mirror cannot see the top of the ladder

Discovery (team → submission ids) uses the community mirror `georgymamarin/kaggriculture-episodes`.
It is fine in the middle and bottom of the leaderboard and **wrong at the top**. Measured
2026-08-22 on its 49,985-episode `episodes.csv`:

- the **rank-1 team appeared in 3 episodes**, the newest from 08-18, four days stale;
- the **highest rating anywhere in its last 24 hours was 2,124**, against a rank-1 LB score
  of **3,142** — the mirror's population simply does not contain the top clique;
- so discovery returned that team's **retired** submission (peaked ~1,505 and stopped playing
  on 08-19) rather than the one holding rank 1.

Nothing about that failure looks like a failure: the selection reports "40 wins found", every
replay is at the pin, and the corpus that comes out is a faithful clone of **the wrong agent**.
The tell is `rating_at_play` — compare it against the team's `lb_score` before fetching.

Two consequences, both now in the tool:

1. **`--submissions ID[,ID…]`** bypasses mirror discovery. For a top team, find the active
   submission id from an opponent's episode list: pull `ListEpisodes` for a frontier
   submission already in `index.csv` and read `opp_submission_id` where
   `opp_team_id` is the team you want. Four calls found it on 2026-08-22.
2. Selection **no longer stops at the first submission that fills the quota**. "The 40 most
   recent wins" is a statement about the *team*; ranking by date within one submission and
   stopping there silently selects whichever submission discovery listed first.

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
is never assumed from the fetch date.

**The low tiers are where this bites.** A rank-10 team plays constantly, so its "30 most
recent wins" all sit inside the last few days; a rank-3,000 team plays rarely, so its 30 reach
back past the 2026-08-15 engine bump. On the first field-tier run **38 of 120 selected
episodes came back 1.32.6** — one whole team's entire allocation — and each cost a 30 MB
download before it could be discarded. Selection therefore floors at `PIN_SHIPPED`
(`--since`, default 2026-08-16), which is free; a tier left short after discards is topped up
with `--top-up` rather than quietly shipped as a 3-team centroid wearing a 4-team label.

**Corpus as built, 2026-08-22** (LB snapshot `2026-08-22T02:56:57`): **352 episodes,
185.0 MB packed** (10.4 GB raw-equivalent), **3.70% of the 5 GB cap**, **100% engine 1.32.7**,
100% winning seats, 0 orphan files, 0 missing replays.

| tier | episodes | teams (LB rank) | `rating_at_play` | dates |
|---|---|---|---|---|
| `frontier` | 120 | 4 (2, 5, 8, 10) | 600–3,051 (μ 2,277) | 08-18 … 08-22 |
| `upper` | 120 | 4 (108, 126, 170, 187) | 1,947–2,572 (μ 2,221) | 08-20 … 08-22 |
| `field` | 112 | 4 (2828, 2829, 3100, 3136) | 676–1,655 (μ 963) | 08-15 … 08-22 |

`field` is 112 rather than 120 because team 16622993 had only 22 wins at the pin. Note
`rating_at_play` is the *submission's* ladder rating in that episode, not the team's LB score
(`lb_score`) — a rank-2 team's fresh submission starts at 600 and climbs, which is why the
frontier tier's floor is 600.

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
