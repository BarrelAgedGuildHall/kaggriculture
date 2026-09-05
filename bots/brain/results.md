# Brain v2 results ? original fertilizer valuation

2026-09-05 UTC; branch `feature/brain-economics`. `bots/brain/main.py` is exactly
`bots/brain_fertilizer/main.py`; v1 is frozen at `bots/brain_v1_frozen/main.py`.
No public-agent code, action tapes, replay-derived opening, identity routing, or
imitation objective is used. Engine constants and observations drive decisions.

## Result

Fertilizer is retained. It is reserved, purchased, carried and applied only when the
estimated extra harvest from its three-day lifetime exceeds 1.4 times its market
price plus $50 for labor/delivery. The seller keeps at most eight needed units.
Existing crop, labor and selling rules otherwise remain unchanged.

Held-out seeds 1100?1149, both seats, 100 games per matchup:

| Matchup | Wins | 95% Wilson interval | Mean bank | Opponent bank |
|---|---:|---:|---:|---:|
| Brain v1 vs V46 | 2/100 | 0.5?7.0% | $69,205 | $114,267 |
| Brain v2 vs V46 | 2/100 | 0.5?7.0% | $71,489 | $113,634 |
| Brain v2 vs Brain v1 | 89/100 | 81.4?93.8% | $80,587 | $77,466 |

Against V46, the same-seed, same-seat margin improves **$2,917/game**.
Bootstrap resampling the 50 seed pairs gives 95% interval **+$1,581 to +$4,282**.
This is an improvement over our own bot, not evidence of strong-ladder readiness.

## Component measurement

40 additional engine-instrumented games per arm, seeds 1100?1119, both seats:

| KPI | v1 | v2 |
|---|---:|---:|
| Strawberry units sold/game | 43.6 | 65.1 |
| Final bank | $68,178 | $71,224 |
| Fertilizer sales revenue | $14,045 | $13,418 |
| Productive actions | 41.29% | 41.84% |
| Moves/productive action | 1.38 | 1.35 |
| Wages/game | $5,669 | $5,669 |
| Plants becoming weeds/game | 4.9 | 6.5 |
| Final unsold marked value | $111 | $113 |

The plant-loss counter also includes lifespan decay; it is not a pure drought
measure. Increased losses warrant checking the fertilizer/water priority interaction.

## Rejected experiments

- Eight-animal herd: 0/12 vs V46, margin -$55,268 versus control -$44,894.
- Forecast crop allocation: 0/12, margin -$67,922. The forecast helped the opponent.
- Opportunistic adjacent deposits: held-out 0/100 vs V46; margin improvement
  $770, paired-seed CI [-$3,441, +$4,818]; 59/100 vs v1. Not integrated.
- Delivery detours: tuning 2/20 vs V46, margin -$40,441, without a convincing
  advantage over control -$40,698. Not advanced.

The first crop-forecast build failed all 12 games because Kaggle loads the last
callable and the new class was appended after `agent`. Moving registration before
`agent` fixed this; the reported 0/12 strategy result is the subsequent clean run.
Concurrent registration also collided in the arena registry. The JSON was recovered
and the original retained at `data/results/economics-registry-recovery-original.txt`.
All subsequent parallel experiments used separate arena directories.

## Verification

- Candidate screen, confirmation and KPI games: zero crashes/timeouts after the
  unrelated rejected forecast build was repaired.
- Standalone packaged 720-turn episode: both DONE, $101,811 vs random, maximum
  call 42.6 ms. Artifact: `dist/brain-fertilizer/main.py`.
- Contended arena candidate p99: 20.5 ms; max 563.3 ms. The harness's stricter
  400 ms max gate failed; no engine timeout occurred. The standalone result above
  was obtained separately with the current pinned interpreter, not a fresh venv.
- `uv run pytest -q`: 175 passed, 2 pre-existing ghost-corpus failures
  (`rank1` tier and index/file mismatch), as in the preceding cycle.
- Compilation and `git diff --check` pass. No separate lint configuration exists.
- GitHub access through the configured Git credential broker succeeded. Draft PR #37
  (`feature/original-brain-fertilizer`) contains only original Brain files. No direct
  change to main and no competition submission.


Evidence: `data/results/{fertilizer-confirm-100,brain-v1-confirm-100,
neardelivery-confirm-100}/results.json`, `economics-paired-deltas.json`, and
`data/results/kpi/{fertilizer-confirm-kpi-40,brain-v1-confirm-kpi-40}/kpi.json`.
The finite opening-allocation search is `scripts/search_brain_opening.py`:
10 policies (control plus 9 cells), seeds 20?29 in both seats, V46 opponent.
A winning cell still needs confirmation outside all seeds used above.

## Opening allocation follow-up (2026-09-05)

Ten original parameter settings played 200 tuning games, all without faults. The
best screen changed opening wheat/melon and later wheat/strawberry to 50/50 each.
On fresh seeds 1200-1249 (both seats) it beat v2 85/100, but lost 100/100 to V46
and worsened the reference margin by $10,324/game (seed-pair bootstrap CI
-$14,862 to -$5,674). Control v2 also went 0/100 versus V46, with a less negative
margin. The opening is rejected and the retained Brain source is unchanged.

V2's two fresh V46 bands now total 2/200 wins. Its 89/100 against v1 is real, but
should not be presented as evidence of high ladder strength. No new submission.

Evidence: `data/results/{opening-confirm-100,fertilizer-opening-control-100}/results.json`,
`data/results/opening-paired-delta.json`, and `data/results/brain-opening-search/`.
Next experiment: `scripts/search_brain_fertilizer_priority.py`, three task priorities
on fresh seeds 1300-1349, both seats, 100 games per arm against V46. It does not
change forecast values or crop allocation. Results remain pending.

## Autopilot cycle 1: labor and capital experiments

The retained Brain v2 source remains unchanged. Completed comparisons:

| Mechanism | Games | Result vs V46 | Decision |
|---|---:|---|---|
| Fertilizer priority 0 | 100 | 0 wins; margin delta -$58, CI -$1,124 to +$978 | Reject |
| Fertilizer priority 1 | 100 | 0 wins; delta -$258, CI -$1,236 to +$737 | Reject |
| Softer distance/priority routing | 100 | 0 wins; delta -$1,828, CI -$5,423 to +$1,657 | Reject; also 24/100 vs v2 |
| Distance before fixed worker ownership | 20 | 0 wins; screen margin improves $2,796 | Unconfirmed |
| Reward-per-distance job assignment | 20 | 0 wins; screen margin worsens $10,741 | Reject |
| Limit speculative empty pastures | 20 | 0 wins; screen margin improves $3,716 | Keep as search parameter, unconfirmed |
| Buy approved land before seeds/animals | 20 | 0 wins; screen margin worsens $6,801 | Reject |
| Land ordering plus pasture limit | 20 | 0 wins; screen margin worsens $14,649 | Reject |
| Keep late-season cap at 12 hands | 20 | 0 wins; delta +$523, CI -$307 to +$1,331 | Fresh-seed confirmation running |
| Raise peak cap to 14 | 20 | 0 wins; delta +$535, CI -$425 to +$1,489 | Unconfirmed |

All figures compare the same seeds and both seats with the unchanged control.
Confidence intervals resample whole seed pairs. There were no engine crashes or
timeouts; several local maximum-call checks exceeded the stricter 400 ms threshold
under concurrent load. Raw evidence: `data/results/autopilot1-paired-deltas.json`,
`late-labor-screen-deltas.json`, and the dispatch/capital/late-labor result folders.
The combined completed priority/dispatch/capital/late-labor comparisons total 820
games including their controls. New final-seed evidence for v2 vs V46 is 4/400 wins
across seeds 1100-1149, 1200-1249, 1300-1349 and 1400-1449.

Own-replay diagnosis (`data/results/brain-own-diagnostic/main_vs_main_seed51.json`):
on day 4 the starting quadrant had 15 plants, 2 animals and 8 empty pastures, with
$295 cash. Most observed plant-to-weed transitions were natural lifespan decay,
not drought. The farmer resets to the default spawn each night, so repositioning
him late to change the next morning's spawn was rejected from engine inspection.

A bounded reward-only evolutionary search is now running. It varies 15 parameters
of our own policy (crop/capital allocation, herd, hiring, feed, harvest threshold,
and care priority). Four generations of 16 policies use seeds 150-157, both seats,
against V46 and tapeselect. The objective balances final bank share across both
opponents. Four finalists and the control use seeds 1500-1519 for selection; the
selected policy and control then use fresh seeds 1600-1649, 100 games per opponent.
No imitation loss, replay action targets, public-agent source, or fingerprints are
used. `scripts/search_brain_evolution.py` has no automatic integration/submission.
100 sampled parameter combinations compile, and its control reproduces the frozen
v2 source exactly. The first 12 completed policies (384 games) had no faults and no
wins against either reference; their scores are training results, not improvements.

---

# Brain v1 results

Branch `bot/brain`; bot `bots/brain/main.py`. Closed-loop only: no tapes,
replay lookup, opponent identity keys, or recorded action sequences.

## Change

Integrated Labor B1 behind the `hire_count` / `dispatch` contract. The immutable
turn view now exposes empty livestock structures and the coordinator's complete job
graph, so the labor module routes Brain's existing feed, pickup/place, and shed jobs
instead of replacing them with its narrower rig job list.

- Hiring estimates daily work from plants, plantable seed/tiles, animals, urgent
  backlog, weeds, and pending animal setup. It buys the next hand only while its
  Fibonacci wage is below conservative marginal work value and cash reserve survives.
- Dispatch assigns coordinates to deterministic unit anchors (persistent compact
  clusters), executes an in-place task first, then routes by survival deadline,
  task priority, cluster ownership, and distance. Second-miss feed/water are emergency
  exceptions; normal order is harvest, water, plant/setup, care/feed/logistics, deposit.
- Unassigned units pre-position toward live work; they do not pass while a reachable
  productive job exists.

## Verification and arena

- Exact world replica, five seeds: sale revenue/inventory, town drain, and refreshed
  price maximum errors all **0** (`test_world.py`).
- Four 720-turn checks, Brain in both seats vs random and V46: **4/4 DONE**, zero
  crashes/timeouts. Standalone p99 **7.465 ms**, max **97.045 ms**.
- Reference starter, 64 paired games: **64/64 = 100%**, Wilson 95% CI
  **94.3-100%**; banks **$109,268.8 vs $3,544.3**, p99 **4.969 ms**,
  zero crashes/timeouts.
- Money sanity, 20 paired games (seeds 0-9): Brain **$65,640.8**, V46
  **$106,338.7**; Brain = **61.73%** of V46 (v0: 62.31%). Win rate **2/20 =
  10.0%**, CI **2.8-30.1%**.
- Required paired n=100 (seeds 0-49): **3/100 = 3.0%**, Wilson 95% CI
  **1.0-8.5%**; banks **$64,394.2 vs $108,973.2**, margin **-$44,579.0**;
  zero crashes/timeouts, p99 **41.254 ms**, max **886.421 ms**. This run overlapped
  another 16-worker arena job but still passed every harness gate.
- `ghost_eval` was not run because n=100 was below the requested 10% threshold.

## Same-seed KPIs (20 games each)

All three rows were re-extracted with the same action-aligned census on seeds 0-9.
Watered fraction is WATER actions per distinct observed plant-day.

| KPI | Brain v1 | Brain v0 | V46 |
|---|---:|---:|---:|
| Productive-action fraction | **41.00%** | 29.97% | 44.37% |
| Idle unit-slots / game | **104.8** | 1,181.2 | 244.2 |
| Watered fraction | 95.45% | 67.80% | 89.04% |
| Wages / season | **$5,661** | $7,452 | $7,530 |
| Unsold marked value | $73.70 | $89.30 | **$0** |
| Bank @72 | $513 | $468 | $352 |
| Bank @240 | $398 | $195 | $1,389 |
| Bank @480 | $23,892 | $28,327 | $46,434 |
| Bank @719 | $65,641 | $65,426 | $106,339 |

Full rows: `data/results/brain-v1-kpi-comparison/kpi_compare.json`. Arena reports:
`data/results/brain-v1-fault-timing-gate/`, `brain-v1-v46-n20/`, and
`brain-v1-v46-n100/`; starter report: `data/results/brain-v1-starter-n64/`.

## Verdict and next leak

Labor B1 is integrated and closes the idle-slot and wage leaks: idle is below the
300 target and wages below $6.5k. Productive rate improves **+11.0 pp** but misses
the 44% target by **3.37 pp**; its 95.45% water rate also indicates excess low-value
watering.

Money remains below 90% because labor quantity is no longer the binding KPI: v1 has
fewer idle slots and lower wages than V46, yet by turn 480 holds only **51.5%** of
V46's bank. The next biggest leak is **midseason task value / market realization**
(crop-livestock mix and selling), not more hands. Next test: integrate Team B's
quote-aware selling module, then use realized dollars per productive action to prune
safe-day watering and low-value routes.
