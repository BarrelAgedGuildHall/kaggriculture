# Brain v3: retain late-season workers

2026-09-05 UTC. The sole change from v2 removes the fixed worker-cap cuts on
days 25 and 28; hiring still requires observed work, affordable wages and cash
reserve. Source now skips discarded legacy dispatch work, with unchanged tested v3 actions;
its pre-optimization policy is frozen at `bots/brain_late_labor_keep12/main.py`. V2 remains
frozen at `bots/brain_fertilizer/main.py`. No imitation or opponent action data.

Fresh seeds 1700-1749, both seats, 100 games per row:

| Opponent | v3 wins | v2 wins | Paired bank-margin gain | 95% seed-pair bootstrap CI |
|---|---:|---:|---:|---:|
| V46 | 4 | 4 | +$901 | +$538 to +$1262 |
| tapeselect | 0 | 0 | +$667 | +$277 to +$1098 |

Retained for consistent small gains against independent references. Direct v2
self-play is worse (28/100 wins); reference wins are unchanged. This is not a
strong-ladder candidate and has not been submitted. Absolute v3 banks against
V46 and tapeselect are $73,808 and $59,517 respectively.

All 300 confirmation games had zero crashes or engine timeouts. The concurrent
harness's strict 400 ms maximum gate failed (max 680 ms); p99 remained below 29 ms.
Separate packaged 720-turn game vs random: both DONE, bank $135,422, maximum call
21.209 ms. Artifact `dist/brain-late-labor/main.py`, using the pinned current
interpreter in an isolated working directory, not a fresh virtual environment.

Evidence: `data/results/late-labor-{keep12,control}-confirm-100/results.json`,
`late-labor-{keep12,control}-tapeselect-100/results.json`,
`late-labor-confirm-delta.json`, and `late-labor-tapeselect-delta.json`.
Fixed historical top-team opponents: v3 and v2 each win 5/107 (4.67%). Bank-margin
means are -$53,485 and -$54,168: gain $684/game. Averaging within the 63 distinct
seeds gives gain $654/seed, bootstrap CI +$313 to +$1023. Both runs have zero
crashes/engine timeouts and p99 below 26 ms. Evidence:
`data/results/brain-v{2,3}-ghost-107/ghost_eval.json`, `brain-v3-ghost-paired.json`.
The first completed runs hit a Windows console encoding error before saving;
rerunning with `python -X utf8` saved identical game results. Those opponents are
benchmarks only; their actions never enter our strategy or training.

The short-cycle opening experiment played 160 games across control, all-wheat,
50/50 wheat/carrot, and all-carrot on seeds 180-189 against both references.
Every variant won 0/40; control won 2/40. Carrot variants did not improve both
reference margins. All-wheat gains were +$1021 vs V46 (CI -$13435 to +$14541) and
+$3527 vs tapeselect (CI -$5329 to +$12354): insufficient evidence, not retained.
All 160 games had zero crash/timeout flags. Evidence:
`data/results/brain-short-opening/results.json`, `short-opening-wheat-paired.json`.

`scripts/compare_paired.py` checks identical complete opponent/seed/seat sets and
reports whole-seed bootstrap intervals without dropping faulted games. Its actual
late-labor replay of stored results reproduces the previous $667.24 estimate and
[$277.17, $1097.77] interval exactly. Full suite remains 175 passed and the same two
external-corpus failures; compilation and whitespace checks pass.

---

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
| Keep late-season cap at 12 hands | 100 fresh | 4 wins; delta +$901, CI +$538 to +$1,262 | Checking second reference before promotion |
| Raise peak cap to 14 | 20 | 0 wins; delta +$535, CI -$425 to +$1,489 | Unconfirmed |

All figures compare the same seeds and both seats with the unchanged control.
Confidence intervals resample whole seed pairs. There were no engine crashes or
timeouts; several local maximum-call checks exceeded the stricter 400 ms threshold
under concurrent load. Raw evidence: `data/results/autopilot1-paired-deltas.json`,
`late-labor-screen-deltas.json`, and the dispatch/capital/late-labor result folders.
The combined completed priority/dispatch/capital/late-labor comparisons total 820
games including their controls. With the new late-labor control, final-seed evidence for v2 vs V46 is 8/500 wins
across seeds 1100-1149, 1200-1249, 1300-1349, 1400-1449 and 1700-1749.
The keep12 candidate also went 4/100 vs V46 but only 28/100 vs v2. Its $901
reference-margin gain is statistically supported; a matched 100-game comparison
against tapeselect is running for both arms before any promotion. Both arms had
zero crashes/timeouts, but their maxima under concurrent load exceeded 400 ms.

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

## Reward-only search final comparison

The completed search measured 52 unique tuning policies, five validation policies,
and two final policies (2464 games). The selected policy is
`bots/brain_search_0058/main.py`; it has not replaced v3.

Unseen seeds 1600-1649, both seats, 100 games per opponent: selected policy wins
0/100 against V46 and 0/100 against tapeselect. Compared with frozen v2, paired
bank-margin gains are +$2921 against V46 (95% seed-pair bootstrap CI -$1382 to
+$7212) and +$7610 against tapeselect (CI +$1873 to +$13627). Final candidate
and control have no crash/timeout flags. Candidate p99 is below 8 ms and maximum
208.229 ms. Evidence: `data/results/brain-evolution/{final.json,paired-final.json}`.

This supports a margin gain against one reference, with uncertainty against the
other and no wins. The comparison against current v3 was interrupted at the
founder's request; after explicit resumption it is being rerun with two workers
under `evolution-v3-resumed-final-100`. Do not treat that incomplete run as evidence.

## Resumed comparison against current v3

The full two-worker rerun completed 200 games on seeds 1600-1649: v3 wins 1/100
against V46 and 0/100 against tapeselect. Zero crashes/timeouts; p99 8.327 ms,
maximum 159.274 ms. Combined with the prior 1700-1749 band, v3 is 5/200 versus V46.

Selected search policy 0058 versus current v3: paired margin +$2099 against V46
(95% seed-pair CI -$2277 to +$6428) and +$7064 against tapeselect (CI +$1389 to
+$13007). The search candidate still wins 0/100 against each reference. It is not
promoted: a reference margin improves, but V46 improvement remains uncertain and
wins have not increased. Evidence `data/results/evolution-v3-resumed-final-100/`
and `data/results/brain-evolution/paired-v3-final.json`.

Next mechanism: feed distribution, holding herd size and economics fixed.
`scripts/experiment_brain_feed_delivery.py` tests control, one carrier, up to eight
carriers with smaller loads, and distributed pickup at an earlier priority. Each
gets eight seeds (190-197), both seats, against V46 and tapeselect: 128 games total,
two workers, sequential registrations. No promotion from a tuning result alone.

Diagnostic evidence from our existing v2 seed-51 replay: day20/hour23 has 13 of
18 animals still unfed while six workers carry 35 wheat. This is a pre-final-action
snapshot, not a complete daily starvation count. It supports investigating feed
routing and task deadlines; it does not establish causality. Census saved at
`data/results/brain-own-diagnostic/feed-census.json`. No opponent actions were
used to construct the policy.

## Feed-delivery screen and next deadline experiment

All128 screen games completed on eight paired seeds190-197 against both
references, without crash/timeout flags. No delivery variant is retained.

| Variant | V46 wins /16 | tapeselect wins /16 | Margin delta vs V46 | Margin delta vs tapeselect |
|---|---:|---:|---:|---:|
| Control | 0 | 0 | 0 | 0 |
| One carrier | 2 | 0 | -$19725 | -$9647 |
| Distributed | 0 | 0 | +$1164 | -$62 |
| Distributed, earlier pickup | 2 | 0 | +$559 | +$4177 |

One carrier is a clear margin regression: paired 95% seed-bootstrap intervals
[-$32081,-$6594] against V46 and [-$18390,-$1204] against tapeselect. Both intervals
for each distributed variant include zero. The earlier-pickup variant remains
unconfirmed despite two wins; no promotion from this small tuning cohort.
Evidence: `data/results/brain-feed-delivery/{results.json,concentrated-paired.json,
distributed-paired.json,distributed-early-paired.json}`.

Replaying only our own final unit actions before the daily refresh in saved v2
seed51 confirms 398 animal-days: 71.61% fed,69.35% both fed and cared. On day20
only5/18 animals received feed even after the final action. This is single-game
diagnostic evidence, not an evaluation cohort. The engine unit-action phase was
used directly; opponent actions were replaced with PASS for this own-farm census.
`data/results/brain-own-diagnostic/feed-final-census.json`.

`scripts/experiment_brain_feed_deadlines.py` tests feed priority raised at hour12,
feed at12 plus care at16, and feed at6 plus care at12. The two-day survival
emergency remains more urgent, delivery remains unchanged, and last-day useless
husbandry is still excluded. The byte-identical completed v3 control is reused;
three variants each run32 games against both references, seeds190-197, two workers.
It is running; results must be observed before conclusions or integration.

## Husbandry deadline screening

All96 deadline games completed, zero crash/timeout flags. Against the matched
control, feed12 changes margins -$6376 vs V46 /+$12554 vs tapeselect; feed6/care12
changes -$3628/+$6071. Both are rejected for weakening the V46 point estimate.

Selected feed12/care16:2/16 wins vs V46,0/16 vs tapeselect, control0/16 each.
Paired margin gains +$2659 against V46 (95% seed-pair CI -$6698 to +$11668) and
+$10917 against tapeselect (CI +$2966 to +$20452). This small tuning cohort
supports further testing, not promotion. `data/results/brain-feed-deadlines/`.

Fresh confirmation now runs through `scripts/confirm_brain_husbandry.py`: frozen
v3 and `bots/brain_feed_deadline_both12/main.py`, each100 games per reference,
seeds1800-1849, both seats, two workers sequentially. Total400 games. It writes
`data/results/husbandry-confirm/paired.json`. Only if both reference point margins
improve and at least one interval excludes zero, with no fault flags, does it
run40 instrumented games per arm against V46. No automatic integration/submission.

The existing engine-phase KPI extractor is now included in this branch with
animal-day, fed-animal-day and fed+cared-animal-day counters. These are captured
after final daily unit actions and before refresh; days without a refresh are
excluded. On the saved full720-turn seed51 replay it reproduces the independent
census exactly:398 animal-days,285 fed,276 fed+cared, bank$82827,DONE. Aggregate
fractions also match exactly. Evidence `feed-kpi-verified.json` in the own diagnostic
folder. Compilation/whitespace checks pass; no new full-suite claim is made.

## Prepared value-aware husbandry (not game-tested)

The existing seed51 replay has day20 quotes wheat$38,milk$229,wool$11; at day25,
wheat$39,milk$235,wool$1. Fixed equal husbandry priority ignores this difference.
Saved quote observations: `data/results/brain-own-diagnostic/market-census.json`.
The pinned engine confirms care is banked after production, scheduled production
clears pending bonuses, and two unfed days cause escape.

`scripts/experiment_brain_husbandry_value.py` builds three independent variants of
frozen v3: gate care by payoff; also reduce feeding to maintenance when the bonus
is uneconomic; add midday deadlines to economically useful husbandry. Care needs
an in-season payment date, pending-bonus room, and a current sale quote exceeding
half a wheat quote plus $35 labor allowance. Feeding protects a valuable pending
bonus on a production day and always preserves the existing survival-emergency
rule. There is no forecast trained from replay actions or opponent identities.

Build-only execution is complete. Each variant passed120 direct policy calls on
recorded own observations, bypassing the agent exception fallback, plus four
boundary checks for low-value wool, terminal care, saturated pending bonuses and
the production-day reset. These are functional checks, not wins or economic
validation. `data/results/brain-husbandry-value/offline-checks.json`.

No new game job was started alongside the active husbandry confirmation. Only
after that job (including any KPI runs) finishes, `--run` starts96 paired screen
games against the existing byte-identical v3 control on seeds190-197, two workers.
Any apparent gain still requires fresh-seed confirmation. Main bot is unchanged.

## Dispatch CPU check (not integrated)

The coordinator computes legacy dispatch even when LaborB1 replaces its complete
output. `scripts/benchmark_brain_dispatch_cost.py` creates a candidate that skips
this discarded matching work only under the exact existing B1-ownership condition;
the late final-day delivery path still uses the original logic.

Direct policy calls on1438 recorded observations across both seats return exactly
the same actions, with no mismatches or hidden exception fallback. Six interleaved
full-state CPU measurements have median1.578125s for frozen v3 and1.453125s for
the candidate:7.92% less policy CPU on this replay workload. Initial shorter batches
were too close to the CPU timer resolution; the full-state measurements supersede
them. This is not an arena throughput or win-rate claim.

Evidence `data/results/brain-dispatch-cost/results.json`; generated source
`bots/brain_fast_dispatch/main.py`. Retained source and running arena snapshots
are unchanged. A standalone full-game check remains required before integration,
after the active confirmation and any conditional component runs finish.

## Fresh husbandry confirmation

All400 paired games finished, seeds1800-1849 against both references. Candidate
and v3 each win2/100 against V46 and0/100 against tapeselect; no crash/timeout
flags. Candidate p99 7.354ms,maximum100.220ms. Candidate margin gains are +$2009
against V46 (95% seed-pair CI -$2747 to +$6723) and +$4610 against tapeselect
(CI +$1433 to +$7677). Own bank gains are $1334 and $557 respectively; much of the
tapeselect margin gain comes from lower opponent revenue. Wins have not improved.

`data/results/husbandry-confirm/paired.json` qualifies the policy for component
measurement, not automatic promotion. Control40 KPI games are complete; candidate40
KPI games are running. Retained v3 is unchanged and no submission was made.

## Original live submission diagnostic

Own submission56023381 metadata returned39 scored completed games,18 wins and
mean bank$64833. This is its observed low-rating cohort, not the strong local
benchmark. Latest inspected own episode105889015 completed720 turns, both DONE,
own statuses only ACTIVE/DONE, bank$47320 versus$124526. It does not show an engine
crash or timeout. Exact phase analysis reports productive-action fraction30.81%,
1109 idle unit-slots,$7434 wages,398animal-days,353fed,352fed+care, and zero plants
lost at the deadline. This is the older live v0 and a different board, so it is not
a paired comparison with v3. Evidence in `data/results/brain-live-diagnostic/`:
`episodes.json`, `episode-105889015-replay.json`, and `kpi.json`.

Authenticated leaderboard at2026-09-05 19:43UTC: original v0 rating637.3; historical
v47 2183.4; historical tapeselect2235.1. Leader Jesse Bullard2959.3, tenth2837.6.
Only our own replay was fetched for diagnosis; no actions enter strategy training.

## Completed husbandry component result and retained CPU improvement

Forty matched instrumented games per arm completed with no faults. Fed fraction
rose72.44% to81.09%; fed+cared fraction70.62% to80.15%. However, bank declined
$79004.575 to$77382.225 (-$1622.35) on this40-game subset. The arena database gives
exactly the same subset banks; this differs from the full100-game mean because it
covers only the first20 seeds, not because of instrumentation divergence.

Milk units154.525 to158.7 but milk income fell$2084.15; wool units150.6 to173.2 and
income rose$2469. Strawberry income fell$824.725 and wheat income$1363.525. Wages
fell$49.4; productive fraction40.66% to41.13%. Better husbandry alone has not
established a reliable broad economic or win improvement. The deadline candidate
is not promoted; its full-cohort positive tapeselect margin remains recorded.
Evidence `data/results/kpi/husbandry-{control,candidate}-kpi-40/kpi.json`.

The CPU-only candidate passed isolated packaged720-turn verification: both DONE,
bank$101315 vs random, mean1.277ms,max43.842ms. Current pinned interpreter, not a
fresh venv. With1438 action-equivalence checks and the measured7.92% policy CPU
reduction, the redundant-dispatch skip is now retained in `bots/brain/main.py`.
This changes runtime, not the tested strategy; no new win-rate claim. Frozen v3
remains available. Artifact `dist/brain-fast-dispatch/main.py`.

The previous controller has exited. Value-aware husbandry now runs with two
workers: `scripts/experiment_brain_husbandry_value.py --run`, three variants,
96 games, matched existing v3 control, isolated arena. No new submission.

## Value-aware husbandry screen (2026-09-05)
All 96 games completed without crash/timeout flags, seeds190-197 both seats.
Current quotes and exact production timing only; no imitation. Comparisons use
matched frozen v3 control from feed-delivery-control-screen-16.
- Care-only gating: 0/16 wins against each reference. Margin -$8706 vs V46
  (seed-pair CI -$19106 to -$756), +$4237 vs tapeselect (CI -$1319 to +$9482). Rejected.
- Maintenance feeding: 2/16 V46, 0/16 tapeselect. Margin +$2125 vs V46
  (CI -$9531 to +$12358), +$4526 vs tapeselect (CI +$1399 to +$7657).
- Value-aware deadlines: 2/16 V46, 0/16 tapeselect. Margin +$761 vs V46
  (CI -$9546 to +$10614), +$8136 vs tapeselect (CI +$3404 to +$13117).
  Own bank gains $4330/$4213. Selected for fresh confirmation, not promoted.
Evidence: data/results/brain-husbandry-value/{results.json,*-paired.json}.
Fresh confirmation: scripts/confirm_brain_value.py, seeds1900-1949, both seats,
100 games per opponent per arm; two workers, sequential arms. Conditional
40-game KPI runs per arm use the same positive-margin/no-fault gate as the
previous controller. No automatic promotion or submission.
