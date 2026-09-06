# Brain v4: value-based feeding and care

Retained 2026-09-05 after fresh paired confirmation and standalone verification.
Current source bots/brain/main.py matches bots/brain_value_v4/main.py.
Care must have a payable production date, spare bonus capacity and enough
current product value to cover feed and labor. Feeding conserves uneconomic
work while preserving survival and pending bonus payments; deadlines are
feed at hour12 and care at16. Observation/engine rules only, no imitation.

Fresh seeds1900-1949, both seats,100games per opponent per arm:
- V46: v4 8/100 vs v3 2/100; paired margin +$6447.26,
  seed-pair CI +$2407.71 to +$10559.25; own bank +$3336.40.
- Tapeselect: both0/100; paired margin +$6174.12,
  CI +$2720.08 to +$9633.73; own bank +$2251.72.
All400games had no crash/timeout flags. Candidatep99 6.793ms,max94.158ms.
V46 gained6 wins and lost none relative to control, but whole-seed win-rate
bootstrap interval is0 to14 percentage points. Margin evidence is clearer.
This is a local improvement, not demonstrated strong-ladder performance.

Matched40game V46 economic subset: bank71287 ->75226.45 (+$3939.45),
fed fraction72.98% ->72.99%, fed+cared71.22% ->55.92%, wages6945.55 ->6934.25.
Productive-action fraction40.91% ->39.74%; action count is not economic value.
Milk units157.8 ->136.5,wool148.025 ->152.825,wheat316.4 ->322.925.
Both80 KPI games had zero faults. Controller81517 fully exited.
Evidence data/results/husbandry-value-confirm/{paired.json,paired-wins.json}
and data/results/kpi/husbandry-value-{control,candidate}-kpi-40/kpi.json.

Retained CPU optimization added to measured policy with1438/1438 identical
saved-state actions. Packaged dist/brain-value-v4/main.py passed720turns,
bothDONE, bank98379 vs random,mean1.286ms,max60.188ms. Current pinned
interpreter; random is unseeded, so its bank is not a paired benchmark.
No submission: current original live rating637.3 at last prior20:01 reading;
v4 still far weaker than the historical active benchmark in local tests.

Next: reachable-work screen running against v4,64games,two workers,
seeds210-217,both seats; no automatic promotion. Herd caps remain a separate
prepared experiment and must rebuild from v4 before testing.

## Historical v3: retain late-season workers

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

## Prepared demand-aware herd trial (2026-09-05)
Own-action purchase census found one requested animal purchase on/after day19
in the saved v2 seed51 game (one sheep day26), and none in the inspected v0
live episode105889015. These are requested orders, not verified purchases;
two games cannot establish a population rate. Late cutoff alone is lower
priority than herd allocation. Evidence brain-own-diagnostic/animal-purchase-census.json.

Prepared scripts/experiment_brain_herd_demand.py: keep seven-cow cap, limit
sheep cap to min(11, floor + 4 * revealed YARN_STORE count), floor2 or5.
Duplicate shops counted, existing animals retained, no opponent actions used.
Lower herd targets also release future pasture reservations for crops.
This is an experimental capacity rule, not a claim of optimal shop allocation.
Build-only: control and both variants each passed120 direct policy calls on
our saved observations; each variant passed four shop-count boundary checks.
No arena games launched while value confirmation is active. --run will use
fresh matched control plus both candidates, seeds200-207, both seats against
V46/tapeselect,96games total,two workers sequentially. Rebuild from the retained
champion with --baseline if the active value confirmation changes it.
Evidence data/results/brain-herd-demand/offline-checks.json.

## Cycle 8: fresh control and nightly reachability diagnostic
Fresh1900-1949 control COMPLETE200games:2/100 V46,0/100 tapeselect,
no crash/timeout flags,p99 6.071ms,max95.937ms. Banks72095.9 vs113921.8
(V46),63665.4 vs124728.8(tapeselect). Retained v3 now9/400 V46 (2.25%)
on four disjoint seed bands1600-1900,0/400 tapeselect. Candidate arm running;
no comparison or promotion yet. data/results/husbandry-value-control-1900-n100/results.json.

Authenticated API20:01:44UTC: original v0 637.3, historical v47 2182.7,
tapeselect2235.1; leader Jesse Bullard2948.6,tenth2834.7. No submission.

On current v3 evaluated against our saved v2 seed51 observations, tracing
709 B1 calls found180 fallback moves among6630 unit actions, all beforeday11.
Separately,501/6369 task assignments had travel distance exceeding the number
of remaining moves before a final action at hour23. This is observational
policy diagnostics, not a full-game causal estimate or win improvement.
Evidence data/results/brain-own-diagnostic/{dispatch-fallback,unreachable-work}.json.
Prepared scripts/experiment_brain_reachable_work.py: rejects travel assignments
that cannot execute before night; same filter on fallback travel. Exact
on-tile actions remain available at hour23. Hands vanish and farmer resets
nightly, so unreachable late travel has no next-day positioning value.
Passed719 direct policy calls and4 boundary checks (exact task at final hour,
one-step task with/without enough time, empty graph). No game run yet.
Evidence data/results/brain-reachable-work/offline-checks.json.
Rebuild with --baseline if retained champion changes. After active controller
finishes, screen this independently from the already prepared herd caps.

## Cycle 9: reachability verification and paired runner
Added optional --run to scripts/experiment_brain_reachable_work.py:64games,
control/candidate each16 per V46/tapeselect, seeds210-217,both seats,two
workers sequentially, isolated data/arena-reachable-work. Rejects existing
result tags, checks subprocess exit, compares exact paired seed/seat sets.
No automatic promotion. Not launched while husbandry confirmation runs.

Tracing both policies on identical own saved observations verified zero
unreachable assignments in candidate vs501 control. Control6369 assignments,
candidate6073; fallback moves180 ->192,709 dispatcher calls each. This is a
mechanism check, not measured full-game productivity or win improvement.
Evidence data/results/brain-reachable-work/assignment-check.json;
reproducer artisans/ceo/bench/diagnose_reachable_work.py (workspace root).
Build-only checks still pass719 policy calls and4 boundaries. Whitespace clean.
Purchase diagnostic now explicitly uses observation[t] and action[t+1];
requested-order counts unchanged. Reproducer artisans/ceo/bench/animal_purchase_census.py.
Active husbandry-value candidate has90/200 games observed at this update;
no paired result available and retained champion unchanged.

## Reachability screen against v4 (completed cycle 10)
64games complete, seeds210-217 both seats, zero faults. Both policies0/16
wins against each reference. Margin gain +$2260 V46 (seed-pair CI -$4783 to
+$8395), +$974 tapeselect (CI -$3838 to +$7320). Own bank +$5375/+$6579,
but opponent bank also rose. No promotion: reference margins uncertain.
Evidence data/results/brain-reachable-work/paired.json. Session28951 exited.
Herd-demand screen now running from retained v4,96games,two workers,
fresh matched control,seeds200-207,both seats. No combination with reachability.

## Herd-demand screen and broader v4 benchmark (cycle 11)
All96games finished without crash/timeout flags; all policies0/16 wins vs
both references, seeds200-207 both seats. No herd variant promoted.
- Sheep2: own bank +$9942 V46/+$8917 tape; margin +$1206 V46
  (seed-pair CI -$6583 to +$9032), -$3243 tape (CI -$15266 to +$6629).
- Sheep5: own bank +$934 V46/+$7593 tape; margin +$5719 V46
  (CI -$1444 to +$11678), +$1653 tape (CI -$12161 to +$12841).
Opponents also benefited in several comparisons; higher own bank alone does
not establish competitive improvement. These measurements do not by themselves
identify the mechanism. data/results/brain-herd-demand/{sheep2,sheep5}-paired.json.
Controller56292 exited. V4 unchanged.

Started107game v4 historical-opponent benchmark with2workers, session26616,
scripts/ghost_eval.py --agent bots/brain_value_v4/main.py --workers2
--tag brain-v4-ghost-107 (use spaces after option names). Opponent tapes remain
black-box benchmarks, not strategy inputs/training examples. No result yet.
Added scripts/compare_ghost_results.py: asserts identical complete opponent,
seed and seat sets, retains faults and clusters bootstrap by seed. Reproduced
existing v3-v2 result exactly:683.663551/game,653.753968/seed,
CI313.468254..1022.976190,107games63seeds. Evidence brain-v3-ghost-reproduced.json.

## Completed v4 broader benchmark and tomato screen (cycle 12)
Historical opponent benchmark completed107/107:5wins102losses, unchanged
wins from v3, zero crashes/engine timeouts/local >1s turns. Mean bank70529.3,
mean margin -48332.3; p50 1.619ms,p99 4.167ms. Different worker counts from
historical v3 prevent attributing timing differences to policy changes.
Matched result: margin +$5152.327103/game over v3;63-distinct-seed mean
+$5694.206349,seed-bootstrap CI +$2954.920635 to +$8475.396825.
Supports retaining v4, but still loses102/107 historical games. No live submission.
Evidence data/results/brain-v4-ghost-107/ghost_eval.json and brain-v4-ghost-paired.json.
Session26616 fully exited.

Added scripts/experiment_brain_tomato.py. Two engine-supported crop allocations
replace10 or20 percentage points of post-bootstrap wheat with tomato; strawberry
and melon shares stay25%/20%. Bootstrap unchanged. Tomato seed cap24.
Only original observation-driven policy; no opponent actions or imitation.
Control and both candidates each passed120 direct policy calls; candidates
passed4 allocation-boundary checks each. Build-only results in
 data/results/brain-tomato/offline-checks.json.
Screen running from v4:96games, fresh seeds220-227,both seats,V46/tapeselect,
two workers sequentially. Automatic paired reports, no automatic promotion.

## Completed tomato screen and expansion test (cycle 13)
96games complete, zero crashes/timeout flags, all policies0/16 wins vs each
reference on seeds220-227 both seats. No allocation promoted.
Tomato10 margin +$1528 V46 (seed-pair CI -1090..4557), -$748 tape
(CI -10290..10133). Own bank -2543/+3898.
Tomato20 margin +$3477 V46 (CI -1358..8584), -$1667 tape
(CI -11362..11075). Own bank -2028/+5401.
Evidence data/results/brain-tomato/{tomato10,tomato20}-paired.json.
Controller64778 fully exited; v4 unchanged.

Prepared/started scripts/experiment_brain_expansion.py from v4. Land variant
raises target from3 to4quadrants atday12, permits third land purchase only
throughday18, keeps existing $5200 cash threshold and first-two-land policy.
Labor variant additionally raises the phase/config ceiling12 ->14 hands;
existing work, marginal value and affordability gates still control actual
hires. This does not claim14 hands will be hired or profitable.
Control and variants120policy calls each; variants4phase/cap checks each.
Four modified-observation checks verified BUY_LAND present day18 and absent
day19 for both variants with3ownedquadrants,$10000,seed/feed stock.
Evidence data/results/brain-expansion/{offline-checks,purchase-boundaries}.json.
Screen running96games,seeds230-237,both seats,V46/tapeselect,two workers
sequentially,session27979. No automatic promotion or submission.

## Expansion rejected; shared-price screen (cycle 14)
All96expansion games complete,zero faults,allpolicies0/16wins vs each reference.
Fourth land: margin -$9428 V46 (seed-pair CI -14985..-4556), -$6384 tape
(CI -10142..-2831). Own bank -2434/+2372.
Fourth land + higher labor ceiling: margin -$11860 V46 (CI -18660..-6825),
-$7556 tape (CI -11843..-3556). Own bank -1203/-1923.
Allfour margin intervals belowzero; both rejected. No causal attribution to
land versus changed market effects from these game totals alone.
Evidence data/results/brain-expansion/{land,labor}-paired.json;27979 exited.

New script experiment_brain_market_pressure.py uses public rival animal tiles
and exact engine prices. Estimates visible held product plus base production
and existing pending bonuses over3days,capped at24units per product. Adds
half/full estimated rival-revenue reduction from our added unit to the v4
care/feed value. Floor-price sales add no stock and get zero pressure value.
This assumes some rival output sells after our added unit; it is a testable
approximation, not an exact forecast or copied action. No opponent identity.
Both variants120policy calls and4boundary checks passed. Full weight matched
v4 actions on120modified observations with no rival tiles,zero mismatches.
Evidence data/results/brain-market-pressure/{offline-checks,zero-exposure-check}.json.
Running96games from v4,fresh seeds240-247,both seats,V46/tapeselect,two workers
sequentially,session2729. Automatic paired reports,no automatic adoption.

## Shared-price screen and diagnostic follow-up (cycle 15)
96games complete,zero faults. V4 control4/16 V46 wins; half/full pressure
variants2/16 each. All0/16 tape. Neither variant promoted.
Half margin -$1223 V46 (seed-pair CI -2893..77), +$895 tape (CI -442..1937).
Full margin -$524 V46 (CI -2528..1318), +$65 tape (CI -2410..1979).
Evidence data/results/brain-market-pressure/{half,full}-paired.json.2729 exited.

Independent engine-phase reconstruction of our inventories verified all29
nightly shed and bank states in each of two saved games. Overflow losses:
v2 seed51 lost9units (5carrot,1melon,3wheat); older livev0 lost29units
(9fertilizer,16wool,3milk,1strawberry). Not current-v4 cohort evidence or
realized-money loss. Lower-priority lead. data/results/brain-own-diagnostic/overnight-loss.json;
reproducer artisans/ceo/bench/diagnose_overnight_loss.py. No strategy training.
V4 on saved own v2 observations:84duplicate travel destinations among3699
travel assignments across719calls. Lower-priority lead; no game effect claim.
Evidence duplicate-travel.json; reproducer artisans/ceo/bench/diagnose_duplicate_travel.py.

Returning to reachable-work for a larger fresh confirmation. Its earlier
8-seed screen had positive margins both references but intervals crossingzero;
this is uncertainty resolution, not adoption. Rebuilt from v4 and rechecked
719policy calls/4boundaries. Frozen candidate bots/brain_reachable_work/main.py.
New scripts/confirm_brain_reachable.py:100games/opponent/arm,400total,
fresh seeds2000-2049,both seats,two workers sequentially. Conditional40 KPI
games per arm (20seeds x2seats) only if both margins positive, at least oneCIpositive,
and no faults. No automatic promotion/submission. Rejects existing result tags
and checks subprocess exits. Session53698 running from20:47UTC.

## Lean-watering preparation and action diagnostics (cycle 16)
Exact own unit-action census: savedv2 seed51 only5ineffective PICKUP actions
plus109PASS; olderlivev0 only1109PASS unchanged. No failed movement in either.
Immediate reversals:158/1811consecutive move pairs(v2),62/2336(v0). These older
single-game diagnostics do not establish a current-v4 cohort bottleneck.
Evidence data/results/brain-own-diagnostic/{unit-action-census,route-reversals}.json;
reproducer artisans/ceo/bench/diagnose_unit_actions.py. Engine permits movement
onto LOCKED tiles, so no path restriction was added.

Prepared experiment_brain_lean_water.py. Early variant removes forced hourly
watering before first yield; lean removes forced watering whenever there is
no survival emergency, non-ongoing yield gain, or payable fertilizer bonus.
Preserves planting-day survival (new plants start consecutive_unwatered=1),
all yield-window watering and fertilized ongoing-production ticks.
Both120policy calls and9fertilizer timing/capacity boundaries passed.
Ten isolated engine fixtures (5crops xfertilized/unfertilized) preserved total
harvest under daily,early,lean schedules with actual WATER/HARVEST/refresh calls.
Examples: wheat5 ->4water actions,same4/6yield; strawberry17 ->9lean,same4/8yield;
melon11 ->8unfertilized or7fertilized,same6yield. Continuous fertilizer is a
fixture condition, not free fertilizer in the bot. Immediate labor availability
and common harvest schedules in fixtures do not prove full-game gains.
Evidence data/results/brain-lean-water/{offline-checks,engine-growth-checks}.json;
reproducer artisans/ceo/bench/verify_lean_water_growth.py.
Optional --run ready:96games,control/early/lean,seeds250-257,both seats,
V46/tapeselect,two workers sequentially. NOT launched while reachability
confirmation53698 runs. Rebuild from champion if it changes before the screen.

## Reachability confirmation complete (cycle 17)
400games finished,zero crash/timeout flags. Freshseeds2000-2049,both seats:
V4 3/100 V46,0/100tape; reachable4/100 V46,0/100tape. Candidatep99 6.751ms,
max103.762ms. Margin +$3330.11 V46 (seed-pair CI -1517.56..8144.25),
+$874.18 tape (CI -2117.20..3843.00); own bank +1137.61/+1201.12.
No promotion: intervals still includezero. Component gate not met, so no KPI
jobs ran. Controller53698 fully exited. V4 retained11/200 held-outV46,
0/200tape across1900/2000bands,5/107historical wins.
Evidence data/results/reachable-work-confirm/paired.json and
 data/results/reachable-work-{control,candidate}-2000-n100/results.json.

Started prepared lean-water screen from retained v4,session45976 at21:04UTC:
96games,control/early/lean each16per V46/tapeselect,seeds250-257,both seats,
two workers sequentially. Source rebuilt and policy/boundary checks executed
by runner. No overlap with prior controller, no automatic promotion.

## Watering screen and fresh confirmation (cycle 18)
All 96 screen games completed without crashes/timeouts. All arms won 0/16
against each reference on paired seeds250-257, both seats.
Early: margin +3463.50 V46 (CI -13581.38..14534.25), +11048.69 tapeselect
(CI -64.06..23725.81); own bank +11655.56/+9321.56.
Lean: margin +3538.81 V46 (CI -13492.88..14562.44), +10776.50 tapeselect
(CI -1315.13..24310.31); own bank +11810.44/+7843.69.
Neither adopted; intervals include zero. Evidence:
data/results/brain-lean-water/{early,lean}-paired.json.
Conservative early variant selected for fresh confirmation because its
combined margin is slightly higher and it removes fewer watering actions.
Frozen bots/brain_early_water/main.py; scripts/confirm_brain_early_water.py
runs 400 games on seeds2100-2149, both seats, two workers sequentially,
against retained v4 and both black-box references. Conditional component
checks follow only if both margins improve and one interval excludes zero.
No automatic promotion or submission. Controller started 21:10UTC.

## Prepared egg production screen (cycle 19)
Independent mechanism: replace premium cow/sheep output with earlier goose
production. Test caps12 and18 against retained v4; use COOP construction in
both dispatch paths. No opponent actions or identities enter the strategy.
Both variants and control passed120 policy calls. Variant outputs buy only
geese and never request pasture construction on these saved own states.
Actual engine fixture built a coop, placed a goose and verified seven daily
feed/care/harvest refreshes: held output0,0,0,4,2,2,2 and8 harvested eggs.
Fixture supplies immediate labor/feed, so it establishes mechanics, not profit.
Evidence data/results/brain-eggs/{offline-checks,engine-checks}.json;
fixture reproducer artisans/ceo/bench/verify_goose_growth.py.
Prepared scripts/experiment_brain_eggs.py (build-only default). Optional --run
screens96 games, seeds260-267, both seats, V46/tapeselect, two workers
sequentially. NOT started while watering confirmation7316 is active. Rebuild
from the retained champion before running if watering changes that champion.

Two saved original games also checked for terminal stock: older v2 carried
one fertilizer; live v0 carried3milk/1wool, with zero sellable shed stock in
both. This is not a v4 cohort and does not rule out terminal losses elsewhere.
Evidence data/results/brain-own-diagnostic/terminal-stock.json.

## Conservative watering confirmation complete (cycle21)
400 games on fresh seeds2100-2149,both seats,zero crashes/timeouts.
Candidate8/100V46 versus v4 4/100;both0/100tapeselect. Candidatep99 5.659ms,
max102.29ms. Paired margin V46+$2600.36 (CI -3458.38..7988.27),tapeselect
-$1123.57 (CI -5045.10..2626.64). Own bank +6159.07/-9.65.
No promotion: neither margin interval excludes zero and tape mean is negative.
Conditional component criteria were not met; no additional KPI games ran.
Evidence data/results/early-water-confirm/paired.json and individual
 early-water-{control,candidate}-2100-n100/results.json. Controller7316 exited.
V4 retained with pooled15/300V46,0/300tapeselect,5/107historical wins.
Started the prepared egg screen from unchanged v4 after the controller exited:
96games,seeds260-267,both seats,two workers sequentially,control/egg12/egg18.
Each policy passed120 saved-state calls again at launch. No automatic promotion.

## Egg screen rejected; daytime delivery trial (cycle22)
All96 egg games completed without crashes/timeouts; all policies0/16 against
each reference on seeds260-267,both seats. Egg12 margin -26202.06V46
(CI -40998.31..-11350.38),-23577.25tape(-41632.13..-8115.00).
Egg18 -28894.63V46(-41782.19..-16696.81),-28386.31tape(-48601.56..-10465.38).
All four intervals exclude zero negatively. Both rejected, v4 retained.
Evidence data/results/brain-eggs/{egg12,egg18}-paired.json;62699 exited.

New independent short-delivery mechanism: units carrying goods with at least
$250 expected sale value may detour to a shed within1 or3 steps throughhour18.
Exclude wheat carriers, unplaced animals and fertilizer carriers when crop
fertilizer is reserved. Reserve shed capacity across proposed deliveries.
Use own market-price/sale functions; sale occurs after shed observation updates.
No opponent action or identity input. This sacrifices work for earlier sales;
there is no measured benefit yet. Source scripts/experiment_brain_day_delivery.py.
Both variants passed120 policy calls then719 full saved-state calls;near1
changes15states,near3 changes70. Late-hour/final-day actions remain identical.
Evidence data/results/brain-day-delivery/{offline-checks,action-checks}.json.
Started96game screen after egg controller exited:seeds270-277,both seats,
control/near1/near3 each16perV46/tapeselect,two workers sequentially.

## Daytime delivery screen complete (cycle23)
All96games completed without faults,all policies0/16 per reference.
Near1 margin V46+$1262.00(CI -6151.06..10091.06),tapeselect+$3274.75
(CI733.00..6474.38);own bank+6916.06/+2083.13.
Near3 margin V46+$43.06(CI-3779.94..4335.75),tape+$4530.44
(CI1800.06..7179.75);own bank+1338.25/+450.50.
Neither promoted. Near1 chosen for confirmation: shorter detours, greater
own-bank gain and larger V46 mean gain; combined reference margin gains
are nearly tied. This is screen-based selection, requiring fresh evaluation.
Evidence data/results/brain-day-delivery/{near1,near3}-paired.json.
Frozen bots/brain_day_delivery/main.py; scripts/confirm_brain_day_delivery.py
started after screen92328 exited.400games,seeds2200-2249,both seats,
controlv4/candidate each100perV46/tapeselect,two workers sequentially.
Conditional component runs only if both margin means positive and one
paired interval excludes zero, with no faults. No automatic adoption/upload.

## Delivery override diagnostic after resume
On719 saved observations from our older v2 game, near1 changes17 unit actions
across15 states:1care,2water,3pickup,1fertilizer collection and10 moves.
No displaced watering action had consecutive_unwatered>=1. Near3 changes91
unit actions across70states and displaces6 survival-watering actions. These
are counterfactual policy calls, not actual losses or a measured v4 cohort.
The active near1 confirmation remains unchanged. This supports checking
work displacement before trying longer delivery detours again.
Evidence data/results/brain-day-delivery/override-census.json; reproducer
 artisans/ceo/bench/diagnose_delivery_overrides.py. Original observations only.

## Prepared joint dispatch (resumed cycle2)
Independent mechanism: assign workers jointly within each job priority group,
minimizing total cluster departures then distance. Preserve original on-tile
work, inventory eligibility, priority order, and fallback behavior. Dummy
assignments permit idle workers where no eligible job exists. No opponent
actions, identities, or imitation inputs. Baseline retained v4.
Own rectangular assignment solver matched exhaustive optima on70 small
random matrices (including negative costs), plus empty input.719 saved own
observations passed,114 changed states,p99 1.568ms,max18.462ms under arena
contention. This is functional/algorithm validation, not game performance.
Evidence data/results/brain-joint-dispatch/{offline-checks,assignment-checks}.json;
reproducer artisans/ceo/bench/check_joint_dispatch.py.
Prepared scripts/experiment_brain_joint_dispatch.py; default build-only.
Optional --run:64games,control/joint each16perV46/tapeselect,seeds280-287,
both seats,two workers sequentially. NOT running while delivery81011 runs.
Rebuild from retained champion if it changes before this screen starts.

## Prepared value-aware wheat pickup (resumed cycle3)
Found v4 feeding jobs use survival/value gating while pickup demand still
counts every unfed animal. Separate candidate applies the same eligibility
to need_wheat; survival feeding remains included. Purchases, feed decisions,
and carrier distribution remain as in v4. No opponent action/identity inputs.
719 policy calls pass,15changedstates;696 checks show pickup demand positive
exactly when feeding jobs exceed carried wheat and shed wheat is available.
Summed requested pickup quantities817 ->652 on saved older own observations;
this counts proposed jobs across states, not executed pickups or economic gain.
Evidence data/results/brain-feed-pickup/{offline-checks,job-checks}.json;
reproducer artisans/ceo/bench/check_feed_pickup.py.
Prepared scripts/experiment_brain_feed_pickup.py,build-only default. Optional
--run screens64games,seeds290-297,both seats,control/value each16perV46/tape,
two workers sequentially. NOT running. Queue behind delivery confirmation
and prepared joint dispatch; rebuild from retained champion when started.

## Delivery confirmation complete (resumed cycle3)
400freshgames completed,seeds2200-2249,both seats,zero crashes/timeouts.
Candidate and v4 each0/100 against both references. Candidatep99 6.289ms,
max98.033ms. Margin +$990.40V46(CI -3522.44..5369.25),+$1294.80tapeselect
(CI -591.34..3238.37). Own bank -1385.57/+11.29. Not promoted: both intervals
include zero. Component criteria not met, no KPI games ran.81011 fully exited.
Evidence data/results/day-delivery-resumed-confirm/paired.json. Interrupted
44games excluded. V4 retained15/400held-outV46,0/400tape,5/107historical.
Started prepared joint-dispatch screen from unchanged v4 after81011 exited;
64games,seeds280-287,both seats,two workers sequentially. Both policies
passed120 saved-state calls again at launch. Value-aware pickup remains queued.

## Joint dispatch and pickup screens complete (resumed cycle4)
Joint:64games,zero faults,all arms0/16perreference. MarginV46+$6302.75
(CI83..12699.50),tape-$5436.44(-11782.50..1320.13);ownbank-2300.81/-3387.44.
Not promoted: mixed reference performance and lower ownbank.37988 exited.
Pickup:64games,zero faults,all arms0/16perreference. MarginV46+$4223.50
(CI-6266.44..15420.88),tape+$1419.94(-5057.56..9886.19);ownbank+8204.19/+1775.
Positive means but wide intervals; unconfirmed, not promoted.22364 exited.
Reports data/results/brain-joint-dispatch/joint-paired.json and
 data/results/brain-feed-pickup/value-paired.json.

Storage correction: day-delivery, joint-dispatch and feed-pickup screens
used data/arena-eggs due to a copied runner default. Their unique tags and
sequential execution kept results separate. Read-only checks verified all
seven32game arms, identical pair sets within each experiment and zero faults.
Evidence data/results/brain-day-delivery/arena-location-checks.json.
Historical rows stay in that arena. Defaults now use separate directories;
existing result guards still prevent overwriting tags. Confirmations already
used their own correct directories. No bot policy changed in this correction.

Started joint-travel screen after22364 exited:96games,seeds300-307,both seats,
control/free/soft each16perV46/tape,two workers sequentially. Uses the tested
joint solver, with territory penalty0 or4 instead of1000. This tests interaction
of joint matching and territory cost; prior softer greedy assignment differed.
Both120policy calls pass. Baseline v4 unchanged, no imitation inputs.
Scripts/experiment_brain_joint_travel.py; data/arena-joint-travel and
 data/results/brain-joint-travel. No automatic promotion/submission.

## Joint-travel screen complete; pickup confirmation (resumed cycle5)
96games,zero faults,seeds300-307,both seats. Control and soft each2/32wins;
free0/32. Free marginV46+1457.44(CI-6100.56..9087),tape-6394.31
(-17809.75..4625.44);ownbank+3094.44/+10316.25. SoftmarginV46+2364.63
(-4315.94..7461.50),tape+2378(-825.06..5663.31);ownbank+1322.50/-3147.81.
No adoption: intervals include zero and free loses reference wins.
Evidence data/results/brain-joint-travel/{free,soft}-paired.json.23868 exited.

Selected value-aware pickup for fresh confirmation: corrects a specific
feeding/pickup mismatch and had positive ownbank/margin means against both
references in its screen. This is screen-based selection, not confirmation.
Frozen bots/brain_feed_pickup/main.py; scripts/confirm_brain_feed_pickup.py
runs400games,seeds2300-2349,both seats,controlv4/candidate each100perreference,
two workers sequentially. Conditional component measurements only if both
margins positive and one interval excludes zero, nofaults. No auto adoption.
Started after23868 exited. V4 retained15/400held-outV46,0/400tape.

## Prepared earlier crew investment (resumed cycle6)
Observed retained-v4 component aggregate: four hands/day through day6,
mean bank$368.20 atstep240(day10),$25150.43 at480,$75226.45 terminal.
Evidence data/results/kpi/husbandry-value-candidate-kpi-40/kpi.json; this
is a40game measured cohort, not a causal proof that labor is constrained.
Test higher desired hire floors6/8 beforeday10 under existing phase caps,
wage-value and cash-reserve checks. Initial plan cap5 still applies, so the
variant names do not promise6/8 actual hands. Day10 onward unchanged.
Both719policycalls pass:floor6 changes23states,34extra requested hires;
floor8 changes27states,68extra requested hires across older saved own states.
Each1438 zero-cash/late-hour checks pass; later-day actions identical.
These are counterfactual requests, not executed hires or economic gain.
Evidence data/results/brain-early-crew/{offline-checks,hiring-checks}.json;
reproducer artisans/ceo/bench/check_early_crew.py.
Prepared scripts/experiment_brain_early_crew.py; build-only default.
Optional --run:96games,seeds310-317,both seats,control/floor6/floor8 each16per
V46/tapeselect,two workers sequentially. NOT running while pickup40068 runs.
Rebuild from retained champion if it changes. No imitation inputs.

## Care-demand idea parked after functional checks (resumed cycle8)
Prepared scripts/experiment_brain_care_demand.py to price care bonuses using
half/full revealed town consumption through the next bonus date, capped at
three days. Pending immediate production retains current quote. No opponent
sales/actions/identities modeled; future player supply is unmodeled, so this
is an optimistic hypothetical input, not a dependable price forecast.
648consumption boundary checks perarm passed (duplicates,unknownshops,
interval endpoints,zerohorizon);719policycalls each passed with finalday
identical. Both variants changed ZERO actions on the older own replay.
Parked without an arena run due to no observed action effect on this sample.
This does not prove no effect elsewhere. Early-crew remains next experiment.
Evidence data/results/brain-care-demand/{offline-checks,forecast-checks}.json;
reproducer artisans/ceo/bench/check_care_demand.py.

## Pickup confirmation complete (resumed cycle8)
400games,seeds2300-2349,both seats,zero crashes/timeouts. Candidate4/100V46
versusv4 6/100,both0/100tapeselect. Candidatep99 6.058ms,max95.908ms.
MarginV46+$3253.19(CI-565.36..7221.67),tape+$352.05(-2434.23..3179.74);
ownbank+610.49/+2643.45. Not promoted: intervals include zero and no win lift.
Conditional component criteria not met; no KPI games ran.40068 exited.
Evidence data/results/feed-pickup-confirm/paired.json and individual reports.
V4 retained21/500held-outV46,0/500tape,5/107historical wins.
Started prepared early-crew screen from unchanged v4 after40068 exited:
96games,seeds310-317,both seats,two workers sequentially;control/floor6/floor8
120saved-state calls each passed again. No automatic adoption/submission.

## Early crew rejected; fresh v4 cash diagnostic (resumed cycle9)
Earlycrew96games,zero faults,all arms0/16perreference. Floor6marginV46
-318.56(CI-14343.06..13636),tape-3614.31(-10053.31..3653.81);ownbank+8869.31/+3761.50.
Floor8marginV46+4042.63(-4564.25..12208.19),tape-4374.13(-9683.75..755.31);
ownbank+7643.50/-577.31. Neither promoted.44880 exited.
Evidence data/results/brain-early-crew/{floor6,floor8}-paired.json.

Fresh currentv4 diagnostic seed2400,both seats sequentially: both ownbank51918
versus84198,ownDONE,productive fraction38.51%,wages6977. Two games on one
seed, not a strength estimate or added to held-out pooled result. Saved full
records under data/results/brain-v4-diagnostic/seat{0,1}.json and own KPI files.
Bench/measure_v4_diagnostic.py reproducer;81024 exited before any new arena.
Exact engine market replay verified all719owncash transitions in seat0:
initial3000 +grosssales78549 -expenses29631 =51918. Wheat purchases5214,
wages6977,land3000,animals8300,seeds6140. Wool sales5353 vs sheep capital5500;
fertilizer and feed/labor attribution are not included in that comparison.
Do not infer sheep unprofitable solely from wool. Cash-ledger.json contains
own operation/day breakdown; bench/reconcile_v4_cash.py reproduces.

Started cow-heavy18animal screen:12cow/6sheep or18cow versus7cow/11sheep v4.
Same herd ramp/pasture type; separates product mix from herd contraction and
previous goose trial. Both719current-v4 saved-statecalls pass,368/439changed
states;30day ramp/cap/housing checks perarm pass. Data/results/brain-milk-mix/
policy-checks.json;bench/check_milk_mix.py. No opponent actions/identity inputs.
Scripts/experiment_brain_milk_mix.py --run:96games,seeds330-337,both seats,
two workers sequentially,control/milk12/milk18 each16perV46/tapeselect.
No automatic adoption/submission. V4 retained unchanged.

## Milk mix rejected; ongoing crop batching (resumed cycle10)
96milk-mix games,zero faults,all arms0/16perreference. Milk12marginV46+12
(CI-9586.75..11029),tape-1290.69(-9236.19..6412.63);ownbank-2153.75/-1919.75.
Milk18marginV46-16771.94(-29582.69..-3738.63),tape-9458.63(-18095.81..1725.44);
ownbank-21897.19/-16740.88. Neither adopted; all-cow clearly worseV46.
10374 exited. Data/results/brain-milk-mix/{milk12,milk18}-paired.json.

Own v4 diagnostic product-demand.json shows seed2400 town demand354strawberry,
ours68sold; final strawberry price178 versus milk/wool/fertilizer1. This is
one seed, not generalized unmet demand (rival supply also matters). Crop-work.json
counts46strawberry harvest requests for68sold,31fertilize requests;13plant requests,
4unused seeds. Requests are not verified successful actions in this census.

New bounded batching for ongoing crops: require2or3held units, but flush at
last production age, liquidation, or before next production would overflow.
Conservatively assume fertilized next yield if fertilizer active, even if water
is not yet applied. Other crops unchanged. Actual engine fixtures for tomato/
strawberry,with/without continuous fertilizer, preserve4/8units. Batch3 cuts
harvest actions4->2 in all four fixtures;batch2 cuts only unfertilized4->2.
Fixture labor/watering/fertilizer supplied immediately; not game-profit proof.
Data/results/brain-crop-batch/engine-checks.json; bench/check_crop_batch.py.
Scripts/experiment_brain_crop_batch.py --run started after10374 exited:
96games,seeds340-347,both seats,two workers,control/batch2/batch3 each16per
V46/tapeselect.120saved-state calls each passed. V4 retained, no auto adoption.

## Crop batching rejected; earlier boosted wheat harvest (resumed cycle11)
96games,seeds340-347,both seats,zero faults;all arms0/16perreference.
Batch2 marginV46+51.75(CI-1517.88..1460.63),tape+718.88(-826.13..2499.06);
ownbank-14.13/-235.81. Batch3 marginV46-1516(-2424.81..-591.06),
tape-835.88(-2596.63..1279);ownbank-729.25/-666.25. Neither adopted.
Evidence data/results/brain-crop-batch/{batch2,batch3}-paired.json.
Earlier four actual engine fixtures preserved4/8units while reducing harvest
calls, but this did not yield a demonstrated full-game gain. V4 retained.
Fresh v4 harvest-timing.json census:0capped mature crops before target age;
174plant-observation snapshots of five-unit wheat before target age. These
are repeated observations, not unique plants, and motivate the next screen.


Session9722: uv run python scripts/experiment_brain_boost_harvest.py --run.
Started23:52UTC after crop-batch46002 exited.64games,seeds350-357,both seats,
two workers,control/boost5 each16perV46/tapeselect. Harvest mature wheat at
five units before target age only if the quote after6additional inventory >30.
Heuristic: five units atday3 vs six atday4 frees land earlier; rate threshold
assumes60combined seed/labor cost and unchanged prices. Not profit proof.
719fresh ownv4 observations pass,38changed actions. Actual engine fixtures
with continuousfert verify day3 five units vs day4 six. No imitation.
Arena data/arena-boost-harvest; logs,engine-checks.json and eventual
boost5-paired.json in data/results/brain-boost-harvest. No automatic adoption.

## Early harvest result and crop retirement (resumed cycle12)
Early wheat harvest64games,seeds350-357,both seats,zero faults. Candidate/control
both2/16V46,0/16tape. MarginV46+731.19(CI165..1346.88),tape-462.38
(-1770.25..692.69);ownbank+687.63/-568.94. Candidatep99 5.199ms,max102.547.
Mixed reference result, no win lift; not promoted or expanded. Session9722 exited.
Evidence data/results/brain-boost-harvest/boost5-paired.json.
Crop-horizon variants require2/3harvest events left before planting ongoing
crops, and remove corresponding seed purchases.30day plan-boundary checks perarm
and18actual engine harvest-count fixtures pass. However719currentv4 observations
each produce0changed actions. Parked without games, not an economic result.
Scripts/experiment_brain_crop_horizon.py;data/results/brain-crop-horizon/engine-checks.json.
Retained v4 unchanged21/500held-outV46,0/500tape; screen seeds excluded.

Session60520: uv run python scripts/experiment_brain_crop_retirement.py --run.
96games,seeds370-377,both seats,two workers,sequential control/stop/clear arms,
each16games perV46/tapeselect. Stop watering ongoing crops once final production
has arrived; clear additionally schedules DIG after all held yield is collected.
Original engine-derived mechanism; references remain black-box opponents.
719ownv4 saved-state calls perarm pass;17/28changed states. Four actual engine
fixtures tomato/strawberry,with/withoutcontinuousfert preserve4/8units with no
water on retirement day and verify emptied tile DIG. Immediate-service fixtures
do not establish full-game gains. Arena data/arena-crop-retirement; logs and
engine-checks.json in data/results/brain-crop-retirement. No auto adoption.

## Crop retirement result and unique travel (resumed cycle13)
96games,seeds370-377,both seats,zero faults,all arms0/16perreference.
Stop marginV46-640.88(CI-1420.13..97.25),tape-235.44(-723.75..207.13),
ownbank-487.50/-89.88. Clear marginV46-547.69(-1128.25..29.31),
tape+214.75(-560..975),ownbank-559.75/+221.56. Neither adopted.
Data/results/brain-crop-retirement/{stop,clear}-paired.json;60520 fully exited.
Fixture savings did not translate into demonstrated game gains. V4 retained.
Exact engine own-unit census on seed2400 validates all719cash transitions:
4189movement actions,126intentional PASS,only3unchanged PICKUP actions. Allother
requested unit actions changed state. This is one game, not population evidence.
Data/results/brain-v4-diagnostic/noops.json;bench/diagnose_v4_noops.py.
Motivates avoiding redundant travel rather than treating general invalid commands
as the throughput problem. Reference actions used solely for exact engine phases,
never action imitation or strategy fitting.

Session45190: uv run python scripts/experiment_brain_unique_travel.py --run.
Started00:02UTC after60520 exited.64games,seeds380-387,both seats,two workers,
sequential control/unique,each16perV46/tapeselect. Preserve existing simultaneous
on-tile jobs, then reserve each target against duplicate travel assignments.
Remove fallback travel toward remaining jobs that require unavailable supplies.
Original task-matching mechanism, no opponent action or identity routing.
Six targeted assignment checks pass,719currentv4 saved-state calls pass with139
changed actions. Arena data/arena-unique-travel;data/results/brain-unique-travel/
contains dispatch-checks.json,logs and eventual unique-paired.json. No auto adoption.

## Unique travel result and priced fertilizer collection (resumed cycle14)
64games,seeds380-387,both seats,zero faults. Candidate2/16V46 vscontrol0,
both0/16tapeselect. MarginV46+3060.13(CI-5485.50..11253.25),tape-1921.94
(-7251.44..2885.94);ownbank-116.56/-408.31. Candidatep99 4.926ms,max126.359.
Both intervals include0 and references disagree; not promoted or expanded.
45190 fully exited. Data/results/brain-unique-travel/unique-paired.json.
V4 retained21/500held-outV46,0/500tapeselect; screen seeds excluded.

Session28814: uv run python scripts/experiment_brain_fertilizer_collection.py --run.
Started00:06UTC after45190 exited.96games,seeds390-397,both seats,two workers,
sequential control/floor10/floor30,each16perV46/tapeselect. Suppress fertilizer
collection when conservative sale quote <=10/30 and shed+carried stock covers
max(2,current profitable fertilizer-job reserve); no buffer on terminalday29.
Existing fertilizer investments and job ordering preserved. No imitation.
719currentv4 saved-state calls each pass;30/92changed states. All653/534states
with high quote or stock below2 preserve exact baseline actions. Own diagnostic
364collections includes52atquote<=10 and132atquote<=30; quotes at observation,
not realized sale prices. Data/results/brain-fertilizer-collection/policy-checks.json.
Arena data/arena-fertilizer-collection; logs and eventual floor10/floor30-paired.json
under same result directory. No automatic adoption or submission.

## Positive fertilizer screen and fresh confirmation (resumed cycle15)
96games,seeds390-397,both seats,zero faults,all arms0/16perreference.
Floor10 marginV46+1409.94(CI680.50..2252.88),tape+1731.50(771.25..2778.25),
ownbank+1201.63/+1565. Floor30 marginV46+1900.88(85.31..4218.63),
tape+2416.38(453.75..4752),ownbank+1744.19/+1449.75. Floor30p995.025ms,
max102.233. Both margin intervals exclude0 for both variants; neither wins.
Choose floor30 for larger mean reference gains; its superiority to floor10
is not established. Fresh confirmation handles screen selection uncertainty.
Data/results/brain-fertilizer-collection/{floor10,floor30}-paired.json;28814 exited.
Mechanism: skip collection at quote<=30 when currentstock coversmax(2,profitable
crop-use reserve);terminaldaybuffer0. Existing fertilizer investment unchanged.

Prepared animal-harvest value30/value90 variants using heldyield times conservative
post-harvest inventory quote.719currentv4 calls each pass,89/96changed states;
452/448states without eligible low-value harvest retain exact baseline actions.
Own90harvest census:50heldvalues<=30,58<=90. Quotes,not realized proceeds.
No games started; prioritize fertilizer confirmation. Source
scripts/experiment_brain_animal_harvest_value.py;data/results/brain-animal-harvest-value/
policy-checks.json. Both preserve feeding/care rules; no economic gain established.

Session77840: uv run python scripts/confirm_brain_fertilizer_collection.py.
Started00:11UTC after28814 exited.400fresh games,seeds2500-2549,both seats,
two workers sequential control/candidate,each100perV46/tapeselect. Frozen
candidate bots/brain_fertilizer_collection/main.py equals screen floor30.
Retained control bots/brain_value_v4/main.py. Arena data/arena-fertilizer-collection-confirm;
logs and eventual paired.json under data/results/fertilizer-collection-confirm.
If both margins improve and at least one seed-pair CI excludes0, with zero
crash/timeout flags, runner follows with40V46KPI games each(80total) on2500-2519.
No automatic promotion or submission. V4 remains best pending confirmation.

## Collection supply checks (resumed cycle16)
scripts/check_brain_collection_jobs.py passes1438own saved observations from
both currentv4 diagnostic seats:326job lists filtered,184actions changed.
Candidate job list exactly equals baseline minus eligible fertilizer-collection
jobs; all other jobs and ordering preserved.21synthetic inventory/stock boundary
cases pass,including2where stock>=2 but below profitable crop-job reserve.
Those cases retain collection and exact baseline actions. This verifies the
functional guard, not realized crop supply or full-game economic benefit.
Evidence data/results/fertilizer-collection-confirm/job-checks.json.
Confirmation77840 continues unchanged:control70/200completed at00:14UTC,
candidate not started. No competing game job; two workers. No decision from
partial runs. Prepared animal-harvest screen remains unrun.

## Outcome review prepared (resumed cycle17)
scripts/compare_paired_outcomes.py reads actual arena result0 rather than inferring
outcomes from money margins. Ties counthalf, faults remain included. It validates
complete matching seed/seat sets and reproduces aggregate evaluator win score.
Verified on two completed32+32game comparisons: unique travel gains2V46wins,
0losses,score difference+12.5pp but whole-seedCI0..37.5pp includes0; tape0.
Fertilizer floor30 screen gains0wins vsboth despite positive money margins.
Evidence brain-unique-travel/paired-outcomes.json and
brain-fertilizer-collection/paired-outcomes.json under data/results/.
After confirmation completion run this reviewer with arena
data/arena-fertilizer-collection-confirm, candidate/control tags
fertilizer-collection-{candidate,control}-2500-n100 and output
data/results/fertilizer-collection-confirm/paired-outcomes.json.
Current77840control140/200 at00:17UTC; candidate pending. No competing game,
no promotion, no upload. V4 retained while fresh confirmation finishes.

## Confirmation control complete; action metrics verified (resumed cycle18)
Control200/200completed:3/100V46,0/100tape,zero crash/timeout flags,
p996.637ms,max94.520. Ownmeanbank75393.4/61471.1; opponents111006/117815.8.
Result data/results/fertilizer-collection-control-2500-n100/results.json.
V4 pooled now24/600V46,0/600tapeselect; completed control band2500 added once.
Candidate phase of77840 started00:19UTC,200games pending; no extra arena.

KPI analyzer now includes per-operation successful_unit_actions, from existing
exact engine unit-call outcomes, and their mean over games. Both future
conditional component runs will use this same analyzer. Neither bot changed.
Replay validation preserves all45old fields exactly on actualv4 seed2400.
New counts match independent exact census:364COLLECT_FERTILIZER,51FERTILIZE,
134successful PICKUP(137requested),4189moves; all17operationcounts match.
Two-row aggregate preserves same means. Data/results/brain-v4-diagnostic/
kpi-action-checks.json;bench/check_kpi_action_counts.py. This lets us distinguish
less collection from less profitable fertilizer use if confirmation passes.

## Running now
Session77840 now executing conditional KPI phase of
scripts/confirm_brain_fertilizer_collection.py after400arena games completed.
80component games:control then frozen candidate,40V46games each,seeds2500-2519
both seats,two workers. Control10/40 at00:28UTC. Per-operation action metrics
enabled and verified for both runs. Tags fertilizer-collection-{control,candidate}-kpi-40;
reports under data/results/kpi/. No concurrent game job.
Frozen bots/brain_fertilizer_collection/main.py equals screen floor30. V4 retained
pending KPI review and standalone game. Staged67,201-byte dist/
brain-fertilizer-collection-review/main.py exactly matches candidate and compiles;
staging ran no game and no upload. Package verification waits until77840 exits.

## Fertilizer fresh confirmation complete (resumed cycle20)
400games,seeds2500-2549,both seats,zero crashes/timeouts. Candidate5/100V46
vscontrol3/100;both0/100tape. V46margin+1259.04(CI714.94..1803.54),
tape+650.07(-51.49..1374.09);ownbank+1465.84/+1528.76. Candidatep995.425ms,
max109.166. Both means positive,one clear interval:preplanned KPI phase triggered.
Actual-outcome paired review:4V46wins gained,2lost,score+2pp with seed-pairCI
-4..8pp includes0. No established win-rate lift. Tape0difference. Economic
evidence strongest vsV46; tape margin remains uncertain. No automatic promotion.
Data/results/fertilizer-collection-confirm/{paired,paired-outcomes}.json.
V4 control band2500 already counted:24/600V46,0/600tapeselect. Do not doublecount.

## V5 retained after component and standalone review (resumed cycle21)
bots/brain/main.py matches bots/brain_value_v5/main.py and frozen
bots/brain_fertilizer_collection/main.py. V4 plus priced fertilizer collection:
skip at conservativequote<=30 when stock coversmax(2,profitable crop-job reserve),
terminaldaybuffer0. All other job types/order unchanged on matched observations.
Fresh2500-2549 both seats:5/100V46 vs v4 3/100,0/100tapeselect for both.
MarginV46+1259.04(seed-pairCI714.94..1803.54),tape+650.07(-51.49..1374.09).
Ownbank+1465.84/+1528.76.4winsgained,2lost;win-scoreCI-4..8pp includes0.
All400games zero faults;candidatep995.425ms,max109.166. Economic improvement
established vsV46; tape margin uncertain. Do not transfer v4's pooled wins to v5.
80KPI games completed,all40+40banks exactly match confirmation subset.
Bank69986.875->71093.65(+1106.775,CI366.45..1918.45);collections350.9->243.975;
applications46.125->34.575. Fewer fertilizer applications, not preserved usage.
Crop units618.275->634.1,all five crop unit means nondecreasing(4increase,tomato0).
Wages6943.8->6947.4;productivefraction39.48%->38.15%. Counts are not value.
Evidence data/results/fertilizer-collection-confirm/{paired,paired-outcomes,component-review}.json
and data/results/kpi/fertilizer-collection-{control,candidate}-kpi-40/kpi.json.
Standalone dist/brain-fertilizer-collection-review/main.py matches frozen candidate;
720turns,bothDONE,bank108623 vsunseededrandom,mean1.273ms,max40.721ms.
Random bank is not paired performance. Review passed;v5 promoted locally only.
V5 unsubmitted, not strong-ladder ready. V4 retained frozen for comparison.

## Running now
Session75558: uv run python scripts/experiment_brain_animal_harvest_value.py --run.
Started00:35UTC after77840 fully exited and standalone review passed.96games,
seeds400-407,both seats,two workers,sequential control/value30/value90 arms,
each16perV46/tapeselect. Built from newly retained v5,not v4. Minimum held-yield
salevalue30/90 for animalharvest; other jobs unchanged. No automatic adoption.
719own savedv4 observations perarm pass againstv5 reference;91/99changedactions,
452/448states without low-value jobs preserve exact v5 actions. Census remains
from v4's actual game,not new v5 games. Arena data/arena-animal-harvest-value;
logs,policy-checks.json and eventual paired.json under data/results/brain-animal-harvest-value.
Next: assess this screen, then broaden v5 reference evaluation if no new candidate.

## Animal-harvest result and broader v5 evaluation (resumed cycle22)
## Running now
Session7465: uv run python -X utf8 scripts/ghost_eval.py --agent
bots/brain_value_v5/main.py --ghost-dir D:/GuildhallData/kaggler/datasets/
kaggriculture/tapes/top10-2026-09-04 --workers 2 --tag brain-v5-ghost-107.
Started00:39UTC after75558 fully exited.107fixed historical games; all filenames
match v4's episode/seat set. Black-box opponents only; no imitation or identity
routing in our policy. Log data/results/brain-v5-ghost-107/run.log, eventual
ghost_eval.json. No concurrent game. V5 retained unchanged and unsubmitted.
On completion: scripts/compare_ghost_results.py --candidate
data/results/brain-v5-ghost-107/ghost_eval.json --control
data/results/brain-v4-ghost-107/ghost_eval.json --out data/results/brain-v5-ghost-paired.json.
Reviewer matches episodes,seats,seeds,teams,recorded banks; clusters by sharedseed,
retains faults. Actual prior v4-v3 result reproduced exactly:107games,63seeds,
margin+5152.327gameweighted,+5694.206seedweightedCI2954.921..8475.397,0wins gained/lost.
Evidence data/results/brain-v4-ghost-paired-recheck.json. Fixed ghosts do not
establish live ladder strength. Do not treat multiple episodes as independentseeds.

## Animal harvest screen completed (resumed cycle22)
96games,seeds400-407,both seats,zero faults,all arms0/16perreference.
Value30 marginV46-426.44(CI-2056.94..793.69),tape+733.50(-220.63..1802),
ownbank+1922.50/+228.38. Value90 marginV46+851.50(-1380..3543),
tape-59.38(-1098.06..1062),ownbank+757.25/-502.56. No promotion or confirmation.
All intervals include0 and references disagree. V5 remains retained.
Data/results/brain-animal-harvest-value/{value30,value90}-paired.json;75558 exited.

## Historical result and active-reference evaluation (resumed cycle23)
## Running now
Session8991: uv run python scripts/evaluate.py --agent bots/brain_value_v5/main.py
--opponents bots/v47/main.py bots/tapeselect/main.py --n 100 --seed-base 2600
--workers 2 --tag v5-active-reference-2600-n100.
KAGGRI_ARENA_DIR=data/arena-v5-active-reference. Started00:44UTC after7465 exited.
200fresh games,seeds2600-2649,both seats; one experiment,two workers. Historical
v47 is the stronger latest-two submission reference; both opponents black-box.
No source/actions copied or routed into v5. Direct test of replacement readiness,
not a new policy or a promotion. Results data/results/v5-active-reference-2600-n100/results.json.
Do not combine v47 wins with V46 wins. V5 remains unchanged and unsubmitted.

## V5 historical evaluation complete (resumed cycle23)
107games,63distinctseeds:5wins,102losses,0ties,matching v4 with0winsgained/lost.
Zero crashes,engine timeouts,local>1s turns. Meanbank71405.72,meanmargin-47771,
p501.587ms,p993.943ms. No causal speed claim from separate timed runs.
Paired v5-v4 margin+561.308gameweighted;seedweighted+480.5,CI-77.214..1074.325
includes0. Ownbank+876.411gameweighted;seedweighted+771.627,CI228.532..1329.873.
Bank improves, broader margin remains uncertain; no new win evidence or ladder claim.
Exactepisode/seat/seed/team/recordedbank matches verified;faults retained.
Data/results/brain-v5-ghost-107/ghost_eval.json and brain-v5-ghost-paired.json.
7465 fully exited. V5 retained based on earlier fresh V46 economic improvement;
direct active-reference measurement now addresses submission replacement risk.

## Nearby urgency prepared (resumed cycle24)
Current direct v47/tapeselect benchmark8991 continues,90/200completed at00:49UTC.
No result inferred from partial games. Retained v5 and submission unchanged.
Prepared scripts/experiment_brain_nearby_urgency.py: exact on-tile work of priority
>=2 yields its automatic precedence if an eligible priority<=0 job is within
1or2steps and no other available eligible worker is already on that target.
Urgent on-tile jobs remain first; existing remote matcher handles released workers.
Independent original dispatch idea; no opponent source/actions/identity features.
Seven assignment cases perarm pass(nearby,covered,urgent-on-tile,inventory,
distance boundary,nonurgent).719own savedv4 observations againstv5 each pass;
210/274changedstates,all10final-liquidationstates identical. No game gain claimed.
Data/results/brain-nearby-urgency/dispatch-checks.json;bench/check_nearby_urgency.py.
Build-only completed. After8991 exits and its replacement-readiness result is
recorded, run --run for96games,seeds410-417,both seats,two workers,sequential
control/near1/near2 each16perV46/tapeselect. Arena data/arena-nearby-urgency.
Do not overlap arenas or promote from offline checks.

## Replacement benchmark result and urgency screen (resumed cycle25)
## Running now
Session63652: uv run python scripts/experiment_brain_nearby_urgency.py --run.
Started00:53UTC after8991 fully exited.96games,seeds410-417,both seats,two workers,
sequential control/near1/near2 each16perV46/tapeselect. All built from frozenv5.
On-tile priority>=2 jobs relinquish automatic precedence when eligible urgent
priority<=0 work is1/2steps away and not covered by another available eligible
worker already at that target. Existing remote assignment and urgent-on-tile
work preserved. Seven focused checks and719own observations perarm already pass.
Arena data/arena-nearby-urgency; logs and eventual near1/near2-paired.json under
data/results/brain-nearby-urgency. No overlapping arena or automatic promotion.
Next:review complete paired results;confirm promising candidate on fresh seeds.

## Active-reference benchmark finished (resumed cycle25)
200fresh games,seeds2600-2649,both seats:V5 0/100v47 and0/100tapeselect.
Zero crashes/timeouts,p995.433ms,max101.831. Meanownbank68516vs122776(v47),
66600vs123129(tape). Moneymeans rounded; exact summary in
data/results/v5-active-reference-2600-n100/results.json.8991 fully exited.
This does not support replacing the stronger historical active submission;
no upload. V5 remains locally best on earlier economics,not competitive on this
direct test. Added100tapeselect games once to V5 heldouttotal0/200;v47 tracked
separately fromV46. #1 objective not achieved; no founder input needed now.
