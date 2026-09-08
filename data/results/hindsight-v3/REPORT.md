# Hindsight Solver v3 — Many-draw contested

Engine `1.32.7`; seeds 5005, 5033, 5043, 5072, 5026, 5038, 5024, 5064, 5063, 5090; one alternating seat per seed; live opponent `bots\v46\main.py`. Objective: mean paired margin.

## Panel results

| strategy | mean margin | worst margin | mean our bank | mean opponent bank |
|---|---:|---:|---:|---:|
| brain_strawberry baseline | $-34,407 | $-47,542 | $78,795 | $113,202 |
| hindsight-v3 best | $-34,407 | $-47,542 | $78,795 | $113,202 |

| strategy | seed | seat | our bank | opponent bank | margin |
|---|---:|---:|---:|---:|---:|
| brain_strawberry baseline | 5005 | 1 | $43,189 | $74,117 | $-30,928 |
| brain_strawberry baseline | 5033 | 1 | $61,932 | $104,389 | $-42,457 |
| brain_strawberry baseline | 5043 | 1 | $80,816 | $118,708 | $-37,892 |
| brain_strawberry baseline | 5072 | 0 | $80,278 | $107,018 | $-26,740 |
| brain_strawberry baseline | 5026 | 0 | $88,877 | $132,772 | $-43,895 |
| brain_strawberry baseline | 5038 | 0 | $60,717 | $97,687 | $-36,970 |
| brain_strawberry baseline | 5024 | 0 | $70,006 | $95,489 | $-25,483 |
| brain_strawberry baseline | 5064 | 0 | $96,837 | $116,966 | $-20,129 |
| brain_strawberry baseline | 5063 | 1 | $119,628 | $151,664 | $-32,036 |
| brain_strawberry baseline | 5090 | 0 | $85,671 | $133,213 | $-47,542 |
| hindsight-v3 best | 5005 | 1 | $43,189 | $74,117 | $-30,928 |
| hindsight-v3 best | 5033 | 1 | $61,932 | $104,389 | $-42,457 |
| hindsight-v3 best | 5043 | 1 | $80,816 | $118,708 | $-37,892 |
| hindsight-v3 best | 5072 | 0 | $80,278 | $107,018 | $-26,740 |
| hindsight-v3 best | 5026 | 0 | $88,877 | $132,772 | $-43,895 |
| hindsight-v3 best | 5038 | 0 | $60,717 | $97,687 | $-36,970 |
| hindsight-v3 best | 5024 | 0 | $70,006 | $95,489 | $-25,483 |
| hindsight-v3 best | 5064 | 0 | $96,837 | $116,966 | $-20,129 |
| hindsight-v3 best | 5063 | 1 | $119,628 | $151,664 | $-32,036 |
| hindsight-v3 best | 5090 | 0 | $85,671 | $133,213 | $-47,542 |

## Best plan

Exact raw bots/brain_strawberry/main.py behavior (plan-space anchor).

## Per-day bank (ours)

| day | baseline | best |
|---:|---:|---:|
| 0 | $936 | $936 |
| 1 | $413 | $413 |
| 2 | $484 | $484 |
| 3 | $149 | $149 |
| 4 | $246 | $246 |
| 5 | $309 | $309 |
| 6 | $258 | $258 |
| 7 | $206 | $206 |
| 8 | $384 | $384 |
| 9 | $454 | $454 |
| 10 | $42 | $42 |
| 11 | $964 | $964 |
| 12 | $476 | $476 |
| 13 | $1,702 | $1,702 |
| 14 | $8,123 | $8,123 |
| 15 | $11,434 | $11,434 |
| 16 | $16,308 | $16,308 |
| 17 | $20,607 | $20,607 |
| 18 | $23,313 | $23,313 |
| 19 | $24,708 | $24,708 |
| 20 | $28,672 | $28,672 |
| 21 | $30,331 | $30,331 |
| 22 | $36,042 | $36,042 |
| 23 | $43,138 | $43,138 |
| 24 | $48,054 | $48,054 |
| 25 | $52,006 | $52,006 |
| 26 | $57,422 | $57,422 |
| 27 | $61,814 | $61,814 |
| 28 | $68,126 | $68,126 |
| 29 | $78,795 | $78,795 |

## Sensitivity

| parameter | correlation with mean margin |
|---|---:|
| `sell.FERTILIZER` | -0.495 |
| `last.STRAWBERRY` | -0.433 |
| `hands.d25_29` | -0.421 |
| `last.WHEAT` | -0.363 |
| `last.CARROT` | -0.278 |
| `fertilize.WHEAT` | -0.271 |
| `sell.STRAWBERRY` | -0.252 |
| `mix.days_23_26.WHEAT` | -0.246 |
| `mix.days_23_26.MELON` | +0.238 |
| `hands.d0_4` | +0.231 |
| `fertilize.STRAWBERRY` | -0.192 |
| `mix.opening.MELON` | +0.182 |

Associations are observational within the correlated evolutionary population.

## Search efficiency

The resumable log contains 592 attempts: 581 completed panels, 10 pruned, and 1 raced out. It ran for 106.0 cumulative minutes and executed 5851 games (0.920 games/s). The best first appeared at completed evaluation 1; gains over the final 100 and 400 completed panels were $0 and $0. Final heat 0.250; prune slack $1,000,000,000; race slack $1,000,000,000; worker cap 8.

The cumulative 106.0 minutes comprise the parity/search smoke and the uninterrupted 100-minute production run. The production run used eight workers. The day-18/day-23 pruning and fixed two-seed racing machinery remained active; after the smoke showed that the corrected raw incumbent pruned almost every mutation at a $50k slack, the production run used effectively non-binding slacks so that 581 complete 10-seed panels were measured. Racing seeds were 5005 (dairy) and 5033 (no milk/wool).

## Ladder calibration submission

`brain_hs2` was submitted first as authorized. Kaggle submission **56085372**, uploaded **2026-09-07 22:28:50.693 UTC**; artifact `dist/brain-hs2/main.py`, **77,380 bytes**. Packaging verification completed a 720-turn episode with DONE/DONE before upload.

## Town-draw census

The census ran 100 real no-op-vs-no-op episodes, seeds 5000–5099, on engine 1.32.7. Every seed reached eight shop instances. Counts below are seeds with at least one shop serving the product.

| shop-served product | seeds / 100 |
|---|---:|
| milk | 98 |
| wool | 62 |
| strawberry | 100 |
| egg | 92 |

Because strawberry appeared everywhere and only seed 5033 lacked both dairy channels, the panel was stratified by realized season drain and missing channels rather than simple presence.

| seed | class | milk drain | wool drain | strawberry drain | egg drain |
|---:|---|---:|---:|---:|---:|
| 5005 | balanced-high milk+wool; dairy race seed | 444 | 246 | 588 | 156 |
| 5033 | no milk/no wool; crop+egg race seed | 30 | 30 | 696 | 534 |
| 5043 | wool-heavy dairy | 336 | 750 | 372 | 66 |
| 5072 | milk-heavy dairy | 660 | 174 | 534 | 66 |
| 5026 | low-milk/no-wool, carrot-heavy | 120 | 30 | 354 | 264 |
| 5038 | strawberry-heavy/no-wool | 300 | 30 | 750 | 246 |
| 5024 | wool-only dairy channel | 30 | 246 | 246 | 336 |
| 5064 | milk-heavy; no wool/no egg | 768 | 30 | 642 | 30 |
| 5063 | product-variety sparse; no carrot/tomato | 84 | 606 | 408 | 480 |
| 5090 | low strawberry, mixed moderate | 228 | 138 | 120 | 264 |

## Base-plan parity

Before the fix the nominal base plan was not the raw policy. After adding a `raw_bot` plan-space anchor, mutations explicitly leave that anchor and enable the existing knobs.

| seed | raw before | adapter before | residual before | raw after | adapter after | action mismatches | daily-curve mismatches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5005 | $43,189 | $34,103 | −$9,086 | $43,189 | $43,189 | 0 / 719 | 0 / 30 |
| 5033 | $61,932 | $44,613 | −$17,319 | $61,932 | $61,932 | 0 / 719 | 0 / 30 |
| 5026 | $88,877 | $65,443 | −$23,434 | $88,877 | $88,877 | 0 / 719 | 0 / 30 |

## Emission

The winning plan is the raw anchor, so `bots/brain_hs3/main.py` is emitted as a standalone source file without appending the parameter adapter. On seed 5005 versus V46 it banked **$43,189 in-process and $43,189 emitted**.

## Fresh screen, seeds 4100–4107 in both seats

Intervals are whole-seed paired bootstraps with 10,000 resamples. Each row is 16 games (8 seeds × 2 seats).

| bot | opponent | bank [95% CI] | paired margin [95% CI] | faults | p99 | max |
|---|---|---:|---:|---:|---:|---:|
| hs3 | V46 | $78,147 [$67,975, $86,572] | −$32,081 [−$43,045, −$21,852] | 0 | 7.06 ms | 249.18 ms |
| control | V46 | $78,147 [$67,975, $86,572] | −$32,081 [−$43,045, −$21,852] | 0 | 6.83 ms | 119.64 ms |
| hs3 | tapeselect | $61,837 [$47,673, $73,600] | −$46,256 [−$51,767, −$41,323] | 0 | 6.68 ms | 155.89 ms |
| control | tapeselect | $61,837 [$47,673, $73,600] | −$46,256 [−$51,767, −$41,323] | 0 | 6.72 ms | 121.26 ms |

Direct hs3-minus-control bank and margin deltas are **$0 [95% CI $0, $0]** against both opponents. Therefore hs3 does **not** beat control by $5k; it is short by exactly **$5,000/game** because the 581-panel optimum was the raw control itself, not a mutation.

## Fresh-screen KPI versus V46

The KPI run covered 32 games on the same eight seeds in both seats. Mean final bank was $76,557; faults 0; productive-action fraction 37.14%; moves per productive action 1.63; wages $6,949/game; town-demand share 35.61%.

| product | units sold/game | realized dollars/game | realized/base |
|---|---:|---:|---:|
| wheat | 244.0 | $9,123 | 1.50× |
| carrot | 113.6 | $6,670 | 1.68× |
| tomato | 0.0 | $0 | — |
| strawberry | 164.9 | $30,580 | 1.55× |
| melon | 72.5 | $12,559 | 0.69× |
| egg | 0.0 | $0 | — |
| milk | 138.3 | $19,831 | 0.90× |
| wool | 133.2 | $13,814 | 0.52× |
| fertilizer | 222.2 | $13,359 | 0.60× |

## Parameters that mattered

The three largest within-population associations with panel margin were fertilizer sell policy (**r = −0.495** for metering), last strawberry planting day (**r = −0.433**), and hands on days 25–29 (**r = −0.421**). These are observational correlations among adapter candidates. The stronger result is categorical: all searched deviations lost to the exact raw anchor, which first appeared at completed evaluation 1 and gained $0 over both the final 100 and final 400 full panels.
