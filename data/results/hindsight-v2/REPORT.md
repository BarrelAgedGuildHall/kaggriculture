# Hindsight Solver v2 — Contested

Engine `1.32.7`; seeds 3910, 3911, 3912, 3913; one alternating seat per seed; live opponent `bots\v46\main.py`. Objective: mean paired margin.

## Panel results

| strategy | mean margin | mean our bank | mean opponent bank |
|---|---:|---:|---:|
| brain_strawberry baseline | $-32,669 | $76,423 | $109,092 |
| hindsight-v2 best | $-39,984 | $73,755 | $113,738 |

| strategy | seed | seat | our bank | opponent bank | margin |
|---|---:|---:|---:|---:|---:|
| brain_strawberry baseline | 3910 | 0 | $67,994 | $115,084 | $-47,090 |
| brain_strawberry baseline | 3911 | 1 | $64,932 | $81,013 | $-16,081 |
| brain_strawberry baseline | 3912 | 0 | $79,978 | $120,560 | $-40,582 |
| brain_strawberry baseline | 3913 | 1 | $92,788 | $119,711 | $-26,923 |
| hindsight-v2 best | 3910 | 0 | $90,915 | $129,587 | $-38,672 |
| hindsight-v2 best | 3911 | 1 | $70,575 | $102,509 | $-31,934 |
| hindsight-v2 best | 3912 | 0 | $66,246 | $103,607 | $-37,361 |
| hindsight-v2 best | 3913 | 1 | $67,283 | $119,251 | $-51,968 |

## Best plan

Open 20 wheat, 5 melon. Replant targets: days 10 16 wheat 60%/melon 17%/strawberry 23%; days 17 22 wheat 55%/carrot 1%/melon 20%/strawberry 24%; days 23 26 wheat 72%/carrot 23%/strawberry 5%. Land: NE never, SW never, SE never. Animals: no goose, cow 6 from d0 every 1d x1, sheep 11 from d0 every 1d x1. Hands by five-day block: 5/8/10/9/10/10. Fertilise: wheat, carrot, melon. Metered: carrot, tomato, strawberry, melon, egg; all others immediate.

## Per-day bank (ours)

| day | baseline | best |
|---:|---:|---:|
| 0 | $936 | $872 |
| 1 | $413 | $373 |
| 2 | $484 | $439 |
| 3 | $148 | $630 |
| 4 | $245 | $952 |
| 5 | $263 | $420 |
| 6 | $191 | $417 |
| 7 | $228 | $261 |
| 8 | $236 | $510 |
| 9 | $412 | $440 |
| 10 | $38 | $705 |
| 11 | $814 | $718 |
| 12 | $310 | $866 |
| 13 | $1,807 | $2,870 |
| 14 | $7,443 | $9,972 |
| 15 | $10,382 | $12,386 |
| 16 | $15,537 | $18,498 |
| 17 | $19,506 | $24,924 |
| 18 | $22,487 | $27,490 |
| 19 | $23,588 | $32,181 |
| 20 | $26,582 | $37,403 |
| 21 | $28,216 | $38,732 |
| 22 | $34,808 | $43,275 |
| 23 | $43,021 | $48,156 |
| 24 | $48,065 | $49,324 |
| 25 | $51,976 | $53,444 |
| 26 | $56,638 | $58,140 |
| 27 | $61,662 | $59,312 |
| 28 | $67,485 | $68,924 |
| 29 | $76,423 | $73,755 |

## Sensitivity

| parameter | correlation with mean margin |
|---|---:|
| `land.NE` | +0.889 |
| `animal.COW.total` | -0.878 |
| `mix.days_23_26.STRAWBERRY` | +0.876 |
| `land.SW` | +0.875 |
| `last.STRAWBERRY` | -0.848 |
| `last.CARROT` | -0.822 |
| `sell.EGG` | +0.816 |
| `hands.d25_29` | -0.798 |
| `sell.MELON` | +0.786 |
| `hands.d15_19` | -0.785 |
| `hands.d20_24` | -0.774 |
| `mix.days_23_26.CARROT` | -0.745 |

Associations are observational within the correlated evolutionary population.

## Search efficiency

The resumable log contains 1576 attempts: 834 completed panels, 568 pruned, and 174 raced out. It ran for 101.0 cumulative minutes and executed 4379 games (0.722 games/s). The best first appeared at completed evaluation 602; gains over the final 100 and 400 completed panels were $0 and $998. Final heat 0.250; prune slack $12,000; race slack $8,000; worker cap 8.

The v1 eight-worker smoke executed 32 games in 42.75 seconds, or **0.749 games/s**. V2 executed **0.722 games/s**. Attempt throughput rose from 0.187/s to 0.260/s (1.39x), not the targeted 3–5x: the engine-game rate was unchanged and 834 finalists still paid for all four games. Racing/pruning saved games, but not enough to produce a 3x attempt-rate gain.

## Seat-symmetry verification

The raw baseline was rerun in both seats. Final bank and every one of the 30 daily bank points matched exactly.

| seed | seat 0 final | seat 1 final | unequal daily points |
|---:|---:|---:|---:|
| 3910 | $67,994 | $67,994 | 0 / 30 |
| 3911 | $64,932 | $64,932 | 0 / 30 |

Verdict: symmetric for this deterministic baseline/panel, so search games use seed parity as the seat (0/1/0/1 for 3910–3913). A prune timing check on the v1 contested plan at seed 3910 ran 719 steps in 6.81 s without pruning and stopped at step 456 in 4.11 s with pruning; its bank at the gate was $5,023.

## Baseline verdict

The best searched plan **did not beat the raw brain_strawberry baseline**. Its panel margin was $−39,984 versus baseline $−32,669: $7,315 worse, and $12,315 short of the requested +$5,000 improvement. The parameter adapter's search space could not reproduce the stronger raw-bot behavior on this panel, and selection on four fixed seeds overfit: the held-out screen below is substantially worse again.

## Fresh-seed screen (4000–4007, both seats)

Each row has 16 games. Margin intervals use a whole-seed paired bootstrap (10,000 resamples). Both bots had zero crashes and zero timeouts. A single host-jitter turn made the arena max-time gate fail for each bot, but p99 remained far below budget.

| bot | opponent | mean bank | mean paired margin | paired 95% CI | faults | p99 turn |
|---|---|---:|---:|---:|---:|---:|
| brain_hs2 | V46 | $55,479 | $−60,328 | [$−83,468, $−37,404] | 0 | 4.32 ms |
| brain_strawberry | V46 | $78,462 | $−35,557 | [$−49,078, $−26,461] | 0 | 7.85 ms |
| brain_hs2 | tapeselect | $47,227 | $−90,723 | [$−99,789, $−81,930] | 0 | 4.35 ms |
| brain_strawberry | tapeselect | $78,130 | $−56,825 | [$−59,699, $−53,799] | 0 | 7.98 ms |

Paired directly against the control on identical games, hs2 lost $24,771 margin/game versus V46 (95% CI $−42,627 to $−7,438) and $33,898 versus tapeselect (95% CI $−42,427 to $−25,108).

## KPI money (32 games vs V46, seeds 4000–4015)

| product | mean units sold | mean realised dollars | realised/base |
|---|---:|---:|---:|
| WHEAT | 282.1 | $7,887 | 1.12x |
| CARROT | 6.0 | $290 | 1.38x |
| TOMATO | 0.0 | $0 | — |
| STRAWBERRY | 9.6 | $1,992 | 1.74x |
| MELON | 29.4 | $6,291 | 0.85x |
| EGG | 0.0 | $0 | — |
| MILK | 169.1 | $20,202 | 0.75x |
| WOOL | 251.7 | $27,523 | 0.55x |
| FERTILIZER | 369.0 | $14,673 | 0.40x |

Mean final bank was $51,301; faults 0; wages $2,915; productive-action fraction 28.48%; moves per productive action 2.27. The three strongest observational sensitivities were land.NE (+0.889), cow total (−0.878), and late strawberry mix (+0.876). They are correlated-population associations, not isolated causal effects.

## Standalone verification

`bots/brain_hs2/main.py` contains the source strategy, adapter, and literal plan; uses standard-library imports only; and leaves `agent` as the last inserted callable. Seed 3910 versus V46 returned **$90,915 in-process and $90,915 emitted**. A fresh Python 3.11 venv containing only kaggle-environments 1.32.7 and its dependencies reran the emitted file to DONE/DONE with the same $90,915 bank.
