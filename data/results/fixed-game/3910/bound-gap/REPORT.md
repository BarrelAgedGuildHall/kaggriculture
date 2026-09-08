# Bound-gap check

Integerized witness: **$329**. Best repaired goose/carrot schedule: **$83,764**. Gap to the valid LP bound: **$297,926**; gap to the pinned $138,797 action tape: **$55,033**.

The witness requested 49 geese by day 9, but the real run had only 8 live then (maximum 16 on day 13), never acquired a sheep, and the serviced herd collapsed to zero on day 15. `search.py --help` exposes no start-schedule option, so the conditional 20-minute seeded search was not run.

## Integerized LP witness: daily ledger

| Day | Bank | Produced | Sold | Move | Act | Idle |
|---:|---:|---|---|---:|---:|---:|
| 0 | $238 | — | WHEAT 30 | 61 | 48 | 30 |
| 1 | $197 | — | — | 68 | 41 | 30 |
| 2 | $228 | — | FERTILIZER 4, WHEAT 3 | 78 | 39 | 21 |
| 3 | $183 | CARROT 15 | CARROT 3, FERTILIZER 3, WHEAT 22 | 98 | 45 | 42 |
| 4 | $178 | CARROT 6, EGG 22, WHEAT 3 | CARROT 12, EGG 4, FERTILIZER 7, WHEAT 87 | 129 | 56 | 0 |
| 5 | $268 | EGG 12 | CARROT 6, EGG 20, FERTILIZER 6, WHEAT 61 | 126 | 51 | 0 |
| 6 | $249 | EGG 12 | EGG 10, FERTILIZER 6 | 120 | 71 | 9 |
| 7 | $195 | EGG 11 | EGG 10 | 88 | 51 | 0 |
| 8 | $278 | EGG 12 | EGG 14 | 107 | 64 | 32 |
| 9 | $240 | EGG 12 | EGG 8 | 75 | 56 | 8 |
| 10 | $369 | EGG 15, MELON 4 | EGG 15 | 108 | 63 | 37 |
| 11 | $426 | EGG 14 | EGG 18 | 93 | 62 | 52 |
| 12 | $139 | EGG 8, MILK 6 | EGG 7, FERTILIZER 66 | 131 | 76 | 0 |
| 13 | $27 | EGG 20, MILK 12 | — | 100 | 78 | 0 |
| 14 | $61 | EGG 7, MILK 6 | — | 66 | 46 | 0 |
| 15 | $146 | EGG 11, MILK 8 | — | 82 | 52 | 0 |
| 16 | $171 | EGG 17, MILK 4 | EGG 12 | 75 | 59 | 0 |
| 17 | $20 | EGG 14, MILK 2 | — | 96 | 60 | 0 |
| 18 | $33 | EGG 14, MILK 2 | — | 50 | 40 | 0 |
| 19 | $24 | EGG 13, MILK 1 | — | 73 | 61 | 0 |
| 20 | $196 | EGG 7, MILK 3 | EGG 4, FERTILIZER 4, WHEAT 10 | 121 | 57 | 0 |
| 21 | $5 | EGG 24, MILK 2 | EGG 2, FERTILIZER 1, WHEAT 3 | 96 | 63 | 0 |
| 22 | $44 | EGG 13, MILK 4 | FERTILIZER 1 | 60 | 52 | 0 |
| 23 | $3 | EGG 7, MILK 1 | FERTILIZER 1 | 92 | 64 | 0 |
| 24 | $16 | EGG 13, MILK 3 | FERTILIZER 1 | 90 | 66 | 0 |
| 25 | $96 | EGG 14, MILK 1 | FERTILIZER 1 | 78 | 56 | 0 |
| 26 | $174 | EGG 17, MILK 5 | FERTILIZER 1 | 123 | 75 | 0 |
| 27 | $61 | EGG 12, MILK 1 | FERTILIZER 1 | 118 | 80 | 0 |
| 28 | $27 | EGG 20, MILK 7 | FERTILIZER 1 | 99 | 57 | 0 |
| 29 | $1,002 | EGG 14, MILK 1 | FERTILIZER 1 | 93 | 44 | 54 |

## Measured iterations

| Schedule | Bank | Faults | p99 ms |
|---|---:|---:|---:|
| v0 | $329 | 0 | 2.960 |
| v1 | $190 | 0 | 1.986 |
| v2 | $8,432 | 0 | 2.318 |
| v3 | $31,190 | 0 | 3.195 |
| v4 | $79,679 | 0 | 3.296 |
| v5 | $83,764 | 0 | 1.453 |

## Product comparison

| Product | Bound units | Bound avg $ | Playable units | Playable avg $ | Town drain |
|---|---:|---:|---:|---:|---:|
| WHEAT | 283 | $36.17 | 184 | $40.98 | 282 |
| CARROT | 379 | $56.24 | 11 | $36.91 | 462 |
| TOMATO | 0 | — | 0 | — | 192 |
| STRAWBERRY | 36 | $178.17 | 0 | — | 66 |
| MELON | 150 | $216.67 | 4 | $272.75 | 30 |
| EGG | 2,224 | $39.35 | 250 | $43.14 | 120 |
| MILK | 276 | $223.94 | 133 | $218.91 | 228 |
| WOOL | 655 | $239.68 | 266 | $247.50 | 606 |
| FERTILIZER | 693 | $36.43 | 169 | $85.08 | 0 |

## Gap attribution

- labour travel: $69,518
- cash timing: $136,356
- pricing model: $9,104
- animal ramp: $54,765
- other: $28,183

The attribution is an exact reconciliation, not five independent counterfactuals: price uses the LP envelope on the units actually sold; cash timing is the missing animal-product envelope before service; animal ramp is the shortfall from the requested cohorts' perfect-care envelope; labour/travel is missing crop and fertilizer revenue; other is the residual for wheat, expenses, speculation, fractionality, storage, and interactions.

## Verdict

No: the best playable goose/carrot family reached **$83,764**, which is **$55,033 below** the pinned $138,797 schedule. The most-wrong single assumption is cash timing: the LP credits end-of-season, all-drain-first revenue early enough to finance its animal ramp, worth **$136,356** in this decomposition. The real integerized plan could place only 8 of 49 intended geese by day 9 and then lost the herd when liquidity could no longer fund the promised hands and feed.
