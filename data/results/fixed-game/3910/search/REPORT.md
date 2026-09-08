# Fixed game round 2 — Engineer B

Best pinned replay: **$138,797**, 0 faults, p99 1.557 ms.

## Search

7,735 logged evaluations, 620 accepted moves, 31.5 aggregate wall-minutes across three sequential 8-worker passes.

## Bank curve

| day | best | exec_b reported (un-pinned shops) | delta |
|---:|---:|---:|---:|
| 0 | $205 | $1,621 | $-1,416 |
| 1 | $307 | $666 | $-359 |
| 2 | $213 | $575 | $-362 |
| 3 | $318 | $540 | $-222 |
| 4 | $429 | $311 | $+118 |
| 5 | $432 | $670 | $-238 |
| 6 | $1,282 | $959 | $+323 |
| 7 | $199 | $2,060 | $-1,861 |
| 8 | $346 | $178 | $+168 |
| 9 | $2,550 | $933 | $+1,617 |
| 10 | $1,319 | $3,553 | $-2,234 |
| 11 | $2,127 | $4,570 | $-2,443 |
| 12 | $19,957 | $5,829 | $+14,128 |
| 13 | $18,570 | $11,428 | $+7,142 |
| 14 | $29,457 | $20,501 | $+8,956 |
| 15 | $29,925 | $26,966 | $+2,959 |
| 16 | $39,634 | $31,060 | $+8,574 |
| 17 | $43,332 | $40,428 | $+2,904 |
| 18 | $48,877 | $48,117 | $+760 |
| 19 | $57,564 | $54,501 | $+3,063 |
| 20 | $67,429 | $63,945 | $+3,484 |
| 21 | $70,342 | $74,059 | $-3,717 |
| 22 | $81,721 | $79,270 | $+2,451 |
| 23 | $84,562 | $87,866 | $-3,304 |
| 24 | $97,128 | $94,919 | $+2,209 |
| 25 | $102,203 | $100,993 | $+1,210 |
| 26 | $104,003 | $111,664 | $-7,661 |
| 27 | $114,620 | $116,235 | $-1,615 |
| 28 | $122,782 | $121,530 | $+1,252 |
| 29 | $138,797 | $138,413 | $+384 |

## Product ledger

| product | units | revenue | realised/base |
|---|---:|---:|---:|
| wheat | 159 | $5,638 | 1.418x |
| carrot | 3 | $105 | 1.000x |
| tomato | 0 | $0 | — |
| strawberry | 0 | $0 | — |
| melon | 57 | $14,151 | 0.993x |
| egg | 0 | $0 | — |
| milk | 145 | $30,658 | 1.321x |
| wool | 412 | $100,497 | 1.220x |
| fertilizer | 477 | $25,907 | 0.543x |

## Work and loss

Observed unit slots: 6,725; moving 3,923; productive actions 2,141; idle 618; ineffective non-pass 43.
Final undelivered: $0; observed rot: $0; net slippage versus base: $-5,326 (a scarcity premium), despite fertilizer losing $21,793 versus base.

## Schedule and binding resources

The winning NW block opens with 12 wheat across the north, 11 melons across the south, and one carrot; it buys no land. It front-loads 17 sheep and keeps only 6 cows, runs 4–12 hands/day (mean 8.83), meters selected fertilizer/wheat/melon/milk sale days, and finishes with no inventory.
Cash binds days 0–9 (bank repeatedly near $200). The 25 NW tiles bind placement around the day-10 melon/animal turnover, but a land+carrot relaxation lost $9.1k, showing labour is the economic constraint from days 10–24. Late, wool/milk remain below town drain, so drain does not bind them; zero fertilizer drain binds fertilizer price and creates the $21.8k adverse slippage.

Ablations from the winner: dairy herd −$63,931; dairy hands −$15,577; remove melon block −$13,277; sell-all −$2,046.

## Pinned-instance discrepancy

The historical $138,413 reference used action-coupled Ice Cream/Yarn/... shops. Its verbatim 7baa291 router replays to $76,009 under the mandated pinned Yarn/Pet Cafe/... shops, so $138,413 is not a reproducible parity target on this instance.
