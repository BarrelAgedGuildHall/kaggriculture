# Router round 4

## 3910 travel attribution

| task | moving turns | share | turns / $1k realised revenue |
|---|---:|---:|---:|
| feed | 1,626 | 39.76% | 11.03 |
| care/milk/shear | 733 | 17.92% | 4.97 |
| fertilizer collection | 654 | 15.99% | 4.44 |
| haul to shed | 536 | 13.11% | 3.64 |
| water | 221 | 5.40% | 1.50 |
| plant/build | 218 | 5.33% | 1.48 |
| harvest | 76 | 1.86% | 0.52 |
| fertilize | 26 | 0.64% | 0.18 |
| sell | 0 | 0.00% | 0.00 |
| total | 4,090 | 100.00% | 27.76 |

Realised revenue: **$147,356**.

| hand | feed | care/milk/shear | fertilizer collection | haul | water | plant/build | harvest | fertilize | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 54 | 109 | 85 | 22 | 31 | 42 | 7 | 0 | 350 |
| 1 | 172 | 55 | 70 | 50 | 26 | 46 | 7 | 0 | 426 |
| 2 | 156 | 66 | 77 | 47 | 27 | 21 | 11 | 3 | 408 |
| 3 | 142 | 75 | 68 | 41 | 20 | 25 | 3 | 5 | 379 |
| 4 | 124 | 79 | 73 | 49 | 15 | 17 | 10 | 5 | 372 |
| 5 | 140 | 58 | 72 | 42 | 21 | 27 | 6 | 8 | 374 |
| 6 | 167 | 56 | 45 | 52 | 18 | 8 | 9 | 5 | 360 |
| 7 | 169 | 48 | 32 | 52 | 18 | 12 | 2 | 0 | 333 |
| 8 | 120 | 59 | 46 | 54 | 20 | 9 | 6 | 0 | 314 |
| 9 | 130 | 58 | 42 | 45 | 9 | 3 | 5 | 0 | 292 |
| 10 | 118 | 35 | 30 | 39 | 9 | 1 | 9 | 0 | 241 |
| 11 | 87 | 23 | 11 | 28 | 4 | 5 | 1 | 0 | 159 |
| 12 | 31 | 8 | 2 | 10 | 3 | 2 | 0 | 0 | 56 |
| 13 | 6 | 2 | 1 | 5 | 0 | 0 | 0 | 0 | 14 |
| 14 | 10 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |

## Router measurements

| router | seed | bank | delta | move | act | idle | faults | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3910 | $139,281 | $0 | 4,090 | 2,275 | 628 | 0 | 2.108 |
| baseline | 5000 | $126,610 | $0 | 3,474 | 2,014 | 925 | 0 | 1.831 |
| baseline | 5001 | $136,502 | $0 | 4,060 | 2,260 | 680 | 0 | 2.037 |
| full carry | 3910 | $134,913 | -$4,368 | 4,063 | 2,240 | 649 | 0 | 1.710 |
| full carry | 5000 | $120,907 | -$5,703 | 3,656 | 2,053 | 1,047 | 0 | 1.530 |
| full carry | 5001 | $132,789 | -$3,713 | 4,107 | 2,239 | 620 | 0 | 1.654 |
| capacity husbandry crew | 3910 | $4,694 | -$134,587 | 3,649 | 1,844 | 1,428 | 0 | 1.385 |
| capacity husbandry crew | 5000 | $24,838 | -$101,772 | 3,222 | 1,765 | 1,505 | 0 | 1.380 |
| capacity husbandry crew | 5001 | $16,995 | -$119,507 | 3,731 | 1,852 | 1,300 | 0 | 1.391 |
| animal bundle hard | 3910 | $134,097 | -$5,184 | 3,942 | 2,298 | 726 | 0 | 1.635 |
| animal bundle hard | 5000 | $94,031 | -$32,579 | 3,069 | 1,990 | 1,248 | 0 | 1.449 |
| animal bundle hard | 5001 | $134,914 | -$1,588 | 3,951 | 2,298 | 839 | 0 | 1.747 |
| animal bundle soft | 3910 | $135,953 | -$3,328 | 3,848 | 2,336 | 808 | 0 | 1.646 |
| animal bundle soft | 5000 | $93,624 | -$32,986 | 3,050 | 1,978 | 1,254 | 0 | 1.558 |
| animal bundle soft | 5001 | $133,456 | -$3,046 | 3,942 | 2,282 | 840 | 0 | 1.704 |
| locality 20 | 3910 | $139,281 | $0 | 4,090 | 2,275 | 628 | 0 | 3.253 |
| locality 20 | 5000 | $126,610 | $0 | 3,474 | 2,014 | 925 | 0 | 2.745 |
| locality 20 | 5001 | $136,502 | $0 | 4,060 | 2,260 | 680 | 0 | 3.168 |
| locality 40 | 3910 | $136,656 | -$2,625 | 4,096 | 2,257 | 597 | 0 | 3.194 |
| locality 40 | 5000 | $122,559 | -$4,051 | 3,552 | 2,043 | 950 | 0 | 2.895 |
| locality 40 | 5001 | $136,506 | +$4 | 4,063 | 2,256 | 683 | 0 | 3.065 |
| locality 100 | 3910 | $136,582 | -$2,699 | 4,104 | 2,269 | 759 | 0 | 2.859 |
| locality 100 | 5000 | $121,769 | -$4,841 | 3,591 | 2,053 | 889 | 0 | 2.483 |
| locality 100 | 5001 | $131,701 | -$4,801 | 3,951 | 2,214 | 682 | 0 | 2.773 |
| locality 300 | 3910 | $138,104 | -$1,177 | 3,907 | 2,333 | 869 | 0 | 2.976 |
| locality 300 | 5000 | $124,655 | -$1,955 | 3,348 | 2,089 | 1,097 | 0 | 2.500 |
| locality 300 | 5001 | $134,374 | -$2,128 | 3,920 | 2,284 | 764 | 0 | 2.744 |
| balanced husbandry | 3910 | $118,355 | -$20,926 | 3,784 | 2,182 | 983 | 0 | 1.583 |
| balanced husbandry | 5000 | $114,126 | -$12,484 | 3,397 | 1,996 | 1,130 | 0 | 1.491 |
| balanced husbandry | 5001 | $130,730 | -$5,772 | 3,890 | 2,209 | 906 | 0 | 1.551 |

Kept router: **baseline**. Default travel router: **baseline**. No-regression gate: **1/9 passed**. Travel-improvement + no-regression gate: **0/9 passed**. Target delta: **+$0 / +$8,000**.

## 3910 search

| minutes | workers | evaluations | eval/min | prior | best | delta | replay | faults | move | act | idle | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15.12 | 6 | 1,995 | 131.90 | $139,281 | $139,432 | +$151 | $139,432 | 0 | 4,020 | 2,257 | 604 | 1.733 |

## Paths

| artifact | path |
|---|---|
| attribution | `data/results/fixed-game/3910/execution/router-r4-attribution/ledger.json` |
| three-seed ledgers | `data/results/fixed-game/{3910,5000,5001}/execution/router-r4-*` |
| search | `data/results/fixed-game/3910/search-router-round4/summary.json` |
| search replay | `data/results/fixed-game/3910/execution/router-r4-search-winner/ledger.json` |

## Verification

| check | result |
|---|---:|
| fixed-game replay faults | 0 |
| pytest | 46 passed |
| py_compile | 2 passed |
| CLI removals | 0 |
| maximum worker processes | 6 |
