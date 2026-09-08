# Fixed-game tape polish: seed 3910

Pinned reference: **$76,009**.  Polished arbiter replay: **$76,870** (**+$861**).

> Specification conflict: the cited $138,413 is the same commit's unpinned, action-coupled run. Under the required `pin.py` schedule the policy replays to $76,009; its unpinned 720-action tape replays under the pin to $75,808. Therefore $138.4k is not the pinned reference.

Search: 3,600 evaluations, 6 chains, 6.41 minutes, zero faults, fixed-plan signature identical. Evaluation p99 1414.2 ms.

## Bank by day

| day | reference | polished |
|---:|---:|---:|
| 0 | $1,621 | $1,621 |
| 1 | $666 | $666 |
| 2 | $575 | $575 |
| 3 | $542 | $542 |
| 4 | $315 | $315 |
| 5 | $617 | $617 |
| 6 | $1,052 | $1,052 |
| 7 | $342 | $342 |
| 8 | $187 | $187 |
| 9 | $962 | $962 |
| 10 | $2,642 | $2,642 |
| 11 | $3,107 | $3,107 |
| 12 | $4,338 | $4,338 |
| 13 | $8,620 | $8,620 |
| 14 | $13,617 | $13,617 |
| 15 | $17,933 | $17,933 |
| 16 | $23,395 | $23,395 |
| 17 | $28,394 | $28,394 |
| 18 | $33,333 | $33,333 |
| 19 | $39,383 | $39,383 |
| 20 | $41,735 | $41,735 |
| 21 | $45,705 | $45,705 |
| 22 | $48,365 | $48,365 |
| 23 | $51,963 | $51,963 |
| 24 | $55,419 | $55,419 |
| 25 | $57,801 | $57,801 |
| 26 | $61,266 | $61,266 |
| 27 | $64,810 | $64,810 |
| 28 | $66,309 | $66,309 |
| 29 | $76,009 | $76,870 |

## Hand-turns by day

| day | ref move | ref act | ref idle | polished move | polished act | polished idle |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 47 | 51 | 41 | 47 | 51 | 41 |
| 1 | 58 | 33 | 48 | 58 | 33 | 48 |
| 2 | 64 | 37 | 38 | 64 | 37 | 38 |
| 3 | 68 | 40 | 31 | 68 | 40 | 31 |
| 4 | 91 | 48 | 0 | 91 | 48 | 0 |
| 5 | 113 | 47 | 2 | 113 | 47 | 2 |
| 6 | 95 | 54 | 13 | 95 | 54 | 13 |
| 7 | 102 | 56 | 4 | 102 | 56 | 4 |
| 8 | 108 | 54 | 0 | 108 | 54 | 0 |
| 9 | 108 | 48 | 0 | 108 | 48 | 0 |
| 10 | 147 | 78 | 0 | 147 | 78 | 0 |
| 11 | 153 | 74 | 0 | 153 | 74 | 0 |
| 12 | 145 | 82 | 0 | 145 | 82 | 0 |
| 13 | 136 | 90 | 1 | 136 | 90 | 1 |
| 14 | 142 | 86 | 0 | 142 | 86 | 0 |
| 15 | 158 | 92 | 0 | 158 | 92 | 0 |
| 16 | 154 | 95 | 1 | 154 | 95 | 1 |
| 17 | 158 | 92 | 0 | 158 | 92 | 0 |
| 18 | 156 | 94 | 0 | 156 | 94 | 0 |
| 19 | 156 | 94 | 0 | 156 | 94 | 0 |
| 20 | 156 | 94 | 0 | 156 | 94 | 0 |
| 21 | 156 | 94 | 0 | 156 | 94 | 0 |
| 22 | 155 | 95 | 0 | 155 | 95 | 0 |
| 23 | 157 | 93 | 0 | 157 | 93 | 0 |
| 24 | 156 | 94 | 0 | 156 | 94 | 0 |
| 25 | 117 | 89 | 0 | 117 | 89 | 0 |
| 26 | 121 | 85 | 0 | 121 | 85 | 0 |
| 27 | 118 | 88 | 0 | 118 | 88 | 0 |
| 28 | 119 | 87 | 0 | 119 | 87 | 0 |
| 29 | 124 | 46 | 27 | 124 | 46 | 27 |

Season total: reference 3,738 moving / 2,210 acting / 206 idle; polished 3,738 / 2,210 / 206.

## Gains by mutation

- sale_shift: +$861

## Remaining travel and idle

- Day 15: 158 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.
- Day 17: 158 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.
- Day 23: 157 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.
- Day 18: 156 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.
- Day 19: 156 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.
- Day 20: 156 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.
- Day 21: 156 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.
- Day 24: 156 moving, 0 idle. Idle hotspots: none. Moving remains the tape's point-to-point travel; idle is explicit PASS/no assigned job.

## Ceiling

For the pinned schedule, merely end-loading every actually sold unit after all town drain—an infeasible upper bound that ignores the 100-unit shed and liquidity—would bank **$126,769** at the reference physical output. Perfect routing could raise output, but >$150k would require roughly another $23k beyond even that sale-timing bound. A defensible ceiling for this fixed economic plan is **$135k–$145k**, with $150k an aggressive outer bound, not a reachable tape-polish target under this pinned shop schedule.
