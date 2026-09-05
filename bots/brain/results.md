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
