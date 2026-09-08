# Prefix-snapshot evaluation

## Numbers first

- Exactness: **50/50** random mutations of the pinned 3910 winner matched full replay to the dollar; zero failures across all eight mutation classes.
- Full replay, 10 minutes / 4 workers: **1,182 evaluations**, **117.43 evaluations/min**, best **$139,432**, replay **$139,432**, zero faults.
- Prefix reuse, 10 minutes / 4 workers: **1,627 evaluations**, **161.80 evaluations/min**, best **$139,432**, replay **$139,432**, zero faults.
- Throughput change: **1.378x (+37.8%)**. The 2x target was not reached because most winner mutations genuinely affect days 0-8; exact reuse cannot skip those transitions.
- Seed 5002, 25 minutes / 4 workers: **4,081 evaluations**, **162.85 evaluations/min**, best and search replay **$100,016**, up **$4,335** from $95,681, zero faults.
- Independent `execute.py` arbiter: **$100,016**, zero faults; policy p99 **1.422 ms**, max **26.426 ms**.

Prefix mode is exposed as `--prefix-eval` / `--no-prefix-eval` and defaults on after the 50/50 exactness gate. It restores the incumbent at the first affected day; conservative day-zero mutations first compare deterministic router actions against stored parent observations and skip unchanged engine days.
