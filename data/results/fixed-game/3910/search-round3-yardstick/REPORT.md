# Fixed game search, round 3

## Numbers first

- Pinned 3910, 30 minutes / 8 workers: **$138,797 before; $139,281 after** (**+$484**). The saved action tape and a separate `execute.py` real-engine run both replay to $139,281 with zero faults.
- Starting-family bests in the same budget: round-2 incumbent **$139,281**; tomato/carrot crop-heavy **$129,097**; mixed **$104,597**; goose/carrot/sheep **$103,090**.
- Search curve: 5 minutes **$138,797**; 10 **$139,029**; 20 **$139,171**; 30 **$139,281**.
- Throughput: **5,493 evaluations / 30.10 min = 182.5 evaluations/min**, versus round 2's 7,735 / 31.48 min = 245.7/min. Block mutations cost throughput, but improved the replayed bank.
- Policy timing on the winning real-engine replay: **p99 1.762 ms**, max 28.56 ms. Search and all reported ablations had zero faults.

## Same-schedule router effects

All comparisons replay the round-3 winning schedule, changing only the named router option.

| change | bank | delta | moving turns | finding |
|---|---:|---:|---:|---|
| no zero-drain fertilizer cutoff | $135,326 | baseline | 4,105 | 396 fertilizer sold for $21,896; $17,704 below base value |
| stop collecting below $10 | **$139,281** | **+$3,955** | 4,090 | same fertilizer sales; low-price collection work was displaced |
| reference crop layout | $129,216 | baseline | 4,108 | ignores scheduled block positions |
| scheduled melon/wheat block layout | **$139,281** | **+$10,065** | 4,090 | searched shed-relative locations pay materially |
| swap melon and wheat target positions | $131,098 | **-$8,183** | 4,094 | confirms crop identity and shed distance matter |
| normalize animal strips over active herd | $124,827 | **-$14,454** | 4,372 | rejected: the reference router already performs the better contiguous sweep |

Travel remains **4,090 moving turns**, 2,275 acting and 628 idle on the winner. It is not solved; the attempted herd reassignment made both travel and bank worse.

## Mechanisms

Search now has block recropping, block day shifts, block-position swaps, whole-herd timing shifts, hands +/-1 over day ranges, per-product sell-on-sight/hold/never policies, simulated-annealing acceptance, and periodic restarts from each worker's top six. Starts explicitly cover the LP's goose/carrot/sheep family, tomato/carrot crop-heavy, mixed, and the prior incumbent.

The winning 3910 schedule remains in the incumbent family: 13 wheat, 9 melon and 3 carrot planting markers; 11 sheep, 7 cows, no geese; 3-14 hands/day. The structural alternatives improved substantially but did not beat it.

## Generalization

Seed 5006 was pinned independently and verified across all 720 pass observations. Its shops are Brunch, Ice Cream, Pet Cafe, Brunch, Ice Cream, Pet Cafe, Farmers Market, Bakery. A fixed **20-minute / 4-worker** run (the Engineer already occupied eight workers) completed 2,124 evaluations at 105.8/min. Best was **$93,886, mixed family**, replayed identically through `execute.py`, zero faults. The unadapted 3910 schedule started at $51,121.

## CLI compatibility

Existing `search.py` flags are unchanged: `--workers`, `--evaluations`, `--minutes`, and `--seed`. New optional flags are `--instance`, `--tag`, `--temperature`, and `--top-k`, all with defaults. Existing `execute.py` positional schedule plus `--seed`, `--workers`, `--tag`, and `--no-idle-fix` still work; new router/instance flags are optional.

Next: share top-k schedules across workers, rather than only within each worker, and seed restarts with cross-family block grafts.
