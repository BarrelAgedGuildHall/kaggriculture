# TapeSelect results — 2026-09-04

Engine 1.32.7. Branch `bot/tapeselect`. Candidate library: 107 fresh top-10 tapes + 13 V46 tapes, all candidates wrapped with V46 weed repair and final liquidation. The full 107-ghost sweep projected over 90 minutes under concurrent host load, so the declared fallback was used: 60 ghosts, exactly 6 per top-10 team.

Matrix: **7,200 rows = 120 tapes × 60 ghosts**; 0 duplicate/missing pairs; 0 failures. CSV: `warehouse/datasets/kaggriculture/tapes/matrix-2026-09-04.csv`.

## Overall

Best mean: `105498422_0` — mean +16,997, worst -18,813, 35/60 wins (58.3%).
Maximin: `105499136_1` — mean +12,466, worst -17,970, 26/60 wins (43.3%). They are **not the same**.

## First-shop winner and V46 regret

| key | n | best tape | mean | worst | V46 tape | V46 mean | regret |
|---|---:|---|---:|---:|---|---:|---:|
| BAKERY | 8 | `105498422_0` | +26,375 | -3,675 | `v46-b` | -23,582 | +49,956 |
| BRUNCH_SPOT | 8 | `105498460_1` | +15,137 | -15,554 | `v46-c` | -18,452 | +33,588 |
| FARMERS_MARKET | 8 | `105502931_0` | +21,584 | -16,694 | `v46-d` | -19,841 | +41,426 |
| ICE_CREAM_SHOP | 7 | `105495705_0` | +11,252 | -1,593 | `v46-c` | -16,586 | +27,838 |
| PET_CAFE | 6 | `105502932_0` | +67,854 | -22,429 | `v46-b` | -34,893 | +102,747 |
| PIZZA_SHOP | 5 | `105502929_0` | +6,243 | +1,052 | `v46-b` | -14,646 | +20,889 |
| SMOOTHIE_SHOP | 6 | `105506529_1` | +27,044 | -964 | `v46-c` | +11,949 | +15,096 |
| YARN_STORE | 12 | `105498422_0` | +17,340 | -6,422 | `v46-a` | -22,905 | +40,245 |

## First-two-shop winner and V46 regret

| key | n | best tape | mean | worst | V46 tape | V46 mean | regret |
|---|---:|---|---:|---:|---|---:|---:|
| BAKERY>BAKERY | 2 | `105494776_0` | +59,962 | -3,328 | `v46-b` | -19,874 | +79,836 |
| BAKERY>FARMERS_MARKET | 1 | `105506517_1` | +11,773 | +11,773 | `v46-b` | -24,471 | +36,244 |
| BAKERY>PIZZA_SHOP | 2 | `105491380_1` | +7,956 | +6,496 | `v46-b` | -34,436 | +42,392 |
| BAKERY>YARN_STORE | 3 | `105502932_0` | +37,751 | -5,486 | `v46-a` | -12,581 | +50,332 |
| BRUNCH_SPOT>BAKERY | 2 | `105490583_0` | +2,762 | +2,654 | `v46-a` | -15,736 | +18,498 |
| BRUNCH_SPOT>FARMERS_MARKET | 2 | `105498460_1` | +31,889 | +13,710 | `v46-e` | -9,543 | +41,432 |
| BRUNCH_SPOT>PET_CAFE | 4 | `105498460_1` | +18,722 | -7,006 | `v46-c` | -13,527 | +32,250 |
| FARMERS_MARKET>BAKERY | 1 | `105494817_0` | +28,188 | +28,188 | `v46-b` | -13,756 | +41,944 |
| FARMERS_MARKET>ICE_CREAM_SHOP | 1 | `v46-j` | +6,552 | +6,552 | `v46-e` | -24,933 | +31,485 |
| FARMERS_MARKET>PET_CAFE | 3 | `105494776_0` | +52,724 | -4,101 | `v46-d` | -27,222 | +79,946 |
| FARMERS_MARKET>SMOOTHIE_SHOP | 1 | `105487019_1` | +45,472 | +45,472 | `v46-c` | -17,554 | +63,026 |
| FARMERS_MARKET>YARN_STORE | 2 | `105491380_1` | +24,128 | -366 | `v46-a` | -701 | +24,829 |
| ICE_CREAM_SHOP>BAKERY | 1 | `105502926_0` | +67,062 | +67,062 | `v46-c` | -5,000 | +72,062 |
| ICE_CREAM_SHOP>ICE_CREAM_SHOP | 1 | `105506529_1` | +11,977 | +11,977 | `v46-c` | +521 | +11,456 |
| ICE_CREAM_SHOP>PET_CAFE | 2 | `105502932_0` | +21,182 | +7,703 | `v46-c` | -20,627 | +41,808 |
| ICE_CREAM_SHOP>SMOOTHIE_SHOP | 3 | `105498447_0` | +3,599 | +268 | `v46-c` | -23,457 | +27,056 |
| PET_CAFE>BRUNCH_SPOT | 1 | `105502932_0` | +112,988 | +112,988 | `v46-c` | -13,599 | +126,587 |
| PET_CAFE>ICE_CREAM_SHOP | 2 | `105494776_0` | +61,752 | -911 | `v46-c` | -19,270 | +81,022 |
| PET_CAFE>YARN_STORE | 3 | `105502932_0` | +56,959 | +1,645 | `v46-a` | -24,414 | +81,372 |
| PIZZA_SHOP>BRUNCH_SPOT | 2 | `105502929_0` | +5,644 | +1,052 | `v46-c` | -6,916 | +12,560 |
| PIZZA_SHOP>PIZZA_SHOP | 3 | `105486864_1` | +7,323 | -3,963 | `v46-b` | -12,406 | +19,729 |
| SMOOTHIE_SHOP>PET_CAFE | 4 | `105491245_0` | +41,107 | +9,813 | `v46-c` | +25,635 | +15,472 |
| SMOOTHIE_SHOP>PIZZA_SHOP | 2 | `105499136_1` | +9,941 | -1,606 | `v46-a` | -1,676 | +11,617 |
| YARN_STORE>BRUNCH_SPOT | 5 | `105498422_0` | +21,279 | -2,862 | `v46-a` | -21,594 | +42,874 |
| YARN_STORE>FARMERS_MARKET | 1 | `105491250_1` | +22,863 | +22,863 | `v46-a` | -13,880 | +36,743 |
| YARN_STORE>ICE_CREAM_SHOP | 4 | `105490529_0` | +2,010 | +0 | `v46-a` | -26,711 | +28,722 |
| YARN_STORE>YARN_STORE | 2 | `105502808_1` | +53,158 | -1,820 | `v46-a` | -23,079 | +76,237 |

## Opponent SELL clusters

Days 0–9 dominant SELL product produced two clusters: FERTILIZER=46, WHEAT=14. Best tape changed in 10/14 first-shop cells and 8/17 two-shop cells (cells require n≥2). This was not useful out of sample: leave-one-team-out shop-only 53.3%, +17,479; shop+cluster 43.3%, +10,537; delta -10.0%, -6,942. Opponent conditioning is excluded.

## Deployable exact-prefix policies

Shops are not visible until step 72. Both policies start with `105502932_0` and only switch to tapes sharing its exact first 72 actions (13-tape lineage); no day-6 splice is used. There are 23 distinct step-72 and 39 step-144 prefixes in the library.

| first shop | robust tape (`main.py`) | mean tape |
|---|---|---|
| BAKERY | `105498422_0` | `105498422_0` |
| BRUNCH_SPOT | `105502808_1` | `105502932_0` |
| FARMERS_MARKET | `105498422_0` | `105502931_0` |
| ICE_CREAM_SHOP | `105502797_0` | `105502797_0` |
| PET_CAFE | `105498422_0` | `105502932_0` |
| PIZZA_SHOP | `105502797_0` | `105502797_0` |
| SMOOTHIE_SHOP | `105502797_0` | `105502797_0` |
| YARN_STORE | `105502932_0` | `105498422_0` |

Robust in-sample: 32/60 (53.3%), mean +18,270, worst -10,120. Mean policy: 31/60 (51.7%), mean +21,123, worst -22,429.
Leave-one-team-out rebuilt robust policy: 30/60 = 50.0%, mean +17,575. Rebuilt mean policy: 30/60 = 50.0%, mean +17,733.

## Confirmations

Actual V46 router vs same ghosts: 11/60 = 18.3% (Wilson 10.6–29.9%), mean -15,236, p99 0.630 ms. Thus robust leave-one-team-out is +31.7 pp and +32,811 margin over V46.

| policy vs V46 | n | wins | win rate (Wilson 95%) | mean margin | p99 | max | faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| robust | 200 | 190 | 95.0% (91.0%–97.3%) | +19,585 | 1.915 ms | 240.827 ms | 0 crash, 0 timeout |
| mean | 200 | 188 | 94.0% (89.8%–96.5%) | +19,591 | 1.693 ms | 195.198 ms | 0 crash, 0 timeout |

Robust selector vs starter: 64/64 = 100.0% (Wilson 94.3-100.0%), mean margin +153,223, p99 0.944 ms, max 165.752 ms, 0 faults.

Matrix candidate timing: p99 of game-p99 = 0.246 ms; worst game-p99 = 3.793 ms.

## Two-slot portfolio

1. Robust selector (`main.py`): 95.0% vs V46; matrix worst −10,120.
2. Mean selector (`tapeselect-mean.py` experiment artifact): 94.0% vs V46; matrix mean +21,123. It differs on 4/8 shop routes; across ghosts the pair has 7 outcome disagreements and either policy wins 35/60 (58.3%).

Both clear V46 head-to-head with Wilson lower bounds >50% and both rebuild to 50.0% leave-one-team-out vs V46's 18.3% on the same ghost cohort.

Next: refresh the matrix with disjoint post-2026-09-04 top-team tapes and require the robust table to retain its shop choices before using opponent SELL conditioning.
