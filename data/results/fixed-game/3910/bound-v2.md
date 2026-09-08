# Fixed game round 2: meaningful economic upper bound

## Numbers first

**Valid upper bound: $381,690. Realistic-travel variant: $326,649.** The valid total is a $359,490 farm LP plus a separately valid $22,200 ceiling for wheat carry trading. The realistic variant uses 1.842 measured moves per non-move action and is diagnostic, not certified.

The exact `7baa291` reference has a provenance mismatch. It realizes **$138,413 only under its endogenous shop draw** (`ICE_CREAM_SHOP / YARN_STORE / ICE_CREAM_SHOP / BAKERY / SMOOTHIE_SHOP / YARN_STORE / SMOOTHIE_SHOP / FARMERS_MARKET`). With the fixed instance's requested pin (`YARN_STORE / PET_CAFE / YARN_STORE / PIZZA_SHOP / BAKERY / PET_CAFE / PIZZA_SHOP / SMOOTHIE_SHOP`) the same committed agent realizes **$76,009**, not $138,413.

Fixing the reference trajectory's actual quantities and expenses and applying this pinned model's all-drain-first price envelope gives **$121,988 versus $138,413**. That negative $16,425 difference is not model tightness: it is the different shop instance. On the actual pinned run, the fixed-schedule model value is **$123,154 versus $76,009**, so measured model looseness is **$47,145**. The valid bound's excess over that fixed pinned schedule is **$258,536**. Both 720-action tapes are saved beside this report.

Engine: `kaggle-environments==1.32.7`; source SHA-256 `BC8A54879EF02C7EA64B8B333D6A976F0EA65C4949149D01F463F23BCCEE653E`. “Engine lines” below refer to `.venv/Lib/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`.

## Product trading: spread, slots, and value

- Only `WHEAT` and `FERTILIZER` are legal `BUY_PRODUCT` items (engine lines 596–605). A buy is quoted at `market_price(inventory-1)`; after the buy the inventory is exactly `inventory-1`, so an immediate sell receives the identical quote. There is **no markup/spread at unchanged inventory** (lines 598–601, 652–671).
- The engine takes at most ten submitted market orders each turn (lines 551–560). Quantity is carried inside one parsed order and processed unit by unit (lines 583–628, 639–648), so one bulk buy plus one bulk sell uses two slots. The shed caps held product at 100 (lines 665–671).
- Fertilizer has no town drain and therefore no carry profit. Wheat has 282 units of pinned town drain, so holding wheat across a drain tick can profit. The model does not allow bought product to masquerade as farm output. Instead it adds a **$22,200 valid trading ceiling**: at each of the 120 exact wheat-drain ticks, it grants 100 held units the maximum integer-price appreciation caused by that tick, then sums the grants. This honors the 100-item shed and is looser than the true two-slot/cash/timing path because it grants a freshly full shed at every tick. Thus $22,200 is what speculation is worth *in the bound*; its feasible value is lower.

## Why each valid constraint contains every real game

### Labour

The hire schedule is exactly Fibonacci `1,1,2,3,5,…` by that day's zero-based hire index (engine lines 690–709). Contrary to the task premise, hands **do not persist and do not draw recurring wages**: end of day deletes all hands and resets `hires_today` (lines 873–882). Ten order slots × 24 turns gives an engine syntactic maximum of 240 hires/day, plus the permanent farmer. The LP carries the first 40 prefix hires with their exact costs; hire 41 alone costs more than any coarse all-tiles physical revenue ceiling, so hires 41–240 cannot improve a nonnegative-bank solution. Every represented worker supplies 24 turns. Daily constraints charge unavoidable same-day work and a true travel floor; a second season-pooled constraint charges planting, watering, building, pickup, placement, feeding, care, fertilizer collection, and minimum harvests. Pooling work across days and allowing fractional hands only relaxes reality.

### Purchases, structures, animals, and feed

Seeds and animal prices are the engine constants (crop/animal table, lines 11–22); land follows NE→SW→SE at $1,000/$2,000/$4,000 (lines 95–97, 712–725). Coops and pastures cost $0 but require one build action on an empty owned tile (lines 493–502). Placement requires a matching unoccupied structure and consumes one carried animal (lines 377–391), hence one animal per structure and one structure per tile. The LP charges build + pickup + place, assigns every purchase to a tile, and every live animal consumes aggregate tile capacity. It relaxes persistent structure identity/free disposal, so the real game is at least as constrained.

`FEED` consumes exactly one carried wheat and is limited to once per animal/day (lines 505–512). Wheat must originate in grown stock or legal `BUY_PRODUCT WHEAT` (lines 598–603). The LP has those two sources, charges a feed action, requires enough pooled feed to avoid two consecutive misses, and additionally requires every modeled care bonus to have a feed unit. Bought feed is charged at the $1 price floor rather than its usually higher quote; pooling feed across animals and days is optimistic.

### Tile-indexed crops and travel

The model has `plant_at[crop,tile,day]` for all 5 × 100 × 30 crop choices and `animal_at[type,tile,day]` for all 3 × 100 × 30 placements. Per-tile/day occupancy permits one crop/structure choice; quadrant assignment is linked to the ordered land purchases. Ongoing crops occupy through their fourth scheduled production; one-time crops occupy through their max-yield age. Free fractional coloring, free weeds/digging, and relaxed persistent animal identity enlarge the real set.

Planting starts with one unwatered miss and initial yield 0 for ongoing crops or 1 for one-time crops (lines 215–225). Watering is once per day; one-time crops add at most one unit, or two with fertilizer, only in their growth window (lines 431–443). Two consecutive missed days kill a crop; ongoing production follows first-yield age/interval, caps at four productions, and rot then removes one held unit every two turns (lines 752–802). The LP grants free fertilizer and lifetime cap yield, but charges at least `max(ceil((last_age+1)/2), ceil(new_units/2))` watering visits plus plant and harvest. Real crops need no fewer visits and yield no more.

The four shed access coordinates are `(4,4),(5,4),(4,5),(5,5)` (lines 132–139); movement is orthogonal one tile/turn (lines 323–331). Market orders are remote, so true tile-to-market travel is **zero**. At each day all units respawn at shed access (lines 879–882), so reaching the furthest worked tile costs at least its exact Manhattan shed distance. Feed/collection visits also require at least `distinct visited animal tiles − workers` moves. These are valid lower bounds, while chaining, return trips, carrying, watering travel, and most inter-tile motion are omitted. The realistic variant instead multiplies non-move work by 1.842.

### Animal ramp and fertilizer joint product

Animals escape after two consecutive unfed ends of day. At scheduled ages they add base one plus banked care only when fed, cap held output at 4/6/6, then reset the bank; fed+care also creates the next bonus and every surviving animal exposes one fertilizer (engine lines 805–833). The LP's cohort/day survival variables gate every production tick. Its perfect-care envelopes are goose `4 then 2/tick`, cow `6 then 3/tick`, sheep `6 then 4/tick`; it also caps units by base ticks plus care actions and care by fed animal-days. Harvest actions are at least sold units divided by held cap. Fertilizer is at most one collection per surviving animal-day and costs a collection action. Pooled care/feed, fractional cohorts, perfect harvest timing, and delayed fertilizer collection are all favorable relaxations.

### Market timing and price

There is **no daily market reset**. Inventory starts at `I0=10000`; below `I0` receives the scarcity branch, above it the glut branch, floored at $1 (engine lines 178–206). Every transition resolves unit actions, then market orders, then town consumption, then rot, and only at hour 23 performs end-of-day growth/reset (lines 935–950). Shop drain fires every four steps and town-center drain every 24, then prices refresh (lines 728–749). Sales at $1 deliberately add no supply (lines 652–661).

For each product the LP uses one cumulative season curve: it moves **all pinned drain before all own sales**, then walks marginal integer quotes in order; when price reaches $1, subsequent units remain at $1 without adding supply. Any real sale sees no less inventory than this all-drain-first construction and therefore no higher quote. Crediting that final envelope early for cash and using block-first marginal prices are relaxations; total modeled revenue cannot exceed it.

### Cash, storage, slots, and integrality

Cash starts at $3,000 and cannot borrow. Daily flow charges seeds, animals, exact prefix wages, land, and feed; same-day netting and early credit from the season market envelope relax within-day liquidity. Each active bulk item/order plus each atomic hire/land order counts against 240 daily slots, a relaxation of the true ten-per-turn timing. Shed and carried-inventory capacities, overflow, pickup/drop routing, and order placement hour are omitted. Every integer choice is relaxed to a fraction. Therefore these constraints cannot lower the optimum below a feasible real game's bank.

## Attaining LP schedule by five-day block

Fractional values are the LP witness, not an executable recommendation.

| Days | Hands by day | Animals bought | Tiles planted | Sales |
|---|---|---|---|---|
| 0–4 | 5, 6, 7, 7, 7 | 25.4 goose, 5.5 cow, 1.3 sheep | 47.2 wheat, 53.9 carrot | 215.5 wheat, 139.5 carrot, 25.9 egg, 40.0 fertilizer |
| 5–9 | 7, 8, 8, 8, 8 | 23.5 goose, 3.5 cow, 22.1 sheep | 30.5 carrot, 15.5 melon | 67.5 wheat, 196.9 carrot, 315.5 egg, 74.9 fertilizer |
| 10–14 | 8, 8, 8, 8, 8 | none | 6.2 carrot | 977.9 egg, 39.3 wool, 65.6 milk, 82.9 fertilizer |
| 15–19 | 8, 8, 8, 8, 8 | none | 3.5 carrot, 9.0 strawberry, 9.5 melon | 905.0 egg, 147.9 fertilizer |
| 20–24 | 8, 8, 8, 8, 8 | none | 0.5 carrot | 53.3 fertilizer |
| 25–29 | 8, 8, 8, 8, 8 | none | 0.2 carrot | 293.7 fertilizer, 150.0 melon, 210.4 milk, 42.6 carrot, 36.0 strawberry, 615.7 wool |

The witness postpones much inventory to the end because the valid all-drain-first curve makes timing free; this is a visible relaxation, not a proposed route.

Optional execution through `execute.py --workers 1 --tag bound-v2` banked **$3,736** with zero faults. The executor truncates the fractional quantities and cannot realize pooled/early-credit relaxations, so this is only a schema smoke test, not evidence against the bound.

## Binding constraints by day

- Cash floor: days **0–6 and 8**.
- Daily labour: days **7, 9–13, 15–17, 19, 21–22, 24–29**. The pooled season labour constraint also binds.
- Tiles: days **0–2, 4–28** (not 3 or 29).
- Market-order slots: **none**.

Machine-readable daily primal slacks and LP duals are in `schedule-v2.json` and `bound-details-v2.json`.

## Three largest remaining overstatements

1. **Empirical travel gap: about $55,041.** Replacing only the valid minimal travel floors with 1.842 moves/action lowers $381,690 to $326,649. This is diagnostic and overlaps routing/storage timing.
2. **All-drain-first market timing: $47,145 on the pinned reference schedule.** Holding its actual quantities and expenses fixed raises $76,009 to $123,154. This is the directly measured model looseness on a real pinned trajectory.
3. **Wheat carry safety allowance: at most $22,200.** It grants a fresh 100-unit position the maximum appreciation at every drain tick, ignoring cash, existing shed contents, and the need to alternate buy/sell orders. A feasible carry policy should capture only a fraction.

These estimates overlap and must not be added. Other known optimism—fractional tiles/hands, pooled care/feed, free fertilizer on crops, no shed overflow, and early revenue credit—remains, which is why $381.7k is a yardstick rather than a prediction.
