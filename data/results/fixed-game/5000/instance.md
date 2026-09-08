# Fixed Kaggriculture instance: seed 5000

Engine: kaggle-environments 1.32.7 / Kaggriculture 0.1.0. Source SHA-256: `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.

Seat 0, opponent `pass`, $3,000 starting bank, 30 days × 24 turns. The framework stores 720 observations (step 0 is initial), hence 719 action transitions.

## Pinned town

| Day | Shop | Products |
|---:|---|---|
| 3 | BRUNCH_SPOT | EGG, WHEAT, STRAWBERRY |
| 6 | SMOOTHIE_SHOP | STRAWBERRY, MILK |
| 9 | YARN_STORE | WOOL |
| 12 | PET_CAFE | CARROT |
| 15 | BRUNCH_SPOT | EGG, WHEAT, STRAWBERRY |
| 18 | PIZZA_SHOP | MILK, TOMATO, WHEAT |
| 21 | ICE_CREAM_SHOP | STRAWBERRY, MILK, WHEAT |
| 24 | ICE_CREAM_SHOP | STRAWBERRY, MILK, WHEAT |

Daily drain (town center plus all shops active that day):

| Day | WHEAT | CARROT | TOMATO | STRAWBERRY | MELON | EGG | MILK | WOOL | FERTILIZER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| 3 | 7 | 1 | 1 | 7 | 1 | 7 | 1 | 1 | 0 |
| 4 | 7 | 1 | 1 | 7 | 1 | 7 | 1 | 1 | 0 |
| 5 | 7 | 1 | 1 | 7 | 1 | 7 | 1 | 1 | 0 |
| 6 | 7 | 1 | 1 | 13 | 1 | 7 | 7 | 1 | 0 |
| 7 | 7 | 1 | 1 | 13 | 1 | 7 | 7 | 1 | 0 |
| 8 | 7 | 1 | 1 | 13 | 1 | 7 | 7 | 1 | 0 |
| 9 | 7 | 1 | 1 | 13 | 1 | 7 | 7 | 13 | 0 |
| 10 | 7 | 1 | 1 | 13 | 1 | 7 | 7 | 13 | 0 |
| 11 | 7 | 1 | 1 | 13 | 1 | 7 | 7 | 13 | 0 |
| 12 | 7 | 13 | 1 | 13 | 1 | 7 | 7 | 13 | 0 |
| 13 | 7 | 13 | 1 | 13 | 1 | 7 | 7 | 13 | 0 |
| 14 | 7 | 13 | 1 | 13 | 1 | 7 | 7 | 13 | 0 |
| 15 | 13 | 13 | 1 | 19 | 1 | 13 | 7 | 13 | 0 |
| 16 | 13 | 13 | 1 | 19 | 1 | 13 | 7 | 13 | 0 |
| 17 | 13 | 13 | 1 | 19 | 1 | 13 | 7 | 13 | 0 |
| 18 | 19 | 13 | 7 | 19 | 1 | 13 | 13 | 13 | 0 |
| 19 | 19 | 13 | 7 | 19 | 1 | 13 | 13 | 13 | 0 |
| 20 | 19 | 13 | 7 | 19 | 1 | 13 | 13 | 13 | 0 |
| 21 | 25 | 13 | 7 | 25 | 1 | 13 | 19 | 13 | 0 |
| 22 | 25 | 13 | 7 | 25 | 1 | 13 | 19 | 13 | 0 |
| 23 | 25 | 13 | 7 | 25 | 1 | 13 | 19 | 13 | 0 |
| 24 | 31 | 13 | 7 | 31 | 1 | 13 | 25 | 13 | 0 |
| 25 | 31 | 13 | 7 | 31 | 1 | 13 | 25 | 13 | 0 |
| 26 | 31 | 13 | 7 | 31 | 1 | 13 | 25 | 13 | 0 |
| 27 | 31 | 13 | 7 | 31 | 1 | 13 | 25 | 13 | 0 |
| 28 | 31 | 13 | 7 | 31 | 1 | 13 | 25 | 13 | 0 |
| 29 | 31 | 13 | 7 | 31 | 1 | 13 | 25 | 13 | 0 |

Fertilizer has no town drain. Every other product has center drain 1/day. Each active multi-product shop adds 6/day per listed product; each single-product shop adds 12/day.

### RNG coupling and pin

_end_of_day seeds random.Random((seed*1_000_003)^day), passes it through _spawn_weeds (one draw per empty unlocked tile), then rng.choice(sorted(SHOPS)); occupancy changes RNG position before choice.

run original end-of-day including weed RNG and original choice, then replace only a newly appended shop with this list's shop for that unlock day. This preserves weeds and every other rule while making shops action-independent.

## Map

Coordinates use `(x,y)`, origin northwest. NW (x=0..4,y=0..4) is initially owned. NE costs $1,000, SW $2,000, SE $4,000 and must be bought in that order. Each is 25 tiles. The central shed is not a tile; access tiles are (4,4), (5,4), (4,5), (5,5). Market and water are non-spatial.

```text
      x=0 1 2 3 4 | 5 6 7 8 9
y=0..4   NW owned | NE locked ($1,000)
          ---------+------------------
y=5..9  SW locked  | SE locked
         ($2,000)  | ($4,000)
```

Every tile is enumerated in `instance.json` with coordinate, quadrant, initial type/ownership, and quadrant unlock price.

## Calendar and action economy

Within a turn the engine applies unit actions, processes up to 10 market-order slots (bulk quantities are allowed), applies town consumption on its cadence, decays plants, then performs the day refresh after hour 23. At refresh it updates crops and animals, rolls weeds, drops all carried inventory into the 100-unit shed (discarding overflow), removes hired hands, respawns the farmer, resets daily hire count, then unlocks a shop when due.

One farmer is permanent. Each hand lasts one day and also gets 24 actions. Daily hires cost Fibonacci(index): 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610. One orthogonal move or farm operation costs one unit turn; market orders cost no unit turn. Buildings are free but BUILD costs one turn.

## Crops

| Crop | Seed | First yield age | Max-yield age | Interval | Cap | Ongoing |
|---|---:|---:|---:|---:|---:|---|
| WHEAT | 10 | 2 | 4 | 0 | 6 | False |
| CARROT | 20 | 2 | 3 | 0 | 4 | False |
| TOMATO | 50 | 8 | 8 | 1 | 4 | True |
| STRAWBERRY | 100 | 10 | 10 | 2 | 4 | True |
| MELON | 80 | 10 | 12 | 0 | 6 | False |

Planting starts with one missed-water count; a second consecutive missed end-of-day turns it into a weed. One-time crops begin at 1 unit and gain on watered days in the latter half of the growth window (fertilizer makes the gain 2), capped at max_yield. Ongoing crops schedule up to four productions; base is 1 and a currently fertilized, watered production is 2. Fertilizer lasts the application day plus two. At lifespan, held yield decays by one every two turns until weed.

## Animals

| Animal | Buy | Structure | Structure $ | First yield age | Interval | Held cap | Product |
|---|---:|---|---:|---:|---:|---:|---|
| GOOSE | 300 | COOP | 0 | 4 | 1 | 4 | EGG |
| COW | 400 | PASTURE | 0 | 8 | 2 | 6 | MILK |
| SHEEP | 500 | PASTURE | 0 | 6 | 3 | 6 | WOOL |

Each animal consumes one wheat on a FEED action per serviced day; two consecutive missed feeds makes it escape. Fed+CARE banks +1 for the next fed production, where the whole bank joins base yield subject to the held cap. If harvested each tick and fed+cared daily, first/later tick yields are goose 4/2, cow 6/3, sheep 6/4 (production consumes the old care bank before today's care is banked). Every surviving animal exposes one fertilizer after each day refresh; uncollected fertilizer does not stack.

## Exact price functions

For inventory `I`, `x=|I-10000|`. Below I0: `round(base + amp_below*f_below(x))`; at/above I0: `round(base - amp_above*f_above(x))`; then floor at $1. `amp = target*base/f(T)`. Hinge uses `u=x/T; f=u+8*max(0,u-1)^2`. Python's integer rounding is the engine rounding.

| Product | Base | T | Scarcity shape/target/amp | Glut shape/target/amp |
|---|---:|---:|---|---|
| WHEAT | 25 | 400 | sqrt / 0.8 / 1 | log / 0.2 / 0.834172869 |
| CARROT | 35 | 450 | hinge / 1.0 / 35 | sqrt / 0.7 / 1.15494108 |
| TOMATO | 60 | 200 | hinge / 0.4 / 24 | sqrt / 0.6 / 2.54558441 |
| STRAWBERRY | 120 | 100 | sqrt / 0.7 / 8.4 | linear / 1.6 / 1.92 |
| MELON | 250 | 300 | log / 0.2 / 8.76100122 | sq / 3.6 / 0.01 |
| EGG | 50 | 332 | hinge / 0.4 / 20 | log / 0.2 / 1.72172085 |
| MILK | 160 | 122 | sqrt / 0.6 / 8.69143162 | linear / 1.6 / 2.09836066 |
| WOOL | 200 | 105 | log / 0.2 / 8.57736087 | sq / 3.2 / 0.0580498866 |
| FERTILIZER | 100 | 200 | linear / 0.4 / 0.2 | linear / 0.4 / 0.2 |

Sales at $1 pay $1 but do not add market inventory. Only wheat and fertilizer can be bought as products. Seeds cost the crop's seed price; animals cost the table price.
