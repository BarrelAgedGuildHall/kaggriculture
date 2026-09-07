# Fixed-game economic bound

**Valid upper bound: $2,631,002. Realistic-routing relaxation: $2,493,148.** Both solved by SciPy/HiGHS; the first is the certified relaxation. The realistic number uses 1.842 measured Exec B moves per non-move action and is diagnostic, not a proof bound.

Exec B routing measurement under the pin: final bank $97,255; 3,782 moves / 2,053 non-move actions = 1.842. On actual day 20: 142 / 85 = 1.671.

## Why every valid-model constraint is a relaxation

- **Cash/no borrowing.** Daily cash starts at $3,000 and is nonnegative. Seeds, animals, Fibonacci wages, land, and bought feed are charged on their purchase day; revenue is credited on the sale day. Netting within a day ignores intra-day sequencing, which can only help. Bought feed is priced at the $1 engine floor and cannot be resold; actual bought wheat is never cheaper.
- **Production/growth.** Every crop receives its lifetime cap at its earliest legal harvest age and its tile is released immediately. Actual watering/fertilizer can only produce no earlier and no more. Each fed animal production tick is allowed its full held cap (4/6/6) immediately, which upper-bounds the base-plus-care ramp. Free cohort disposal relaxes escape/retirement.
- **Feed.** Cohorts alive on day d impose the unavoidable cumulative lower bound floor((age+1)/2) feeds; feeds are pooled across animals and may be bought at $1, both easier than real per-animal wheat handling. Feed actions still consume labour.
- **Tiles/quadrants.** At most 25 tiles per unlocked quadrant are occupied, and land follows NE→SW→SE at $1k/$2k/$4k. The model ignores weeds and gives accelerated crop release, so all real layouts map inside it.
- **Labour/travel.** Required actions include plant+harvest, animal build+pickup+place, production harvest, feed, and fertilizer collection. Water and CARE are omitted. The valid map-derived travel lower bound is zero because shed-access tiles are productive/spawn tiles and consecutive actions may occur without movement. Thus required turns never exceed real required turns. The realistic variant multiplies these actions by the measured routing ratio and loses the proof guarantee.
- **Orders/storage.** Active order classes plus individual HIRE/land orders are capped at the real 240 slots/day. Bulk quantities use one slot, as in-engine. Shed/inventory capacity and end-of-day overflow are omitted, enlarging feasibility.
- **Revenue.** For each product/day the concave block curve uses the highest real marginal quote in each block after *all* pinned drain through that day. It resets prior sales to zero every day and lets end-of-day drain precede sales. Both raise quotes. Block maxima upper-bound every rounded marginal; melon and fertilizer therefore retain explicit dead-stock pools rather than linear base-price revenue.
- **Legal wheat speculation.** The production model alone would miss BUY_PRODUCT/SELL carry trades. A separate $1,366,100 credit closes that gap: at most 100 product units can enter the shed per transition × 719 = 71,900; fertilizer has zero external drain and zero profitable round trip; each wheat unit is granted the maximum $19 price rise across all 282 pinned drain units. This deliberately double-counts timing opportunities and is therefore an upper allowance.
- **Integrality/hire cap.** Tiles, animals, hands, land and order indicators are fractional in the final LP, enlarging the real integer set. Forty prefix-ordered hire slots/day cover any economically relevant real plan: the 40th Fibonacci wage alone is $102,334,155, above a coarse all-tiles/all-products physical revenue ceiling for this 30-day instance.

## Binding resources by day

Valid model — zero cash: 0, 1, 2, 3, 4, 5, 6, 7, 8; labour: 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29; tiles: 0, 1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28.

Realistic model — zero cash: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9; labour: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29; tiles: 0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27.

The model is an LP relaxation, so the duals are defined. `schedule.json` records per-day labour value ($ gained per extra unit turn), tile value ($ per extra tile-day), cash-floor shadow cost, and primal binding flags. Values are local sensitivities of this relaxation.

| Day | Cash floor shadow | Labour $/turn | Tile $/tile-day |
|---:|---:|---:|---:|
| 0 | $1.46 | $0.74 | $58.37 |
| 1 | $6.64 | $0.00 | $531.41 |
| 2 | $1.00 | $0.81 | $451.48 |
| 3 | $3.94 | $0.72 | $0.00 |
| 4 | $0.62 | $0.40 | $77.45 |
| 5 | $2.44 | $0.86 | $353.96 |
| 6 | $0.31 | $0.35 | $0.00 |
| 7 | $0.23 | $0.46 | $36.30 |
| 8 | $0.15 | $0.62 | $329.85 |
| 9 | $0.00 | $0.54 | $316.92 |
| 10 | $0.00 | $0.54 | $316.92 |
| 11 | $0.00 | $0.54 | $316.92 |
| 12 | $0.00 | $0.54 | $316.92 |
| 13 | $0.00 | $0.54 | $316.92 |
| 14 | $0.00 | $0.54 | $316.92 |
| 15 | $0.00 | $0.54 | $894.83 |
| 16 | $0.00 | $0.54 | $108.67 |
| 17 | $0.00 | $0.54 | $510.92 |
| 18 | $0.00 | $0.54 | $1,148.25 |
| 19 | $0.00 | $0.54 | $142.59 |
| 20 | $0.00 | $0.54 | $316.92 |
| 21 | $0.00 | $0.54 | $1,308.32 |
| 22 | $0.00 | $0.54 | $108.67 |
| 23 | $0.00 | $0.54 | $316.92 |
| 24 | $0.00 | $0.54 | $1,342.25 |
| 25 | $0.00 | $0.54 | $108.67 |
| 26 | $0.00 | $0.54 | $475.66 |
| 27 | $0.00 | $0.54 | $1,182.88 |
| 28 | $0.00 | $0.33 | $88.12 |
| 29 | $0.00 | $0.54 | $0.00 |

## Bank curves

| Day | Valid UB | Realistic routing |
|---:|---:|---:|
| 0 | $0 | $0 |
| 1 | $0 | $0 |
| 2 | $0 | $0 |
| 3 | $0 | $0 |
| 4 | $0 | $0 |
| 5 | $0 | $0 |
| 6 | $0 | $0 |
| 7 | $0 | $0 |
| 8 | $0 | $0 |
| 9 | $9,215 | $0 |
| 10 | $18,721 | $3,351 |
| 11 | $28,911 | $6,235 |
| 12 | $42,879 | $11,355 |
| 13 | $57,042 | $19,516 |
| 14 | $71,440 | $27,794 |
| 15 | $90,773 | $41,075 |
| 16 | $110,449 | $54,646 |
| 17 | $130,402 | $68,551 |
| 18 | $159,985 | $92,071 |
| 19 | $190,195 | $116,277 |
| 20 | $227,227 | $147,191 |
| 21 | $271,437 | $185,370 |
| 22 | $328,935 | $236,751 |
| 23 | $406,528 | $308,296 |
| 24 | $511,788 | $411,294 |
| 25 | $619,333 | $512,804 |
| 26 | $728,674 | $617,466 |
| 27 | $859,276 | $758,067 |
| 28 | $1,059,309 | $921,105 |
| 29 | $2,631,002 | $2,493,148 |

## Schedule summary (valid relaxation, five-day blocks)

| Days | Hires | Animals | Plantings | Sales |
|---|---:|---|---|---|
| 0–4 | 8.547883 | GOOSE 6.25, COW 18.7, SHEEP 7.30159 | WHEAT 144 | FERTILIZER 45.3508, WHEAT 371.52, EGG 25 |
| 5–9 | 27.939276 | SHEEP 43.1031, COW 30.8952 | none | WHEAT 161.523, EGG 75, FERTILIZER 322.006, MILK 1.45936, WOOL 39.4397 |
| 10–14 | 35.0 | none | none | MILK 81.54, FERTILIZER 500 |
| 15–19 | 33.849233 | none | none | MILK 350, FERTILIZER 500 |
| 20–24 | 33.0 | none | none | MILK 700, WOOL 475, FERTILIZER 500 |
| 25–29 | 30.462479000000002 | none | none | MILK 1400, WOOL 1478.35, FERTILIZER 500 |

## Largest overstatements (estimated)

1. Speculative-wheat safety allowance: **$1,366,100** is granted in full although cash and routing make 71,900 profitable carry units unattainable.
2. Daily market reset/end-of-day drain timing: expected overstatement **$120k–$300k**, largest in wool/milk and the dead-stock fertilizer/melon pools.
3. Full held-cap animal yield without CARE plus accelerated crop caps: expected **$100k–$250k**.

These are sensitivity ranges, not certified decompositions; overlapping mechanisms mean they must not be summed.
