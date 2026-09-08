# Drain-matched portfolio — extended hindsight executor

Engine 1.32.7. **Result: $86,494 mean, $74,032 worst** on seeds 5000–5011. This missed the $130k / $100k target. The final search completed 744 three-seed attempts and 248 full 12-seed evaluations; the selected plan first appeared 133 full evaluations before the end and was not improved again (time-boxed plateau). Four workers maximum were used. Every `KAGGRI_ARENA_DIR` was absolute (`C:\Guildhall\guilds\kaggler\workshop\repo\data\arena-ext2-*`).

## What is known and how capacity changes

At day 0, `obs.town.unlocked_shops` is empty. The executor knows the centre's one-unit/day pull, the eight future unlock dates, the uniform eight-shop census, product membership, single-product 2× pull, duplicate-instance rule, and 4-turn shop cadence; it does not know future shop identities. `drain_forecast(obs)` uses the exact identities already visible and expected membership for unrevealed slots. Crops use this remaining-season forecast because blocks turn over. Livestock is irreversible, so it uses only the exact revealed run-rate and stops adding animals when a later forecast is lower. A new shop changes targets on the next turn. A zero ratio always disables its line.

Worked example: seed 5005 at day 12 showed `SMOOTHIE_SHOP, ICE_CREAM_SHOP, SMOOTHIE_SHOP, PET_CAFE`. Exact revealed daily drain was milk 19, strawberry 19, carrot 13, and 1 for wool/egg/melon; remaining-season forecasts including four unknown shops were milk 24.25, strawberry 26, carrot 18.25, wool/egg 4.5, melon 1. The selected ratios produced herd targets **14 cows, 1 sheep, 0 geese** and block targets **64 wheat, 39 strawberry, 3 carrot, 2 melon, 0 tomato**. The farm had 14 cows/1 sheep at the snapshot; physical acreage and crop turnover constrain simultaneous live blocks.

## Backward compatibility

The unmodified v1 seed-3910 plan returned **$136,554**, status `DONE`, through `executor_ext.load_agent`, exactly matching Cycle 5.

## Solo, pass opponent

| seed | raw brain_strawberry | exec_b | ext2 |
|---:|---:|---:|---:|
| 5000 | $105,379 | $84,296 | **$104,241** |
| 5001 | $137,630 | $86,508 | **$87,545** |
| 5002 | $126,399 | $95,749 | **$95,114** |
| 5003 | $119,500 | $93,892 | **$84,218** |
| 5004 | $124,935 | $95,826 | **$87,581** |
| 5005 | $127,687 | $93,497 | **$95,835** |
| 5006 | $106,494 | $83,142 | **$74,915** |
| 5007 | $118,657 | $61,058 | **$75,171** |
| 5008 | $136,614 | $88,950 | **$97,285** |
| 5009 | $137,165 | $66,979 | **$74,032** |
| 5010 | $131,417 | $83,328 | **$84,704** |
| 5011 | $110,389 | $74,801 | **$77,283** |
| **mean** | **$123,522** | **$84,002** | **$86,494** |
| **worst** | **$105,379** | **$61,058** | **$74,032** |

Product revenue on milk-heavy seed 5005 (three Smoothie, three Ice Cream): milk **$84,068**/275 units, wool $4,223/19, fertilizer $22,211/332, strawberry $4,303/19, melon $2,985/11, wheat $1,326/47, carrot $633/18, egg $0. No zero-milk draw occurred in the fixed 12-seed panel; the closest was seed 5009, whose only milk shop (Pizza) appeared sixth on day 18: milk **$10,374**/47, wool $49,865/206, fertilizer $20,232/281, strawberry $6,009/30, wheat $3,252/103, melon $3,257/12, carrot $635/18, egg $0.

## Contested against V46, alternating seat

| seed | seat | ext2 / V46 / margin | raw / V46 / margin |
|---:|---:|---:|---:|
| 3910 | 0 | $65,686 / $109,633 / **−$43,947** | $67,994 / $115,084 / −$47,090 |
| 3911 | 1 | $42,006 / $104,538 / **−$62,532** | $64,932 / $81,013 / −$16,081 |
| 3912 | 0 | $74,354 / $160,381 / **−$86,027** | $79,978 / $120,560 / −$40,582 |
| 3913 | 1 | $61,759 / $120,406 / **−$58,647** | $92,788 / $119,711 / −$26,923 |
| 5000 | 0 | $27,038 / $101,460 / **−$74,422** | $101,665 / $131,063 / −$29,398 |
| 5001 | 1 | $79,115 / $160,167 / **−$81,052** | $104,736 / $136,308 / −$31,572 |
| 5002 | 0 | $25,901 / $118,121 / **−$92,220** | $65,853 / $82,362 / −$16,509 |
| 5003 | 1 | $58,952 / $152,968 / **−$94,016** | $66,474 / $92,830 / −$26,356 |

All 12 ext2 solo games, 12 raw solo games, 12 exec_b solo games, eight ext2/V46 games, eight raw/V46 controls, and the legacy control finished `DONE`: **zero faults**. The conservative worst-game p99 was **3.79 ms**; maximum observed ext2 turn was **214.64 ms**, below the 1 s limit.

## Largest remaining loss

The selected plan still sells **zero eggs on every seed**. Seed 5009 had four egg shops (three Bakery plus Brunch), yet ext2 earned $0 from eggs and banked $74,032 versus raw's $137,165: a **$63,133 solo loss**. Ratio search repeatedly rejected geese because the inherited placement/service path displaced more dairy and crop work than eggs returned. Contested exposure remains worse: on seed 5003 the ext2 margin was −$94,016 versus raw's −$26,356, a $67,660 deterioration.

The single next change should be a dedicated goose/egg service lane (correct coop placement, feed/care/harvest/deposit capacity reserved per goose) before searching portfolio ratios again.
