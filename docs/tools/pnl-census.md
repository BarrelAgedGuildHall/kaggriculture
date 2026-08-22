# The P&L census — `tools/pnl_census.py`

Shared measurement tooling: a per-line income statement and terminal balance sheet for one
Kaggriculture episode, aggregated over a seed battery. **Teams compete on agents, never on books**
(plan `2026-08-21-balance-sheet-program/plan.md` §6a.5) — both teams read the same statement, so a
line's contribution means the same thing in a Team A review and a Team B one.

It is a diagnostic. It changes no agent code and gates nothing on its own; it is the input to
allocation decisions (which assets to own, which action to spend next).

## Run it

```bash
uv run python tools/pnl_census.py --a millrace:feedchain --b agents/team_a/bc \
    --seeds 0-7 --workers 8 --out-dir artisans/<you>/bench/runs/<exp>
```

Writes `pnl_seed<N>.json` per episode plus `pnl_aggregate.json`, and prints the report. Agent specs
are resolved by `tools/agent_spec.py`: `pass|random|starter`, a `.py` path, a directory containing
`main.py`, `probe:<Preset>` / `probe:{json}`, `millrace:...`. `--swap-half` plays the back half of
the seed list with the seats swapped.

## How each number is produced

| Block | Method |
|---|---|
| revenue / purchases | exact `_commit_unit` interception (SELL / BUY_PRODUCT / BUY_SEED / BUY_ANIMAL), attributed to a player by farm identity rebuilt at the top of every `interpreter` call |
| production | units read off the tile *before* the HARVEST lands — exact, never inferred from sales |
| tile-days | every owned tile sampled at hour 0 of each day, keyed by the line occupying it |
| actions | `_apply_unit_action` is patched and the tile diffed **across that single call**, so hour-23 actions are classified correctly (a diff around `env.step` misclassifies all of them: the end-of-day refresh runs inside the same step) |
| terminal | cash, shed + carried inventory walked down the sell curve, unharvested pipeline, seeds at cost, capital written to zero on day 29 |

## The two corrections that define this version

Both from `2026-08-21-team-b-business-lens/ruling-02-m4-and-sprint2-order.md`.

### §3.3 — fertilizer is a joint product, not a line

The engine sets `fertilizer_available` on an animal tile **unconditionally** on the animal's
schedule, and `COLLECT_FERTILIZER` converts one available tile into exactly one unit (it is refused
on a tile with no animal). Crediting that revenue to a standalone FERTILIZER line while charging
the feed bill to the animals understates every animal line — and goose most, since geese are the
cheapest animal-days per dollar of capital and therefore emit the most fertilizer per dollar
invested. A herd-sizing allocator sorting on the old table could cull the herd and be surprised
when a fifth of gross revenue left with it.

So:

- every executed collection is attributed to **the animal on the collected tile**, read before the
  action — exact, not apportioned;
- fertilizer sales revenue is allocated to the animal lines **pro rata by units collected**;
- the **collection actions move onto those lines**, which is the "net of its own collection
  actions" term: it rides in the `contribution_per_action` denominator, so the netting needs no
  assumed opportunity-cost constant (and no circular λ);
- if fertilizer was also *bought*, only `collected / (collected + bought)` of the revenue is a joint
  product; the rest is resale of purchased stock and stays on the memo line;
- `FERTILIZER` survives as a **memo line** (`joint_product: true`, printed with a `*`) carrying the
  gross figures in `memo_gross_revenue` / `memo_units_sold`.

The allocation redistributes revenue and never creates it: the sum of the lines' `gross_revenue`
equals the statement's `gross_revenue`, and that identity is asserted by the tests.

Purchased **fertilizer spend** is charged symmetrically, pro rata by FERTILIZE event, to the crop
lines that received it. An agent that buys fertilizer and issues no FERTILIZE reports the spend in
`unallocated_input_spend` rather than as a silent zero.

### §3.1 — gross and contribution are different denominators of value

They must never be compared with each other, so they are separately labelled everywhere:

| Per line | Meaning |
|---|---|
| `sales_revenue` | own-product SELL dollars |
| `fertilizer_revenue_allocated` | joint-product share (animal lines) |
| `gross_revenue` | `sales_revenue + fertilizer_revenue_allocated` |
| `direct_cost` | seed + charged feed + charged fertilizer (cash inputs only) |
| `contribution` | `gross_revenue - direct_cost` |
| `gross_per_action` / `contribution_per_action` | the two denominators, printed as `$g/act` and `$c/act` |
| `gross_per_tile_day` / `contribution_per_tile_day` | same pair per tile-day |

The statement carries a `lambda_opp` block:

- `reference` = total contribution / productive actions. This is the number a marginal task must
  beat when the action budget is fully spent, and it is the same definition the agent's live ledger
  uses (`(gross_today - cogs_today) / productive_actions`). **Quote this, not gross/action.**
- `gross_per_action` beside it, so the gap is visible rather than conflated.
- `ladder` — every executing line sorted by contribution/action with its action share and the
  cumulative share, i.e. the budget cut, readable.
- `marginal_line` / `marginal_contribution_per_action` — the lowest contribution/action line still
  consuming ≥1% of the budget: the task a dispatch pricer would cut first. A line below that share
  is a rounding error, not a trade-off, and is never named as marginal.

## Reconciliation caveats (read before quoting a line)

- `total_contribution` is a **transfer view**, not the bank. It excludes capex, hire and land, and
  it charges purchased wheat to the animals as feed while the WHEAT line still books its own sales.
  Bank is `terminal.cash`.
- Feed is charged at the season's realised **average purchased** wheat price, split pro rata by FEED
  event; `feed_charged_margin` is the alternative view (every FEED event at the purchase price, i.e.
  grown wheat charged at its opportunity cost).
- The WHEAT line's `internal_transfer_revenue` is a memo: never banked, never in `gross_revenue`.

## Schema note for existing readers

The column names are new (`sales_revenue` / `gross_revenue` / `gross_per_action` /
`contribution_per_action` replace `revenue` / `rev_per_action` / `contrib_per_action`). That is
deliberate: after §3.3 an animal line's revenue **means something different**, and keeping the old
names would have made a changed number look unchanged. Readers of the old
`agents/team_b/tools/pnl_census.py` output must switch keys along with the tool.
