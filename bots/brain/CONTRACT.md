# Brain v1 module contract

This is the stable seam between the Brain and independently measured modules. The
submission is packed into `main.py`; these names and signatures remain available
there even when development modules are inlined.

## Immutable turn view

`View.from_obs(obs) -> View` is called exactly once at the start of each turn.
`View` is a frozen value object containing:

- `player`, `step`, `day`, `hour`, and `turns_left`;
- `money`, `tiles`, `farmer`, `hands`, `unit_positions`, `unit_inventories`,
  `shed`, `seeds`, `unlocked_quadrants`, and `hires_today` for our farm;
- `market_inventory` and `market_prices` for the shared market;
- `town_shops` including duplicate shop instances;
- `opponent`: the visible opponent money, tiles, farmer, hands, unlocked land,
  and hires (never an invented opponent shed);
- derived tuples `plants`, occupied `animals`, `empty_structures`, `empty_tiles`,
  `weeds`, and `shed_access_tiles`;
- optional immutable `jobs`, populated by the coordinator immediately before labor
  dispatch as `(priority, (x, y), action, need_item, skip_if_has)` tuples. This lets a
  labor module route the complete coordinator graph without redefining livestock or
  shed logistics.

Coordinates are always `(x, y)`. Collections exposed by `View` are snapshots;
modules must not mutate them or retain mutable references to `obs`.

## Day plan

`season.plan_day(view: View) -> DayPlan`

`DayPlan` is a frozen value object, recomputed at the first observed turn of each
day and reused for that day. It contains `phase`, `target_hands`,
`target_quadrants`, `crop_weights`, `animal_targets`, `sell_floors`,
`liquidating`, and `reserve_cash`. Values are decisions, not actions.

## Swappable pure modules

The required fixed signatures are:

```python
labor.hire_count(view: View, plan: DayPlan) -> int
labor.dispatch(view: View, plan: DayPlan) -> dict[int, list]
planting.choose(view: View, plan: DayPlan) -> dict[tuple[int, int], str]
market.orders(view: View, plan: DayPlan) -> list[list]
season.plan_day(view: View) -> DayPlan
```

`hire_count` returns the number of `HIRE` orders to request this turn (normally
zero or the gap to the daily target). `dispatch` returns one legal-shaped unit
action for selected unit indices, where index 0 is the farmer and 1..N are hands;
the coordinator inserts `PASS` for missing indices. `planting.choose` maps empty
owned tiles to target crop or animal (`GOOSE`, `COW`, `SHEEP`). `market.orders`
returns ordered market actions and never more than 10. The coordinator may remove
orders made infeasible by the shared cash/order budget but never changes module
state, because modules have no state.

Module selection is by `config["modules"]`, mapping each role to a registered
implementation name. A Team B replacement implements one signature, is added to
that registry, and changes only its role's config value.

## Invariants

- Pure, deterministic, and side-effect-free: no file/network/console I/O, clock
  dependence, global mutation, or observation mutation.
- No randomness unless the module owns an explicit fixed seed in `config`; equal
  inputs must produce equal outputs.
- Each module completes within 30 ms; the whole policy targets p99 below 100 ms
  and must always remain below the engine's 1 s turn deadline.
- At most one action per unit, actions align exactly with current hand order, and
  at most 10 market orders are returned.
- Thresholds live in `config` with a one-line engine/economic justification.
- The top-level `agent(obs)` wraps planning and coordination in a safe `PASS`
  fallback and never raises.
- KPI collection observes decisions/results but cannot affect any module output.
- No tapes, replay lookup, recorded action sequences, or opponent identity keys.

## Compatibility

Modules may read only fields documented above. New `View`/`DayPlan` fields may be
added with defaults; existing fields and signatures are not removed or renamed
during v0. A replacement is accepted only after its component KPI and paired-game
result are reported.
