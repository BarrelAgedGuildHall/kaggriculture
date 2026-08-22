"""The balance sheet: one pure `mark_*` function per plan §1 row, and the Sheet they compose.

Lifted from exp-N1's `nav_probe.py`, which is the reference implementation and stays the
offline oracle. Two things changed in the lift, both required by S-1:

1. **No engine import.** The probe took the engine module for its rulebook tables and its
   `market_price`; here the tables come from `nav.rules` and every price comes from a curve
   *fitted at runtime* (`nav.curves`).
2. **The lines follow the plan's rows, not the probe's composite.** The probe folded the
   fertilizer entitlement and the feed liability into `mark_animal`; they are their own lines
   here (rows 10a and 4), because a caller has to be able to see a liability that a
   value-destroying asset is hiding. The totals are the same.

The horizon, and why the terminal identity is not a special case
---------------------------------------------------------------
`H(i) = 719 - i` is the number of turns in which any further action can happen (exp-N1 §1(a):
a SELL still executes on step 718, so the terminal snapshot is index **719**, not 718).
`T_ticks(d) = 29 - d` is the number of remaining end-of-day ticks; day 29's never runs, so
from day 29's morning there is no production, no animal refresh and no free auto-drop of
carried inventory into the shed.

Every mark asks what its asset can still deliver **within H**. At `H = 0` every non-cash line
is therefore zero *by the same formula that values it at H = 700* -- there is no
`if last_day:` branch anywhere in this file, and `tests/test_nav.py` contains a static check
(T1b) that fails the suite if one appears. Writing the terminal case by hand would make
`Phi(terminal) == 0` unfalsifiable, which is exactly what exp-N1's Test 1 exists to detect.

Two engine facts the marks encode, both measured, both contradicting the original spec:
  * **There is no weed hazard on a planted tile.** `_spawn_weeds` converts only empty tiles;
    a plant dies from `consecutive_unwatered >= 2` and nothing else. So `P(survive) = 1`
    while the watering obligation is met, and 0 when a second miss is already unavoidable.
  * **A one-time crop starts at `yield_units = 1`** when planted, and each WATER inside the
    bonus window adds +1 (+2 fertilized), capped at `max_yield`.

`nav` never decides anything: these functions answer *what is this worth*. What to do about
the number stays in the team's own agent code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .book import DemandBook
from .curves import CurveSet
from .lam import LambdaPair
from .rules import (ANIMALS, CROPS, LAST_DAY, PRICE_FLOOR, SHED_CAPACITY, TERMINAL_INDEX,
                    TURNS_PER_DAY, hours_left, shed_access_tiles, ticks_left)

VARIANTS = ("spot", "walk", "drain")


# ------------------------------------------------------------------------------ the pool
class LiquidationPool:
    """The sell side of the market as it is walked down by one marking pass.

    Statefulness is the point: marking every line against the untouched spot price is
    `M-spot`, and the whole question exp-N1 asked is how much that overstates. One pool is
    created per `Sheet`, lives in `MarkContext.pool`, and is the only mutable thing a mark
    touches -- the marks never mutate their own inputs (test T6).

    Variants (exp-N1 §3):
      * `spot`  -- every unit at the current quoted price; no walk, no drain.
      * `walk`  -- the closed-form curve integral from wherever this pass has walked the pool.
      * `drain` -- as `walk`, but a unit delivered on a LATER day is sold into a pool the town
                   has already eaten, so the pool is advanced by the remaining town drain the
                   first time a future delivery of that product is marked.
    """

    def __init__(self, curves: CurveSet, book: DemandBook, market, step: int,
                 variant: str = "walk") -> None:
        inv = market["inventory"] if isinstance(market, dict) else market.inventory
        prices = market["prices"] if isinstance(market, dict) else market.prices
        self.curves = curves
        self.book = book
        self.variant = variant
        self.step = int(step)
        self.inv = {k: int(v) for k, v in inv.items()}
        self.spot = {k: int(v) for k, v in prices.items()}
        self._drained: set[str] = set()
        self.provisional = False

    def take(self, item: str, n: int, future: bool = False) -> float:
        """Proceeds of selling `n` units of `item` given everything already taken this pass."""
        n = int(n)
        if n <= 0 or item not in self.spot:
            return 0.0
        if self.variant == "spot":
            return float(self.spot.get(item, PRICE_FLOOR)) * n
        curve = self.curves[item]
        if curve.confidence <= 0.0:
            self.provisional = True
        if self.variant == "drain" and future and item not in self._drained:
            self._drained.add(item)
            self.inv[item] = self.inv.get(item, curve.i0) - int(
                self.book.remaining_drain(item, self.step, TERMINAL_INDEX))
        inv = self.inv.get(item, curve.i0)
        value = curve.marginal_revenue(inv, n)
        self.inv[item] = inv + curve.priced_units(inv, n)
        return value

    def buy(self, item: str, n: int) -> float:
        """Cost of BUYING `n` units (row 10b: the pool is two-sided). Buying moves it the
        other way; the engine quotes at `inventory - 1` and then decrements."""
        n = int(n)
        if n <= 0 or item not in self.spot:
            return 0.0
        curve = self.curves[item]
        if self.variant == "spot":
            return float(self.spot.get(item, PRICE_FLOOR)) * n
        if curve.confidence <= 0.0:
            self.provisional = True
        inv = self.inv.get(item, curve.i0)
        cost = curve.marginal_cost(inv, n)
        self.inv[item] = inv - n
        return cost


# ------------------------------------------------------------------------------ records
@dataclass(frozen=True)
class Line:
    """One row of the sheet, in dollars. Liabilities are negative lines, not a second side."""
    name: str
    value: float
    detail: Mapping[str, float] = field(default_factory=dict)
    provisional: bool = False

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value,
                "detail": {k: v for k, v in self.detail.items()},
                "provisional": self.provisional}


@dataclass(frozen=True)
class MarkContext:
    """The immutable context every mark needs -- the thing that makes the marks pure.

    `pool` is the one deliberate exception: it is the liquidation accumulator for THIS
    marking pass (see LiquidationPool), created fresh per Sheet and never shared.
    """
    step: int
    day: int
    t_ticks: int
    hours_left: int
    curves: CurveSet
    book: DemandBook
    lam: LambdaPair
    variant: str
    pool: LiquidationPool
    liquidation_order: tuple[str, ...]
    shed_capacity: int = SHED_CAPACITY
    cash: float = 0.0             # additive: mark_structures needs to know a purchase is affordable
    shed_units: int = 0           # additive: BUY_ANIMAL into a full shed is silently refused


@dataclass(frozen=True)
class Sheet:
    step: int
    day: int
    seat: int
    variant: str
    lam: LambdaPair
    cash: float
    lines: tuple[Line, ...]
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    schema_version: int = 1

    @property
    def nav(self) -> float:
        return self.cash + sum(l.value for l in self.lines)

    @property
    def phi(self) -> float:
        """`nav - cash`: everything the seat owns that is not yet money."""
        return sum(l.value for l in self.lines)

    @property
    def provisional(self) -> bool:
        return any(l.provisional for l in self.lines)

    def line(self, name: str) -> Line | None:
        for l in self.lines:
            if l.name == name:
                return l
        return None

    def as_dict(self) -> dict:
        """JSON-safe; this is the S-3 telemetry record."""
        return {
            "schema_version": self.schema_version, "step": self.step, "day": self.day,
            "seat": self.seat, "variant": self.variant, "lam": self.lam.as_dict(),
            "cash": self.cash, "nav": self.nav, "phi": self.phi,
            "provisional": self.provisional,
            "lines": {l.name: l.value for l in self.lines},
            "detail": {l.name: dict(l.detail) for l in self.lines},
            "diagnostics": dict(self.diagnostics),
        }


# ------------------------------------------------------------------------------ helpers
def shed_distance(pos) -> int:
    return min(abs(pos[0] - x) + abs(pos[1] - y) for x, y in shed_access_tiles())


def _keepalive_waters(day: int, until_day: int) -> int:
    """A plant survives one missed night, so watering every other day keeps it alive."""
    return math.ceil(max(0, until_day - day) / 2)


def _in_window_water_days(tile, day: int) -> list[int]:
    """Days on which watering this one-time crop still adds yield."""
    cd = CROPS[tile["crop"]]
    start = tile["planted_day"] + (cd["max_yield_day"] + 1) // 2
    end = tile["planted_day"] + cd["max_yield_day"]
    lo = max(day, start)
    if tile.get("watered_today"):
        lo = max(lo, day + 1)              # today's water is already spent
    return list(range(lo, min(end, LAST_DAY) + 1))


def _tiles(farm):
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict):
                yield tile


# ------------------------------------------------------------------------------- the marks
def mark_cash(farm) -> Line:                                                    # row 1
    """Face value. The only terminal truth."""
    return Line("cash", float(farm["money"]))


def mark_shed(shed, ctx: MarkContext) -> Line:                                  # rows 2, 12
    """Shed stock, walked in the measured liquidation order (steepest pool first).

    Sellable iff at least one turn remains: ten market orders fit in a single turn and there
    are nine products, so one turn liquidates the whole shed. At `H = 0` this is zero -- the
    terminal identity falling out of the formula, not out of a special case.
    """
    detail: dict[str, float] = {}
    if ctx.hours_left < 1 or not shed:
        return Line("shed", 0.0, detail)
    order = [p for p in ctx.liquidation_order if p in shed]
    order += [p for p in shed if p not in ctx.liquidation_order]
    for item in order:
        n = int(shed.get(item, 0) or 0)
        if n > 0 and item in ctx.pool.spot:
            detail[item] = ctx.pool.take(item, n, future=False)
    return Line("shed", sum(detail.values()), detail, provisional=ctx.pool.provisional)


def mark_carried(inventories, farm, ctx: MarkContext) -> Line:                  # row 2, row 14
    """Goods held in unit inventories. They must reach a shed-access tile to be sellable.

    Free while an end-of-day tick remains (`_drop_inventories_to_shed` runs for nothing);
    after that they cost the walk to the shed plus one DROP action, and a unit that cannot
    reach the shed inside `H` is carrying worthless cargo -- the day-29 trap, priced.
    """
    total, detail = 0.0, {}
    positions = [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands", [])]
    free_drop = ctx.t_ticks >= 1
    for u, inv in enumerate(inventories or []):
        if not inv or not any(int(v) > 0 for v in inv.values()):
            continue
        pos = positions[u] if u < len(positions) else shed_access_tiles()[0]
        distance = shed_distance(pos)
        if not free_drop:
            if distance + 1 > ctx.hours_left:
                continue                                  # cannot reach the shed before the bell
            total -= ctx.lam.opp                          # one DROP action
        for item, n in inv.items():
            if item in ctx.pool.spot and int(n) > 0:
                v = ctx.pool.take(item, int(n), future=False)
                detail[item] = detail.get(item, 0.0) + v
                total += v
    return Line("carried", total, detail, provisional=ctx.pool.provisional)


def mark_seeds(seeds, ctx: MarkContext) -> Line:                                # row 3
    """A seed is worth the NPV of the crop it can still grow, never its cost.

    Zero after the crop's last profitable planting day, which is what makes it go to zero at
    the horizon. (`pnl_census.py` marks seeds at `seeds_left_cost`; that is a COST view and
    the wrong mark for a balance sheet. Recorded, not reconciled.)
    """
    total, detail = 0.0, {}
    for crop, n in (seeds or {}).items():
        n = int(n or 0)
        if n <= 0 or crop not in CROPS:
            continue
        v = _npv_fresh_plant(crop, ctx)
        if v > 0:
            detail[crop] = v * n
            total += v * n
    return Line("seeds", total, detail, provisional=ctx.pool.provisional)


def _npv_fresh_plant(crop: str, ctx: MarkContext) -> float:
    """Value of planting one seed of `crop` today, on the minimal correct schedule."""
    cd = CROPS[crop]
    fake = {"kind": "PLANT", "crop": crop, "planted_day": ctx.day,
            "yield_units": 0 if cd["ongoing"] else 1, "watered_today": False,
            "consecutive_unwatered": 1, "fertilized_until_day": -1}
    return mark_crop(fake, ctx).value - ctx.lam.opp          # the PLANT action itself


def mark_crop(tile, ctx: MarkContext) -> Line:                                  # rows 5, 6, 15
    """Crop in the ground: units still deliverable by the bell, marked and charged.

    Row 6 (service obligations) is the `lam.opp` charge on every watering and harvest the
    schedule requires. Row 15 (weeds) is `P(survive)`: the engine spawns weeds only on EMPTY
    tiles, so a planted tile faces no hazard and `P(survive) = 1` -- but only because the
    keep-alive waterings are in the action count below, i.e. survival is *bought*, not
    assumed. If those actions do not fit inside `H` the whole mark is zero.
    """
    crop = tile.get("crop")
    if crop not in CROPS or ctx.hours_left <= 0:
        return Line("crop", 0.0, {})
    cd = CROPS[crop]

    if cd["ongoing"]:
        # Production fires at each end-of-day tick whose `next_day` hits the interval; day
        # 29's tick never runs, so `next_day` stops at LAST_DAY.
        events = 0
        for next_day in range(ctx.day + 1, LAST_DAY + 1):
            since = next_day - tile["planted_day"] - cd["first_yield_day"]
            if since < 0 or since % cd["interval"] != 0:
                continue
            if since // cd["interval"] + 1 > cd["max_yield"]:
                continue
            events += 1
        units = int(tile.get("yield_units", 0)) + events
        if units <= 0:
            return Line("crop", 0.0, {})
        # `yield_units` is capped at max_yield as a HELD quantity, so the extra units only
        # exist if a HARVEST is scheduled between events.
        harvests = max(1, math.ceil(units / cd["max_yield"]))
        actions = _keepalive_waters(ctx.day, LAST_DAY) + harvests
        if actions > ctx.hours_left:
            return Line("crop", 0.0, {})
        value = ctx.pool.take(crop, units, future=True) - ctx.lam.opp * actions
        return Line("crop", value, {crop: float(units), "actions": float(actions)},
                    provisional=ctx.pool.provisional)

    waters = _in_window_water_days(tile, ctx.day)
    units = int(tile.get("yield_units", 0))
    for d in waters:
        bonus = 2 if int(tile.get("fertilized_until_day", -1)) >= d else 1
        units = min(cd["max_yield"], units + bonus)
    harvest_day = max(ctx.day, tile["planted_day"] + cd["first_yield_day"])
    if waters:
        harvest_day = max(harvest_day, waters[-1])
    if harvest_day > LAST_DAY or units <= 0:
        return Line("crop", 0.0, {})
    actions = len(waters) + _keepalive_waters(ctx.day, harvest_day) + 1          # + HARVEST
    if harvest_day >= LAST_DAY:
        actions += 1                                                            # + manual DROP
    if actions > ctx.hours_left:
        return Line("crop", 0.0, {})
    value = ctx.pool.take(crop, units, future=(harvest_day > ctx.day)) - ctx.lam.opp * actions
    return Line("crop", value, {crop: float(units), "actions": float(actions)},
                provisional=ctx.pool.provisional)


@dataclass(frozen=True)
class _AnimalSchedule:
    """What one animal can still deliver inside H, and what it costs to get it.

    Shared by the three animal lines so they cannot disagree with each other: production
    (row 4/7), the fertilizer entitlement (row 10a) and the feed liability (row 4, negative).
    """
    feasible: bool
    units: int
    produce_days: int
    feed_days: int
    collect_days: int
    harvests: int
    actions: int


def animal_schedule(tile, ctx: MarkContext) -> _AnimalSchedule:
    animal = ANIMALS.get(tile.get("animal"))
    if animal is None or ctx.hours_left <= 0:
        return _AnimalSchedule(False, 0, 0, 0, 0, 0, 0)
    days = list(range(ctx.day, LAST_DAY))          # days whose END-OF-DAY tick still runs
    pending = int(tile.get("pending_care_bonus", 0) or 0)      # row 7: accrued care bonus
    produced = 0
    for d in days:
        since = (d + 1) - tile["placed_day"] - animal["first_yield_day"]
        if since >= 0 and since % animal["interval"] == 0:
            produced += 1 + (pending if d == ctx.day else 0)
            pending = 0
    units = int(tile.get("yield_units", 0) or 0) + produced
    feed_days = len(days)                          # 1 WHEAT + 1 FEED action per remaining day
    collect_days = len(days)                       # COLLECT_FERTILIZER, one per remaining day
    harvests = max(1, math.ceil(units / animal["max_held"])) if units else 0
    actions = feed_days + collect_days + harvests
    return _AnimalSchedule(actions <= ctx.hours_left, units, len(days), feed_days,
                           collect_days, harvests, actions)


def mark_animal(tile, ctx: MarkContext, schedule: _AnimalSchedule | None = None) -> Line:
    """Animal production stream, charged for the harvest actions it needs (rows 4, 6, 7).

    The fertilizer entitlement and the feed liability are their OWN lines (rows 10a and 4) --
    see `mark_fertilizer_entitlement` and `mark_feed_liability`. Together the three can be
    net negative, which is the point: a mark that clamps at zero cannot tell you that an
    asset is destroying value, and on day 29 an animal is exactly that.
    """
    s = schedule or animal_schedule(tile, ctx)
    animal = ANIMALS.get(tile.get("animal"))
    if animal is None or not s.feasible or s.units <= 0:
        return Line("animal", 0.0, {})
    revenue = ctx.pool.take(animal["product"], s.units, future=True)
    value = revenue - ctx.lam.opp * s.harvests
    return Line("animal", value, {animal["product"]: float(s.units),
                                  "harvests": float(s.harvests)},
                provisional=ctx.pool.provisional)


def mark_fertilizer_entitlement(tile, ctx: MarkContext,
                                schedule: _AnimalSchedule | None = None) -> Line:   # row 10a
    """1 unit of FERTILIZER per animal per day, perishable, bought with an action.

    **May go negative and is not clamped.** Once the FERTILIZER pool floors (measured around
    day 20-23) the entitlement is a ~$1 unit collected with a ~$50 action, and the sheet has
    to be able to say so.
    """
    s = schedule or animal_schedule(tile, ctx)
    if not s.feasible or s.collect_days <= 0:
        return Line("fertilizer", 0.0, {})
    revenue = ctx.pool.take("FERTILIZER", s.collect_days, future=True)
    value = revenue - ctx.lam.opp * s.collect_days
    return Line("fertilizer", value, {"units": float(s.collect_days),
                                      "revenue": revenue},
                provisional=ctx.pool.provisional)


def mark_feed_liability(tile, ctx: MarkContext,
                        schedule: _AnimalSchedule | None = None) -> Line:            # row 4
    """One WHEAT per animal per day at the walked BUY cost, plus the FEED action. Negative.

    Both players are structural wheat buyers, so this liability is priced on the same
    two-sided pool that our own wheat sales move.
    """
    s = schedule or animal_schedule(tile, ctx)
    if not s.feasible or s.feed_days <= 0:
        return Line("feed", 0.0, {})
    cost = ctx.pool.buy("WHEAT", s.feed_days)
    value = -(cost + ctx.lam.opp * s.feed_days)
    return Line("feed", value, {"wheat_units": float(s.feed_days), "wheat_cost": cost},
                provisional=ctx.pool.provisional)


def mark_structures(tiles, ctx: MarkContext) -> Line:                                # row 9
    """An empty coop/pasture is worth one action's option value iff an animal could still be
    placed in it: an end-of-day tick must remain for it to produce, cash must afford one, and
    the shed must have headroom -- `BUY_ANIMAL` into a full shed is silently refused."""
    n = 0
    for tile in tiles:
        if tile.get("kind") in ("COOP", "PASTURE") and "animal" not in tile:
            n += 1
    if not n or ctx.t_ticks < 1 or ctx.hours_left < 2:
        return Line("structures", 0.0, {})
    if ctx.cash < min(a["cost"] for a in ANIMALS.values()) or ctx.shed_units >= ctx.shed_capacity:
        return Line("structures", 0.0, {})
    return Line("structures", ctx.lam.opp * n, {"empty": float(n)})


def mark_land(tiles, ctx: MarkContext) -> Line:                                      # row 8
    """Zero, always. There is no SELL_LAND and no salvage; tile-days are a diagnostic, never
    a price. The line exists so the row is visibly accounted for rather than forgotten."""
    unlocked = sum(1 for t in tiles if not isinstance(t, str))
    return Line("land", 0.0, {"tile_days": float(unlocked * max(0, LAST_DAY - ctx.day))})


def covenant_charge(shed, ctx: MarkContext) -> float:                                # row 2
    """Units above the 100-slot shed covenant and what they are worth, cheapest destroyed
    first. Reported as a diagnostic and never netted into NAV -- a large unmodelled charge
    should show up as Test-1b optimism, not be quietly absorbed."""
    total = sum(int(v or 0) for v in (shed or {}).values())
    excess = max(0, total - ctx.shed_capacity)
    if not excess:
        return 0.0
    pool = LiquidationPool(ctx.curves, ctx.book, {"inventory": ctx.pool.inv, "prices": ctx.pool.spot},
                           ctx.step, variant="walk")
    value, left = 0.0, excess
    for item in reversed(ctx.liquidation_order):
        n = min(left, int(shed.get(item, 0) or 0))
        if n:
            value += pool.take(item, n)
            left -= n
        if not left:
            break
    return value


def intangible_diagnostic(ctx: MarkContext) -> float:
    """Reported, NEVER in nav (spec §3.8): the value of what the seat knows rather than owns
    -- here, the option value the demand book carries over the products we can still sell."""
    return sum(ctx.book.hinge_value(p, ctx.curves[p], ctx.pool.inv.get(p))
               for p in ctx.pool.spot)


# ------------------------------------------------------------------------------- the sheet
def build_sheet(farm, private, market, town, step: int, seat: int, curves: CurveSet,
                book: DemandBook, lam: LambdaPair, variant: str = "drain",
                shed_capacity: int = SHED_CAPACITY, public_only: bool = False) -> Sheet:
    """Compose every line for one seat at one sample point.

    `public_only=True` marks a seat whose `private` block we cannot see (the opponent, §5):
    shed, carried goods and seeds are dropped from the sheet and reported in
    `diagnostics["unmarkable_lines"]` instead of being guessed.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    day = int(step) // TURNS_PER_DAY
    private = private or {}
    shed = {} if public_only else dict(private.get("shed", {}) or {})
    shed_units = sum(int(v or 0) for v in shed.values())
    pool = LiquidationPool(curves, book, market, step, variant=variant)
    ctx = MarkContext(
        step=int(step), day=day, t_ticks=ticks_left(day), hours_left=hours_left(step),
        curves=curves, book=book, lam=lam, variant=variant, pool=pool,
        liquidation_order=curves.liquidation_order(pool.inv, day=day),
        shed_capacity=shed_capacity, cash=float(farm["money"]), shed_units=shed_units,
    )

    cash = mark_cash(farm)
    lines: list[Line] = []
    if not public_only:
        lines.append(mark_shed(shed, ctx))
        lines.append(mark_carried(private.get("inventories", []), farm, ctx))

    tiles = list(_tiles(farm))
    crops = _fold("crops", (mark_crop(t, ctx) for t in tiles if t.get("kind") == "PLANT"))
    animal_tiles = [t for t in tiles if "animal" in t]
    schedules = [(t, animal_schedule(t, ctx)) for t in animal_tiles]
    animals = _fold("animals", (mark_animal(t, ctx, s) for t, s in schedules))
    fertilizer = _fold("fertilizer", (mark_fertilizer_entitlement(t, ctx, s) for t, s in schedules))
    feed = _fold("feed", (mark_feed_liability(t, ctx, s) for t, s in schedules))
    lines += [crops, animals, fertilizer, feed]
    if not public_only:
        lines.append(mark_seeds(private.get("seeds", {}), ctx))
    lines.append(mark_structures(tiles, ctx))
    lines.append(mark_land([t for row in farm["tiles"] for t in row], ctx))

    diagnostics = {
        "covenant_charge": covenant_charge(shed, ctx) if not public_only else 0.0,
        "shed_units": float(shed_units),
        "hours_left": float(ctx.hours_left),
        "t_ticks": float(ctx.t_ticks),
        "curve_confidence": min((curves[p].confidence for p in pool.spot), default=0.0),
        "tripwire": float(len(curves.tripwire())),
    }
    if public_only:
        diagnostics["unmarkable_lines"] = "shed,carried,seeds"
    return Sheet(step=int(step), day=day, seat=int(seat), variant=variant, lam=lam,
                 cash=cash.value, lines=tuple(lines), diagnostics=diagnostics)


def _fold(name: str, lines) -> Line:
    total, detail, provisional = 0.0, {}, False
    for l in lines:
        total += l.value
        provisional = provisional or l.provisional
        for k, v in l.detail.items():
            detail[k] = detail.get(k, 0.0) + v
    return Line(name, total, detail, provisional)
