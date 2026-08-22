"""expB-304b-b0 — the landing-inventory projection (ruling-04 §2.1, R-36).

The question every acreage rule has to answer before it can price a tile:

> If I plant one more tile of `p` today, what will the pool look like on the day
> its units land — and therefore what will the marginal unit fetch?

Ruling-04 §2.1 states the projection:

    I_land(p) = I_now
              + own_pipeline_landing(p, L)
              + opponent_est_landing(p, L)
              - DemandBook.drain_over_steps(p, step+1, step+L)
              - expected_extra_unlock_drain(p, L)

Two standing instructions from the ruling are honoured literally here:

  * **Do not re-derive the drain.**  `nav.DemandBook` wraps `probe_core
    ._drain_per_day`, which is already validated; every drain term below is a
    `DemandBook` call and this module contains no drain arithmetic of its own.
  * **The unlock posterior is not optional.**  A strawberry landing in ten days
    faces the drain of a *future* unlock set, and that term is the only thing
    separating this projection from the constant it is meant to replace.  It is
    integrated day by day rather than applied at the end-state rate, because a
    draw that lands on day 26 does not drain the pool on day 20.

Nothing here decides anything.  It answers "what will the pool be" and
"how much of that is posterior rather than exact"; b1 is where a rule reads it.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nav.book import DemandBook                                   # noqa: E402
from nav.rules import (ANIMALS, CROPS, SHOPS, TERMINAL_INDEX,     # noqa: E402
                       TOWN_CENTER_PRODUCTS, TURNS_PER_DAY)

# Structural lead time, in STEPS, from "I commit a unit of production capacity"
# to "its first units are in my hands".  Straight off the engine's own tables --
# no constant of ours.  FERTILIZER is absent on purpose: it is a joint product
# with no lead of its own and (engine `_town_consume`) no town drain at all.
LEAD_STEPS = {}
for _crop, _c in CROPS.items():
    LEAD_STEPS[_crop] = int(_c["first_yield_day"]) * TURNS_PER_DAY
for _a in ANIMALS.values():
    LEAD_STEPS[_a["product"]] = int(_a["first_yield_day"]) * TURNS_PER_DAY


def per_draw_extra_rate(product: str, shop_interval: int) -> float:
    """Expected units/day added to `product`'s drain by ONE further unlock draw.

    The draw is uniform with replacement over the eight shops (`_advance_day`),
    and a single-product shop eats double — the same two facts `DemandBook`
    encodes.  Kept as a helper rather than a magic number so the posterior below
    can be integrated over the window instead of applied at the end rate.
    """
    if product not in TOWN_CENTER_PRODUCTS:
        return 0.0
    per_draw = 0.0
    for products in SHOPS.values():
        if product in products:
            per_draw += (2 if len(products) == 1 else 1) / len(SHOPS)
    return per_draw * (TURNS_PER_DAY / max(1, shop_interval))


def expected_extra_unlock_drain(book: DemandBook, product: str, step: int, lead: int) -> float:
    """Units the *not yet drawn* shops are expected to take out of the pool over
    `(step, step + lead]`.

    Integrated day by day: on day d the expected extra rate is
    `per_draw_rate * (draws that have landed by d)`, and a draw scheduled for
    day 27 contributes nothing to a window that closes on day 22.  Applying
    `DemandBook.expected_extra_drain_per_day` flat across the window (the
    obvious shortcut) over-counts by roughly a factor of two on a long lead,
    which on STRAWBERRY is the difference between the rule growing and not.
    """
    if product not in TOWN_CENTER_PRODUCTS or lead <= 0:
        return 0.0
    rate = per_draw_extra_rate(product, book.shop_interval)
    if rate <= 0.0:
        return 0.0
    day_now = step // TURNS_PER_DAY
    draws_now = book.remaining_draws(day_now)
    if draws_now <= 0:
        return 0.0
    total = 0.0
    last_step = min(int(step) + int(lead), TERMINAL_INDEX)
    for s in range(int(step) + 1, last_step + 1):
        d = s // TURNS_PER_DAY
        landed = draws_now - book.remaining_draws(d)
        if landed > 0:
            total += rate * landed / TURNS_PER_DAY
    return total


def curve_state(curve) -> dict:
    """A frozen, exact snapshot of a `nav.PriceCurve`'s fit at this instant.

    Recorded alongside every projection so the back-test can later separate two
    very different failures that produce the same wrong quote:

      * the **inventory** model was wrong (the drain/pipeline arithmetic), or
      * the inventory was right and the **curve extrapolation** was wrong.

    Without this decomposition a failed G-b0.1 is uninterpretable — it would
    condemn the drain-aware rule for an error in a fitted curve the rule merely
    reads.  The snapshot is exact rather than a grid: `price_at` is
    `base -+ c * shape(f, |inv - i0|, T)` and those five numbers per side are the
    whole model.
    """
    curve.below.fit()
    curve.above.fit()
    return {
        "base": curve.base, "i0": curve.i0,
        "below": [curve.below.shape, curve.below.T, curve.below.c],
        "above": [curve.above.shape, curve.above.T, curve.above.c],
        "n_points": curve.n_points,
        "obs_min": (min(curve._observed) if curve._observed else None),
        "obs_max": (max(curve._observed) if curve._observed else None),
    }


def price_from_state(state: dict, inventory: float) -> float:
    """`nav.PriceCurve.price_at` replayed from a `curve_state` snapshot."""
    from nav.curves import shape as _shape           # noqa: PLC0415
    from nav.rules import PRICE_FLOOR                # noqa: PLC0415
    base, i0 = state.get("base"), state.get("i0")
    if base is None:
        return float(PRICE_FLOOR)
    d = abs(float(inventory) - float(i0))
    if d == 0:
        return float(base)
    func, T, c = state["above"] if inventory > i0 else state["below"]
    if func is None:
        return float(base)
    delta = c * _shape(func, d, T)
    p = base - delta if inventory > i0 else base + delta
    return max(float(PRICE_FLOOR), p)


@dataclass
class Projection:
    """One (product, step -> step+lead) projection, with its terms exposed.

    Every term is reported separately on purpose: G-b0.1 can fail because the
    drain is wrong, because our own pipeline estimate is wrong, or because the
    opponent term is wrong, and a single number cannot tell those apart.
    """
    product: str
    step: int
    lead: int
    land_step: int
    inv_now: int
    spot: float
    own_pipeline: float = 0.0
    opponent_est: float = 0.0
    exact_drain: float = 0.0
    posterior_drain: float = 0.0
    inv_land: float = 0.0
    quote_land: float = 0.0
    provisional: bool = False
    curve: dict = field(default_factory=dict)
    terms: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "product": self.product, "step": self.step, "lead": self.lead,
            "land_step": self.land_step, "inv_now": self.inv_now,
            "spot": round(self.spot, 3),
            "own_pipeline": round(self.own_pipeline, 3),
            "opponent_est": round(self.opponent_est, 3),
            "exact_drain": round(self.exact_drain, 3),
            "posterior_drain": round(self.posterior_drain, 3),
            "inv_land": round(self.inv_land, 3),
            "quote_land": round(self.quote_land, 3),
            "provisional": self.provisional,
            "curve": self.curve,
        }


def tile_pipeline(product: str, farm, step: int, lead: int) -> float:
    """Units of `product` that the tiles of `farm` will issue over `(step, step+lead]`,
    plus what is already sitting unharvested on them.

    Deliberately reads only `farm["tiles"]`, because **`obs.farms` is public for
    BOTH seats** (`nav.opponent`: money, every tile, the farmer and hands).  That
    is what lets the opponent term below be a *measurement* of their standing
    acreage rather than an extrapolation of their past selling — and on a
    ten-day-lead crop the difference is the whole projection: a trailing-rate
    estimate of a supply that has not started flowing yet is exactly zero, which
    is the one number that cannot be right about a mirror opponent growing the
    same crop on the same clock.
    """
    land_step = step + lead
    total = 0.0
    for row in (farm or {}).get("tiles", []) or []:
        for t in row:
            if not isinstance(t, dict):
                continue
            if "animal" in t:
                spec = ANIMALS.get(t["animal"])
                if not spec or spec["product"] != product:
                    continue
                total += float(t.get("yield_units", 0) or 0)
                interval = max(1, int(spec.get("interval", 1))) * TURNS_PER_DAY
                # events still to fire inside the window, capped by what the
                # animal can still hold (`max_held`)
                events = max(0, (land_step - step) // interval)
                total += min(float(spec.get("max_held", 0)), float(events))
                continue
            crop = t.get("crop")
            if crop != product:
                continue
            spec = CROPS.get(crop) or {}
            total += float(t.get("yield_units", 0) or 0)
            if spec.get("ongoing"):
                interval = max(1, int(spec.get("interval", 1))) * TURNS_PER_DAY
                events = max(0, (land_step - step) // interval)
                total += float(min(events, int(spec.get("max_yield", 0))))
    return total


def internal_consumption(product: str, farm, step: int, lead: int,
                         feeds_per_day: float = 1.0) -> float:
    """Units of `product` this farm will take OUT of circulation itself over the
    window, before anything reaches the market.

    Exactly one product has structural internal consumption: `FEED` takes one
    **WHEAT** from a unit's inventory (`_apply_unit_action`: `_inv_take(inv,
    "WHEAT", 1)`), and an animal that goes two consecutive days unfed escapes
    (`_daily_refresh_animals`).  So a farm with a herd is a wheat *sink*, and on
    a big herd it is a net BUYER of wheat no matter how much it grows.

    Ignoring this term is not a small error: it inverts the sign of the whole
    wheat projection, because gross production says "pool up, price down" while
    the truth is "pool down, price up".  Measured on 16 seat-seasons before this
    term existed, WHEAT's direction was right on 19% of material moves — reliably
    *worse* than a coin, which is the signature of a missing term rather than a
    noisy one.

    Read off `farm["tiles"]`, which is public for both seats, so the opponent's
    herd is priced by exactly the same call.  `feeds_per_day = 1.0` is the
    keep-producing cadence; the escape constraint alone would only force 0.5, so
    this term is a band, and the caller reports it as such.
    """
    if product != "WHEAT":
        return 0.0
    herd = 0
    for row in (farm or {}).get("tiles", []) or []:
        for t in row:
            if isinstance(t, dict) and "animal" in t:
                herd += 1
    days = lead / TURNS_PER_DAY
    return float(herd) * days * float(feeds_per_day)


def own_pipeline_landing(product: str, farm, private, step: int, lead: int) -> float:
    """Our committed supply over the window: the tile pipeline plus what is
    already in the shed and in our units' hands.

    Deliberately the *committed* pipeline and not a forecast of our own future
    planting — the projection's job is to price the next tile against a world
    that already contains everything we have decided, and folding in a guess
    about decisions not yet taken would make the rule chase its own tail (the
    ration-not-race error expB-220 measured at 21.9%).
    """
    total = tile_pipeline(product, farm, step, lead)
    shed = (private or {}).get("shed", {}) or {}
    total += float(shed.get(product, 0))
    for inv in (private or {}).get("inventories", []) or []:
        total += float((inv or {}).get(product, 0))
    # NET of what we will eat ourselves: a farm with a herd is a wheat sink.
    return total - internal_consumption(product, farm, step, lead)


def opponent_est_landing(opponent_book, product: str, lead: int, step: int = 0,
                         their_farm=None) -> tuple[float, bool]:
    """Their expected contribution to the pool over the window.

    Two estimators, in order of how much they are actually *known*:

    1. **Their standing acreage**, read straight off `obs.farms[them]["tiles"]`,
       which is public and exact.  This is the term that sees a mirror
       opponent's ten-day strawberry crop before a single unit of it has been
       sold.  Preferred whenever the farm is supplied.
    2. **The trailing flow residual** from `nav.OpponentBook`, used when no farm
       is available.  It is backward-looking by construction and therefore blind
       to any supply that has not started flowing.

    Their shed and seeds are hidden either way (`unmarkable_residual`), so the
    reading is always flagged `provisional`: it is a lower bound on their supply,
    never a measurement of it.
    """
    provisional = True
    if their_farm is not None:
        # Same two terms as our own side, minus the shed we cannot see.  Their
        # herd is public, so their wheat consumption is priced exactly.
        return (tile_pipeline(product, their_farm, step, lead)
                - internal_consumption(product, their_farm, step, lead)), provisional
    if opponent_book is None:
        return 0.0, provisional
    seen = max(1, int(getattr(opponent_book, "steps_seen", 0) or 0))
    rate = float(opponent_book.net_flow.get(product, 0.0)) / seen
    return rate * float(lead), provisional


def project(nav, product: str, lead: int | None = None, extra_units: float = 0.0,
            farm=None, private=None, their_farm=None) -> Projection:
    """The §2.1 projection for `product`, `lead` steps out from `nav.step`.

    `extra_units` is the marginal supply whose effect is being priced — b1 passes
    the yield of the tile it is considering; the back-test passes 0, because it
    is asking what the pool did, not what one more tile would have done to it.
    """
    step = int(nav.step)
    lead = int(LEAD_STEPS.get(product, TURNS_PER_DAY)) if lead is None else int(lead)
    land_step = min(step + lead, TERMINAL_INDEX)
    curve = nav.curves[product]
    # the live pool, straight off the observation the caller just fed in
    market = nav.obs["market"] if isinstance(nav.obs, dict) else getattr(nav.obs, "market")
    inv_map = market["inventory"] if isinstance(market, dict) else market.inventory
    inv_now = int(inv_map[product])

    own = own_pipeline_landing(product, farm, private, step, lead) if farm is not None else 0.0
    opp, provisional = opponent_est_landing(nav.opponent, product, lead, step=step,
                                            their_farm=their_farm)
    exact = nav.book.drain_over_steps(product, step + 1, land_step)
    post = expected_extra_unlock_drain(nav.book, product, step, land_step - step)

    inv_land = inv_now + own + opp + float(extra_units) - exact - post
    inv_land = max(0.0, inv_land)
    return Projection(
        product=product, step=step, lead=lead, land_step=land_step, inv_now=inv_now,
        spot=float(curve.price_at(inv_now)), own_pipeline=own, opponent_est=opp,
        exact_drain=exact, posterior_drain=post, inv_land=inv_land,
        quote_land=float(curve.price_at(inv_land)), provisional=provisional,
        curve=curve_state(curve),
    )


def season_revenue(curve, book: DemandBook, product: str, units: float,
                   start_inv: int, first_step: int, last_step: int, chunks: int = 60,
                   opponent_units: float | None = None) -> float:
    """Revenue of selling `units` of `product` spread evenly across
    `[first_step, last_step]`, with the town draining the pool in between.

    This is the quantity ruling-04 §3.1's $47.4 / $147.3 table actually measures:
    a *season's* realised revenue, not a block sale.  The two are very different
    numbers and conflating them is what makes a naive marginal-revenue reading of
    a season volume come out at the price floor — over thirty days the town takes
    a large multiple of the standing pool out of it, so a season that sells 190
    strawberries does NOT walk the curve 190 units down.

    `opponent_units` is the OTHER seat's volume into the same pool.  It moves the
    inventory but earns us nothing, and leaving it out is what makes a mirror
    census irreproducible: the $47.4 / $147.3 points were measured on a board
    where both seats were selling, so the pool absorbed twice the per-seat
    volume.  Defaults to `units` (the mirror), which is the census's own regime.

    Uniform sales against a uniform drain, integrated in `chunks` pieces; the
    pool is floored at zero, and the curve's own $1 floor handles the rest.
    """
    units = float(units)
    if units <= 0:
        return 0.0
    opp = float(units if opponent_units is None else opponent_units)
    drain = book.drain_over_steps(product, first_step, last_step)
    inv = float(start_inv)
    per_chunk_u = units / chunks
    per_chunk_o = opp / chunks
    per_chunk_d = drain / chunks
    revenue = 0.0
    for _ in range(chunks):
        revenue += per_chunk_u * curve.price_at(max(0.0, inv))
        inv = max(0.0, inv + per_chunk_u + per_chunk_o - per_chunk_d)
    return revenue


def marginal_revenue_at_landing(nav, product: str, units: float, lead: int | None = None,
                                farm=None, private=None) -> float:
    """Revenue of selling `units` of `product` into the pool AS IT WILL BE when
    they land — the quantity b1's hurdle test needs, and the one `MR` in
    ruling-04 §3.1's $47 / $147 table means.
    """
    p = project(nav, product, lead=lead, farm=farm, private=private)
    return nav.curves[product].marginal_revenue(int(round(p.inv_land)), int(round(units)))
