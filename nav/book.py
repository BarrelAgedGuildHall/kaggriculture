"""The demand book: exactly how much of each product the town will take, and what is still
unknown about it (plan §1 row 11).

Everything here is either exact from `obs.town` or an explicit posterior over the engine's
own draw. Nothing is estimated from our own history, because the town's appetite is a
property of the board, not of how we have been playing.

Exact, from the unlock set on the board (engine `_town_consume`):
  * each unlocked shop instance eats 1 of each product it lists, every `townShopSellInterval`
    (4) steps -- **2x for a single-product shop** (YARN_STORE, PET_CAFE);
  * the Town Center eats 1 of every non-FERTILIZER product every `townCenterSellInterval`
    (24) steps;
  * **FERTILIZER has no town drain at all** -- its pool never recovers on the sell side,
    which is why a floored fertilizer pool stays floored.

Posterior, over what has not unlocked yet (engine `_advance_day`): every 3 days one shop is
drawn **uniformly with replacement** from the eight, up to 8 instances. So the chance a given
shop is present at least once in the remaining draws is `1 - (1 - 1/8)^R`. Variety is not
guaranteed and the same shop can unlock repeatedly.
"""
from __future__ import annotations

from .rules import (CENTER_SELL_INTERVAL, MAX_SHOP_INSTANCES, SHOP_SELL_INTERVAL, SHOPS,
                    TERMINAL_INDEX, TURNS_PER_DAY, LAST_DAY, TOWN_CENTER_PRODUCTS)

SHOP_UNLOCK_INTERVAL_DAYS = 3      # configuration.townShopUnlockInterval


def _ticks(interval: int, from_step: int, to_step: int) -> int:
    """How many steps in (from_step, to_step] are divisible by `interval`."""
    if to_step <= from_step:
        return 0
    return to_step // interval - from_step // interval


class DemandBook:
    """Exact remaining town demand plus the posterior over future unlocks."""

    def __init__(self, shop_interval: int = SHOP_SELL_INTERVAL,
                 center_interval: int = CENTER_SELL_INTERVAL) -> None:
        self.shop_interval = max(1, int(shop_interval))
        self.center_interval = max(1, int(center_interval))
        self.unlocked: tuple[str, ...] = ()
        self.step = 0
        self.day = 0
        self._per_shop_cache: dict[str, float] | None = None

    # -- evidence ---------------------------------------------------------------------------
    def observe(self, town, step: int) -> None:
        shops = tuple(town.get("unlocked_shops", []) if isinstance(town, dict)
                      else getattr(town, "unlocked_shops", []) or [])
        if shops != self.unlocked:
            self._per_shop_cache = None       # only a new unlock changes the drain
        self.unlocked = shops
        self.step = int(step)
        self.day = int(step) // TURNS_PER_DAY

    # -- exact --------------------------------------------------------------------------------
    def _per_interval(self) -> dict[str, float]:
        """Units removed per shop-interval tick, per product, at the current unlock set."""
        if self._per_shop_cache is None:
            d = {}
            for shop in self.unlocked:
                products = SHOPS.get(shop, ())
                mult = 2 if len(products) == 1 else 1
                for item in products:
                    d[item] = d.get(item, 0.0) + mult
            self._per_shop_cache = d
        return self._per_shop_cache

    def drain_per_day(self, product: str) -> float:
        """Exact units per day at the current unlock set (probe_core `_drain_per_day`, which
        is already validated -- not re-derived)."""
        if product not in TOWN_CENTER_PRODUCTS:
            return 0.0                                   # FERTILIZER
        per_day = TURNS_PER_DAY / self.center_interval   # town centre: 1/day
        per_day += self._per_interval().get(product, 0.0) * (TURNS_PER_DAY / self.shop_interval)
        return per_day

    def drain_over_steps(self, product: str, first_step: int, last_step: int) -> float:
        """Units removed by the ticks that FIRE on steps [first_step, last_step] inclusive.

        A snapshot at index `i` shows the board before step `i` is processed, so the tick at
        step `i` has not fired yet -- which is why the two callers below differ by one.
        """
        if product not in TOWN_CENTER_PRODUCTS or last_step < first_step:
            return 0.0
        n_shop = _ticks(self.shop_interval, first_step - 1, last_step)
        n_ctr = _ticks(self.center_interval, first_step - 1, last_step)
        return self._per_interval().get(product, 0.0) * n_shop + n_ctr

    def remaining_drain(self, product: str, from_step: int | None = None,
                        to_step: int = TERMINAL_INDEX) -> float:
        """Units the town removes from `product`'s pool between `from_step` and the bell.
        Exact at the current unlock set; future unlocks only add to it, so it is a lower
        bound (`expected_extra_drain_per_day` is the posterior on top)."""
        a = self.step if from_step is None else int(from_step)
        return self.drain_over_steps(product, a + 1, int(to_step) - 1)

    # -- posterior ----------------------------------------------------------------------------
    def remaining_draws(self, day: int | None = None) -> int:
        """Unlock draws left: one every 3 days, capped at 8 instances in total."""
        d = self.day if day is None else int(day)
        draws = sum(1 for nd in range(d + 1, LAST_DAY + 1) if nd > 0 and nd % SHOP_UNLOCK_INTERVAL_DAYS == 0)
        return max(0, min(draws, MAX_SHOP_INSTANCES - len(self.unlocked)))

    def unlock_posterior(self, day: int | None = None) -> dict[str, float]:
        """P(shop appears at least once among the remaining draws), uniform with replacement."""
        r = self.remaining_draws(day)
        p_miss = 1.0 - 1.0 / len(SHOPS)
        p = 1.0 - p_miss ** r if r else 0.0
        return {shop: p for shop in SHOPS}

    def expected_extra_drain_per_day(self, product: str, day: int | None = None) -> float:
        """Expected units/day the town will *additionally* take once the remaining draws land."""
        if product not in TOWN_CENTER_PRODUCTS:
            return 0.0
        r = self.remaining_draws(day)
        if not r:
            return 0.0
        per_draw = 0.0
        for shop, products in SHOPS.items():
            if product in products:
                per_draw += (2 if len(products) == 1 else 1) / len(SHOPS)
        return per_draw * r * (TURNS_PER_DAY / self.shop_interval)

    def hinge_value(self, product: str, curve=None, inventory: int | None = None,
                    day: int | None = None) -> float:
        """Option value of a hinge product: the extra units the town is *expected* to take
        over the rest of the season if the draws fall its way, marked through the curve.

        Reported as an option value and nothing more. Whether that justifies a CapEx (the
        CARROT/PET_CAFE, WOOL/YARN_STORE, EGG, TOMATO trades) is the team's call, not nav's.
        """
        d = self.day if day is None else int(day)
        extra_per_day = self.expected_extra_drain_per_day(product, d)
        if extra_per_day <= 0:
            return 0.0
        days_left = max(0, LAST_DAY - d)
        units = extra_per_day * days_left
        if curve is None:
            return units
        inv = curve.i0 if inventory is None else int(inventory)
        return curve.marginal_revenue(inv, int(round(units)))

    def as_dict(self) -> dict:
        return {
            "unlocked_shops": list(self.unlocked),
            "drain_per_day": {p: self.drain_per_day(p) for p in TOWN_CENTER_PRODUCTS},
            "remaining_draws": self.remaining_draws(),
            "p_shop_unlocks": round(1.0 - (1.0 - 1.0 / len(SHOPS)) ** self.remaining_draws(), 4),
        }
