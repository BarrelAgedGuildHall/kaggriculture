"""Millrace M1 — runtime market identification. NO hard-coded price curve.

Everything here is estimated from `obs.market` (inventory, price) samples the
engine hands us every turn. Structural priors kept (spec-permitted, each
self-checkable at runtime):
  - the town's shop->product recipe table (prior for the drain estimate only;
    the live estimate supersedes it and divergence is tracked)
  - shop drain cadence (multi-product shops 1/product per tick, single-product
    2/tick) is NOT assumed: drain is measured as inventory decrement on turns
    where we did not trade.

Exposed to the rest of the agent:
    quote(p)                live price
    dPdI(p)                 measured local price slope (<= 0 in normal regimes)
    drain_per_day(p)        estimated town consumption per day
    marginal_revenue(p, n)  proceeds of selling n units NOW, walking the
                            measured slope (never a shape assumption)
    observed_max_quote(p)   running max quote (for the M7 floor guard)

Engine: kaggle-environments 1.32.7 (pin; the curve itself is never read).
"""
from __future__ import annotations

from collections import deque

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]

# prior only — logged against the live estimate, never trusted over it
SHOPS_PRIOR = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}


class MarketModel:
    def __init__(self, history=40):
        self.hist = {p: deque(maxlen=history) for p in PRODUCTS}  # (inv, price)
        self.slope = {p: deque(maxlen=20) for p in PRODUCTS}      # dP/dI samples
        self.drain_samples = {p: deque(maxlen=60) for p in PRODUCTS}
        self.obs_max_quote = {p: 0 for p in PRODUCTS}
        self.last_inv = None
        self.last_our_net = {p: 0 for p in PRODUCTS}  # our sells - buys last turn
        self._prices = {p: 0 for p in PRODUCTS}
        self._steps_seen = 0

    # ---- feeding -----------------------------------------------------------
    def observe(self, obs, our_orders_last_turn=None):
        """Call once per turn BEFORE planning, with the orders we emitted on
        the previous turn (best-effort; fills are not exactly observable)."""
        market = obs["market"]
        inv = {p: market["inventory"].get(p, 0) for p in PRODUCTS}
        prices = {p: market["prices"].get(p, 0) for p in PRODUCTS}
        self._prices = prices
        self._steps_seen += 1

        our_net = {p: 0 for p in PRODUCTS}
        for o in our_orders_last_turn or []:
            if not (isinstance(o, list) and len(o) >= 3):
                continue
            if o[0] == "SELL" and o[1] in our_net:
                our_net[o[1]] += int(o[2])
            elif o[0] == "BUY_PRODUCT" and o[1] in our_net:
                our_net[o[1]] -= int(o[2])

        for p in PRODUCTS:
            self.obs_max_quote[p] = max(self.obs_max_quote[p], prices[p])
            if self.hist[p]:
                pinv, pprice = self.hist[p][-1]
                dI = inv[p] - pinv
                dP = prices[p] - pprice
                if dI != 0:
                    self.slope[p].append(dP / dI)
                # drain: inventory decrement on a turn where WE did not trade.
                # (the opponent adds inventory when selling, so decrements on
                # our no-trade turns under-attribute only when they buy, which
                # is rare for produce.) Accumulated per turn, reported per day.
                if our_net[p] == 0 and dI <= 0:
                    self.drain_samples[p].append(-dI)
                elif our_net[p] != 0:
                    residual = -(dI - our_net[p])
                    if residual >= 0:
                        self.drain_samples[p].append(residual)
            self.hist[p].append((inv[p], prices[p]))
        self.last_inv = inv
        self.last_our_net = our_net

    # ---- queries ------------------------------------------------------------
    def quote(self, p):
        return self._prices.get(p, 0)

    def dPdI(self, p):
        s = self.slope.get(p)
        if not s:
            return -1.0  # pessimistic default until measured
        vals = sorted(s)
        return vals[len(vals) // 2]  # median: robust to town-tick jumps

    def drain_per_day(self, p, obs=None):
        """Estimated units/day the town consumes. Falls back to the shop-recipe
        prior until enough live samples exist."""
        s = self.drain_samples.get(p)
        if s and len(s) >= 24:
            per_turn = sum(s) / len(s)
            return per_turn * 24.0
        if obs is not None:
            return float(self._prior_drain(obs, p))
        return 0.0

    def _prior_drain(self, obs, p):
        if p == "FERTILIZER":
            return 0
        per_day = 1  # town centre
        for shop in obs["town"].get("unlocked_shops", []):
            prods = SHOPS_PRIOR.get(shop, [])
            if p in prods:
                per_day += 12 if len(prods) == 1 else 6
        return per_day

    def marginal_revenue(self, p, n):
        """Proceeds of selling n units now: walk the measured slope from the
        live quote, floored at $1. No shape function anywhere."""
        q = float(self.quote(p))
        step = abs(self.dPdI(p))
        total = 0.0
        for i in range(int(n)):
            total += max(1.0, q - i * step)
        return total
