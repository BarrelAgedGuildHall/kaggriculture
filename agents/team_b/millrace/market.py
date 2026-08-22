"""Millrace M1 — runtime market identification. NO hard-coded price curve.

Everything here is estimated from `obs.market` (inventory, price) samples the
engine hands us every turn. Structural priors kept (spec-permitted, each
self-checkable at runtime):
  - the town's shop->product recipe table (prior for the drain estimate only;
    the live estimate supersedes it and divergence is tracked)
  - the town consumption cadence (`tick_mod` = 4: shops consume on steps
    divisible by 4). SELF-CHECK: for a product neither player can buy
    (everything except WHEAT/FERTILIZER), an inventory decrement on an
    off-tick turn with no sale of ours is impossible under this assumption;
    if observed, `tick_ok` flips false and the drain estimator falls back to
    the no-trade-turn sampler.

Estimators (v2, after the 2026-08-20 M1 acceptance run exposed two biases in
v1 — window quantisation on low-drain products and opponent-sale masking in
mirrors, est 0.8/day vs truth 7.0 for EGG):
  - **Price table.** `market_price` is a pure function of inventory, so every
    observed (inventory, price) pair is an exact point of the live curve.
    `marginal_revenue` walks the OBSERVED table (exact where visited, linear
    interpolation between neighbours, measured-slope extrapolation outside),
    which captures curvature that a constant-slope walk misses.
  - **Drain by daily balance.** Per completed day:
        drain = our_net + opponent_net_est - deltaI
    with opponent_net measured exactly on off-tick turns (off-tick, the only
    inventory movers are the two players) and scaled by tick/off-tick turn
    counts for the tick turns. Days where the product sat on the $1 floor are
    skipped (floored sales do not add inventory). Estimate = median of the
    last few completed days.

Exposed to the rest of the agent:
    quote(p)                live price
    dPdI(p)                 measured local price slope (<= 0 in normal regimes)
    drain_per_day(p)        estimated town consumption per day
    drain_live(p)           (estimate, n_days) from the balance estimator only
    marginal_revenue(p, n)  proceeds of selling n units NOW along the observed
                            price table (never a shape assumption)
    observed_max_quote(p)   running max quote (for the M7 floor guard)

Engine: kaggle-environments 1.32.7 (pin; the curve itself is never read).
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]

BUYABLE = ("WHEAT", "FERTILIZER")  # the only products either player can buy

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
        self.drain_samples = {p: deque(maxlen=60) for p in PRODUCTS}  # v1 fallback
        self.obs_max_quote = {p: 0 for p in PRODUCTS}
        # observed price curve: inventory -> price (exact points of the live fn)
        self.seen = {p: {} for p in PRODUCTS}
        self._seen_keys = {p: [] for p in PRODUCTS}  # sorted, for interpolation
        # daily-balance drain estimator
        self.drain_days = {p: deque(maxlen=6) for p in PRODUCTS}
        self.tick_mod = 4     # structural prior, self-checked (see observe)
        self.tick_ok = True
        self._day = None
        self._d_dI = {p: 0 for p in PRODUCTS}
        self._d_our = {p: 0 for p in PRODUCTS}
        self._d_opp_off = {p: 0 for p in PRODUCTS}
        self._d_tick = 0
        self._d_off = 0
        self._d_floored = {p: False for p in PRODUCTS}
        self._shops_sig = ()
        self.last_inv = None
        self.last_our_net = {p: 0 for p in PRODUCTS}
        self._prices = {p: 0 for p in PRODUCTS}
        self._steps_seen = 0

    # ---- feeding -----------------------------------------------------------
    def _close_day(self):
        scale = self._d_tick / max(1, self._d_off)
        for p in PRODUCTS:
            if self._d_floored[p]:
                continue  # $1 sales are invisible in inventory: skip the day
            opp = self._d_opp_off[p] * (1.0 + scale)
            drain = self._d_our[p] + opp - self._d_dI[p]
            self.drain_days[p].append(max(0.0, drain))
        for p in PRODUCTS:
            self._d_dI[p] = 0
            self._d_our[p] = 0
            self._d_opp_off[p] = 0
            self._d_floored[p] = False
        self._d_tick = 0
        self._d_off = 0

    def observe(self, obs, our_orders_last_turn=None):
        """Call once per turn BEFORE planning, with the orders we emitted on
        the previous turn (best-effort; fills are not exactly observable)."""
        market = obs["market"]
        day = obs["day"]
        step = day * 24 + obs["hour"]
        inv = {p: market["inventory"].get(p, 0) for p in PRODUCTS}
        prices = {p: market["prices"].get(p, 0) for p in PRODUCTS}
        self._prices = prices
        self._steps_seen += 1

        if self._day is not None and day != self._day:
            self._close_day()
        self._day = day

        # Town drain is piecewise-constant in the (observable) unlocked-shop
        # set: when a shop unlocks, completed-day samples for its products are
        # stale. Reset those windows (recipe prior used only as a reset hint —
        # if it were wrong the sliding window still converges within 6 days).
        shops = obs["town"].get("unlocked_shops", [])
        sig = tuple(shops)
        if sig != self._shops_sig:
            new_shops = list(shops)
            for s in self._shops_sig:
                if s in new_shops:
                    new_shops.remove(s)
            for shop in new_shops:
                for p in SHOPS_PRIOR.get(shop, []):
                    self.drain_days[p].clear()
            self._shops_sig = sig

        our_net = {p: 0 for p in PRODUCTS}
        for o in our_orders_last_turn or []:
            if not (isinstance(o, list) and len(o) >= 3):
                continue
            if o[0] == "SELL" and o[1] in our_net:
                our_net[o[1]] += int(o[2])
            elif o[0] == "BUY_PRODUCT" and o[1] in our_net:
                our_net[o[1]] -= int(o[2])

        # `_town_consume(step=s)` runs inside the engine step whose result is
        # the observation at s+1, so the decrement observed at obs step `step`
        # comes from tick condition (step - 1) % tick_mod == 0.
        is_tick = ((step - 1) % self.tick_mod == 0)
        if self.last_inv is not None:
            if is_tick:
                self._d_tick += 1
            else:
                self._d_off += 1

        for p in PRODUCTS:
            self.obs_max_quote[p] = max(self.obs_max_quote[p], prices[p])
            # exact point of the live price curve
            if inv[p] not in self.seen[p]:
                self.seen[p][inv[p]] = prices[p]
                insort(self._seen_keys[p], inv[p])
            if self.hist[p]:
                pinv, pprice = self.hist[p][-1]
                dI = inv[p] - pinv
                dP = prices[p] - pprice
                if dI != 0:
                    self.slope[p].append(dP / dI)
                # v1 fallback sampler: no-trade decrements
                if our_net[p] == 0 and dI <= 0:
                    self.drain_samples[p].append(-dI)
                elif our_net[p] != 0:
                    residual = -(dI - our_net[p])
                    if residual >= 0:
                        self.drain_samples[p].append(residual)
                # daily balance ledger
                self._d_dI[p] += dI
                self._d_our[p] += our_net[p]
                if not is_tick:
                    self._d_opp_off[p] += dI - our_net[p]
                    # tick-cadence self-check: off-tick, only players move
                    # inventory; nobody can buy a non-BUYABLE product, so a
                    # decrement with no sale of ours breaks the assumption.
                    if p not in BUYABLE and our_net[p] == 0 and dI < 0:
                        self.tick_ok = False
            if prices[p] <= 1:
                self._d_floored[p] = True
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

    def drain_live(self, p):
        """(estimate, n_completed_days) from the daily-balance estimator."""
        d = self.drain_days.get(p)
        if not d:
            return None, 0
        vals = sorted(d)
        return vals[len(vals) // 2], len(d)

    def drain_per_day(self, p, obs=None):
        """Estimated units/day the town consumes. Preference order: daily
        balance (if the tick self-check holds), no-trade sampler, prior."""
        if self.tick_ok:
            est, n = self.drain_live(p)
            if n >= 1:
                return est
        s = self.drain_samples.get(p)
        if s and len(s) >= 24:
            return sum(s) / len(s) * 24.0
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

    def price_at(self, p, inventory):
        """Price at a hypothetical inventory: exact where observed, linear
        interpolation between observed neighbours, measured-slope extrapolation
        outside the observed range. Floored at $1."""
        table = self.seen[p]
        if inventory in table:
            return float(table[inventory])
        keys = self._seen_keys[p]
        if not keys:
            return float(self._prices.get(p, 0))
        i = bisect_left(keys, inventory)
        if 0 < i < len(keys):
            lo, hi = keys[i - 1], keys[i]
            plo, phi = table[lo], table[hi]
            f = (inventory - lo) / (hi - lo)
            return max(1.0, plo + f * (phi - plo))
        # outside the observed range: extrapolate from the nearest edge
        edge = keys[0] if i == 0 else keys[-1]
        step = self.dPdI(p)
        return max(1.0, table[edge] + step * (inventory - edge))

    def marginal_revenue(self, p, n):
        """Proceeds of selling n units now: walk the OBSERVED price table from
        the live inventory (engine walk: a $1 sale does not add inventory).
        No shape function anywhere."""
        inv = (self.last_inv or {}).get(p)
        if inv is None:
            q = float(self.quote(p))
            return sum(max(1.0, q) for _ in range(int(n)))
        total = 0.0
        cur = inv
        for _ in range(int(n)):
            price = self.price_at(p, cur)
            total += max(1.0, price)
            if price > 1:
                cur += 1
        return total

    def marginal_buy_cost(self, p, n, skip=0):
        """Cost of buying `n` units of a BUYABLE product now — the sell-side
        mirror of `marginal_revenue`, and the number a make-or-buy decision
        needs.

        Engine (`_process_market`, 1.32.7): a BUY_PRODUCT unit is quoted at
        `market_price(item, inventory - 1)`, i.e. at the POST-buy inventory,
        and each unit bought decrements market inventory — so the price walks
        UP as you buy, the exact opposite of the sell walk.

        `skip` lets a caller price the *marginal* units of a larger purchase:
        with `skip=k` the walk starts k units in, which is what "the cost of
        the last two units of today's residual feed shortfall" means. There is
        no floor on the way up (the floor only binds at $1 going down).
        """
        inv = (self.last_inv or {}).get(p)
        if inv is None:
            q = max(1.0, float(self.quote(p)))
            return q * int(n)
        total = 0.0
        cur = inv - int(skip)
        for _ in range(int(n)):
            total += max(1.0, self.price_at(p, cur - 1))
            cur -= 1
        return total
