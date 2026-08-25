"""The runtime-measured price curve, and the closed-form integral of it.

**The agent cannot read the price curve.** `obs.market` carries only the current inventory
and the current price per product; `market["params"]` exists only in an episode that
overrides `configuration.marketParams`. So marking a future or a metered sale means
reconstructing the curve from what has been observed (spec §3).

What makes that tractable, all verified in the engine:

1. At step 0 every product sits at `inventory = I0` with `price = base`, so **base and I0 are
   observed exactly** on the first turn of every episode.
2. `price(inv) = base +- amp * f(|inv - I0|, T)`, one shape family per side,
   `f in {linear, sq, sqrt, log, hinge}`, floored at 1 and rounded to an integer.
3. Every later turn yields one more exact point per product, and our own sales walk the
   curve deliberately.

`PriceCurve` is therefore a per-product, per-side **online least-squares fit** with `base`
and `I0` pinned. Writing `price(inv) = base -+ c * g(d)` (d = |inv - I0|) makes every shape
except `hinge` a one-parameter model whose optimum is closed-form (`c = Sgy / Sgg`), so a new
point costs O(number of candidate models) and the refit is O(1) in the number of points.
`hinge` carries a second parameter (the knee `T`) and is searched over a fixed grid.

Closed form, not a walk (exp-N1 §7.4)
-------------------------------------
The probe priced `n` units with a per-unit Python loop. Here the same quantity is the
**midpoint integral** of the fitted curve,

    sum_{k=0..n-1} p(inv+k)  ~=  integral from inv-1/2 to inv+n-1/2 of p(x) dx,

whose error is O(p''/24) per unit -- far below the +-0.5 the engine's integer rounding
already imposes -- and which costs the same for n = 1 and n = 900. The `$1` floor is not
smooth, so it is handled exactly rather than integrated: a sale at `$1` does **not** move the
pool (`_commit_unit`: "Sales at $1 do not increase market supply"), so once the quote hits the
floor every further unit fetches exactly `$1`. The crossing inventory is found by bisection,
O(log n).

The buy side follows the engine exactly: `BUY_PRODUCT` quotes at `inventory - 1` and then
decrements, so `marginal_cost(inv, n) = sum_{k=1..n} p(inv - k)`. (The probe walked
`p(inv), p(inv-1), ...`, one unit short of the engine; recorded here, engine wins.)

Provenance: a read says what backed it (R-52)
---------------------------------------------
A curve is fitted per *side*, and the two sides are observed independently: a product we only
ever sell walks its pool up, so the glut side fills with points while the scarcity side stays
empty. The old fallback returned `base` on a side with no fit -- a flat curve, i.e. "sell as
much as you like, no price impact" -- and nothing in the return value distinguished that from
a well-fitted read. It is not a symmetric loss of accuracy; it biases every acreage and
capital decision toward over-production.

So every read now carries its **provenance**, worst-of over whatever it touched:

    observed        an exact recorded (inventory, price) point
    fitted          the side has a fit and the query is inside (or just past) its reach
    extrapolated    the query is beyond `EXTRAPOLATION_FACTOR x side_reach`
    base_fallback   the side has NO fit -- the read is `base`, and is not evidence
    floor           the curve has no `base` yet -- the read is PRICE_FLOOR

`price_at`/`quote` keep their signatures and their numbers exactly; `price_at_marked` is the
marked read, and every aggregate read (`marginal_revenue`, `marginal_cost`, `priced_units`,
`units_above`, `_integral`) leaves `curve.last_read` behind for the caller that cares. The
bookkeeping is O(1) in the number of units: the inventory interval is split analytically by
side and by reach, never walked.

**The gate clause that goes with it (ruling-05 R-52 item 5).** Any offline projection,
back-test or calibration pair whose read touched `base_fallback` (or `floor`) is **VOID** --
excluded from the denominator, reported as a separate count, never scored as a miss. A model
cannot be refuted on data where it was never consulted. And no irreversible action (a
purchase, a plant, a hire) may be taken on such a read. `nav` states the classes and reports
them; enforcing either clause is the caller's job, because `nav` never decides anything.

Nothing in this module decides anything: it answers "what is this worth" and "how much
evidence backs that answer" (`confidence`, `side_confidence`, `diagnostics`, `tripwire`).
"""
from __future__ import annotations

import math

from .rules import (HINGE_GAIN, MARKET_I0_FALLBACK, PRICE_FLOOR, PRODUCTS, SHAPES)

# Knee grid for the two-parameter `hinge` shape: geometric, ratio 1.15 from 40 to ~2200.
# Measured (starter mirror, seed 7, the steepest case in a normal episode -- TOMATO's below
# side runs past its knee): ratio 1.4 leaves p95 |quote - true| = 3 units over the observed
# range, ratio 1.15 leaves 2. The extra models cost ~30 us/turn, ~0.6% of the 5 ms budget.
HINGE_T_GRID = (40, 46, 53, 61, 70, 81, 93, 107, 123, 141, 163, 187, 215, 247, 284, 327,
                376, 432, 497, 572, 658, 756, 870, 1000, 1150, 1323, 1521, 1750, 2011)

# Tripwire thresholds. Both must be exceeded, because either alone cries wolf: a relative
# threshold alone fires on a $2 product, an absolute one alone fires on any steep curve the
# fit is merely extrapolating along (TOMATO's hinge reaches |err| = 3 in ordinary play, which
# is 1% of its price there). Measured out-of-sample relative error in normal episodes is
# ~0-1%; an unrepresentable curve is tens of percent.
# Above this many distinct integer price levels in one summation range, the midpoint integral
# is used; at or below it the quotes are summed exactly (see `_sum_quotes`).
MAX_EXACT_LEVELS = 12

RESIDUAL_TOLERANCE = 2.0      # price units
RELATIVE_TOLERANCE = 0.05     # fraction of the observed price
TRIPWIRE_WINDOW = 8           # out-of-sample residuals kept per side
TRIPWIRE_MIN_POINTS = 4       # never fire on a curve that has barely been observed

# Exponential forgetting on the sufficient statistics: each new point discounts the ones
# before it. 1.0 = plain least squares over the whole episode. Below 1 the fit becomes local
# -- which is what the marks want, because the pool is walked from where it is now, and a
# one-step-ahead extrapolation is what the tripwire measures.
#
# Measured over 12 episodes (starter mirror + Team B routed, seeds 3/7/11, default AND
# g4-perturbed params), counting how often the tripwire condition held with a full window:
#   gamma  1.00 -> 187 firings (all false: TOMATO's below-side hinge past its knee)
#   gamma  0.99 ->  52 firings (all false)
#   gamma  0.97 ->   0 firings, |quote - true| over the unobserved range p95 = 2, max = 5
#   gamma  0.90 ->   0 firings, but max error 7 -- forgetting too fast costs accuracy
# 0.97 is the point where the false-positive rate reaches zero without paying for it.
FORGET = 0.97

# How far past the observed reach of a side a fit is still called `fitted`. A parameter, not a
# constant: the shapes are monotone and one-parameter, so a modest extension of the observed
# range is the fit doing its job, while a query many times past the furthest point ever seen
# is the model speaking about a market it has never been shown. Start at 1.5 (R-52).
EXTRAPOLATION_FACTOR = 1.5

# --------------------------------------------------------------------------- read provenance
# Ordered from "nothing was read" to "worst", so `worst_provenance` is a max over the index.
PROVENANCE_ORDER = ("none", "observed", "fitted", "extrapolated", "base_fallback", "floor")
_PROVENANCE_RANK = {p: i for i, p in enumerate(PROVENANCE_ORDER)}

# A read in this class is not evidence: the curve was never consulted, it answered with a
# constant. Offline pairs that touched one are VOID; no irreversible action may rest on one.
VOID_PROVENANCE = frozenset({"base_fallback", "floor"})

# `provisional` on a mark: the read is either not evidence at all, or is the fit speaking
# well outside the range it was fitted on.
PROVISIONAL_PROVENANCE = frozenset({"extrapolated", "base_fallback", "floor"})


def worst_provenance(*provenances) -> str:
    """The worst class among the arguments (each a class name or an iterable of them)."""
    worst = "none"
    for p in provenances:
        if p is None:
            continue
        if not isinstance(p, str):
            p = worst_provenance(*p)
        if _PROVENANCE_RANK.get(p, 0) > _PROVENANCE_RANK[worst]:
            worst = p
    return worst


def read_is_void(provenance) -> bool:
    """True when this read leaned on a constant instead of on evidence (R-52 item 5).

    Accepts a class name, a `last_read`/`Line`-style mapping carrying `"provenance"`, or a
    `(value, provenance)` pair -- so an analysis script can pass whatever it happens to hold.
    """
    if provenance is None:
        return False
    if isinstance(provenance, str):
        return provenance in VOID_PROVENANCE
    if isinstance(provenance, dict):
        return read_is_void(provenance.get("provenance"))
    if isinstance(provenance, (tuple, list)) and len(provenance) == 2:
        return read_is_void(provenance[1])
    return read_is_void(getattr(provenance, "provenance", None))


def read_is_provisional(provenance) -> bool:
    """True when a mark resting on this read must be flagged `provisional`."""
    if isinstance(provenance, str):
        return provenance in PROVISIONAL_PROVENANCE
    if provenance is None:
        return False
    if isinstance(provenance, dict):
        return read_is_provisional(provenance.get("provenance"))
    if isinstance(provenance, (tuple, list)) and len(provenance) == 2:
        return read_is_provisional(provenance[1])
    return read_is_provisional(getattr(provenance, "provenance", None))


# ----------------------------------------------------------------------------- shape algebra
def shape(func: str, x: float, T: float | None = None) -> float:
    """The engine's `_shape`, transcribed. `f(0) = 0` for every family."""
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def shape_integral(func: str, x: float, T: float | None = None) -> float:
    """`integral from 0 to x of shape(func, t) dt`, x >= 0. The closed form of §7.4."""
    x = max(0.0, x)
    if func == "linear":
        return 0.5 * x * x
    if func == "sq":
        return x ** 3 / 3.0
    if func == "sqrt":
        return (2.0 / 3.0) * x ** 1.5
    if func == "log":
        return (1.0 + x) * math.log(1.0 + x) - x
    if func == "log10":
        return ((1.0 + x) * math.log(1.0 + x) - x) / math.log(10.0)
    if func == "hinge":
        if not T or T <= 0:
            return 0.5 * x * x
        u = x / T
        return T * (0.5 * u * u + HINGE_GAIN * max(0.0, u - 1.0) ** 3 / 3.0)
    return 0.5 * x * x


def amp_from_target(func: str, target: float, base: float, T: float) -> float:
    """The engine's `amp = target * base / f(T, T)` -- used by `from_params` and by tests."""
    return target * base / shape(func, T, T)


class _SideFit:
    """Sufficient statistics for one side of one product's curve.

    Model: `y = c * g_m(d)` for each candidate model m, where `d = |inv - I0|` and
    `y = |price - base|`. For every m the least-squares optimum is `c = Sgy/Sgg` and the
    residual sum is `Syy - Sgy^2/Sgg`, so a new point is O(#models) and a refit is O(#models)
    -- no history is kept and nothing is refitted from scratch.
    """

    __slots__ = ("models", "sgg", "sgy", "syy", "n", "w", "seen", "dirty",
                 "shape", "T", "c", "sse", "rmse", "residuals", "reach")

    def __init__(self) -> None:
        self.models: list[tuple[str, float | None]] = [(f, None) for f in SHAPES if f != "hinge"]
        self.models += [("hinge", float(t)) for t in HINGE_T_GRID]
        self.sgg = [0.0] * len(self.models)
        self.sgy = [0.0] * len(self.models)
        self.syy = 0.0
        self.n = 0
        self.w = 0.0
        self.seen: set[int] = set()
        self.dirty = True
        self.shape: str | None = None
        self.T: float | None = None
        self.c = 0.0
        self.sse = 0.0
        self.rmse = 0.0
        self.residuals: list[tuple[float, float]] = []
        self.reach: float | None = None      # largest |inv - I0| ever observed on this side

    def add(self, d: float, y: float, key: int) -> None:
        if key in self.seen:
            return                      # the same inventory carries no new information
        self.seen.add(key)
        self.n += 1
        self.reach = d if self.reach is None else max(self.reach, d)
        self.w = self.w * FORGET + 1.0
        self.syy = self.syy * FORGET + y * y
        for i, (func, T) in enumerate(self.models):
            g = shape(func, d, T)
            self.sgg[i] = self.sgg[i] * FORGET + g * g
            self.sgy[i] = self.sgy[i] * FORGET + g * y
        self.dirty = True

    def fit(self) -> None:
        if not self.dirty:
            return
        self.dirty = False
        if self.n == 0:
            self.shape, self.c, self.T, self.sse, self.rmse = None, 0.0, None, 0.0, 0.0
            return
        best = None
        for i, (func, T) in enumerate(self.models):
            if self.sgg[i] <= 0.0:
                continue
            c = self.sgy[i] / self.sgg[i]
            sse = max(0.0, self.syy - self.sgy[i] * self.sgy[i] / self.sgg[i])
            # `hinge` buys its fit with a second parameter; charge it one, so it only wins
            # when it explains genuinely more. (Plain SSE would pick it almost always.)
            k = 2 if func == "hinge" else 1
            penalised = sse * (1.0 + k / max(1.0, self.w))
            if best is None or penalised < best[0]:
                best = (penalised, func, T, c, sse)
        _, self.shape, self.T, self.c, self.sse = best
        self.rmse = math.sqrt(self.sse / max(1e-9, self.w))

    def predict(self, d: float) -> float | None:
        """Signed magnitude `c * g(d)`, or None while the side has no fit."""
        self.fit()
        if self.shape is None:
            return None
        return self.c * shape(self.shape, d, self.T)

    def integral(self, d0: float, d1: float) -> float:
        """`integral of c*g(d) dd` from d0 to d1 (both >= 0)."""
        self.fit()
        if self.shape is None:
            return 0.0
        return self.c * (shape_integral(self.shape, d1, self.T) - shape_integral(self.shape, d0, self.T))

    def note_residual(self, err: float, price: float) -> None:
        """One out-of-sample prediction error, absolute and relative to the observed price."""
        self.residuals.append((err, err / max(1.0, abs(price))))
        if len(self.residuals) > TRIPWIRE_WINDOW:
            self.residuals.pop(0)


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


# ----------------------------------------------------------------------------- the curve
class PriceCurve:
    """One product's price curve, fitted from observed `(inventory, price)` pairs."""

    def __init__(self, product: str, base: float | None = None, i0: int | None = None) -> None:
        self.product = product
        self.base: float | None = float(base) if base is not None else None
        self.i0: int = int(i0) if i0 is not None else MARKET_I0_FALLBACK
        self.i0_observed = i0 is not None
        self.above = _SideFit()          # glut side: inventory > I0, price below base
        self.below = _SideFit()          # scarcity side: inventory < I0, price above base
        self.exact: dict | None = None   # set only by `from_params` (offline/tests)
        self.n_points = 0
        self._observed: dict[int, int] = {}   # inventory -> exact observed integer price
        self.extrapolation_factor = float(EXTRAPOLATION_FACTOR)
        # What backed the most recent aggregate read: {"provenance": <worst class touched>,
        # "units_by_class": {class: units}}. `None` until something has been read.
        self.last_read: dict | None = None

    # -- construction -----------------------------------------------------------------------
    @classmethod
    def from_params(cls, product: str, params: dict) -> "PriceCurve":
        """Exact curve from a MARKET_PARAMS-shaped dict. OFFLINE/test use only: giving this to
        the online path would make the curve-agnostic test vacuous."""
        c = cls(product, base=params["base"], i0=params.get("I0", MARKET_I0_FALLBACK))
        c.i0_observed = True
        c.exact = dict(params)
        return c

    # -- evidence ---------------------------------------------------------------------------
    def observe(self, inventory: int, price: int, step: int | None = None) -> float | None:
        """Record one exact point. Returns the OUT-OF-SAMPLE prediction error for this point
        (None when the point was already known or the curve had no fit yet) -- that residual,
        measured before the point is folded in, is what the engine-bump tripwire watches."""
        inventory = int(inventory)
        price = int(price)
        if step == 0 or self.base is None:
            # Step 0 pins the two parameters that are never fitted. `base` is also re-pinned
            # if this is the first observation we ever see (an offline splice mid-episode):
            # then it is a guess and `i0_observed` stays False.
            if step == 0:
                self.base = float(price)
                self.i0 = inventory
                self.i0_observed = True
            elif self.base is None:
                self.base = float(price)
        known = inventory in self._observed
        err = None
        if not known and self.confidence > 0.0:
            err = abs(self.quote(inventory) - price)
            side = self.above if inventory > self.i0 else self.below
            side.note_residual(err, price)
        self._observed[inventory] = price
        d = abs(inventory - self.i0)
        if d > 0:
            y = abs(price - self.base)
            side = self.above if inventory > self.i0 else self.below
            side.add(d, y, inventory)
        if not known:
            self.n_points += 1
        return err

    @property
    def confidence(self) -> float:
        """0..1 diagnostic (never a decision): how much evidence backs the fit, discounted by
        how well it fits. 0 while only the base point is known -- marks that used the curve at
        that point are tagged `provisional`."""
        n = self.above.n + self.below.n
        if self.exact is not None:
            return 1.0
        if n < 2:
            return 0.0
        self.above.fit()
        self.below.fit()
        rmse = max(self.above.rmse, self.below.rmse)
        return min(1.0, n / 8.0) / (1.0 + rmse)

    def _side(self, side: str) -> _SideFit:
        if side in ("above", "glut"):
            return self.above
        if side in ("below", "scarcity"):
            return self.below
        raise ValueError(f"side must be 'above'/'glut' or 'below'/'scarcity', got {side!r}")

    def side_of(self, inventory: float) -> str:
        """`'above'` (glut, price below base) or `'below'` (scarcity, price above base).

        `inventory == I0` is reported as `'below'`; nothing is read there anyway, because
        `d == 0` is the exactly-observed base point.
        """
        return "above" if inventory > self.i0 else "below"

    def side_confidence(self, side: str) -> float:
        """The aggregate `confidence`, restricted to the evidence on ONE side.

        This is the statistic the aggregate one cannot be: a curve with forty points on the
        glut side and none on the scarcity side reports high `confidence` while every
        scarcity-side read is pure `base` (R-52 N-2). The invariant worth relying on is

            `side_confidence(side) == 0.0`  <=>  a read on that side is `base_fallback`

        so a caller can gate on the number without knowing about provenance classes.
        """
        fit = self._side(side)
        if self.exact is not None:
            return 1.0
        if fit.n <= 0:
            return 0.0
        fit.fit()
        return min(1.0, fit.n / 8.0) / (1.0 + fit.rmse)

    def side_reach(self, side: str) -> float | None:
        """The largest `|inventory - I0|` ever observed on that side, or None if never."""
        if self.exact is not None:
            return None
        return self._side(side).reach

    # -- provenance --------------------------------------------------------------------------
    def price_at_marked(self, inventory: float) -> tuple[float, str]:
        """`price_at(inventory)` and what backed it -- the marked read (R-52 item 2).

        The price is bit-for-bit what `price_at` returns; only the second element is new.
        """
        price = self.price_at(inventory)
        return price, self._record_point_read(inventory)

    def _provenance_at(self, inventory: float) -> str:
        """Which class a single-point read at `inventory` falls in."""
        if self.base is None:
            return "floor"
        if self.exact is not None:
            return "observed"
        inv_i = int(inventory)
        if inv_i == inventory and inv_i in self._observed:
            return "observed"
        d = abs(inventory - self.i0)
        if d == 0:
            return "observed" if self.i0_observed else "fitted"
        fit = self.above if inventory > self.i0 else self.below
        fit.fit()
        if fit.shape is None:
            return "base_fallback"
        reach = fit.reach or 0.0
        return "fitted" if d <= self.extrapolation_factor * reach else "extrapolated"

    def _classify_units(self, lo: float, hi: float) -> dict[str, float]:
        """Split the inventory interval [lo, hi] into provenance classes, by units.

        Analytic, O(1): each side contributes at most two pieces (inside its reach, past the
        extrapolation factor), so pricing one unit and pricing nine hundred cost the same.
        Exactly-observed points are NOT looked up here -- classifying an observed unit as
        `fitted` errs on the pessimistic side, which is the direction that is safe.

        `lo`/`hi` are the FIRST and LAST inventory a read prices, not the midpoint interval
        the integral uses: half a unit of overhang across `I0` is not evidence that the read
        leaned on the other side, and it must not decide the worst class.
        """
        if hi < lo:
            return {}
        span = hi - lo
        if self.base is None:
            return {"floor": span}
        if self.exact is not None:
            return {"observed": span}
        out: dict[str, float] = {}

        def add(cls: str, units: float) -> None:
            if units > 0:
                out[cls] = out.get(cls, 0.0) + units

        i0 = float(self.i0)
        for name, fit, a, b in (("below", self.below, lo, min(hi, i0)),
                                ("above", self.above, max(lo, i0), hi)):
            if b <= a:
                continue
            fit.fit()
            if fit.shape is None:
                add("base_fallback", b - a)
                continue
            # d runs from |a - i0| to |b - i0|; the fitted band is d <= factor * reach.
            d_lo, d_hi = (i0 - b, i0 - a) if name == "below" else (a - i0, b - i0)
            edge = self.extrapolation_factor * (fit.reach or 0.0)
            add("fitted", max(0.0, min(d_hi, edge) - d_lo))
            add("extrapolated", max(0.0, d_hi - max(d_lo, edge)))
        return out

    def _record_read(self, lo: float, hi: float, units: float | None = None) -> str:
        """Set `last_read` from the interval an aggregate read touched. Returns the class.

        `units`, when given, is what the class weights are normalised to -- `n` units priced at
        `n` consecutive inventories span `n - 1`, and the caller cares about units.
        """
        if hi <= lo:
            return self._record_point_read(lo)
        classes = self._classify_units(lo, hi)
        if units is not None and classes:
            scale = float(units) / sum(classes.values())
            classes = {k: v * scale for k, v in classes.items()}
        provenance = worst_provenance(classes.keys())
        self.last_read = {"provenance": provenance, "units_by_class": classes}
        return provenance

    def _record_point_read(self, inventory: float) -> str:
        """Set `last_read` from a single-point read (one unit in one class)."""
        cls = self._provenance_at(inventory)
        self.last_read = {"provenance": cls, "units_by_class": {cls: 1.0}}
        return cls

    @property
    def fitted_shape(self) -> tuple[str | None, str | None]:
        self.below.fit()
        self.above.fit()
        return (self.below.shape, self.above.shape)

    # -- the curve --------------------------------------------------------------------------
    def price_at(self, inventory: float) -> float:
        """Continuous fitted price at `inventory`, floored at PRICE_FLOOR (not rounded)."""
        if self.base is None:
            return float(PRICE_FLOOR)
        if self.exact is not None:
            return max(float(PRICE_FLOOR), self._exact_price(inventory))
        inv_i = int(inventory)
        if inv_i == inventory and inv_i in self._observed:
            return float(self._observed[inv_i])      # an exact point beats any fit
        d = abs(inventory - self.i0)
        if d == 0:
            return float(self.base)
        side = self.above if inventory > self.i0 else self.below
        delta = side.predict(d)
        if delta is None:
            return float(self.base)                  # no evidence on this side yet
        p = self.base - delta if inventory > self.i0 else self.base + delta
        return max(float(PRICE_FLOOR), p)

    def quote(self, inventory: int) -> int:
        """What the engine would print: `max(PRICE_FLOOR, int(round(price)))`."""
        return max(PRICE_FLOOR, int(round(self.price_at(inventory))))

    def _exact_price(self, inventory: float) -> float:
        p = self.exact
        base, i0, T = float(p["base"]), p.get("I0", self.i0), p["T"]
        if inventory < i0:
            f, target = p["below_func"], p["below_target"]
            return base + amp_from_target(f, target, base, T) * shape(f, i0 - inventory, T)
        f, target = p["above_func"], p["above_target"]
        return base - amp_from_target(f, target, base, T) * shape(f, inventory - i0, T)

    # -- the closed-form integral -----------------------------------------------------------
    def _integral(self, a: float, b: float) -> float:
        """`integral of price(x) dx` from a to b (a <= b), split at I0. Exact for the fitted
        model; for `exact` curves it integrates the true curve."""
        if b <= a:
            return 0.0
        self._record_read(a, b)
        if self.base is None:
            return float(PRICE_FLOOR) * (b - a)
        total = 0.0
        i0 = float(self.i0)
        lo, hi = a, min(b, i0)
        if hi > lo:                                   # scarcity side (price above base)
            if self.exact is not None:
                total += self._exact_integral(lo, hi, below=True)
            else:
                total += self.base * (hi - lo) + self.below.integral(i0 - hi, i0 - lo)
        lo, hi = max(a, i0), b
        if hi > lo:                                   # glut side (price below base)
            if self.exact is not None:
                total += self._exact_integral(lo, hi, below=False)
            else:
                total += self.base * (hi - lo) - self.above.integral(lo - i0, hi - i0)
        return total

    def _exact_integral(self, a: float, b: float, below: bool) -> float:
        p = self.exact
        base, i0, T = float(p["base"]), float(p.get("I0", self.i0)), p["T"]
        f = p["below_func"] if below else p["above_func"]
        target = p["below_target"] if below else p["above_target"]
        amp = amp_from_target(f, target, base, T)
        if below:
            return base * (b - a) + amp * (shape_integral(f, i0 - a, T) - shape_integral(f, i0 - b, T))
        return base * (b - a) - amp * (shape_integral(f, b - i0, T) - shape_integral(f, a - i0, T))

    def _floor_inventory(self, lo: int, hi: int) -> int:
        """Smallest integer inventory in [lo, hi+1] whose quote is at the floor (hi+1 if the
        quote never floors in that range). The quote is monotone non-increasing in inventory,
        so this is a bisection -- O(log n), never a walk."""
        if self.quote(hi) > PRICE_FLOOR:
            return hi + 1
        if self.quote(lo) <= PRICE_FLOOR:
            return lo
        a, b = lo, hi                       # quote(a) > floor, quote(b) <= floor
        while b - a > 1:
            m = (a + b) // 2
            if self.quote(m) > PRICE_FLOOR:
                a = m
            else:
                b = m
        return b

    def _sum_quotes(self, lo: int, hi: int) -> float:
        """`sum of quote(x)` for integer x in [lo, hi].

        Two regimes, because the engine's `int(round(...))` is what the integral cannot see:

        * **Few price levels** (a flat stretch of a cheap product -- WHEAT quoted at $21 for
          two hundred units) -- the rounding bias is systematic and does NOT cancel, so the
          quotes are summed exactly, one bisection per level. Measured: this is where the
          plain integral was worst, 2.2% on WHEAT.
        * **Many levels** -- the fractional parts sweep the whole interval, the rounding
          cancels, and the midpoint integral is right to a few parts in 10^5. One evaluation
          regardless of `n`.
        """
        if hi < lo:
            return 0.0
        f = self._floor_inventory(lo, hi)
        total = float(PRICE_FLOOR) * (hi - f + 1) if f <= hi else 0.0
        hi = min(f - 1, hi)
        if hi < lo:
            return total
        levels = self.quote(lo) - self.quote(hi)
        if levels <= MAX_EXACT_LEVELS:
            return total + self._sum_quotes_by_level(lo, hi)
        return total + self._integral(lo - 0.5, hi + 0.5)

    def _sum_quotes_by_level(self, lo: int, hi: int) -> float:
        """Exact sum over a range spanning few integer price levels: for each level, bisect
        for the last inventory that still quotes it."""
        total, x = 0.0, lo
        while x <= hi:
            v = self.quote(x)
            a, b = x, hi                      # quote(a) == v, quote(b) <= v
            if self.quote(b) == v:
                a = b
            else:
                while b - a > 1:
                    m = (a + b) // 2
                    if self.quote(m) >= v:
                        a = m
                    else:
                        b = m
            total += float(v) * (a - x + 1)
            x = a + 1
        return total

    def marginal_revenue(self, inventory: int, n: int) -> float:
        """Revenue of selling `n` units into this book.

        Each unit sold above the floor pushes the pool up by one; units sold *at* the floor do
        not move it (engine `_commit_unit`), so they all fetch exactly `$1`. That is plan §1
        row 12's floored-pool recovery falling out of the mechanics rather than a constant.
        """
        n = int(n)
        if n <= 0:
            return 0.0
        inventory = int(inventory)
        f = self._floor_inventory(inventory, inventory + n - 1)
        n_priced = max(0, min(n, f - inventory))
        value = (self._sum_quotes(inventory, inventory + n_priced - 1)
                 + float(PRICE_FLOOR) * (n - n_priced))
        self._record_read(inventory, inventory + n - 1, units=n)
        return value

    def priced_units(self, inventory: int, n: int) -> int:
        """How many of `n` units sold actually move the pool -- i.e. fetch more than the
        floor. The rest are the floored-pool tail and leave the inventory where it is."""
        n = int(n)
        if n <= 0:
            return 0
        inventory = int(inventory)
        f = self._floor_inventory(inventory, inventory + n - 1)
        self._record_read(inventory, inventory + n - 1, units=n)
        return max(0, min(n, f - inventory))

    def marginal_cost(self, inventory: int, n: int) -> float:
        """Cost of BUYING `n` units: the engine quotes at `inventory - 1` and decrements, so
        the units are priced at `inventory-1 .. inventory-n` (row 10b, the two-sided pool)."""
        n = int(n)
        if n <= 0:
            return 0.0
        inventory = int(inventory)
        cost = self._sum_quotes(inventory - n, inventory - 1)
        self._record_read(inventory - n, inventory - 1, units=n)
        return cost

    def units_above(self, inventory: int, fraction: float) -> int:
        """Headroom: how many units can be sold before the quote falls below `fraction * base`
        (the steep-pool diagnostic that orders liquidation, spec §3.8)."""
        if self.base is None:
            self._record_point_read(inventory)
            return 0
        target = fraction * self.base
        if self.price_at(inventory) < target:
            self._record_point_read(inventory)
            return 0
        lo, hi = 0, 1
        while hi < 1 << 20 and self.price_at(inventory + hi) >= target:
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            m = (lo + hi) // 2
            if self.price_at(inventory + m) >= target:
                lo = m
            else:
                hi = m
        self._record_read(float(inventory), float(inventory + lo))
        return lo

    # -- diagnostics ------------------------------------------------------------------------
    def residual_median(self) -> float:
        """Median absolute out-of-sample error over the tripwire window (both sides)."""
        return max(_median([r[0] for r in self.below.residuals]),
                   _median([r[0] for r in self.above.residuals]))

    def residual_median_relative(self) -> float:
        return max(_median([r[1] for r in self.below.residuals]),
                   _median([r[1] for r in self.above.residuals]))

    def diagnostics(self) -> dict:
        below, above = self.fitted_shape
        return {
            "product": self.product, "base": self.base, "i0": self.i0,
            "i0_observed": self.i0_observed, "n_points": self.n_points,
            "confidence": round(self.confidence, 4),
            "below_confidence": round(self.side_confidence("below"), 4),
            "above_confidence": round(self.side_confidence("above"), 4),
            "below_reach": self.side_reach("below"), "above_reach": self.side_reach("above"),
            "last_read": (self.last_read or {}).get("provenance", "none"),
            "below_shape": below, "above_shape": above,
            "below_rmse": round(self.below.rmse, 3), "above_rmse": round(self.above.rmse, 3),
            "below_T": self.below.T, "above_T": self.above.T,
            "residual_median": round(self.residual_median(), 3),
            "residual_median_relative": round(self.residual_median_relative(), 4),
            "n_residuals": len(self.below.residuals) + len(self.above.residuals),
        }


# ----------------------------------------------------------------------------- all nine
class CurveSet:
    """All nine products; the object an agent keeps for the whole episode."""

    def __init__(self, tolerance: float = RESIDUAL_TOLERANCE,
                 relative_tolerance: float = RELATIVE_TOLERANCE) -> None:
        self.curves = {p: PriceCurve(p) for p in PRODUCTS}
        self.tolerance = float(tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.step = -1
        self.last_errors: dict[str, float] = {}
        self._order_cache: tuple[int, float, tuple[str, ...]] | None = None

    @classmethod
    def from_params(cls, params: dict) -> "CurveSet":
        """Exact curves from an explicit MARKET_PARAMS-shaped dict. OFFLINE/test use only."""
        cs = cls()
        for p, entry in params.items():
            if p in cs.curves:
                cs.curves[p] = PriceCurve.from_params(p, entry)
        return cs

    def __getitem__(self, product: str) -> PriceCurve:
        if product not in self.curves:
            self.curves[product] = PriceCurve(product)
        return self.curves[product]

    def __contains__(self, product: str) -> bool:
        return product in self.curves

    def observe(self, market, step: int | None = None) -> None:
        """One point per product per turn. O(9 * #models) and only when a pool actually moved."""
        inv = market["inventory"] if isinstance(market, dict) else market.inventory
        prices = market["prices"] if isinstance(market, dict) else market.prices
        self.step = self.step + 1 if step is None else int(step)
        self.last_errors = {}
        for product in inv:
            err = self[product].observe(int(inv[product]), int(prices[product]), step=self.step)
            if err is not None:
                self.last_errors[product] = err

    # -- the engine-bump tripwire ------------------------------------------------------------
    def tripwire(self, tolerance: float | None = None,
                 relative_tolerance: float | None = None) -> dict[str, float]:
        """Products whose recent OUT-OF-SAMPLE prediction error is persistently too large:
        the fitted curve has stopped describing the market it is fitted to.

        This is the runtime half of plan §1 row 16, and the thing that makes the doctrine
        "all curves measured at runtime" *checkable* instead of merely stated. It fires when
        a curve is not representable by the five shapes (a new engine release adding one), or
        when the fit has broken -- and it stays quiet for a curve that is merely unfamiliar
        or freshly rebalanced, because `nav` re-derives those from observation. The residual
        is measured against each point BEFORE that point is folded into the fit, so it is a
        genuine out-of-sample error and not a goodness-of-fit statistic.

        `nav` reports it and does nothing about it; what to do is the caller's decision.
        Returns `{product: median relative error}`.
        """
        tol = self.tolerance if tolerance is None else float(tolerance)
        rel_tol = self.relative_tolerance if relative_tolerance is None else float(relative_tolerance)
        out = {}
        for p, c in self.curves.items():
            n = len(c.below.residuals) + len(c.above.residuals)
            if n < TRIPWIRE_MIN_POINTS:
                continue
            if c.residual_median() > tol and c.residual_median_relative() > rel_tol:
                out[p] = round(c.residual_median_relative(), 4)
        return out

    def diagnostics(self) -> dict[str, dict]:
        return {p: c.diagnostics() for p, c in self.curves.items()}

    def liquidation_order(self, inventory: dict, fraction: float = 0.5,
                          day: int | None = None) -> tuple[str, ...]:
        """Steepest pool first: the measured headroom above `fraction * base`, ascending.

        Re-derived from the fitted curves, never a frozen constant -- the probe's WOOL..EGG
        ordering is what this returns on the shipped engine, but it follows a rebalance
        without a code change. Cached per day (§7: it only changes as the pools move).
        """
        d = self.step // 24 if day is None else int(day)
        if self._order_cache and self._order_cache[0] == d and self._order_cache[1] == fraction:
            return self._order_cache[2]

        def headroom(p: str) -> tuple[int, str]:
            inv = int(inventory.get(p, self[p].i0)) if inventory else self[p].i0
            return (self[p].units_above(inv, fraction), p)

        order = tuple(p for _, p in sorted((headroom(p) for p in self.curves)))
        self._order_cache = (d, fraction, order)
        return order
