"""Terminal-bank line accounting for `exp-014`'s factored critic.

The factored critic (plan.md section 2, exp-014) decomposes the value head over "terminal-bank
lines". For that to be a decomposition rather than a naming convention the lines have to be
*exact*: their sum must be the bank, with no residual to hide a bug in.

**The engine gives us that exactly.** `farm["money"]` is mutated in exactly six places in
`kaggle_environments/envs/kaggriculture/kaggriculture.py` (verified against 1.32.7, engine sha
`bc8a54879ef0`):

    _commit_unit   SELL          money += price                 (line 657)
    _commit_unit   BUY_PRODUCT   money -= price                 (line 669)
    _commit_unit   BUY_SEED      money -= CROPS[item]["seed"]   (line 676)
    _commit_unit   BUY_ANIMAL    money -= ANIMALS[item]["cost"] (line 684)
    _do_hire                     money -= _hire_cost(...)       (line 706)
    _do_buy_land                 money -= land cost             (line 719)

so

    final_bank == startingMoney + sell revenue - buys - hires - land

holds to the cent, and `Accountant.residual()` asserts it on every episode. Anything that
breaks the identity (an engine bump, a missed cash path) shows up as a non-zero residual
rather than as a quietly mis-attributed critic target.

**Correction to plan.md's line list, and it matters.** The plan names "feed bill" as a
terminal-bank line. There is no cash feed bill: feeding consumes *shed items*, never money.
The feed bill is the forgone sale value of the wheat/fertilizer that went into an animal
instead of the market -- a shadow cost that is an allocation *within* the sell/buy lines, not a
seventh cash line. The same is true of "liquidation leak" (inventory still in the shed at the
horizon is money we never received). Both are kept, as SHADOW_LINES: auxiliary regression
targets for the critic trunk, never members of the value decomposition. Mixing them in would
have made the lines sum to something that is not the bank.

The accountant is *observational*. It wraps three engine functions and `env.interpreter`, takes
no decisions, and changes no action; `tests/test_ppo_lines.py` pins that an instrumented
episode and a clean one produce byte-identical banks.
"""
from __future__ import annotations

import numpy as np

from .bcpath import F

# The nine shared market pools, in the engine's own order. Identical to
# `kaggriculture.PRODUCTS` and to `z["products"]` in Ops' S-2 label shard, which is what makes
# a rollout-computed label and an S-2 corpus label the same column.
PRODUCTS = tuple(F.PRODUCTS)
N_PRODUCTS = len(PRODUCTS)
_PIDX = {p: i for i, p in enumerate(PRODUCTS)}

# Per-product flow rows, matching S-2's `p{i}_sold` / `p{i}_sold_floored` / `p{i}_bought`.
FLOWS = ("sold", "sold_floored", "bought")
_FIDX = {name: i for i, name in enumerate(FLOWS)}

# The exact cash decomposition. Sum(LINES) + starting money == final bank.
LINES = (
    "sell_staple",        # SELL revenue: WHEAT, CARROT
    "sell_premium",       # SELL revenue: TOMATO, STRAWBERRY, MELON
    "sell_animal",        # SELL revenue: EGG, MILK, WOOL
    "sell_fertilizer",    # SELL revenue: FERTILIZER  (its own line -- expB-307's two-sided
                          # thesis lives here and it is the only product we both buy and sell)
    "cost_inputs",        # -(BUY_PRODUCT + BUY_SEED)
    "cost_capital",       # -(BUY_ANIMAL + BUY_LAND)
    "cost_labour",        # -HIRE
)
N_LINES = len(LINES)

# Not cash. Auxiliary regression targets only -- see the module docstring.
SHADOW_LINES = ("feed_bill", "liquidation_leak")

_STAPLE = ("WHEAT", "CARROT")
_PREMIUM = ("TOMATO", "STRAWBERRY", "MELON")
_ANIMAL = ("EGG", "MILK", "WOOL")

_IDX = {name: i for i, name in enumerate(LINES)}


def line_of(op, item):
    """(engine op, item) -> index into LINES, or None if the op moves no cash."""
    if op == "SELL":
        if item in _STAPLE:
            return _IDX["sell_staple"]
        if item in _PREMIUM:
            return _IDX["sell_premium"]
        if item in _ANIMAL:
            return _IDX["sell_animal"]
        if item == "FERTILIZER":
            return _IDX["sell_fertilizer"]
        return None
    if op in ("BUY_PRODUCT", "BUY_SEED"):
        return _IDX["cost_inputs"]
    if op in ("BUY_ANIMAL", "BUY_LAND"):
        return _IDX["cost_capital"]
    if op == "HIRE":
        return _IDX["cost_labour"]
    return None


_ACTIVE = []          # innermost accountant, or empty
_PATCHED = False


def _install(K):
    """Patch the engine's three cash paths ONCE per process.

    Patch-and-restore around every episode would be a data race the moment two accountants
    exist in one interpreter; a permanent wrapper that no-ops when `_ACTIVE` is empty costs one
    list truth-test per commit and cannot get out of step with itself.
    """
    global _PATCHED
    if _PATCHED:
        return
    orig_commit, orig_hire, orig_land = K._commit_unit, K._do_hire, K._do_buy_land

    def commit(op, item, price, farm, private, market, shed_capacity=100):
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if ok and _ACTIVE:
            _ACTIVE[-1].note(farm, op, item, float(price))
        return ok

    def do_hire(farm, private, board_size, mult=K.FARM_HAND_COST_MULT):
        before = float(farm["money"])
        out = orig_hire(farm, private, board_size, mult)
        spent = before - float(farm["money"])
        if spent and _ACTIVE:
            _ACTIVE[-1].note(farm, "HIRE", "-", spent)
        return out

    def do_land(farm, board_size):
        before = float(farm["money"])
        out = orig_land(farm, board_size)
        spent = before - float(farm["money"])
        if spent and _ACTIVE:
            _ACTIVE[-1].note(farm, "BUY_LAND", "-", spent)
        return out

    K._commit_unit, K._do_hire, K._do_buy_land = commit, do_hire, do_land
    _PATCHED = True


class Accountant:
    """Per-episode terminal-bank line totals for both seats.

    Use as a context manager around a single `env.run(...)`:

        with Accountant(K).attach(env) as acc:
            env.run(agents)
        lines = acc.totals(seat)          # (N_LINES,) dollars, signed
        assert abs(acc.residual(seat, bank, start)) < 1e-6
    """

    def __init__(self, K):
        self.K = K
        _install(K)
        self.tot = np.zeros((2, N_LINES), dtype=np.float64)
        self.units = np.zeros((2, N_LINES), dtype=np.int64)
        # (seat, flow, product) running unit counts -- the S-2 flow block, computed live.
        self.prod = np.zeros((2, len(FLOWS), N_PRODUCTS), dtype=np.int64)
        self.price_floor = float(getattr(K, "PRICE_FLOOR", 1))
        self.start_money = None
        self._fmap = {}
        self._on_step = []

    # -- wiring ---------------------------------------------------------------
    def on_step(self, fn):
        """Register `fn(state, env, acc)`, called at the START of every interpreter call.

        The one patch point stays the one patch point: an auxiliary recorder that needs a
        per-step snapshot (market inventory, unlocked shops, the running flow tally) hangs off
        this instead of monkeypatching the engine a second time -- which is the data race the
        module docstring exists to prevent.
        """
        self._on_step.append(fn)
        return self

    def attach(self, env):
        inner = env.interpreter

        def interpreter(state, env_):
            farms = state[0].observation["farms"]
            self._fmap = {id(f): i for i, f in enumerate(farms)}
            if self.start_money is None and len(farms) == 2:
                # Captured rather than assumed: `startingMoney` is configuration, and the
                # residual assertion is worthless if its constant can drift from the engine's.
                # `len(farms) == 2` because `env.reset` runs the interpreter once on a state
                # whose observation has no farms yet -- that call must not be mistaken for the
                # first turn (it is also why `Recorder` dedupes on the step index).
                self.start_money = [float(f["money"]) for f in farms]
            for fn in self._on_step:
                fn(state, env_, self)
            return inner(state, env_)

        env.interpreter = interpreter
        return self

    def __enter__(self):
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc):
        if _ACTIVE and _ACTIVE[-1] is self:
            _ACTIVE.pop()
        return False

    # -- accumulation ---------------------------------------------------------
    def note(self, farm, op, item, price):
        p = self._fmap.get(id(farm))
        if p is None:
            return
        j = _PIDX.get(item)
        if j is not None:
            # S-2 flow semantics, verbatim: a SELL at the $1 floor pays the seller but adds no
            # market supply, so `sold_floored` is the part of `net_true` no inventory-based
            # observer can ever recover. `_commit_unit` is called once per UNIT, so these are
            # unit counts, not order counts.
            if op == "SELL":
                self.prod[p, _FIDX["sold"], j] += 1
                if price <= self.price_floor:
                    self.prod[p, _FIDX["sold_floored"], j] += 1
            elif op == "BUY_PRODUCT":
                self.prod[p, _FIDX["bought"], j] += 1
        i = line_of(op, item)
        if i is None:
            return
        self.tot[p, i] += price if op == "SELL" else -price
        self.units[p, i] += 1

    # -- reading --------------------------------------------------------------
    def totals(self, seat):
        """Lines for `seat`, ordered as LINES, in dollars (revenue +, cost -)."""
        return self.tot[seat].copy()

    def residual(self, seat, bank, starting_money=None):
        """bank - (startingMoney + sum(lines)). Zero on a correct decomposition.

        `starting_money` defaults to the value observed at the first interpreter call, so the
        identity is checked against the engine's own configuration rather than a constant in this
        file that an engine rebalance could silently invalidate.
        """
        if starting_money is None:
            starting_money = (self.start_money[seat] if self.start_money is not None else 3000.0)
        return float(bank) - (float(starting_money) + float(self.tot[seat].sum()))

    def net_units(self, seat):
        """(9,) units this seat SOLD minus BOUGHT over the episode -- S-2's `net_true`."""
        return (self.prod[seat, _FIDX["sold"], :]
                - self.prod[seat, _FIDX["bought"], :]).astype(np.int64)
