"""lambda: the price of an action, always an explicit argument -- never a module default.

Two numbers, because a turn spent is two different costs depending on the question:

* `wage` -- the **floor**: what an action costs when it is bought as farm-hand labour.
* `opp`  -- the **opportunity cost** of a unit-action: what the marginal action would have
  earned doing the next-best thing.

exp-N1 §5.3 measured that the two tests it ran disagree, and the disagreement is the finding,
not noise: Test 1b (what a day of liquidation actually converts) favours lambda ~ 50, the
revenue-per-productive-action estimate; Test 2 (what predicts the final bank) favours
lambda = 15, the derived shadow price. They answer different questions.

**Documented recommendation, not a default:** lambda_opp = 15 for the critic-feature use
(exp-014), lambda_opp = 50 for liquidation accounting. `Nav.sheet()` requires the caller to
say which, or inherits the `LambdaPolicy` the caller configured. `nav` never picks for you.
"""
from __future__ import annotations


class LambdaPair:
    """`wage` and `opp` in dollars per action. Immutable and hashable, so a Sheet can carry it."""

    __slots__ = ("wage", "opp")

    def __init__(self, opp: float, wage: float | None = None) -> None:
        object.__setattr__(self, "opp", float(opp))
        object.__setattr__(self, "wage", float(opp if wage is None else wage))

    def __setattr__(self, *_a):
        raise AttributeError("LambdaPair is immutable")

    def __eq__(self, other) -> bool:
        return isinstance(other, LambdaPair) and (self.opp, self.wage) == (other.opp, other.wage)

    def __hash__(self) -> int:
        return hash((self.opp, self.wage))

    def __repr__(self) -> str:
        return f"LambdaPair(opp={self.opp:g}, wage={self.wage:g})"

    def as_dict(self) -> dict:
        return {"opp": self.opp, "wage": self.wage}


# The two calibrated arms of exp-N1 §5.3, named so an experiment row can say which it used.
LAMBDA_CRITIC = LambdaPair(opp=15.0, wage=15.0)        # Test 2 arm: predicts the final bank
LAMBDA_LIQUIDATION = LambdaPair(opp=50.0, wage=15.0)   # Test 1b arm: what liquidation converts


def fixed_lambda(opp: float, wage: float | None = None):
    """A constant policy. `policy.at(step)` -> LambdaPair."""
    pair = LambdaPair(opp, wage)

    class _Fixed:
        def at(self, step: int) -> LambdaPair:
            return pair

        def __repr__(self) -> str:
            return f"fixed_lambda({pair})"

    return _Fixed()


class LiveLambda:
    """The expB-302 rolling estimator: 3-day rolling `(gross - cogs) / productive_actions`
    scaled by `(1 - idle_fraction)`, floored at `wage`.

    Stateful and **fed by the caller**: gross, COGS, productive actions and idle fraction are
    the agent's own accounting, not anything `nav` can read out of an observation. Until a
    full window has been recorded it returns the fallback pair, so an agent that forgets to
    feed it gets the documented constant rather than a silently drifting number.
    """

    def __init__(self, wage: float = 15.0, fallback_opp: float = 15.0, window_days: int = 3) -> None:
        self.wage = float(wage)
        self.fallback = LambdaPair(opp=float(fallback_opp), wage=float(wage))
        self.window_days = int(window_days)
        self.days: list[tuple[int, float, float, float, float]] = []

    def record(self, day: int, gross: float, cogs: float, productive_actions: float,
               idle_fraction: float = 0.0) -> None:
        self.days = [d for d in self.days if d[0] != int(day)]
        self.days.append((int(day), float(gross), float(cogs), float(productive_actions),
                          float(idle_fraction)))
        self.days.sort()
        if len(self.days) > self.window_days:
            self.days = self.days[-self.window_days:]

    def at(self, step: int | None = None) -> LambdaPair:
        if len(self.days) < self.window_days:
            return self.fallback
        gross = sum(d[1] for d in self.days)
        cogs = sum(d[2] for d in self.days)
        actions = sum(d[3] for d in self.days)
        idle = sum(d[4] for d in self.days) / len(self.days)
        if actions <= 0:
            return self.fallback
        opp = (gross - cogs) / actions * (1.0 - idle)
        return LambdaPair(opp=max(self.wage, opp), wage=self.wage)

    def __repr__(self) -> str:
        return f"LiveLambda(window={self.window_days}d, n={len(self.days)}, at={self.at()})"
