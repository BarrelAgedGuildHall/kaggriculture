"""`Nav`: the per-turn object an agent holds, and the offline replay that reproduces it.

Two phases, for the runtime budget (spec §7):

* `observe(obs)` is **cheap and every turn**: one curve point per product, the town's unlock
  set, the opponent's flow residual. O(9 products + 1 town + 2 farms).
* `sheet()` is the **expensive one**: O(tiles + herd). Callers may run it every turn (Team A's
  feature path) or once a day (Team B's allocator). That choice is the caller's.

Nothing here decides anything. `Nav` answers *what is this worth* and *what is exactly
inferable*; every judgement about what to do with the number stays in the agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from .book import DemandBook
from .curves import CurveSet, RESIDUAL_TOLERANCE
from .lam import LAMBDA_CRITIC, LambdaPair
from .opponent import OpponentBook
from .rules import SHED_CAPACITY, TURNS_PER_DAY
from .sheet import Sheet, VARIANTS, build_sheet


def obs_get(obs, key: str, default=None):
    """Observations arrive as a `Struct` in the engine and as a plain dict offline."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def step_of(obs) -> int:
    """`obs.step` when the framework supplies it, else `day * 24 + hour` -- which is the same
    number, because the interpreter advances day/hour to the snapshot's own index."""
    step = obs_get(obs, "step")
    if step is None:
        step = int(obs_get(obs, "day", 0)) * TURNS_PER_DAY + int(obs_get(obs, "hour", 0))
    return int(step)


@dataclass(frozen=True)
class NavConfig:
    """Everything a caller may tune. No thresholds, no policy -- see the spec's one rule."""
    variant: str = "drain"
    lam: LambdaPair = LAMBDA_CRITIC
    shed_capacity: int = SHED_CAPACITY
    residual_tolerance: float = RESIDUAL_TOLERANCE


class Nav:
    def __init__(self, config: NavConfig | None = None) -> None:
        self.config = config or NavConfig()
        if self.config.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.curves = CurveSet(tolerance=self.config.residual_tolerance)
        self.book = DemandBook()
        self.opponent = OpponentBook(self.book)
        self.obs = None
        self.step = -1
        self.seat = 0
        self.turns = 0

    # -- the cheap phase ----------------------------------------------------------------------
    def observe(self, obs, own_flow=None) -> "Nav":
        """Call once per turn, before anything else. `own_flow` is {product: net units WE put
        into the pool on the step that just resolved} -- see `nav.opponent`."""
        step = step_of(obs)
        market, town, farms = obs_get(obs, "market"), obs_get(obs, "town"), obs_get(obs, "farms")
        self.seat = int(obs_get(obs, "player", 0) or 0)
        self.curves.observe(market, step=step)
        self.book.observe(town, step)
        if farms and len(farms) > 1:
            self.opponent.observe(market, farms, self.seat, step, own_flow=own_flow)
        self.obs = obs
        self.step = step
        self.turns += 1
        return self

    # -- the expensive phase ------------------------------------------------------------------
    def sheet(self, *, variant: str | None = None, lam: LambdaPair | None = None,
              obs=None) -> Sheet:
        """The full mark for our seat."""
        obs = obs if obs is not None else self.obs
        if obs is None:
            raise RuntimeError("Nav.sheet() before Nav.observe()")
        seat = int(obs_get(obs, "player", 0) or 0)
        farms = obs_get(obs, "farms")
        return build_sheet(
            farms[seat], obs_get(obs, "private", {}) or {}, obs_get(obs, "market"),
            obs_get(obs, "town"), step_of(obs), seat, self.curves, self.book,
            lam or self.config.lam, variant or self.config.variant,
            shed_capacity=self.config.shed_capacity,
        )

    def opponent_sheet(self, *, variant: str | None = None, lam: LambdaPair | None = None,
                       obs=None) -> Sheet:
        """Public-only marks for the other seat (§5). Their shed, carried goods and seeds are
        not in it -- `diagnostics["unmarkable_lines"]` says so, and `OpponentBook
        .unmarkable_residual()` carries the public evidence about them."""
        obs = obs if obs is not None else self.obs
        if obs is None:
            raise RuntimeError("Nav.opponent_sheet() before Nav.observe()")
        seat = int(obs_get(obs, "player", 0) or 0)
        farms = obs_get(obs, "farms")
        if not farms or len(farms) < 2:
            raise RuntimeError("no opponent in this observation")
        them = 1 - seat
        sheet = build_sheet(
            farms[them], None, obs_get(obs, "market"), obs_get(obs, "town"), step_of(obs),
            them, self.curves, self.book, lam or self.config.lam, variant or self.config.variant,
            shed_capacity=self.config.shed_capacity, public_only=True,
        )
        return sheet

    def phi_rel(self, *, variant: str | None = None, lam: LambdaPair | None = None,
                obs=None) -> float:
        """`(nav - cash)_us - (nav - cash)_them`.

        Terminally zero (both sides are zero at H = 0) and win-aligned: the competition ranks
        on money, so the quantity that matters is not how much we are worth but how much more
        we are worth. Their hidden shed and seeds are NOT in their side, so this is an upper
        bound on the true relative potential; `nav.opponent.OpponentBook
        .unmarkable_residual()` is what the caller inspects to judge the size of the gap.
        """
        ours = self.sheet(variant=variant, lam=lam, obs=obs)
        theirs = self.opponent_sheet(variant=variant, lam=lam, obs=obs)
        return ours.phi - theirs.phi

    # -- offline ------------------------------------------------------------------------------
    @classmethod
    def replay(cls, observations, config: NavConfig | None = None, own_flows=None) -> "Nav":
        """Rebuild the online object from a recorded trace. Feeding the same observations in
        the same order must produce the same state, which is what lets exp-N1/S-3 mark
        recorded episodes with the very code the agent runs."""
        nav = cls(config)
        for i, obs in enumerate(observations):
            nav.observe(obs, own_flow=(own_flows[i] if own_flows else None))
        return nav

    @classmethod
    def replay_sheets(cls, observations, at=None, config: NavConfig | None = None,
                      variant: str | None = None, lam: LambdaPair | None = None):
        """Yield `(step, Sheet)` at the requested snapshot indices (default: every one) while
        replaying. The curves only ever see points up to that step, exactly as online."""
        nav = cls(config)
        want = None if at is None else set(int(a) for a in at)
        for obs in observations:
            nav.observe(obs)
            if want is None or nav.step in want:
                yield nav.step, nav.sheet(variant=variant, lam=lam)

    # -- diagnostics --------------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "step": self.step, "turns": self.turns, "seat": self.seat,
            "tripwire": self.curves.tripwire(),
            "curves": self.curves.diagnostics(),
            "book": self.book.as_dict(),
            "opponent": self.opponent.as_dict(),
        }
