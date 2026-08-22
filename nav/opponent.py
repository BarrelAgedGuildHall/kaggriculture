"""The opponent's books from public state and exactly inferable flows (plan §1 row 13).

This is the piece exp-N1 explicitly did not build (§7.3) and the one exp-016's dual head and
Team B's M2 allocator both want. It rests on one fact about the observation:

> **`obs.farms` is public for BOTH seats.** Money, every tile, the farmer and hands, hires
> today and unlocked quadrants -- all of it. Only `obs.private` (shed, seeds, unit
> inventories) is our seat's alone.

So the opponent's crops, animals, structures and cash can be marked with exactly the same
functions that mark ours (`build_sheet(..., public_only=True)`), and their shed/seeds are the
**unmarkable residual** -- reported, never guessed.

The flow inference is the other half. The market pool moves by
`delta_inventory = sales - buys - town_drain`, and the town drain is exact from `obs.town`.
So, per product and per step:

    their_net_flow = delta_inventory + town_drain - our_net_flow

Our own executed flow is the one term `nav` cannot read: our shed changes for several reasons
in one turn and the engine never reports what filled. The caller passes it (`own_flow`, which
its own executor knows exactly); without it the whole residual is attributed to the opponent
and every affected reading is flagged `provisional`. `nav` reports the ambiguity rather than
resolving it by assumption.
"""
from __future__ import annotations

from .book import DemandBook
from .rules import PRODUCTS


class OpponentBook:
    """Their public cash and their inferred market flow, accumulated per product."""

    def __init__(self, book: DemandBook) -> None:
        self.book = book
        self.seat: int | None = None
        self.step = -1
        self._prev_inv: dict[str, int] | None = None
        self._prev_step: int | None = None
        self.net_flow = {p: 0.0 for p in PRODUCTS}       # + = they put units into the pool
        self.own_flow_total = {p: 0.0 for p in PRODUCTS}
        self.provisional = False                          # True once a step lacked `own_flow`
        self.cash_value = 0.0
        self.cash_start: float | None = None
        self.steps_seen = 0
        self.steps_without_own_flow = 0

    # -- evidence -----------------------------------------------------------------------------
    def observe(self, market, farms, seat: int, step: int, own_flow=None) -> None:
        """`own_flow`: {product: net units WE put into the pool this step} (sales positive,
        buys negative). Omit it and the residual is attributed to the opponent, flagged."""
        inv = market["inventory"] if isinstance(market, dict) else market.inventory
        inv = {k: int(v) for k, v in inv.items()}
        step = int(step)
        self.seat = 1 - int(seat) if len(farms) > 1 else int(seat)
        them = farms[self.seat]
        self.cash_value = float(them["money"])
        if self.cash_start is None:
            self.cash_start = self.cash_value

        if self._prev_inv is not None and step > (self._prev_step or 0):
            self.steps_seen += 1
            if own_flow is None:
                self.provisional = True
                self.steps_without_own_flow += 1
            for p, now in inv.items():
                delta = now - self._prev_inv.get(p, now)
                drain = self.book.drain_over_steps(p, self._prev_step, step - 1)
                ours = float((own_flow or {}).get(p, 0.0))
                self.own_flow_total[p] = self.own_flow_total.get(p, 0.0) + ours
                self.net_flow[p] = self.net_flow.get(p, 0.0) + (delta + drain - ours)
        self._prev_inv = inv
        self._prev_step = step
        self.step = step

    # -- readings -------------------------------------------------------------------------------
    def net_sales(self, product: str) -> float:
        """Net units the opponent has put into the pool (negative = they are net buyers).
        Exact when `own_flow` has been supplied on every step; otherwise it also contains our
        own flow and `provisional` is True."""
        return self.net_flow.get(product, 0.0)

    def cash(self) -> float:
        """`obs.farms[them]["money"]` -- public, exact, no inference at all."""
        return self.cash_value

    def cash_earned(self) -> float:
        return self.cash_value - (self.cash_start if self.cash_start is not None else self.cash_value)

    def unmarkable_residual(self) -> dict:
        """What we cannot see. Reported, never guessed.

        Their shed and seeds are hidden, so their sheet is missing exactly the `shed`,
        `carried` and `seeds` lines. `net_sales` is the public evidence about how much has
        already flowed out of that hidden stock -- evidence, not an estimate of it.
        """
        return {
            "hidden": "shed,seeds,inventories",
            "missing_lines": ("shed", "carried", "seeds"),
            "net_sales": dict(self.net_flow),
            "provisional": self.provisional,
            "steps_seen": self.steps_seen,
            "steps_without_own_flow": self.steps_without_own_flow,
        }

    def as_dict(self) -> dict:
        return {"seat": self.seat, "cash": self.cash_value, "cash_earned": self.cash_earned(),
                "net_sales": dict(self.net_flow), "provisional": self.provisional}
