"""expB-304b-b0 — invariants of the landing-inventory projection.

These pin the two engine facts the projection would silently get wrong, both of
which it DID get wrong on the first cut and which the back-test caught:

  1. `FEED` consumes one WHEAT (`_apply_unit_action`: `_inv_take(inv, "WHEAT", 1)`),
     so a farm with a herd is a wheat SINK.  Counting gross production as market
     supply inverts the sign of the whole wheat projection.
  2. MELON is in **zero** shops, so its entire town drain is the town centre's
     1 unit/day — a hard constant, not a draw.  STRAWBERRY is in four of eight,
     so its drain swings with the unlock draw.  A test that let melon's drain
     drift would let a "drain-aware" melon rule read a number that does not exist.

Plus the structural ones: the lead table comes from the engine's own CROPS /
ANIMALS tables, and the unlock posterior is integrated over the window rather
than applied at the end-state rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "agents" / "team_b" / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import drain_project as dp                       # noqa: E402
from nav.book import DemandBook                  # noqa: E402
from nav.rules import SHOPS, TURNS_PER_DAY       # noqa: E402


def _farm_with(tiles):
    return {"tiles": tiles}


def _blank(n=4):
    return [[None] * n for _ in range(n)]


# --------------------------------------------------------------- the wheat sink
def test_feed_makes_a_herd_a_wheat_sink():
    tiles = _blank()
    tiles[0][0] = {"animal": "COW", "yield_units": 0}
    tiles[0][1] = {"animal": "GOOSE", "yield_units": 0}
    farm = _farm_with(tiles)
    # two animals, a four-day window, one feed per animal per day
    demand = dp.internal_consumption("WHEAT", farm, step=0, lead=4 * TURNS_PER_DAY)
    assert demand == pytest.approx(8.0)


def test_internal_consumption_is_wheat_only():
    tiles = _blank()
    tiles[0][0] = {"animal": "COW", "yield_units": 0}
    farm = _farm_with(tiles)
    for product in ("STRAWBERRY", "MELON", "MILK", "EGG", "WOOL", "CARROT"):
        assert dp.internal_consumption(product, farm, 0, 240) == 0.0


def test_own_pipeline_is_net_of_the_herd():
    """A wheat farm with a big herd projects NEGATIVE net supply -- it buys."""
    tiles = _blank(6)
    tiles[0][0] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 3, "planted_day": 0}
    for i in range(1, 6):
        tiles[1][i % 6] = {"animal": "COW", "yield_units": 0}
    farm = _farm_with(tiles)
    private = {"shed": {"WHEAT": 5}, "inventories": [{}]}
    net = dp.own_pipeline_landing("WHEAT", farm, private, step=0, lead=2 * TURNS_PER_DAY)
    # 3 standing + 5 shed = 8 gross, against 5 cows x 2 days = 10 eaten
    assert net == pytest.approx(-2.0)


# --------------------------------------------------------------- the drain facts
def test_melon_has_no_shop_and_therefore_a_constant_drain():
    assert [s for s, prods in SHOPS.items() if "MELON" in prods] == []
    for unlocked in ((), tuple(SHOPS), ("FARMERS_MARKET", "PIZZA_SHOP")):
        book = DemandBook()
        book.unlocked = unlocked
        assert book.drain_per_day("MELON") == pytest.approx(1.0)


def test_strawberry_drain_moves_with_the_unlock_draw():
    lo, hi = DemandBook(), DemandBook()
    lo.unlocked = ("YARN_STORE", "PET_CAFE")
    hi.unlocked = ("BRUNCH_SPOT", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "FARMERS_MARKET")
    assert lo.drain_per_day("STRAWBERRY") == pytest.approx(1.0)
    assert hi.drain_per_day("STRAWBERRY") == pytest.approx(25.0)


def test_fertilizer_has_no_drain_and_no_lead():
    book = DemandBook()
    book.unlocked = tuple(SHOPS)
    assert book.drain_per_day("FERTILIZER") == 0.0
    assert "FERTILIZER" not in dp.LEAD_STEPS


# --------------------------------------------------------------- the posterior
def test_unlock_posterior_is_integrated_not_applied_flat():
    """A draw scheduled for day 27 must not drain a window that closes on day 12.

    The flat shortcut (`expected_extra_drain_per_day * days`) over-counts, and on
    STRAWBERRY's ten-day lead that is the difference between growing and not.
    """
    book = DemandBook()
    book.unlocked = ()
    book.observe({"unlocked_shops": []}, 0)
    lead = 10 * TURNS_PER_DAY
    integrated = dp.expected_extra_unlock_drain(book, "STRAWBERRY", step=0, lead=lead)
    flat = book.expected_extra_drain_per_day("STRAWBERRY", 0) * (lead / TURNS_PER_DAY)
    assert integrated < flat
    assert integrated > 0.0


def test_lead_table_comes_from_the_engine_tables():
    from nav.rules import ANIMALS, CROPS
    assert dp.LEAD_STEPS["STRAWBERRY"] == CROPS["STRAWBERRY"]["first_yield_day"] * TURNS_PER_DAY
    assert dp.LEAD_STEPS["MELON"] == CROPS["MELON"]["first_yield_day"] * TURNS_PER_DAY
    assert dp.LEAD_STEPS["EGG"] == ANIMALS["GOOSE"]["first_yield_day"] * TURNS_PER_DAY


# --------------------------------------------------------------- season revenue
def test_season_revenue_counts_the_opponents_volume_in_the_same_pool():
    """A mirror census's realised price reflects BOTH seats selling."""
    from nav.curves import PriceCurve
    curve = PriceCurve("MELON", base=250, i0=10000)
    curve.observe(10000, 250, step=0)
    curve.observe(10100, 150, step=1)
    book = DemandBook()
    book.unlocked = ()
    solo = dp.season_revenue(curve, book, "MELON", 100, 10000, 240, 719, opponent_units=0)
    mirror = dp.season_revenue(curve, book, "MELON", 100, 10000, 240, 719)
    assert mirror < solo          # the same units fetch less into a fuller pool


def test_price_from_state_replays_the_curve_exactly():
    from nav.curves import PriceCurve
    curve = PriceCurve("WHEAT", base=30, i0=10000)
    for inv, price in ((10000, 30), (10050, 26), (10100, 23), (10200, 19)):
        curve.observe(inv, price, step=0 if inv == 10000 else 1)
    state = dp.curve_state(curve)
    for inv in (9900, 10000, 10120, 10400):
        assert dp.price_from_state(state, inv) == pytest.approx(curve.price_at(inv), rel=1e-9)
