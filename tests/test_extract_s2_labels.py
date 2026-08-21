"""Tests for the S-2 label extractor.

The load-bearing property is the market inventory identity: for every product and
every transition,

    inv[t+1] - inv[t] == sum_players (sold_above_floor - bought) - town_drain(t)

If that closes on a freshly simulated episode, the instrumentation is attributing
every executed unit to the right player and the drain model matches `_town_consume`.
A silent instrumentation failure (which is the realistic failure mode -- the engine
deep-copies state between steps, so a stale object-identity map yields zero commits)
shows up here as a large residual, not as an exception.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _engine import ENV_NAME, import_engine  # noqa: E402
from extract_s2_labels import (  # noqa: E402
    NO_FLOOR,
    _count_multiples,
    _kg,
    price_at_cap,
    extract_episode,
    floor_inventory,
    town_drain,
    validation_report,
    PRODUCT_COLUMNS,
    SCALAR_COLUMNS,
)


@pytest.fixture(scope="module")
def synthetic_replay(tmp_path_factory):
    """A short, real episode: 12 days of the built-in starter agent against itself."""
    ke = import_engine()
    env = ke.make(ENV_NAME, configuration={"episodeSteps": 288}, info={"seed": 7}, debug=False)
    env.run(["starter", "starter"])
    path = tmp_path_factory.mktemp("s2") / "synthetic.json"
    path.write_text(json.dumps(env.toJSON()), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def extracted(synthetic_replay):
    res = extract_episode(synthetic_replay, episode_id=1)
    assert res.get("ok"), res.get("error")
    return res


def test_count_multiples():
    # inclusive on both ends, and the lo == 0 edge is the one that bites
    assert _count_multiples(0, 0, 4) == 1
    assert _count_multiples(0, 7, 4) == 2      # 0, 4
    assert _count_multiples(1, 7, 4) == 1      # 4
    assert _count_multiples(4, 8, 4) == 2      # 4, 8
    assert _count_multiples(5, 3, 4) == 0
    assert _count_multiples(0, 719, 24) == 30


# Under 1.32.7's shipped MARKET_PARAMS these two curves are still well above $1 at the
# reachability cap (100,000 units, ~10x the starting pool), so no play can crash them to
# the floor and their headroom is unbounded. Values are the quote AT the cap.
UNFLOORABLE = {"WHEAT": 15, "EGG": 30}

# The seven pools that do floor, and where (measured, 1.32.7 defaults). These are the
# numbers Team B's thin-pool marks rest on: 59-842 units above the 10,000 start.
FLOORABLE = {
    "CARROT": 10_842, "TOMATO": 10_529, "STRAWBERRY": 10_062, "MELON": 10_158,
    "MILK": 10_076, "WOOL": 10_059, "FERTILIZER": 10_493,
}


def test_floor_inventory_is_the_first_dollar_quote():
    kg = _kg()
    assert set(FLOORABLE) | set(UNFLOORABLE) == set(kg.PRODUCTS)
    for product, expected in FLOORABLE.items():
        inv_floor = floor_inventory(kg, product, None)
        assert inv_floor == expected, product
        assert kg.market_price(product, inv_floor, None) == kg.PRICE_FLOOR
        assert kg.market_price(product, inv_floor - 1, None) > kg.PRICE_FLOOR


def test_wheat_and_egg_pools_cannot_be_crashed_to_the_floor():
    """Regression guard for the sentinel, and a live check on the engine's economy.

    If a rebalance ever gives WHEAT or EGG a reachable $1 floor -- or takes one away
    from another product -- this fails, which is exactly the signal the balance-sheet
    program wants: pool headroom marks change meaning across an engine bump.
    """
    kg = _kg()
    for product, quote_at_cap in UNFLOORABLE.items():
        assert floor_inventory(kg, product, None) == NO_FLOOR, product
        assert price_at_cap(kg, product, None) == quote_at_cap, product
    for product in FLOORABLE:
        assert price_at_cap(kg, product, None) == kg.PRICE_FLOOR, product


def test_price_curve_is_non_increasing():
    """`headroom_static = inv_floor - inv` is only meaningful if price is monotone."""
    kg = _kg()
    for product in kg.PRODUCTS:
        inv_floor = floor_inventory(kg, product, None)
        top = 20_000 if inv_floor == NO_FLOOR else inv_floor + 2
        prices = [kg.market_price(product, i, None) for i in range(0, top)]
        assert all(b <= a for a, b in zip(prices, prices[1:])), product


def test_town_drain_matches_the_engine():
    kg = _kg()
    cfg = {"townShopSellInterval": 4, "townCenterSellInterval": 24}
    products = list(kg.PRODUCTS)

    # off-tick: nothing moves
    assert set(town_drain(kg, cfg, ["BAKERY"], 5, products).values()) == {0}
    # shop tick: BAKERY is a two-product shop, so 1x each
    d = town_drain(kg, cfg, ["BAKERY"], 4, products)
    assert d["EGG"] == 1 and d["WHEAT"] == 1 and d["MILK"] == 0
    # single-product shops consume at 2x, and duplicates stack (drawn with replacement)
    d = town_drain(kg, cfg, ["YARN_STORE", "YARN_STORE"], 4, products)
    assert d["WOOL"] == 4
    # step 0 is both a shop tick and a town-centre tick; FERTILIZER never drains
    d = town_drain(kg, cfg, ["YARN_STORE"], 0, products)
    assert d["WOOL"] == 3 and d["FERTILIZER"] == 0 and d["MELON"] == 1


def test_identity_closes_exactly(extracted):
    resid = extracted["cols"]["identity_residual"]
    assert resid.shape[0] == extracted["rows"]
    assert int(np.abs(resid).max()) == 0, "market inventory identity did not close"


def test_instrumentation_actually_saw_trades(extracted):
    """Guards the silent-zero failure mode: a stale farm id map records nothing."""
    cols = extracted["cols"]
    assert int(cols["p0_sold"].sum() + cols["p1_sold"].sum()) > 0


def test_replay_parity_on_the_market_slice(extracted):
    assert extracted["market_mismatched_steps"] == 0
    assert extracted["rewards_match"]


def test_visible_and_true_flow_differ_only_by_floored_sales(extracted):
    cols = extracted["cols"]
    for pl in (0, 1):
        delta = cols[f"p{pl}_net_true"] - cols[f"p{pl}_net_visible"]
        assert np.array_equal(delta, cols[f"p{pl}_sold_floored"])


def test_headroom_is_consistent(extracted):
    cols = extracted["cols"]
    bounded = cols["inv_floor"] != NO_FLOOR
    # unbounded pools carry the sentinel everywhere, never a fabricated number
    assert (cols["headroom_static"][~bounded] == NO_FLOOR).all()
    assert (cols["headroom_actual"][~bounded] == NO_FLOOR).all()
    assert (cols["headroom_known"][~bounded] == NO_FLOOR).all()
    assert (cols["headroom_static"][bounded] >= 0).all()
    # on a floorable pool: quoting above $1 means room left, quoting $1 means none
    at_floor = (cols["price_pre"] <= 1) & bounded
    above = (cols["price_pre"] > 1) & bounded
    assert (cols["headroom_static"][at_floor] == 0).all()
    assert (cols["headroom_static"][above] > 0).all()
    assert np.array_equal(cols["headroom_actual"][bounded],
                          (cols["headroom_static"] + cols["drain_remaining_actual"])[bounded])
    # remaining drain is a suffix sum, so it can only decrease as the episode runs
    d = cols["drain_remaining_actual"]
    assert (np.diff(d, axis=0) <= 0).all()
    assert (cols["drain_remaining_actual"] >= 0).all()


def test_drain_remaining_known_matches_actual_once_shops_stop_unlocking(extracted):
    """The two variants must agree on the final day: no draws are left to differ over."""
    cols = extracted["cols"]
    assert np.array_equal(cols["drain_remaining_known"][-1], cols["drain_remaining_actual"][-1])
    # the runtime-knowable variant freezes the shop set, so it can never over-count
    assert (cols["drain_remaining_known"] <= cols["drain_remaining_actual"]).all()


def test_future_flow_is_a_suffix_sum(extracted):
    cols = extracted["cols"]
    for pl in (0, 1):
        fut = cols[f"p{pl}_future_net_true"]
        cur = cols[f"p{pl}_net_true"]
        assert np.array_equal(fut[-1], cur[-1])
        assert np.array_equal(fut[:-1], cur[:-1] + fut[1:])


def test_no_engine_globals_left_patched(extracted):
    """The extractor monkeypatches two engine functions; it must put them back."""
    kg = _kg()
    assert kg._commit_unit.__module__ == kg.__name__
    assert kg._process_market.__module__ == kg.__name__


def test_all_declared_columns_are_present(extracted):
    for c in PRODUCT_COLUMNS:
        assert c in extracted["cols"], c
        assert extracted["cols"][c].shape[0] == extracted["rows"]
    for c in SCALAR_COLUMNS:
        assert c in extracted["scal"], c


def test_validation_report_runs(extracted):
    kg = _kg()
    out = {c: extracted["cols"][c] for c in PRODUCT_COLUMNS}
    out.update({c: extracted["scal"][c] for c in SCALAR_COLUMNS})
    out["products"] = np.array(kg.PRODUCTS, dtype="<U12")
    report = validation_report(out, [{"episode_id": 1}])
    assert report["identity"]["max_abs_residual"] == 0
    assert report["rows"] == extracted["rows"]
    assert set(report["per_product"]) == set(kg.PRODUCTS)
