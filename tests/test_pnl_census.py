"""Tests for the shared P&L census (tools/pnl_census.py).

The allocation arithmetic is tested on a synthetic census (no engine, instant), the instrumentation
itself on one short real episode. Ruling: `2026-08-21-team-b-business-lens/ruling-02-m4-and-sprint2-order.md`
§3.1 (labelled gross vs contribution columns + λ_opp reference) and §3.3 (fertilizer is a joint
product of an animal-day, not a line).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import pnl_census as pc  # noqa: E402

ANIMALS = ("GOOSE", "COW", "SHEEP")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = ("EGG", "MILK", "WOOL", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "FERTILIZER")


def _census(**over):
    """A minimal but complete raw census for one player, shaped exactly like run_pnl's output."""
    st = {
        "sales": {p: {"units": 0, "dollars": 0.0} for p in PRODUCTS},
        "buy_product": {p: {"units": 0, "dollars": 0.0} for p in ("WHEAT", "FERTILIZER")},
        "buy_product_prices": {"WHEAT": [], "FERTILIZER": []},
        "buy_seed": {c: {"units": 0, "dollars": 0.0} for c in CROPS},
        "buy_animal": {a: {"units": 0, "dollars": 0.0} for a in ANIMALS},
        "hire": {"count": 0, "dollars": 0.0},
        "land": {"count": 0, "dollars": 0.0},
        "refused": {"buy_animal_shed_full": 0, "buy_product_shed_full": 0, "cash": 0},
        "harvest_units": {p: 0 for p in PRODUCTS},
        "feed_events": {a: 0 for a in ANIMALS},
        "fert_collected": 0,
        "fert_collected_by_line": {a: 0 for a in ANIMALS},
        "fert_collected_unattributed": 0,
        "fertilize_on": {c: 0 for c in CROPS},
        "acts": {b: {"executed": 0, "noop": 0} for b in
                 ("produce", "maintain", "invest", "logistics", "idle", "unknown")},
        "acts_by_op": {},
        "unit_turns": 0,
        "lines": {ln: pc._blank_line() for ln in pc.LINE_PRODUCT},
        "tile_days_other": {"EMPTY_STRUCT": 0, "WEED": 0, "FALLOW": 0, "LOCKED": 0},
        "hands_by_day": [2, 2],
        # expB-304c: the shed-overflow accounting the conservation identity needs
        "destroyed": {},
        "destroyed_by_path": {"drop": 0, "end_of_day": 0},
        "shed_days": [],
        "shed_full_turns": 0,
        "placed_animals": {a: 0 for a in ANIMALS},
        "terminal": {"cash": 1000.0, "shed": {}, "shed_value": 0, "carried": {}, "carried_value": 0,
                     "unharvested": {}, "unharvested_value": 0, "seeds_left": {},
                     "seeds_left_cost": 0, "capital_written_off": 0, "animals_alive": 0},
    }
    st.update(over)
    return st


def _herd_census():
    """A goose-heavy herd: 300 fertilizer units collected (goose 210 / cow 60 / sheep 30) and sold
    for $12,000, animal products sold beside them, wheat bought and fed."""
    st = _census()
    st["sales"]["FERTILIZER"] = {"units": 300, "dollars": 12000.0}
    st["sales"]["EGG"] = {"units": 100, "dollars": 3000.0}
    st["sales"]["MILK"] = {"units": 50, "dollars": 4000.0}
    st["sales"]["WOOL"] = {"units": 20, "dollars": 2000.0}
    st["fert_collected"] = 300
    st["fert_collected_by_line"] = {"GOOSE": 210, "COW": 60, "SHEEP": 30}
    st["buy_product"]["WHEAT"] = {"units": 100, "dollars": 4000.0}
    st["feed_events"] = {"GOOSE": 60, "COW": 30, "SHEEP": 10}
    st["harvest_units"].update({"EGG": 100, "MILK": 50, "WOOL": 20})
    # productive actions: collection lands on the emitting animal line (that is the fix)
    st["lines"]["GOOSE"]["acts"] = {"produce": 210 + 40, "maintain": 60, "invest": 0}
    st["lines"]["COW"]["acts"] = {"produce": 60 + 20, "maintain": 30, "invest": 0}
    st["lines"]["SHEEP"]["acts"] = {"produce": 30 + 10, "maintain": 10, "invest": 0}
    st["lines"]["GOOSE"]["tile_days"] = 200
    st["lines"]["COW"]["tile_days"] = 60
    st["lines"]["SHEEP"]["tile_days"] = 40
    st["acts"]["produce"]["executed"] = 370
    st["acts"]["maintain"]["executed"] = 100
    st["unit_turns"] = 1000
    return st


# ------------------------------------------------------------------ §3.3 allocation

def test_fertilizer_revenue_is_allocated_to_the_emitting_animal_lines():
    alloc = pc.allocate_fertilizer(_herd_census())
    assert alloc["units_collected"] == 300
    assert alloc["collected_by_line"] == {"GOOSE": 210, "COW": 60, "SHEEP": 30}
    assert alloc["joint_share_of_supply"] == 1.0
    assert alloc["allocated_by_line"]["GOOSE"] == pytest.approx(12000 * 210 / 300)
    assert alloc["allocated_by_line"]["COW"] == pytest.approx(12000 * 60 / 300)
    assert alloc["allocated_by_line"]["SHEEP"] == pytest.approx(12000 * 30 / 300)
    assert alloc["unallocated_revenue"] == pytest.approx(0.0)


def test_fertilizer_line_becomes_a_memo_line_and_nothing_is_double_counted():
    stmt = pc.income_statement(_herd_census())
    memo = stmt["lines"]["FERTILIZER"]
    assert memo["joint_product"] is True
    assert memo["memo_gross_revenue"] == 12000.0
    assert memo["gross_revenue"] == 0.0, "allocated away: the memo line keeps no revenue of its own"
    assert memo["productive_actions"] == 0, "collection actions moved to the animal lines"
    # the allocation redistributes revenue, it never creates or destroys it
    assert sum(v["gross_revenue"] for v in stmt["lines"].values()) == pytest.approx(stmt["gross_revenue"])


def test_goose_is_understated_without_the_allocation():
    """The §3.3 finding: crediting fertilizer to a standalone line understates the animal lines,
    goose most (it emits the most units per dollar of capital)."""
    stmt = pc.income_statement(_herd_census())
    goose = stmt["lines"]["GOOSE"]
    assert goose["sales_revenue"] == 3000.0
    assert goose["fertilizer_revenue_allocated"] == pytest.approx(8400.0)
    assert goose["gross_revenue"] == pytest.approx(11400.0)
    # feed is charged pro-rata by FEED event: goose took 60 of 100 events on a $4,000 bill
    assert goose["direct_cost"] == pytest.approx(2400.0)
    assert goose["contribution"] == pytest.approx(9000.0)
    # ... and the collection actions are netted through the denominator, not assumed away
    assert goose["productive_actions"] == 250 + 60
    assert goose["contribution_per_action"] == pytest.approx(9000.0 / 310, rel=1e-3)


def test_bought_fertilizer_is_not_credited_to_the_herd():
    st = _herd_census()
    st["buy_product"]["FERTILIZER"] = {"units": 100, "dollars": 500.0}
    st["sales"]["FERTILIZER"] = {"units": 400, "dollars": 16000.0}
    alloc = pc.allocate_fertilizer(st)
    assert alloc["joint_share_of_supply"] == pytest.approx(300 / 400)
    assert sum(alloc["allocated_by_line"].values()) == pytest.approx(12000.0)
    assert alloc["unallocated_revenue"] == pytest.approx(4000.0), "resale of purchased stock stays on the memo line"
    stmt = pc.income_statement(st)
    assert stmt["lines"]["FERTILIZER"]["gross_revenue"] == pytest.approx(4000.0)
    assert sum(v["gross_revenue"] for v in stmt["lines"].values()) == pytest.approx(stmt["gross_revenue"])


def test_no_fertilizer_collected_leaves_every_animal_line_untouched():
    st = _census()
    st["sales"]["EGG"] = {"units": 10, "dollars": 300.0}
    stmt = pc.income_statement(st)
    assert stmt["fertilizer_allocation"]["units_collected"] == 0
    assert stmt["lines"]["GOOSE"]["fertilizer_revenue_allocated"] == 0.0
    assert stmt["lines"]["FERTILIZER"]["gross_revenue"] == 0.0


def test_purchased_fertilizer_is_charged_to_the_crops_that_used_it():
    st = _census()
    st["buy_product"]["FERTILIZER"] = {"units": 20, "dollars": 600.0}
    st["fertilize_on"]["MELON"] = 15
    st["fertilize_on"]["WHEAT"] = 5
    st["sales"]["MELON"] = {"units": 10, "dollars": 2000.0}
    st["lines"]["MELON"]["acts"] = {"produce": 10, "maintain": 15, "invest": 2}
    stmt = pc.income_statement(st)
    assert stmt["lines"]["MELON"]["fert_charged_bill"] == pytest.approx(450.0)
    assert stmt["lines"]["WHEAT"]["fert_charged_bill"] == pytest.approx(150.0)
    assert stmt["unallocated_input_spend"] == 0.0
    # ... and an agent that buys fertilizer but issues no FERTILIZE says so out loud
    st["fertilize_on"] = {c: 0 for c in CROPS}
    assert pc.income_statement(st)["unallocated_input_spend"] == 600.0


# ------------------------------------------------------------------ §3.1 columns / lambda

def test_gross_and_contribution_are_separately_labelled_on_every_line():
    stmt = pc.income_statement(_herd_census())
    for ln, v in stmt["lines"].items():
        for k in ("sales_revenue", "fertilizer_revenue_allocated", "gross_revenue", "direct_cost",
                  "contribution", "gross_per_action", "contribution_per_action",
                  "gross_per_tile_day", "contribution_per_tile_day"):
            assert k in v, f"{ln} is missing the {k} column"
    goose = stmt["lines"]["GOOSE"]
    assert goose["gross_per_action"] > goose["contribution_per_action"], \
        "the two denominators must not be the same number (§3.1's whole point)"


def test_lambda_opp_reference_is_on_the_contribution_denominator():
    stmt = pc.income_statement(_herd_census())
    lam = stmt["lambda_opp"]
    prod = stmt["productive_actions"]
    assert lam["reference"] == pytest.approx(stmt["total_contribution"] / prod, rel=1e-3)
    assert lam["gross_per_action"] == pytest.approx(stmt["gross_revenue"] / prod, rel=1e-3)
    assert lam["reference"] < lam["gross_per_action"], "contribution/action is below gross/action"
    assert stmt["contribution_per_productive_action"] == lam["reference"]


def test_lambda_ladder_is_sorted_and_names_the_marginal_line_at_the_cut():
    stmt = pc.income_statement(_herd_census())
    ladder = stmt["lambda_opp"]["ladder"]
    vals = [r["contribution_per_action"] for r in ladder]
    assert vals == sorted(vals, reverse=True)
    assert ladder[-1]["cum_action_share"] == pytest.approx(1.0, abs=0.02)
    assert stmt["lambda_opp"]["marginal_line"] == ladder[-1]["line"]
    assert stmt["lambda_opp"]["marginal_contribution_per_action"] == vals[-1]


def test_a_rounding_error_line_is_not_the_marginal_line():
    st = _herd_census()
    st["sales"]["CARROT"] = {"units": 1, "dollars": 1.0}
    st["lines"]["CARROT"]["acts"] = {"produce": 1, "maintain": 0, "invest": 0}   # 0.3% of the budget
    stmt = pc.income_statement(st)
    assert stmt["lambda_opp"]["marginal_line"] != "CARROT"


# ------------------------------------------------------------------ aggregation

def test_aggregate_reports_the_allocation_and_both_denominators():
    stmt = pc.income_statement(_herd_census())
    rows = [{"seed": s, "agents": ["a", "b"], "engine": "1.32.7", "final_money": [1, 2],
             "statuses": ["DONE", "DONE"], "statements": [stmt, stmt]} for s in (0, 1)]
    rep = pc.aggregate(rows)
    top = rep["per_agent"]["a"]
    assert top["gross_per_productive_action"]["mean"] > top["contribution_per_productive_action"]["mean"]
    assert top["fertilizer_allocation"]["allocated_by_line"]["GOOSE"]["mean"] == pytest.approx(8400.0)
    assert top["lambda_opp_reference"]["mean"] == pytest.approx(stmt["lambda_opp"]["reference"])
    assert top["lines"]["GOOSE"]["gross_revenue"]["mean"] == pytest.approx(11400.0)
    pc.print_report(rep)   # the printer must survive a real report (it reads many optional keys)


# ------------------------------------------------------------------ one real episode

@pytest.mark.parametrize("steps", [96])
def test_short_episode_is_internally_consistent(steps):
    """Instrumentation smoke: 4 days of starter-vs-starter. Asserts the identities that must hold
    for any episode, whatever the agents did."""
    raw = pc.run_pnl("starter", "starter", seed=0, steps=steps)
    assert raw["engine"] == "1.32.7"
    for player in raw["players"]:
        collected = player["fert_collected_by_line"]
        assert sum(collected.values()) + player["fert_collected_unattributed"] == player["fert_collected"]
        assert player["lines"]["FERTILIZER"]["acts"] == {"produce": 0, "maintain": 0, "invest": 0}, \
            "no action may be attributed to the joint-product line"
        stmt = pc.income_statement(player)
        assert sum(v["gross_revenue"] for v in stmt["lines"].values()) == pytest.approx(stmt["gross_revenue"])
        assert stmt["lines"]["FERTILIZER"]["joint_product"] is True
        for a in ANIMALS:
            assert stmt["lines"][a]["productive_actions"] >= collected[a]
