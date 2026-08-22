"""expB-304c — the census's unit-conservation identity, and the leak it exposes.

`DROP` and `_drop_inventories_to_shed` both compute `room = max(0, capacity - sum(shed))`,
`take = min(n, room)` and then `del inv[item]` UNCONDITIONALLY: whatever did not fit is
destroyed, with no refusal, no market event and no money delta. Nothing on the revenue side
of the census can see it, so before it was measured the identity

    produced + bought - sold - consumed - terminal_stock - destroyed == 0

did not close, and the residual was silently spread over every line's realised revenue --
the numbers expB-304b sorts on (ruling-03 section 3.4 reason 4).

These tests pin the engine behaviour the instrumentation depends on, so a version bump that
changes it fails here rather than in a quiet mis-measurement.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

from pnl_census import unit_conservation  # noqa: E402


def test_end_of_day_shed_overflow_is_destroyed_not_refused():
    """The premise of the whole experiment, asserted against the engine itself."""
    private = {"shed": {"WHEAT": 98}, "inventories": [{"MELON": 10}], "seeds": {}}
    K._drop_inventories_to_shed(private, 100)
    assert private["shed"]["MELON"] == 2          # only the room was taken
    assert private["inventories"][0] == {}        # and the rest is GONE, not left behind
    assert sum(private["shed"].values()) == 100


def test_drop_action_overflow_is_destroyed_too():
    private = {"shed": {"WHEAT": 100}, "inventories": [{"MELON": 5}], "seeds": {}}
    before = sum(private["shed"].values()) + sum(private["inventories"][0].values())
    K._drop_inventories_to_shed(private, 100)
    after = sum(private["shed"].values()) + sum(private["inventories"][0].values())
    assert before - after == 5                     # a full shed evaporates the whole drop


def test_identity_closes_when_destroyed_is_accounted_and_not_before():
    """A hand-built player state: 10 melons made, 3 sold, 2 left, 5 evaporated."""
    st = {
        "harvest_units": {p: 0 for p in K.PRODUCTS},
        "sales": {p: {"units": 0, "dollars": 0.0} for p in K.PRODUCTS},
        "buy_product": {"WHEAT": {"units": 0}, "FERTILIZER": {"units": 0}},
        "buy_animal": {a: {"units": 0} for a in K.ANIMALS},
        "fert_collected": 0,
        "fertilize_on": {c: 0 for c in K.CROPS},
        "feed_events": {a: 0 for a in K.ANIMALS},
        "placed_animals": {a: 0 for a in K.ANIMALS},
        "destroyed": {"MELON": 5},
        "destroyed_by_path": {"drop": 0, "end_of_day": 5},
        "shed_days": [10, 20], "shed_full_turns": 0,
        "terminal": {"shed": {"MELON": 2}, "carried": {}},
    }
    st["harvest_units"]["MELON"] = 10
    st["sales"]["MELON"]["units"] = 3
    c = unit_conservation(st)
    assert c["per_item"]["MELON"]["residual"] == 0
    assert c["destroyed_units"] == 5
    assert c["residual_abs"] == 0
    assert c["items_closed"] == c["items"]

    st["destroyed"] = {}                            # the pre-expB-304c world
    st["destroyed_by_path"] = {"drop": 0, "end_of_day": 0}
    c2 = unit_conservation(st)
    assert c2["per_item"]["MELON"]["residual"] == 5  # 5 units of phantom production


def test_animals_and_fertilizer_are_consumption_terms():
    st = {
        "harvest_units": {p: 0 for p in K.PRODUCTS},
        "sales": {p: {"units": 0, "dollars": 0.0} for p in K.PRODUCTS},
        "buy_product": {"WHEAT": {"units": 0}, "FERTILIZER": {"units": 4}},
        "buy_animal": {a: {"units": 0} for a in K.ANIMALS},
        "fert_collected": 6,
        "fertilize_on": {c: 0 for c in K.CROPS},
        "feed_events": {a: 0 for a in K.ANIMALS},
        "placed_animals": {a: 0 for a in K.ANIMALS},
        "destroyed": {},
        "destroyed_by_path": {"drop": 0, "end_of_day": 0},
        "shed_days": [], "shed_full_turns": 0,
        "terminal": {"shed": {}, "carried": {}},
    }
    st["buy_animal"]["GOOSE"]["units"] = 3
    st["placed_animals"]["GOOSE"] = 3               # bought 3, placed 3, none in the shed
    st["fertilize_on"]["WHEAT"] = 10                # 6 collected + 4 bought, all applied
    c = unit_conservation(st)
    assert c["per_item"]["GOOSE"]["residual"] == 0
    assert c["per_item"]["FERTILIZER"]["residual"] == 0
