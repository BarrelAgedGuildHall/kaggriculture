"""ruling-03 section 1.5 — invariants of the weed classifier and of R-15's DIG.

The classification is the evidence for a pre-registered falsification test, so the
arithmetic it rests on is pinned against the engine's own refresh loop rather than
against a restatement of it:

  1. `plant_events_realised` reproduces, for every ongoing crop and every age, the
     number of production events a simulated `_daily_refresh_plants` actually fires;
  2. a non-ongoing crop still standing on the board has realised nothing, whatever
     its age (its yield is a WATER bonus that only a HARVEST -- which removes the
     tile -- can realise);
  3. the two independent derivations of RETIREMENT agree: an ongoing tile killed by
     lifespan decay must have reached `max_yield`, because `_new_plant` leaves
     `max_lifespan_step` at -1 and only the terminal event ever sets it;
  4. `dig_terminal` fires on a spent ongoing tile and never on a live one, and the
     flag defaults OFF so the control lineage is untouched.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agents" / "team_b"))
sys.path.insert(0, str(REPO / "agents" / "team_b" / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

from feed_audit import plant_events_realised  # noqa: E402
from millrace.feedchain import FEEDCHAIN_DEFAULTS  # noqa: E402

TURNS_PER_DAY = 24


def _simulate_events(crop, planted_day, kill_day):
    """Count production events by running the engine's own refresh on one tile.

    The tile is watered every day so it survives to `kill_day`; the kill itself is
    modelled the way the engine does it -- `_daily_refresh_plants` weeds the tile
    BEFORE the production block on the day it dies -- so the refresh is run for
    days `planted_day .. kill_day - 1`.
    """
    farm = {"tiles": [[K._new_plant(crop, planted_day, TURNS_PER_DAY)]]}
    events = 0
    for day in range(planted_day, kill_day):
        tile = farm["tiles"][0][0]
        before = tile["yield_units"]
        tile["watered_today"] = True
        K._daily_refresh_plants(farm, day, TURNS_PER_DAY)
        tile = farm["tiles"][0][0]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            break
        if tile["yield_units"] > before:
            events += 1
        # empty the tile so the max_yield CAP never hides a later event
        tile["yield_units"] = 0
    return events


def test_events_realised_matches_the_engine_for_every_ongoing_crop_and_age():
    for crop, cd in K.CROPS.items():
        if not cd["ongoing"]:
            continue
        for kill_day in range(0, 30):
            got, max_yield, ongoing, _ = plant_events_realised(K, crop, kill_day)
            want = min(cd["max_yield"], _simulate_events(crop, 0, kill_day))
            assert ongoing is True
            assert max_yield == cd["max_yield"]
            assert got == want, (crop, kill_day, got, want)


def test_a_standing_non_ongoing_crop_has_realised_nothing():
    for crop, cd in K.CROPS.items():
        if cd["ongoing"]:
            continue
        for age in (0, 5, 12, 25):
            events, max_yield, ongoing, last = plant_events_realised(K, crop, age)
            assert (events, ongoing, last) == (0, False, None)
            assert max_yield == cd["max_yield"]


def test_lifespan_decay_of_an_ongoing_tile_implies_max_yield_was_reached():
    """The instrument's cross-check, stated as an engine property.

    `_new_plant` gives an ongoing crop `max_lifespan_step = -1` and `_decay_plants`
    skips anything negative, so an ongoing tile can only reach the decay path after
    line 802 has set the step -- which happens only at `production_count == max_yield`.
    """
    for crop, cd in K.CROPS.items():
        if not cd["ongoing"]:
            continue
        assert K._new_plant(crop, 0, TURNS_PER_DAY)["max_lifespan_step"] == -1
        farm = {"tiles": [[K._new_plant(crop, 0, TURNS_PER_DAY)]]}
        terminal_day = None
        for day in range(0, 40):
            tile = farm["tiles"][0][0]
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                break
            tile["watered_today"] = True
            tile["yield_units"] = 0
            K._daily_refresh_plants(farm, day, TURNS_PER_DAY)
            tile = farm["tiles"][0][0]
            if isinstance(tile, dict) and tile.get("max_lifespan_step", -1) >= 0:
                terminal_day = day
                break
        assert terminal_day is not None, crop
        events, max_yield, _, _ = plant_events_realised(K, crop, terminal_day + 1)
        assert events == max_yield, (crop, terminal_day, events, max_yield)


def test_dig_terminal_defaults_off_and_requires_dig():
    assert FEEDCHAIN_DEFAULTS["dig_terminal"] is False
    assert FEEDCHAIN_DEFAULTS["dig"] is False


def test_spent_ongoing_tile_is_recognised_and_a_live_one_is_not():
    """The exact predicate `feedchain` uses to offer a terminal DIG."""
    crop = next(c for c, cd in K.CROPS.items() if cd["ongoing"])
    live = K._new_plant(crop, 0, TURNS_PER_DAY)
    assert not (live.get("max_lifespan_step", -1) >= 0 and live.get("yield_units", 0) <= 0)

    farm = {"tiles": [[K._new_plant(crop, 0, TURNS_PER_DAY)]]}
    for day in range(0, 40):
        tile = farm["tiles"][0][0]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            break
        tile["watered_today"] = True
        K._daily_refresh_plants(farm, day, TURNS_PER_DAY)
        tile = farm["tiles"][0][0]
        if isinstance(tile, dict) and tile.get("max_lifespan_step", -1) >= 0:
            # still holding its final harvest: NOT diggable yet
            assert tile["yield_units"] > 0
            assert not (tile["max_lifespan_step"] >= 0 and tile["yield_units"] <= 0)
            tile["yield_units"] = 0        # the terminal HARVEST lands
            assert tile["max_lifespan_step"] >= 0 and tile["yield_units"] <= 0
            return
    raise AssertionError("no terminal event reached for " + crop)
