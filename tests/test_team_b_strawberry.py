"""expB-303 — lifecycle tests for `ongoing` crops (STRAWBERRY).

Written BEFORE the strawberry acreage A/B, per the expB-101 lesson: a chassis
validity bug inverts A/B signs, and Millrace's crop code has only ever handled
non-ongoing crops (WHEAT / CARROT / MELON). Every assertion below is checked
against the real `kaggle_environments` 1.32.7 engine functions, not against our
model of them.

The engine facts these pin (CROPS["STRAWBERRY"] = seed 100, first_yield_day 10,
max_yield_day 10, interval 2, max_yield 4, ongoing True):

  * `_new_plant` starts an ongoing crop at yield_units 0 (not 1) and with
    max_lifespan_step -1 — it does not decay until its last production event.
  * production resolves in `_daily_refresh_plants` at the END of the day whose
    age is 9, 11, 13, 15 (the refresh uses `next_day`), adding 1 unit — or 2
    when the tile was watered that day AND `fertilized_until_day >= day`.
  * `yield_units` is capped at max_yield 4, so the tile must be harvested
    between events or production is silently thrown away.
  * one FERTILIZE covers day..day+2, so an application on an age-9 day also
    covers the age-11 event: two applications buy all four events.
  * HARVEST does not clear an ongoing tile, and refuses before first_yield_day.
  * two consecutive unwatered days weed the tile out at ANY age, so watering
    must continue past `max_yield_day` — the exact bug the old
    `age < cd["maxday"]` water gate in `plant_tasks` would have caused.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _engine import import_engine  # noqa: E402

import_engine()
import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: E402

from millrace import probe_core as pc  # noqa: E402
from millrace.service import plant_tasks  # noqa: E402

TPD = 24


def _farm(tile=None):
    return {"tiles": [[tile if (x, y) == (0, 0) else None for x in range(3)]
                      for y in range(3)],
            "farmer": [0, 0], "hands": [], "money": 0.0, "hires_today": 0}


def _private():
    return {"shed": {}, "seeds": {"STRAWBERRY": 5}, "inventories": [{}]}


def _run_season(fertilize_days=(), water=True, harvest=True, plant_day=0,
                until_day=18):
    """Drive the real engine day by day for a single strawberry tile.

    Returns (units_harvested, final_tile)."""
    farm = _farm()
    priv = _private()
    K._apply_unit_action(farm, priv, 0, ["PLANT", "STRAWBERRY"], 3, plant_day, TPD)
    harvested = 0
    for day in range(plant_day, plant_day + until_day + 1):
        tile = farm["tiles"][0][0]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            break
        # the planting day counts as unwatered (`consecutive_unwatered` starts
        # at 1), so a tile that is not watered on the day it is planted weeds
        # out that same night
        if water:
            K._apply_unit_action(farm, priv, 0, ["WATER"], 3, day, TPD)
        if day - plant_day in fertilize_days:
            priv["inventories"][0]["FERTILIZER"] = 1
            K._apply_unit_action(farm, priv, 0, ["FERTILIZE"], 3, day, TPD)
        if harvest and farm["tiles"][0][0].get("yield_units", 0) > 0:
            before = priv["inventories"][0].get("STRAWBERRY", 0)
            K._apply_unit_action(farm, priv, 0, ["HARVEST"], 3, day, TPD)
            harvested += priv["inventories"][0].get("STRAWBERRY", 0) - before
        K._daily_refresh_plants(farm, day, TPD)
    return harvested, farm["tiles"][0][0]


# ------------------------------------------------------------------ engine
def test_engine_ongoing_starts_at_zero_yield_and_never_decays_early():
    t = K._new_plant("STRAWBERRY", 0, TPD)
    assert t["yield_units"] == 0          # non-ongoing crops start at 1
    assert t["max_lifespan_step"] == -1   # set only after the last event


def test_engine_unfertilized_strawberry_yields_four_units():
    units, tile = _run_season()
    assert units == 4, f"expected 4 units unfertilized, got {units}"
    assert isinstance(tile, dict) and tile.get("kind") == "PLANT"


def test_engine_two_fertilize_applications_double_the_tile_to_eight():
    # ages 9 and 13: each covers day..day+2, i.e. the 9/11 and the 13/15 events
    units, _ = _run_season(fertilize_days=(9, 13))
    assert units == 8, f"expected 8 units with 2 applications, got {units}"


def test_engine_yield_caps_at_four_so_unharvested_production_is_lost():
    units, tile = _run_season(fertilize_days=(9, 13), harvest=False)
    assert units == 0
    assert tile["yield_units"] == 4, "cap is max_yield, surplus events are dropped"


def test_engine_two_unwatered_days_weed_the_tile_out_at_any_age():
    _, tile = _run_season(water=False, until_day=3)
    assert tile == {"kind": "WEED"}


def test_engine_watering_must_continue_past_max_yield_day():
    """The old `age < cd["maxday"]` water gate stops at age 10 and kills the
    tile before three of its four production events."""
    farm = _farm()
    priv = _private()
    K._apply_unit_action(farm, priv, 0, ["PLANT", "STRAWBERRY"], 3, 0, TPD)
    for day in range(0, 12):
        if day and day < K.CROPS["STRAWBERRY"]["max_yield_day"]:  # the buggy gate
            K._apply_unit_action(farm, priv, 0, ["WATER"], 3, day, TPD)
        K._daily_refresh_plants(farm, day, TPD)
    assert farm["tiles"][0][0] == {"kind": "WEED"}


def test_engine_harvest_refuses_before_first_yield_day():
    farm = _farm()
    priv = _private()
    K._apply_unit_action(farm, priv, 0, ["PLANT", "STRAWBERRY"], 3, 0, TPD)
    farm["tiles"][0][0]["yield_units"] = 2  # would-be-illegal state
    K._apply_unit_action(farm, priv, 0, ["HARVEST"], 3, 5, TPD)
    assert priv["inventories"][0].get("STRAWBERRY", 0) == 0


def test_engine_harvest_leaves_an_ongoing_tile_in_place():
    farm = _farm()
    priv = _private()
    K._apply_unit_action(farm, priv, 0, ["PLANT", "STRAWBERRY"], 3, 0, TPD)
    farm["tiles"][0][0]["yield_units"] = 2
    K._apply_unit_action(farm, priv, 0, ["HARVEST"], 3, 10, TPD)
    assert priv["inventories"][0]["STRAWBERRY"] == 2
    assert farm["tiles"][0][0]["kind"] == "PLANT"  # non-ongoing would be None


# --------------------------------------------------- our crop table + tasks
def test_crop_table_matches_the_engine_for_every_crop_we_grow():
    for crop, ours in pc.CROPS.items():
        eng = K.CROPS[crop]
        assert ours["seed"] == eng["seed"], crop
        assert ours["first"] == eng["first_yield_day"], crop
        assert ours["maxday"] == eng["max_yield_day"], crop
        assert bool(ours.get("ongoing")) == eng["ongoing"], crop
        if eng["ongoing"]:
            assert ours["interval"] == eng["interval"], crop
            assert ours["max_yield"] == eng["max_yield"], crop


def _tile(age, yield_units=0, watered=False, fert_until=-1, unwatered=0):
    return {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 20 - age,
            "watered_today": watered, "consecutive_unwatered": unwatered,
            "yield_units": yield_units, "max_lifespan_step": -1,
            "fertilized_until_day": fert_until}


def _tasks(tile, day=20, last_day=29, fert=None):
    P = dict(pc.DEFAULTS)
    return plant_tasks([((1, 1), tile)], day, last_day, P, pc.CROPS, fert)


def _ops(tasks):
    return {t[2][0] for t in tasks}


def test_plant_tasks_keeps_watering_an_ongoing_crop_past_max_yield_day():
    for age in (10, 11, 12, 13, 14, 15):
        assert "WATER" in _ops(_tasks(_tile(age))), f"age {age} lost its water task"


def test_plant_tasks_stops_watering_after_the_last_production_event():
    assert "WATER" not in _ops(_tasks(_tile(16)))


def test_plant_tasks_harvests_an_ongoing_crop_from_first_yield_day():
    assert "HARVEST" in _ops(_tasks(_tile(10, yield_units=2)))
    assert "HARVEST" not in _ops(_tasks(_tile(8, yield_units=2)))


def test_plant_tasks_prioritises_harvest_on_a_production_eve():
    """A tile that produces tonight and is not emptied loses the whole event to
    the max_yield cap, so it must outrank routine harvesting (band 4)."""
    eve = [t for t in _tasks(_tile(11, yield_units=4)) if t[2][0] == "HARVEST"]
    calm = [t for t in _tasks(_tile(12, yield_units=4)) if t[2][0] == "HARVEST"]
    assert eve and calm
    assert eve[0][0] < calm[0][0]


def test_plant_tasks_fertilizes_only_on_production_eves():
    always = lambda crop, t: 3.4
    fired = [age for age in range(0, 17)
             if "FERTILIZE" in _ops(_tasks(_tile(age), fert=always))]
    assert fired == [9, 11, 13, 15], fired


def test_plant_tasks_does_not_re_fertilize_inside_an_active_window():
    always = lambda crop, t: 3.4
    # applied on the age-9 day: fertilized_until_day covers the age-11 event
    tile = _tile(11, fert_until=20)
    assert "FERTILIZE" not in _ops(_tasks(tile, fert=always))


# ------------------------------------------- expB-303 shipped acreage invariant

def test_shipped_entry_carries_the_strawberry_acreage():
    """The submission entry must construct the gated expB-303 build.

    `strawberry_tiles` is a fixed constant and therefore a logged §0 violation
    (issue #16): the section-0-clean "mr" rule lost to fixed-16 by 13pp head to
    head, so the constant ships until expB-304 derives the acreage from
    dispatch-queue pricing. FEEDCHAIN_DEFAULTS stay at 0 so
    `millrace:feedchain` and melonrace@725fcf7 remain byte-identical control
    lineages.
    """
    src = (REPO_ROOT / "agents" / "team_b" / "millrace" / "main.py").read_text(encoding="utf-8")
    assert '"strawberry_tiles": 16' in src
    from millrace.feedchain import FEEDCHAIN_DEFAULTS
    assert FEEDCHAIN_DEFAULTS["strawberry_tiles"] == 0
    assert FEEDCHAIN_DEFAULTS["strawberry_rule"] == "fixed"
