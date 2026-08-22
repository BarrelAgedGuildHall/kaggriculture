"""Tests for the G-1 ghost-corpus fetch and its per-commodity census.

The two things worth guarding are the two that fail *silently* in production:

  1. the census's instrumentation can go blind (the engine hands the interpreter a fresh
     deep copy of the state each step, so a player-id map built at the wrong moment
     attributes nothing) -- and a blind run looks exactly like "nobody traded";
  2. the town-drain model can drift from `_town_consume`, which shows up only as a
     non-zero market-inventory identity residual.

Both are checked against a real corpus replay when one is on disk, and skipped cleanly
when it is not (CI has no warehouse).
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import census_ghost_corpus as census  # noqa: E402
import fetch_ghost_corpus as fetch  # noqa: E402


def _corpus() -> Path:
    return Path(os.environ.get("GUILDHALL_WAREHOUSE")
                or (REPO.parents[3] / "warehouse")) / "datasets" / "ghost-corpus"


def _a_replay() -> Path | None:
    idx = _corpus() / "index.csv"
    if not idx.exists():
        return None
    with open(idx, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = _corpus() / row["replay_path"]
            if p.exists():
                return p
    return None


# ------------------------------------------------------------------ pure logic


def test_tiers_are_the_ruling_s_bands():
    """The ruling's three behavioural bands are fixed; review tiers may be added beside
    them, but they must not disturb the bands or their wins-only invariant."""
    assert fetch.RULING_TIERS == ("frontier", "upper", "field")
    for name, band in {"frontier": (1, 10), "upper": (100, 300), "field": (2800, 3300)}.items():
        assert fetch.TIERS[name] == band
    assert fetch.TARGET_PER_TIER == 120 and fetch.MIN_TEAMS_PER_TIER == 4


def test_rank1_is_the_single_top_team():
    assert fetch.TIERS["rank1"] == (1, 1)


def test_cap_is_five_decimal_gb():
    """'5 GB' is read conservatively: 5e9, not 5 GiB."""
    assert fetch.CAP_BYTES == 5_000_000_000


def test_engine_pin_is_shared_by_fetch_and_census():
    assert fetch.ENGINE_PIN == census.PIN


def test_index_fields_carry_provenance_and_engine():
    for required in ("episode_id", "tier", "team_id", "lb_rank", "rating_at_play",
                     "engine_version", "bytes_gz", "replay_path"):
        assert required in fetch.INDEX_FIELDS


def test_census_fields_carry_units_and_realised_price():
    """The pricing table needs money, not only units (ruling-02-addendum-02 §1)."""
    for required in ("p0_units_sold", "p1_units_sold", "p0_realised_price",
                     "p1_realised_price", "p0_sell_revenue", "town_drain_units",
                     "price_mean", "identity_residual_abs"):
        assert required in census.FIELDS


def test_town_drain_matches_the_engine_s_shop_tables():
    """Single-product shops consume at 2x; the town centre fires on its own interval."""
    kg = census._kg()
    products = list(kg.PRODUCTS)
    cfg = {"townShopSellInterval": 4, "townCenterSellInterval": 24}
    shop = next(iter(kg.SHOPS))
    items = kg.SHOPS[shop]
    mult = 2 if len(items) == 1 else 1

    off_tick = census.town_drain(kg, cfg, [shop], 1, products)
    assert all(v == 0 for v in off_tick.values()), "no drain off a shop tick"

    on_tick = census.town_drain(kg, cfg, [shop], 4, products)
    for item in items:
        assert on_tick[item] == mult

    # step 0 is both a shop tick and a town-centre tick
    both = census.town_drain(kg, cfg, [], 0, products)
    for item in kg.TOWN_CENTER_PRODUCTS:
        if item in both:
            assert both[item] == 1


# ------------------------------------------------------------------ real replay


@pytest.mark.skipif(_a_replay() is None, reason="no ghost corpus on this machine")
def test_corpus_replays_are_gzipped_and_pinned():
    p = _a_replay()
    assert p.suffix == ".gz", "corpus replays are stored packed"
    with gzip.open(p, "rt", encoding="utf-8") as f:
        replay = json.load(f)
    assert replay["module_version"] == census.PIN
    assert replay.get("info", {}).get("seed") is not None, "needed to re-step deterministically"


@pytest.mark.skipif(_a_replay() is None, reason="no ghost corpus on this machine")
def test_census_closes_the_inventory_identity_and_saw_real_trades():
    res = census.census_episode(_a_replay())
    assert res.get("ok"), res.get("error")
    assert res["market_parity_mismatches"] == 0, "local pool diverged from the recorded pool"
    # the identity is exact, not a tolerance
    assert all(r["identity_residual_abs"] == 0 for r in res["rows"])
    # ...and the instrumentation was not blind: a real ladder episode trades
    assert sum(r["p0_units_sold"] + r["p1_units_sold"] for r in res["rows"]) > 0
    # realised price is only reported where units actually moved
    for r in res["rows"]:
        for pl in (0, 1):
            if r[f"p{pl}_units_sold"] == 0:
                assert r[f"p{pl}_realised_price"] == ""
            else:
                assert r[f"p{pl}_realised_price"] > 0


@pytest.mark.skipif(not (_corpus() / "index.csv").exists(), reason="no corpus index")
def test_index_rows_are_wins_at_the_pinned_engine():
    """Every row is at the pin; the three ruling tiers are wins-only; and because the index
    has no `outcome` column, `bank_margin` has to BE the outcome -- a consumer that wants
    the old wins-only invariant filters on its sign."""
    with open(_corpus() / "index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        assert r["engine_version"] == fetch.ENGINE_PIN
        assert r["tier"] in fetch.TIERS
        margin = float(r["bank"]) - float(r["opp_bank"])
        assert float(r["bank_margin"]) == margin
        assert margin != 0, "a tie is neither a win nor a loss and is never selected"
        if r["tier"] in fetch.RULING_TIERS:
            assert margin > 0, "the ruling tiers are winning episodes only"


def _fake_lb(rank: int = 1, team: str = "T1") -> list[dict]:
    return [{"Rank": str(rank), "TeamId": team, "Score": "3142.4", "TeamName": "Top"}]


def _ep(eid: int, end: str, won: bool, team: str = "T1", sub: str = "s") -> dict:
    return {"episode_id": eid, "end_time": end, "seat": 0, "team_id": team,
            "team_name": "Top", "submission_id": sub, "rating_at_play": 3000.0,
            "rating_after": 3010.0, "bank": 100 if won else 10, "opp_team_id": "X",
            "opp_submission_id": "y", "opp_rating_at_play": 2900.0,
            "opp_bank": 10 if won else 100, "won": won}


def test_selection_ranks_recency_across_submissions_not_within_one(monkeypatch):
    """A team's "N most recent wins" is a statement about the TEAM. Stopping at the first
    submission that fills the quota selects whichever submission discovery happened to list
    first -- which, at the top of the ladder, is the retired one."""
    old = [_ep(i, f"2026-08-17T0{i}:00:00Z", True, sub="old") for i in range(1, 4)]
    new = [_ep(10 + i, f"2026-08-22T0{i}:00:00Z", True, sub="new") for i in range(1, 4)]
    monkeypatch.setattr(fetch, "mirror_submissions_by_team", lambda: {"T1": ["old", "new"]})
    monkeypatch.setattr(fetch, "episodes_for_submission",
                        lambda sub: old if sub == "old" else new)
    picked = fetch.select(_fake_lb(), per_team=2, target=2, only_tier="rank1",
                          since="2026-08-16", min_teams=1)["rank1"]
    assert [r["episode_id"] for r in picked] == [13, 12], "the newest two episodes overall"


def test_selection_can_take_losses_and_keeps_them_out_of_the_ruling_tiers(monkeypatch):
    eps = ([_ep(i, f"2026-08-2{i}T01:00:00Z", True) for i in range(1, 4)]
           + [_ep(20 + i, f"2026-08-2{i}T02:00:00Z", False) for i in range(1, 4)])
    monkeypatch.setattr(fetch, "mirror_submissions_by_team", lambda: {"T1": ["s"]})
    monkeypatch.setattr(fetch, "episodes_for_submission", lambda sub: eps)

    review = fetch.select(_fake_lb(), per_team=2, target=3, only_tier="rank1",
                          since="2026-08-16", min_teams=1, losses_per_team=1)["rank1"]
    assert sum(1 for r in review if r["won"]) == 2
    assert sum(1 for r in review if not r["won"]) == 1
    assert [r["episode_id"] for r in review if not r["won"]] == [23], "most recent loss"

    # the default (losses_per_team=0) must still be wins-only
    wins_only = fetch.select(_fake_lb(), per_team=3, target=3, only_tier="rank1",
                             since="2026-08-16", min_teams=1)["rank1"]
    assert all(r["won"] for r in wins_only)


def test_selection_accepts_explicit_submissions_instead_of_the_mirror(monkeypatch):
    """The mirror barely covers the top of the ladder, so discovery must be overridable."""
    def boom():
        raise AssertionError("mirror discovery must not run when submissions are given")
    monkeypatch.setattr(fetch, "mirror_submissions_by_team", boom)
    monkeypatch.setattr(fetch, "episodes_for_submission",
                        lambda sub: [_ep(1, "2026-08-22T01:00:00Z", True, sub=sub)])
    picked = fetch.select(_fake_lb(), per_team=1, target=1, only_tier="rank1",
                          since="2026-08-16", min_teams=1, submissions=["55614463"])["rank1"]
    assert [r["submission_id"] for r in picked] == ["55614463"]


def test_kept_leaderboard_snapshot_name_is_a_legal_windows_filename(tmp_path,
                                                                    monkeypatch):
    """The LB csv's name carries a colon-bearing timestamp; copying it through unchanged
    fails with WinError 123 *after* the new download has already replaced the old file."""
    z = tmp_path / "kaggriculture.zip"
    z.write_bytes(b"not-a-zip")
    monkeypatch.setattr(fetch, "read_leaderboard",
                        lambda p: ([], "kaggriculture-publicleaderboard-2026-08-22T14:03:57.csv"))
    fetch.keep_snapshot(z)
    kept = list(tmp_path.glob("kaggriculture-*.zip"))
    assert len(kept) == 1 and ":" not in kept[0].name
    assert "2026-08-22T14-03-57" in kept[0].name


def test_selection_floors_at_the_engine_ship_date():
    """1.32.7 shipped 2026-08-15; selecting older wins only buys 30 MB downloads that get
    discarded, and it bites the LOW tiers hardest because they play rarely."""
    assert fetch.PIN_SHIPPED == "2026-08-16"


@pytest.mark.skipif(not (_corpus() / "index.csv").exists(), reason="no corpus index")
def test_corpus_has_no_orphan_files_or_missing_replays():
    """index.csv is authoritative: every row has a file, every file has a row."""
    corpus = _corpus()
    with open(corpus / "index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    indexed = {int(r["episode_id"]) for r in rows}
    assert len(indexed) == len(rows), "duplicate episode ids in the index"
    on_disk = {int(p.name.split(".")[0])
               for tier in fetch.TIERS for p in (corpus / tier).glob("*.json.gz")}
    assert indexed == on_disk


@pytest.mark.skipif(not (_corpus() / "index.csv").exists(), reason="no corpus index")
def test_corpus_is_under_the_cap():
    with open(_corpus() / "index.csv", newline="", encoding="utf-8") as f:
        packed = sum(int(r["bytes_gz"]) for r in csv.DictReader(f))
    assert packed < fetch.CAP_BYTES
