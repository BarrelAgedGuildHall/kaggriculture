"""Tests for the ladder-record / Pit-calibration tooling.

THE LOAD-BEARING PROPERTY is the exclusion of Kaggle's upload-validation episode. That
episode is SELF-PLAY -- both seats carry our own submission and team id -- and it is the one
that stamps the initial 600 (`rating_before` 0 -> `rating_after` 600 on both seats). Counted,
it silently adds +2 W to every agent's record, reports `rating_start` as 0 instead of 600, and
produces one bogus "the ladder's rating movement disagrees with the bank comparison" row. It
is a plausible-looking corruption of a number that goes straight into the submission ledger,
so it is asserted in code (`split_validation`) and pinned here rather than left to a comment.

Everything below runs on synthetic rows: no network, no Kaggle credentials, no replays.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from episode_index import FIELDS  # noqa: E402
from ladder_record import (  # noqa: E402
    LadderDataError,
    MIN_GAMES,
    check_result_consistency,
    elo_mle,
    games_needed,
    ledger_cell,
    linfit,
    read_seat_rows,
    result_of,
    split_validation,
    summarise,
    wilson,
)

SUB = 55671796
OUR_TEAM = 16740404


def row(episode_id, *, opp_team=99, rating_before=600.0, rating_after=610.0,
        reward=70000.0, opp_reward=60000.0, end_time="2026-08-21T15:15:37Z",
        submission_id=SUB, seat=0, opp_rating=650.0):
    return {
        "episode_id": episode_id, "end_time": end_time, "seat": seat,
        "team_id": OUR_TEAM, "team_name": "BarrelAged", "submission_id": submission_id,
        "rating_before": rating_before, "rating_after": rating_after, "reward": reward,
        "opp_team_id": opp_team, "opp_team_name": "somebody", "opp_rating_before": opp_rating,
        "opp_reward": opp_reward,
        "won": int(reward > opp_reward), "tied": int(reward == opp_reward),
    }


def validation_rows():
    """The upload-validation episode as Kaggle actually reports it: one episode, BOTH seats
    ours, rating 0 -> 600 on each, and the two seats disagree on the bank (one 'wins')."""
    return [
        row(900001, opp_team=OUR_TEAM, rating_before=0.0, rating_after=600.0,
            reward=37542.0, opp_reward=44215.0, seat=0, opp_rating=0.0,
            end_time="2026-08-21T15:10:00Z"),
        row(900001, opp_team=OUR_TEAM, rating_before=0.0, rating_after=600.0,
            reward=44215.0, opp_reward=37542.0, seat=1, opp_rating=0.0,
            end_time="2026-08-21T15:10:00Z"),
    ]


def ladder_rows():
    """Three real games: W, L, W."""
    return [
        row(1, rating_before=600.0, rating_after=660.0, reward=70000.0, opp_reward=60000.0,
            end_time="2026-08-21T15:20:00Z"),
        row(2, rating_before=660.0, rating_after=610.0, reward=50000.0, opp_reward=80000.0,
            end_time="2026-08-21T15:30:00Z"),
        row(3, rating_before=610.0, rating_after=655.0, reward=90000.0, opp_reward=20000.0,
            end_time="2026-08-21T15:40:00Z"),
    ]


# --------------------------------------------------------------- the load-bearing exclusion


def test_validation_episode_is_excluded_from_the_record():
    rec = summarise(SUB, validation_rows() + ladder_rows())
    assert rec["n"] == 3, "the self-play validation episode must not count as ladder games"
    assert (rec["W"], rec["L"], rec["T"]) == (2, 1, 0)
    assert rec["validation_episodes"] == 1
    assert rec["rating_start"] == 600.0, "rating_start must be the first LADDER game's 600"
    assert not rec["inconsistent"], "the validation episode is the only self-disagreeing row"


def test_counting_the_validation_episode_is_what_the_exclusion_prevents():
    """Pins the exact size of the corruption, so a regression is recognisable in a diff.

    Both seats of the validation episode belong to us, so leaving it in adds TWO rows: by the
    bank comparison one is a win and one is a loss (+1 W, +1 L), and by the ladder's rating
    movement BOTH are wins (0 -> 600), which is also where the spurious rating/bank
    disagreement comes from.
    """
    naive = validation_rows() + ladder_rows()
    clean = summarise(SUB, naive)

    assert len(naive) == clean["n"] + 2
    assert sum(r["won"] for r in naive) == clean["W"] + 1
    assert sum(1 for r in naive if r["won"] == 0) == clean["L"] + 1
    assert [result_of(r) for r in validation_rows()] == ["W", "W"]
    assert clean["rating_start"] == 600.0 and naive[0]["rating_before"] == 0.0


def test_leaked_validation_row_is_caught_by_the_independent_signature():
    """If the team-id filter is weakened, the zero `rating_before` still gives it away."""
    leaked = validation_rows()[0]
    leaked["opp_team_id"] = 12345          # no longer looks like self-play...
    with pytest.raises(LadderDataError, match="rating_before == 0"):
        split_validation([leaked] + ladder_rows())


def test_unfiltered_csv_rows_are_rejected():
    """An episode_index CSV has one row per (episode, seat), so it carries the OPPONENT's
    rows too. Summarising those as ours would double-count every game."""
    opponent = row(1, submission_id=99999999)
    with pytest.raises(LadderDataError, match="span 2 submissions"):
        split_validation(ladder_rows() + [opponent])


def test_more_than_one_validation_episode_is_refused():
    second = [dict(r, episode_id=900002) for r in validation_rows()]
    with pytest.raises(LadderDataError, match="at most one"):
        split_validation(validation_rows() + second + ladder_rows())


def test_read_seat_rows_keeps_only_our_seat(tmp_path):
    path = tmp_path / "episodes-sub-1.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ladder_rows() + [row(1, submission_id=99999999, seat=1)])
    rows = read_seat_rows(path, SUB)
    assert len(rows) == 3
    assert {r["submission_id"] for r in rows} == {SUB}
    split_validation(rows)   # must not raise now that the opponent's rows are gone


# --------------------------------------------------------------- result derivation


def test_result_comes_from_the_rating_movement():
    assert result_of(row(1, rating_before=600.0, rating_after=660.0)) == "W"
    assert result_of(row(1, rating_before=660.0, rating_after=600.0)) == "L"
    assert result_of(row(1, rating_before=660.0, rating_after=660.0)) == "T"


def test_rating_bank_disagreement_is_reported():
    """Kaggriculture's ladder result IS the bank comparison (141/141 real games as of
    2026-08-22). A disagreement means a leak or a change at Kaggle -- it must surface."""
    assert check_result_consistency(ladder_rows()) == []
    liar = row(7, rating_before=600.0, rating_after=660.0, reward=10.0, opp_reward=99999.0)
    bad = check_result_consistency([liar])
    assert len(bad) == 1 and bad[0]["by_rating"] == "W" and bad[0]["by_bank"] == "L"


# --------------------------------------------------------------- statistics


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson(15, 31)
    assert lo < 100 * 15 / 31 < hi
    assert 30 < lo < 35 and 63 < hi < 68        # the rebuild-v2 record, 2026-08-22
    assert math.isnan(wilson(0, 0)[0])


def test_elo_mle_recovers_a_known_strength():
    """Generate expected scores from a known R against a spread of opponents; the MLE must
    return that R (this is the estimator the whole calibration read rests on)."""
    true_r = 1200.0
    opponents = [900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
    games = [(o, 1 / (1 + 10 ** ((o - true_r) / 400))) for o in opponents for _ in range(20)]
    r, se = elo_mle(games)
    assert r == pytest.approx(true_r, abs=1.0)
    assert 0 < se < 40


def test_elo_mle_is_undefined_for_a_clean_sweep():
    assert math.isnan(elo_mle([(600.0, 1.0), (700.0, 1.0)])[0])
    assert math.isnan(elo_mle([])[0])


def test_elo_mle_matches_the_closed_form_at_fifty_percent():
    """Against a single repeated opponent, a 50% score puts you exactly at their rating."""
    r, _ = elo_mle([(700.0, 1.0), (700.0, 0.0)] * 10)
    assert r == pytest.approx(700.0, abs=0.5)


def test_games_needed_shrinks_with_the_square_of_the_gap():
    assert games_needed(330) == pytest.approx(8.5, abs=0.5)
    assert games_needed(100) == pytest.approx(93, abs=2)
    assert games_needed(100) == pytest.approx(games_needed(200) * 4, rel=1e-6)


def test_linfit_is_exact_on_a_line():
    slope, r = linfit([(0.0, 1.0), (10.0, 3.0), (20.0, 5.0)])
    assert slope == pytest.approx(0.2)
    assert r == pytest.approx(1.0)


# --------------------------------------------------------------- ledger output


def test_ledger_cell_refuses_to_read_a_thin_record():
    thin = summarise(SUB, validation_rows() + ladder_rows())
    assert thin["n"] < MIN_GAMES
    cell = ledger_cell(thin, "2026-08-22T02:24Z")
    assert "TOO EARLY TO READ" in cell
    assert "2-1-0" in cell and "2026-08-22T02:24Z" in cell


def test_ledger_cell_reports_a_full_record():
    rows = validation_rows()
    for i in range(MIN_GAMES + 2):
        rows.append(row(100 + i, rating_before=600.0 + i,
                        rating_after=600.0 + i + (5 if i % 2 else -5),
                        reward=70000.0 if i % 2 else 50000.0,
                        opp_reward=60000.0, end_time=f"2026-08-21T16:{i:02d}:00Z"))
    rec = summarise(SUB, rows)
    cell = ledger_cell(rec, "2026-08-22T02:24Z")
    assert "TOO EARLY" not in cell
    assert "opponent-adjusted Elo" in cell and "win " in cell
    assert f"{rec['n']} games" in cell


def test_summarise_handles_a_submission_with_only_a_validation_episode():
    rec = summarise(SUB, validation_rows())
    assert rec["n"] == 0 and rec["validation_episodes"] == 1
    assert "no ladder games" in rec["note"]
    assert "no ladder games yet" in ledger_cell(rec, "2026-08-22T02:24Z")
