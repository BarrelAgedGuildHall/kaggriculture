"""S-3: the Pit's per-episode NAV telemetry (arena/navlog.py).

The properties worth pinning, in order of how much they would cost if they broke:

1. **Off is off.** A run with `nav_rate=0` -- every gate, every screen, every confirm -- writes
   nothing and carries no telemetry flag on its jobs. This is the promise that S-3 did not
   change what a measured arm measures.
2. **The terminal snapshot is always sampled**, whatever the sampling interval. `phi` there is 0
   by construction (H = 0) and `nav == cash == the realised bank`; that identity is the whole
   validity check the trajectories exist to support (exp-N1 Test 1b).
3. **Sampling is deterministic in `game_id`**, so a partial recording is an unbiased sample of
   pairings rather than of whatever the ladder happened to be playing when someone looked.
4. Rows survive the round trip: sqlite, the remote-delta merge, and the JSON API.

Run: uv run pytest -q tests/test_nav_telemetry.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# See tests/test_arena_push.py: `arena.paths` reads KAGGRI_ARENA_DIR once, at import, and the
# other arena modules bind their directories from it at THEIR import -- so whoever redirects
# first owns the scratch dir for the session and a later module adopts it.
if "arena.paths" in sys.modules:
    import arena.paths as paths                                        # noqa: E402

    _TMP = paths.ARENA_DIR
    assert _TMP != paths.DATA_DIR / "arena", "the arena dir must be redirected before arena is imported"
else:
    _TMP = Path(tempfile.mkdtemp(prefix="arena-nav-test-"))
    os.environ["KAGGRI_ARENA_DIR"] = str(_TMP)
    import arena.paths as paths                                        # noqa: E402

    importlib.reload(paths)
    assert paths.ARENA_DIR == _TMP

from arena import navlog, scheduler, serve, store  # noqa: E402
from arena.paths import DB_PATH  # noqa: E402

STEPS = 48          # two game days: enough for a real trajectory, fast


def _nav(**kw):
    con = store.connect(DB_PATH)
    try:
        return store.nav_rows(con, **kw)
    finally:
        con.close()


# ------------------------------------------------------------------ pure helpers
def test_sampling_is_deterministic_and_honours_the_rate():
    import hashlib
    # real ids are `scheduler.game_id` = a sha1 prefix, so use that shape and not a counter
    ids = [hashlib.sha1(str(i).encode()).hexdigest()[:16] for i in range(1000)]
    assert not any(navlog.should_record(g, 0) for g in ids), "rate 0 is off"
    assert all(navlog.should_record(g, 100) for g in ids)
    chosen = [g for g in ids if navlog.should_record(g, 10)]
    assert chosen == [g for g in ids if navlog.should_record(g, 10)], "stable across calls"
    assert 50 < len(chosen) < 160, f"~10% of 1000, got {len(chosen)}"
    # a bigger rate is a superset: raising it never drops a game already being recorded
    assert set(chosen) <= {g for g in ids if navlog.should_record(g, 25)}


def test_the_terminal_snapshot_is_always_sampled():
    assert navlog.sample_steps(720, 24)[-1] == 719
    assert navlog.sample_steps(720, 24)[:3] == [0, 24, 48]
    assert navlog.sample_steps(720, 1000) == [0, 719], "a huge interval still marks both ends"
    assert navlog.sample_steps(1, 24) == [0]
    assert navlog.sample_steps(0, 24) == []


# ------------------------------------------------------------------ a real (tiny) run
def test_telemetry_is_off_unless_asked():
    before = len(_nav(limit=10_000))
    res = scheduler.run_match(["pass", "random"], mode="screen", n=2, workers=1, steps=STEPS, quiet=True)
    assert not any(j.get("nav_telemetry") for j in res["jobs"]), "no job carries the flag"
    assert len(_nav(limit=10_000)) == before, "a default run records nothing"


def test_a_recorded_run_marks_both_seats_and_matches_the_realised_bank():
    res = scheduler.run_match(["pass", "starter"], mode="screen", n=2, workers=1, steps=STEPS,
                              quiet=True, nav_rate=100)
    rows = _nav(run_id=res["run_id"], limit=100)
    assert len(rows) == 2 * len(res["rows"]), "one row per (game, seat)"
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game_id"], {})[r["seat"]] = r
    for game in res["rows"]:
        pair = by_game[game["game_id"]]
        assert set(pair) == {0, 1}
        for seat, r in pair.items():
            assert r["error"] == "", r["error"]
            assert r["agent"] == game[f"p{seat}"]
            assert r["nav_schema"] >= 1 and len(r["nav_source"]) == 16
            assert r["variant"] == navlog.VARIANT and r["lam_opp"] > 0
            pts = navlog.trajectory(r)
            assert len(pts) == r["n_points"] > 1
            assert [p["step"] for p in pts] == navlog.sample_steps(STEPS, r["sample_every"])
            assert pts[-1]["cash"] == game[f"money{seat}"], "the last mark's cash IS the bank"
            assert r["final_cash"] == pts[-1]["cash"] and r["final_nav"] == pts[-1]["nav"]
            assert set(pts[0]) == {"step", "day", "cash", "nav", "phi", "provisional",
                                   "lines", "diagnostics"}, "detail is dropped by default"


def test_partial_rate_records_a_subset_and_re_runs_are_stable():
    res = scheduler.run_match(["pass", "random"], mode="screen", n=8, workers=2, steps=STEPS,
                              quiet=True, nav_rate=50)
    flagged = {j["game_id"] for j in res["jobs"] if j.get("nav_telemetry")}
    assert 0 < len(flagged) < len(res["jobs"]), f"{len(flagged)} of {len(res['jobs'])}"
    got = {r["game_id"] for r in _nav(run_id=res["run_id"], limit=100)}
    assert got == flagged
    again = scheduler.run_match(["pass", "random"], mode="screen", n=8, workers=2, steps=STEPS,
                                quiet=True, nav_rate=50)
    assert {j["game_id"] for j in again["jobs"] if j.get("nav_telemetry")} == flagged


def test_detail_is_optional():
    res = scheduler.run_match(["pass", "starter"], mode="screen", n=2, workers=1, steps=STEPS,
                              quiet=True, nav_rate=100, nav_detail=True, nav_sample_every=16)
    r = _nav(run_id=res["run_id"], limit=1)[0]
    pts = navlog.trajectory(r)
    assert "detail" in pts[0] and isinstance(pts[0]["detail"], dict)
    assert [p["step"] for p in pts] == navlog.sample_steps(STEPS, 16)


# ------------------------------------------------------------------ calibration
def _synthetic(n=12, noise=0.0):
    """n games, both seats, where NAV at step 24 is a perfect predictor of the bank."""
    rows = []
    for i in range(n):
        for seat in (0, 1):
            nav = 1000.0 + 100 * i + (500 if seat == 0 else 0)
            money = nav + (noise if seat == 0 else 0)
            rows.append({
                "game_id": f"g{i:04d}", "seat": seat, "money": money,
                "result": 1.0 if seat == 0 else 0.0,      # seat 0 always wins, and always leads
                "points": json.dumps([{"step": 0, "day": 0, "cash": 3000.0, "nav": 3000.0,
                                       "phi": 0.0, "provisional": False, "lines": {}},
                                      {"step": 24, "day": 1, "cash": 3000.0, "nav": nav,
                                       "phi": nav - 3000.0, "provisional": False, "lines": {}}]),
            })
    return rows


def test_calibration_reads_a_perfect_predictor():
    table = {t["step"]: t for t in navlog.calibration(_synthetic())}
    assert table[24]["r_nav"] == 1.0
    assert table[24]["r_cash"] is None, "a constant cash column has no correlation, not a fake one"
    assert table[24]["sign_acc"] == 1.0 and table[24]["n_pairs"] == 12
    assert table[0]["n_pairs"] == 0 and table[0]["n_ties"] == 12, "a zero gap has no sign to score"


def test_a_loss_is_not_counted_as_a_tie():
    """Regression: `float(row['result'] or 0.5)` turns a LOSS (0.0) into a tie, which silently
    drops every losing game out of the sign accuracy -- half the sample, always the same half."""
    rows = _synthetic(n=10)
    for r in rows:                       # flip the outcome: now seat 0 leads on NAV and LOSES
        r["result"] = 0.0 if r["seat"] == 0 else 1.0
    t = {x["step"]: x for x in navlog.calibration(rows)}[24]
    assert t["n_pairs"] == 10 and t["n_ties"] == 0
    assert t["sign_acc"] == 0.0, "leading on NAV and losing every time is 0% accuracy, not 100%"


def test_calibration_refuses_to_report_a_tiny_sample():
    assert navlog.calibration(_synthetic(n=2)) == []


# ------------------------------------------------------------------ round trips
def test_rows_merge_from_a_remote_delta(tmp_path):
    delta = tmp_path / "delta.sqlite"
    res = scheduler.run_match(["pass", "random"], mode="screen", n=2, workers=1, steps=STEPS,
                              quiet=True, nav_rate=100, db_path=delta,
                              run_id="deltarun-" + os.urandom(3).hex())
    assert len(_nav(run_id=res["run_id"], limit=10)) == 0, "it went to the delta, not the main store"
    con = store.connect(DB_PATH)
    store.import_db(con, delta)
    con.close()
    merged = _nav(run_id=res["run_id"], limit=10)
    assert len(merged) == 2 * len(res["rows"])
    assert all(navlog.trajectory(r) for r in merged)


def test_import_of_a_pre_s3_store_is_not_an_error(tmp_path):
    import sqlite3
    old = tmp_path / "old.sqlite"
    con = store.connect(old)
    con.execute("DROP TABLE nav_traj")           # a store written before S-3 existed
    con.commit()
    con.close()
    assert "nav_traj" not in {r[0] for r in sqlite3.connect(str(old)).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    main = store.connect(DB_PATH)
    store.import_db(main, old)                   # must not raise
    main.close()


def test_the_api_serves_trajectories():
    res = scheduler.run_match(["pass", "starter"], mode="screen", n=2, workers=1, steps=STEPS,
                              quiet=True, nav_rate=100)
    out = serve.api_nav(run=res["run_id"])
    assert out["n"] == 2 * len(res["rows"]) and out["total"] >= out["n"]
    row = out["rows"][0]
    assert isinstance(row["points"], list) and isinstance(row["diagnostics"], dict)
    lean = serve.api_nav(run=res["run_id"], points=False)
    assert "points" not in lean["rows"][0], "points=0 leaves out the blob"
    json.dumps(out)                              # the whole payload must be JSON-serialisable
