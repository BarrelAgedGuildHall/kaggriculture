"""Arena tests: registry round-trip, a tiny screen run, ratings on synthetic results, gate logic.

The arena state is redirected to a temp dir through KAGGRI_ARENA_DIR (read at import time by
arena.paths), so these tests never touch data/arena/. Run: uv run pytest -q tests/test_arena.py
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_TMP = Path(tempfile.mkdtemp(prefix="arena-test-"))
os.environ["KAGGRI_ARENA_DIR"] = str(_TMP)

import arena.paths as paths  # noqa: E402

importlib.reload(paths)
assert paths.ARENA_DIR == _TMP

from arena import gates, ratings, registry, scheduler, store  # noqa: E402
from arena.paths import ARENA_DIR, DB_PATH  # noqa: E402

TRIVIAL = REPO / "agent" / "main.py"   # the PASS placeholder


# ----------------------------------------------------------------------------- registry
def test_registry_round_trip():
    a = registry.register_file(TRIVIAL, "pass-file", "A", notes="placeholder")
    assert a["id"].startswith("pass-file@")
    assert (ARENA_DIR / "agents" / a["entry"]).read_bytes() == TRIVIAL.read_bytes()
    again = registry.register_file(TRIVIAL, "pass-file", "A")
    assert again["id"] == a["id"], "same bytes -> same immutable id"
    assert registry.resolve("pass-file")["id"] == a["id"]
    assert registry.resolve(a["id"])["id"] == a["id"]
    b = registry.resolve("random")
    assert b["kind"] == "builtin" and b["team"] == "resident"
    with pytest.raises(KeyError):
        registry.resolve("nobody-here")
    registry.set_champion("A", "pass-file")
    assert registry.get_champion("A")["id"] == a["id"]
    reg = registry.load()
    assert reg["champions"]["A"] == a["id"]


def test_game_ids_are_seat_and_seed_specific():
    g1 = scheduler.game_id("a@1", "b@1", 5, "1.32.7")
    assert g1 == scheduler.game_id("a@1", "b@1", 5, "1.32.7")
    assert g1 != scheduler.game_id("b@1", "a@1", 5, "1.32.7")
    assert g1 != scheduler.game_id("a@1", "b@1", 6, "1.32.7")


def test_parse_seeds():
    assert scheduler.parse_seeds(None, "screen", 64) == list(range(0, 32))
    assert scheduler.parse_seeds(None, "confirm", 400) == list(range(1000, 1200))
    assert scheduler.parse_seeds("7-9", "screen", 64) == [7, 8, 9]
    assert scheduler.parse_seeds("3,1", "screen", 64) == [3, 1]


# ----------------------------------------------------------------------------- a real tiny run
def test_screen_random_vs_pass_two_seeds():
    """2 seeds x 2 seats = 4 games, 48 steps each (2 days: fast). PASS keeps 3000; random spends."""
    res = scheduler.run_match(["random", "pass"], mode="screen", n=4, workers=2, steps=48, quiet=True)
    rows = res["rows"]
    assert len(rows) == 4
    assert {r["seed"] for r in rows} == {0, 1}
    assert sorted((r["p0"].split("@")[0], r["seed"]) for r in rows) == [("pass", 0), ("pass", 1), ("random", 0), ("random", 1)]
    for r in rows:
        assert r["status0"] == "DONE" and r["status1"] == "DONE", r
        assert r["steps_played"] == 48
        assert r["crash0"] == r["crash1"] == 0 and r["harness_error"] == ""
        pass_side = 0 if r["p0"].startswith("pass") else 1
        assert r[f"money{pass_side}"] == 3000.0
        assert r[f"p99_{pass_side}"] >= 0.0
    con = store.connect(DB_PATH)
    stored = store.games_between(con, rows[0]["p0"], rows[0]["p1"])
    con.close()
    assert len(stored) == 4, "every game is upserted into the sqlite store"
    # Re-running the same seeds upserts (same game ids), it does not duplicate.
    res2 = scheduler.run_match(["random", "pass"], mode="screen", n=4, workers=2, steps=48, quiet=True)
    con = store.connect(DB_PATH)
    assert len(store.games_between(con, rows[0]["p0"], rows[0]["p1"])) == 4
    con.close()
    assert {r["game_id"] for r in res2["rows"]} == {r["game_id"] for r in rows}


def test_crashing_agent_is_flagged_and_loses(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def agent(obs):\n    if obs.step == 5:\n        raise RuntimeError('boom')\n"
                   "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n", encoding="utf-8")
    registry.register_file(bad, "boom", "B")
    res = scheduler.run_match(["boom", "pass"], mode="screen", n=2, workers=1, steps=24, quiet=True,
                              keep_replay="crash")
    assert len(res["rows"]) == 2
    for r in res["rows"]:
        side = 0 if r["p0"].startswith("boom") else 1
        assert r[f"crash{side}"] == 1
        assert "boom" in r[f"error{side}"]
        assert (r["result0"] == 0.0) if side == 0 else (r["result0"] == 1.0), "a crash is scored as a loss"
        assert r["replay_path"] and Path(r["replay_path"]).exists(), "crash replays are kept"


# ----------------------------------------------------------------------------- ratings
def _g(p0, p1, r0, seed=0):
    return {"p0": p0, "p1": p1, "result0": r0, "seed": seed, "crash0": 0, "crash1": 0, "timeout0": 0, "timeout1": 0,
            "p99_0": 1.0, "p99_1": 1.0, "max_0": 2.0, "max_1": 2.0, "money0": 100.0, "money1": 90.0,
            "created_at": "2026-08-19T00:00:00"}


def test_bt_and_elo_orderings_on_synthetic_results():
    games = []
    # strong beats mid 80%, mid beats weak 80%, strong beats weak 95%; ties sprinkled in
    for i in range(100):
        games.append(_g("strong", "mid", 1.0 if i < 80 else 0.0, i))
        games.append(_g("mid", "weak", 1.0 if i < 80 else 0.0, i))
        games.append(_g("weak", "strong", 0.0 if i < 95 else (0.5 if i < 98 else 1.0), i))
    bt = ratings.bradley_terry(games)
    assert bt["strong"] > bt["mid"] > bt["weak"]
    assert abs(sum(bt.values())) < 1e-6, "centred"
    # BT log-odds for an 80/20 pairing is ~log(4) = 1.39
    assert 1.0 < bt["strong"] - bt["mid"] < 1.8
    disp = ratings.bt_display(bt)
    assert disp["strong"] > disp["mid"] > disp["weak"]
    el = ratings.elo(games)
    assert el["strong"] > el["mid"] > el["weak"]
    h2h = ratings.head_to_head(games)
    c = h2h["strong"]["mid"]
    assert c["n"] == 100 and c["wins"] == 80 and c["ci_lo"] < 0.8 < c["ci_hi"]
    assert h2h["mid"]["strong"]["rate"] == pytest.approx(0.2)


def test_wilson_interval_basic():
    lo, hi = ratings.wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert ratings.wilson(0, 0) == (0.0, 0.0)
    lo, hi = ratings.wilson(64, 64)
    assert lo > 0.9 and hi == 1.0


# ----------------------------------------------------------------------------- gates
def test_gate_logic_on_store(tmp_path):
    db = tmp_path / "g.sqlite"
    con = store.connect(db)
    champ = registry.register_file(TRIVIAL, "champ", "A")
    regfile = tmp_path / "cand.py"
    regfile.write_text("def agent(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n# cand\n", encoding="utf-8")
    cand = registry.register_file(regfile, "cand", "A")
    registry.set_champion("A", champ["id"])
    base = {"run_id": "t", "mode": "confirm", "steps": 720, "engine": "1.32.7", "arena_sha": "x", "host": "t",
            "status0": "DONE", "status1": "DONE", "over_budget0": 0, "over_budget1": 0, "error0": "", "error1": "",
            "harness_error": "", "steps_played": 720, "wall_s": 1.0, "replay_path": "", "p50_0": 1, "p50_1": 1}
    rows = []
    for k in range(200):
        seed = 1000 + k
        for p0, p1, r0 in ((cand["id"], champ["id"], 1.0 if k % 10 else 0.0), (champ["id"], cand["id"], 0.0 if k % 10 else 1.0)):
            rows.append({**base, **_g(p0, p1, r0, seed), "game_id": scheduler.game_id(p0, p1, seed, "1.32.7")})
    store.upsert_rows(con, rows)
    con.close()
    res = gates.evaluate("cand", team="A", db_path=db)
    assert res["pass"], res
    assert res["n"] == 400 and res["win_rate"] == pytest.approx(0.9)
    # Not enough games on the confirm set -> FAIL
    res = gates.evaluate("cand", team="A", db_path=db, min_games=401)
    assert not res["pass"] and any("enough_games" in r for r in res["reasons"])
    # A single crash -> FAIL
    con = store.connect(db)
    bad = dict(rows[0]); bad["crash0"] = 1
    store.upsert_rows(con, [bad]); con.close()
    res = gates.evaluate("cand", team="A", db_path=db)
    assert not res["pass"] and any("no_crashes" in r for r in res["reasons"])
    # Slow p99 -> FAIL
    con = store.connect(db)
    bad = dict(rows[0]); bad["p99_0"] = 500.0
    store.upsert_rows(con, [bad]); con.close()
    res = gates.evaluate("cand", team="A", db_path=db)
    assert not res["pass"] and any("p99_under_limit" in r for r in res["reasons"])
    # 50/50 -> CI lower bound below 0.5 -> FAIL
    con = store.connect(db)
    even = [{**r, "result0": 0.5} for r in rows]
    store.upsert_rows(con, even); con.close()
    res = gates.evaluate("cand", team="A", db_path=db)
    assert not res["pass"] and any("beats_champion" in r for r in res["reasons"])
    # No champion -> FAIL with explanation
    res = gates.evaluate("cand", team="B", db_path=db)
    assert not res["pass"] and "no champion" in res["reasons"][0]


def test_store_import_merges_by_game_id(tmp_path):
    a = store.connect(tmp_path / "a.sqlite")
    b = store.connect(tmp_path / "b.sqlite")
    row = {**_g("x", "y", 1.0, 3), "game_id": "same", "run_id": "r", "mode": "screen", "steps": 720, "engine": "e",
           "arena_sha": "s", "host": "h", "status0": "DONE", "status1": "DONE", "over_budget0": 0, "over_budget1": 0,
           "error0": "", "error1": "", "harness_error": "", "steps_played": 720, "wall_s": 1.0, "replay_path": "",
           "p50_0": 1, "p50_1": 1}
    store.upsert_rows(b, [row, {**row, "game_id": "other"}])
    store.upsert_rows(a, [{**row, "result0": 0.0}])
    b.close()
    n = store.import_db(a, tmp_path / "b.sqlite")
    assert n == 2
    games = store.all_games(a)
    assert len(games) == 2 and {g["game_id"] for g in games} == {"same", "other"}
    assert next(g for g in games if g["game_id"] == "same")["result0"] == 1.0, "imported row replaces by id"
    a.close()
