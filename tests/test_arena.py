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


# ------------------------------------------------------------------- module isolation
# Regression for the 2026-08-19 arena bug: two directory-agents whose main.py resolved
# siblings with a bare `import policy` were silently played as the SAME policy inside a
# shared worker (first import wins sys.modules; byte-identical banks at --workers 1).

def _write_dir_agent(root: Path, policy_body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "policy.py").write_text(policy_body, encoding="utf-8")
    (root / "main.py").write_text("import policy\n\n\ndef agent(obs):\n    return policy.act(obs)\n",
                                  encoding="utf-8")
    return root


def test_subprocess_agents_do_not_share_modules(tmp_path):
    """Direct proxy check, no engine: colliding `policy` modules stay distinct policies."""
    from arena.isolation import SubprocessAgent
    _write_dir_agent(tmp_path / "alpha", "def act(obs):\n    return {'who': 'alpha', 'step': obs.step}\n")
    _write_dir_agent(tmp_path / "beta", "def act(obs):\n    return {'who': 'beta', 'step': obs.step}\n")
    a = SubprocessAgent(tmp_path / "alpha" / "main.py", name="alpha")
    b = SubprocessAgent(tmp_path / "beta" / "main.py", name="beta")
    try:
        obs = {"step": 7, "remainingOverageTime": 60}
        cfg = {"actTimeout": 1}
        ra, rb = a(obs, cfg), b(obs, cfg)
        ra2 = a(obs, cfg)
    finally:
        a.close()
        b.close()
    assert ra == {"who": "alpha", "step": 7}
    assert rb == {"who": "beta", "step": 7}, "second agent must play its OWN policy module"
    assert ra2 == ra


def test_colliding_directory_agents_play_their_own_policies(tmp_path):
    """Engine-level regression at workers=1 (the proven-corrupt case): a PASS-only agent and
    a HIRE-at-hour-0 agent, both with a sibling module named `policy`, must produce
    different action streams (visible as different banks) on the same fixed seed."""
    _write_dir_agent(tmp_path / "iso_pass",
                     "def act(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n")
    _write_dir_agent(tmp_path / "iso_hire",
                     "def act(obs):\n"
                     "    hands = obs.farms[obs.player]['hands']\n"
                     "    market = [['HIRE']] if obs.hour == 0 else []\n"
                     "    return {'farmer': ['PASS'], 'hands': [['PASS'] for _ in hands], 'market': market}\n")
    registry.register_file(tmp_path / "iso_pass", "iso-pass", "A")
    registry.register_file(tmp_path / "iso_hire", "iso-hire", "B")
    res = scheduler.run_match(["iso-pass", "iso-hire"], mode="screen", n=2, workers=1, steps=48, quiet=True)
    assert len(res["rows"]) == 2
    for r in res["rows"]:
        assert r["status0"] == "DONE" and r["status1"] == "DONE", r
        assert r["crash0"] == r["crash1"] == 0 and r["harness_error"] == "", r
        pass_side = 0 if r["p0"].startswith("iso-pass") else 1
        hire_side = 1 - pass_side
        assert r[f"money{pass_side}"] == 3000.0, "PASS agent never spends"
        assert r[f"money{hire_side}"] < 3000.0, \
            "colliding-module agents were played as the same policy (sys.modules leak)"


def test_registered_id_covers_gitignored_files():
    """2026-08-19 hazard: a clean committed dir got a git-sha id that ignored gitignored
    weights, so re-registering changed weights silently resolved to the old snapshot."""
    src = REPO / "agent"                      # committed directory agent (the PASS placeholder)
    marker = src / "zz_test_weights.npz"      # *.npz is gitignored
    try:
        marker.write_bytes(b"weights-v1")
        e1 = registry.register_file(src, "idhaz", "A")
        assert e1["sha"].startswith("h"), "a snapshot holding gitignored files must not use the git sha"
        assert (ARENA_DIR / "agents" / e1["id"] / "zz_test_weights.npz").read_bytes() == b"weights-v1"
        marker.write_bytes(b"weights-v2")
        e2 = registry.register_file(src, "idhaz", "A")
        assert e2["id"] != e1["id"], "changed weight bytes must yield a new id"
        assert (ARENA_DIR / "agents" / e2["id"] / "zz_test_weights.npz").read_bytes() == b"weights-v2"
    finally:
        marker.unlink(missing_ok=True)


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


# ----------------------------------------------------------------------------- hub + serve
def test_hub_plan_round_prefers_least_played_active():
    from arena import hub
    registry.ensure_residents()
    a = registry.register_file(TRIVIAL, "hub-a", "A")
    registry.register_file(REPO / "agents" / "team_a" / "v0a.py", "hub-b", "B")
    # bench the built-ins so the plan is deterministic over our two agents
    for b in ("random", "starter", "pass", "pass-file", "boom", "cand", "champ"):
        try:
            registry.set_flag(b, "active", False)
        except KeyError:
            pass
    plan = hub.plan_round()
    assert plan is not None
    ids = {plan["anchor"]["id"]} | {o["id"] for o in plan["opponents"]}
    assert a["id"] in ids and len(ids) >= 2
    assert plan["next_seed"] >= hub.LADDER_SEED_BASE
    # pausing is a durable flag file
    assert hub.set_paused(True) is True
    assert hub.is_paused()
    assert hub.set_paused(False) is False
    cfg = hub.set_config(workers=3, games_per_pairing=6)
    assert cfg["workers"] == 3 and cfg["games_per_pairing"] == 6


def test_registry_flags_bench_and_retire():
    a = registry.register_file(TRIVIAL, "flaggy", "B")
    assert registry.is_active(registry.resolve("flaggy"))
    registry.set_flag("flaggy", "active", False)
    assert not registry.is_active(registry.resolve("flaggy"))
    registry.set_flag("flaggy", "active", True)
    registry.set_retired("flaggy", True)
    got = registry.resolve("flaggy")
    assert got.get("retired") and not registry.is_active(got)
    registry.set_retired("flaggy", False)
    assert not registry.resolve("flaggy").get("retired")
    assert a["id"] == registry.resolve("flaggy")["id"]


def test_serve_api_endpoints_and_controls():
    import json as _json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from arena import serve as srv
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        def get(path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
                return _json.loads(r.read())

        def post(path, body):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                         data=_json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return _json.loads(r.read())

        assert get("/health") == {"ok": True}
        lb = get("/api/leaderboard")
        assert "table" in lb and "h2h" in lb and "champions" in lb
        ags = get("/api/agents")
        assert any(x["name"] == "random" for x in ags["agents"])
        st = get("/api/status")
        assert st["engine"] and "hub" in st
        assert isinstance(get("/api/games?limit=5"), list)
        assert "live" in get("/api/runs")
        r = post("/api/control/bench", {"agent": "flaggy"})
        assert r["agent"]["active"] is False
        r = post("/api/control/bench", {"agent": "flaggy", "undo": True})
        assert r["agent"]["active"] is True
        r = post("/api/control/pause", {})
        assert r["paused"] is True
        r = post("/api/control/resume", {})
        assert r["paused"] is False
        with pytest.raises(urllib.error.HTTPError):
            post("/api/control/bench", {"agent": "no-such-agent"})
    finally:
        httpd.shutdown()
        httpd.server_close()


# ----------------------------------------------------------------- auditor fixes (2026-08-19)
def test_gate_counts_only_confirm_mode_games(tmp_path):
    """Ladder/screen rows sitting on confirm seeds must not count as gate evidence."""
    db = tmp_path / "m.sqlite"
    con = store.connect(db)
    champ = registry.resolve("champ")
    cand = registry.resolve("cand")
    registry.set_champion("A", champ["id"])
    base = {"run_id": "t", "steps": 720, "engine": "1.32.7", "arena_sha": "x", "host": "t",
            "status0": "DONE", "status1": "DONE", "over_budget0": 0, "over_budget1": 0, "error0": "", "error1": "",
            "harness_error": "", "steps_played": 720, "wall_s": 1.0, "replay_path": "", "p50_0": 1, "p50_1": 1}
    rows = []
    for k in range(10):
        seed = 1000 + k
        for mode in ("confirm", "ladder"):
            gid = scheduler.game_id(cand["id"], champ["id"], seed, "1.32.7") + ("L" if mode == "ladder" else "")
            rows.append({**base, **_g(cand["id"], champ["id"], 1.0, seed), "mode": mode, "game_id": gid})
    store.upsert_rows(con, rows)
    con.close()
    res = gates.evaluate("cand", team="A", db_path=db, min_games=10, seed_lo=1000, seed_hi=1199)
    assert res["n"] == 10, "only the mode=confirm rows count"
    assert res["excluded_non_confirm"] == 10
    assert "10 non-confirm games on these seeds excluded" in res["checks"]["enough_games"]["detail"]


def test_single_ladder_implementation_on_the_2000_band():
    """The hub is the only ladder (Auditor blocker 1): the CLI mode is gone, the seed band is
    shared from paths.LADDER_SEED_BASE and sits strictly above the confirm band."""
    from arena import hub
    from arena.paths import CONFIRM_N, CONFIRM_SEED_BASE, LADDER_SEED_BASE
    assert LADDER_SEED_BASE > CONFIRM_SEED_BASE + CONFIRM_N // 2 - 1
    assert hub.LADDER_SEED_BASE is LADDER_SEED_BASE
    assert not hasattr(scheduler, "run_ladder")
    with pytest.raises(ValueError, match="hub"):
        scheduler.run_match(["random", "pass"], mode="ladder", n=2, quiet=True)
    plan = hub.plan_round()
    assert plan is None or plan["next_seed"] >= LADDER_SEED_BASE


def test_control_register_is_locked_down(tmp_path):
    """Auditor blocker 2: API registration only accepts agents/ paths at a git ref, and the
    result starts benched; raw path / code_b64 need the default-off unsafe flag."""
    import base64 as b64
    from arena import serve as srv
    with pytest.raises(ValueError, match="agents/"):
        srv.control("register", {"name": "evil", "team": "A", "git_ref": "main", "path": "scripts/evaluate.py"})
    with pytest.raises(ValueError, match="agents/"):
        srv.control("register", {"name": "evil", "team": "A", "git_ref": "main", "path": "agents/../scripts/x.py"})
    with pytest.raises(ValueError, match="only git_ref"):
        srv.control("register", {"name": "evil", "team": "A", "path": str(tmp_path / "evil.py")})
    with pytest.raises(ValueError, match="only git_ref"):
        srv.control("register", {"name": "evil", "team": "A",
                                 "code_b64": b64.b64encode(b"import os\n").decode()})
    # with the unsafe flag ON (local dev), path registration works but still starts benched
    from arena import hub
    hub.HUB_CONFIG.write_text('{"allow_unsafe_register": true}', encoding="utf-8")
    try:
        f = tmp_path / "ok.py"
        f.write_text("def agent(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n", encoding="utf-8")
        out = srv.control("register", {"name": "api-reg", "team": "B", "path": str(f)})
        assert out["agent"]["active"] is False, "API registrations start benched"
        assert not registry.is_active(registry.resolve("api-reg"))
    finally:
        hub.HUB_CONFIG.unlink(missing_ok=True)
