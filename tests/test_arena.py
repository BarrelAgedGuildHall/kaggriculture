"""Arena tests: registry round-trip, a tiny screen run, ratings on synthetic results, gate logic.

The arena state is redirected to a temp dir through KAGGRI_ARENA_DIR (read at import time by
arena.paths), so these tests never touch data/arena/. Run: uv run pytest -q tests/test_arena.py
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import tempfile
import time
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


def test_crashing_directory_agent_leaks_no_child_processes(tmp_path, monkeypatch):
    """2026-08-19 audit finding 1, engine level: after a multi-step game against a
    permanently-broken agent (import-time failure -> `_kill()` on every call that reaches
    the child), every child ever spawned must have been reaped -- an unreaped kill leaves
    zombie processes on the hub host."""
    import subprocess as _subprocess

    from arena import runner

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "main.py").write_text(
        "import totally_missing_module\n\n\ndef agent(obs):\n    return {}\n", encoding="utf-8")
    entry = registry.register_file(broken, "broken-import", "B")
    passer = registry.resolve("pass")

    spawned: list = []
    real_popen = _subprocess.Popen

    class RecordingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(_subprocess, "Popen", RecordingPopen)
    job = {"game_id": "reap-test", "p0": entry, "p1": passer, "seed": 0, "steps": 24,
           "keep_replay": "none", "mode": "screen", "run_id": "reap-test"}
    row = runner.play_game(job)

    # Snapshot returncodes BEFORE any poll/wait of our own -- poll() reaps, hiding the bug.
    unreaped = [p for p in spawned if p.returncode is None]
    for p in unreaped:   # never leak real zombies out of the test itself
        try:
            p.kill()
            p.wait(timeout=3)
        except Exception:
            pass
    assert row["harness_error"] == "", row
    side = 0 if row["p0"].startswith("broken-import") else 1
    assert row[f"crash{side}"] == 1 and row["result0"] == (0.0 if side == 0 else 1.0), \
        "a permanently-broken agent is scored a crash-loss"
    assert len(spawned) >= 1, "the broken agent must actually have spawned a child"
    assert not unreaped, \
        f"{len(unreaped)}/{len(spawned)} killed children were never wait()ed (zombie leak)"


def test_kill_reaps_live_child(tmp_path):
    """2026-08-19 audit finding 1, the deterministic core: `_kill()` on a LIVE child must
    reap it (kill + wait), like `close()` does. Pre-fix it only killed -- on POSIX a
    killed-but-unwaited child is a zombie until the parent (the hub process) exits, and
    `__call__` respawns a fresh child on every subsequent call, so the leak is unbounded."""
    from arena.isolation import SubprocessAgent

    _write_dir_agent(tmp_path / "alive", "def act(obs):\n    return {'ok': obs.step}\n")
    proxy = SubprocessAgent(tmp_path / "alive" / "main.py", name="alive")
    try:
        assert proxy({"step": 1, "remainingOverageTime": 60}, {"actTimeout": 1}) == {"ok": 1}
        child = proxy.proc
        assert child is not None and child.poll() is None, "child must be alive before _kill"
        proxy._kill()
        assert proxy.proc is None
        assert child.returncode is not None, \
            "_kill() left the child un-reaped (zombie on POSIX); it must wait() like close()"
    finally:
        proxy.close()


def test_stdlib_shadowing_matches_real_loader(tmp_path):
    """2026-08-19 audit finding 2: the child must sys.path.APPEND the agent dir like the real
    loader (kaggle_environments.agent.get_last_callable), so a sibling module named after a
    stdlib module (here `string`) is shadowed BY the stdlib in both loaders. insert(0) made
    the sibling win in the Pit only -- silent Pit-vs-Kaggle behaviour drift."""
    import arena.engine as eng
    from arena.isolation import SubprocessAgent

    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "string.py").write_text("MARKER = 'FAKE_STRING_MODULE'\n", encoding="utf-8")
    (shadow / "main.py").write_text(
        "import string\n\n\ndef agent(obs, config):\n"
        "    return {'marker': getattr(string, 'MARKER', 'REAL_STDLIB')}\n", encoding="utf-8")

    eng.import_engine()
    from kaggle_environments.agent import get_last_callable
    raw = (shadow / "main.py").read_text(encoding="utf-8")
    real_agent = get_last_callable(raw, path=str(shadow / "main.py"))
    real_result = real_agent({"step": 0}, {})

    proxy = SubprocessAgent(shadow / "main.py", name="shadow")
    try:
        pit_result = proxy({"step": 0, "remainingOverageTime": 60}, {"actTimeout": 1})
    finally:
        proxy.close()

    assert real_result == {"marker": "REAL_STDLIB"}, "the real loader shadows agent siblings with the stdlib"
    assert pit_result == real_result, \
        "Pit loader resolves a stdlib-shadowing sibling import differently from the real loader"


def test_zero_arg_agent_matches_real_loader_trim(tmp_path):
    """2026-08-19 audit finding 3: the real loader trims args to co_argcount with NO floor
    (args[:co_argcount]), so a legal zero-argument agent is called with (). The old `or 2`
    clobbered co_argcount == 0 into a TypeError crash-loss. Two-arg agents keep (obs, config)."""
    from arena.isolation import SubprocessAgent

    zero = tmp_path / "zero"
    zero.mkdir()
    (zero / "main.py").write_text(
        "def agent():\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n", encoding="utf-8")
    two = tmp_path / "two"
    two.mkdir()
    (two / "main.py").write_text(
        "def agent(obs, config):\n    return {'step': obs.step, 'act': config.actTimeout}\n",
        encoding="utf-8")

    obs = {"step": 3, "remainingOverageTime": 60}
    cfg = {"actTimeout": 1}
    z = SubprocessAgent(zero / "main.py", name="zero")
    t = SubprocessAgent(two / "main.py", name="two")
    try:
        za = z(obs, cfg)
        ta = t(obs, cfg)
    finally:
        z.close()
        t.close()
    assert za == {"farmer": ["PASS"], "hands": [], "market": []}, \
        "a zero-arg agent is legal and must be called with zero args, not (obs, config)"
    assert ta == {"step": 3, "act": 1}


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
    stamp = hub.PAUSE_FLAG.read_text(encoding="utf-8")
    assert hub.set_paused(True) is True
    assert hub.PAUSE_FLAG.read_text(encoding="utf-8") == stamp, \
        "re-pausing keeps the original 'paused since' stamp (it is evidence for a contaminated arm)"
    assert hub.set_paused(False) is False
    cfg = hub.set_config(workers=3, games_per_pairing=6)
    assert cfg["workers"] == 3 and cfg["games_per_pairing"] == 6


# ------------------------------------------------------------------- pause leases (2026-08-21)
# Defect: the hub ladder resumed underneath a live manual evaluation (an explicit
# /api/control/resume by another actor while a team measured timing), contaminating the arms.

def test_lease_blocks_resume_until_released():
    from arena import hub, leases
    lease = hub.acquire_lease("engineer-b", "expB-303 gate battery", ttl_s=600)
    try:
        assert hub.is_paused(), "acquiring a lease pauses the ladder"
        assert [l["id"] for l in leases.active()] == [lease["id"]]
        with pytest.raises(leases.LeaseHeld) as e:
            hub.set_paused(False)
        assert lease["id"] in str(e.value) and "engineer-b" in str(e.value)
        assert "expB-303 gate battery" in str(e.value), "the refusal names who is running what"
        assert hub.is_paused(), "a refused resume must not have touched the flag"
    finally:
        hub.release_lease(lease["id"], resume=True)
    assert not leases.active()
    assert not hub.is_paused(), "releasing the last lease restores normal resume behaviour"


def test_lease_lapses_after_its_ttl(monkeypatch):
    from arena import hub, leases
    lease = leases.acquire("modeler", "exp-N1 traces", ttl_s=leases.DEFAULT_TTL_S)
    hub.set_paused(True)
    try:
        with pytest.raises(leases.LeaseHeld):
            hub.set_paused(False)
        monkeypatch.setattr(leases, "_epoch", lambda: lease["acquired_at"] + leases.DEFAULT_TTL_S + 1)
        assert leases.active() == [], "an abandoned lease lapses; it cannot wedge the ladder forever"
        assert hub.set_paused(False) is False
        with pytest.raises(KeyError):
            leases.renew(lease["id"])          # expired leases are not renewable
    finally:
        monkeypatch.undo()
        for l in leases.active():
            leases.release(l["id"])
        hub.set_paused(False)


def test_lease_renew_extends_from_now(monkeypatch):
    from arena import leases
    lease = leases.acquire("ops", "ingestion", ttl_s=600)
    try:
        t1 = lease["acquired_at"] + 300
        monkeypatch.setattr(leases, "_epoch", lambda: t1)
        renewed = leases.renew(lease["id"], ttl_s=600)
        assert renewed["renewals"] == 1
        assert renewed["expires_at"] == pytest.approx(t1 + 600)
        assert renewed["expires_in_s"] == 600, "renew extends from now, not from the original grant"
    finally:
        monkeypatch.undo()
        leases.release(lease["id"])


def test_force_resume_breaks_the_lease_and_is_logged():
    from arena import hub, leases
    lease = hub.acquire_lease("architect-b", "expB-306", ttl_s=600)
    try:
        with pytest.raises(ValueError, match="actor"):
            hub.set_paused(False, force=True)       # an override must say who is overriding
        assert leases.active(), "a rejected force must not break anything"
        assert hub.set_paused(False, force=True, actor="ceo") is False
        assert leases.active() == [], "force resume releases the leases it overrides"
    finally:
        for l in leases.active():
            leases.release(l["id"])
    log = leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()
    events = [json.loads(ln) for ln in log]
    assert any(e["event"] == "force-break" and e["actor"] == "ceo" and lease["id"] in e["lease_ids"]
               for e in events), "every override is on the audit log"
    assert any(e["event"] == "acquire" and e["lease_id"] == lease["id"] for e in events)


def test_lease_requires_owner_and_purpose():
    from arena import leases
    with pytest.raises(ValueError, match="owner"):
        leases.acquire("", "something")
    with pytest.raises(ValueError, match="purpose"):
        leases.acquire("engineer", "  ")
    assert leases.active() == []


# The two papercuts the first real users hit (2026-08-21), fixed in serve.py:
#  - `acquire` returns the id under "id", so callers echo "id" back at renew/release, which
#    documented only "lease_id";
#  - a client that read `leases` at one level of /api/status and found nothing concluded the
#    ladder was free while resume was in fact being refused.

def test_lease_api_accepts_either_id_key():
    from arena import hub, leases, serve
    rec = serve.control("lease", {"op": "acquire", "owner": "ops", "purpose": "ingestion",
                                  "ttl_s": 600})["lease"]
    try:
        renewed = serve.control("lease", {"op": "renew", "id": rec["id"], "ttl_s": 600})["lease"]
        assert renewed["id"] == rec["id"], "the key `acquire` hands back must work on the way in"
        with pytest.raises(ValueError, match="lease_id"):
            serve.control("lease", {"op": "release"})
        out = serve.control("lease", {"op": "release", "id": rec["id"]})
        assert out["lease"]["id"] == rec["id"] and out["leases"] == []
    finally:
        for l in leases.active():
            leases.release(l["id"])
        hub.set_paused(False)


def test_status_reports_leases_at_both_levels():
    from arena import hub, leases, serve
    assert serve.api_status()["lease_held"] is False
    lease = hub.acquire_lease("modeler", "exp-012b screen", ttl_s=600)
    try:
        st = serve.api_status()
        assert st["lease_held"] is True
        assert [l["id"] for l in st["leases"]] == [lease["id"]], "top level carries the leases"
        assert [l["id"] for l in st["hub"]["leases"]] == [lease["id"]], "so does hub state"
        assert serve.api_leases()["lease_held"] is True
    finally:
        hub.release_lease(lease["id"], resume=True)
    assert serve.api_status()["lease_held"] is False


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

        # pause lease over the API: acquire -> resume refused (409) -> release -> resume works
        r = post("/api/control/lease", {"op": "acquire", "owner": "engineer", "purpose": "timing arm",
                                        "ttl_s": 600})
        lease_id = r["lease"]["id"]
        try:
            assert r["paused"] is True and len(r["leases"]) == 1
            assert get("/api/leases")["leases"][0]["id"] == lease_id
            assert get("/api/status")["hub"]["leases"][0]["owner"] == "engineer"
            with pytest.raises(urllib.error.HTTPError) as e:
                post("/api/control/resume", {})
            assert e.value.code == 409
            payload = _json.loads(e.value.read())
            assert payload["leases"][0]["id"] == lease_id and "timing arm" in payload["error"]
            assert get("/api/hub")["paused"] is True, "the refused resume left the ladder paused"
            r = post("/api/control/lease", {"op": "renew", "lease_id": lease_id, "ttl_s": 900})
            assert r["lease"]["renewals"] == 1
        finally:
            r = post("/api/control/lease", {"op": "release", "lease_id": lease_id})
        assert r["leases"] == []
        r = post("/api/control/resume", {})
        assert r["paused"] is False
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


# ------------------------------------------------- lease hardening (Auditor review of PR #19)
def test_control_rejects_a_lease_body_with_no_op():
    """F1: `op` used to default to "list", so a mis-keyed or missing op turned a mutation into a
    read that answered 200 with an empty lease list -- indistinguishable from "the ladder is free"."""
    from arena import serve as srv
    with pytest.raises(ValueError, match="unknown request key"):
        srv.control("lease", {"action": "acquire", "owner": "someone", "purpose": "x"})
    with pytest.raises(ValueError, match='lease needs an "op"'):
        srv.control("lease", {})
    with pytest.raises(ValueError, match='lease needs an "op"'):
        srv.control("lease", {"owner": "someone", "purpose": "x", "ttl_s": 60})
    with pytest.raises(ValueError, match="unknown lease op"):
        srv.control("lease", {"op": "aquire"})


def test_control_rejects_unknown_body_keys():
    from arena import serve as srv
    with pytest.raises(ValueError, match="unknown request key"):
        srv.control("resume", {"forced": True})
    with pytest.raises(ValueError, match="unknown request key"):
        srv.control("lease", {"op": "acquire", "owner": "o", "purpose": "p", "ttl": 60})
    with pytest.raises(ValueError, match="unknown request key"):
        srv.control("champion", {"team": "A", "agent": "x", "force": True})
    with pytest.raises(ValueError, match="unknown control action"):
        srv.control("resune", {})


def test_control_still_accepts_the_documented_bodies():
    from arena import hub, serve as srv
    try:
        out = srv.control("lease", {"op": "acquire", "owner": "engineer", "purpose": "hardening test",
                                    "ttl_s": 60})
        lid = out["lease"]["id"]
        assert srv.control("lease", {"op": "list"})["lease_held"] is True
        assert srv.control("lease", {"op": "renew", "id": lid, "ttl_s": 90})["lease"]["ttl_s"] == 90
        srv.control("lease", {"op": "release", "lease_id": lid, "resume": True})
        assert srv.control("lease", {"op": "list"})["lease_held"] is False
    finally:
        for l in leases_mod().active():
            leases_mod().release(l["id"])
        hub_mod().set_paused(False)


def leases_mod():
    from arena import leases
    return leases


def hub_mod():
    from arena import hub
    return hub


def test_unreadable_lease_state_reads_as_held_and_refuses_resume():
    """F2: `except Exception: state = {}` turned a truncated leases.json into "no leases held" --
    resume went from 409 to 200 with no log row. Unknown state is held state now."""
    from arena import hub, leases
    lease = hub.acquire_lease("engineer", "F2 regression", ttl_s=600)
    try:
        leases.LEASES_PATH.write_text('{"leases": {"x": ', encoding="utf-8")   # truncated mid-write
        st = leases.status()
        assert st["lease_held"] is True and st["leases"] == []
        assert "unreadable" in st["lease_state_error"]
        with pytest.raises(leases.LeaseStateUnreadable):
            leases.active()
        with pytest.raises(leases.LeaseHeld):        # LeaseStateUnreadable IS a LeaseHeld -> 409
            hub.set_paused(False)
        assert hub.is_paused(), "a refused resume must not have touched the flag"
        # the API reports it at both levels rather than answering "free"
        from arena import serve as srv
        assert srv.api_leases()["lease_held"] is True
        assert srv.api_status()["lease_state_error"]
        assert srv.api_status()["hub"]["lease_held"] is True
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        assert any(e["event"] == "state-unreadable" for e in events), "and it says so in the log"
    finally:
        leases.LEASES_PATH.unlink(missing_ok=True)
        for l in leases.active():
            leases.release(l["id"])
        hub.set_paused(False)
    assert lease["id"]


def test_force_resume_quarantines_an_unreadable_lease_file():
    """The one way past: deliberate, attributed, logged -- and the bad file is kept, renamed. It is
    never deleted, because "no file" would be back to "no leases held"."""
    from arena import hub, leases
    hub.set_paused(True)
    leases.LEASES_PATH.write_text("not json at all", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="actor"):
            hub.set_paused(False, force=True, actor="")      # an override must name someone
        assert hub.is_paused()
        assert hub.set_paused(False, force=True, actor="ceo") is False
        assert not leases.LEASES_PATH.exists()
        moved = list(leases.LEASES_PATH.parent.glob(leases.LEASES_PATH.name + ".corrupt-*"))
        assert moved and moved[0].read_text(encoding="utf-8") == "not json at all"
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        assert any(e["event"] == "state-quarantined" and e["actor"] == "ceo" for e in events)
    finally:
        for f in leases.LEASES_PATH.parent.glob(leases.LEASES_PATH.name + ".corrupt-*"):
            f.unlink()
        leases.LEASES_PATH.unlink(missing_ok=True)
        hub.set_paused(False)


def test_lease_writes_are_serialised_and_use_unique_temp_files():
    """F3: read-modify-write with a fixed .tmp name under a threading server. 12 concurrent
    acquires must all survive.

    Written with `shared=True` since exclusivity landed -- the point of the test is the *write*
    storm (no lost update, no stray temp file), not how many holders the policy allows. The
    exclusivity policy has its own tests below.
    """
    import threading as th
    from arena import leases
    errors, ids = [], []

    def take(i):
        try:
            ids.append(leases.acquire(f"owner{i}", f"purpose {i}", ttl_s=120, shared=True)["id"])
        except Exception as exc:               # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [th.Thread(target=take, args=(i,)) for i in range(12)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, errors
        assert len(ids) == 12
        assert {l["id"] for l in leases.active()} == set(ids), "no acquire may be lost to a race"
        assert not list(leases.LEASES_PATH.parent.glob(leases.LEASES_PATH.name + ".tmp*")), \
            "temp files are cleaned up and uniquely named"
        assert not leases.LOCK_PATH.exists(), "the cross-process lock is always released"
    finally:
        for l in leases.active():
            leases.release(l["id"])


def test_hub_reconciles_a_lease_held_over_an_unpaused_ladder():
    """F4: the loop only checked the pause flag, so a lease could be live while the ladder played.
    A live lease means the ladder must not play -- the hub re-asserts the pause and logs it."""
    from arena import hub, leases
    lease = leases.acquire("modeler", "F4 regression", ttl_s=120)   # no pause taken with it
    hub.PAUSE_FLAG.unlink(missing_ok=True)
    try:
        assert not hub.is_paused()
        assert hub.reconcile_pause() is True
        assert hub.reconcile_pause() is False, "already reconciled: no second pause, no second log row"
        assert hub.is_paused(), "a held lease re-asserts the pause instead of dealing games under it"
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        rec = [e for e in events if e["event"] == "reconciled-pause"]
        assert rec and lease["id"] in rec[-1]["lease_ids"]
    finally:
        leases.release(lease["id"])
        hub.set_paused(False)


# ------------------------------------------------- deploy identity (F7)
def test_arena_code_hash_is_stable_and_line_ending_independent(tmp_path, monkeypatch):
    """arena_sha can be a value declared in a systemd unit at deploy time and never updated (it
    read 12a5398 while the hub ran code well past it), so `is the deploy live?` needs an answer
    computed from the files themselves."""
    from arena import engine as eng2
    h1 = eng2.arena_code_hash()
    assert len(h1) == 12 and h1 == eng2.arena_code_hash()

    pkg = tmp_path / "arena"
    pkg.mkdir()
    (pkg / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (pkg / "b.py").write_bytes(b"z = 3\n")
    monkeypatch.setattr(eng2, "__file__", str(pkg / "engine.py"))
    monkeypatch.setitem(eng2._CODE_HASH, "at", 0.0)
    crlf = eng2.arena_code_hash()
    (pkg / "a.py").write_bytes(b"x = 1\ny = 2\n")
    monkeypatch.setitem(eng2._CODE_HASH, "at", 0.0)
    assert eng2.arena_code_hash() == crlf, "CRLF vs LF must not change the deploy identity"
    (pkg / "b.py").write_bytes(b"z = 4\n")
    monkeypatch.setitem(eng2._CODE_HASH, "at", 0.0)
    assert eng2.arena_code_hash() != crlf, "a real code change must change it"


def test_arena_sha_source_says_where_the_sha_came_from(monkeypatch):
    from arena import engine as eng2
    monkeypatch.setenv("KAGGRI_ARENA_SHA", "12a5398")
    eng2.arena_git_sha.cache_clear()
    try:
        assert eng2.arena_git_sha() == "12a5398"
        assert eng2.arena_sha_source() == "env", "a declared sha is flagged as declared"
    finally:
        monkeypatch.delenv("KAGGRI_ARENA_SHA")
        eng2.arena_git_sha.cache_clear()


def test_status_reports_both_identities():
    from arena import serve as srv
    st = srv.api_status()
    assert st["arena_code_hash"] and st["arena_sha_source"] in ("env", "git", "none")
# ----------------------------------------------------------------------------- seed bands
def test_seed_bands_are_declared_and_resolvable():
    """The confirm band used to be a module constant, so every gate in the org's history
    inherited the same 200 seeds. It is a named, overridable band now (and the defaults are
    unchanged: screen 0-199, confirm 1000-1199, ladder 2000+)."""
    from arena.paths import SEED_BANDS, band_of, band_seeds, bands_of, resolve_band
    assert SEED_BANDS["screen"] == (0, 199)
    assert SEED_BANDS["confirm"] == (1000, 1199)
    assert SEED_BANDS["ladder"] == (2000, None)
    assert resolve_band("confirm") == (1000, 1199)
    assert resolve_band("3000-3199") == (3000, 3199)
    assert resolve_band("5000-") == (5000, None)
    with pytest.raises(ValueError, match="unknown seed band"):
        resolve_band("nope")
    assert band_seeds("confirm", 3) == [1000, 1001, 1002]
    assert band_of(0) == "screen" and band_of(1100) == "confirm" and band_of(9999) == "ladder"
    assert band_of(500) is None
    assert bands_of([1, 1100, 1101]) == ["screen", "confirm"]
    assert bands_of([500]) == ["unbanded"]


def test_band_seeds_refuses_to_run_off_the_end_of_a_band():
    """Silently borrowing seeds from the next band is how a held-out set stops being held out."""
    from arena.paths import band_seeds
    assert len(band_seeds("confirm", 200)) == 200
    with pytest.raises(ValueError, match="holds 200 seeds"):
        band_seeds("confirm", 201)


def test_seed_bands_can_be_overridden_by_env(monkeypatch):
    import importlib

    import arena.paths as p
    monkeypatch.setenv("KAGGRI_SEED_BANDS", "confirm=3000-3199,confirmB=4000-4199")
    try:
        importlib.reload(p)
        assert p.SEED_BANDS["confirm"] == (3000, 3199)
        assert p.SEED_BANDS["confirmB"] == (4000, 4199)
        assert p.CONFIRM_SEED_BASE == 3000
        assert p.SEED_BANDS["screen"] == (0, 199), "unnamed bands keep their defaults"
    finally:
        monkeypatch.delenv("KAGGRI_SEED_BANDS")
        importlib.reload(p)
        assert p.SEED_BANDS["confirm"] == (1000, 1199)


def test_parse_seeds_accepts_any_declared_band():
    assert scheduler.parse_seeds("ladder", "screen", 6) == [2000, 2001, 2002]
    assert scheduler.parse_seeds("confirm", "screen", 4) == [1000, 1001]


def test_gate_states_the_band_it_ran_on(tmp_path):
    """Every gate result names its band -- a pooled claim is a claim about which bands."""
    db = tmp_path / "band.sqlite"
    con = store.connect(db)
    champ = registry.register_file(TRIVIAL, "band-champ", "B")
    f = tmp_path / "bandcand.py"
    f.write_text("def agent(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n# band\n",
                 encoding="utf-8")
    cand = registry.register_file(f, "band-cand", "B")
    registry.set_champion("B", champ["id"])
    base = {"run_id": "b", "mode": "confirm", "steps": 720, "engine": "1.32.7", "arena_sha": "x", "host": "t",
            "status0": "DONE", "status1": "DONE", "over_budget0": 0, "over_budget1": 0, "error0": "", "error1": "",
            "harness_error": "", "steps_played": 720, "wall_s": 1.0, "replay_path": "", "p50_0": 1, "p50_1": 1}
    rows = []
    for seed in list(range(1000, 1005)) + list(range(3000, 3005)):
        rows.append({**base, **_g(cand["id"], champ["id"], 1.0, seed),
                     "game_id": scheduler.game_id(cand["id"], champ["id"], seed, "1.32.7")})
    store.upsert_rows(con, rows)
    con.close()
    res = gates.evaluate("band-cand", team="B", db_path=db, min_games=5)
    assert res["band"] == "confirm" and (res["seed_lo"], res["seed_hi"]) == (1000, 1199)
    assert res["n"] == 5, "the default band must not silently reach into another band"
    assert "band confirm" in res["checks"]["enough_games"]["detail"]
    assert "band confirm" in gates.format_result(res)
    other = gates.evaluate("band-cand", team="B", db_path=db, band="3000-3199", min_games=5)
    assert other["n"] == 5 and other["band"] == "3000-3199"
    assert other["pass"] and res["pass"], "the same agent, two disjoint bands -- what pooling needs"


# ----------------------------------------------------------------------------- NB-6 integrity
def _row(run_id, seed=7, result0=1.0, **over):
    r = {**_g("p0@1", "p1@1", result0, seed), "game_id": scheduler.game_id("p0@1", "p1@1", seed, "1.32.7"),
         "run_id": run_id, "mode": "confirm", "steps": 720, "engine": "1.32.7", "arena_sha": "s", "host": "h",
         "status0": "DONE", "status1": "DONE", "over_budget0": 0, "over_budget1": 0, "error0": "", "error1": "",
         "harness_error": "", "steps_played": 720, "wall_s": 1.0, "replay_path": "", "p50_0": 1, "p50_1": 1}
    r.update(over)
    return r


def test_rerunning_a_pairing_keeps_the_measurement_it_replaces(tmp_path):
    """NB-6: game_id is sha1(p0|p1|seed|engine) and rows upsert, so a re-run silently destroyed the
    earlier measurement. The upsert stays (it is what makes remote deltas merge); what it replaces
    is now kept in the `superseded` ledger."""
    con = store.connect(tmp_path / "nb6.sqlite")
    store.upsert_rows(con, [_row("run-A", result0=1.0, money0=500.0)])
    store.upsert_rows(con, [_row("run-B", result0=0.0, money0=100.0)])
    games = store.all_games(con)
    assert len(games) == 1 and games[0]["run_id"] == "run-B", "the upsert semantics are unchanged"
    old = store.superseded_rows(con)
    assert len(old) == 1
    assert old[0]["run_id"] == "run-A" and old[0]["superseded_by_run"] == "run-B"
    assert old[0]["result0"] == 1.0 and old[0]["money0"] == 500.0
    con.close()


def test_same_run_rewrites_are_not_supersessions(tmp_path):
    con = store.connect(tmp_path / "same.sqlite")
    store.upsert_rows(con, [_row("run-A")])
    store.upsert_rows(con, [_row("run-A", result0=0.5)])   # progress rewrite / re-import
    assert store.superseded_rows(con) == []
    con.close()


def test_integrity_report_finds_the_orphaned_run(tmp_path):
    """The 2026-08-21 case: a run row claiming n_games=400 with zero games left in the store."""
    con = store.connect(tmp_path / "orphan.sqlite")
    store.upsert_rows(con, [_row("run-A", seed=s) for s in range(4)])
    store.record_run(con, {"run_id": "run-A", "mode": "confirm", "agents": "a,b", "n_games": 4,
                           "seeds": "0-3", "engine": "1.32.7", "arena_sha": "s", "host": "h",
                           "workers": 2, "wall_s": 1.0, "started_at": "2026-08-21T08:02:54",
                           "finished_at": "2026-08-21T08:12:00"})
    rep = store.integrity_report(con)
    assert rep["ok"] and rep["incomplete_runs"] == []

    store.upsert_rows(con, [_row("run-B", seed=s) for s in range(4)])
    store.record_run(con, {"run_id": "run-B", "mode": "confirm", "agents": "a,b", "n_games": 4,
                           "seeds": "0-3", "engine": "1.32.7", "arena_sha": "s", "host": "h",
                           "workers": 2, "wall_s": 1.0, "started_at": "2026-08-21T09:00:00",
                           "finished_at": "2026-08-21T09:10:00"})
    rep = store.integrity_report(con)
    assert not rep["ok"]
    bad = rep["incomplete_runs"]
    assert len(bad) == 1 and bad[0]["run_id"] == "run-A"
    assert bad[0]["n_games"] == 4 and bad[0]["stored"] == 0 and bad[0]["missing"] == 4
    assert bad[0]["superseded_by"] == {"run-B": 4}
    assert rep["superseded_total"] == 4
    assert len(store.superseded_rows(con, run_id="run-A")) == 4
    con.close()


def test_integrity_report_flags_games_with_no_run_row(tmp_path):
    con = store.connect(tmp_path / "noRun.sqlite")
    store.upsert_rows(con, [_row("run-ghost")])
    rep = store.integrity_report(con)
    assert rep["orphan_games"] == {"run-ghost": 1} and not rep["ok"]
    con.close()


def test_import_records_supersession_across_stores(tmp_path):
    """A node run superseding a noisy local one is intended behaviour -- it just has to leave a
    trace, so a contaminated arm can be told apart from a clean re-run afterwards."""
    local = store.connect(tmp_path / "local.sqlite")
    node = store.connect(tmp_path / "node.sqlite")
    store.upsert_rows(local, [_row("local-run", result0=0.0, p99_0=900.0)])
    store.upsert_rows(node, [_row("node-run", result0=1.0, p99_0=12.0)])
    node.close()
    store.import_db(local, tmp_path / "node.sqlite")
    assert store.all_games(local)[0]["p99_0"] == 12.0
    old = store.superseded_rows(local)
    assert len(old) == 1 and old[0]["run_id"] == "local-run" and old[0]["p99_0"] == 900.0
    local.close()


# ------------------------------------------- lease exclusivity (incident 2026-08-21 21:08-22:00)
# engineer-b held a lease for the expB-304a gate battery; modeler took a SECOND one 15 minutes
# later and started its own battery on the same 16-core host. One arm ran at load 36.67 and
# recorded 16 spurious timeouts, p99 68.7 ms and a 2647 ms worst turn for an agent that measures
# p99 7.2 ms quiesced -- and an over-budget turn is scored a LOSS regardless of bank. The lease
# API allowed it by design: a lease only blocked the ladder, never another operative.

def test_a_second_lease_is_refused_and_names_the_holder():
    from arena import hub, leases
    first = hub.acquire_lease("engineer-b", "expB-304a gate battery", ttl_s=600)
    try:
        with pytest.raises(leases.LeaseConflict) as e:
            leases.acquire("modeler", "D-1/D-2/D-3 exp-013 halt diagnostics", ttl_s=600)
        msg = str(e.value)
        assert first["id"] in msg and "engineer-b" in msg, "the refusal says who to ask"
        assert "expB-304a gate battery" in msg, "and what they are running"
        assert "arena lease release" in msg, "and how to proceed"
        assert [l["id"] for l in leases.active()] == [first["id"]], \
            "a refused acquire creates nothing"
        assert isinstance(e.value, leases.LeaseHeld), "so the HTTP layer answers 409"
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        ref = [ev for ev in events if ev["event"] == "acquire-refused"]
        assert ref and ref[-1]["owner"] == "modeler" and first["id"] in ref[-1]["held_by"], \
            "and the refusal itself is evidence, on the lease log"
    finally:
        hub.release_lease(first["id"], resume=True)
    second = leases.acquire("modeler", "exp-013 diagnostics", ttl_s=600)   # released: next in
    leases.release(second["id"])


def test_shared_leases_coexist_but_never_with_an_exclusive_one():
    from arena import leases
    a = leases.acquire("ops", "daily ingestion window", ttl_s=600, shared=True)
    try:
        b = leases.acquire("registrar", "log tidy", ttl_s=600, shared=True)
        assert {l["id"] for l in leases.active()} == {a["id"], b["id"]}
        assert all(l["shared"] for l in leases.active())
        with pytest.raises(leases.LeaseConflict, match="exclusive lease"):
            leases.acquire("engineer-b", "expB-305 gate battery", ttl_s=600)
        leases.release(b["id"])
        with pytest.raises(leases.LeaseConflict):
            leases.acquire("engineer-b", "expB-305 gate battery", ttl_s=600)
    finally:
        for l in leases.active():
            leases.release(l["id"])
    excl = leases.acquire("engineer-b", "expB-305 gate battery", ttl_s=600)
    try:
        assert excl["shared"] is False, "exclusive is the default"
        with pytest.raises(leases.LeaseConflict, match="shared lease"):
            leases.acquire("ops", "ingestion", ttl_s=600, shared=True)
    finally:
        leases.release(excl["id"])


def test_an_expired_lease_does_not_block_the_next_one(monkeypatch):
    """Exclusivity must not turn an abandoned lease into a wedge: expiry still wins."""
    from arena import leases
    old = leases.acquire("modeler", "abandoned battery", ttl_s=60)
    monkeypatch.setattr(leases, "_epoch", lambda: old["acquired_at"] + 61)
    try:
        fresh = leases.acquire("engineer", "gate run", ttl_s=600)
        assert [l["id"] for l in leases.active()] == [fresh["id"]]
    finally:
        monkeypatch.undo()
        for l in leases.active():
            leases.release(l["id"])


def test_two_client_race_leaves_exactly_one_holder():
    """The race the incident could have been: N clients acquire at once, one wins.

    Same machinery as the F3 write-storm test (12 threads through the whole public path), but
    asserting the *policy*: 1 lease granted, 11 refusals, and the file still parses."""
    import threading as th
    from arena import leases
    granted, refused, other = [], [], []
    start = th.Barrier(12)

    def take(i):
        start.wait(timeout=30)
        try:
            granted.append(leases.acquire(f"racer{i}", f"battery {i}", ttl_s=120)["id"])
        except leases.LeaseConflict as exc:
            refused.append(exc)
        except Exception as exc:                   # noqa: BLE001 - reported below
            other.append(exc)

    threads = [th.Thread(target=take, args=(i,)) for i in range(12)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not other, other
        assert len(granted) == 1, f"exactly one client may hold the Pit, got {granted}"
        assert len(refused) == 11
        assert [l["id"] for l in leases.active()] == granted
        assert not leases.LOCK_PATH.exists()
    finally:
        for l in leases.active():
            leases.release(l["id"])


# argv: <owner> <gate path>. The gate is a real barrier: every child announces itself, then
# spins until the parent opens the gate, so all of them are inside `acquire` at once. Without it
# python's own startup cost (~200 ms, and unequal) staggers the children and the race never runs.
TWO_PROCESS_RACE = '''
import os, sys, time
sys.path.insert(0, {repo!r})
import arena.leases as L
gate = sys.argv[2]
open(gate + ".ready-" + sys.argv[1], "w").close()
deadline = time.time() + 120
while not os.path.exists(gate) and time.time() < deadline:
    time.sleep(0.002)
try:
    print("OK " + L.acquire(sys.argv[1], "race", ttl_s=120)["id"])
except L.LeaseConflict:
    print("CONFLICT")
'''


def test_two_process_race_leaves_exactly_one_holder(tmp_path):
    """The threading test cannot see the cross-process window: `_LOCK` is per-process, and two
    shells on the Pit host each read "no leases" and each wrote themselves in. That is what the
    O_EXCL lock file closes, and only real processes prove it."""
    import subprocess
    import time as _t
    from arena import leases
    prog = TWO_PROCESS_RACE.format(repo=str(REPO))
    env = dict(os.environ, KAGGRI_ARENA_DIR=str(ARENA_DIR))
    gate = tmp_path / "go"
    n = 4
    procs = [subprocess.Popen([sys.executable, "-c", prog, f"proc{i}", str(gate)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
             for i in range(n)]
    try:
        deadline = _t.time() + 120
        while len(list(tmp_path.glob("go.ready-*"))) < n and _t.time() < deadline:
            _t.sleep(0.02)
        assert len(list(tmp_path.glob("go.ready-*"))) == n, "all clients must be armed first"
        gate.write_text("go", encoding="utf-8")            # they all enter acquire() together
        outs = [p.communicate(timeout=120) for p in procs]
        for p, (_, err) in zip(procs, outs):
            assert p.returncode == 0, err
        verdicts = [out.strip().split()[0] for out, _ in outs]
        assert verdicts.count("OK") == 1, f"exactly one process may win: {verdicts}"
        assert verdicts.count("CONFLICT") == n - 1
        assert len(leases.active()) == 1
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for l in leases.active():
            leases.release(l["id"])
        leases.LOCK_PATH.unlink(missing_ok=True)


def test_a_stale_lock_file_is_broken_rather_than_wedging_the_arena(monkeypatch):
    """A crashed holder must not lock the Pit out of its own state file forever."""
    from arena import leases
    leases.LOCK_PATH.write_text("pid 999999 acquire (crashed)\n", encoding="utf-8")
    monkeypatch.setattr(leases, "LOCK_STALE_S", 0.0)
    try:
        rec = leases.acquire("engineer", "after a crash", ttl_s=60)
        assert leases.active()[0]["id"] == rec["id"]
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        assert any(e["event"] == "lock-broken" for e in events), "and it says so in the log"
    finally:
        monkeypatch.undo()
        for l in leases.active():
            leases.release(l["id"])
        leases.LOCK_PATH.unlink(missing_ok=True)


def test_a_broken_holder_does_not_delete_its_successors_lock(monkeypatch):
    """Auditor L-2, reproduced: A is inside, its lock is broken as stale, B takes its own -- and
    A must not delete B's lock on the way out (observed A-in, B-in, A-out, B-out, with the lock
    file gone, so a *third* process could enter while B was still mutating).

    Two threads model two processes: the in-process `_LOCK` is neutered, which is exactly the
    part the lock FILE exists to cover, and LOCK_STALE_S is compressed from 20 s to 0.4 s so the
    stall does not have to be real."""
    import threading as th
    from arena import leases
    inside, tokens = [], {}
    monkeypatch.setattr(leases, "LOCK_STALE_S", 0.4)
    monkeypatch.setattr(leases, "_LOCK", contextlib.nullcontext())

    def slow():                                    # A: stalls long enough to look abandoned
        with leases._state_lock("slow"):
            inside.append("A-in")
            tokens["A"] = leases._lock_holder()
            time.sleep(1.2)
            inside.append("A-out")

    def fast():                                    # B: arrives after A's lock has gone stale
        time.sleep(0.6)
        with leases._state_lock("fast"):
            inside.append("B-in")
            tokens["B"] = leases._lock_holder()
            deadline = time.time() + 20             # stay inside until A has left
            while "A-out" not in inside and time.time() < deadline:
                time.sleep(0.02)
            tokens["after_A_left"] = leases._lock_holder()
            inside.append("B-out")

    ta, tb = th.Thread(target=slow), th.Thread(target=fast)
    try:
        ta.start(); tb.start(); ta.join(timeout=30); tb.join(timeout=30)
        assert inside == ["A-in", "B-in", "A-out", "B-out"], \
            f"the interleaving under test did not occur ({inside}); the assertions below are void"
        assert tokens["A"] != tokens["B"], "each holder writes its own identity into the lock"
        assert tokens["after_A_left"] == tokens["B"], \
            "B's lock survived A's exit -- the unconditional unlink here is what let a THIRD in"
        assert not leases.LOCK_PATH.exists(), "and B still cleaned up after itself"
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        assert any(e["event"] == "lock-broken" for e in events[-8:]), "the break is on the log"
        lost = [e for e in events if e["event"] == "lock-lost"]
        assert lost and lost[-1]["held_token"] == tokens["A"] and lost[-1]["found"] == tokens["B"], \
            "and so is the fact that A ran outside its lock -- anything A wrote is suspect"
    finally:
        monkeypatch.undo()
        leases.LOCK_PATH.unlink(missing_ok=True)


def test_a_broken_out_holder_cannot_write_over_the_new_holders_state(monkeypatch):
    """The other half of L-2. No timeout-based lock can stop a pathologically stalled holder from
    overlapping its successor -- but it can stop the stalled one's WRITE, which is the part that
    corrupts a lease file. Every write re-checks the lock identity first and fails closed."""
    from arena import leases
    monkeypatch.setattr(leases, "_LOCK", contextlib.nullcontext())
    before = leases.LEASES_PATH.read_text(encoding="utf-8") if leases.LEASES_PATH.exists() else None
    try:
        with leases._state_lock("slow"):
            leases.LOCK_PATH.write_text("successors-token\npid 1 acquire\n", encoding="utf-8")
            with pytest.raises(leases.LockLost) as e:
                leases._save({"leases": {"invented": {"expires_at": 1, "acquired_at": 0}}})
            assert isinstance(e.value, leases.LeaseHeld), "unknown state is HELD state -> 409"
            assert "NOT applied" in str(e.value)
        now = leases.LEASES_PATH.read_text(encoding="utf-8") if leases.LEASES_PATH.exists() else None
        assert now == before, "the refused write left the successor's state exactly as it was"
        assert leases.LOCK_PATH.read_text(encoding="utf-8").startswith("successors-token"), \
            "and did not take the successor's lock with it"
        lost = [e for e in _events() if e["event"] == "lock-lost"]
        assert any(e["action"] == "save" and e["found"] == "successors-token" for e in lost), \
            "the refused write is on the log; anything that process reported is suspect"
    finally:
        monkeypatch.undo()
        leases.LOCK_PATH.unlink(missing_ok=True)


def test_a_stale_break_is_owned_by_one_breaker_and_leaves_no_debris():
    """The break is a rename, so two would-be breakers cannot both 'win' it -- and the renamed
    file is not left lying around next to leases.json."""
    from arena import leases
    leases.LOCK_PATH.write_text("some-dead-token\npid 999999 acquire\n", encoding="utf-8")
    old_stale = leases.LOCK_STALE_S
    leases.LOCK_STALE_S = -1.0             # not 0.0: a fresh file's age can round to exactly 0
    try:
        assert leases._lock_is_stale() is True
        assert not leases.LOCK_PATH.exists()
        assert leases._lock_is_stale() is True, "a lock that is already gone just means 'retry'"
        assert not list(leases.LOCK_PATH.parent.glob(leases.LOCK_PATH.name + ".broken-*"))
    finally:
        leases.LOCK_STALE_S = old_stale
        leases.LOCK_PATH.unlink(missing_ok=True)


def test_a_live_lock_fails_closed_instead_of_writing_anyway(monkeypatch):
    from arena import leases
    leases.LOCK_PATH.write_text("pid 999999 acquire\n", encoding="utf-8")
    monkeypatch.setattr(leases, "LOCK_WAIT_S", 0.05)
    try:
        with pytest.raises(leases.LeaseLockTimeout) as e:
            leases.acquire("engineer", "blocked", ttl_s=60)
        assert isinstance(e.value, leases.LeaseHeld), "unknown state is HELD state -> 409"
        assert leases.status()["lease_held"] is False, \
            "a jammed lock must not invent a lease; only the mutation is refused"
    finally:
        monkeypatch.undo()
        leases.LOCK_PATH.unlink(missing_ok=True)


# ------------------------------------------------- the run-start guard: `match`/`gate` refuse
# `acquire` stops a second *lease*; the contamination only needed a second *run*.

def test_match_and_gate_refuse_to_start_under_a_foreign_lease():
    from arena import cli, leases
    lease = leases.acquire("engineer-b", "expB-304a confirm arm", ttl_s=600)
    try:
        with pytest.raises(leases.LeaseHeld) as e:
            leases.guard_run("match", detail="probe")
        assert lease["id"] in str(e.value) and "--ignore-lease" in str(e.value)
        assert cli.main(["match", "random", "pass", "--n", "2", "--steps", "8"]) == cli.REFUSED_EXIT
        assert cli.main(["gate", "random", "--vs", "pass"]) == cli.REFUSED_EXIT
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        refusals = [ev for ev in events if ev["event"] == "guard-refused"]
        assert {r["action"] for r in refusals} >= {"match", "gate"}
    finally:
        leases.release(lease["id"])


def test_the_guard_lets_the_leaseholder_through_by_id_and_by_env(monkeypatch):
    from arena import leases
    lease = leases.acquire("engineer", "my own battery", ttl_s=600)
    try:
        assert [l["id"] for l in leases.guard_run("match", lease_id=lease["id"])] == [lease["id"]]
        monkeypatch.setenv(leases.LEASE_ENV, lease["id"])
        assert leases.guard_run("gate"), "the env fallback carries the same declaration"
        monkeypatch.setenv(leases.LEASE_ENV, "someone-elses-id")
        with pytest.raises(leases.LeaseHeld):
            leases.guard_run("gate")
    finally:
        monkeypatch.delenv(leases.LEASE_ENV, raising=False)
        leases.release(lease["id"])


def test_ignore_lease_is_allowed_but_recorded():
    from arena import leases
    lease = leases.acquire("modeler", "long unmeasured training", ttl_s=600)
    try:
        seen = leases.guard_run("match", ignore=True, detail="screen v0a v1")
        assert [l["id"] for l in seen] == [lease["id"]]
        events = [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]
        ig = [e for e in events if e["event"] == "guard-ignored"]
        assert ig and ig[-1]["detail"] == "screen v0a v1" and lease["id"] in ig[-1]["lease_ids"], \
            "an override leaves a trace, so contaminated timing can be voided afterwards"
    finally:
        leases.release(lease["id"])


def test_the_guard_runs_free_when_nobody_holds_the_pit():
    from arena import leases
    assert leases.active() == []
    assert leases.guard_run("match") == []


def test_the_guard_fails_closed_on_an_unreadable_lease_file():
    from arena import leases
    leases.LEASES_PATH.write_text('{"leases": {"x": ', encoding="utf-8")
    try:
        with pytest.raises(leases.LeaseStateUnreadable):
            leases.guard_run("match")
    finally:
        leases.LEASES_PATH.unlink(missing_ok=True)


def test_the_guard_fails_closed_when_a_declared_hub_cannot_be_read():
    """The remote check is opt-in precisely so it can fail closed: an unreachable hub is not
    evidence that the host it runs is free."""
    from arena import leases
    with pytest.raises(leases.LeaseCheckFailed, match="treated as held"):
        leases.guard_run("match", hub_url="http://127.0.0.1:9")   # discard port: never listening


# ---------------------------------------- the OTHER direction of the guard (Auditor L-1, 08-22)
# The guard refused to start *under* a foreign lease but never TOOK one, and said nothing at all
# when none existed. So the incident was fixed in the form it happened (holder first, intruder
# second) and not in the mirror form -- a leaseless measured run, then somebody acquires and
# starts a battery under it -- which was the easy default, because nothing prompted for a lease.

def _stub_run_match(monkeypatch, during=None):
    """Replace the game loop with a stub, optionally running `during()` inside the run."""
    from arena import scheduler
    def fake(agents, **kw):
        if during is not None:
            during()
        return {"rows": [], "agents": [], "mode": kw.get("mode", "screen"),
                "run_id": "stub-run", "wall_s": 0.0}
    monkeypatch.setattr(scheduler, "run_match", fake)


def _events():
    from arena import leases
    return [json.loads(l) for l in leases.LEASE_LOG.read_text(encoding="utf-8").splitlines()]


def test_a_leaseless_match_takes_a_lease_of_its_own(monkeypatch):
    from arena import cli, hub, leases
    seen_inside = {}

    def during():
        seen_inside["leases"] = [dict(l) for l in leases.active()]
        seen_inside["paused"] = hub.is_paused()

    _stub_run_match(monkeypatch, during)
    assert leases.active() == [] and not hub.is_paused()
    assert cli.main(["match", "random", "pass", "--n", "2", "--lease-owner", "engineer"]) == 0

    held = seen_inside["leases"]
    assert len(held) == 1, "a run that declares no lease now takes one, so nothing can join it"
    assert held[0]["owner"] == "engineer" and held[0]["shared"] is False
    assert "match" in held[0]["purpose"] and "random" in held[0]["purpose"], \
        "the refusal somebody else gets has to say what is running"
    assert seen_inside["paused"] is True, "a run that plays games also holds the ladder off"

    assert leases.active() == [], "and it releases on the way out -- no TTL wait for the next run"
    assert not hub.is_paused(), "and puts the ladder back the way it found it"
    ev = _events()
    mine = [e["event"] for e in ev
            if e.get("lease_id") == held[0]["id"] and e["event"] in ("acquire", "release")]
    assert mine == ["acquire", "release"]
    assert any(e["event"] == "pause-restored" for e in ev[-4:])


def test_a_second_leaseless_run_is_refused_while_the_first_is_playing(monkeypatch):
    """The mirror form of the 2026-08-21 incident, end to end: two leaseless `arena match`es.

    Before the fix both proceeded silently and shared the host. Now the first one's own lease is
    what refuses the second."""
    from arena import cli, leases
    intruder = {}

    def during():
        intruder["rc"] = cli.main(["match", "random", "pass", "--n", "2"])

    _stub_run_match(monkeypatch, during)
    assert cli.main(["match", "random", "pass", "--n", "2", "--lease-owner", "engineer-b"]) == 0
    assert intruder["rc"] == cli.REFUSED_EXIT, "the second run must be refused, not silently share"
    refused = [e for e in _events() if e["event"] == "guard-refused"]
    assert refused and refused[-1]["action"] == "match"
    assert leases.active() == []


def test_no_lease_is_the_documented_opt_out_and_is_logged_loudly(monkeypatch, capsys):
    """Scripted callers that genuinely want no lease say so once -- and it leaves a trace, so a
    timing claim from an unleased run can be challenged afterwards."""
    from arena import cli, leases
    inside = {}
    _stub_run_match(monkeypatch, lambda: inside.update(n=len(leases.active())))
    assert cli.main(["match", "random", "pass", "--n", "2", "--no-lease"]) == 0
    assert inside["n"] == 0, "--no-lease takes nothing"
    out = capsys.readouterr().out
    assert "--no-lease" in out and "contestable" in out
    un = [e for e in _events() if e["event"] == "run-unleased"]
    assert un and un[-1]["action"] == "match"


def test_a_declared_lease_is_used_rather_than_a_second_one_taken(monkeypatch):
    """The scripted-battery path: one lease for the whole battery, exported once. Nothing about
    it changes -- in particular the run must not take a lease under the operator's lease."""
    from arena import cli, leases
    lease = leases.acquire("engineer-b", "expB-303 battery", ttl_s=600)
    inside = {}
    _stub_run_match(monkeypatch, lambda: inside.update(ids=[l["id"] for l in leases.active()]))
    try:
        monkeypatch.setenv(leases.LEASE_ENV, lease["id"])
        assert cli.main(["match", "random", "pass", "--n", "2"]) == 0
        assert inside["ids"] == [lease["id"]], "the battery's own lease, and only it"
        assert [l["id"] for l in leases.active()] == [lease["id"]], \
            "and the run does not release a lease it did not take"
    finally:
        monkeypatch.delenv(leases.LEASE_ENV, raising=False)
        leases.release(lease["id"])


def test_ignore_lease_still_takes_nothing(monkeypatch):
    """`--ignore-lease` means 'leases are not my problem for this run': it overrides the refusal
    AND skips the acquire, so it cannot surprise a script by pausing the ladder."""
    from arena import cli, leases
    inside = {}
    _stub_run_match(monkeypatch, lambda: inside.update(n=len(leases.active())))
    assert cli.main(["match", "random", "pass", "--n", "2", "--ignore-lease"]) == 0
    assert inside["n"] == 0


def test_a_node_run_leases_the_node_not_this_desktop(monkeypatch):
    """The games run on the node, so the lease that matters is the node's (Auditor L-4). Guarding
    here would check a lease file no game touches -- and a stale local lease would refuse a
    perfectly good node run."""
    from arena import cli, leases, remote
    calls = {}
    monkeypatch.setattr(remote, "run_remote", lambda args: calls.setdefault("argv", args) and 0 or 0)
    stale = leases.acquire("modeler", "something local", ttl_s=600)
    try:
        assert cli.main(["match", "random", "pass", "--node", "workflows-box"]) == 0
        assert "argv" in calls, "the local guard must not stand between a --node run and the node"
        assert [l["id"] for l in leases.active()] == [stale["id"]], "and it takes nothing here"
    finally:
        leases.release(stale["id"])


def test_remote_forwards_every_lease_flag(monkeypatch):
    """Skipping the local guard is only safe if the node-side run gets the whole declaration."""
    import argparse
    from arena import remote
    args = argparse.Namespace(lease="L1", lease_owner="engineer-b", lease_hub="http://h:8765",
                              lease_ttl=7200, ignore_lease=False, no_lease=True)
    argv = []
    for flag in ("lease", "lease_owner", "lease_hub", "lease_ttl"):
        val = getattr(args, flag, None)
        if val:
            argv += ["--" + flag.replace("_", "-"), str(val)]
    for flag in ("ignore_lease", "no_lease"):
        if getattr(args, flag, False):
            argv += ["--" + flag.replace("_", "-")]
    assert argv == ["--lease", "L1", "--lease-owner", "engineer-b",
                    "--lease-hub", "http://h:8765", "--lease-ttl", "7200", "--no-lease"]
    src = (Path(remote.__file__)).read_text(encoding="utf-8")
    for flag in ("lease_owner", "lease_hub", "lease_ttl", "no_lease"):
        assert flag in src, f"remote.py must forward --{flag.replace('_', '-')}"


def test_a_gate_leases_without_stopping_the_ladder():
    """`arena gate` plays no games: it needs exclusivity against a battery, but no pause. And a
    lease that paused the ladder for two seconds and did not resume it would be worse than the
    problem -- nothing resumes a ladder that `reconcile_pause` stopped."""
    from arena import hub, leases
    assert not hub.is_paused()
    with hub.lease_for_run("auditor", "gate: cand vs B", ttl_s=60, pause=False) as rec:
        assert [l["id"] for l in leases.active()] == [rec["id"]]
        assert not hub.is_paused(), "a read does not stop the ladder"
    assert leases.active() == [] and not hub.is_paused()


def test_a_run_lease_leaves_a_ladder_that_was_already_paused_paused():
    """Restoring means 'as we found it', not 'running'. A ladder somebody else stopped is theirs."""
    from arena import hub, leases
    hub.set_paused(True)
    try:
        with hub.lease_for_run("engineer", "match: screen a b", ttl_s=60):
            assert hub.is_paused()
        assert hub.is_paused(), "we did not pause it, so it is not ours to resume"
        assert leases.active() == []
    finally:
        hub.set_paused(False)


def test_a_run_lease_is_released_even_when_the_run_raises():
    from arena import hub, leases
    with pytest.raises(ZeroDivisionError):
        with hub.lease_for_run("engineer", "match: boom", ttl_s=60):
            1 / 0
    assert leases.active() == [], "a crashed run must not leave the Pit leased for its whole TTL"
    assert not hub.is_paused()


def test_leases_held_releases_on_every_exit_path():
    """The Python half of the EXIT-trap convention."""
    from arena import leases
    with leases.held("ops", "ingest window", ttl_s=60, shared=True) as rec:
        assert [l["id"] for l in leases.active()] == [rec["id"]]
    assert leases.active() == []
    with pytest.raises(KeyboardInterrupt):
        with leases.held("ops", "ingest window", ttl_s=60) as rec:
            raise KeyboardInterrupt
    assert leases.active() == [], "Ctrl-C is the case that actually happens"
    with leases.held("ops", "vanishing", ttl_s=60) as rec:
        leases.release(rec["id"])                  # somebody force-broke it under us
    assert any(e["event"] == "release-noop" for e in _events()[-3:]), \
        "releasing a lease that is already gone is logged, not raised out of a finally"


# ------------------------------------------- probe isolation (Auditor L-3, 2026-08-22)
# The auditor ran a probe of mine the way its own docstring documented, minus the env var it did
# not set itself; it wrote leases.json, lease-log.jsonl and a `paused` flag into a real arena dir.
# On the Pit host that invocation pauses the live ladder, and nothing resumes it.

def test_require_scratch_arena_refuses_a_live_arena_dir(tmp_path, monkeypatch):
    from arena import paths as P
    monkeypatch.delenv("KAGGRI_ARENA_DIR", raising=False)
    with pytest.raises(P.LiveArenaRefused, match="default arena dir"):
        P.require_scratch_arena("this probe")

    monkeypatch.setenv("KAGGRI_ARENA_DIR", str(tmp_path / "live"))
    live = tmp_path / "live"
    live.mkdir()
    (live / "arena.sqlite").write_text("not really", encoding="utf-8")
    monkeypatch.setattr(P, "ARENA_DIR", live)
    with pytest.raises(P.LiveArenaRefused, match="live arena state"):
        P.require_scratch_arena("this probe")

    scratch = tmp_path / "scratch"
    monkeypatch.setenv("KAGGRI_ARENA_DIR", str(scratch))
    monkeypatch.setattr(P, "ARENA_DIR", scratch)
    monkeypatch.setattr(P, "AGENTS_DIR", scratch / "agents")
    monkeypatch.setattr(P, "REPLAYS_DIR", scratch / "replays")
    monkeypatch.setattr(P, "RUNS_DIR", scratch / "runs")
    monkeypatch.setattr(P, "PROGRESS_DIR", scratch / "progress")
    assert P.require_scratch_arena("this probe") == scratch
    assert (scratch / P.SCRATCH_MARKER).exists()
    assert P.require_scratch_arena("this probe") == scratch, "a marked scratch dir is reusable"


# --------------------------------------------------- verify baselines (Auditor finding P21-3)
# `arena verify` exited 1 on every store in the org, mostly for the intended "a clean node run
# supersedes a noisy local one" reason, so the exit code carried no signal and the real orphans
# drowned. A per-store baseline makes "new supersessions only" expressible.

def _incomplete_store(tmp_path, name="base.sqlite"):
    """A store with one run that lost all its games to a later one, plus an orphan run_id."""
    con = store.connect(tmp_path / name)
    store.record_run(con, {"run_id": "run-A", "mode": "confirm", "agents": "a,b", "n_games": 4,
                           "started_at": "2026-08-01T00:00:00"})
    store.upsert_rows(con, [_row("run-A", seed=s) for s in range(4)])
    store.upsert_rows(con, [_row("run-B", seed=s) for s in range(4)])     # takes every row
    store.upsert_rows(con, [_row("run-ghost", seed=99)])                  # no run row at all
    return con


def test_verify_baseline_records_then_reports_only_what_is_new(tmp_path):
    con = _incomplete_store(tmp_path)
    rep = store.integrity_report(con)
    assert not rep["ok"] and len(rep["incomplete_runs"]) == 1 and rep["orphan_games"]

    bpath = store.baseline_path(tmp_path / "base.sqlite")
    assert bpath.name == "base.sqlite" + store.BASELINE_SUFFIX, "the baseline lives beside its DB"
    assert store.load_baseline(bpath) is None
    assert store.compare_baseline(rep, None)["ok"] is False, "no baseline => everything is new"

    store.record_baseline(rep, bpath, note="intended: node run superseded the local one")
    cmp1 = store.compare_baseline(rep, store.load_baseline(bpath))
    assert cmp1["ok"] and not cmp1["new_incomplete_runs"] and not cmp1["new_orphan_games"]
    assert cmp1["accepted_incomplete"] == 1
    assert cmp1["accepted_orphans"] == len(rep["orphan_games"]) == 2   # run-B and run-ghost
    assert cmp1["baseline"]["note"].startswith("intended")
    con.close()


def test_verify_baseline_still_flags_a_new_and_a_worsened_finding(tmp_path):
    con = _incomplete_store(tmp_path)
    base = store.record_baseline(store.integrity_report(con), store.baseline_path(tmp_path / "base.sqlite"))
    assert base["incomplete_runs"]["run-A"]["missing"] == 4

    # a NEW run loses its games, and the already-accepted one loses more
    store.record_run(con, {"run_id": "run-C", "mode": "confirm", "agents": "a,b", "n_games": 2,
                           "started_at": "2026-08-02T00:00:00"})
    store.upsert_rows(con, [_row("run-C", seed=s) for s in (10, 11)])
    store.upsert_rows(con, [_row("run-D", seed=s) for s in (10, 11)])
    con.execute("UPDATE runs SET n_games = 6 WHERE run_id = 'run-A'")      # it played more
    con.commit()

    cmp2 = store.compare_baseline(store.integrity_report(con),
                                  store.load_baseline(store.baseline_path(tmp_path / "base.sqlite")))
    flagged = {r["run_id"]: r for r in cmp2["new_incomplete_runs"]}
    assert set(flagged) == {"run-A", "run-C"}, "new runs AND accepted runs that got worse"
    assert flagged["run-A"]["accepted_missing"] == 4 and flagged["run-A"]["missing"] == 6
    assert flagged["run-C"]["accepted_missing"] is None
    assert cmp2["ok"] is False
    con.close()


def test_verify_cli_exit_codes_follow_the_baseline(tmp_path, capsys):
    from arena import cli
    db = str(tmp_path / "cli.sqlite")
    _incomplete_store(tmp_path, "cli.sqlite").close()
    assert cli.main(["verify", "--db", db]) == 1, "no baseline: the raw answer, as before"
    assert "no baseline" in capsys.readouterr().out
    assert cli.main(["verify", "--db", db, "--baseline", "record", "--note", "accepted"]) == 0
    assert cli.main(["verify", "--db", db]) == 0, "and nothing is new afterwards"
    out = capsys.readouterr().out
    assert "OK: nothing new since the baseline" in out and "accepted, not shown above" in out
    assert cli.main(["verify", "--db", db, "--baseline", "ignore"]) == 1, \
        "--baseline ignore keeps the unfiltered check available"
