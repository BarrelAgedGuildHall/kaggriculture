"""Fast smoke tests: the engine runs, the pin matches the recorded version, an episode is
deterministic for a given seed, and the replay checker reproduces a locally generated replay.
Run: uv run pytest -q
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _engine import ENV_NAME, engine_version, import_engine  # noqa: E402


def test_engine_version_matches_version_md():
    recorded = (REPO / "docs" / "engine" / "VERSION.md").read_text(encoding="utf-8")
    m = re.search(r"`kaggle-environments` \(pinned\) \| \*\*([\d.]+)\*\*", recorded)
    assert m, "VERSION.md must state the pinned version"
    assert engine_version() == m.group(1)


def test_engine_file_sha_matches_version_md():
    import hashlib

    from _engine import engine_dir

    sha = hashlib.sha256((engine_dir() / "kaggriculture.py").read_bytes()).hexdigest()
    recorded = (REPO / "docs" / "engine" / "VERSION.md").read_text(encoding="utf-8")
    assert sha in recorded, "kaggriculture.py sha256 differs from docs/engine/VERSION.md -- engine changed?"


def _play(agent_a, agent_b, seed, steps=48):
    ke = import_engine()
    env = ke.make(ENV_NAME, configuration={"episodeSteps": steps, "seed": seed})
    env.run([agent_a, agent_b])
    return env


def test_agent_is_legal_and_makes_money():
    env = _play(str(REPO / "agent" / "main.py"), "starter", seed=0)
    final = env.steps[-1]
    assert [s.status for s in final] == ["DONE", "DONE"]
    # 48 steps = two days: enough to prove the opening spends and produces, not to score it.
    assert final[0].observation.farms[0]["money"] != 3000


def test_episode_is_deterministic_for_seed():
    a = _play("starter", "starter", seed=7, steps=120)
    b = _play("starter", "starter", seed=7, steps=120)
    assert json.dumps(a.toJSON()["steps"][-1][0]["observation"], sort_keys=True) == json.dumps(
        b.toJSON()["steps"][-1][0]["observation"], sort_keys=True
    )


def test_replay_check_reproduces_local_replay(tmp_path):
    env = _play("starter", "starter", seed=11, steps=72)
    replay = tmp_path / "local.json"
    replay.write_text(json.dumps(env.toJSON()), encoding="utf-8")
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / "replay_check.py"), str(replay)], text=True, capture_output=True
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "RESULT ALL_REPRODUCE" in res.stdout


def test_replay_check_detects_tampering(tmp_path):
    env = _play("starter", "starter", seed=11, steps=72)
    data = env.toJSON()
    data["info"]["seed"] = 12345
    replay = tmp_path / "tampered.json"
    replay.write_text(json.dumps(data), encoding="utf-8")
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / "replay_check.py"), str(replay)], text=True, capture_output=True
    )
    # A different seed changes weeds/shops eventually; within 72 steps (3 days) the first
    # shop draw happens at end of day 2, so a mismatch is expected.
    assert res.returncode == 1, res.stdout


def test_agent_survives_a_full_episode_vs_random():
    """The competition gate: 720 turns against the built-in `random` agent with no
    exception, no bad status, and every turn inside the 1 s actTimeout. A crash on the
    ladder is a guaranteed loss, so this is the test that must never be skipped."""
    env = _play(str(REPO / "agent" / "main.py"), "random", seed=3, steps=720)
    final = env.steps[-1]
    assert len(env.steps) == 720
    assert [s.status for s in final] == ["DONE", "DONE"]
    assert final[0].observation.farms[0]["money"] > 3000

    traces = [lg[0].get("stderr", "") for lg in env.logs if lg and isinstance(lg[0], dict)]
    assert not any("Traceback" in t for t in traces), "agent raised during the episode"

    durations = [lg[0]["duration"] for lg in env.logs
                 if lg and isinstance(lg[0], dict) and "duration" in lg[0]]
    assert durations and max(durations) < 1.0, f"turn over the 1 s budget: {max(durations):.3f}s"


def test_agent_returns_a_wellformed_action_for_a_degenerate_observation():
    """Hardening: the policy must never propagate an exception, whatever it is handed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kagg_agent", REPO / "agent" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for bad in ({}, {"player": 0}, {"player": 0, "farms": []}, {"player": 5, "farms": [{}]}):
        out = mod.agent(bad)
        assert set(out) == {"farmer", "hands", "market"}
        assert isinstance(out["farmer"], list) and out["farmer"][0] in mod.UNIT_OPS
        assert out["hands"] == [] and out["market"] == []


@pytest.mark.parametrize("fresh", [False])
def test_make_submission_builds_and_verifies(fresh):
    args = [sys.executable, str(SCRIPTS / "make_submission.py"), "--verify", "--steps", "48"]
    if not fresh:
        args.append("--no-fresh-venv")
    res = subprocess.run(args, text=True, capture_output=True, cwd=str(REPO))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "VERIFY OK" in res.stdout
