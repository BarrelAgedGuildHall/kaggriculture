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


def test_placeholder_agent_is_legal_and_keeps_starting_money():
    env = _play(str(REPO / "agent" / "main.py"), "starter", seed=0)
    final = env.steps[-1]
    assert [s.status for s in final] == ["DONE", "DONE"]
    assert final[0].observation.farms[0]["money"] == 3000


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


@pytest.mark.parametrize("fresh", [False])
def test_make_submission_builds_and_verifies(fresh):
    args = [sys.executable, str(SCRIPTS / "make_submission.py"), "--verify", "--steps", "48"]
    if not fresh:
        args.append("--no-fresh-venv")
    res = subprocess.run(args, text=True, capture_output=True, cwd=str(REPO))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "VERIFY OK" in res.stdout
