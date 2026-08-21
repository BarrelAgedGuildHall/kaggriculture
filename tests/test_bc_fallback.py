"""Degraded-mode behaviour of the Team A BC entry point (audit finding NB-2).

Before this, a weights-load failure left `_POLICY = None`; the first `agent()` call retried,
threw again, and the `except` returned `{"farmer": ["PASS"], ...}` -- for all 720 turns. The
"fallback" was a guaranteed lost episode dressed up as resilience, and PPO churning weights
files makes a load failure an operational risk rather than a hypothetical one.

These tests assert the intended behaviour instead: a load failure is announced once on
stderr and the episode is then *played* by the vendored weights-free v0 heuristic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BC = ROOT / "agents" / "team_a" / "bc"
V0 = ROOT / "agents" / "team_a" / "v0.py"
sys.path.insert(0, str(BC))

pytest.importorskip("numpy")

SUBSTITUTION = ("DEBUG_TIMING = True",
                "DEBUG_TIMING = False   # vendored copy: never print from the fallback path")


def _body(text):
    """Everything after the module docstring."""
    first = text.index('"""')
    return text[text.index('"""', first + 3) + 3:].lstrip("\n")


def test_fallback_is_v0_verbatim_under_one_substitution():
    """The fallback is a frozen copy, not a fork. If v0 changes, re-vendor it."""
    expected = _body(V0.read_text(encoding="utf-8").replace(*SUBSTITUTION, 1))
    got = _body((BC / "fallback.py").read_text(encoding="utf-8"))
    assert got == expected, "agents/team_a/bc/fallback.py has drifted from agents/team_a/v0.py"


def test_fallback_never_prints_per_turn():
    """DEBUG_TIMING in the vendored copy must be off -- the fallback runs on a per-turn path."""
    import fallback
    assert fallback.DEBUG_TIMING is False


def _degraded_episode(tmp_path, seed=0):
    """Play a full episode with unloadable weights: main.py (degraded) vs v0.

    The agent is passed as a *file path* so kaggle-environments exec's it exactly as the
    real loader does -- a fresh module namespace per episode, which is what makes the
    latched module-level degraded state testable at all.
    """
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz file, not even close")
    old = os.environ.get("BC_MEMBERS")
    os.environ["BC_MEMBERS"] = str(corrupt)
    try:
        from kaggle_environments import make
        env = make("kaggriculture", configuration={"seed": seed}, debug=False)
        env.run([str(BC / "main.py"), str(V0)])
    finally:
        if old is None:
            os.environ.pop("BC_MEMBERS", None)
        else:
            os.environ["BC_MEMBERS"] = old
    # kaggle-environments redirects each agent's stderr into env.logs, not the process's.
    stderr = "".join(step[0].get("stderr", "") for step in (env.logs or []) if step)
    return env, stderr


def test_corrupt_weights_play_a_real_episode(tmp_path):
    """The intended behaviour: announce once, then play v0 -- not PASS for 720 turns."""
    env, stderr = _degraded_episode(tmp_path)

    assert "TEAM_A_BC_DEGRADED" in stderr, "degraded mode must be announced on stderr"
    assert stderr.count("TEAM_A_BC_DEGRADED") == 1, "announce once, not every turn"
    assert "weights-load" in stderr

    final = env.steps[-1][0]
    money = final["observation"]["farms"][0]["money"]
    assert money > 0, f"degraded episode banked {money}; the fallback did not play"
    # A PASS-for-720-turns agent ends at its starting bank. v0 banks tens of thousands
    # against another v0; anything in that region proves a real policy ran.
    assert money > 5000, f"degraded episode banked only {money} -- the fallback is not farming"
    assert env.steps[-1][0]["status"] == "DONE"
    assert env.steps[-1][1]["status"] == "DONE"


def test_degraded_agent_emits_non_pass_actions(tmp_path):
    """Structural proof, independent of the money threshold."""
    env, _ = _degraded_episode(tmp_path, seed=1)
    acted = sum(1 for s in env.steps
                if isinstance(s[0].get("action"), dict)
                and (s[0]["action"].get("market")
                     or s[0]["action"].get("farmer", ["PASS"])[0] != "PASS"))
    assert acted > 100, f"only {acted} non-PASS turns in a degraded episode"
