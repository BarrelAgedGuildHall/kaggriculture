"""In-process episode runner for PPO rollouts.

Training infrastructure ONLY. The Pit (`arena`) stays the sole evaluation authority; no
number produced here is ever quoted as a result (spec section 3). The arena's per-agent
subprocess isolation is right for evaluation and far too slow for rollouts, so this runs
`kaggle_environments` in-process with both policies as plain callables.

The `step 0` calibration in `calibrate.py` is what earns this file the right to exist: it
plays the same agents on the same seeds here and in the Pit and checks the banks agree.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena import engine as arena_engine      # noqa: E402  (quiet kaggle_environments import)

from . import bcpath                          # noqa: E402,F401
from .actor import RecordingPolicy            # noqa: E402
from .bcpath import P, WEIGHTS_DIR            # noqa: E402

REWARD_SCALE = 50_000.0
BASE_MEMBERS = sorted(str(p) for p in WEIGHTS_DIR.glob("*.npz"))
PI0_MEMBERS = [str(WEIGHTS_DIR / "rebuild-m0.npz")]


def make_policy(spec, safety=3, greedy=True, temperature=1.0, seed=0, record=False):
    """spec: 'pi0' | 'base' | a path/list of .npz paths."""
    if spec == "pi0":
        members = PI0_MEMBERS
    elif spec == "base":
        members = BASE_MEMBERS
    elif isinstance(spec, (list, tuple)):
        members = [str(m) for m in spec]
    else:
        members = [str(spec)]
    if len(members) == 1 and (record or not greedy):
        return RecordingPolicy(members, safety=safety, greedy=greedy,
                               temperature=temperature, seed=seed, record=record)
    return P.BCPolicy(members, safety=safety)


def _callable(pol):
    def fn(obs, config=None):
        return pol.act(obs)
    return fn


def play(seed, seat, actor, opponent, steps=720):
    """Play one episode. `actor` sits in `seat`. Returns (reward_actor, reward_opp, steps)."""
    ke = arena_engine.import_engine()
    env = ke.make("kaggriculture", configuration={"seed": int(seed), "episodeSteps": int(steps)},
                  debug=False)
    agents = [None, None]
    agents[seat] = _callable(actor)
    agents[1 - seat] = _callable(opponent)
    env.run(agents)
    last = env.steps[-1]
    money = [float(last[0]["observation"]["farms"][i]["money"]) for i in (0, 1)]
    rewards = [float(last[i]["reward"]) for i in (0, 1)]
    statuses = [last[i]["status"] for i in (0, 1)]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"episode seed={seed} ended {statuses}")
    # spec section 1.1: the trainer asserts the engine's terminal reward IS the seat's money.
    for i in (0, 1):
        if abs(rewards[i] - money[i]) > 1e-6:
            raise RuntimeError(f"reward/money mismatch seat {i}: {rewards[i]} vs {money[i]}")
    return rewards[seat], rewards[1 - seat], len(env.steps)


def rollout(seed, seat, actor_members, opp_members, rng_seed, temperature=1.0,
            safety=3, steps=720):
    """One recorded episode. Returns (turns, R_actor, R_opp)."""
    actor = RecordingPolicy(actor_members, safety=safety, greedy=False,
                            temperature=temperature, seed=rng_seed, record=True)
    opp = make_policy(opp_members, safety=safety, greedy=True)
    r_us, r_them, _ = play(seed, seat, actor, opp, steps=steps)
    return actor.turns, r_us, r_them
