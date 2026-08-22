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

from . import auxlabels as AUX                # noqa: E402
from . import bcpath                          # noqa: E402,F401
from . import lines as LINES                  # noqa: E402
from .actor import RecordingPolicy            # noqa: E402
from .bcpath import F, P, WEIGHTS_DIR         # noqa: E402

REWARD_SCALE = 50_000.0
BASE_MEMBERS = sorted(str(p) for p in WEIGHTS_DIR.glob("*.npz"))
# `pi0` is the distilled single net accepted in exp-012b. It lives outside the repo (weights
# are git-ignored), so the path is an environment knob rather than a constant -- and it
# defaults to member 0, which is what exp-011 ran on, so an unset variable degrades to the
# documented exp-011 configuration instead of silently loading the wrong policy.
PI0_MEMBERS = [os.environ.get("PI0_NPZ") or str(WEIGHTS_DIR / "rebuild-m0.npz")]


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


def engine_module():
    """The kaggriculture module itself (not the `kaggle_environments` wrapper)."""
    arena_engine.import_engine()
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    return K


def play(seed, seat, actor, opponent, steps=720, instrument=None):
    """Play one episode. `actor` sits in `seat`. Returns (reward_actor, reward_opp, steps).

    `instrument` is an optional `lines.Accountant`: attached to the env and entered around
    `env.run`, it is observational only (`tests/test_ppo_lines.py` pins byte-identical banks with
    and without it), so the training and evaluation paths stay the same episode.
    """
    ke = arena_engine.import_engine()
    env = ke.make("kaggriculture", configuration={"seed": int(seed), "episodeSteps": int(steps)},
                  debug=False)
    agents = [None, None]
    agents[seat] = _callable(actor)
    agents[1 - seat] = _callable(opponent)
    if instrument is not None:
        instrument.attach(env)
        with instrument:
            env.run(agents)
    else:
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
            safety=3, steps=720, record_units=False, unit_subsample=1.0,
            unit_temperature=None, record_lines=False, aux_names=()):
    """One recorded episode. Returns (turns, unit_rows, R_actor, R_opp, lines, aux).

    `record_lines` attaches `lines.Accountant`, giving exp-014's factored critic its exactly
    decomposed terminal-bank target; `aux_names` additionally computes the rollout-side auxiliary
    labels (`auxlabels.py`, ratified by addendum-01 A-4.1). Both are off by default, so exp-013b's
    scalar-critic control runs the identical code path with the identical cost.
    """
    actor = RecordingPolicy(actor_members, safety=safety, greedy=False,
                            temperature=temperature, seed=rng_seed, record=True,
                            record_units=record_units, unit_subsample=unit_subsample,
                            unit_temperature=unit_temperature)
    opp = make_policy(opp_members, safety=safety, greedy=True)
    aux_names = tuple(aux_names or ())
    acc = rec = None
    if record_lines or aux_names:
        K = engine_module()
        acc = LINES.Accountant(K)
        if aux_names:
            rec = AUX.Recorder(K, want_nav=("nav" in aux_names))
            acc.on_step(rec)
    r_us, r_them, _ = play(seed, seat, actor, opp, steps=steps, instrument=acc)

    ep_lines = None
    if acc is not None and record_lines:
        ep_lines = acc.totals(seat)
        # The decomposition is only a decomposition if it closes. A rollout whose lines do not sum
        # to the bank would hand the factored critic a silently mis-attributed target, so it fails
        # the episode rather than training on it. `tests/test_ppo_lines.py` pins the identity
        # offline; this is the same assertion on the live training path, where what it catches is
        # an engine bump adding a cash path.
        resid = acc.residual(seat, r_us)
        if abs(resid) > 1e-6:
            raise RuntimeError(f"line identity broken on seed={seed} seat={seat}: "
                               f"residual {resid:.6f}")
    ep_aux = None
    if rec is not None:
        wanted = [int(t["day"]) * F.TURNS_PER_DAY + int(t["hour"]) for t in actor.turns]
        ep_aux = rec.labels(acc, seat, aux_names, wanted)
    return actor.turns, actor.unit_rows, r_us, r_them, ep_lines, ep_aux
