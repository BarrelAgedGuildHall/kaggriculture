"""Branchable fixed-game simulator backed by the real Kaggriculture interpreter."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._engine import to_plain
from scripts.fixed_game.pin import DEFAULT_INSTANCE, PASS, load_instance, make_env


class Sim:
    """A seat-0 simulator with a permanently passing seat 1.

    The engine interpreter has no mutable module-level game state.  All game
    state is in the two agent states plus invariant ``env.info['seed']`` and
    configuration.  Snapshots therefore copy ``env.state`` only; history is not
    part of transition semantics.
    """

    def __init__(self, instance: dict[str, Any] | str | Path = DEFAULT_INSTANCE):
        self.instance = load_instance(instance) if isinstance(instance, (str, Path)) else copy.deepcopy(instance)
        self.env = make_env(self.instance)

    def snapshot(self) -> Any:
        # The framework, not the Kaggriculture interpreter, derives the next
        # public ``observation.step`` from len(env.steps).  Preserve that cursor
        # along with the interpreter state; the old history values themselves
        # are never read by a transition.
        return {
            "state": copy.deepcopy(self.env.state),
            "steps_len": len(self.env.steps),
            "logs_len": len(self.env.logs),
        }

    def restore(self, state: Any) -> None:
        self.env.state = copy.deepcopy(state["state"])
        del self.env.steps[int(state["steps_len"]):]
        del self.env.logs[int(state["logs_len"]):]

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.env.done:
            raise RuntimeError("fixed game is already done")
        self.env.step([action, PASS])
        return to_plain(self.env.state[0].observation)

    def observation(self) -> dict[str, Any]:
        return to_plain(self.env.state[0].observation)

    def bank(self) -> float:
        return float(self.env.state[0].observation.farms[0].money)

    def day(self) -> int:
        return int(self.env.state[0].observation.day)

    def done(self) -> bool:
        return bool(self.env.done)


def replay(actions: Sequence[dict[str, Any]], instance: dict[str, Any] | str | Path = DEFAULT_INSTANCE) -> float:
    """Final arbiter: replay JSON actions through ``Environment.step`` + real interpreter.

    Kaggle records step 0 as the initialized observation, so a 720-step episode
    consumes 719 transitions.  A conventional 720-action file is accepted; its
    trailing action is ignored after the engine reaches DONE.
    """
    if len(actions) not in (719, 720):
        raise ValueError(f"expected 719 engine transitions or a 720-action padded list, got {len(actions)}")
    sim = Sim(instance)
    for action in actions:
        if sim.done():
            break
        sim.step(action)
    if not sim.done():
        raise RuntimeError("action list ended before the engine reached DONE")
    return sim.bank()


def verify(instance: dict[str, Any] | str | Path = DEFAULT_INSTANCE, *, seed: int = 3910) -> dict[str, Any]:
    """Verify 50 randomized branch excursions against a straight replay."""
    rng = random.Random(seed)
    actions = [copy.deepcopy(PASS) for _ in range(720)]
    # Include harmless but state-changing actions so snapshots cover nontrivial state.
    actions[0] = {"farmer": ["BUILD_COOP"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}
    actions[1] = {"farmer": ["DIG"], "hands": [], "market": []}

    straight = Sim(instance)
    straight_obs = []
    step_times = []
    for action in actions:
        if straight.done():
            break
        t0 = time.perf_counter_ns()
        straight_obs.append(straight.step(action))
        step_times.append(time.perf_counter_ns() - t0)

    branched = Sim(instance)
    branch_points = sorted(rng.sample(range(1, len(straight_obs) - 1), 50))
    snapshot_times = []
    for i, action in enumerate(actions[:len(straight_obs)]):
        if i in branch_points:
            t0 = time.perf_counter_ns()
            snap = branched.snapshot()
            snapshot_times.append(time.perf_counter_ns() - t0)
            alt = copy.deepcopy(PASS)
            branched.step(alt)
            branched.restore(snap)
        got = branched.step(action)
        if got != straight_obs[i]:
            raise AssertionError(f"branch replay diverged at transition {i}")

    arbiter_bank = replay(actions, instance)
    result = {
        "transitions_compared": len(straight_obs),
        "branch_points": len(branch_points),
        "identical_observations_every_transition": True,
        "straight_bank": straight.bank(),
        "branched_bank": branched.bank(),
        "arbiter_bank": arbiter_bank,
        "mean_ms_per_step": sum(step_times) / len(step_times) / 1e6,
        "median_ms_per_step": sorted(step_times)[len(step_times) // 2] / 1e6,
        "mean_ms_per_snapshot": sum(snapshot_times) / len(snapshot_times) / 1e6,
        "median_ms_per_snapshot": sorted(snapshot_times)[len(snapshot_times) // 2] / 1e6,
    }
    if not (result["straight_bank"] == result["branched_bank"] == result["arbiter_bank"]):
        raise AssertionError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actions", nargs="?", type=Path)
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.actions:
        result: Any = {"bank": replay(json.loads(args.actions.read_text(encoding="utf-8")), args.instance)}
    else:
        result = verify(args.instance)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
