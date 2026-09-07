"""Pin Kaggriculture town unlocks without changing any other engine rule.

The engine uses one day-local RNG for weed rolls and the subsequent shop draw.
Consequently the number of empty tiles changes the shop.  The wrapper below lets
the real end-of-day code consume exactly the same random stream, then replaces
only a newly appended shop with the instance's recorded shop for that day.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._engine import import_engine, to_plain

DEFAULT_INSTANCE = REPO_ROOT / "data" / "results" / "fixed-game" / "3910" / "instance.json"
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
_PATCH_LOCK = threading.RLock()


def load_instance(path: str | Path = DEFAULT_INSTANCE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schedule_by_day(instance: dict[str, Any]) -> dict[int, str]:
    return {int(row["day"]): str(row["shop"]) for row in instance["town"]["shop_unlocks"]}


@contextmanager
def pinned_town(instance: dict[str, Any]) -> Iterator[None]:
    """Temporarily patch only the post-weed shop result.

    This is process-global because the engine resolves ``_end_of_day`` through
    its module globals.  A lock prevents two local simulators from overlapping.
    """
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    wanted = schedule_by_day(instance)
    original = kg._end_of_day

    def pinned_end_of_day(state: Any, env: Any, day: int) -> None:
        town = state[0].observation.town
        before = len(town.get("unlocked_shops", []))
        original(state, env, day)
        next_day = day + 1
        shops = town.get("unlocked_shops", [])
        if len(shops) > before:
            if next_day not in wanted:
                raise RuntimeError(f"instance has no pinned shop for unlock day {next_day}")
            shops[-1] = wanted[next_day]

    with _PATCH_LOCK:
        kg._end_of_day = pinned_end_of_day
        try:
            yield
        finally:
            kg._end_of_day = original


def pinned_interpreter(instance: dict[str, Any]):
    """Return an interpreter wrapper suitable for assigning to ``env.interpreter``."""
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    def interpret(state: Any, env: Any):
        with pinned_town(instance):
            return kg.interpreter(state, env)

    return interpret


def make_env(instance: dict[str, Any], *, debug: bool = True):
    ke = import_engine()
    cfg = copy.deepcopy(instance["configuration"])
    cfg["seed"] = int(instance["seed"])
    env = ke.make("kaggriculture", configuration=cfg, debug=debug)
    env.interpreter = pinned_interpreter(instance)
    return env


def _run_actions(instance: dict[str, Any], build_once: bool, pinned: bool) -> Any:
    ke = import_engine()
    cfg = copy.deepcopy(instance["configuration"])
    cfg["seed"] = int(instance["seed"])
    env = ke.make("kaggriculture", configuration=cfg, debug=True)
    if pinned:
        env.interpreter = pinned_interpreter(instance)

    def player(obs: Any):
        if build_once and int(obs.step) == 0:
            return {"farmer": ["BUILD_COOP"], "hands": [], "market": []}
        return PASS

    env.run([player, "pass"])
    return env


def verify(instance_path: str | Path = DEFAULT_INSTANCE) -> dict[str, Any]:
    """Run the two acceptance checks and return machine-readable evidence."""
    instance = load_instance(instance_path)
    plain_unpinned = _run_actions(instance, False, False)
    plain_pinned = _run_actions(instance, False, True)
    built_unpinned = _run_actions(instance, True, False)
    built_pinned = _run_actions(instance, True, True)

    unpinned_obs = [to_plain(s[0].observation) for s in plain_unpinned.steps]
    pinned_obs = [to_plain(s[0].observation) for s in plain_pinned.steps]
    pass_shops = pinned_obs[-1]["town"]["unlocked_shops"]
    built_pin_shops = to_plain(built_pinned.steps[-1][0].observation)["town"]["unlocked_shops"]
    built_raw_shops = to_plain(built_unpinned.steps[-1][0].observation)["town"]["unlocked_shops"]
    result = {
        "pass_unpinned_equals_pinned_all_720_observations": unpinned_obs == pinned_obs,
        "pass_final_bank_unpinned": unpinned_obs[-1]["farms"][0]["money"],
        "pass_final_bank_pinned": pinned_obs[-1]["farms"][0]["money"],
        "pass_pinned_shops": pass_shops,
        "build_once_unpinned_shops": built_raw_shops,
        "build_once_pinned_shops": built_pin_shops,
        "different_actions_same_pinned_shops": pass_shops == built_pin_shops,
        "coupling_demonstrated_unpinned_shop_change": pass_shops != built_raw_shops,
    }
    if not all(result[k] for k in (
        "pass_unpinned_equals_pinned_all_720_observations",
        "different_actions_same_pinned_shops",
        "coupling_demonstrated_unpinned_shop_change",
    )):
        raise AssertionError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.instance)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
