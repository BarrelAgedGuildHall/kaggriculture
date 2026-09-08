"""Record fixed-game control banks under an instance's pinned shop schedule."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import multiprocessing as mp
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fixed_game.sim import PASS, Sim

BOTS = {
    "brain_strawberry": ROOT / "bots" / "brain_strawberry" / "main.py",
    "exec_b": ROOT / "bots" / "exec_b" / "main.py",
    "V46": ROOT / "bots" / "v46" / "main.py",
    "tapeselect": ROOT / "bots" / "tapeselect" / "main.py",
}


def _plain_action(action: Any, hands: int) -> dict[str, Any]:
    if not isinstance(action, dict):
        action = {}
    farmer = action.get("farmer", PASS["farmer"])
    farmer = list(farmer) if isinstance(farmer, list) and farmer else ["PASS"]
    raw_hands = action.get("hands", [])
    raw_hands = raw_hands if isinstance(raw_hands, list) else []
    hand_actions = []
    for index in range(hands):
        item = raw_hands[index] if index < len(raw_hands) else ["PASS"]
        hand_actions.append(list(item) if isinstance(item, list) and item else ["PASS"])
    market = action.get("market", [])
    market = [list(order) for order in market[:10] if isinstance(order, list)] if isinstance(market, list) else []
    return {"farmer": farmer, "hands": hand_actions, "market": market}


def _run_one(name: str, agent_path: str, instance_path: str, output: mp.Queue) -> None:
    try:
        spec = importlib.util.spec_from_file_location(f"fixed_control_{name}", agent_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load agent: {agent_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sim = Sim(instance_path)
        while not sim.done():
            obs = sim.observation()
            action = module.agent(copy.deepcopy(obs))
            sim.step(_plain_action(action, len(obs["farms"][0].get("hands", []))))
        output.put({"bank": sim.bank()})
    except BaseException as exc:
        output.put({"error": f"{type(exc).__name__}: {exc}"})


def isolated_bank(name: str, agent_path: Path, instance_path: Path) -> float:
    context = mp.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_run_one, args=(name, str(agent_path), str(instance_path), output))
    process.start()
    try:
        result = output.get(timeout=180)
    except queue.Empty as exc:
        process.terminate()
        raise TimeoutError(f"{name} exceeded 180 seconds") from exc
    finally:
        process.join(timeout=10)
    if process.exitcode not in (0, None):
        raise RuntimeError(f"{name} exited with code {process.exitcode}")
    if "error" in result:
        raise RuntimeError(f"{name}: {result['error']}")
    return float(result["bank"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    instance = args.instance or ROOT / "data" / "results" / "fixed-game" / str(args.seed) / "instance.json"
    output = args.output or instance.parent / "controls.json"
    controls = {name: isolated_bank(name, path, instance) for name, path in BOTS.items()}
    result = {
        "engine": "kaggle-environments 1.32.7",
        "seed": args.seed,
        "seat": 0,
        "opponent": "pass",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "controls": controls,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
