"""Solve one pinned pass/pass Kaggriculture seed and emit a training record."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fixed_game import build_instance
from scripts.fixed_game.pin import PASS
from scripts.fixed_game.sim import Sim, replay

RESULTS = ROOT / "data" / "results" / "fixed-game"
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
OWNED_SEARCH_FILES = ("search.py", "execute.py", "sim.py", "pin.py", "__init__.py")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_instance(seed: int, output_dir: Path) -> dict[str, Any]:
    unlocks = build_instance.pass_shop_unlocks(seed)
    instance = build_instance.build(seed, unlocks)
    _write_json(output_dir / "instance.json", instance)
    (output_dir / "instance.md").write_text(build_instance.markdown(instance), encoding="utf-8")
    pin = {
        "schema_version": 1,
        "seed": seed,
        "source": "unmodified engine pass/pass run",
        "shop_unlocks": instance["town"]["shop_unlocks"],
        "semantics": instance["town"]["pin_semantics"],
    }
    _write_json(output_dir / "pin.json", pin)
    return instance


def _stage_search(instance_path: Path, output_dir: Path, seed: int) -> Path:
    """Create an isolated exact copy because search.py hard-codes seed 3910 paths."""
    stage = output_dir / "_runtime"
    if stage.exists():
        shutil.rmtree(stage)
    fixed = stage / "scripts" / "fixed_game"
    fixed.mkdir(parents=True)
    source = ROOT / "scripts" / "fixed_game"
    for name in OWNED_SEARCH_FILES:
        source_path = source / name
        if source_path.exists():
            shutil.copy2(source_path, fixed / name)
        else:
            # A builder may be replacing a tracked file in the shared worktree.
            # The last committed CLI is the safe backward-compatible fallback.
            payload = subprocess.check_output(["git", "show", f"HEAD:scripts/fixed_game/{name}"], cwd=ROOT)
            (fixed / name).write_bytes(payload)
    shutil.copytree(source / "schedules", fixed / "schedules")
    shutil.copy2(ROOT / "scripts" / "_engine.py", stage / "scripts" / "_engine.py")
    init = ROOT / "scripts" / "__init__.py"
    if init.exists():
        shutil.copy2(init, stage / "scripts" / "__init__.py")
    for staged_seed in {3910, seed}:
        staged_result = stage / "data" / "results" / "fixed-game" / str(staged_seed)
        staged_result.mkdir(parents=True)
        shutil.copy2(instance_path, staged_result / "instance.json")
    prior = RESULTS / "3910" / "search" / "best-schedule.json"
    if prior.exists():
        prior_dir = stage / "data" / "results" / "fixed-game" / "3910" / "search"
        prior_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prior, prior_dir / "best-schedule.json")
    return stage


def _run_search(stage: Path, seed: int, minutes: float, workers: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(stage / "scripts" / "fixed_game" / "search.py"),
               "--seed", str(seed), "--minutes", str(minutes), "--workers", str(workers),
               "--evaluations", "10000000"]
    search_source = (stage / "scripts" / "fixed_game" / "search.py").read_text(encoding="utf-8")
    modern_cli = 'add_argument("--instance"' in search_source
    if modern_cli:
        command.extend(["--instance", str(stage / "data" / "results" / "fixed-game" / str(seed) / "instance.json")])
    completed = subprocess.run(command, cwd=stage, env=env, check=True)
    search_dir = stage / "data" / "results" / "fixed-game" / str(seed if modern_cli else 3910) / "search"
    return {"command": command, "returncode": completed.returncode, "directory": search_dir}


def _run_polish(stage: Path, seed: int, workers: int, actions: Path) -> Path:
    fixed = stage / "scripts" / "fixed_game"
    shutil.copy2(ROOT / "scripts" / "fixed_game" / "polish.py", fixed / "polish.py")
    staged_actions = stage / "best-actions.json"
    shutil.copy2(actions, staged_actions)
    agent = stage / "tape_agent.py"
    agent.write_text(
        "import json\nfrom pathlib import Path\n"
        "A=json.loads((Path(__file__).parent/'best-actions.json').read_text())\n"
        "def agent(obs): return A[min(int(obs.step),len(A)-1)]\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(fixed / "polish.py"), "--agent", str(agent),
               "--seed", str(seed), "--chains", str(min(workers, 6))]
    subprocess.run(command, cwd=stage, env=env, check=True)
    return stage / "data" / "results" / "fixed-game" / "3910" / "polish"


def _copy_directory(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def _replay_dataset(actions: list[dict[str, Any]], instance_path: Path) -> dict[str, Any]:
    """Replay through the real interpreter while capturing exact observations and sales."""
    import kaggle_environments.envs.kaggriculture.kaggriculture as kg

    revenue: dict[str, float] = defaultdict(float)
    sold: Counter[str] = Counter()
    current = {"turn": -1, "day": -1}
    original_commit = kg._commit_unit

    def watched_commit(op, item, price, farm, private, market, shed_capacity=100):
        ok = original_commit(op, item, price, farm, private, market, shed_capacity)
        if ok and op == "SELL":
            sold[str(item)] += 1
            revenue[str(item)] += float(price)
        return ok

    sim = Sim(instance_path)
    records = []
    curve: dict[int, float] = {}
    hands: dict[int, int] = defaultdict(int)
    kg._commit_unit = watched_commit
    try:
        for turn, action in enumerate(actions):
            if sim.done():
                break
            obs = sim.observation()
            day = int(obs["day"])
            hands[day] = max(hands[day], len(obs["farms"][0].get("hands", [])))
            current.update(turn=turn, day=day)
            records.append({"turn": turn, "observation": obs, "action": copy.deepcopy(action),
                            "applied": True, "terminal": False})
            sim.step(action)
            if sim.done() or sim.day() != day:
                curve[day] = sim.bank()
    finally:
        kg._commit_unit = original_commit
    if not sim.done():
        raise RuntimeError("best tape ended before DONE")
    if len(records) != 719:
        raise AssertionError(f"expected 719 applied records, got {len(records)}")
    final_obs = sim.observation()
    records.append({"turn": 719, "observation": final_obs,
                    "action": copy.deepcopy(actions[719]), "applied": False, "terminal": True})
    return {
        "bank": sim.bank(),
        "bank_curve": [curve[d] for d in range(30)],
        "product_revenue": {p: revenue[p] for p in PRODUCTS},
        "sold_units": {p: sold[p] for p in PRODUCTS},
        "hands_per_day": [hands[d] for d in range(30)],
        "training_records": records,
    }


def _season_drain(instance: dict[str, Any]) -> dict[str, int]:
    return {p: sum(int(row["drain"][p]) for row in instance["town"]["daily_drain"]) for p in PRODUCTS}


def _schedule_profile(schedule: dict[str, Any]) -> dict[str, Any]:
    crops: dict[str, list[list[int]]] = defaultdict(list)
    herd: Counter[str] = Counter()
    hands = []
    for day in schedule["days"]:
        hands.append(int(day.get("hires", 0)))
        for row in day.get("plantings", []):
            crops[str(row["crop"])].append([int(x) for x in row["tile"]])
        for row in day.get("purchases", []):
            if row.get("item") in ("GOOSE", "COW", "SHEEP"):
                herd[str(row["item"])] += int(row.get("qty", 1))
    crop_blocks = {}
    for crop, tiles in crops.items():
        crop_blocks[crop] = {
            "tiles": len(set(map(tuple, tiles))),
            "bounding_box": [min(x for x, _ in tiles), min(y for _, y in tiles),
                             max(x for x, _ in tiles), max(y for _, y in tiles)],
        }
    return {"crop_blocks": crop_blocks, "herd_purchases": dict(herd), "scheduled_hires_by_day": hands}


def _run_bound(instance_path: Path, output_dir: Path, timeout_minutes: float) -> tuple[float | None, str | None]:
    command = [sys.executable, str(ROOT / "scripts" / "fixed_game" / "bound_v2.py"),
               "--instance", str(instance_path), "--output-dir", str(output_dir)]
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=timeout_minutes * 60)
        details = json.loads((output_dir / "bound-details-v2.json").read_text(encoding="utf-8"))
        return float(details["valid"]["objective_final_bank"]), None
    except subprocess.TimeoutExpired:
        return None, f"bound_v2 exceeded {timeout_minutes:g} minutes"


def _summary_markdown(solved: dict[str, Any]) -> str:
    shops = ", ".join(row["shop"] for row in solved["pinned_shop_schedule"])
    drains = sorted(solved["season_drain"].items(), key=lambda x: x[1], reverse=True)
    highlights = ", ".join(f"{p.lower()} {n}" for p, n in drains[:4])
    profile = solved["schedule_profile"]
    crops = ", ".join(f"{p.lower()} {v['tiles']} tiles @ {v['bounding_box']}" for p, v in profile["crop_blocks"].items()) or "none"
    herd = ", ".join(f"{n} {a.lower()}" for a, n in profile["herd_purchases"].items()) or "none"
    return "\n".join([
        f"# Solved fixed game: seed {solved['seed']}", "",
        f"Best verified bank: **${solved['best_bank']:,.0f}**. Bound: " +
        (f"**${solved['bound']:,.0f}**." if solved["bound"] is not None else "not computed."), "",
        f"Wall time: {solved['wall_minutes']:.1f} minutes; search budget {solved['search_minutes']:.1f} minutes on {solved['workers']} workers.", "",
        f"Shops: {shops}.", "",
        f"Season drain highlights: {highlights}.", "",
        f"Winning schedule: crops {crops}; herd {herd}; scheduled hands range "
        f"{min(profile['scheduled_hires_by_day'])}-{max(profile['scheduled_hires_by_day'])}/day.", "",
        "The 720 observation/action records are embedded in `solved.json`; record 719 is terminal and its padded PASS action is not applied.", "",
    ])


def _update_index(solved: dict[str, Any]) -> None:
    path = RESULTS / "INDEX.md"
    header = "# Fixed-game solved dataset\n\n| Seed | Shops | Drain highlights | Best bank | Bound | Wall min |\n|---:|---|---|---:|---:|---:|\n"
    old = path.read_text(encoding="utf-8") if path.exists() else header
    lines = [line for line in old.splitlines() if not line.startswith(f"| {solved['seed']} |")]
    shops = ", ".join(row["shop"].replace("_", " ") for row in solved["pinned_shop_schedule"])
    drains = sorted(solved["season_drain"].items(), key=lambda x: x[1], reverse=True)[:3]
    highlights = ", ".join(f"{p.lower()} {n}" for p, n in drains)
    bound = f"${solved['bound']:,.0f}" if solved["bound"] is not None else "—"
    lines.append(f"| {solved['seed']} | {shops} | {highlights} | ${solved['best_bank']:,.0f} | {bound} | {solved['wall_minutes']:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def solve(seed: int, minutes: float, workers: int, with_bound: bool, with_polish: bool,
          bound_timeout_minutes: float = 10.0) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    started = time.time()
    output_dir = RESULTS / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    instance = _make_instance(seed, output_dir)
    stage = _stage_search(output_dir / "instance.json", output_dir, seed)
    try:
        search_run = _run_search(stage, seed, minutes, workers)
        search_dir = output_dir / "search"
        _copy_directory(search_run["directory"], search_dir)
        action_path = search_dir / "best-actions.json"
        if with_polish:
            polish_dir = _run_polish(stage, seed, workers, action_path)
            _copy_directory(polish_dir, output_dir / "polish")
            polished = output_dir / "polish" / "best-actions.json"
            if replay(json.loads(polished.read_text(encoding="utf-8")), output_dir / "instance.json") >= replay(json.loads(action_path.read_text(encoding="utf-8")), output_dir / "instance.json"):
                action_path = polished
        actions = json.loads(action_path.read_text(encoding="utf-8"))
        schedule = json.loads((search_dir / "best-schedule.json").read_text(encoding="utf-8"))
        verified = replay(actions, output_dir / "instance.json")
        dataset = _replay_dataset(actions, output_dir / "instance.json")
        if dataset["bank"] != verified:
            raise AssertionError({"replay": verified, "dataset_replay": dataset["bank"]})
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    bound = None
    bound_error = None
    if with_bound:
        bound, bound_error = _run_bound(output_dir / "instance.json", output_dir, bound_timeout_minutes)
    search_summary = json.loads((output_dir / "search" / "summary.json").read_text(encoding="utf-8"))
    solved = {
        "schema_version": 1,
        "seed": seed,
        "engine": instance["engine"],
        "pinned_shop_schedule": instance["town"]["shop_unlocks"],
        "season_drain": _season_drain(instance),
        "best_bank": verified,
        "bound": bound,
        "bound_error": bound_error,
        "bank_curve": dataset["bank_curve"],
        "product_revenue": dataset["product_revenue"],
        "sold_units": dataset["sold_units"],
        "hands_per_day": dataset["hands_per_day"],
        "best_schedule": schedule,
        "schedule_profile": _schedule_profile(schedule),
        "actions": actions,
        "training_records": dataset["training_records"],
        "search_minutes": minutes,
        "search_wall_seconds": float(search_summary["wall_seconds"]),
        "workers": workers,
        "polish_requested": with_polish,
        "selected_tape": "polish/best-actions.json" if action_path.parent.name == "polish" else "search/best-actions.json",
        "verification": {"arbiter": "scripts.fixed_game.sim.replay", "real_engine": True,
                         "transitions": 719, "matched_bank": True},
        "wall_minutes": (time.time() - started) / 60,
    }
    _write_json(output_dir / "solved.json", solved)
    (output_dir / "summary.md").write_text(_summary_markdown(solved), encoding="utf-8")
    _update_index(solved)
    print(json.dumps({k: solved[k] for k in ("seed", "best_bank", "bound", "bound_error", "search_wall_seconds", "wall_minutes")}, indent=2))
    return solved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bound", action="store_true")
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--bound-timeout-minutes", type=float, default=10.0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    solve(args.seed, args.minutes, args.workers, args.bound, args.polish, args.bound_timeout_minutes)


if __name__ == "__main__":
    main()
