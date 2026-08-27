"""Package-bound turn trace capture, validation, extraction, and reproduction.

The pipeline is measurement-only. It records the exact observation presented to each
seat immediately before the call and the exact value emitted by that seat. It neither
classifies actions nor alters engine state, rewards, agents, or arena records.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from arena import engine as arena_engine
from arena.isolation import SubprocessAgent
from scripts._engine import to_plain
from tools.agent_spec import arity_wrap

from . import (
    MANIFEST_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_SCHEMA_SOURCE = Path(__file__).resolve().parent / "schema" / "turn-trace-v2.schema.json"
STATE_SCHEMA_SOURCE = Path(__file__).resolve().parent / "schema" / "framework-state-v1.schema.json"
BUILTINS = ("pass", "random", "starter")
PACKAGE_HASH_FORMAT = "sha256(canonical-json(package-files-v1))"
GAME_HASH_FORMAT = "sha256(canonical-json(game-identity-v1))"
SEMANTIC_HASH_FORMAT = "sha256(canonical-jsonl(records-without-timing))"
STATE_HASH_FORMAT = "sha256(canonical-jsonl(framework-state-records-v1))"


class TraceContractError(ValueError):
    """Raised when a trace or identity fails closed."""


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TraceContractError(f"cannot read JSON {path}: {exc}") from exc


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not label:
        raise TraceContractError(f"agent label has no safe characters: {value!r}")
    return label


def _engine_identity() -> dict:
    source = arena_engine.engine_dir() / "kaggriculture.py"
    return {
        "name": "kaggriculture",
        "version": arena_engine.engine_version(),
        "sha256": sha256_file(source),
    }


def _copy_package(source: Path, destination: Path) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=False)
    if source.is_file():
        files = [(source, Path("main.py"))]
    else:
        files = []
        for item in sorted(source.rglob("*")):
            if item.is_symlink():
                raise TraceContractError(f"package links are not retained: {item}")
            if not item.is_file() or "__pycache__" in item.parts or item.suffix == ".pyc" or ".git" in item.parts:
                continue
            files.append((item, item.relative_to(source)))
    if not files:
        raise TraceContractError(f"agent package contains no files: {source}")
    inventory = []
    for src, rel in files:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        inventory.append({
            "path": rel.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    if not (destination / "main.py").is_file():
        raise TraceContractError(f"directory package has no main.py: {source}")
    return inventory


def _package_hash(files: list[dict]) -> str:
    return canonical_hash({"format": "package-files-v1", "files": files})


def _snapshot_agent(label: str, spec: str, packages_root: Path, engine: dict) -> dict:
    label = _safe_label(label)
    if spec in BUILTINS:
        descriptor = {
            "format": "engine-builtin-v1",
            "builtin": spec,
            "engine_version": engine["version"],
            "engine_sha256": engine["sha256"],
        }
        package_sha = canonical_hash(descriptor)
        return {
            "label": label,
            "agent_id": f"{label}@{package_sha[:12]}",
            "kind": "builtin",
            "entry": spec,
            "source_spec": spec,
            "package_path": None,
            "package_sha256": package_sha,
            "package_hash_format": PACKAGE_HASH_FORMAT,
            "files": [],
        }

    source = Path(spec)
    if not source.is_absolute():
        source = REPO_ROOT / source
    source = source.resolve()
    try:
        source_spec = source.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise TraceContractError(f"agent package must be inside the repository: {source}") from exc
    if not source.exists():
        raise TraceContractError(f"agent package does not exist: {source_spec}")
    package_path = Path("packages") / label
    inventory = _copy_package(source, packages_root / label)
    package_sha = _package_hash(inventory)
    return {
        "label": label,
        "agent_id": f"{label}@{package_sha[:12]}",
        "kind": "file",
        "entry": "main.py",
        "source_spec": source_spec,
        "package_path": package_path.as_posix(),
        "package_sha256": package_sha,
        "package_hash_format": PACKAGE_HASH_FORMAT,
        "files": inventory,
    }


def _expand_seeds(value) -> list[int]:
    if isinstance(value, list):
        seeds = [int(seed) for seed in value]
    elif isinstance(value, dict) and set(value) == {"start", "count"}:
        start, count = int(value["start"]), int(value["count"])
        seeds = list(range(start, start + count))
    else:
        raise TraceContractError("seeds must be a list or exactly {start, count}")
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise TraceContractError(f"seeds must be non-empty, unique, and non-negative: {seeds}")
    return seeds


def _episode_semantics(episode_steps: int) -> dict:
    if episode_steps < 2:
        raise TraceContractError("episode_steps must leave an initial and terminal state")
    return {
        "episode_steps": episode_steps,
        "stored_states_expected": episode_steps,
        "action_calls_per_seat_expected": episode_steps - 1,
        "initial_state_index": 0,
        "final_action_step": episode_steps - 2,
        "terminal_state_index": episode_steps - 1,
    }


def _full_configuration(base: dict, episode_steps: int) -> dict:
    if "episodeSteps" in base or "seed" in base:
        raise TraceContractError("configuration must not override episodeSteps or seed")
    ke = arena_engine.import_engine()
    requested = {**base, "episodeSteps": episode_steps, "seed": 0}
    env = ke.make("kaggriculture", configuration=requested, debug=False)
    actual = to_plain(env.configuration)
    if int(actual["episodeSteps"]) != episode_steps:
        raise TraceContractError("engine contradicted the configured episodeSteps contract")
    return actual


def _game_identity(p0: dict, p1: dict, seed: int, episode_steps: int,
                   action_turns: int, configuration: dict, engine: dict) -> dict:
    return {
        "format": "game-identity-v1",
        "p0_agent_id": p0["agent_id"],
        "p0_package_sha256": p0["package_sha256"],
        "p1_agent_id": p1["agent_id"],
        "p1_package_sha256": p1["package_sha256"],
        "engine_version": engine["version"],
        "engine_sha256": engine["sha256"],
        "seed": int(seed),
        "action_turns": int(action_turns),
        "episode_steps": int(episode_steps),
        "configuration": configuration,
    }


def _job(p0_label: str, p1_label: str, seed: int, expected: str, agents: dict,
         episode_steps: int, action_turns: int, configuration: dict, engine: dict) -> dict:
    p0, p1 = agents[p0_label], agents[p1_label]
    identity = _game_identity(
        p0, p1, seed, episode_steps, action_turns, configuration, engine
    )
    return {
        "game_id": canonical_hash(identity),
        "game_hash_format": GAME_HASH_FORMAT,
        "p0": p0_label,
        "p1": p1_label,
        "seed": int(seed),
        "expected": expected,
        "identity": identity,
    }


def prepare_bundle(config_path: Path, bundle: Path) -> dict:
    if bundle.exists() and any(bundle.iterdir()):
        raise TraceContractError(f"output bundle is not empty: {bundle}")
    bundle.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    if config.get("schema_version") != "kaggriculture.trace-config.v2":
        raise TraceContractError("unsupported trace config schema_version")
    episode_steps = int(config.get("episode_steps", 0))
    semantics = _episode_semantics(episode_steps)
    action_turns = semantics["action_calls_per_seat_expected"]
    engine = _engine_identity()
    configuration = _full_configuration(
        dict(config.get("configuration") or {}), episode_steps
    )
    packages_root = bundle / "packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    agents = {}
    for label, value in sorted((config.get("agents") or {}).items()):
        if not isinstance(value, dict) or set(value) != {"spec"}:
            raise TraceContractError(f"agent {label!r} must contain exactly a spec")
        agents[label] = _snapshot_agent(label, str(value["spec"]), packages_root, engine)
    if len(agents) < 2:
        raise TraceContractError("at least two exact agents are required")
    jobs = []
    for pairing in config.get("pairings") or []:
        a, b = pairing.get("agent_a"), pairing.get("agent_b")
        if a not in agents or b not in agents:
            raise TraceContractError(f"pairing names unknown agent: {a!r}, {b!r}")
        expected = pairing.get("expected", "normal")
        if expected not in ("normal", "fault"):
            raise TraceContractError(f"expected must be normal or fault, got {expected!r}")
        for seed in _expand_seeds(pairing.get("seeds")):
            jobs.append(_job(
                a, b, seed, expected, agents, episode_steps, action_turns,
                configuration, engine,
            ))
            if pairing.get("swap_seats", True):
                jobs.append(_job(
                    b, a, seed, expected, agents, episode_steps, action_turns,
                    configuration, engine,
                ))
    ids = [job["game_id"] for job in jobs]
    if not jobs or len(ids) != len(set(ids)):
        raise TraceContractError("the expanded plan has no jobs or duplicate game identities")

    trace_schema_target = bundle / "schema" / TRACE_SCHEMA_SOURCE.name
    state_schema_target = bundle / "schema" / STATE_SCHEMA_SOURCE.name
    trace_schema_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TRACE_SCHEMA_SOURCE, trace_schema_target)
    shutil.copyfile(STATE_SCHEMA_SOURCE, state_schema_target)
    normalized_config = copy.deepcopy(config)
    write_json(bundle / "input-config.json", normalized_config)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "trace_schema": {
            "version": TRACE_SCHEMA_VERSION,
            "path": "schema/turn-trace-v2.schema.json",
            "sha256": sha256_file(trace_schema_target),
        },
        "state_schema": {
            "version": STATE_SCHEMA_VERSION,
            "path": "schema/framework-state-v1.schema.json",
            "sha256": sha256_file(state_schema_target),
        },
        "input_config": {
            "path": "input-config.json",
            "sha256": sha256_file(bundle / "input-config.json"),
        },
        "engine": engine,
        "semantics": semantics,
        "action_turns": action_turns,
        "episode_steps": episode_steps,
        "configuration": configuration,
        "agents": agents,
        "jobs": jobs,
    }
    write_json(bundle / "plan.json", plan)
    return plan


def _gzip_records(path: Path, records: list[dict]) -> tuple[str, str, int]:
    semantic = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as out:
            for record in records:
                line = canonical_bytes(record) + b"\n"
                out.write(line)
                semantic.update(canonical_bytes(_semantic_record(record)) + b"\n")
    return sha256_file(path), semantic.hexdigest(), path.stat().st_size


def _gzip_state_records(path: Path, records: list[dict]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as out:
            for record in records:
                out.write(canonical_bytes(record) + b"\n")
    return sha256_file(path), path.stat().st_size


def _read_gzip_records(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as src:
            for number, line in enumerate(src, 1):
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise TraceContractError(f"invalid JSON at {path}:{number}") from exc
    except (OSError, EOFError) as exc:
        raise TraceContractError(f"cannot read trace {path}: {exc}") from exc


def _semantic_record(record: dict) -> dict:
    value = dict(record)
    value.pop("timing", None)
    return value


def _duration_summary(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"calls": 0, "mean_ns": 0, "p50_ns": 0, "p99_ns": 0, "max_ns": 0}
    return {
        "calls": len(ordered),
        "mean_ns": round(statistics.fmean(ordered)),
        "p50_ns": ordered[len(ordered) // 2],
        "p99_ns": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "max_ns": ordered[-1],
    }


def _fault_message_sha256(exc: Exception) -> str:
    """Hash the terminal exception line, not a path-bearing subprocess traceback."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    terminal = lines[-1] if lines else type(exc).__name__
    return hashlib.sha256(terminal.encode("utf-8", errors="replace")).hexdigest()


def _fault_fingerprint(faults: list[dict]) -> str | None:
    if not faults:
        return None
    return canonical_hash({"format": "fault-fingerprint-v1", "faults": faults})


def _load_agent(agent: dict, bundle: Path, configuration):
    if agent["kind"] == "builtin":
        import kaggle_environments.envs.kaggriculture.kaggriculture as engine_module

        fn = {
            "pass": engine_module.pass_agent,
            "random": engine_module.random_agent,
            "starter": engine_module.starter_agent,
        }[agent["entry"]]
        return arity_wrap(fn, configuration), None
    entry = bundle / agent["package_path"] / agent["entry"]
    child = SubprocessAgent(entry, name=agent["agent_id"])
    return lambda observation: child(observation, configuration), child


def _seat_identity(job: dict, seat: int, agents: dict, engine: dict) -> dict:
    own = agents[job[f"p{seat}"]]
    other = agents[job[f"p{1 - seat}"]]
    return {
        "agent_id": own["agent_id"],
        "agent_package_sha256": own["package_sha256"],
        "opponent_id": other["agent_id"],
        "opponent_package_sha256": other["package_sha256"],
        "engine_version": engine["version"],
        "engine_sha256": engine["sha256"],
        "seed": job["seed"],
        "seat": seat,
        "game_id": job["game_id"],
    }


def capture_episode(job: dict, agents: dict, engine: dict, configuration: dict,
                    episode_steps: int, action_turns: int, bundle_text: str) -> dict:
    bundle = Path(bundle_text)
    if _engine_identity() != engine:
        raise TraceContractError("engine identity changed between planning and capture")
    ke = arena_engine.import_engine()
    requested = {key: value for key, value in configuration.items() if key not in ("seed",)}
    requested.update({"episodeSteps": episode_steps, "seed": job["seed"]})
    env = ke.make("kaggriculture", configuration=requested, debug=False)
    if to_plain(env.configuration) != configuration or int(to_plain(env.info)["seed"]) != job["seed"]:
        raise TraceContractError("engine configuration or seed contradicted the planned identity")
    loaded = []
    closers = []
    for seat in (0, 1):
        fn, closer = _load_agent(agents[job[f"p{seat}"]], bundle, env.configuration)
        loaded.append(fn)
        if closer is not None:
            closers.append(closer)

    records = []
    observations_by_state = {}
    durations = [[], []]
    faults = []
    wall_start = time.perf_counter_ns()
    env.reset()
    try:
        for turn in range(action_turns):
            if env.done:
                raise TraceContractError(f"engine terminated before action turn {turn}")
            # Shared observation fields physically live only on state[0]. The framework's
            # agent runner calls this exact pinned-core method to materialize each seat's
            # runtime view; retaining env.state[i].observation directly would omit `step`
            # from seat 1 and would not be the pre-action observation the agent received.
            shared_states = [env._Environment__get_shared_state(seat) for seat in (0, 1)]
            observations = [to_plain(state.observation) for state in shared_states]
            if any(int(obs.get("step", -1)) != turn for obs in observations):
                raise TraceContractError(f"engine observation step contradicted action turn {turn}")
            observations_by_state[turn] = observations
            actions = [None, None]
            turn_fault = False
            for seat in (0, 1):
                started = time.perf_counter_ns()
                fault = None
                try:
                    actions[seat] = to_plain(loaded[seat](shared_states[seat].observation))
                    if not isinstance(actions[seat], dict):
                        raise TypeError("emitted action is not a JSON object")
                except Exception as exc:  # the trace must retain, then stop, on any agent fault
                    actions[seat] = None
                    fault = {
                        "type": type(exc).__name__,
                        "message_sha256": _fault_message_sha256(exc),
                    }
                    faults.append({"turn": turn, "seat": seat, **fault})
                    turn_fault = True
                duration = time.perf_counter_ns() - started
                durations[seat].append(duration)
                records.append({
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "game_id": job["game_id"],
                    "turn": turn,
                    "state_index": turn,
                    "seat": seat,
                    "identity": _seat_identity(job, seat, agents, engine),
                    "pre_action_observation": observations[seat],
                    "emitted_action": actions[seat],
                    "completion": {"emitted": fault is None, "faulted": fault is not None},
                    "fault": fault,
                    "timing": {"duration_ns": duration},
                })
            if turn_fault:
                break
            env.step(actions)
        wall_ns = time.perf_counter_ns() - wall_start
    finally:
        for closer in closers:
            closer.close()

    stored_state_records = [{
        "schema_version": STATE_SCHEMA_VERSION,
        "game_id": job["game_id"],
        "state_index": state_index,
        "framework_state": to_plain(state),
        "runtime_observations": observations_by_state.get(state_index),
    } for state_index, state in enumerate(env.steps)]
    normal = (
        not faults
        and len(records) == action_turns * 2
        and len(stored_state_records) == episode_steps
        and env.done
    )
    final_statuses = [str(state.status) for state in env.state]
    trace_rel = Path("traces") / f"{job['game_id']}.jsonl.gz"
    trace_path = bundle / trace_rel
    exact_sha, semantic_sha, size = _gzip_records(trace_path, records)
    states_rel = Path("states") / f"{job['game_id']}.jsonl.gz"
    states_path = bundle / states_rel
    states_sha, states_size = _gzip_state_records(states_path, stored_state_records)
    calls_by_seat = [sum(record["seat"] == seat for record in records) for seat in (0, 1)]
    semantics = _episode_semantics(episode_steps)
    return {
        "game_id": job["game_id"],
        "p0_agent_id": agents[job["p0"]]["agent_id"],
        "p1_agent_id": agents[job["p1"]]["agent_id"],
        "seed": job["seed"],
        "expected": job["expected"],
        "completion": {
            "normal": normal,
            "episode_steps_configured": episode_steps,
            "stored_states_expected": semantics["stored_states_expected"],
            "stored_states": len(stored_state_records),
            "action_calls_per_seat_expected": semantics["action_calls_per_seat_expected"],
            "action_calls_by_seat": calls_by_seat,
            "final_action_step_expected": semantics["final_action_step"],
            "terminal_state_index_expected": semantics["terminal_state_index"],
            "statuses": final_statuses,
        },
        "faults": faults,
        "fault_fingerprint_sha256": _fault_fingerprint(faults),
        "timing": {
            "wall_ns": wall_ns,
            "seat0": _duration_summary(durations[0]),
            "seat1": _duration_summary(durations[1]),
        },
        "trace": {
            "path": trace_rel.as_posix(),
            "sha256": exact_sha,
            "semantic_sha256": semantic_sha,
            "semantic_hash_format": SEMANTIC_HASH_FORMAT,
            "bytes": size,
            "records": len(records),
        },
        "states": {
            "path": states_rel.as_posix(),
            "sha256": states_sha,
            "semantic_sha256": states_sha,
            "semantic_hash_format": STATE_HASH_FORMAT,
            "bytes": states_size,
            "records": len(stored_state_records),
        },
    }


def _manifest_semantic_view(manifest: dict) -> dict:
    value = copy.deepcopy(manifest)
    value.pop("semantic_sha256", None)
    value.get("execution", {}).pop("workers", None)
    for episode in value.get("episodes", []):
        episode.pop("timing", None)
        episode.get("trace", {}).pop("sha256", None)
        episode.get("trace", {}).pop("bytes", None)
        episode.get("states", {}).pop("sha256", None)
        episode.get("states", {}).pop("bytes", None)
    return value


def _write_semantic_extraction(bundle: Path, episodes: list[dict], out_rel: str = "extracted/turns.jsonl.gz") -> dict:
    out = bundle / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as dst:
            for episode in episodes:
                for record in _read_gzip_records(bundle / episode["trace"]["path"]):
                    dst.write(canonical_bytes(_semantic_record(record)) + b"\n")
                    count += 1
    return {"path": out_rel, "sha256": sha256_file(out), "bytes": out.stat().st_size,
            "records": count, "timing_excluded": True}


def execute_plan(plan: dict, bundle: Path, workers: int) -> dict:
    if workers < 1:
        raise TraceContractError("workers must be positive")
    args = [(job, plan["agents"], plan["engine"], plan["configuration"],
             plan["episode_steps"], plan["action_turns"], str(bundle))
            for job in plan["jobs"]]
    if workers == 1:
        episodes = [capture_episode(*values) for values in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(capture_episode, *values) for values in args]
            episodes = [future.result() for future in futures]
    extraction = _write_semantic_extraction(bundle, episodes)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "trace_schema": plan["trace_schema"],
        "state_schema": plan["state_schema"],
        "plan": {"path": "plan.json", "sha256": sha256_file(bundle / "plan.json")},
        "engine": plan["engine"],
        "execution": {"episode_count": len(episodes), "workers": workers},
        "episodes": episodes,
        "extraction": extraction,
    }
    manifest["semantic_sha256"] = canonical_hash(_manifest_semantic_view(manifest))
    write_json(bundle / "manifest.json", manifest)
    return manifest


def _verify_package(bundle: Path, agent: dict) -> None:
    if agent["kind"] == "builtin":
        expected = canonical_hash({
            "format": "engine-builtin-v1",
            "builtin": agent["entry"],
            "engine_version": _engine_identity()["version"],
            "engine_sha256": _engine_identity()["sha256"],
        })
        if expected != agent["package_sha256"]:
            raise TraceContractError(f"builtin package identity mismatch: {agent['agent_id']}")
        return
    root = bundle / agent["package_path"]
    inventory = []
    for declared in agent["files"]:
        path = root / declared["path"]
        if not path.is_file():
            raise TraceContractError(f"package file missing: {path}")
        actual = {"path": declared["path"], "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != declared:
            raise TraceContractError(f"package file identity mismatch: {path}")
        inventory.append(actual)
    actual_paths = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    if actual_paths != [item["path"] for item in inventory]:
        raise TraceContractError(f"package inventory mismatch: {agent['agent_id']}")
    if _package_hash(inventory) != agent["package_sha256"]:
        raise TraceContractError(f"package hash mismatch: {agent['agent_id']}")


def _validate_plan_contract(bundle: Path, plan: dict) -> None:
    config = read_json(bundle / "input-config.json")
    if config.get("schema_version") != "kaggriculture.trace-config.v2":
        raise TraceContractError("unsupported retained trace config schema_version")
    episode_steps = int(config.get("episode_steps", 0))
    semantics = _episode_semantics(episode_steps)
    if plan.get("episode_steps") != episode_steps:
        raise TraceContractError("configured/plan episodeSteps mismatch")
    if plan.get("action_turns") != semantics["action_calls_per_seat_expected"]:
        raise TraceContractError("action calls do not derive from configured episodeSteps")
    if plan.get("semantics") != semantics:
        raise TraceContractError("stored-state/action-call semantics mismatch")
    expected_configuration = _full_configuration(
        dict(config.get("configuration") or {}), episode_steps
    )
    if plan.get("configuration") != expected_configuration:
        raise TraceContractError("configured/plan engine configuration mismatch")
    for job in plan.get("jobs", []):
        expected = canonical_hash(_game_identity(
            plan["agents"][job["p0"]], plan["agents"][job["p1"]], job["seed"],
            episode_steps, plan["action_turns"], plan["configuration"], plan["engine"],
        ))
        if job.get("game_id") != expected or job.get("identity", {}).get("episode_steps") != episode_steps:
            raise TraceContractError(f"planned game identity mismatch: {job.get('game_id')}")


def _validate_episode(bundle: Path, plan: dict, job: dict, episode: dict) -> dict:
    if episode["game_id"] != job["game_id"]:
        raise TraceContractError("episode order or game identity mismatch")
    expected_game = canonical_hash(_game_identity(
        plan["agents"][job["p0"]], plan["agents"][job["p1"]], job["seed"],
        plan["episode_steps"], plan["action_turns"], plan["configuration"], plan["engine"],
    ))
    if expected_game != job["game_id"]:
        raise TraceContractError(f"game identity hash mismatch: {job['game_id']}")
    trace_path = bundle / episode["trace"]["path"]
    if sha256_file(trace_path) != episode["trace"]["sha256"] or trace_path.stat().st_size != episode["trace"]["bytes"]:
        raise TraceContractError(f"trace file hash/size mismatch: {trace_path}")
    records = list(_read_gzip_records(trace_path))
    semantic = hashlib.sha256()
    per_seat = [0, 0]
    faults = []
    by_turn: dict[int, set[int]] = {}
    calls = {}
    for record in records:
        semantic.update(canonical_bytes(_semantic_record(record)) + b"\n")
        if record.get("schema_version") != TRACE_SCHEMA_VERSION or record.get("game_id") != job["game_id"]:
            raise TraceContractError(f"trace schema/game mismatch: {job['game_id']}")
        turn, seat = record.get("turn"), record.get("seat")
        if not isinstance(turn, int) or seat not in (0, 1):
            raise TraceContractError(f"invalid turn/seat: {job['game_id']}")
        if record.get("state_index") != turn:
            raise TraceContractError(f"action/state index mismatch: {job['game_id']} turn={turn} seat={seat}")
        if (turn, seat) in calls:
            raise TraceContractError(f"duplicate action call: {job['game_id']} turn={turn} seat={seat}")
        calls[(turn, seat)] = record
        by_turn.setdefault(turn, set()).add(seat)
        per_seat[seat] += 1
        expected_identity = _seat_identity(job, seat, plan["agents"], plan["engine"])
        if record.get("identity") != expected_identity:
            raise TraceContractError(f"seat identity mismatch: {job['game_id']} turn={turn} seat={seat}")
        obs = record.get("pre_action_observation")
        if not isinstance(obs, dict) or int(obs.get("step", -1)) != turn or int(obs.get("player", -1)) != seat:
            raise TraceContractError(f"pre-action observation mismatch: {job['game_id']} turn={turn} seat={seat}")
        completion = record.get("completion") or {}
        fault = record.get("fault")
        action = record.get("emitted_action")
        if fault is None:
            if completion != {"emitted": True, "faulted": False} or not isinstance(action, dict):
                raise TraceContractError(f"missing emitted action: {job['game_id']} turn={turn} seat={seat}")
        else:
            if completion != {"emitted": False, "faulted": True} or action is not None:
                raise TraceContractError(f"fault/action contradiction: {job['game_id']} turn={turn} seat={seat}")
            faults.append({"turn": turn, "seat": seat, **fault})
        duration = (record.get("timing") or {}).get("duration_ns")
        if not isinstance(duration, int) or duration < 0:
            raise TraceContractError(f"invalid timing: {job['game_id']} turn={turn} seat={seat}")
    turns_seen = sorted(by_turn)
    if turns_seen != list(range(len(turns_seen))) or any(by_turn[turn] != {0, 1} for turn in turns_seen):
        raise TraceContractError(f"truncated or unpaired turns: {job['game_id']}")
    if len(records) != episode["trace"]["records"] or semantic.hexdigest() != episode["trace"]["semantic_sha256"]:
        raise TraceContractError(f"trace record/semantic hash mismatch: {job['game_id']}")
    if faults != episode["faults"]:
        raise TraceContractError(f"fault manifest mismatch: {job['game_id']}")
    if _fault_fingerprint(faults) != episode.get("fault_fingerprint_sha256"):
        raise TraceContractError(f"fault fingerprint mismatch: {job['game_id']}")
    if job["expected"] == "normal" and (
        faults or per_seat != [plan["action_turns"], plan["action_turns"]]
    ):
        raise TraceContractError(
            f"normal trace is not {plan['action_turns']}-call complete: {job['game_id']}"
        )

    states_path = bundle / episode["states"]["path"]
    states_descriptor = episode["states"]
    if (
        sha256_file(states_path) != states_descriptor["sha256"]
        or states_path.stat().st_size != states_descriptor["bytes"]
        or states_descriptor.get("semantic_sha256") != states_descriptor["sha256"]
    ):
        raise TraceContractError(f"stored state file hash/size mismatch: {states_path}")
    state_records = list(_read_gzip_records(states_path))
    if len(state_records) != states_descriptor["records"]:
        raise TraceContractError(f"stored state record count mismatch: {job['game_id']}")
    if [state.get("state_index") for state in state_records] != list(range(len(state_records))):
        raise TraceContractError(f"stored state indices are not contiguous: {job['game_id']}")
    for state_record in state_records:
        state_index = state_record["state_index"]
        if (
            state_record.get("schema_version") != STATE_SCHEMA_VERSION
            or state_record.get("game_id") != job["game_id"]
        ):
            raise TraceContractError(f"stored state schema/game mismatch: {job['game_id']}")
        framework_state = state_record.get("framework_state")
        if not isinstance(framework_state, list) or len(framework_state) != 2:
            raise TraceContractError(f"stored framework state shape mismatch: {job['game_id']} state={state_index}")
        runtime_observations = state_record.get("runtime_observations")
        state_calls = [calls.get((state_index, seat)) for seat in (0, 1)]
        if any(state_calls):
            if not isinstance(runtime_observations, list) or len(runtime_observations) != 2:
                raise TraceContractError(f"stored runtime observation missing: {job['game_id']} state={state_index}")
            for seat, call in enumerate(state_calls):
                if call is None or call["pre_action_observation"] != runtime_observations[seat]:
                    raise TraceContractError(f"action/state observation mismatch: {job['game_id']} state={state_index} seat={seat}")
                if call["fault"] is None and (job["expected"] == "normal" or state_index < len(state_records) - 1):
                    # The pinned framework stores a submitted action on the state
                    # produced by that call, one index after its pre-action state.
                    action_state_index = state_index + 1
                    stored_action = state_records[action_state_index]["framework_state"][seat].get("action")
                    if stored_action != call["emitted_action"]:
                        raise TraceContractError(f"emitted action/stored state mismatch: {job['game_id']} state={state_index} seat={seat}")
        elif runtime_observations is not None:
            raise TraceContractError(f"stored state has observations without calls: {job['game_id']} state={state_index}")

    completion = episode["completion"]
    if completion["action_calls_by_seat"] != per_seat:
        raise TraceContractError(f"completion counts mismatch: {job['game_id']}")
    semantics = plan["semantics"]
    expected_completion = {
        "episode_steps_configured": plan["episode_steps"],
        "stored_states_expected": semantics["stored_states_expected"],
        "action_calls_per_seat_expected": semantics["action_calls_per_seat_expected"],
        "final_action_step_expected": semantics["final_action_step"],
        "terminal_state_index_expected": semantics["terminal_state_index"],
    }
    for key, value in expected_completion.items():
        if completion.get(key) != value:
            raise TraceContractError(f"completion semantic mismatch ({key}): {job['game_id']}")
    if completion.get("stored_states") != len(state_records):
        raise TraceContractError(f"stored state completion count mismatch: {job['game_id']}")
    if job["expected"] == "normal":
        if faults or per_seat != [plan["action_turns"], plan["action_turns"]]:
            raise TraceContractError(f"normal trace is not {plan['action_turns']}-call complete: {job['game_id']}")
        if sorted(by_turn) != list(range(plan["action_turns"])):
            raise TraceContractError(f"normal action steps mismatch: {job['game_id']}")
        if len(state_records) != plan["episode_steps"]:
            raise TraceContractError(f"normal stored state count mismatch: {job['game_id']}")
        final_state = state_records[-1]
        final_statuses = [str(state.get("status")) for state in final_state["framework_state"]]
        final_step = final_state["framework_state"][0].get("observation", {}).get("step")
        if (
            not completion["normal"]
            or completion["statuses"] != ["DONE", "DONE"]
            or final_statuses != ["DONE", "DONE"]
            or final_state["state_index"] != semantics["terminal_state_index"]
            or final_step != semantics["terminal_state_index"]
            or final_state["runtime_observations"] is not None
        ):
            raise TraceContractError(f"normal completion mismatch: {job['game_id']}")
    elif not faults or completion["normal"]:
        raise TraceContractError(f"expected synthetic fault was not retained: {job['game_id']}")
    return {
        "records": len(records),
        "calls_by_seat": per_seat,
        "stored_states": len(state_records),
        "faults": len(faults),
    }


def validate_bundle(bundle: Path, write: bool = True) -> dict:
    plan = read_json(bundle / "plan.json")
    manifest = read_json(bundle / "manifest.json")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TraceContractError("unsupported plan or manifest schema version")
    schema_path = bundle / plan["trace_schema"]["path"]
    if (
        sha256_file(schema_path) != plan["trace_schema"]["sha256"]
        or plan["trace_schema"]["version"] != TRACE_SCHEMA_VERSION
        or manifest.get("trace_schema") != plan["trace_schema"]
    ):
        raise TraceContractError("trace schema identity mismatch")
    state_schema_path = bundle / plan["state_schema"]["path"]
    if (
        sha256_file(state_schema_path) != plan["state_schema"]["sha256"]
        or plan["state_schema"]["version"] != STATE_SCHEMA_VERSION
        or manifest.get("state_schema") != plan["state_schema"]
    ):
        raise TraceContractError("state schema identity mismatch")
    if _engine_identity() != plan["engine"] or manifest["engine"] != plan["engine"]:
        raise TraceContractError("engine identity mismatch")
    if sha256_file(bundle / "input-config.json") != plan["input_config"]["sha256"]:
        raise TraceContractError("input config identity mismatch")
    if sha256_file(bundle / "plan.json") != manifest["plan"]["sha256"]:
        raise TraceContractError("plan identity mismatch")
    _validate_plan_contract(bundle, plan)
    for agent in plan["agents"].values():
        _verify_package(bundle, agent)
    if len(plan["jobs"]) != len(manifest["episodes"]):
        raise TraceContractError("planned/captured episode count mismatch")
    results = [
        _validate_episode(bundle, plan, job, episode)
        for job, episode in zip(plan["jobs"], manifest["episodes"], strict=True)
    ]
    extraction = _write_semantic_extraction(bundle, manifest["episodes"])
    if extraction != manifest["extraction"]:
        raise TraceContractError("deterministic extraction identity mismatch")
    semantic_manifest = canonical_hash(_manifest_semantic_view(manifest))
    if semantic_manifest != manifest.get("semantic_sha256"):
        raise TraceContractError("semantic manifest hash mismatch")
    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "valid": True,
        "schema": plan["trace_schema"],
        "state_schema": plan["state_schema"],
        "semantics": plan["semantics"],
        "plan_sha256": sha256_file(bundle / "plan.json"),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "semantic_manifest_sha256": semantic_manifest,
        "extraction": extraction,
        "episodes": len(results),
        "normal_episodes": sum(job["expected"] == "normal" for job in plan["jobs"]),
        "fault_episodes": sum(job["expected"] == "fault" for job in plan["jobs"]),
        "records": sum(result["records"] for result in results),
        "stored_states": sum(result["stored_states"] for result in results),
        "faults": sum(result["faults"] for result in results),
        "checks": [
            "schema-hash",
            "engine-identity",
            "package-inventory",
            "game-identity",
            "trace-file-hash",
            "stored-state-file-hash",
            "seat-turn-completeness",
            "configured-state-call-semantics",
            "terminal-state-status",
            "pre-action-observation",
            "action-state-correspondence",
            "emitted-action-or-fault",
            "timing",
            "semantic-determinism",
            "extraction-hash",
        ],
    }
    if write:
        write_json(bundle / "validation-manifest.json", validation)
    return validation


def run_pipeline(config: Path, bundle: Path, workers: int) -> dict:
    plan = prepare_bundle(config, bundle)
    manifest = execute_plan(plan, bundle, workers)
    validation = validate_bundle(bundle)
    return {"plan": plan, "manifest": manifest, "validation": validation}


def reproduce_bundle(source: Path, destination: Path, workers: int) -> dict:
    validate_bundle(source, write=False)
    if destination.exists() and any(destination.iterdir()):
        raise TraceContractError(f"reproduction output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "input-config.json"):
        shutil.copyfile(source / name, destination / name)
    shutil.copytree(source / "schema", destination / "schema")
    shutil.copytree(source / "packages", destination / "packages")
    plan = read_json(destination / "plan.json")
    reproduced = execute_plan(plan, destination, workers)
    validation = validate_bundle(destination)
    original = read_json(source / "manifest.json")
    normal_indices = [
        index for index, job in enumerate(plan["jobs"]) if job["expected"] == "normal"
    ]
    fault_indices = [
        index for index, job in enumerate(plan["jobs"]) if job["expected"] == "fault"
    ]
    original_trace_hashes = [
        original["episodes"][index]["trace"]["semantic_sha256"] for index in normal_indices
    ]
    reproduced_trace_hashes = [
        reproduced["episodes"][index]["trace"]["semantic_sha256"] for index in normal_indices
    ]
    original_state_hashes = [
        original["episodes"][index]["states"]["semantic_sha256"] for index in normal_indices
    ]
    reproduced_state_hashes = [
        reproduced["episodes"][index]["states"]["semantic_sha256"] for index in normal_indices
    ]
    original_fault_fingerprints = [
        original["episodes"][index]["fault_fingerprint_sha256"] for index in fault_indices
    ]
    reproduced_fault_fingerprints = [
        reproduced["episodes"][index]["fault_fingerprint_sha256"] for index in fault_indices
    ]
    trace_match = original_trace_hashes == reproduced_trace_hashes
    state_match = original_state_hashes == reproduced_state_hashes
    fault_match = original_fault_fingerprints == reproduced_fault_fingerprints
    result = {
        "reproduced": trace_match and state_match and fault_match
        and original["semantic_sha256"] == reproduced["semantic_sha256"],
        "source_semantic_manifest_sha256": original["semantic_sha256"],
        "reproduced_semantic_manifest_sha256": reproduced["semantic_sha256"],
        "normal_trace_semantic_hashes_match": trace_match,
        "normal_state_semantic_hashes_match": state_match,
        "fault_fingerprints_match": fault_match,
        "normal_episodes": len(normal_indices),
        "fault_episodes": len(fault_indices),
        "episodes": len(reproduced["episodes"]),
        "validation_manifest_sha256": sha256_file(destination / "validation-manifest.json"),
    }
    write_json(destination / "reproduction.json", result)
    if not result["reproduced"]:
        raise TraceContractError("semantic reproduction mismatch")
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="capture, validate, and extract in one command")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    validate = sub.add_parser("validate", help="validate an existing bundle")
    validate.add_argument("--bundle", type=Path, required=True)
    extract = sub.add_parser("extract", help="rebuild and hash the deterministic state/action corpus")
    extract.add_argument("--bundle", type=Path, required=True)
    reproduce = sub.add_parser("reproduce", help="re-run exact retained packages and compare semantic hashes")
    reproduce.add_argument("--bundle", type=Path, required=True)
    reproduce.add_argument("--out", type=Path, required=True)
    reproduce.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_pipeline(args.config.resolve(), args.out.resolve(), args.workers)
            summary = result["validation"]
        elif args.command == "validate":
            summary = validate_bundle(args.bundle.resolve())
        elif args.command == "extract":
            bundle = args.bundle.resolve()
            manifest = read_json(bundle / "manifest.json")
            summary = _write_semantic_extraction(bundle, manifest["episodes"])
            if summary != manifest["extraction"]:
                raise TraceContractError("extracted corpus does not match the manifest")
        else:
            summary = reproduce_bundle(args.bundle.resolve(), args.out.resolve(), args.workers)
    except TraceContractError as exc:
        print(f"TRACE FAIL {exc}")
        return 1
    print("TRACE OK " + json.dumps(summary, sort_keys=True))
    return 0
