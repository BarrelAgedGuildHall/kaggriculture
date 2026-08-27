from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trace_retention import pipeline

SCRIPT = REPO / "scripts" / "trace_pipeline.py"
FAULT_AGENT = "trace_retention/fixtures/fault_agent.py"


def _config(tmp_path: Path, *, episode_steps: int = 5, include_fault: bool = True) -> Path:
    pairings = [{
        "agent_a": "pass-control",
        "agent_b": "starter-control",
        "seeds": [7],
        "swap_seats": True,
        "expected": "normal",
    }]
    agents = {
        "pass-control": {"spec": "pass"},
        "starter-control": {"spec": "starter"},
    }
    if include_fault:
        agents["synthetic-fault"] = {"spec": FAULT_AGENT}
        pairings.append({
            "agent_a": "synthetic-fault",
            "agent_b": "pass-control",
            "seeds": [8],
            "swap_seats": False,
            "expected": "fault",
        })
    value = {
        "schema_version": "kaggriculture.trace-config.v2",
        "episode_steps": episode_steps,
        "configuration": {},
        "agents": agents,
        "pairings": pairings,
    }
    path = tmp_path / "config.json"
    pipeline.write_json(path, value)
    return path


def _rewrite_episode(bundle: Path, index: int, records: list[dict], *, counts=None) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = pipeline.read_json(manifest_path)
    episode = manifest["episodes"][index]
    exact, semantic, size = pipeline._gzip_records(bundle / episode["trace"]["path"], records)
    episode["trace"].update(
        sha256=exact, semantic_sha256=semantic, bytes=size, records=len(records)
    )
    if counts is not None:
        episode["completion"]["action_calls_by_seat"] = counts
    manifest["extraction"] = pipeline._write_semantic_extraction(bundle, manifest["episodes"])
    manifest["semantic_sha256"] = pipeline.canonical_hash(
        pipeline._manifest_semantic_view(manifest)
    )
    pipeline.write_json(manifest_path, manifest)


def _rewrite_states(bundle: Path, index: int, records: list[dict]) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = pipeline.read_json(manifest_path)
    episode = manifest["episodes"][index]
    sha256, size = pipeline._gzip_state_records(
        bundle / episode["states"]["path"], records
    )
    episode["states"].update(
        sha256=sha256,
        semantic_sha256=sha256,
        bytes=size,
        records=len(records),
    )
    manifest["semantic_sha256"] = pipeline.canonical_hash(
        pipeline._manifest_semantic_view(manifest)
    )
    pipeline.write_json(manifest_path, manifest)


def test_one_command_captures_validates_extracts_and_retains_fault(tmp_path):
    config = _config(tmp_path)
    bundle = tmp_path / "bundle"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "run", "--config", str(config), "--out", str(bundle), "--workers", "1"],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRACE OK" in result.stdout
    validation = pipeline.read_json(bundle / "validation-manifest.json")
    assert validation["valid"] is True
    assert validation["episodes"] == 3
    assert validation["normal_episodes"] == 2
    assert validation["fault_episodes"] == 1
    assert validation["faults"] == 1
    manifest = pipeline.read_json(bundle / "manifest.json")
    assert all(
        ep["completion"]["episode_steps_configured"] == 5
        and ep["completion"]["stored_states"] == 5
        and ep["completion"]["action_calls_by_seat"] == [4, 4]
        and ep["completion"]["final_action_step_expected"] == 3
        and ep["completion"]["terminal_state_index_expected"] == 4
        for ep in manifest["episodes"][:2]
    )
    fault = manifest["episodes"][2]
    assert fault["faults"] == [{
        "message_sha256": fault["faults"][0]["message_sha256"],
        "seat": 0,
        "turn": 3,
        "type": "AgentProcessError",
    }]


def test_negative_trace_file_tampering_fails_hash_check(tmp_path):
    bundle = tmp_path / "bundle"
    pipeline.run_pipeline(_config(tmp_path, include_fault=False), bundle, workers=1)
    manifest = pipeline.read_json(bundle / "manifest.json")
    trace = bundle / manifest["episodes"][0]["trace"]["path"]
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(pipeline.TraceContractError, match="trace file hash/size mismatch"):
        pipeline.validate_bundle(bundle, write=False)


def test_truncated_but_rehashed_normal_trace_fails_completeness(tmp_path):
    bundle = tmp_path / "bundle"
    pipeline.run_pipeline(_config(tmp_path, include_fault=False), bundle, workers=1)
    manifest = pipeline.read_json(bundle / "manifest.json")
    episode = manifest["episodes"][0]
    records = list(pipeline._read_gzip_records(bundle / episode["trace"]["path"]))[:-2]
    _rewrite_episode(bundle, 0, records, counts=[3, 3])
    with pytest.raises(pipeline.TraceContractError, match="normal trace is not 4-call complete"):
        pipeline.validate_bundle(bundle, write=False)


def test_rehashed_seat_identity_mismatch_fails_closed(tmp_path):
    bundle = tmp_path / "bundle"
    pipeline.run_pipeline(_config(tmp_path, include_fault=False), bundle, workers=1)
    manifest = pipeline.read_json(bundle / "manifest.json")
    episode = manifest["episodes"][0]
    records = list(pipeline._read_gzip_records(bundle / episode["trace"]["path"]))
    records[0] = copy.deepcopy(records[0])
    records[0]["identity"]["agent_package_sha256"] = "0" * 64
    _rewrite_episode(bundle, 0, records)
    with pytest.raises(pipeline.TraceContractError, match="seat identity mismatch"):
        pipeline.validate_bundle(bundle, write=False)


def test_semantic_manifest_and_extraction_are_deterministic(tmp_path):
    config = _config(tmp_path, include_fault=False)
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = pipeline.run_pipeline(config, first, workers=1)["manifest"]
    two = pipeline.run_pipeline(config, second, workers=1)["manifest"]
    assert one["semantic_sha256"] == two["semantic_sha256"]
    assert one["extraction"]["sha256"] == two["extraction"]["sha256"]
    assert [ep["trace"]["semantic_sha256"] for ep in one["episodes"]] == [
        ep["trace"]["semantic_sha256"] for ep in two["episodes"]
    ]


def test_reproduction_uses_retained_packages_and_matches_semantics(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "reproduced"
    pipeline.run_pipeline(_config(tmp_path, include_fault=True), source, workers=1)
    result = pipeline.reproduce_bundle(source, destination, workers=1)
    assert result["reproduced"] is True
    assert result["normal_trace_semantic_hashes_match"] is True
    assert result["normal_state_semantic_hashes_match"] is True
    assert result["fault_fingerprints_match"] is True


def test_config_refuses_episode_steps_override(tmp_path):
    config = pipeline.read_json(_config(tmp_path, include_fault=False))
    config["configuration"] = {"episodeSteps": 720}
    pipeline.write_json(tmp_path / "config.json", config)
    with pytest.raises(pipeline.TraceContractError, match="must not override"):
        pipeline.prepare_bundle(tmp_path / "config.json", tmp_path / "bundle")


def test_exact_competition_plan_derives_719_calls_and_step_718(tmp_path):
    plan = pipeline.prepare_bundle(
        _config(tmp_path, episode_steps=720, include_fault=False), tmp_path / "bundle"
    )
    assert plan["configuration"]["episodeSteps"] == 720
    assert plan["episode_steps"] == 720
    assert plan["action_turns"] == 719
    assert plan["semantics"] == {
        "episode_steps": 720,
        "stored_states_expected": 720,
        "action_calls_per_seat_expected": 719,
        "initial_state_index": 0,
        "final_action_step": 718,
        "terminal_state_index": 719,
    }


def test_rehashed_extra_call_fails_exact_derived_count(tmp_path):
    bundle = tmp_path / "bundle"
    pipeline.run_pipeline(_config(tmp_path, include_fault=False), bundle, workers=1)
    manifest = pipeline.read_json(bundle / "manifest.json")
    episode = manifest["episodes"][0]
    records = list(pipeline._read_gzip_records(bundle / episode["trace"]["path"]))
    extra = []
    for seat in (0, 1):
        record = copy.deepcopy(records[-2 + seat])
        record["turn"] = 4
        record["state_index"] = 4
        record["pre_action_observation"]["step"] = 4
        extra.append(record)
    _rewrite_episode(bundle, 0, records + extra, counts=[5, 5])
    with pytest.raises(pipeline.TraceContractError, match="normal trace is not 4-call complete"):
        pipeline.validate_bundle(bundle, write=False)


def test_rehashed_missing_paired_seat_fails_closed(tmp_path):
    bundle = tmp_path / "bundle"
    pipeline.run_pipeline(_config(tmp_path, include_fault=False), bundle, workers=1)
    manifest = pipeline.read_json(bundle / "manifest.json")
    episode = manifest["episodes"][0]
    records = list(pipeline._read_gzip_records(bundle / episode["trace"]["path"]))
    _rewrite_episode(bundle, 0, records[:-1], counts=[4, 3])
    with pytest.raises(pipeline.TraceContractError, match="truncated or unpaired turns"):
        pipeline.validate_bundle(bundle, write=False)


def test_rehashed_action_state_mismatch_fails_closed(tmp_path):
    bundle = tmp_path / "bundle"
    pipeline.run_pipeline(_config(tmp_path, include_fault=False), bundle, workers=1)
    manifest = pipeline.read_json(bundle / "manifest.json")
    episode = manifest["episodes"][0]
    records = list(pipeline._read_gzip_records(bundle / episode["trace"]["path"]))
    records[0] = copy.deepcopy(records[0])
    records[0]["emitted_action"] = {"farmer": ["PASS", "PASS"], "hands": [], "market": []}
    _rewrite_episode(bundle, 0, records)
    with pytest.raises(pipeline.TraceContractError, match="emitted action/stored state mismatch"):
        pipeline.validate_bundle(bundle, write=False)


def test_rehashed_unexpected_terminal_state_fails_closed(tmp_path):
    bundle = tmp_path / "bundle"
    pipeline.run_pipeline(_config(tmp_path, include_fault=False), bundle, workers=1)
    manifest = pipeline.read_json(bundle / "manifest.json")
    episode = manifest["episodes"][0]
    states = list(pipeline._read_gzip_records(bundle / episode["states"]["path"]))
    states[-1] = copy.deepcopy(states[-1])
    states[-1]["framework_state"][0]["status"] = "ACTIVE"
    _rewrite_states(bundle, 0, states)
    with pytest.raises(pipeline.TraceContractError, match="normal completion mismatch"):
        pipeline.validate_bundle(bundle, write=False)
