"""`arena push`: one snapshot to a hub, content-verified, ONE registry key merged.

The arena state is redirected to a temp dir through KAGGRI_ARENA_DIR (read at import time by
arena.paths), so these tests never touch data/arena/. The "hub" here is the same arena dir with
the agent's entry and snapshot removed first -- which is exactly the state a hub is in before a
push, and it lets the test assert the thing that actually matters: what the merge does to the
OTHER entries and to the champion pointers.

Run: uv run pytest -q tests/test_arena_push.py
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# `arena.paths` reads KAGGRI_ARENA_DIR once, at import, and the other modules bind ARENA_DIR /
# AGENTS_DIR from it at THEIR import -- so a second test module that redirects the arena dir and
# reloads `paths` gets a `paths.AGENTS_DIR` that no longer agrees with `registry.AGENTS_DIR`, and
# every push lands in one directory while the registry is read from another. Whoever redirects
# first therefore owns the scratch dir for the session; a later module adopts it.
if "arena.paths" in sys.modules:
    import arena.paths as paths                                        # noqa: E402

    _TMP = paths.ARENA_DIR
    assert _TMP != paths.DATA_DIR / "arena", "the arena dir must be redirected before arena is imported"
else:
    _TMP = Path(tempfile.mkdtemp(prefix="arena-push-test-"))
    os.environ["KAGGRI_ARENA_DIR"] = str(_TMP)
    import arena.paths as paths                                        # noqa: E402

    importlib.reload(paths)
    assert paths.ARENA_DIR == _TMP

from arena import push, registry, serve  # noqa: E402
from arena.paths import AGENTS_DIR, ARENA_DIR  # noqa: E402

AGENT_SRC = "def agent(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"


def _dir_agent(tmp_path: Path, name: str) -> dict:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("import policy\n\n\ndef agent(obs):\n    return policy.act(obs)\n",
                                  encoding="utf-8")
    (root / "policy.py").write_text("def act(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
                                    encoding="utf-8")
    (root / "weights").mkdir(exist_ok=True)
    (root / "weights" / "w.npz").write_bytes(bytes(range(256)) * 4)      # a binary, byte-exactness
    (root / "__pycache__").mkdir(exist_ok=True)
    (root / "__pycache__" / "junk.pyc").write_bytes(b"never travels")
    return registry.register_file(root, name, "A", notes="test dir agent")


def _forget(agent_id: str) -> None:
    """Put the arena dir into the state a hub is in before the push: no entry, no snapshot."""
    reg = registry.load()
    root = registry._snapshot_root(reg["agents"][agent_id]["entry"])
    del reg["agents"][agent_id]
    registry.save(reg)
    target = AGENTS_DIR / root
    shutil.rmtree(target) if target.is_dir() else target.unlink()


def test_dir_agent_round_trip_is_byte_exact_and_merges_one_key(tmp_path):
    (tmp_path / "solo.py").write_text(AGENT_SRC, encoding="utf-8")
    other = registry.register_file(tmp_path / "solo.py", "bystander", "B")
    registry.set_champion("B", other["id"])
    agent = _dir_agent(tmp_path, "pushme")

    a, files = push.collect(agent["id"])
    names = [rel for rel, _ in files]
    assert names == sorted(names)
    assert f"{agent['id']}/main.py" in names and f"{agent['id']}/weights/w.npz" in names
    assert not any("__pycache__" in n for n in names), "__pycache__ never travels"
    local_hash = push.manifest_hash(files)

    _forget(agent["id"])
    before = registry.load()
    report = push.receive(push.build_payload(a, files, sender="tester"))

    assert report["manifest_sha256"] == local_hash, "hashed again from the hub's own disk"
    assert report["agents_before"] == len(before["agents"])
    assert report["agents_after"] == report["agents_before"] + 1
    assert report["gained"] == [agent["id"]] and report["lost"] == []
    assert report["champions_unchanged"] and report["champions"]["B"] == other["id"]

    reg = registry.load()
    assert reg["agents"][agent["id"]]["active"] is False, "a pushed agent arrives benched"
    assert reg["agents"][agent["id"]]["pushed_from"] == "tester"
    assert reg["agents"][other["id"]] == before["agents"][other["id"]], "no other entry is touched"
    for rel, data in files:
        assert (AGENTS_DIR / rel).read_bytes() == data


def test_single_file_agent_round_trip(tmp_path):
    src = tmp_path / "one.py"
    src.write_text(AGENT_SRC, encoding="utf-8")
    agent = registry.register_file(src, "onefile", "A")
    a, files = push.collect("onefile")
    assert [rel for rel, _ in files] == [f"{agent['id']}.py"]
    _forget(agent["id"])
    report = push.receive(push.build_payload(a, files))
    assert report["root"] == f"{agent['id']}.py"
    assert (AGENTS_DIR / report["root"]).read_text(encoding="utf-8") == AGENT_SRC


def test_id_collision_is_refused_and_force_supersedes(tmp_path):
    agent = _dir_agent(tmp_path, "collide")
    a, files = push.collect(agent["id"])
    _forget(agent["id"])
    push.receive(push.build_payload(a, files))

    with pytest.raises(push.PushConflict) as exc:
        push.receive(push.build_payload(a, files))
    assert agent["id"] in str(exc.value) and "--force" in str(exc.value)

    changed = [(rel, (b"# different\n" + data) if rel.endswith("main.py") else data)
               for rel, data in files]
    report = push.receive(push.build_payload(a, changed, force=True))
    assert report["superseded"] and Path(report["superseded"]["backup"]).exists()
    assert (AGENTS_DIR / f"{agent['id']}/main.py").read_bytes().startswith(b"# different")
    assert report["manifest_sha256"] == push.manifest_hash(changed)
    log = [json.loads(l) for l in (ARENA_DIR / push.PUSH_LOG).read_text(encoding="utf-8").splitlines()]
    assert any(e["event"] == "superseded" and e["agent"] == agent["id"] for e in log)


def test_a_corrupt_transfer_writes_nothing(tmp_path):
    agent = _dir_agent(tmp_path, "corrupt")
    a, files = push.collect(agent["id"])
    _forget(agent["id"])
    payload = push.build_payload(a, files)
    payload["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        push.receive(payload)
    assert agent["id"] not in registry.load()["agents"]
    assert not (AGENTS_DIR / agent["id"]).exists()
    assert not list(AGENTS_DIR.glob(".incoming-*")), "the staging directory is always cleaned up"


@pytest.mark.parametrize("bad", ["../evil.py", "/etc/passwd", "C:/evil.py", "other@sha/main.py"])
def test_paths_outside_the_snapshot_are_refused(tmp_path, bad):
    agent = _dir_agent(tmp_path, "traversal")
    a, files = push.collect(agent["id"])
    _forget(agent["id"])
    payload = push.build_payload(a, [(bad, b"boom")] + list(files))
    with pytest.raises(ValueError):
        push.receive(payload)
    assert not (Path(ARENA_DIR).parent / "evil.py").exists()


def test_entry_point_must_be_among_the_files(tmp_path):
    agent = _dir_agent(tmp_path, "noentry")
    a, files = push.collect(agent["id"])
    _forget(agent["id"])
    without_main = [(rel, d) for rel, d in files if not rel.endswith("main.py")]
    with pytest.raises(ValueError, match="entry point"):
        push.receive(push.build_payload(a, without_main))


def test_builtins_and_ghosts_cannot_be_pushed():
    registry.ensure_residents()
    with pytest.raises(ValueError, match="only file agents|built-ins are residents"):
        push.collect("random")


def test_control_route_checks_keys_and_maps_the_conflict(tmp_path):
    agent = _dir_agent(tmp_path, "viaapi")
    a, files = push.collect(agent["id"])
    _forget(agent["id"])
    body = push.build_payload(a, files)
    out = serve.control("push", body)
    assert out["id"] == agent["id"]
    with pytest.raises(push.PushConflict):
        serve.control("push", body)
    with pytest.raises(ValueError, match="unknown request key"):
        serve.control("push", {**body, "champions": {"A": "hijack"}})
