"""Move ONE agent snapshot from a clone to a hub, content-verified, merged single-entry.

Why this exists
---------------
There was no supported way to get an agent onto the Pit host at all. The two that looked like
one are both wrong:

* `arena/remote.py`'s bundle tars the SENDER's `data/arena/registry.json` and the node extracts
  it over the hub's -- so a ship replaces the hub's champion pointers, roster and
  active/benched/retired flags with whatever the shipping clone happens to hold (issue #15;
  it cost Team A's champion pointer on 2026-08-20, and on 2026-08-21 a clone holding 14 agents
  would have dropped 19 from a hub holding 32).
* `POST /api/control/register {git_ref}` needs a git clone with access to origin, and
  `~/kaggri-arena/repo` on the Pit host is an untarred working tree, not a repo. The `{path}`
  variant is gated behind `allow_unsafe_register`, which the Auditor deliberately left off.

So this is the narrow thing: **one agent, one registry key**. Everything else on the hub --
every other agent entry, every champion pointer, every flag -- is not read, not written and
not diffable by this path, and the response says so in numbers (agents before/after,
lost/gained) so the caller can prove it rather than trust it.

Guarantees
----------
1. **Content-hash verified end to end.** The sender hashes (relative path + bytes) of every
   file; the hub recomputes that over the bytes it received, and again over the bytes it wrote
   to disk, and returns the third hash. The CLI compares it with the first. A silent truncation
   or a transfer that dropped a weights file cannot be mistaken for a success.
2. **Single-entry merge.** The registry is read, ONE key is set, and it is written back with
   the same atomic replace `registry.save` uses. Champions are never touched.
3. **Id collision is refused.** Ids are content-addressed, so the same id is meant to be the
   same bytes; if it is not, something is wrong and `--force` is the deliberate override. A
   forced push backs the old snapshot up, keeps the old entry, and appends a supersession
   record to `push-log.jsonl`.
4. **Benched on arrival.** A push is scheduling code for execution on the Pit host exactly as
   registration is, so a pushed agent starts `active: false` -- somebody activates it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

from . import registry
from .paths import AGENTS_DIR, ARENA_DIR, ensure_dirs

MAX_FILES = 400
MAX_TOTAL_BYTES = 32 * 1024 * 1024        # a snapshot is code + weights, not a dataset
SKIP_PARTS = ("__pycache__", ".git")
PUSH_LOG = "push-log.jsonl"

# The registry is a read-modify-write file and the hub serves requests on threads. This does not
# make the whole registry thread-safe (nothing here does), it keeps two pushes from interleaving.
_LOCK = threading.Lock()

# Copied onto the hub's entry verbatim; anything else the sender sends is dropped. Keeping this
# explicit is what stops a push from carrying a champion pointer or a flag across.
ENTRY_KEYS = ("id", "name", "sha", "team", "kind", "entry", "source", "engine", "engine_sha",
              "registered_at", "notes", "git_ref", "nav_schema_version", "nav_source_hash")


class PushConflict(ValueError):
    """The hub already has this id and `force` was not set (HTTP 409)."""

    def __init__(self, agent_id: str, existing: dict):
        self.agent_id = agent_id
        self.existing = existing
        super().__init__(
            f"{agent_id} is already registered on this hub (registered {existing.get('registered_at')}, "
            f"source {existing.get('source')!r}). Ids are content-addressed, so the same id should be "
            f"the same bytes -- if the bytes really did change, push again with --force, which backs "
            f"the old snapshot up and logs the supersession.")


# --------------------------------------------------------------------------- snapshot files
def snapshot_files(root: Path) -> list[tuple[str, bytes]]:
    """Every file of a snapshot as `(relative posix path, bytes)`, sorted by path.

    `root` is the snapshot root under `data/arena/agents/`: `<id>` for a directory agent,
    `<id>.py` for a single-file one.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"snapshot not found: {root}")
    if root.is_file():
        return [(root.name, root.read_bytes())]
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in SKIP_PARTS for part in p.relative_to(root).parts):
            continue
        rel = PurePosixPath(root.name) / PurePosixPath(*p.relative_to(root).parts)
        out.append((str(rel), p.read_bytes()))
    if not out:
        raise FileNotFoundError(f"snapshot {root} holds no files")
    return sorted(out)


def manifest_hash(files) -> str:
    """sha256 over `(path, bytes)` in path order. Byte-exact on purpose: a .py that changed
    line endings IS a different agent to the arena (the id hashes the same bytes)."""
    h = hashlib.sha256()
    for rel, data in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def _check_size(files) -> int:
    if len(files) > MAX_FILES:
        raise ValueError(f"snapshot holds {len(files)} files; the limit is {MAX_FILES}")
    total = sum(len(d) for _, d in files)
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"snapshot is {total / 1e6:.1f} MB; the limit is {MAX_TOTAL_BYTES / 1e6:.0f} MB "
                         "(model weights belong in the agent, datasets do not)")
    return total


def default_sender() -> str:
    """Who is pushing, for the entry and the log: `user@host`, best effort."""
    import getpass
    import socket
    try:
        return f"{getpass.getuser()}@{socket.gethostname()}"
    except Exception:
        return "unknown"


def build_payload(agent: dict, files, force: bool = False, sender: str = "") -> dict:
    entry = {k: agent[k] for k in ENTRY_KEYS if k in agent}
    return {
        "agent": entry,
        "files": {rel: base64.b64encode(data).decode("ascii") for rel, data in files},
        "manifest_sha256": manifest_hash(files),
        "force": bool(force),
        "sender": sender,
    }


# --------------------------------------------------------------------------- hub side
def _validate_entry(entry: dict) -> dict:
    if not isinstance(entry, dict):
        raise ValueError("push: `agent` must be the sender's registry entry (a JSON object)")
    for key in ("id", "name", "sha", "entry"):
        if not entry.get(key):
            raise ValueError(f"push: the agent entry is missing {key!r}")
    if "@" not in entry["id"]:
        raise ValueError(f"push: agent id must look like name@sha, got {entry['id']!r}")
    if entry.get("kind", "file") != "file":
        raise ValueError(f"push: only file agents can be pushed, not {entry.get('kind')!r} "
                         "(built-ins are residents everywhere; a ghost needs its replay)")
    registry._validate_team(entry.get("team", "resident"))
    return entry


def _validate_paths(names, agent_id: str) -> str:
    """Every file must live inside this agent's own snapshot root, and nowhere else.

    A push writes attacker-chosen paths under `data/arena/`, so this is the whole security
    boundary: no absolute paths, no drive letters, no `..`, no symlink names, and the root must
    be exactly what `registry.register_file` would have created for this id.
    """
    roots = set()
    for name in names:
        if not name or name != name.strip():
            raise ValueError(f"push: bad file name {name!r}")
        norm = name.replace("\\", "/")
        p = PurePosixPath(norm)
        if p.is_absolute() or ":" in norm or ".." in p.parts or any(part in ("", ".") for part in p.parts):
            raise ValueError(f"push: file path must be relative and inside the snapshot, got {name!r}")
        roots.add(p.parts[0])
    if len(roots) != 1:
        raise ValueError(f"push: every file must be under one snapshot root, got {sorted(roots)}")
    root = roots.pop()
    if root not in (agent_id, f"{agent_id}.py"):
        raise ValueError(f"push: snapshot root {root!r} does not belong to {agent_id!r} "
                         f"(expected {agent_id!r} or {agent_id + '.py'!r})")
    return root


def _log(event: str, data: dict) -> None:
    try:
        ensure_dirs()
        with open(ARENA_DIR / PUSH_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **data}) + "\n")
    except OSError:
        pass          # a push must not fail because the log could not be written


def _write_snapshot(root: str, files) -> Path:
    """Materialise the files in a staging directory beside the agents store and return it."""
    ensure_dirs()
    staging = AGENTS_DIR / f".incoming-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        for rel, data in files:
            dest = staging / PurePosixPath(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def receive(payload: dict) -> dict:
    """Accept one pushed snapshot. Returns a report the caller can verify against its own hash."""
    if not isinstance(payload, dict):
        raise ValueError("push: body must be a JSON object")
    entry = _validate_entry(payload.get("agent") or {})
    raw = payload.get("files")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("push: `files` must be a non-empty {relative path: base64} object")
    agent_id = entry["id"]
    root = _validate_paths(raw.keys(), agent_id)
    try:
        files = sorted((str(k).replace("\\", "/"), base64.b64decode(v, validate=True))
                       for k, v in raw.items())
    except Exception as exc:
        raise ValueError(f"push: a file is not valid base64 ({exc})") from None
    total = _check_size(files)

    got = manifest_hash(files)
    want = str(payload.get("manifest_sha256") or "")
    if not want:
        raise ValueError("push: manifest_sha256 is required (it is the transfer's only integrity check)")
    if got != want:
        raise ValueError(f"push: content hash mismatch -- the sender computed {want[:16]}…, "
                         f"the bytes that arrived hash to {got[:16]}…; nothing was written")

    expected_entry = entry["entry"].replace("\\", "/")
    if expected_entry not in dict(files):
        raise ValueError(f"push: the entry point {expected_entry!r} is not among the pushed files")

    with _LOCK:
        reg = registry.load()
        existing = reg["agents"].get(agent_id)
        if existing and not payload.get("force"):
            raise PushConflict(agent_id, existing)

        staging = _write_snapshot(root, files)
        try:
            on_disk = manifest_hash(snapshot_files(staging / root))
            if on_disk != got:
                raise ValueError(f"push: the files did not survive the write ({on_disk[:16]}… != {got[:16]}…)")
            final = AGENTS_DIR / root
            superseded = None
            if final.exists():
                backup = AGENTS_DIR / f"{root}.superseded-{time.strftime('%Y%m%d-%H%M%S')}"
                final.rename(backup)
                superseded = {"agent": agent_id, "backup": str(backup),
                              "previous_entry": existing, "sender": payload.get("sender", "")}
                _log("superseded", superseded)
            try:
                (staging / root).rename(final)
            except Exception:
                if superseded:
                    Path(superseded["backup"]).rename(final)      # put the old snapshot back
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        before = sorted(reg["agents"])
        champions_before = dict(reg.get("champions", {}))
        merged = dict(entry)
        merged.update(kind="file", active=False, pushed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      pushed_from=payload.get("sender", ""), push_sha256=got)
        if existing and existing.get("retired"):
            merged["retired"] = True          # a forced re-push does not un-retire an agent
        reg["agents"][agent_id] = merged
        registry.save(reg)

    after_reg = registry.load()
    after = sorted(after_reg["agents"])
    report = {
        "agent": merged, "id": agent_id, "root": root,
        "files": len(files), "bytes": total,
        "manifest_sha256": manifest_hash(snapshot_files(AGENTS_DIR / root)),
        "agents_before": len(before), "agents_after": len(after),
        "lost": [a for a in before if a not in after],
        "gained": [a for a in after if a not in before],
        "champions": after_reg.get("champions", {}),
        "champions_unchanged": after_reg.get("champions", {}) == champions_before,
        "superseded": superseded, "arena_dir": str(ARENA_DIR),
    }
    _log("push", {k: report[k] for k in ("id", "files", "bytes", "manifest_sha256",
                                         "agents_before", "agents_after", "lost")})
    return report


# --------------------------------------------------------------------------- client side
def collect(agent_spec: str) -> tuple[dict, list[tuple[str, bytes]]]:
    """Resolve a local agent and read its snapshot off disk."""
    agent = registry.resolve(agent_spec)
    if agent.get("kind") != "file":
        raise ValueError(f"{agent['id']} is a {agent.get('kind')} agent; only file agents can be pushed "
                         "(built-ins are residents on every host, and a ghost needs its replay)")
    root = registry._snapshot_root(agent["entry"])
    files = snapshot_files(AGENTS_DIR / root)
    _check_size(files)
    return agent, files
