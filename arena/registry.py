"""Agent registry: immutable `name@sha` entries + champion pointers, in data/arena/registry.json.

Registering a file agent copies it into data/arena/agents/<name>@<sha>[.py] so the arena
always plays the exact bytes that were registered, whatever happens to the source file later.
`sha` is the git short sha of the file when it is committed and clean, else a content hash.
Built-ins (`random`, `starter`, `pass`) are residents and are auto-registered on first use.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import engine as eng
from .paths import AGENTS_DIR, BUILTIN_AGENTS, REGISTRY_PATH, REPO_ROOT, ensure_dirs

TEAMS = ("A", "B", "resident")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def load() -> dict:
    if REGISTRY_PATH.exists():
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        reg = {}
    reg.setdefault("agents", {})
    reg.setdefault("champions", {})
    return reg


def save(reg: dict) -> None:
    ensure_dirs()
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


def _validate_team(team: str) -> str:
    if team not in TEAMS:
        raise ValueError(f"team must be one of {TEAMS}, got {team!r}")
    return team


def register_builtin(name: str, reg: dict | None = None) -> dict:
    if name not in BUILTIN_AGENTS:
        raise ValueError(f"not an engine built-in: {name}")
    reg = reg if reg is not None else load()
    sha = "engine" + eng.engine_version().replace(".", "")
    agent_id = f"{name}@{sha}"
    if agent_id not in reg["agents"]:
        reg["agents"][agent_id] = {
            "id": agent_id, "name": name, "sha": sha, "team": "resident", "kind": "builtin",
            "entry": name, "source": f"kaggle_environments.envs.kaggriculture:{name}_agent",
            "engine": eng.engine_version(), "engine_sha": eng.engine_sha(),
            "registered_at": _now(), "notes": "engine built-in resident",
        }
        save(reg)
    return reg["agents"][agent_id]


def ensure_residents(reg: dict | None = None) -> dict:
    reg = reg if reg is not None else load()
    for b in BUILTIN_AGENTS:
        register_builtin(b, reg)
    return reg


def register_file(path: str | Path, name: str, team: str, notes: str = "", sha: str | None = None) -> dict:
    """Snapshot a .py file (or a directory with main.py) into the agents store."""
    _validate_team(team)
    src = Path(path)
    if not src.is_absolute():
        src = (Path.cwd() / src) if (Path.cwd() / src).exists() else (REPO_ROOT / src)
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(f"agent not found: {path}")
    if src.is_dir() and not (src / "main.py").exists():
        raise FileNotFoundError(f"agent directory has no main.py: {src}")
    sha = sha or eng.file_git_sha(src) or ("h" + eng.content_hash(src))
    agent_id = f"{name}@{sha}"
    reg = load()
    if agent_id in reg["agents"]:
        return reg["agents"][agent_id]  # immutable; same bytes -> same id, nothing to do
    ensure_dirs()
    if src.is_dir():
        dst = AGENTS_DIR / agent_id
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"), dirs_exist_ok=True)
        entry = f"{agent_id}/main.py"
    else:
        dst = AGENTS_DIR / f"{agent_id}.py"
        shutil.copy2(src, dst)
        entry = f"{agent_id}.py"
    try:
        source = str(src.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        source = str(src)
    reg["agents"][agent_id] = {
        "id": agent_id, "name": name, "sha": sha, "team": team, "kind": "file",
        "entry": entry, "source": source,
        "engine": eng.engine_version(), "engine_sha": eng.engine_sha(),
        "registered_at": _now(), "notes": notes,
    }
    save(reg)
    return reg["agents"][agent_id]


def register_entry(entry: dict) -> dict:
    """Insert a fully-formed entry (used by ghosts)."""
    reg = load()
    reg["agents"].setdefault(entry["id"], entry)
    save(reg)
    return reg["agents"][entry["id"]]


def resolve(spec: str, reg: dict | None = None) -> dict:
    """`name@sha` exact, or `name` -> the most recently registered agent of that name.
    Built-in names are auto-registered as residents."""
    reg = reg if reg is not None else load()
    if spec in reg["agents"]:
        return reg["agents"][spec]
    if spec in BUILTIN_AGENTS:
        return register_builtin(spec, reg)
    cands = [a for a in reg["agents"].values() if a["name"] == spec]
    if not cands:
        raise KeyError(f"unknown agent {spec!r}; register it first (`arena register ...`) or use name@sha. "
                       f"Known: {', '.join(sorted(reg['agents'])) or '-'}")
    cands.sort(key=lambda a: a["registered_at"])
    return cands[-1]


def entry_path(agent: dict) -> Path:
    return AGENTS_DIR / agent["entry"]


def list_agents(reg: dict | None = None) -> list[dict]:
    reg = reg if reg is not None else load()
    return sorted(reg["agents"].values(), key=lambda a: (a["team"], a["registered_at"], a["id"]))


def set_champion(team: str, spec: str) -> dict:
    if team not in ("A", "B", "org"):
        raise ValueError("champion team must be A, B or org")
    reg = load()
    agent = resolve(spec, reg)
    reg["champions"][team] = agent["id"]
    save(reg)
    return agent


def get_champion(team: str, reg: dict | None = None) -> dict | None:
    reg = reg if reg is not None else load()
    cid = reg["champions"].get(team)
    return reg["agents"].get(cid) if cid else None


def set_retired(spec: str, retired: bool = True) -> dict:
    """Retired agents keep their rows but are excluded from the leaderboard fit by default
    (the PTCG arena's 'benched' idea): old generations stop distorting the current field."""
    reg = load()
    agent = resolve(spec, reg)
    reg["agents"][agent["id"]]["retired"] = bool(retired)
    save(reg)
    return reg["agents"][agent["id"]]
