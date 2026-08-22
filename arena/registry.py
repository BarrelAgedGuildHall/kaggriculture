"""Agent registry: immutable `name@sha` entries + champion pointers, in data/arena/registry.json.

Registering a file agent copies it into data/arena/agents/<name>@<sha>[.py] so the arena
always plays the exact bytes that were registered, whatever happens to the source file later.
`sha` is the git short sha of the file when it is committed, clean and contains nothing git
does not track (gitignored weights force it); else a content hash over every file, weights
included, so re-registering changed bytes always yields a new id.
Built-ins (`random`, `starter`, `pass`) are residents and are auto-registered on first use.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path  # noqa: F401  (type annotations + relabel's db_path)

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
            "registered_at": _now(), "notes": "engine built-in resident", "active": True,
        }
        save(reg)
    return reg["agents"][agent_id]


def ensure_residents(reg: dict | None = None) -> dict:
    reg = reg if reg is not None else load()
    for b in BUILTIN_AGENTS:
        register_builtin(b, reg)
    return reg


def register_file(path: str | Path, name: str, team: str, notes: str = "", sha: str | None = None,
                  git_ref: str | None = None) -> dict:
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
        "registered_at": _now(), "notes": notes, "active": True, "git_ref": git_ref,
    }
    save(reg)
    return reg["agents"][agent_id]


def register_git_ref(ref: str, path: str, name: str, team: str, notes: str = "", repo: Path | None = None) -> dict:
    """Register `path` as it is at git `ref` (branch, tag or sha): fetches the ref from origin,
    materialises the file(s) into a temp dir and registers them with sha = the commit's short sha.
    Needs a clone with access to origin (the desktop has it; workflows-box needs a deploy key)."""
    import subprocess
    import tempfile
    repo = repo or REPO_ROOT

    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, text=True, capture_output=True, timeout=120)

    fetched = git("fetch", "--quiet", "origin", ref)
    if fetched.returncode != 0:
        git("fetch", "--quiet", "origin")   # a bare sha cannot be fetched by name; refresh everything
    # Prefer what origin just gave us (FETCH_HEAD / origin/<ref>) over a possibly stale local branch.
    r = git("rev-parse", "--short", "FETCH_HEAD^{commit}") if fetched.returncode == 0 else \
        git("rev-parse", "--short", "origin/" + ref + "^{commit}")
    if r.returncode != 0:
        r = git("rev-parse", "--short", "origin/" + ref + "^{commit}")
    if r.returncode != 0:
        r = git("rev-parse", "--short", ref + "^{commit}")
    if r.returncode != 0:
        raise ValueError(f"git ref {ref!r} not found in {repo} (fetch failed or no access): {r.stderr.strip()}")
    sha = r.stdout.strip()
    full = git("rev-parse", sha + "^{commit}").stdout.strip()
    rel = path.replace("\\", "/").strip("/")
    with tempfile.TemporaryDirectory() as td:
        ls = git("ls-tree", "-r", "--name-only", full, "--", rel)
        files = [ln for ln in ls.stdout.splitlines() if ln.strip()]
        if not files:
            raise FileNotFoundError(f"{rel} not found at {ref} ({sha})")
        is_dir = not (len(files) == 1 and files[0] == rel)
        for f in files:
            blob = subprocess.run(["git", "show", f"{full}:{f}"], cwd=repo, capture_output=True, timeout=120)
            if blob.returncode != 0:
                raise FileNotFoundError(f"cannot read {f} at {sha}")
            out = Path(td) / (f[len(rel):].lstrip("/") if is_dir else Path(rel).name)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(blob.stdout)
        src = Path(td) if is_dir else Path(td) / Path(rel).name
        entry = register_file(src, name, team, notes=notes or f"{rel}@{ref}", sha=sha, git_ref=f"{ref}:{rel}@{sha}")
    reg = load()
    reg["agents"][entry["id"]]["source"] = f"git {ref}:{rel} ({sha})"
    save(reg)
    return reg["agents"][entry["id"]]


def set_flag(spec: str, flag: str, value: bool) -> dict:
    """active: takes part in the hub ladder; retired: excluded from the rating fit (rows kept)."""
    if flag not in ("active", "retired"):
        raise ValueError("flag must be active or retired")
    reg = load()
    agent = resolve(spec, reg)
    reg["agents"][agent["id"]][flag] = bool(value)
    if flag == "retired" and value:
        reg["agents"][agent["id"]]["active"] = False
    save(reg)
    return reg["agents"][agent["id"]]


def is_active(agent: dict) -> bool:
    return bool(agent.get("active", True)) and not agent.get("retired") and agent.get("kind") != "ghost"


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


def relabel(spec: str, new_id: str, db_path: Path | None = None, dry_run: bool = False) -> dict:
    """Rename a registered agent id everywhere at once: registry entry, champion pointers,
    the snapshot on disk, and every games/runs row that names it.

    Ids are immutable by design (same bytes -> same id), and this does not change that: it
    exists for the case where the *label* was wrong -- the same agent registered under a git
    sha that has since been rebased, say -- and re-registering would orphan the games already
    played. Orphaning is the failure this prevents: the rows would keep pointing at an id no
    longer in the registry, and the leaderboard would quietly drop a thousand measurements.

    The sqlite rewrite is one transaction and the registry is written with an atomic replace
    AFTER it commits, so a crash leaves either the old state or the new one -- never a
    registry that has moved on from its own rows. The snapshot rename is undone if the
    transaction fails.
    """
    if "@" not in new_id:
        raise ValueError(f"new id must look like name@sha, got {new_id!r}")
    reg = load()
    agent = dict(resolve(spec, reg))
    old_id = agent["id"]
    if old_id == new_id:
        raise ValueError(f"{old_id} is already the id")
    if new_id in reg["agents"]:
        raise ValueError(f"{new_id} is already registered -- ids are immutable, pick another")

    from . import store
    con = store.connect(db_path)
    try:
        n_games = store.count_games_for(con, old_id)
        if dry_run:
            return {"old": old_id, "new": new_id, "games": n_games, "dry_run": True,
                    "champions": [t for t, c in reg["champions"].items() if c == old_id]}

        name, sha = new_id.split("@", 1)
        old_path = AGENTS_DIR / agent["entry"] if agent.get("entry") else None
        new_entry = agent.get("entry")
        renamed = None
        if old_path is not None and old_path.exists() and old_id in str(agent["entry"]):
            new_entry = agent["entry"].replace(old_id, new_id)
            new_path = AGENTS_DIR / new_entry
            if new_path.exists():
                raise ValueError(f"{new_path} already exists -- refusing to overwrite a snapshot")
            old_path.rename(new_path)
            renamed = (new_path, old_path)

        try:
            moved = store.relabel_agent(con, old_id, new_id)
        except Exception:
            if renamed:
                renamed[0].rename(renamed[1])          # put the snapshot back
            raise

        agent.update(id=new_id, name=name, sha=sha, entry=new_entry, relabelled_from=old_id)
        del reg["agents"][old_id]
        reg["agents"][new_id] = agent
        champions = [t for t, c in reg["champions"].items() if c == old_id]
        for team in champions:
            reg["champions"][team] = new_id
        save(reg)                                       # atomic replace, after the commit
        return {"old": old_id, "new": new_id, "champions": champions, "dry_run": False, **moved}
    finally:
        con.close()


def set_retired(spec: str, retired: bool = True) -> dict:
    """Retired agents keep their rows but are excluded from the leaderboard fit and the ladder
    (the PTCG arena's 'benched' idea): old generations stop distorting the current field."""
    return set_flag(spec, "retired", retired)
