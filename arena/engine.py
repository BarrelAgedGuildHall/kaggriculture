"""Quiet import of kaggle_environments + version/sha helpers (mirrors scripts/_engine.py).

`import kaggle_environments` registers every bundled environment and the OpenSpiel bridge
prints ~300 lines of noise to stderr at the file-descriptor level; mute that one import.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .paths import ENV_NAME, REPO_ROOT

_KE = None


def import_engine():
    global _KE
    if _KE is not None:
        return _KE
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        stderr_fd = None
    if stderr_fd is None:
        import kaggle_environments  # noqa: F401
        _KE = kaggle_environments
        return _KE
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull, stderr_fd)
        import kaggle_environments  # noqa: F401
    finally:
        sys.stderr.flush()
        os.dup2(saved, stderr_fd)
        os.close(devnull)
        os.close(saved)
    _KE = kaggle_environments
    return _KE


def engine_version() -> str:
    return import_engine().__version__


def engine_dir() -> Path:
    return Path(import_engine().__file__).resolve().parent / "envs" / ENV_NAME


def engine_sha() -> str:
    return hashlib.sha256((engine_dir() / "kaggriculture.py").read_bytes()).hexdigest()[:12]


@lru_cache(maxsize=1)
def arena_git_sha() -> str:
    """Short git sha of the repo HEAD (+ '-dirty' if the arena package has local edits).
    KAGGRI_ARENA_SHA overrides it (remote runs ship a tarball without .git)."""
    if os.environ.get("KAGGRI_ARENA_SHA"):
        return os.environ["KAGGRI_ARENA_SHA"]
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
                             capture_output=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "arena"], cwd=REPO_ROOT, text=True,
                               capture_output=True, timeout=10).stdout.strip()
        return (sha or "nogit") + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def file_git_sha(path: Path) -> str | None:
    """Short commit sha of the last commit touching `path`, only if the file is committed,
    unmodified in the working tree AND contains nothing git does not track; otherwise None
    (caller falls back to a content hash over every file). The last condition matters: model
    weights are gitignored (*.npz etc.), so a snapshot with weights in it must never be
    identified by the commit sha alone -- re-registering the same name at the same commit
    with different weights would silently resolve to the old snapshot (2026-08-19 hazard)."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--", str(rel)], cwd=REPO_ROOT, text=True,
                                capture_output=True, timeout=10).stdout.strip()
        if status:
            return None
        # gitignored files inside the snapshot (untracked non-ignored ones already show in status)
        ignored = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", str(rel)],
                                 cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
        if any(ln.strip() and "__pycache__" not in ln and not ln.endswith(".pyc") for ln in ignored.splitlines()):
            return None
        sha = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(rel)], cwd=REPO_ROOT, text=True,
                             capture_output=True, timeout=10).stdout.strip()
        return sha or None
    except Exception:
        return None


def content_hash(path: Path) -> str:
    """Short sha256 over a file, or over every file of a directory (sorted, relative names)."""
    h = hashlib.sha256()
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(str(f.relative_to(path)).replace("\\", "/").encode())
            h.update(f.read_bytes())
    else:
        h.update(path.read_bytes())
    return h.hexdigest()[:10]
