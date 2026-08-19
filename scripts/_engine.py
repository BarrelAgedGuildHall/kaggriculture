"""Shared helpers for the scripts: quiet import of kaggle_environments, paths, JSON utils.

`import kaggle_environments` registers every bundled environment, and the OpenSpiel
bridge prints ~300 lines of "Unknown game 'universal_poker'" noise to stderr at the
file-descriptor level (C++ side). `import_engine()` silences that one import only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOCAL_REPLAYS = DATA_DIR / "replays" / "local"
OFFICIAL_REPLAYS = DATA_DIR / "replays" / "official"
ENV_NAME = "kaggriculture"


def import_engine():
    """Import kaggle_environments with stderr muted for the duration of the import."""
    stderr_fd = sys.stderr.fileno()
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
    return kaggle_environments


def engine_dir() -> Path:
    ke = import_engine()
    return Path(ke.__file__).resolve().parent / "envs" / ENV_NAME


def engine_version() -> str:
    return import_engine().__version__


def to_plain(obj):
    """Round-trip through JSON so Struct objects become plain dicts/lists."""
    return json.loads(json.dumps(obj))


def load_agent_callable(spec: str):
    """Resolve an agent spec: a built-in name ('random', 'pass', 'starter'),
    a path to a .py file exposing `agent(obs)`, or 'module:function'."""
    if spec in ("random", "pass", "starter"):
        return spec
    if spec.endswith(".py"):
        return str(Path(spec).resolve())
    if ":" in spec:
        import importlib

        mod, fn = spec.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    raise SystemExit(f"Unknown agent spec: {spec!r} (use a built-in name, a .py path, or module:function)")
