"""Repo-relative locations used by the arena. All state is under the git-ignored data/arena/."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OFFICIAL_REPLAYS = DATA_DIR / "replays" / "official"

# ARENA_DIR can be relocated (tests, remote runs) with the KAGGRI_ARENA_DIR env var.
ARENA_DIR = Path(os.environ.get("KAGGRI_ARENA_DIR") or (DATA_DIR / "arena"))
REGISTRY_PATH = ARENA_DIR / "registry.json"
AGENTS_DIR = ARENA_DIR / "agents"
DB_PATH = ARENA_DIR / "arena.sqlite"
REPLAYS_DIR = ARENA_DIR / "replays"
RUNS_DIR = ARENA_DIR / "runs"           # per-run sqlite deltas (remote runs land here)
PROGRESS_DIR = ARENA_DIR / "progress"   # one json per live run, read by `arena serve`
LEADERBOARD_MD = ARENA_DIR / "leaderboard.md"
LEADERBOARD_JSON = ARENA_DIR / "leaderboard.json"
WORKSHOP_LEADERBOARD = REPO_ROOT.parents[3] / "workshop" / "notes" / "leaderboard-internal.md"

ENV_NAME = "kaggriculture"
BUILTIN_AGENTS = ("random", "starter", "pass")

# Seed hygiene (strategy section 3): tune/screen on 0..199, confirm on held-out 1000..1199.
SCREEN_SEED_BASE = 0
CONFIRM_SEED_BASE = 1000
SCREEN_N = 64
CONFIRM_N = 400
LADDER_SEED_BASE = 2000     # the hub's ladder band; strictly above screen (0..) and confirm (1000..)


def ensure_dirs() -> None:
    for d in (ARENA_DIR, AGENTS_DIR, REPLAYS_DIR, RUNS_DIR, PROGRESS_DIR):
        d.mkdir(parents=True, exist_ok=True)
