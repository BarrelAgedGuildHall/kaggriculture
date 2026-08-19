"""Kaggler's internal arena ("the Pit") -- the single source of truth for which agent is better.

Run it as `uv run python -m arena <command>`; see `python -m arena --help` and the README
section "Arena". Everything persistent lives under the git-ignored `data/arena/`:
`registry.json` (agents + champions), `agents/` (immutable snapshots), `arena.sqlite`
(every game ever played), `replays/` (kept only on crash/timeout/loss), `leaderboard.*`.
"""
from __future__ import annotations

__version__ = "1.0.0"
