"""exp-011 step 0 driver -- see `agents/team_a/ppo/calibrate.py` for what it measures.

Why this thin wrapper exists rather than `python -m agents.team_a.ppo.calibrate`: the arena's
worker pool uses the *spawn* start method, and a spawn child re-imports the parent's
`__main__`. Launched with `-m`, that re-import wedged every worker before it reported ready
(16 games, 0 done, no worker processes, 0.5% CPU -- a silent hang, not a crash). Every arena
driver in this repo is a top-level script for exactly this reason (`scripts/evaluate.py`);
this one follows the pattern instead of fighting it.

    uv run python scripts/ppo_calibrate.py --n 128 --workers 14 --out data/results/exp-011/calibration.json
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo.calibrate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
