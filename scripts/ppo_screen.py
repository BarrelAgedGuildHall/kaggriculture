"""exp-011 acceptance screen driver -- see agents/team_a/ppo/screen.py.

Top-level script on purpose (spawn workers re-import __main__; see scripts/ppo_calibrate.py).

    uv run python scripts/ppo_screen.py --checkpoint data/results/exp-011/run1/ckpt/iter020.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.team_a.ppo.screen import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
