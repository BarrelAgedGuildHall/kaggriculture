"""Put `agents/team_a/bc` on sys.path and re-export its modules.

Every PPO module imports the BC policy through here, so there is exactly one copy of
`policy.py` in the process and the "wrapper used in training is byte-identical to the one in
the packaged artefact" guarantee (spec section 3.2) is structural rather than a promise.
"""
from __future__ import annotations

import sys
from pathlib import Path

BC_DIR = Path(__file__).resolve().parents[1] / "bc"
WEIGHTS_DIR = BC_DIR / "weights"
if str(BC_DIR) not in sys.path:
    sys.path.insert(0, str(BC_DIR))

import features as F      # noqa: E402,F401
import net as N           # noqa: E402,F401
import policy as P        # noqa: E402,F401
