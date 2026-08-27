"""Capture, validate, extract, or reproduce package-bound state/action traces."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from trace_retention.pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

