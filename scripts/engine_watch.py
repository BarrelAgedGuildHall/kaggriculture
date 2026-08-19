"""Engine watch: is there a newer `kaggle-environments` on PyPI than the one we pin?

Kaggle adopts new releases within ~1 day and the Kaggriculture economy has changed between
releases (1.32.6 -> 1.32.7 changed three price curves), so every tuning result is tied to the
pinned version. Run this daily (cron/CI); it exits 1 when PyPI has a newer release so the job
turns red and someone bumps the pin, re-runs `replay_check.py` parity and the round-robin.

Usage:
    uv run python scripts/engine_watch.py            # prints current vs latest, exit 1 if newer
    uv run python scripts/engine_watch.py --json     # machine-readable line
Exit codes: 0 up to date, 1 newer release available, 2 could not reach PyPI.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPI_URL = "https://pypi.org/pypi/kaggle-environments/json"


def pinned_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'"kaggle-environments==([\d.]+)"', text)
    if not m:
        raise SystemExit("pyproject.toml has no exact kaggle-environments pin")
    return m.group(1)


def installed_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("kaggle-environments")
    except Exception:
        return None


def _key(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else 0 for x in v.split("."))


def latest_on_pypi(timeout: float = 20.0) -> tuple[str, str]:
    with urllib.request.urlopen(PYPI_URL, timeout=timeout) as r:
        data = json.load(r)
    latest = data["info"]["version"]
    uploaded = ""
    files = data.get("releases", {}).get(latest) or []
    if files:
        uploaded = max(f.get("upload_time", "") for f in files)
    return latest, uploaded


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    pinned = pinned_version()
    installed = installed_version()
    try:
        latest, uploaded = latest_on_pypi()
    except Exception as exc:
        print(f"ENGINE_WATCH error: could not reach PyPI ({exc!r}); pinned={pinned} installed={installed}")
        return 2
    newer = _key(latest) > _key(pinned)
    out = {"pinned": pinned, "installed": installed, "latest": latest, "latest_uploaded": uploaded, "newer": newer}
    if args.json:
        print(json.dumps(out))
    else:
        state = "NEWER RELEASE AVAILABLE" if newer else "up to date"
        print(f"ENGINE_WATCH {state}: pinned={pinned} installed={installed} pypi_latest={latest} (uploaded {uploaded or '?'})")
        if newer:
            print("  -> bump pyproject.toml + uv.lock, copy docs/engine, re-run scripts/replay_check.py on fresh "
                  "replays, re-run the arena round-robin; tuning results from the old engine are void.")
    return 1 if newer else 0


if __name__ == "__main__":
    sys.exit(main())
