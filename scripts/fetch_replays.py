"""Download official Kaggriculture replays with the Kaggle CLI.

Kaggle publishes every finished ladder episode as ~30 MB JSON: a daily dataset per day
(`kaggle/kaggriculture-episodes-YYYY-MM-DD`, files named `<episode_id>.json`) indexed by
`kaggle/kaggriculture-episodes-index` (manifest.csv). Any single episode can also be pulled
directly with `kaggle competitions replay <episode_id>`, which is what this script does.

Auth: the Kaggle CLI (>=2.2) reads KAGGLE_API_TOKEN from the environment. Never write the
token to a file in this repo. If an old ~/.kaggle/kaggle.json interferes, point
KAGGLE_CONFIG_DIR at an empty directory (this script does that by default).

Usage:
    uv run python scripts/fetch_replays.py --episodes 93459920 93459927
    uv run python scripts/fetch_replays.py --daily 2026-08-16 --n 5          # first N ids of that day's dataset
    uv run python scripts/fetch_replays.py --index                           # just download manifest.csv

Replays land in data/replays/official/ (git-ignored).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import DATA_DIR, OFFICIAL_REPLAYS  # noqa: E402

INDEX_SLUG = "kaggle/kaggriculture-episodes-index"
DAILY_SLUG = "kaggle/kaggriculture-episodes-{date}"


def kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle.cli"]


def run(args: list[str], env: dict, capture=False) -> str:
    cmd = kaggle_cmd() + args
    print("$ kaggle " + " ".join(args), file=sys.stderr)
    res = subprocess.run(cmd, env=env, text=True, capture_output=capture)
    if res.returncode != 0:
        if capture:
            print(res.stdout, res.stderr, file=sys.stderr)
        raise SystemExit(f"kaggle command failed ({res.returncode}): {' '.join(args)}")
    return res.stdout if capture else ""


def list_daily_episode_ids(date: str, env: dict, page_size: int = 500) -> list[int]:
    out = run(["datasets", "files", DAILY_SLUG.format(date=date), "--page-size", str(page_size), "-v"], env, capture=True)
    ids = []
    for line in out.splitlines():
        name = line.split(",")[0].strip()
        if name.endswith(".json") and name[:-5].isdigit():
            ids.append(int(name[:-5]))
    return ids


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", nargs="*", type=int, default=[], help="episode ids to download")
    p.add_argument("--daily", help="date YYYY-MM-DD of a daily dataset to sample episode ids from")
    p.add_argument("--n", type=int, default=3, help="how many episodes to take from --daily")
    p.add_argument("--index", action="store_true", help="download the episodes index manifest to data/index/")
    p.add_argument("--out", default=str(OFFICIAL_REPLAYS))
    p.add_argument("--keep-config", action="store_true", help="do not override KAGGLE_CONFIG_DIR")
    args = p.parse_args(argv)

    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set in the environment (export it in-shell; never write it to a file)")

    env = dict(os.environ)
    tmp = None
    if not args.keep_config and "KAGGLE_CONFIG_DIR" not in env:
        tmp = tempfile.mkdtemp(prefix="nokaggle-")
        env["KAGGLE_CONFIG_DIR"] = tmp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.index:
            idx = DATA_DIR / "index"
            idx.mkdir(parents=True, exist_ok=True)
            run(["datasets", "download", INDEX_SLUG, "-p", str(idx), "--unzip", "-o"], env)

        ids = list(args.episodes)
        if args.daily:
            daily = list_daily_episode_ids(args.daily, env)
            print(f"{len(daily)} episodes listed for {args.daily}; taking {args.n}", file=sys.stderr)
            ids += daily[: args.n]

        for eid in ids:
            target = out / f"episode-{eid}-replay.json"
            if target.exists():
                print(f"exists: {target}", file=sys.stderr)
                continue
            run(["competitions", "replay", str(eid), "-p", str(out)], env)
        if ids:
            print("\n".join(str(out / f"episode-{e}-replay.json") for e in ids))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
