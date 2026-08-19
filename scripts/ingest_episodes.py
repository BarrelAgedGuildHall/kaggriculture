"""Daily ingestion of top Kaggriculture episodes (+ all of our own) into the warehouse.

There is no per-episode metadata (ratings/agents/dates) in the official
`kaggle/kaggriculture-episodes-index` dataset -- that is just a 3 kB manifest.csv listing
20+ daily datasets, each a ~20 GiB dump of raw episode JSON with no ratings attached
(see docs/engine and the competition brief). Per-episode metadata with ratings comes from
the community mirror dataset `georgymamarin/kaggriculture-episodes`, which republishes
`episodes.csv` (episode_id, create_time, end_time, type, state, sub_N/team_N/bank_N/rating_N
per seat) and `agents.csv` alongside its own multi-GB replay parquets -- we download ONLY the
two small csvs (~13 MB total), never the parquets. See
https://www.kaggle.com/competitions/kaggriculture/discussion/731215 for the host's
explanation of how the official daily datasets are built (same selection idea: order by
average seat rating).

Per day, this script:
  1. refreshes the cached episodes.csv / agents.csv metadata (small files only)
  2. selects the top N episodes (by max seat rating, EPISODE_TYPE_PUBLIC only -- ladder
     games, not self-play validation) plus every episode involving one of our own
     submission ids (`--own-submissions`, none registered yet)
  3. fetches each selected episode's replay JSON with `kaggle competitions replay <id>`
     (~30 MB each) unless it is already on disk
  4. records/updates a row per episode in the warehouse index (idempotent -- an episode
     already indexed with its replay file present is skipped, so a partial/interrupted run
     resumes cleanly)

Layout under the warehouse (bulk tier, see CLAUDE.md "Storage tiers"):
    warehouse/datasets/kaggriculture/
        meta/episodes.csv, meta/agents.csv     -- cached community-mirror metadata
        replays/<engine_version>/<episode_id>.json
        index.csv                               -- episode_id, date, sub_0/1, team_0/1,
                                                    bank_0/1, rating_0/1, engine_version,
                                                    selection_reason, fetched_at

Usage:
    uv run python scripts/ingest_episodes.py --last-n-days 3
    uv run python scripts/ingest_episodes.py --days 2026-08-16 2026-08-17 2026-08-18
    uv run python scripts/ingest_episodes.py --last-n-days 1 --top-n 50 --own-submissions 12345 67890
    uv run python scripts/ingest_episodes.py --last-n-days 3 --dry-run     # select + report, no downloads

Rate-limit friendly: a small delay between `kaggle competitions replay` calls
(`--delay-s`, default 1.0), resumable, safe to re-run daily via cron.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import REPO_ROOT  # noqa: E402

MIRROR_SLUG = "georgymamarin/kaggriculture-episodes"
MODULE_VERSION_RE = re.compile(rb'"module_version"\s*:\s*"([^"]+)"')
INDEX_COLUMNS = [
    "episode_id", "date", "sub_0", "team_0", "bank_0", "rating_0",
    "sub_1", "team_1", "bank_1", "rating_1", "engine_version",
    "selection_reason", "fetched_at",
]


def warehouse_dir() -> Path:
    env = os.environ.get("GUILDHALL_WAREHOUSE")
    if env:
        return Path(env)
    # Fallback: repo is normally cloned inside a guild folder that junctions warehouse/.
    guess = REPO_ROOT.parent / "warehouse"
    if guess.exists():
        return guess
    raise SystemExit(
        "Cannot locate the warehouse: $GUILDHALL_WAREHOUSE is not set and no ../warehouse "
        "exists next to the repo. Run this from a Guildhall-launched session, or pass "
        "--warehouse-dir explicitly."
    )


def kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    return [sys.executable, "-m", "kaggle.cli"]


def run_kaggle(args: list[str], env: dict, capture: bool = False) -> str:
    cmd = kaggle_cmd() + args
    print("$ kaggle " + " ".join(args), file=sys.stderr)
    res = subprocess.run(cmd, env=env, text=True, capture_output=capture)
    if res.returncode != 0:
        if capture:
            print(res.stdout, res.stderr, file=sys.stderr)
        raise SystemExit(f"kaggle command failed ({res.returncode}): {' '.join(args)}")
    return res.stdout if capture else ""


def kaggle_env() -> tuple[dict, str | None]:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set in the environment (export it in-shell; never write it to a file)")
    env = dict(os.environ)
    tmp = None
    if "KAGGLE_CONFIG_DIR" not in env:
        tmp = tempfile.mkdtemp(prefix="nokaggle-")
        env["KAGGLE_CONFIG_DIR"] = tmp
    return env, tmp


def refresh_metadata(meta_dir: Path, env: dict) -> tuple[Path, Path]:
    meta_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("episodes.csv", "agents.csv"):
        run_kaggle(["datasets", "download", MIRROR_SLUG, "-f", fname, "-p", str(meta_dir), "-o"], env)
    return meta_dir / "episodes.csv", meta_dir / "agents.csv"


def load_episodes(episodes_csv: Path) -> list[dict]:
    with open(episodes_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_episodes(rows: list[dict], days: list[str], top_n: int, own_submissions: set[str]) -> dict[str, dict]:
    """Return {episode_id: {row, date, reason}} for the requested days."""
    by_day: dict[str, list[dict]] = {d: [] for d in days}
    for row in rows:
        date = row["create_time"][:10]
        if date in by_day:
            by_day[date].append(row)

    selected: dict[str, dict] = {}
    for date, day_rows in by_day.items():
        # Own-submission episodes: any type, always kept.
        own_rows = [
            r for r in day_rows
            if r["sub_0"] in own_submissions or r["sub_1"] in own_submissions
        ]
        for r in own_rows:
            selected[r["episode_id"]] = {"row": r, "date": date, "reason": "own"}

        # Top-N by max seat rating among ladder (public) games only.
        ladder_rows = [r for r in day_rows if r["type"] == "EPISODE_TYPE_PUBLIC"]

        def max_rating(r: dict) -> float:
            try:
                return max(float(r["rating_0"]), float(r["rating_1"]))
            except ValueError:
                return -1.0

        ladder_rows.sort(key=max_rating, reverse=True)
        for r in ladder_rows[:top_n]:
            eid = r["episode_id"]
            if eid not in selected:
                selected[eid] = {"row": r, "date": date, "reason": "top"}
    return selected


def load_index(index_csv: Path) -> dict[str, dict]:
    if not index_csv.exists():
        return {}
    with open(index_csv, newline="", encoding="utf-8") as f:
        return {row["episode_id"]: row for row in csv.DictReader(f)}


def append_index_row(index_csv: Path, row: dict) -> None:
    new_file = not index_csv.exists()
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(index_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def already_fetched(existing: dict, replays_dir: Path, episode_id: str) -> bool:
    row = existing.get(episode_id)
    if not row:
        return False
    ev = row.get("engine_version")
    if not ev:
        return False
    return (replays_dir / ev / f"{episode_id}.json").exists()


def fetch_episode(episode_id: str, env: dict, replays_dir: Path, tmp_dl: Path) -> str:
    """Download one replay, detect its engine version from the JSON header, move it into
    replays/<engine_version>/<episode_id>.json. Returns the engine version."""
    run_kaggle(["competitions", "replay", episode_id, "-p", str(tmp_dl), "-q"], env)
    src = tmp_dl / f"episode-{episode_id}-replay.json"
    if not src.exists():
        raise SystemExit(f"expected replay file missing after download: {src}")
    with open(src, "rb") as f:
        head = f.read(4096)
    m = MODULE_VERSION_RE.search(head)
    engine_version = m.group(1).decode() if m else "unknown"
    dest_dir = replays_dir / engine_version
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{episode_id}.json"
    shutil.move(str(src), str(dest))
    return engine_version


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", nargs="*", help="explicit list of dates YYYY-MM-DD")
    p.add_argument("--last-n-days", type=int, help="ingest the last N calendar days available (UTC today - 1 .. -N)")
    p.add_argument("--top-n", type=int, default=50, help="top episodes per day by max seat rating (default 50)")
    p.add_argument("--own-submissions", nargs="*", default=[], help="our Kaggle submission ids -- always ingested")
    p.add_argument("--warehouse-dir", default=None, help="override $GUILDHALL_WAREHOUSE")
    p.add_argument("--delay-s", type=float, default=1.0, help="delay between replay downloads (rate-limit friendly)")
    p.add_argument("--dry-run", action="store_true", help="select and report, download nothing")
    args = p.parse_args(argv)

    if not args.days and not args.last_n_days:
        p.error("pass --days or --last-n-days")

    if args.days:
        days = list(args.days)
    else:
        today = dt.datetime.now(dt.timezone.utc).date()
        # The mirror dataset lags by ~1 day; start from yesterday.
        days = [str(today - dt.timedelta(days=i)) for i in range(1, args.last_n_days + 1)]

    wh = Path(args.warehouse_dir) if args.warehouse_dir else warehouse_dir()
    base = wh / "datasets" / "kaggriculture"
    meta_dir = base / "meta"
    replays_dir = base / "replays"
    index_csv = base / "index.csv"

    env, cleanup_tmp = kaggle_env()
    try:
        episodes_csv, _agents_csv = refresh_metadata(meta_dir, env)
        rows = load_episodes(episodes_csv)
        own = set(args.own_submissions)
        selected = select_episodes(rows, days, args.top_n, own)
        print(f"days={days} candidates={len(selected)} (top_n={args.top_n}/day, own_submissions={sorted(own) or 'none'})")

        existing = load_index(index_csv)
        n_fetched = n_skipped = n_failed = 0
        total_bytes = 0

        for eid, info in sorted(selected.items(), key=lambda kv: int(kv[0])):
            row, date, reason = info["row"], info["date"], info["reason"]
            if already_fetched(existing, replays_dir, eid):
                n_skipped += 1
                continue
            if args.dry_run:
                print(f"[dry-run] would fetch {eid} ({date}, {reason})")
                continue
            tmp_dl = Path(tempfile.mkdtemp(prefix="episode-dl-"))
            try:
                engine_version = fetch_episode(eid, env, replays_dir, tmp_dl)
            except SystemExit as e:
                print(f"FAILED {eid}: {e}", file=sys.stderr)
                n_failed += 1
                continue
            finally:
                shutil.rmtree(tmp_dl, ignore_errors=True)

            dest = replays_dir / engine_version / f"{eid}.json"
            total_bytes += dest.stat().st_size
            append_index_row(index_csv, {
                "episode_id": eid,
                "date": date,
                "sub_0": row["sub_0"], "team_0": row["team_0"], "bank_0": row["bank_0"], "rating_0": row["rating_0"],
                "sub_1": row["sub_1"], "team_1": row["team_1"], "bank_1": row["bank_1"], "rating_1": row["rating_1"],
                "engine_version": engine_version,
                "selection_reason": reason,
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            })
            existing[eid] = {"engine_version": engine_version}
            n_fetched += 1
            time.sleep(args.delay_s)

        print(
            f"done: fetched={n_fetched} skipped_existing={n_skipped} failed={n_failed} "
            f"new_bytes={total_bytes} ({total_bytes / 1e9:.2f} GB) index={index_csv}"
        )
        return 1 if n_failed else 0
    finally:
        if cleanup_tmp:
            shutil.rmtree(cleanup_tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
