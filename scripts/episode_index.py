"""Build a cheap metadata index of Kaggriculture ladder episodes.

Kaggle's per-episode replay JSON is ~30 MB, so you must not download episodes to find
out whether they are worth having. Two public endpoints make targeted selection cheap:

1. `kaggle datasets files kaggle/kaggriculture-episodes-<date>` lists that day's episode
   ids (file names only, no payload).
2. `POST https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes` with
   `{"ids": [...]}` returns, for each episode: both agents' `submissionId`, `teamId`,
   `reward` (final bank) and `initialScore`/`updatedScore` (ladder rating before/after).
   It also returns team names. No auth needed; it is the same call the website makes.
   `{"submission_id": N}` returns every episode played by one submission.

Output: one CSV row per (episode, seat) at data/index/episodes-<date>.csv with the fields
needed to pick behaviour-cloning teachers: rating, reward, win flag, opponent rating.

Usage:
    uv run python scripts/episode_index.py --daily 2026-08-18
    uv run python scripts/episode_index.py --daily 2026-08-18 --out data/index
    uv run python scripts/episode_index.py --submission 55628543
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import DATA_DIR  # noqa: E402

API = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
DAILY_SLUG = "kaggle/kaggriculture-episodes-{date}"
BATCH = 100


def post(path: str, body: dict, retries: int = 4) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            API + path, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except OSError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def kaggle_cmd() -> list[str]:
    """Prefer this interpreter's kaggle module: a stray `kaggle` on PATH may belong to a
    different (older, broken) Python install."""
    try:
        import kaggle.cli  # noqa: F401
        return [sys.executable, "-m", "kaggle.cli"]
    except Exception:
        exe = shutil.which("kaggle")
        if exe:
            return [exe]
        raise SystemExit("kaggle CLI not available in this interpreter or on PATH")


def daily_episode_ids(date: str, page_size: int = 1000) -> list[int]:
    """Episode ids present in that day's dataset (file names only -- no payload)."""
    env = dict(os.environ)
    tmp = None
    if "KAGGLE_CONFIG_DIR" not in env:
        tmp = tempfile.mkdtemp(prefix="nokaggle-")
        env["KAGGLE_CONFIG_DIR"] = tmp
    try:
        cmd = kaggle_cmd() + ["datasets", "files", DAILY_SLUG.format(date=date),
                              "--page-size", str(page_size), "-v"]
        res = subprocess.run(cmd, env=env, text=True, capture_output=True)
        if res.returncode != 0:
            raise SystemExit(f"kaggle datasets files failed: {res.stderr[:500]}")
        ids = []
        for line in res.stdout.splitlines():
            name = line.split(",")[0].strip()
            if name.endswith(".json") and name[:-5].isdigit():
                ids.append(int(name[:-5]))
        return ids
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def fetch_metadata(ids: list[int]) -> tuple[list[dict], dict[int, str]]:
    episodes, team_names = [], {}
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        d = post("ListEpisodes", {"ids": chunk})
        episodes += d.get("episodes", [])
        for t in d.get("teams", []):
            team_names[t["id"]] = t.get("teamName", "")
        print(f"  metadata {min(i + BATCH, len(ids))}/{len(ids)}", file=sys.stderr)
        time.sleep(0.3)
    return episodes, team_names


def rows_from_episodes(episodes: list[dict], team_names: dict[int, str]) -> list[dict]:
    rows = []
    for ep in episodes:
        agents = ep.get("agents") or []
        if len(agents) != 2 or ep.get("state") != "COMPLETED":
            continue
        for seat, a in enumerate(agents):
            # 'index' is absent for seat 0 in the API payload.
            idx = a.get("index", 0)
            opp = agents[1 - seat]
            reward, opp_reward = a.get("reward"), opp.get("reward")
            if reward is None or opp_reward is None:
                continue
            rows.append({
                "episode_id": ep["id"],
                "end_time": ep.get("endTime", ""),
                "seat": idx,
                "team_id": a.get("teamId"),
                "team_name": team_names.get(a.get("teamId"), ""),
                "submission_id": a.get("submissionId"),
                "rating_before": round(a.get("initialScore") or 0.0, 1),
                "rating_after": round(a.get("updatedScore") or 0.0, 1),
                "reward": reward,
                "opp_team_id": opp.get("teamId"),
                "opp_team_name": team_names.get(opp.get("teamId"), ""),
                "opp_rating_before": round(opp.get("initialScore") or 0.0, 1),
                "opp_reward": opp_reward,
                "won": int(reward > opp_reward),
                "tied": int(reward == opp_reward),
            })
    rows.sort(key=lambda r: (-r["rating_before"], r["episode_id"], r["seat"]))
    return rows


FIELDS = ["episode_id", "end_time", "seat", "team_id", "team_name", "submission_id",
          "rating_before", "rating_after", "reward", "opp_team_id", "opp_team_name",
          "opp_rating_before", "opp_reward", "won", "tied"]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--daily", help="YYYY-MM-DD: index every episode in that day's dataset")
    p.add_argument("--submission", type=int, help="index every episode played by one submission id")
    p.add_argument("--limit", type=int, default=0, help="cap the number of episode ids queried")
    p.add_argument("--out", default=str(DATA_DIR / "index"))
    args = p.parse_args(argv)

    if args.daily:
        ids = daily_episode_ids(args.daily)
        print(f"{len(ids)} episodes listed for {args.daily}", file=sys.stderr)
        if args.limit:
            ids = ids[: args.limit]
        eps, names = fetch_metadata(ids)
        write_csv(rows_from_episodes(eps, names), Path(args.out) / f"episodes-{args.daily}.csv")
    elif args.submission:
        d = post("ListEpisodes", {"submission_id": args.submission})
        names = {t["id"]: t.get("teamName", "") for t in d.get("teams", [])}
        rows = rows_from_episodes(d.get("episodes", []), names)
        write_csv(rows, Path(args.out) / f"episodes-sub-{args.submission}.csv")
    else:
        p.error("give --daily or --submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
