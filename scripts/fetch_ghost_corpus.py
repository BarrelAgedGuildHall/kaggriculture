"""G-1 — build the tiered ghost corpus of top-ladder replays (ruling R-2 Stage 1).

Tiers of public-leaderboard teams so that BC ghosts can be trained on each tier's
behavioural centroid (G-2) and so that the uncontested-pool pricing table has a corpus to
price against:

    frontier  public LB rank 1-10
    upper     public LB rank 100-300
    field     public LB rank 2800-3300   (our own neighbourhood)
    rank1     public LB rank 1 alone     (granular review of the single best team)

The three ruling tiers are WINNING EPISODES ONLY: a behavioural centroid should be the
behaviour that wins.  `rank1` is a review tier and deliberately also carries a few LOSSES,
because a review of one specific team has to see what beats it.  The index does not carry an
`outcome` column -- `bank_margin` is the outcome, strictly positive on a win -- so a
consumer that wants the old invariant filters `bank_margin > 0` (or excludes `rank1`).

Selection is metadata-only and costs no replay bytes:

 1. `kaggle competitions leaderboard --download` gives the rank -> team map.
 2. The community mirror `georgymamarin/kaggriculture-episodes` (small csvs ONLY -- never
    its multi-GB replay parquets) gives, per team, which submission ids have played.
 3. The official `EpisodeService/ListEpisodes {"submission_id": N}` gives that submission's
    CURRENT episode list with both seats' banks and ratings.  The mirror lags ~1 day, so
    the mirror is used for DISCOVERY and the official API for the actual selection.
 4. Only then, one `kaggle competitions replay <id>` per selected episode.

The ~20 GB/day `kaggle/kaggriculture-episodes-YYYY-MM-DD` datasets are never touched.

STORAGE.  A raw replay is ~30 MB and gzips to ~0.45 MB (69x -- the observation dumps are
enormously repetitive), so the corpus is stored gzipped and the raw JSON is deleted as soon
as it is packed.  360 replays land in ~160 MB against the ruling's 5 GB hard cap.  The cap
is enforced on packed bytes with an abort-and-report clause, and peak scratch is one file.

ENGINE PIN.  Every replay carries `module_version`; anything other than the pin is
discarded and counted, per the ruling.  Ghosts are not comparable across an engine bump.

Idempotent and resumable: an episode already packed and indexed is skipped, so an
interrupted run continues where it stopped.  `index.csv` is authoritative -- a .json.gz
with no index row is incomplete and will be re-fetched.

Usage:
    uv run python scripts/fetch_ghost_corpus.py --plan          # select + report, no bytes
    uv run python scripts/fetch_ghost_corpus.py                 # fetch
    uv run python scripts/fetch_ghost_corpus.py --tier frontier --per-team 30
    # the rank-1 review tier: 40 most recent wins + 10 most recent losses, own selection file
    uv run python scripts/fetch_ghost_corpus.py --plan --tier rank1 --per-team 40 \
        --losses 10 --target 50 --min-teams 1 --selection-out selection-rank1.json
    uv run python scripts/fetch_ghost_corpus.py --use-selection <corpus>/meta/selection-rank1.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import REPO_ROOT  # noqa: E402

WAREHOUSE = Path(os.environ.get("GUILDHALL_WAREHOUSE") or (REPO_ROOT.parents[3] / "warehouse"))
CORPUS = WAREHOUSE / "datasets" / "ghost-corpus"
MIRROR_META = WAREHOUSE / "datasets" / "kaggriculture" / "meta"
API = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
MIRROR_SLUG = "georgymamarin/kaggriculture-episodes"

ENGINE_PIN = "1.32.7"
CAP_BYTES = 5 * 10 ** 9            # ruling R-2 G-1: hard cap, abort and report.
                                   # Decimal GB deliberately -- the conservative reading
                                   # of "5 GB" is the smaller number.
TIERS = {
    "frontier": (1, 10),
    "upper": (100, 300),
    "field": (2800, 3300),
    # A review tier, not one of the ruling's three behavioural bands. It overlaps `frontier`
    # by construction; the rank-1 team was NOT among the four `frontier` teams (they are
    # picked by how much ladder history the mirror has seen, and the very top of the LB
    # plays little), which is why it needs its own tier rather than a filter over frontier.
    "rank1": (1, 1),
}
RULING_TIERS = ("frontier", "upper", "field")   # winning episodes only -- see the module doc
TARGET_PER_TIER = 120              # 4 teams x ~30 episodes
MIN_TEAMS_PER_TIER = 4

# Engine 1.32.7 shipped 2026-08-15 and Kaggle adopted it within about a day, so an episode
# that ended before this is almost certainly 1.32.6 and will be discarded after a 30 MB
# download. This bites the LOW tiers specifically: a rank-3,000 team plays rarely, so its
# "30 most recent wins" reach much further back in time than a rank-10 team's do. Measured
# on the first field-tier run: 38 of 120 selected episodes came back 1.32.6, one whole team
# entirely. Filtering at selection is free; filtering after the download is not.
PIN_SHIPPED = "2026-08-16"

INDEX_FIELDS = [
    "episode_id", "tier", "team_id", "team_name", "lb_rank", "lb_score",
    "submission_id", "seat", "rating_at_play", "rating_after", "bank",
    "opp_team_id", "opp_submission_id", "opp_rating_at_play", "opp_bank",
    "bank_margin", "end_time", "engine_version", "bytes_raw", "bytes_gz",
    "replay_path", "fetched_at",
]


# ---------------------------------------------------------------- metadata sources

def post(path: str, body: dict, retries: int = 5) -> dict:
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
    """Prefer this interpreter's kaggle module -- a stray `kaggle` on PATH may belong to a
    different, older Python install."""
    try:
        import kaggle.cli  # noqa: F401
        return [sys.executable, "-m", "kaggle.cli"]
    except Exception:
        exe = shutil.which("kaggle")
        if exe:
            return [exe]
        raise SystemExit("kaggle CLI not available in this interpreter or on PATH")


def kaggle_env() -> dict:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set in the environment "
                         "(it is injected at session launch; never write it to a file)")
    env = dict(os.environ)
    if "KAGGLE_CONFIG_DIR" not in env:       # ignore any stale ~/.kaggle/kaggle.json
        env["KAGGLE_CONFIG_DIR"] = tempfile.mkdtemp(prefix="nokaggle-")
    return env


def refresh_leaderboard(env: dict) -> Path:
    meta = CORPUS / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    keep_snapshot(meta / "kaggriculture.zip")
    cmd = kaggle_cmd() + ["competitions", "leaderboard", "-c", "kaggriculture",
                          "--download", "-p", str(meta)]
    res = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if res.returncode != 0:
        raise SystemExit(f"leaderboard download failed: {res.stderr[:400]}")
    return meta / "kaggriculture.zip"


def keep_snapshot(zip_path: Path) -> None:
    """Archive an existing LB download under its own timestamped name before it is replaced.

    `selection.json` names its snapshot, so overwriting `kaggriculture.zip` would leave a
    frozen selection pointing at a leaderboard nobody can read again -- and a tier added
    later refreshes the LB by definition.
    """
    if not zip_path.exists():
        return
    try:
        _, name = read_leaderboard(zip_path)
    except Exception:
        return
    # the LB csv's name carries a timestamp with colons, which Windows will not have in a
    # filename (WinError 123, and copy2 fails *after* the download has been queued)
    stamp = "".join(c if c.isalnum() or c in "-._" else "-" for c in Path(name).stem)
    kept = zip_path.with_name(f"{zip_path.stem}-{stamp}.zip")
    if not kept.exists():
        shutil.copy2(zip_path, kept)


def read_leaderboard(zip_path: Path) -> tuple[list[dict], str]:
    z = zipfile.ZipFile(zip_path)
    name = z.namelist()[0]
    with io.TextIOWrapper(z.open(name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f)), name


def refresh_mirror_meta(env: dict) -> None:
    """Only episodes.csv -- ~9 MB. Never the replay parquets."""
    MIRROR_META.mkdir(parents=True, exist_ok=True)
    cmd = kaggle_cmd() + ["datasets", "download", MIRROR_SLUG, "-f", "episodes.csv",
                          "-p", str(MIRROR_META), "--force"]
    res = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"  warning: mirror refresh failed ({res.stderr[:200]}); using cached copy",
              file=sys.stderr)


def mirror_submissions_by_team() -> dict[str, list[str]]:
    """team_id -> submission ids, most-recently-played first.  Discovery only."""
    p = MIRROR_META / "episodes.csv"
    if not p.exists():
        raise SystemExit(f"mirror metadata missing: {p} (run without --no-refresh)")
    last: dict[tuple[str, str], str] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e["type"] != "EPISODE_TYPE_PUBLIC" or e["state"] != "COMPLETED":
                continue
            for seat in (0, 1):
                key = (e[f"team_{seat}"], e[f"sub_{seat}"])
                if e["end_time"] > last.get(key, ""):
                    last[key] = e["end_time"]
    out: dict[str, list[str]] = {}
    for (team, sub), t in sorted(last.items(), key=lambda kv: kv[1], reverse=True):
        out.setdefault(team, []).append(sub)
    return out


def episodes_for_submission(sub: str) -> list[dict]:
    """Current, authoritative episode list for one submission (official API)."""
    d = post("ListEpisodes", {"submission_id": int(sub)})
    names = {t["id"]: t.get("teamName", "") for t in d.get("teams", [])}
    rows = []
    for ep in d.get("episodes", []):
        agents = ep.get("agents") or []
        if len(agents) != 2 or ep.get("state") != "COMPLETED":
            continue
        for seat, a in enumerate(agents):
            opp = agents[1 - seat]
            if a.get("teamId") == opp.get("teamId"):
                continue                                    # self-play validation episode
            reward, opp_reward = a.get("reward"), opp.get("reward")
            if reward is None or opp_reward is None:
                continue
            rows.append({
                "episode_id": ep["id"],
                "end_time": ep.get("endTime", ""),
                "seat": a.get("index", 0),
                "team_id": str(a.get("teamId")),
                "team_name": names.get(a.get("teamId"), ""),
                "submission_id": a.get("submissionId"),
                "rating_at_play": round(a.get("initialScore") or 0.0, 1),
                "rating_after": round(a.get("updatedScore") or 0.0, 1),
                "bank": reward,
                "opp_team_id": str(opp.get("teamId")),
                "opp_submission_id": opp.get("submissionId"),
                "opp_rating_at_play": round(opp.get("initialScore") or 0.0, 1),
                "opp_bank": opp_reward,
                "won": reward > opp_reward,
            })
    return rows


# ---------------------------------------------------------------- selection

def select(lb: list[dict], per_team: int, target: int, only_tier: str | None,
           since: str = PIN_SHIPPED, exclude: set[int] | None = None,
           exclude_teams: set[str] | None = None,
           min_teams: int = MIN_TEAMS_PER_TIER,
           losses_per_team: int = 0,
           submissions: list[str] | None = None) -> dict[str, list[dict]]:
    """Pick episodes per tier: the `per_team` most recent WINS of each selected team, plus
    (review tiers only) its `losses_per_team` most recent LOSSES.

    `target` counts selected episodes of both kinds, so a 40-win + 10-loss tier is
    `--per-team 40 --losses 10 --target 50`.

    `submissions` replaces mirror discovery for this run.  The mirror does not see the top
    of the ladder: of its 49,985 episodes on 2026-08-22 the rank-1 team appeared in 3, none
    after 08-18, and the highest rating anywhere in the last day was 2,124 against a rank-1
    LB score of 3,142.  Discovery there returns a team's *retired* submission and the
    selection then clones the wrong agent while looking perfectly healthy.
    """
    by_team = {} if submissions else mirror_submissions_by_team()
    lb_by_team = {r["TeamId"]: r for r in lb}
    plan: dict[str, list[dict]] = {}

    for tier, (lo, hi) in TIERS.items():
        if only_tier and tier != only_tier:
            continue
        band = sorted((r for r in lb if lo <= int(r["Rank"]) <= hi), key=lambda r: int(r["Rank"]))
        # order candidate teams by how much ladder history the mirror has seen for them --
        # the top of the LB plays very little, so rank order alone starves the frontier tier
        cand = sorted(band, key=lambda r: -len(by_team.get(r["TeamId"], [])))
        picked: list[dict] = []
        teams_used = 0
        print(f"\n[{tier}] LB rank {lo}-{hi}: {len(band)} teams in band", file=sys.stderr)
        for row in cand:
            if len(picked) >= target and teams_used >= min_teams:
                break
            team = row["TeamId"]
            if exclude_teams and team in exclude_teams:
                continue
            subs = submissions or by_team.get(team, [])
            if not subs:
                continue
            wins: list[dict] = []
            losses: list[dict] = []
            seen: set[int] = set()
            for sub in subs[:6]:                       # newest submissions first
                try:
                    rows = episodes_for_submission(sub)
                except Exception as e:                 # a dead submission id is not fatal
                    print(f"    sub {sub}: {type(e).__name__} {e}", file=sys.stderr)
                    continue
                for r in rows:
                    if r["team_id"] != team or r["episode_id"] in seen:
                        continue
                    if not r["won"] and not losses_per_team:
                        continue                    # the three ruling tiers are wins-only
                    if since and r["end_time"][:10] < since:
                        continue                    # predates the engine pin -- see PIN_SHIPPED
                    if exclude and r["episode_id"] in exclude:
                        continue                    # already in the corpus
                    if not r["won"] and r["bank"] == r["opp_bank"]:
                        continue         # a tie is neither, and this ladder has never had one
                    seen.add(r["episode_id"])
                    (wins if r["won"] else losses).append(r)
                time.sleep(0.3)
                # NO early break on "quota met". "The N most recent wins" is a statement
                # about the team's episodes, not about one submission's: stopping at the
                # first submission that happens to fill the quota selects whichever
                # submission discovery listed first, which is not necessarily the newest.
            if not wins and not losses:
                continue
            wins.sort(key=lambda r: r["end_time"], reverse=True)     # most recent wins
            losses.sort(key=lambda r: r["end_time"], reverse=True)
            if losses_per_team and len(losses) < losses_per_team:
                print(f"    rank {int(row['Rank']):5d} team {team}: only {len(losses)} losses "
                      f"at the pin (asked for {losses_per_team}) -- reported, not padded",
                      file=sys.stderr)
            take = wins[:per_team] + losses[:losses_per_team]
            for r in take:
                r["tier"] = tier
                r["lb_rank"] = int(row["Rank"])
                r["lb_score"] = row["Score"]
                r["team_name"] = r["team_name"] or row["TeamName"]
            picked += take
            teams_used += 1
            print(f"    rank {int(row['Rank']):5d} team {team}: {len(take):3d} episodes "
                  f"({min(len(wins), per_team)}W of {len(wins)} found"
                  + (f", {min(len(losses), losses_per_team)}L of {len(losses)}"
                     if losses_per_team else "")
                  + f")  running total {len(picked)}", file=sys.stderr)
        plan[tier] = picked
        print(f"  -> {tier}: {len(picked)} episodes from {teams_used} teams", file=sys.stderr)
        if teams_used < min_teams:
            print(f"  !! only {teams_used} teams had usable history in this band "
                  f"(asked for {min_teams}) -- reported, not silently padded",
                  file=sys.stderr)
    return plan


# ---------------------------------------------------------------- index + fetch

def read_index() -> dict[int, dict]:
    p = CORPUS / "index.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {int(r["episode_id"]): r for r in csv.DictReader(f)}


def append_index(row: dict) -> None:
    p = CORPUS / "index.csv"
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in INDEX_FIELDS})


def packed_bytes(index: dict[int, dict]) -> int:
    return sum(int(r.get("bytes_gz") or 0) for r in index.values())


def fetch_one(ep: dict, env: dict, scratch: Path) -> tuple[str, dict | None]:
    """Fetch, engine-check, pack.  Returns (status, index_row)."""
    eid = ep["episode_id"]
    raw = scratch / f"{eid}.json"
    for stale in scratch.glob(f"*{eid}*"):
        stale.unlink(missing_ok=True)
    cmd = kaggle_cmd() + ["competitions", "replay", str(eid), "-p", str(scratch)]
    res = subprocess.run(cmd, env=env, text=True, capture_output=True)
    got = sorted(scratch.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if res.returncode != 0 or not got:
        return (f"fetch_failed: {(res.stderr or res.stdout)[:160]}", None)
    raw = got[-1]
    try:
        with open(raw, encoding="utf-8") as f:
            replay = json.load(f)
    except Exception as e:
        raw.unlink(missing_ok=True)
        return (f"unparseable: {type(e).__name__}", None)

    engine = replay.get("module_version", "")
    if engine != ENGINE_PIN:
        raw.unlink(missing_ok=True)
        return (f"engine_mismatch:{engine or 'unknown'}", None)

    bytes_raw = raw.stat().st_size
    out_dir = CORPUS / ep["tier"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{eid}.json.gz"
    tmp = dest.with_suffix(".gz.part")
    with open(raw, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=4 << 20)
    tmp.replace(dest)
    raw.unlink(missing_ok=True)

    row = dict(ep)
    row.update({
        "bank_margin": ep["bank"] - ep["opp_bank"],
        "engine_version": engine,
        "bytes_raw": bytes_raw,
        "bytes_gz": dest.stat().st_size,
        "replay_path": str(dest.relative_to(CORPUS)).replace("\\", "/"),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })
    return ("ok", row)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true", help="select and report only; fetch nothing")
    p.add_argument("--tier", choices=sorted(TIERS), help="restrict to one tier")
    p.add_argument("--per-team", type=int, default=30)
    p.add_argument("--target", type=int, default=TARGET_PER_TIER, help="episodes per tier")
    p.add_argument("--delay-s", type=float, default=1.0, help="pause between replay fetches")
    p.add_argument("--no-refresh", action="store_true", help="use cached LB/mirror metadata")
    p.add_argument("--use-selection", metavar="JSON", nargs="?", const="",
                   help="skip the selection API entirely and fetch a previously written "
                        "meta/selection.json (Kaggle rate-limits ListEpisodes, and a "
                        "re-selection would also silently drift the corpus)")
    p.add_argument("--cap-bytes", type=int, default=CAP_BYTES)
    p.add_argument("--since", default=PIN_SHIPPED,
                   help="ignore wins that ended before this date (they predate the engine "
                        "pin and would be downloaded only to be discarded)")
    p.add_argument("--min-teams", type=int, default=MIN_TEAMS_PER_TIER,
                   help="stop selecting once the target is met AND this many teams are in")
    p.add_argument("--top-up", action="store_true",
                   help="select only what the corpus is still missing: skip episodes and "
                        "teams already in index.csv, and write meta/selection-topup.json "
                        "instead of overwriting the frozen selection")
    p.add_argument("--losses", type=int, default=0, metavar="N",
                   help="also select each team's N most recent LOSSES (review tiers only -- "
                        "a behavioural review of one team has to see what beats it). The "
                        "index has no outcome column: bank_margin is the outcome")
    p.add_argument("--submissions", metavar="IDS",
                   help="comma-separated submission ids to select from, instead of "
                        "discovering them in the community mirror. Needed for the top of "
                        "the ladder, which the mirror barely covers -- discovery there "
                        "hands back a team's retired submission and looks healthy doing it")
    p.add_argument("--selection-out", metavar="NAME",
                   help="write the frozen selection to meta/<NAME> instead of "
                        "selection.json, so an added tier never overwrites the selection an "
                        "existing tier was built from")
    args = p.parse_args(argv)

    CORPUS.mkdir(parents=True, exist_ok=True)
    env = kaggle_env()

    if args.use_selection is not None:
        sel_path = Path(args.use_selection) if args.use_selection else CORPUS / "meta" / "selection.json"
        sel = json.load(open(sel_path))
        snap_name = sel["snapshot"]
        plan = {t: eps for t, eps in sel["tiers"].items()
                if not args.tier or t == args.tier}
        out_name = sel_path.name
        print(f"using cached selection {sel_path} (snapshot {snap_name}, "
              f"selected {sel['selected_at']})", file=sys.stderr)
    else:
        zip_path = CORPUS / "meta" / "kaggriculture.zip"
        if not args.no_refresh or not zip_path.exists():
            zip_path = refresh_leaderboard(env)
            refresh_mirror_meta(env)
        lb, snap_name = read_leaderboard(zip_path)
        print(f"public LB snapshot: {snap_name} ({len(lb)} teams)", file=sys.stderr)

        have = read_index() if args.top_up else {}
        exclude = set(have)
        # A team already represented in the corpus is not what a short tier needs -- it
        # needs another TEAM, or the tier is a 3-team centroid wearing a 4-team label.
        exclude_teams = {r["team_id"] for r in have.values()
                         if not args.tier or r["tier"] == args.tier} if args.top_up else set()
        if args.top_up:
            print(f"top-up: {len(exclude)} episodes and {len(exclude_teams)} teams already "
                  f"in the corpus will be skipped", file=sys.stderr)
        plan = select(lb, args.per_team, args.target, args.tier,
                      since=args.since, exclude=exclude, exclude_teams=exclude_teams,
                      min_teams=args.min_teams, losses_per_team=args.losses,
                      submissions=([s.strip() for s in args.submissions.split(",") if s.strip()]
                                   if args.submissions else None))
        (CORPUS / "meta").mkdir(parents=True, exist_ok=True)
        out_name = (args.selection_out
                    or ("selection-topup.json" if args.top_up else "selection.json"))
        json.dump({"snapshot": snap_name, "since": args.since, "top_up": args.top_up,
                   "per_team_wins": args.per_team, "per_team_losses": args.losses,
                   "submissions": args.submissions or None,
                   "selected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "tiers": {t: eps for t, eps in plan.items()}},
                  open(CORPUS / "meta" / out_name, "w"), indent=2)

    total = sum(len(v) for v in plan.values())
    print(f"\nselected {total} episodes across {len(plan)} tiers "
          f"-> {CORPUS / 'meta' / out_name}", file=sys.stderr)
    if args.plan:
        return 0

    index = read_index()
    used = packed_bytes(index)
    print(f"index holds {len(index)} episodes, {used/1e6:.1f} MB packed "
          f"(cap {args.cap_bytes/1e9:.1f} GB)", file=sys.stderr)

    scratch = Path(tempfile.mkdtemp(prefix="ghostfetch-"))
    stats = {"ok": 0, "skipped": 0, "engine_mismatch": 0, "failed": 0, "won": 0, "lost": 0}
    mismatches: dict[str, int] = {}
    try:
        for tier, eps in plan.items():
            for i, ep in enumerate(eps, 1):
                eid = ep["episode_id"]
                if eid in index:
                    stats["skipped"] += 1
                    continue
                if used >= args.cap_bytes:
                    print(f"\nABORT: 5 GB cap reached at {used/1e9:.2f} GB "
                          f"({stats['ok']} fetched this run). Nothing further downloaded.",
                          file=sys.stderr)
                    return 2
                status, row = fetch_one(ep, env, scratch)
                if status == "ok":
                    append_index(row)
                    index[eid] = row
                    used += int(row["bytes_gz"])
                    stats["ok"] += 1
                    stats["won" if row["bank_margin"] > 0 else "lost"] += 1
                    print(f"  [{tier} {i}/{len(eps)}] {eid} ok  "
                          f"{int(row['bytes_raw'])/1e6:.1f}MB -> {int(row['bytes_gz'])/1e3:.0f}kB "
                          f"| corpus {used/1e6:.0f} MB", file=sys.stderr)
                elif status.startswith("engine_mismatch"):
                    stats["engine_mismatch"] += 1
                    v = status.split(":", 1)[1]
                    mismatches[v] = mismatches.get(v, 0) + 1
                    print(f"  [{tier} {i}/{len(eps)}] {eid} DISCARDED engine={v}", file=sys.stderr)
                else:
                    stats["failed"] += 1
                    print(f"  [{tier} {i}/{len(eps)}] {eid} {status}", file=sys.stderr)
                time.sleep(args.delay_s)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nfetched {stats['ok']} ({stats['won']}W / {stats['lost']}L), "
          f"skipped {stats['skipped']} (already indexed), "
          f"engine-discarded {stats['engine_mismatch']} {mismatches or ''}, "
          f"failed {stats['failed']}", file=sys.stderr)
    print(f"corpus now {len(index)} episodes, {used/1e6:.1f} MB packed "
          f"({100*used/args.cap_bytes:.2f}% of the 5 GB cap) -> {CORPUS}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
