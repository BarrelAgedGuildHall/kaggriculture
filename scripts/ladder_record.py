"""Ladder records for our Kaggle submissions, and the Pit-vs-ladder calibration read.

Three stages, all from metadata only -- this never downloads a replay and never touches the
~20 GB/day `kaggle/kaggriculture-episodes-*` datasets:

  record     W-L-T, rating trajectory and opponent context per submission, plus the exact
             markdown cell to paste into `workshop/notes/submissions.md`.
  correlate  the Pit's evidence (BT rating, win rate, head-to-head) against the ladder's,
             with rank-order agreement.
  calibrate  opponent-adjusted ladder Elo by maximum likelihood, and the single scale factor
             between Pit BT points and ladder Elo points.

Standing use is the weekly calibration read; see `docs/ops/ladder-record.md`.

THE LOAD-BEARING RULE -- Kaggle's upload validation is a SELF-PLAY episode. Both seats carry
our own submission id and our own team id, and it is the episode that stamps the initial 600
(`rating_before` 0 -> `rating_after` 600 on BOTH seats). Counted, it hands every agent a free
+2 W, a `rating_start` of 0, and one spurious "rating movement disagrees with the bank
comparison" row. It is excluded here by `split_validation()`, which then ASSERTS that no
self-play row survived into the ladder set -- the exclusion is enforced in code rather than
left to a comment, because every stage below shares that one filter.

Usage:
    uv run python scripts/ladder_record.py record --fetch --out data/ladder
    uv run python scripts/ladder_record.py record --from data/ladder --ledger-cells
    uv run python scripts/ladder_record.py correlate --from data/ladder
    uv run python scripts/ladder_record.py calibrate --from data/ladder
    uv run python scripts/ladder_record.py all --fetch --out data/ladder
Exit codes: 0 fine, 1 a consistency check failed under --strict, 2 could not reach a source.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import DATA_DIR  # noqa: E402
from episode_index import post, rows_from_episodes, write_csv  # noqa: E402

HUB = "http://barrelagedworkflows-1.tail547ecd.ts.net:8765"

# Our Kaggle submissions and the Pit agent each one was built from. Maintained by hand
# alongside `workshop/notes/submissions.md`: add a row when the CEO submits. `None` means the
# agent is not (or is no longer) registered on the hub under a resolvable id.
KNOWN_SUBMISSIONS: dict[int, str | None] = {
    55636779: "TeamA-EnsembleBC-v3@12a5398",
    55647331: "TeamB-Millrace-feedchain@e8a76b6",
    55658537: "TeamB-Millrace-melonrace@725fcf7",
    55671796: "TeamA-EnsembleBC-rebuild-v2@a513403",
    55681086: "TeamB-Millrace-strawberry@h9d854a3d8c",
}

# Below this many ladder games an agent is not ranked or calibrated: a 1-1 record carries a
# +-40 pp win-rate interval, and the Elo MLE is not identified at all on a clean sweep.
MIN_GAMES = 10

ELO_SCALE = 400.0
LN10_OVER_SCALE = math.log(10) / ELO_SCALE
EPS = 1e-9


class LadderDataError(RuntimeError):
    """A structural assumption about the episode metadata no longer holds."""


# --------------------------------------------------------------------------- loading


def _num(row: dict) -> dict:
    out = dict(row)
    for k in ("rating_before", "rating_after", "reward", "opp_reward", "opp_rating_before"):
        out[k] = float(row[k])
    for k in ("won", "tied", "seat", "episode_id", "submission_id"):
        out[k] = int(row[k])
    return out


def read_seat_rows(path: Path, submission_id: int) -> list[dict]:
    """Our own seat's rows from an `episode_index.py --submission` CSV.

    That CSV holds one row per (episode, seat), so it contains the OPPONENT's seat too;
    filtering to our `submission_id` is what makes every count below per-agent.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = [_num(r) for r in csv.DictReader(f) if int(r["submission_id"]) == submission_id]
    rows.sort(key=lambda r: (r["end_time"], r["episode_id"], r["seat"]))
    return rows


def split_validation(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition our seat's rows into (ladder games, upload-validation rows).

    Kaggle validates an upload by running the submission against ITSELF, so both seats carry
    our team id. That episode is not a ladder result and must never reach a W-L-T.

    The exclusion is then re-checked against a SECOND, INDEPENDENT signature: the validation
    episode is the one that stamps the initial rating, so it is the only episode a submission
    ever plays with `rating_before == 0`. Checking that no surviving ladder row has a zero
    `rating_before` catches a leak even if the team-id filter above is later weakened or
    bypassed -- a tautological "assert the filter filtered" would not.

    Raises LadderDataError if a validation row leaked into the ladder set, if the rows come
    from more than one submission (the per-seat filter in `read_seat_rows` was skipped, so
    the opponent's rows are being counted as ours), or if more than one validation episode is
    present (the metadata shape changed and every assumption here needs re-checking).
    """
    subs = {r["submission_id"] for r in rows}
    if len(subs) > 1:
        raise LadderDataError(
            f"rows span {len(subs)} submissions {sorted(subs)}; an episode_index CSV holds one "
            "row per (episode, seat), so it must be filtered to our own seat first"
        )
    ladder = [r for r in rows if r["opp_team_id"] != r["team_id"]]
    validation = [r for r in rows if r["opp_team_id"] == r["team_id"]]

    leaked = sorted({r["episode_id"] for r in ladder if r["rating_before"] == 0.0})
    if leaked:
        raise LadderDataError(
            f"upload-validation rows leaked into the ladder set: episodes {leaked} have "
            "rating_before == 0, which only the initial rating-stamping self-play episode "
            "does. Counted, it inflates W-L-T by +2 W and zeroes rating_start."
        )
    if len({r["episode_id"] for r in validation}) > 1:
        raise LadderDataError(
            f"expected at most one upload-validation episode, found "
            f"{sorted({r['episode_id'] for r in validation})}"
        )
    return ladder, validation


def result_of(row: dict) -> str:
    """W/L/T from the ladder's own rating movement."""
    d = row["rating_after"] - row["rating_before"]
    return "W" if d > EPS else ("L" if d < -EPS else "T")


def check_result_consistency(rows: list[dict]) -> list[dict]:
    """Rows where the ladder's rating movement disagrees with the engine's bank comparison.

    Empirically Kaggriculture's ladder result IS the bank comparison (141/141 real games as of
    2026-08-22, zero ties ever), which is why W-L-T can be derived without downloading a
    replay. A disagreement means either a validation row leaked in, or Kaggle changed how the
    ladder scores -- both worth stopping for, so `--strict` turns this into exit 1.
    """
    bad = []
    for r in rows:
        bank = "T" if r["tied"] else ("W" if r["won"] else "L")
        if result_of(r) != bank:
            bad.append({"episode_id": r["episode_id"], "by_rating": result_of(r),
                        "by_bank": bank, "reward": r["reward"], "opp_reward": r["opp_reward"]})
    return bad


# --------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def elo_mle(games: list[tuple[float, float]]) -> tuple[float, float]:
    """Opponent-adjusted Elo strength by maximum likelihood, with the Fisher-information SE.

    `games` is [(opponent_rating, score)] with score in {0, 0.5, 1}. Model:
    P(win) = 1 / (1 + 10^((O - R)/ELO_SCALE)). The raw ladder rating is NOT usable for
    comparing our own agents to each other -- it is path-dependent and each agent played a
    different opponent pool -- so this conditions on the opponents actually faced.

    Returns (nan, nan) for a clean sweep, where the likelihood has no interior maximum.
    """
    if not games:
        return (float("nan"), float("nan"))
    score = sum(s for _, s in games)
    if score <= EPS or score >= len(games) - EPS:
        return (float("nan"), float("nan"))
    lo, hi = -5000.0, 10000.0
    for _ in range(200):  # bisect the score function, which decreases in R
        mid = (lo + hi) / 2
        grad = sum(s - 1 / (1 + 10 ** ((o - mid) / ELO_SCALE)) for o, s in games)
        lo, hi = (mid, hi) if grad > 0 else (lo, mid)
    r = (lo + hi) / 2
    info = 0.0
    for o, _ in games:
        p = 1 / (1 + 10 ** ((o - r) / ELO_SCALE))
        info += LN10_OVER_SCALE ** 2 * p * (1 - p)
    return (r, 1 / math.sqrt(info))


def games_needed(gap_elo: float, z: float = 1.96) -> float:
    """Ladder games EACH to resolve a gap of `gap_elo` between two of our agents at p~0.5."""
    se_one = 1 / math.sqrt(LN10_OVER_SCALE ** 2 * 0.25)   # per game
    return 2 * (se_one * z / gap_elo) ** 2


def linfit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """(slope, Pearson r) of y on x."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxy = sum((x - mx) * (y - my) for x, y in points)
    sxx = sum((x - mx) ** 2 for x, _ in points)
    syy = sum((y - my) ** 2 for _, y in points)
    return (sxy / sxx, sxy / math.sqrt(sxx * syy))


# --------------------------------------------------------------------------- the record


def summarise(submission_id: int, rows: list[dict]) -> dict:
    ladder, validation = split_validation(rows)
    n = len(ladder)
    rec: dict = {
        "submission_id": submission_id,
        "agent": KNOWN_SUBMISSIONS.get(submission_id),
        "n": n,
        "validation_episodes": len({r["episode_id"] for r in validation}),
        "inconsistent": check_result_consistency(ladder),
    }
    if n == 0:
        rec.update({"W": 0, "L": 0, "T": 0, "note": "no ladder games yet"})
        return rec
    w = sum(r["won"] for r in ladder)
    t = sum(r["tied"] for r in ladder)
    ratings = [r["rating_after"] for r in ladder]
    lo, hi = wilson(w, n)
    adj, se = elo_mle([(r["opp_rating_before"], r["won"] + 0.5 * r["tied"]) for r in ladder])
    rec.update({
        "W": w, "L": n - w - t, "T": t,
        "win_pct": 100.0 * w / n, "win_ci": [lo, hi],
        "rating": ratings[-1], "rating_min": min(ratings), "rating_max": max(ratings),
        "rating_start": ladder[0]["rating_before"],
        "opp_rating_mean": sum(r["opp_rating_before"] for r in ladder) / n,
        "adjusted_elo": adj, "adjusted_se": se,
        "bank_mean": sum(r["reward"] for r in ladder) / n,
        "opp_bank_mean": sum(r["opp_reward"] for r in ladder) / n,
        "first": ladder[0]["end_time"], "last": ladder[-1]["end_time"],
        "opponent_team_ids": sorted({r["opp_team_id"] for r in ladder}),
        "trajectory": [[r["end_time"][:19], result_of(r), r["rating_after"],
                        r["opp_rating_before"], r["opp_team_name"]] for r in ladder],
    })
    rec["bank_margin"] = rec["bank_mean"] - rec["opp_bank_mean"]
    return rec


def ledger_cell(rec: dict, as_of: str) -> str:
    """The markdown for the ledger's 'ladder record ... as of' column."""
    if rec["n"] == 0:
        return f"**0-0-0, no ladder games yet, as of {as_of}** -- re-read once it has played"
    body = (f"**{rec['W']}-{rec['L']}-{rec['T']}, rating {rec['rating']:.1f} "
            f"({rec['rating_min']:.1f}-{rec['rating_max']:.1f} over life), {rec['n']} games, "
            f"as of {as_of}**")
    if rec["n"] < MIN_GAMES:
        return body + (f" -- TOO EARLY TO READ (under {MIN_GAMES} games; excluded from the "
                       f"correlation analysis)")
    return body + (f" -- win {rec['win_pct']:.1f}% [{rec['win_ci'][0]:.1f}, {rec['win_ci'][1]:.1f}] "
                   f"vs opponents averaging {rec['opp_rating_mean']:.1f}; opponent-adjusted Elo "
                   f"{rec['adjusted_elo']:.1f} "
                   f"[{rec['adjusted_elo'] - 1.96 * rec['adjusted_se']:.1f}, "
                   f"{rec['adjusted_elo'] + 1.96 * rec['adjusted_se']:.1f}]; bank mu "
                   f"{rec['bank_mean']:,.0f} (margin {rec['bank_margin']:+,.0f})")


# --------------------------------------------------------------------------- sources


def fetch_submission(submission_id: int, out_dir: Path) -> Path:
    """One ListEpisodes call -> one CSV. Metadata only; no replay bytes."""
    d = post("ListEpisodes", {"submission_id": submission_id})
    names = {t["id"]: t.get("teamName", "") for t in d.get("teams", [])}
    rows = rows_from_episodes(d.get("episodes", []), names)
    path = out_dir / f"episodes-sub-{submission_id}.csv"
    write_csv(rows, path)
    return path


def fetch_pit(hub: str, include_retired: bool) -> dict:
    url = f"{hub}/api/leaderboard" + ("?include_retired=1" if include_retired else "")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def load_pit(args) -> tuple[dict, dict]:
    """(canonical, all-agent) hub leaderboards, from --pit-json or the live hub.

    The all-agent fit (`?include_retired=1`) is the only one that carries a BT number for a
    retired agent, but it refits over a different pool -- keep the two in separate columns.
    """
    if args.pit_json:
        canonical = json.loads(Path(args.pit_json).read_text(encoding="utf-8"))
        retired = (json.loads(Path(args.pit_retired_json).read_text(encoding="utf-8"))
                   if args.pit_retired_json else canonical)
        return canonical, retired
    try:
        return fetch_pit(args.hub, False), fetch_pit(args.hub, True)
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"could not reach the Pit hub at {args.hub}: {e}")


def gather(args) -> dict[int, dict]:
    subs = args.submission or sorted(KNOWN_SUBMISSIONS)
    src = Path(args.out if args.fetch else args.source)
    if args.fetch:
        src.mkdir(parents=True, exist_ok=True)
        for s in subs:
            fetch_submission(s, src)
    out = {}
    for s in subs:
        path = src / f"episodes-sub-{s}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path} -- run with --fetch, or point --from at the CSVs")
        out[s] = summarise(s, read_seat_rows(path, s))
    return out


# --------------------------------------------------------------------------- reports


def wlt(rec: dict) -> str:
    return f"{rec['W']}-{rec['L']}-{rec['T']}"


def report_record(records: dict[int, dict], as_of: str, cells: bool) -> None:
    print(f"{'submission':>10} {'agent':32} {'n':>3} {'W-L-T':>10} {'win%':>6} "
          f"{'rating':>8} {'adjElo':>8} {'oppMu':>7} {'bank mu':>9}")
    for s, r in records.items():
        label = (r["agent"] or "?").split("@")[0]
        if r["n"] == 0:
            print(f"{s:>10} {label:32} {0:>3} {'-':>10} {'-':>6} {'-':>8} {'-':>8} {'-':>7} {'-':>9}")
            continue
        adj = "-" if math.isnan(r["adjusted_elo"]) else f"{r['adjusted_elo']:8.1f}"
        print(f"{s:>10} {label:32} {r['n']:>3} {wlt(r):>10} {r['win_pct']:5.1f}% "
              f"{r['rating']:8.1f} {adj:>8} {r['opp_rating_mean']:7.1f} {r['bank_mean']:9,.0f}")
    if cells:
        print("\n-- ledger cells for workshop/notes/submissions.md --")
        for s, r in records.items():
            print(f"\n{s}:\n{ledger_cell(r, as_of)}")


def report_correlate(records: dict[int, dict], pit_all: dict) -> None:
    bt = {t["agent"]: t for t in pit_all["table"]}
    rated = [s for s, r in records.items() if r["n"] >= MIN_GAMES and r["agent"] in bt]
    thin = [s for s, r in records.items() if 0 < r["n"] < MIN_GAMES]
    print(f"\n== PIT vs LADDER (all-agent BT fit, {pit_all['n_games']:,} hub games) ==")
    print(f"{'agent':32} {'PitBT':>9} {'PitWR':>7} {'PitN':>8} | {'rating':>8} {'W-L-T':>10} {'adjElo':>8}")
    for s, r in sorted(records.items(), key=lambda kv: -bt.get(kv[1]["agent"], {}).get("bt", 0)):
        t = bt.get(r["agent"])
        if not t:
            continue
        lad = ("-" if r["n"] == 0
               else f"{r['rating']:8.1f} {wlt(r):>10} {r['adjusted_elo']:8.1f}")
        print(f"{r['agent'].split('@')[0]:32} {t['bt']:9.1f} {100*t['win_rate']:6.1f}% "
              f"{t['games']:8,} | {lad}")
    if thin:
        print(f"  excluded from ranking, under {MIN_GAMES} ladder games: "
              + ", ".join(f"{records[s]['agent'].split('@')[0]} (n={records[s]['n']})" for s in thin))
    if len(rated) < 3:
        print("  not enough agents with ladder data to rank")
        return
    pit_rank = {s: i for i, s in enumerate(sorted(rated, key=lambda s: -bt[records[s]["agent"]]["bt"]))}
    lad_rank = {s: i for i, s in enumerate(sorted(rated, key=lambda s: -records[s]["rating"]))}
    n = len(rated)
    d2 = sum((pit_rank[s] - lad_rank[s]) ** 2 for s in rated)
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    conc = sum(1 for i, a in enumerate(rated) for b in rated[i + 1:]
               if (pit_rank[a] - pit_rank[b]) * (lad_rank[a] - lad_rank[b]) > 0)
    pairs = n * (n - 1) // 2
    print(f"\n  Spearman rho = {rho:.3f}   Kendall tau = {(2*conc-pairs)/pairs:.3f} "
          f"({conc} concordant / {pairs-conc} discordant of {pairs})")
    print("  NOTE: read the per-pair rating gaps, not the concordance count -- the live ladder "
          "is path-dependent\n  and a gap under ~120 Elo is a coin flip at these sample sizes.")


def report_calibrate(records: dict[int, dict], pit_all: dict) -> None:
    bt = {t["agent"]: t for t in pit_all["table"]}
    dev = max(abs(t["bt"] - (1000 + ELO_SCALE / math.log(10) * t["bt_log"])) for t in pit_all["table"])
    print(f"\n== CALIBRATION ==\n  Pit BT is Elo-scaled: max deviation from "
          f"1000 + {ELO_SCALE/math.log(10):.2f}*bt_log over {len(pit_all['table'])} agents "
          f"is {dev:.3f} points -- so Pit BT gaps and ladder Elo gaps are in the SAME units.")
    usable = [s for s, r in records.items()
              if r["n"] >= MIN_GAMES and r["agent"] in bt and not math.isnan(r["adjusted_elo"])]
    if len(usable) < 3:
        print("  not enough agents with ladder data to calibrate")
        return
    slope, r = linfit([(bt[records[s]["agent"]]["bt"], records[s]["adjusted_elo"]) for s in usable])
    print(f"  n={len(usable)} agents   Pearson r = {r:.3f}   "
          f"slope = {slope:.3f} ladder-Elo per Pit-BT point (1.0 = the scale transfers)")
    print(f"\n{'pair':56} {'predicted':>10} {'measured':>10} {'95% CI':>22} {'ok?':>4}")
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            pred = slope * (bt[records[a]["agent"]]["bt"] - bt[records[b]["agent"]]["bt"])
            meas = records[a]["adjusted_elo"] - records[b]["adjusted_elo"]
            se = math.hypot(records[a]["adjusted_se"], records[b]["adjusted_se"])
            lo, hi = meas - 1.96 * se, meas + 1.96 * se
            name = f"{records[a]['agent'].split('@')[0]} - {records[b]['agent'].split('@')[0]}"
            print(f"{name:56} {pred:10.1f} {meas:10.1f} [{lo:8.1f},{hi:8.1f}] "
                  f"{'yes' if lo <= pred <= hi else 'NO':>4}")
    print("  a NO is a PAIR-SPECIFIC divergence: the Pit wrong about one agent rather than "
          "merely mis-scaled.\n  Escalate those; do not escalate the slope alone.")
    print("\n  ladder games needed EACH to resolve a gap between two of our agents:")
    for gap in (50, 100, 200, 330):
        print(f"    {gap:3d} Elo -> ~{games_needed(gap):.0f} games")


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["record", "correlate", "calibrate", "all"], nargs="?",
                   default="all")
    p.add_argument("--submission", type=int, action="append",
                   help="Kaggle submission id (repeatable); default = every known submission")
    p.add_argument("--fetch", action="store_true", help="pull fresh metadata (no replay bytes)")
    p.add_argument("--from", dest="source", default=str(DATA_DIR / "ladder"),
                   help="directory of existing episodes-sub-*.csv files")
    p.add_argument("--out", default=str(DATA_DIR / "ladder"), help="where --fetch writes")
    p.add_argument("--hub", default=HUB)
    p.add_argument("--pit-json", help="saved /api/leaderboard capture instead of the live hub")
    p.add_argument("--pit-retired-json", help="saved ?include_retired=1 capture")
    p.add_argument("--as-of", default="", help="UTC stamp for the ledger cells")
    p.add_argument("--ledger-cells", action="store_true",
                   help="also print the markdown for the submissions.md ladder column")
    p.add_argument("--json", help="write the full record to this path")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any ladder game's rating movement disagrees with its bank")
    args = p.parse_args(argv)

    records = gather(args)
    as_of = args.as_of or "<set --as-of>"

    bad = {s: r["inconsistent"] for s, r in records.items() if r["inconsistent"]}
    if bad:
        print("WARNING: rating movement disagrees with the bank comparison -- either a "
              "validation row leaked in\n         or Kaggle changed how the ladder scores:",
              file=sys.stderr)
        for s, rows in bad.items():
            print(f"  sub {s}: {rows}", file=sys.stderr)

    if args.stage in ("record", "all"):
        report_record(records, as_of, args.ledger_cells or args.stage == "record")
    if args.stage in ("correlate", "calibrate", "all"):
        _, pit_all = load_pit(args)
        if args.stage in ("correlate", "all"):
            report_correlate(records, pit_all)
        if args.stage in ("calibrate", "all"):
            report_calibrate(records, pit_all)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 1 if (bad and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
