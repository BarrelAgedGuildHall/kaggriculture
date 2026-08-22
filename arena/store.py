"""SQLite store for every game the arena has played (data/arena/arena.sqlite, git-ignored).

One row per game, keyed by `game_id` (sha1 of p0|p1|seed|engine, see scheduler.game_id), so
re-running the same pairing on the same seed upserts, and results produced elsewhere (remote
node, another clone) merge by id with `import_db`.

That upsert is deliberate -- a clean node run is *meant* to supersede a noisy local one, and
remote deltas are *meant* to merge -- but it used to be silent: re-running a pairing destroyed the
earlier measurement with no trace, and a `runs` row could end up claiming n_games=400 while not one
game in the store still carried its run_id (2026-08-21, run 20260821-080254-e57458). So a replaced
row whose run_id differs is copied into the `superseded` table first, and `integrity_report`
reports every run whose games no longer add up. The game key itself is unchanged: putting run_id
into it would stop the merge from merging and would make new rows incomparable with every row
already stored.
"""
from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path

from .paths import DB_PATH, ensure_dirs

COLUMNS = [
    ("game_id", "TEXT PRIMARY KEY"), ("run_id", "TEXT"), ("mode", "TEXT"),
    ("p0", "TEXT"), ("p1", "TEXT"), ("seed", "INTEGER"), ("steps", "INTEGER"),
    ("engine", "TEXT"), ("arena_sha", "TEXT"), ("host", "TEXT"),
    ("money0", "REAL"), ("money1", "REAL"), ("status0", "TEXT"), ("status1", "TEXT"),
    ("result0", "REAL"), ("crash0", "INTEGER"), ("crash1", "INTEGER"), ("timeout0", "INTEGER"), ("timeout1", "INTEGER"),
    ("p50_0", "REAL"), ("p99_0", "REAL"), ("max_0", "REAL"), ("p50_1", "REAL"), ("p99_1", "REAL"), ("max_1", "REAL"),
    ("over_budget0", "INTEGER"), ("over_budget1", "INTEGER"),
    ("error0", "TEXT"), ("error1", "TEXT"), ("harness_error", "TEXT"),
    ("steps_played", "INTEGER"), ("wall_s", "REAL"), ("replay_path", "TEXT"), ("created_at", "TEXT"),
]
COLNAMES = [c for c, _ in COLUMNS]

# What is kept when a row is replaced by a different run: enough to re-read the lost measurement
# and to say who overwrote it. Not the full row -- this is a ledger, not a second store.
SUPERSEDED_COLS = ["superseded_at", "superseded_by_run", "game_id", "run_id", "mode", "p0", "p1",
                   "seed", "engine", "arena_sha", "host", "money0", "money1", "result0",
                   "crash0", "crash1", "timeout0", "timeout1", "p99_0", "p99_1", "max_0", "max_1",
                   "created_at"]


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path) if path else DB_PATH
    if path == DB_PATH:
        ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30)
    con.row_factory = sqlite3.Row
    cols = ", ".join(f"{c} {t}" for c, t in COLUMNS)
    con.execute(f"CREATE TABLE IF NOT EXISTS games ({cols})")
    con.execute("CREATE INDEX IF NOT EXISTS idx_games_p0 ON games(p0)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_games_p1 ON games(p1)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_games_run ON games(run_id)")
    con.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, mode TEXT, agents TEXT, n_games INTEGER, seeds TEXT,
        engine TEXT, arena_sha TEXT, host TEXT, workers INTEGER, wall_s REAL, started_at TEXT, finished_at TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS superseded (" + ", ".join(
        f"{c} {'INTEGER' if c.startswith(('seed', 'crash', 'timeout')) else 'REAL' if c.startswith(('money', 'result', 'p99', 'max')) else 'TEXT'}"
        for c in SUPERSEDED_COLS) + ")")
    con.execute("CREATE INDEX IF NOT EXISTS idx_superseded_run ON superseded(run_id)")
    con.commit()
    return con


def _record_superseded(con: sqlite3.Connection, rows: list[dict]) -> int:
    """Copy any stored row that `rows` is about to replace **with a different run** into the
    supersession ledger. Same-run re-writes (a re-import of the same delta, a progress rewrite)
    are not supersessions and are not recorded."""
    ids = [r.get("game_id") for r in rows if r.get("game_id")]
    if not ids:
        return 0
    incoming = {r["game_id"]: r for r in rows if r.get("game_id")}
    kept = []
    for chunk in (ids[i:i + 400] for i in range(0, len(ids), 400)):
        q = f"SELECT * FROM games WHERE game_id IN ({', '.join('?' for _ in chunk)})"
        for row in con.execute(q, tuple(chunk)):
            old = dict(row)
            new_run = incoming[old["game_id"]].get("run_id") or ""
            if (old.get("run_id") or "") == new_run:
                continue
            old["superseded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            old["superseded_by_run"] = new_run
            kept.append(tuple(old.get(c) for c in SUPERSEDED_COLS))
    if kept:
        con.executemany(f"INSERT INTO superseded ({', '.join(SUPERSEDED_COLS)}) "
                        f"VALUES ({', '.join('?' for _ in SUPERSEDED_COLS)})", kept)
    return len(kept)


def upsert_rows(con: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    _record_superseded(con, rows)
    ph = ", ".join("?" for _ in COLNAMES)
    con.executemany(
        f"INSERT OR REPLACE INTO games ({', '.join(COLNAMES)}) VALUES ({ph})",
        [tuple(r.get(c) for c in COLNAMES) for r in rows],
    )
    con.commit()
    return len(rows)


def superseded_rows(con: sqlite3.Connection, run_id: str | None = None) -> list[dict]:
    """The measurements that later runs overwrote (most recent first)."""
    sql = "SELECT * FROM superseded" + (" WHERE run_id = ?" if run_id else "") + " ORDER BY superseded_at DESC"
    return [dict(r) for r in con.execute(sql, (run_id,) if run_id else ())]


def integrity_report(con: sqlite3.Connection) -> dict:
    """Orphan-run integrity check (NB-6).

    A run row records how many games it played. Because games upsert by `game_id`, a later run of
    the same pairing on the same seeds takes those rows over -- so a run can silently end up with
    fewer stored games than it played, or none at all. This reports every such run, what took its
    rows, and any game whose run_id has no run row at all.
    """
    stored = {r["run_id"]: r["n"] for r in
              con.execute("SELECT run_id, COUNT(*) AS n FROM games GROUP BY run_id")}
    lost: dict[str, dict[str, int]] = {}
    for r in con.execute("SELECT run_id, superseded_by_run, COUNT(*) AS n "
                         "FROM superseded GROUP BY run_id, superseded_by_run"):
        lost.setdefault(r["run_id"], {})[r["superseded_by_run"] or "(unknown)"] = r["n"]
    runs, incomplete = [], []
    for r in con.execute("SELECT * FROM runs ORDER BY started_at"):
        run = dict(r)
        have = stored.get(run["run_id"], 0)
        rec = {"run_id": run["run_id"], "mode": run["mode"], "agents": run["agents"],
               "started_at": run["started_at"], "n_games": run["n_games"], "stored": have,
               "missing": max(0, (run["n_games"] or 0) - have),
               "superseded_by": lost.get(run["run_id"], {})}
        runs.append(rec)
        if rec["missing"]:
            incomplete.append(rec)
    known = {r["run_id"] for r in runs}
    orphan_games = {rid: n for rid, n in stored.items() if rid and rid not in known}
    return {
        "runs": runs,
        "incomplete_runs": incomplete,
        "orphan_games": orphan_games,
        "superseded_total": con.execute("SELECT COUNT(*) FROM superseded").fetchone()[0],
        "ok": not incomplete and not orphan_games,
    }


def record_run(con: sqlite3.Connection, run: dict) -> None:
    keys = ["run_id", "mode", "agents", "n_games", "seeds", "engine", "arena_sha", "host", "workers", "wall_s",
            "started_at", "finished_at"]
    con.execute(f"INSERT OR REPLACE INTO runs ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})",
                tuple(run.get(k) for k in keys))
    con.commit()


def all_games(con: sqlite3.Connection, where: str = "", params: tuple = ()) -> list[dict]:
    sql = "SELECT * FROM games" + (f" WHERE {where}" if where else "") + " ORDER BY created_at, game_id"
    return [dict(r) for r in con.execute(sql, params)]


def games_between(con: sqlite3.Connection, a: str, b: str, seed_lo: int | None = None, seed_hi: int | None = None,
                  modes: tuple[str, ...] | None = None) -> list[dict]:
    where = "((p0 = ? AND p1 = ?) OR (p0 = ? AND p1 = ?))"
    params: list = [a, b, b, a]
    if modes:
        where += f" AND mode IN ({', '.join('?' for _ in modes)})"
        params += list(modes)
    if seed_lo is not None:
        where += " AND seed >= ?"
        params.append(seed_lo)
    if seed_hi is not None:
        where += " AND seed <= ?"
        params.append(seed_hi)
    return all_games(con, where, tuple(params))


def import_db(con: sqlite3.Connection, other: Path) -> int:
    """Merge every game row (and run row) from another arena sqlite/CSV into `con` by game_id."""
    other = Path(other)
    if other.suffix.lower() == ".csv":
        with open(other, newline="", encoding="utf-8") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        for r in rows:
            for c, t in COLUMNS:
                if c in r and r[c] not in (None, ""):
                    if t.startswith("INTEGER"):
                        r[c] = int(float(r[c]))
                    elif t.startswith("REAL"):
                        r[c] = float(r[c])
        return upsert_rows(con, rows)
    src = sqlite3.connect(str(other))
    src.row_factory = sqlite3.Row
    rows = [dict(r) for r in src.execute("SELECT * FROM games")]
    n = upsert_rows(con, rows)
    try:
        for r in src.execute("SELECT * FROM runs"):
            record_run(con, dict(r))
    except sqlite3.Error:
        pass
    src.close()
    return n


def count_games_for(con: sqlite3.Connection, agent_id: str) -> int:
    return con.execute("SELECT COUNT(*) FROM games WHERE p0 = ? OR p1 = ?",
                       (agent_id, agent_id)).fetchone()[0]


def relabel_agent(con: sqlite3.Connection, old_id: str, new_id: str) -> dict:
    """Rename an agent id across every row that mentions it, in ONE transaction.

    `game_id` is `sha1(p0|p1|seed|engine)`, so renaming a player without recomputing it would
    leave rows whose key no longer matches their content: re-running that pairing would insert
    a duplicate instead of upserting, and the store would slowly grow two histories of the
    same game. The ids are recomputed here, which is also why this cannot be a plain UPDATE --
    a recomputed id may collide with a row that already exists (the same pairing already
    played under the new name), and the newer row must win exactly as `upsert_rows` would.

    Either every row moves or none does: BEGIN IMMEDIATE ... COMMIT, rolled back on any error.
    """
    from .scheduler import game_id

    rows = [dict(r) for r in con.execute("SELECT * FROM games WHERE p0 = ? OR p1 = ?",
                                         (old_id, old_id))]
    runs = [dict(r) for r in con.execute("SELECT * FROM runs")]
    touched_runs = []
    for run in runs:
        agents = [a for a in (run.get("agents") or "").split(",") if a]
        if old_id in agents:
            run["agents"] = ",".join(new_id if a == old_id else a for a in agents)
            touched_runs.append(run)

    prev = con.isolation_level
    con.isolation_level = None
    try:
        con.execute("BEGIN IMMEDIATE")
        for row in rows:
            con.execute("DELETE FROM games WHERE game_id = ?", (row["game_id"],))
        for row in rows:
            if row["p0"] == old_id:
                row["p0"] = new_id
            if row["p1"] == old_id:
                row["p1"] = new_id
            row["game_id"] = game_id(row["p0"], row["p1"], row["seed"], row["engine"])
        collisions = _record_superseded(con, rows)   # NB-6: the same pairing already played
        ph = ", ".join("?" for _ in COLNAMES)        # under the new name -- do not lose it silently
        con.executemany(f"INSERT OR REPLACE INTO games ({', '.join(COLNAMES)}) VALUES ({ph})",
                        [tuple(r.get(c) for c in COLNAMES) for r in rows])
        for run in touched_runs:
            con.execute("UPDATE runs SET agents = ? WHERE run_id = ?", (run["agents"], run["run_id"]))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.isolation_level = prev
    return {"games": len(rows), "runs": len(touched_runs), "collisions": collisions}


def export_csv(con: sqlite3.Connection, out: Path) -> int:
    rows = all_games(con)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLNAMES)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
