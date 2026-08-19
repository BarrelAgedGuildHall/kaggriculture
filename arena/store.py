"""SQLite store for every game the arena has played (data/arena/arena.sqlite, git-ignored).

One row per game, keyed by `game_id` (sha1 of p0|p1|seed|engine, see scheduler.game_id), so
re-running the same pairing on the same seed upserts, and results produced elsewhere (remote
node, another clone) merge by id with `import_db`.
"""
from __future__ import annotations

import csv
import sqlite3
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
    con.commit()
    return con


def upsert_rows(con: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    ph = ", ".join("?" for _ in COLNAMES)
    con.executemany(
        f"INSERT OR REPLACE INTO games ({', '.join(COLNAMES)}) VALUES ({ph})",
        [tuple(r.get(c) for c in COLNAMES) for r in rows],
    )
    con.commit()
    return len(rows)


def record_run(con: sqlite3.Connection, run: dict) -> None:
    keys = ["run_id", "mode", "agents", "n_games", "seeds", "engine", "arena_sha", "host", "workers", "wall_s",
            "started_at", "finished_at"]
    con.execute(f"INSERT OR REPLACE INTO runs ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})",
                tuple(run.get(k) for k in keys))
    con.commit()


def all_games(con: sqlite3.Connection, where: str = "", params: tuple = ()) -> list[dict]:
    sql = "SELECT * FROM games" + (f" WHERE {where}" if where else "") + " ORDER BY created_at, game_id"
    return [dict(r) for r in con.execute(sql, params)]


def games_between(con: sqlite3.Connection, a: str, b: str, seed_lo: int | None = None, seed_hi: int | None = None) -> list[dict]:
    where = "((p0 = ? AND p1 = ?) OR (p0 = ? AND p1 = ?))"
    params: list = [a, b, b, a]
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


def export_csv(con: sqlite3.Connection, out: Path) -> int:
    rows = all_games(con)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLNAMES)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
