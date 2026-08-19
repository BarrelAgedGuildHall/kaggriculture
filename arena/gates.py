"""Promotion gates: an agent may become champion / be submitted only if, over the confirm set
against the current champion, it has 0 crashes, 0 timeouts, p99 turn <= 150 ms and a Wilson
95 % lower bound on its win rate strictly above 50 %.
"""
from __future__ import annotations

from pathlib import Path

from . import ratings, registry, store
from .paths import CONFIRM_N, CONFIRM_SEED_BASE

P99_LIMIT_MS = 150.0
MAX_LIMIT_MS = 400.0


def evaluate(name: str, team: str = "org", db_path: Path | None = None, opponent: str | None = None,
             seed_lo: int | None = CONFIRM_SEED_BASE, seed_hi: int | None = CONFIRM_SEED_BASE + CONFIRM_N // 2 - 1,
             min_games: int = CONFIRM_N, p99_limit: float = P99_LIMIT_MS, max_limit: float = MAX_LIMIT_MS) -> dict:
    reg = registry.load()
    agent = registry.resolve(name, reg)
    champ = registry.resolve(opponent, reg) if opponent else registry.get_champion(team, reg)
    reasons: list[str] = []
    res = {"agent": agent["id"], "team": team, "champion": champ["id"] if champ else None, "checks": {}, "reasons": reasons}
    if champ is None:
        reasons.append(f"no champion set for team {team!r} (arena champion set --team {team} <agent>); "
                       "first champion is assigned by decision, not by gate")
        res["pass"] = False
        return res
    if champ["id"] == agent["id"]:
        reasons.append("agent is already the champion")
        res["pass"] = False
        return res
    con = store.connect(db_path)
    games = store.games_between(con, agent["id"], champ["id"], seed_lo, seed_hi, modes=("confirm",))
    # Defence in depth (Auditor 2026-08-19): only mode='confirm' rows are evidence. Report what
    # else sits on those seeds so contamination fails loudly instead of silently counting.
    other = store.games_between(con, agent["id"], champ["id"], seed_lo, seed_hi)
    con.close()
    excluded = len(other) - len(games)
    n = len(games)
    st = ratings.per_agent_stats(games).get(agent["id"], {"crashes": 0, "timeouts": 0, "p99_ms": 0.0, "max_ms": 0.0})
    score = sum(float(g["result0"]) if g["p0"] == agent["id"] else 1.0 - float(g["result0"]) for g in games)
    lo, hi = ratings.wilson(score, n)
    checks = {
        "enough_games": (n >= min_games, f"{n} mode=confirm games on seeds {seed_lo}-{seed_hi} (need {min_games}"
                                          + (f"; {excluded} non-confirm games on these seeds excluded" if excluded else "") + ")"),
        "no_crashes": (st["crashes"] == 0, f"{st['crashes']} crashes"),
        "no_timeouts": (st["timeouts"] == 0, f"{st['timeouts']} timeouts"),
        "p99_under_limit": (st["p99_ms"] <= p99_limit, f"worst p99 {st['p99_ms']:.1f} ms (limit {p99_limit:.0f})"),
        "max_under_limit": (st["max_ms"] <= max_limit, f"worst turn {st['max_ms']:.1f} ms (limit {max_limit:.0f})"),
        "beats_champion": (n > 0 and lo > 0.5, f"win rate {100 * (score / n if n else 0):.1f}% "
                                                f"[Wilson 95% {100 * lo:.1f}-{100 * hi:.1f}] vs {champ['id']} (need lower bound > 50%)"),
    }
    res["checks"] = {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()}
    res["n"] = n
    res["excluded_non_confirm"] = excluded
    res["win_rate"] = score / n if n else None
    res["ci95"] = [lo, hi]
    for k, (ok, d) in checks.items():
        if not ok:
            reasons.append(f"{k}: {d}")
    res["pass"] = all(ok for ok, _ in checks.values())
    return res


def format_result(res: dict) -> str:
    lines = [f"GATE {'PASS' if res['pass'] else 'FAIL'}  {res['agent']}  vs champion {res.get('champion')}  (team {res['team']})"]
    for k, c in res.get("checks", {}).items():
        lines.append(f"  [{'ok' if c['ok'] else 'FAIL'}] {k}: {c['detail']}")
    for r in res["reasons"]:
        if not res.get("checks"):
            lines.append(f"  ! {r}")
    return "\n".join(lines)
