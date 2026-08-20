"""Ratings over stored game rows: Bradley-Terry (MM algorithm), Elo, Wilson CI, head-to-head.

Bradley-Terry is what the competition's final tournament uses, so it is the primary number.
Ties count as half a win for each side throughout (standard, slightly conservative).
"""
from __future__ import annotations

import math
from collections import defaultdict


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (ties entered as half successes)."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def _pairs(games: list[dict]):
    """Yield (winner_score_for_p0, p0, p1) per game; result0 in {1, 0.5, 0}."""
    for g in games:
        yield float(g["result0"]), g["p0"], g["p1"]


def bradley_terry(games: list[dict], iters: int = 500, tol: float = 1e-9, prior: float = 0.5) -> dict[str, float]:
    """MM (Hunter 2004) fit of BT strengths with a small symmetric prior (`prior` half-wins vs a
    virtual reference opponent) so undefeated/winless agents get finite scores.
    Returns log-strength per agent, centred at 0 (so 0 = average)."""
    agents = sorted({a for g in games for a in (g["p0"], g["p1"])})
    if not agents:
        return {}
    wins = defaultdict(float)                     # wins[i] = total (fractional) wins of i
    n_ij = defaultdict(float)                     # games between i and j (symmetric)
    for r0, a, b in _pairs(games):
        if a == b:
            continue
        wins[a] += r0
        wins[b] += 1.0 - r0
        n_ij[(a, b)] += 1.0
        n_ij[(b, a)] += 1.0
    p = {a: 1.0 for a in agents}
    ref = "__ref__"
    p[ref] = 1.0
    for a in agents:                               # prior games vs the reference (keeps p finite)
        wins[a] += prior
        wins[ref] += prior
        n_ij[(a, ref)] += 2 * prior
        n_ij[(ref, a)] += 2 * prior
    opp = defaultdict(list)
    for (a, b), n in n_ij.items():
        opp[a].append((b, n))
    for _ in range(iters):
        new = {}
        for a in list(p):
            denom = sum(n / (p[a] + p[b]) for b, n in opp[a])
            new[a] = wins[a] / denom if denom > 0 else p[a]
        # normalise geometric mean to 1
        gm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {k: v / gm for k, v in new.items()}
        delta = max(abs(new[k] - p[k]) for k in p)
        p = new
        if delta < tol:
            break
    logs = {a: math.log(p[a]) for a in agents}
    mean = sum(logs.values()) / len(logs)
    return {a: v - mean for a, v in logs.items()}


BT_SCALE_CENTRE = 1000.0   # display scale borrowed from the PTCG arena: centre + 400*log10(strength)


def bt_display(bt_log: dict[str, float]) -> dict[str, float]:
    """Log-strength -> Elo-like display rating: 1000 + 400*log10(p/geomean). A 400-point gap is a
    10:1 win-odds ratio, 100 points ~ 64% win rate -- the same intuition as Elo, but order-independent."""
    return {a: BT_SCALE_CENTRE + 400.0 * v / math.log(10.0) for a, v in bt_log.items()}


def elo(games: list[dict], k: float = 20.0, start: float = 1500.0) -> dict[str, float]:
    """Sequential Elo in stored order (created_at). Diagnostic only -- order-dependent."""
    r = defaultdict(lambda: start)
    for r0, a, b in _pairs(games):
        if a == b:
            continue
        ea = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
        r[a] += k * (r0 - ea)
        r[b] += k * ((1.0 - r0) - (1.0 - ea))
    return dict(r)


def head_to_head(games: list[dict]) -> dict[str, dict[str, dict]]:
    """h2h[a][b] = {n, wins, losses, ties, score, rate, ci_lo, ci_hi} from a's point of view."""
    acc: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"n": 0, "wins": 0, "losses": 0, "ties": 0}))
    for r0, a, b in _pairs(games):
        if a == b:
            continue
        for me, you, r in ((a, b, r0), (b, a, 1.0 - r0)):
            c = acc[me][you]
            c["n"] += 1
            c["wins" if r == 1.0 else ("losses" if r == 0.0 else "ties")] += 1
    out: dict[str, dict[str, dict]] = {}
    for a, row in acc.items():
        out[a] = {}
        for b, c in row.items():
            score = c["wins"] + 0.5 * c["ties"]
            lo, hi = wilson(score, c["n"])
            out[a][b] = {**c, "score": score, "rate": score / c["n"], "ci_lo": lo, "ci_hi": hi}
    return out


def per_agent_stats(games: list[dict]) -> dict[str, dict]:
    """Games, score, crashes, timeouts, timing p99 (max over games) and money mean per agent."""
    st: dict[str, dict] = defaultdict(lambda: {"games": 0, "score": 0.0, "crashes": 0, "timeouts": 0,
                                               "p99_ms": 0.0, "max_ms": 0.0, "money": []})
    for g in games:
        for i, me in ((0, g["p0"]), (1, g["p1"])):
            s = st[me]
            s["games"] += 1
            s["score"] += float(g["result0"]) if i == 0 else 1.0 - float(g["result0"])
            s["crashes"] += int(g.get(f"crash{i}") or 0)
            s["timeouts"] += int(g.get(f"timeout{i}") or 0)
            s["p99_ms"] = max(s["p99_ms"], float(g.get(f"p99_{i}") or 0.0))
            s["max_ms"] = max(s["max_ms"], float(g.get(f"max_{i}") or 0.0))
            s["money"].append(float(g.get(f"money{i}") or 0.0))
    for s in st.values():
        m = s.pop("money")
        s["money_mean"] = sum(m) / len(m) if m else 0.0
        s["money_std"] = (sum((x - s["money_mean"]) ** 2 for x in m) / len(m)) ** 0.5 if m else 0.0
        s["win_rate"] = s["score"] / s["games"] if s["games"] else 0.0
    return dict(st)
