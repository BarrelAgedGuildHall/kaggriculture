"""Render the internal leaderboard (Markdown + JSON) from everything in the store."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import engine as eng
from . import ratings, registry, store
from .paths import LEADERBOARD_JSON, LEADERBOARD_MD, WORKSHOP_LEADERBOARD


def build(db_path: Path | None = None, engine: str | None = None, include_retired: bool = False) -> dict:
    con = store.connect(db_path)
    games = store.all_games(con, "engine = ?", (engine,)) if engine else store.all_games(con)
    runs = [dict(r) for r in con.execute("SELECT * FROM runs ORDER BY started_at")]
    con.close()
    reg = registry.load()
    retired = {a["id"] for a in reg["agents"].values() if a.get("retired")}
    if retired and not include_retired:
        games = [g for g in games if g["p0"] not in retired and g["p1"] not in retired]
    bt = ratings.bradley_terry(games)
    btd = ratings.bt_display(bt)
    el = ratings.elo(games)
    h2h = ratings.head_to_head(games)
    st = ratings.per_agent_stats(games)
    agents = sorted(st, key=lambda a: -bt.get(a, 0.0))

    def _cls(aid: str) -> str:
        m = reg["agents"].get(aid, {})
        if m.get("kind") == "ghost":
            return "ghost"
        if m.get("kind") == "builtin":
            return "builtin"
        return f"team {m.get('team', '?')}" if m.get("team") in ("A", "B") else "resident"

    by_class: dict[str, dict[str, dict]] = {}
    for g in games:
        for i, me, you in ((0, g["p0"], g["p1"]), (1, g["p1"], g["p0"])):
            if me == you:
                continue
            c = by_class.setdefault(me, {}).setdefault(_cls(you), {"n": 0, "score": 0.0})
            c["n"] += 1
            c["score"] += float(g["result0"]) if i == 0 else 1.0 - float(g["result0"])
    rows = []
    for a in agents:
        meta = reg["agents"].get(a, {})
        s = st[a]
        classes = {k: {"n": v["n"], "rate": round(v["score"] / v["n"], 4), "ci": [round(x, 4) for x in ratings.wilson(v["score"], v["n"])]}
                   for k, v in sorted(by_class.get(a, {}).items())}
        rows.append({
            "by_class": classes, "retired": bool(meta.get("retired")), "notes": meta.get("notes", ""),
            "agent": a, "team": meta.get("team", "?"), "kind": meta.get("kind", "?"),
            "bt": round(btd.get(a, 1000.0), 1), "bt_log": round(bt.get(a, 0.0), 3), "elo": round(el.get(a, 1500.0), 1),
            "games": s["games"], "win_rate": round(s["win_rate"], 4),
            "crashes": s["crashes"], "timeouts": s["timeouts"],
            "p99_ms": round(s["p99_ms"], 2), "max_ms": round(s["max_ms"], 2),
            "money_mean": round(s["money_mean"], 0), "money_std": round(s["money_std"], 0),
        })
    engines = sorted({g["engine"] for g in games})
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine_local": eng.engine_version(), "engines_in_store": engines, "arena_sha": eng.arena_git_sha(),
        "n_games": len(games), "n_runs": len(runs),
        "champions": reg.get("champions", {}),
        "table": rows,
        "h2h": {a: {b: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for b, c in row.items()}
                for a, row in h2h.items()},
        "runs": runs[-20:],
    }


def _short(agent_id: str) -> str:
    return agent_id if len(agent_id) <= 28 else agent_id[:25] + "..."


def render_markdown(lb: dict) -> str:
    out = []
    out.append("# Internal arena leaderboard")
    out.append("")
    out.append(f"_Generated {lb['generated_at']} · engine {lb['engine_local']} (store: {', '.join(lb['engines_in_store']) or '-'}) "
               f"· arena {lb['arena_sha']} · {lb['n_games']} games in {lb['n_runs']} runs._")
    out.append("")
    champs = lb.get("champions") or {}
    out.append("**Champions:** " + (", ".join(f"{t}: `{c}`" for t, c in sorted(champs.items())) or "none set") + "")
    out.append("")
    out.append("**BT** = Bradley-Terry maximum-likelihood rating over all stored games (order-independent; the "
               "final Kaggle tournament is BT), shown as 1000 + 400·log10(strength/geomean) so 100 points ≈ 64% "
               "win odds. **Elo** = sequential K=20, diagnostic only. Win% counts ties as half. Timing = worst "
               "per-game p99/max over all games (framework timer). Gates: 0 crashes, 0 timeouts, p99 ≤ 150 ms, "
               "Wilson lower bound > 50% vs champion on the confirm set. Retired agents are excluded.")
    out.append("")
    out.append("| # | agent | team | BT | Elo | games | win% | crashes | timeouts | p99 ms | max ms | bank mean±std |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(lb["table"], 1):
        out.append(f"| {i} | `{r['agent']}` | {r['team']} | {r['bt']:.0f} | {r['elo']:.0f} | {r['games']} | "
                   f"{100 * r['win_rate']:.1f} | {r['crashes']} | {r['timeouts']} | {r['p99_ms']:.1f} | {r['max_ms']:.1f} | "
                   f"{r['money_mean']:.0f}±{r['money_std']:.0f} |")
    out.append("")
    agents = [r["agent"] for r in lb["table"]]
    if agents:
        out.append("## Head-to-head (row vs column: win% [Wilson 95% CI] n)")
        out.append("")
        out.append("| | " + " | ".join(f"`{_short(a)}`" for a in agents) + " |")
        out.append("|---|" + "---|" * len(agents))
        for a in agents:
            cells = []
            for b in agents:
                if a == b:
                    cells.append("—")
                    continue
                c = lb["h2h"].get(a, {}).get(b)
                cells.append(f"{100 * c['rate']:.0f}% [{100 * c['ci_lo']:.0f}–{100 * c['ci_hi']:.0f}] n={c['n']}" if c else "·")
            out.append(f"| `{_short(a)}` | " + " | ".join(cells) + " |")
        out.append("")
    if lb.get("runs"):
        out.append("## Recent runs")
        out.append("")
        out.append("| run | mode | games | seeds | host | wall s | agents |")
        out.append("|---|---|---|---|---|---|---|")
        for r in lb["runs"][-10:]:
            out.append(f"| {r['run_id']} | {r['mode']} | {r['n_games']} | {r['seeds']} | {r['host']} | {r['wall_s']} | {r['agents']} |")
        out.append("")
    return "\n".join(out)


def write(lb: dict, publish: Path | None = None) -> dict:
    """Write JSON + MD under data/arena and try the workshop path (or `publish`). Returns paths."""
    LEADERBOARD_MD.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(lb)
    LEADERBOARD_MD.write_text(md, encoding="utf-8")
    LEADERBOARD_JSON.write_text(json.dumps(lb, indent=2), encoding="utf-8")
    written = {"md": str(LEADERBOARD_MD), "json": str(LEADERBOARD_JSON), "published": None, "publish_error": None}
    target = publish or (WORKSHOP_LEADERBOARD if WORKSHOP_LEADERBOARD.parent.exists() else None)
    if target is not None:
        try:
            Path(target).write_text(md, encoding="utf-8")
            written["published"] = str(target)
        except Exception as exc:  # e.g. the compliance hook refuses the write
            written["publish_error"] = repr(exc)
    return written
