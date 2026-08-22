"""`python -m arena` command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import engine as eng
from . import registry
from .paths import ARENA_DIR, DB_PATH, RUNS_DIR


def _p(*a, **k):
    print(*a, **k, flush=True)


def cmd_register(args) -> int:
    if args.git_ref:
        if not args.name:
            raise SystemExit("--name is required")
        a = registry.register_git_ref(args.git_ref, args.path, args.name, args.team, args.notes)
    elif args.path in ("random", "starter", "pass"):
        a = registry.register_builtin(args.path)
    else:
        if not args.name:
            raise SystemExit("--name is required for file agents")
        a = registry.register_file(args.path, args.name, args.team, args.notes)
    _p(f"registered {a['id']}  team={a['team']} kind={a['kind']} engine={a['engine']} source={a['source']}")
    return 0


def cmd_register_ghost(args) -> int:
    from .ghost import register_ghost
    a = register_ghost(args.replay, args.seat, args.name, args.notes, skip_validation=args.skip_validation)
    g = a["ghost"]
    v = g.get("validation") or {}
    _p(f"registered {a['id']}  episode={g['episode']} seat={g['seat']} seed={g['seed']} "
       f"recorded_bank={g['recorded_bank']:.0f} replayed_bank={v.get('replayed_bank', float('nan')):.0f} "
       f"drift={100 * v.get('drift', 0):.2f}%")
    _p(f"  limitation: {g['limitation']}")
    return 0


def cmd_list(args) -> int:
    reg = registry.ensure_residents()
    _p(f"{'id':<44} {'team':<9} {'kind':<8} {'state':<8} {'engine':<8} {'registered':<20} source")
    for a in registry.list_agents(reg):
        state = "retired" if a.get("retired") else ("active" if registry.is_active(a) else "benched")
        _p(f"{a['id']:<44} {a['team']:<9} {a['kind']:<8} {state:<8} {a['engine']:<8} {a['registered_at']:<20} {a['source']}")
    if reg["champions"]:
        _p("champions: " + ", ".join(f"{t}={c}" for t, c in sorted(reg['champions'].items())))
    return 0


def _summarize_match(res: dict) -> None:
    from . import ratings
    rows = res["rows"]
    agents = res["agents"]
    if not rows:
        _p("no games played")
        return
    h2h = ratings.head_to_head(rows)
    _p("")
    _p(f"{'matchup':<70} {'n':>4} {'win%':>6} {'95% CI':>13} {'crash':>5} {'t/o':>4} {'p99ms':>8} {'bank A':>9} {'bank B':>9}")
    focus_ids = [agents[0]["id"]] if res["mode"] in ("screen", "confirm") else sorted({g["p0"] for g in rows})
    printed = set()
    for me in focus_ids:
        for you, c in sorted(h2h.get(me, {}).items()):
            key = tuple(sorted((me, you)))
            if key in printed and res["mode"] not in ("screen", "confirm"):
                continue
            printed.add(key)
            sub = [g for g in rows if {g["p0"], g["p1"]} == {me, you}]
            ss = ratings.per_agent_stats(sub)
            m = ss.get(me, {}); y = ss.get(you, {})
            _p(f"{me + ' vs ' + you:<70} {c['n']:>4} {100 * c['rate']:>5.1f}% {100 * c['ci_lo']:>5.1f}-{100 * c['ci_hi']:<5.1f}% "
               f"{m.get('crashes', 0):>5} {m.get('timeouts', 0):>4} {m.get('p99_ms', 0):>8.2f} "
               f"{m.get('money_mean', 0):>9.0f} {y.get('money_mean', 0):>9.0f}")
    bad = [g for g in rows if g["crash0"] or g["crash1"] or g["timeout0"] or g["timeout1"] or g["harness_error"]]
    for g in bad[:5]:
        _p(f"  ! game {g['game_id']} seed={g['seed']} p0={g['p0']} p1={g['p1']} status={g['status0']}/{g['status1']} "
           f"crash={g['crash0']}/{g['crash1']} timeout={g['timeout0']}/{g['timeout1']} "
           f"{(g['harness_error'] or g['error0'] or g['error1'])[-160:]!r}" + (f" replay={g['replay_path']}" if g["replay_path"] else ""))
    if len(bad) > 5:
        _p(f"  ! ... {len(bad) - 5} more flagged games (see the store / replays)")
    _p(f"\nrun {res['run_id']}: {len(rows)} games in {res['wall_s']:.0f}s wall "
       f"({res['wall_s'] / max(1, len(rows)):.1f} s/game) -> {DB_PATH}")


def cmd_match(args) -> int:
    if args.node:
        from .remote import run_remote
        return run_remote(args)
    from .scheduler import run_match
    res = run_match(args.agents, mode=args.mode, n=args.n, seeds=args.seeds, workers=args.workers, steps=args.steps,
                    keep_replay=args.keep_replays, db_path=Path(args.db) if args.db else None,
                    game_timeout=args.game_timeout, quiet=args.quiet, run_id=args.run_id)
    res["db_path"] = args.db
    _summarize_match(res)
    if args.json:
        Path(args.json).write_text(json.dumps({k: v for k, v in res.items() if k != "jobs"}, indent=1, default=str), encoding="utf-8")
    return 0


def cmd_leaderboard(args) -> int:
    from . import leaderboard
    lb = leaderboard.build(Path(args.db) if args.db else None, engine=args.engine, include_retired=args.include_retired)
    md = leaderboard.render_markdown(lb)
    if not args.quiet:
        _p(md)
    w = leaderboard.write(lb, Path(args.publish) if args.publish else None)
    _p(f"wrote {w['md']} and {w['json']}")
    if w["published"]:
        _p(f"published {w['published']}")
    elif w["publish_error"]:
        _p(f"publish to workshop failed: {w['publish_error']} -- copy {w['md']} by hand")
    return 0


def cmd_gate(args) -> int:
    from . import gates
    res = gates.evaluate(args.name, team=args.team, db_path=Path(args.db) if args.db else None, opponent=args.vs,
                         seed_lo=args.seed_lo, seed_hi=args.seed_hi, min_games=args.min_games,
                         p99_limit=args.p99_ms, max_limit=args.max_ms)
    _p(gates.format_result(res))
    return 0 if res["pass"] else 1


def cmd_champion(args) -> int:
    if args.action == "set":
        a = registry.set_champion(args.team, args.agent)
        _p(f"champion[{args.team}] = {a['id']}")
    else:
        reg = registry.load()
        if args.team == "all":
            for t, c in sorted(reg["champions"].items()):
                _p(f"champion[{t}] = {c}")
        else:
            c = registry.get_champion(args.team, reg)
            _p(f"champion[{args.team}] = {c['id'] if c else None}")
    return 0


def cmd_bench(args) -> int:
    a = registry.set_flag(args.agent, "active", args.undo)
    _p(f"{a['id']} active={a.get('active', True)}")
    return 0


def cmd_retire(args) -> int:
    a = registry.set_retired(args.agent, not args.undo)
    _p(f"{a['id']} retired={a.get('retired', False)}")
    return 0


def cmd_import(args) -> int:
    from . import store
    con = store.connect(Path(args.db) if args.db else None)
    total = 0
    for f in args.files:
        n = store.import_db(con, Path(f))
        _p(f"imported {n} games from {f}")
        total += n
    con.close()
    return 0


def cmd_export(args) -> int:
    from . import store
    con = store.connect(Path(args.db) if args.db else None)
    n = store.export_csv(con, Path(args.out))
    con.close()
    _p(f"exported {n} games to {args.out}")
    return 0


def cmd_serve(args) -> int:
    from .serve import serve
    return serve(args.host, args.port, args.verbose, hub=args.hub)


def _hub_call(url: str, path: str, body: dict | None = None) -> dict:
    """Talk to a remote hub's JSON API (so a desktop operator can lease the Pit host)."""
    import json as _json
    import urllib.error
    import urllib.request
    u = url.rstrip("/") + path
    req = urllib.request.Request(u, method="POST" if body is not None else "GET",
                                 data=_json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return _json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = _json.loads(e.read() or b"{}").get("error", str(e))
        raise SystemExit(f"{u} -> {e.code}: {detail}")


def _print_leases(leaseset: list[dict]) -> None:
    if not leaseset:
        _p("no pause lease held (the ladder may be resumed)")
        return
    _p(f"{'lease id':<40} {'owner':<14} {'left':>6} {'held':>6}  purpose")
    for l in leaseset:
        _p(f"{l['id']:<40} {l['owner']:<14} {l['expires_in_s'] // 60:>5}m {l['held_for_s'] // 60:>5}m  {l['purpose']}")


def cmd_lease(args) -> int:
    """Pause leases: mutual exclusion between a manual run and the hub ladder (arena/leases.py)."""
    from . import leases
    op = args.op
    if op == "acquire" and args.ttl is None:
        args.ttl = leases.DEFAULT_TTL_S
    if args.hub:                                   # remote hub over HTTP
        if op == "list":
            out = _hub_call(args.hub, "/api/leases")
        else:
            body = {"op": op}
            if op == "acquire":
                body.update(owner=args.owner, purpose=args.purpose, ttl_s=args.ttl, pause=not args.no_pause)
            elif op == "renew":
                body.update(lease_id=args.lease_id, ttl_s=args.ttl)
            else:
                body.update(lease_id=args.lease_id, resume=args.resume)
            out = _hub_call(args.hub, "/api/control/lease", body)
        if out.get("lease"):
            _p(f"{op}: {out['lease']['id']}")
        if out.get("lease_state_error"):
            _p(f"! LEASE STATE UNREADABLE on the hub -- treated as HELD: {out['lease_state_error']}")
        _print_leases(out.get("leases", []))
        _p(f"ladder paused: {out.get('paused')}")
        return 0
    from . import hub as hubmod
    if op == "acquire":
        rec = hubmod.acquire_lease(args.owner, args.purpose, ttl_s=args.ttl, pause=not args.no_pause)
        _p(f"lease {rec['id']} held by {rec['owner']} for {rec['ttl_s'] // 60} min: {rec['purpose']}")
        _p("the ladder will not resume until it is released or lapses; renew with "
           f"`arena lease renew {rec['id']}`")
        st = hubmod.HUB.status()
        _p(f"ladder paused: {hubmod.is_paused()} (hub phase {st['phase']!r}; wait for 'paused' before measuring)")
    elif op == "renew":
        rec = leases.renew(args.lease_id, args.ttl)
        _p(f"lease {rec['id']} renewed ({rec['expires_in_s'] // 60} min left, renewals {rec['renewals']})")
    elif op == "release":
        rec = hubmod.release_lease(args.lease_id, resume=args.resume)
        _p(f"lease {rec['id']} released after {rec['held_for_s'] // 60} min; ladder paused: {hubmod.is_paused()}")
    else:
        st = leases.status()
        if st["lease_state_error"]:
            _p(f"! LEASE STATE UNREADABLE -- treated as HELD: {st['lease_state_error']}")
        else:
            _print_leases(st["leases"])
        _p(f"ladder paused: {hubmod.is_paused()}")
    return 0


def cmd_info(args) -> int:
    _p(f"arena dir : {ARENA_DIR}")
    _p(f"store     : {DB_PATH} ({'exists' if DB_PATH.exists() else 'missing'})")
    _p(f"runs dir  : {RUNS_DIR}")
    _p(f"engine    : kaggle-environments {eng.engine_version()} (kaggriculture.py sha {eng.engine_sha()})")
    _p(f"arena sha : {eng.arena_git_sha()} (from {eng.arena_sha_source()})")
    _p(f"arena code: {eng.arena_code_hash()}  <- compare with the hub's /api/status to see if a deploy is live")
    from .scheduler import default_workers
    _p(f"workers   : {default_workers()} (cpu_count-2)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena", description="Kaggler internal arena -- see README 'Arena'.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("register", help="register an agent file/dir (snapshot) or a built-in")
    s.add_argument("path", help="path to agent .py (or dir with main.py), or random|starter|pass; with --git-ref: repo-relative path at that ref")
    s.add_argument("--git-ref", help="register the file as it is at this branch/tag/sha (fetched from origin)")
    s.add_argument("--name", help="agent name (id becomes name@sha)")
    s.add_argument("--team", default="resident", choices=registry.TEAMS)
    s.add_argument("--notes", default="")
    s.set_defaults(fn=cmd_register)

    s = sub.add_parser("register-ghost", help="register a replay-ghost resident from an official replay")
    s.add_argument("replay", help="path, episode id, or file in data/replays/official/")
    s.add_argument("--seat", type=int, required=True, choices=(0, 1))
    s.add_argument("--name")
    s.add_argument("--notes", default="")
    s.add_argument("--skip-validation", action="store_true")
    s.set_defaults(fn=cmd_register_ghost)

    s = sub.add_parser("list", help="list registered agents and champions")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("match", help="run games: screen|confirm A vs B [C...], or round-robin (ladder = `arena serve --hub`)")
    s.add_argument("agents", nargs="*", help="agent names or name@sha (round-robin: empty = all registered)")
    s.add_argument("--mode", default="screen", choices=("screen", "confirm", "round-robin"))
    s.add_argument("--n", type=int, help="episodes (per opponent pair); default 64 screen / 400 confirm")
    s.add_argument("--seeds", help="screen | confirm | a-b | a,b,c (default by mode; each seed is played on both seats)")
    s.add_argument("--workers", type=int, help="default cpu_count-2")
    s.add_argument("--steps", type=int, default=720)
    s.add_argument("--keep-replays", default="crash", choices=("none", "crash", "loss", "all"),
                   help="crash: crashes/timeouts; loss: also games the first agent lost")
    s.add_argument("--game-timeout", type=float, default=900.0, help="seconds before a game's worker is killed")
    s.add_argument("--db", help="write rows to this sqlite instead of data/arena/arena.sqlite")
    s.add_argument("--run-id")
    s.add_argument("--json", help="also dump the run summary JSON here")
    s.add_argument("--node", help="run on a remote node (ssh alias/host) and import the results back")
    s.add_argument("--remote-workers", type=int, help="workers on the node (default: its cpu_count-2)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("leaderboard", help="BT + Elo + head-to-head -> data/arena/leaderboard.md/json (+ workshop)")
    s.add_argument("--db")
    s.add_argument("--engine", help="only games from this engine version")
    s.add_argument("--publish", help="also write the Markdown here")
    s.add_argument("--include-retired", action="store_true")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_leaderboard)

    s = sub.add_parser("gate", help="promotion gate for an agent vs the current champion over the confirm set")
    s.add_argument("name")
    s.add_argument("--team", default="org", choices=("A", "B", "org"))
    s.add_argument("--vs", help="compare against this agent instead of the team champion")
    s.add_argument("--min-games", type=int, default=400)
    s.add_argument("--seed-lo", type=int, default=1000)
    s.add_argument("--seed-hi", type=int, default=1199)
    s.add_argument("--p99-ms", type=float, default=150.0)
    s.add_argument("--max-ms", type=float, default=400.0)
    s.add_argument("--db")
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("champion", help="champion get [--team T] | champion set <agent> --team T")
    cs = s.add_subparsers(dest="action", required=True)
    g = cs.add_parser("get")
    g.add_argument("--team", default="all", help="A | B | org | all")
    g.set_defaults(fn=cmd_champion)
    g = cs.add_parser("set")
    g.add_argument("agent")
    g.add_argument("--team", default="org", choices=("A", "B", "org"))
    g.set_defaults(fn=cmd_champion)

    s = sub.add_parser("bench", help="take an agent out of the hub ladder (still rated); --undo to re-activate")
    s.add_argument("agent")
    s.add_argument("--undo", action="store_true")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("retire", help="exclude an agent from the leaderboard fit (rows are kept); --undo to restore")
    s.add_argument("agent")
    s.add_argument("--undo", action="store_true")
    s.set_defaults(fn=cmd_retire)

    s = sub.add_parser("import", help="merge games from other arena sqlite/CSV files (by game_id)")
    s.add_argument("files", nargs="+")
    s.add_argument("--db")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("export", help="export all games to CSV")
    s.add_argument("--out", default=str(ARENA_DIR / "games.csv"))
    s.add_argument("--db")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("serve", help="The Pit dashboard + JSON API (+ --hub: the continuous ladder) on :8765")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the tailnet")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--hub", action="store_true", help="also run the continuous ladder matchmaker")
    s.add_argument("--verbose", action="store_true", help="log every request")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("lease", help="pause leases: hold the ladder while a manual run measures")
    ls = s.add_subparsers(dest="op", required=True)
    g = ls.add_parser("acquire", help="pause the ladder and hold it under a named lease")
    g.add_argument("--owner", required=True, help="who is running it, e.g. engineer-b")
    g.add_argument("--purpose", required=True, help="what is measured, e.g. 'expB-303 gate battery'")
    g.add_argument("--ttl", type=int, default=None, help=f"seconds (default {3600}, max {6 * 3600}); renew to extend")
    g.add_argument("--no-pause", action="store_true", help="take the lease without pausing (already paused)")
    g.add_argument("--hub", help="operate on a remote hub, e.g. http://HOST:8765")
    g.set_defaults(fn=cmd_lease)
    g = ls.add_parser("list", help="show live leases")
    g.add_argument("--hub")
    g.set_defaults(fn=cmd_lease)
    g = ls.add_parser("renew", help="extend a live lease from now (nothing auto-renews)")
    g.add_argument("lease_id")
    g.add_argument("--ttl", type=int, default=None)
    g.add_argument("--hub")
    g.set_defaults(fn=cmd_lease)
    g = ls.add_parser("release", help="release a lease; --resume also restarts the ladder if none remain")
    g.add_argument("lease_id")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--hub")
    g.set_defaults(fn=cmd_lease)

    s = sub.add_parser("info", help="paths, engine, workers")
    s.set_defaults(fn=cmd_info)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
