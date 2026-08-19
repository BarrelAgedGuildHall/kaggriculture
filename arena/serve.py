"""The Pit dashboard: a zero-dependency local web UI + JSON API over the arena store.

    uv run python -m arena serve                 # http://127.0.0.1:8765
    uv run python -m arena serve --port 9000 --host 0.0.0.0   # (tailnet: see README)

Stdlib `http.server` (ThreadingHTTPServer). Everything is read on demand from data/arena/
(sqlite + registry.json + progress/*.json), so it can be started any time, independent of
whether a match is running; the page polls every 3 s. The leaderboard (BT/Elo fit) is cached for
a couple of seconds so polling stays cheap.

API (GET, JSON):
  /api/leaderboard[?engine=1.32.7&include_retired=1]  table (BT, Elo, games, win%, crashes, timeouts, p99/max,
                                                      by opponent class), h2h matrix, champions, engine, runs
  /api/games[?limit=50&agent=<id>&run=<run_id>&flagged=1]   most recent games first
  /api/agents                                        registry entries + champions
  /api/runs                                          live/recent runs from progress files + the runs table
  /api/status                                        engine, arena sha, store size, live run count
Modelled (much simplified) on the PTCG arena's dashboard.py / arena_api.py.
"""
from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import engine as eng
from . import leaderboard, registry, store
from .paths import ARENA_DIR, DB_PATH, PROGRESS_DIR

_LB_CACHE: dict = {"key": None, "at": 0.0, "data": None}
_LB_LOCK = threading.Lock()
LB_TTL_S = 2.0
STALE_RUN_S = 600.0


def api_leaderboard(engine: str | None, include_retired: bool) -> dict:
    key = (engine, include_retired)
    with _LB_LOCK:
        if _LB_CACHE["key"] == key and time.monotonic() - _LB_CACHE["at"] < LB_TTL_S:
            return _LB_CACHE["data"]
        data = leaderboard.build(engine=engine, include_retired=include_retired)
        _LB_CACHE.update(key=key, at=time.monotonic(), data=data)
        return data


def api_games(limit: int = 50, agent: str | None = None, run: str | None = None, flagged: bool = False) -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = store.connect()
    where, params = [], []
    if agent:
        where.append("(p0 = ? OR p1 = ?)"); params += [agent, agent]
    if run:
        where.append("run_id = ?"); params.append(run)
    if flagged:
        where.append("(crash0 OR crash1 OR timeout0 OR timeout1 OR harness_error != '')")
    sql = "SELECT * FROM games" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, game_id LIMIT ?"
    rows = [dict(r) for r in con.execute(sql, (*params, limit))]
    con.close()
    for r in rows:  # trim the long text fields for the wire
        for k in ("error0", "error1", "harness_error"):
            if r.get(k):
                r[k] = r[k][-300:]
    return rows


def api_agents() -> dict:
    reg = registry.ensure_residents()
    return {"agents": registry.list_agents(reg), "champions": reg.get("champions", {})}


def api_runs() -> dict:
    live, recent = [], []
    now = time.time()
    if PROGRESS_DIR.exists():
        for f in sorted(PROGRESS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            age = now - f.stat().st_mtime
            if d.get("status") == "running" and age > STALE_RUN_S:
                d["status"] = "stale"   # the CLI died without finishing; nothing is running
            d["age_s"] = round(age)
            (live if d["status"] == "running" else recent).append(d)
    table = []
    if DB_PATH.exists():
        con = store.connect()
        table = [dict(r) for r in con.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 30")]
        con.close()
    return {"live": live, "recent": recent[:15], "runs": table}


def api_status() -> dict:
    n_games = 0
    if DB_PATH.exists():
        con = store.connect()
        n_games = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        con.close()
    return {"engine": eng.engine_version(), "engine_sha": eng.engine_sha(), "arena_sha": eng.arena_git_sha(),
            "arena_dir": str(ARENA_DIR), "n_games": n_games, "live_runs": len(api_runs()["live"]),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>The Pit — Kaggler internal arena</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1117;--panel:#161a23;--line:#262c3a;--fg:#e6e8ee;--dim:#8b93a7;--acc:#7bd88f;--warn:#f3c969;--bad:#ff6b6b;--a:#6cb6ff;--b:#ff9f6c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;gap:18px;align-items:baseline;padding:14px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:18px;margin:0;letter-spacing:.08em}h2{font-size:13px;margin:0 0 8px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em}
.pill{padding:2px 8px;border-radius:10px;font-size:12px;border:1px solid var(--line);color:var(--dim)}
.pill.live{border-color:var(--acc);color:var(--acc)}.pill.idle{color:var(--dim)}
main{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 20px}section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto}
section.wide{grid-column:1/-1}table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}th{color:var(--dim);font-weight:normal;position:sticky;top:0;background:var(--panel)}
.A{color:var(--a)}.B{color:var(--b)}.res{color:var(--dim)}.ok{color:var(--acc)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;background:var(--acc)}
.mono{color:var(--dim);font-size:12px}.champ{font-weight:bold}.small{font-size:11px;color:var(--dim)}
td.h2h{font-size:11px}.id{max-width:260px;overflow:hidden;text-overflow:ellipsis;display:inline-block;vertical-align:bottom}
footer{padding:8px 20px;color:var(--dim);font-size:12px}
</style></head><body>
<header><h1>THE PIT</h1><span id="status" class="pill idle">connecting…</span><span id="engine" class="mono"></span><span id="champs" class="mono"></span><span id="clock" class="mono" style="margin-left:auto"></span></header>
<main>
 <section class="wide"><h2>Leaderboard <span class="small">(BT = Bradley-Terry over all stored games, 1000 + 400·log10 strength; Elo sequential K=20, diagnostic; win% counts ties as half; p99/max = worst game; gates: 0 crash, 0 timeout, p99 ≤ 150 ms)</span></h2><div id="lb"></div></section>
 <section><h2>Running / queued</h2><div id="runs"></div></section>
 <section><h2>Recent games <span class="small">(latest 40; red = flagged)</span></h2><div id="games"></div></section>
 <section class="wide"><h2>Head-to-head <span class="small">(row vs column: win% [Wilson 95% CI] n)</span></h2><div id="h2h"></div></section>
 <section class="wide"><h2>Agents</h2><div id="agents"></div></section>
</main>
<footer>JSON API: <a href="/api/leaderboard" style="color:var(--a)">/api/leaderboard</a> · <a href="/api/games?limit=50" style="color:var(--a)">/api/games</a> · <a href="/api/agents" style="color:var(--a)">/api/agents</a> · <a href="/api/runs" style="color:var(--a)">/api/runs</a> · <a href="/api/status" style="color:var(--a)">/api/status</a> — refreshes every 3 s</footer>
<script>
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const pct=x=>(100*x).toFixed(1);const teamCls=t=>t==='A'?'A':t==='B'?'B':'res';
const short=id=>id.length>30?id.slice(0,27)+'…':id;
async function j(u){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error(u+' '+r.status);return r.json();}
function lbTable(lb){const ch=new Set(Object.values(lb.champions||{}));let h='<table><tr><th>#</th><th class="l">agent</th><th>team</th><th>BT</th><th>Elo</th><th>games</th><th>win%</th><th>vs builtin</th><th>vs team A</th><th>vs team B</th><th>vs ghost</th><th>crash</th><th>t/o</th><th>p99 ms</th><th>max ms</th><th>bank</th></tr>';
 lb.table.forEach((r,i)=>{const bc=k=>{const c=r.by_class[k];return c?`${pct(c.rate)} <span class="small">n=${c.n}</span>`:'·'};
  const crash=r.crashes?'bad':'ok',to=r.timeouts?'bad':'ok',p99=r.p99_ms>150?'bad':'ok';
  h+=`<tr><td>${i+1}</td><td class="l ${ch.has(r.agent)?'champ':''}"><span class="id" title="${esc(r.agent)}">${esc(r.agent)}</span>${ch.has(r.agent)?' ★':''}</td><td class="${teamCls(r.team)}">${esc(r.team)}</td><td><b>${r.bt.toFixed(0)}</b></td><td>${r.elo.toFixed(0)}</td><td>${r.games}</td><td>${pct(r.win_rate)}</td><td>${bc('builtin')}</td><td>${bc('team A')}</td><td>${bc('team B')}</td><td>${bc('ghost')}</td><td class="${crash}">${r.crashes}</td><td class="${to}">${r.timeouts}</td><td class="${p99}">${r.p99_ms.toFixed(1)}</td><td>${r.max_ms.toFixed(0)}</td><td>${r.money_mean.toFixed(0)}±${r.money_std.toFixed(0)}</td></tr>`;});
 return h+'</table>';}
function h2hTable(lb){const ids=lb.table.map(r=>r.agent);if(!ids.length)return '<span class="small">no games yet</span>';let h='<table><tr><th class="l"></th>'+ids.map(a=>`<th title="${esc(a)}">${esc(short(a.split('@')[0]))}</th>`).join('')+'</tr>';
 ids.forEach(a=>{h+=`<tr><td class="l" title="${esc(a)}">${esc(short(a))}</td>`;ids.forEach(b=>{if(a===b){h+='<td class="h2h">—</td>';return;}const c=(lb.h2h[a]||{})[b];h+=c?`<td class="h2h ${c.ci_lo>0.5?'ok':c.ci_hi<0.5?'bad':''}">${(100*c.rate).toFixed(0)}% [${(100*c.ci_lo).toFixed(0)}–${(100*c.ci_hi).toFixed(0)}] n=${c.n}</td>`:'<td class="h2h">·</td>';});h+='</tr>';});return h+'</table>';}
function runsBlock(rs){let h='';if(!rs.live.length)h+='<div class="small">idle — no match running</div>';
 rs.live.forEach(r=>{const p=r.total?r.done/r.total:0;h+=`<div><b>${esc(r.mode)}</b> ${esc(r.run_id)} <span class="small">${esc(r.host)} · ${r.workers} workers · seeds ${esc(r.seeds)}</span><div class="bar"><i style="width:${(100*p).toFixed(1)}%"></i></div><div class="small">${r.done}/${r.total} games (${r.total-r.done} queued) · ${esc(r.agents.join(', '))}${r.last?` · last: ${esc(short(r.last.p0))} ${r.last.money0.toFixed(0)} – ${r.last.money1.toFixed(0)} ${esc(short(r.last.p1))}${r.last.flag?' <span class=bad>flag</span>':''}`:''}</div></div>`;});
 if(rs.recent.length){h+='<table><tr><th class="l">recent run</th><th>mode</th><th>games</th><th>wall s</th><th class="l">agents</th></tr>';rs.recent.slice(0,8).forEach(r=>{h+=`<tr><td class="l">${esc(r.run_id)} <span class="small ${r.status==='stale'?'warn':''}">${esc(r.status)}</span></td><td>${esc(r.mode)}</td><td>${r.done}/${r.total}</td><td>${r.wall_s??''}</td><td class="l small">${esc(r.agents.map(short).join(', '))}</td></tr>`;});h+='</table>';}
 return h;}
function gamesTable(gs){if(!gs.length)return '<span class="small">none</span>';let h='<table><tr><th class="l">when</th><th class="l">p0</th><th>bank</th><th>bank</th><th class="l">p1</th><th>seed</th><th>mode</th><th>p99 ms</th></tr>';
 gs.forEach(g=>{const flag=g.crash0||g.crash1||g.timeout0||g.timeout1||g.harness_error;const w0=g.result0===1,w1=g.result0===0;
  h+=`<tr class="${flag?'bad':''}"><td class="l small">${esc(g.created_at.slice(5,16).replace('T',' '))}</td><td class="l ${w0?'ok':''}" title="${esc(g.p0)}">${esc(short(g.p0))}</td><td>${g.money0.toFixed(0)}</td><td>${g.money1.toFixed(0)}</td><td class="l ${w1?'ok':''}" title="${esc(g.p1)}">${esc(short(g.p1))}</td><td>${g.seed}</td><td>${esc(g.mode)}</td><td>${Math.max(g.p99_0,g.p99_1).toFixed(1)}</td></tr>`;});return h+'</table>';}
function agentsTable(a){let h='<table><tr><th class="l">id</th><th>team</th><th>kind</th><th>engine</th><th class="l">registered</th><th class="l">source / notes</th></tr>';
 a.agents.forEach(x=>{h+=`<tr class="${x.retired?'res':''}"><td class="l">${esc(x.id)}${x.retired?' <span class=small>(retired)</span>':''}</td><td class="${teamCls(x.team)}">${esc(x.team)}</td><td>${esc(x.kind)}</td><td>${esc(x.engine)}</td><td class="l small">${esc(x.registered_at)}</td><td class="l small">${esc(x.source)}${x.notes?' — '+esc(x.notes):''}${x.ghost?' — '+esc(x.ghost.limitation):''}</td></tr>`;});return h+'</table>';}
async function tick(){try{const [lb,rs,gs,ag,st]=await Promise.all([j('/api/leaderboard'),j('/api/runs'),j('/api/games?limit=40'),j('/api/agents'),j('/api/status')]);
 $('lb').innerHTML=lbTable(lb);$('h2h').innerHTML=h2hTable(lb);$('runs').innerHTML=runsBlock(rs);$('games').innerHTML=gamesTable(gs);$('agents').innerHTML=agentsTable(ag);
 const s=$('status');s.textContent=rs.live.length?`RUNNING ${rs.live.length}`:'IDLE';s.className='pill '+(rs.live.length?'live':'idle');
 $('engine').textContent=`engine ${st.engine} · arena ${st.arena_sha} · ${st.n_games} games`;
 $('champs').textContent='champions: '+(Object.entries(lb.champions||{}).map(([t,c])=>`${t}=${c}`).join('  ')||'none');
 $('clock').textContent='updated '+new Date().toLocaleTimeString();}catch(e){$('status').textContent='ERROR '+e.message;$('status').className='pill bad';}}
tick();setInterval(tick,3000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ThePit/1.0"

    def log_message(self, fmt, *args):  # quiet by default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = HTTPStatus.OK) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/health":
                return self._json({"ok": True})
            if u.path == "/api/leaderboard":
                return self._json(api_leaderboard(q.get("engine"), q.get("include_retired") in ("1", "true")))
            if u.path == "/api/games":
                return self._json(api_games(int(q.get("limit", 50)), q.get("agent"), q.get("run"), q.get("flagged") in ("1", "true")))
            if u.path == "/api/agents":
                return self._json(api_agents())
            if u.path == "/api/runs":
                return self._json(api_runs())
            if u.path == "/api/status":
                return self._json(api_status())
            if u.path == "/api/leaderboard.md":
                lb = api_leaderboard(q.get("engine"), False)
                return self._send(HTTPStatus.OK, leaderboard.render_markdown(lb).encode("utf-8"), "text/markdown; charset=utf-8")
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # never take the server down for one bad request
            return self._json({"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def serve(host: str = "127.0.0.1", port: int = 8765, verbose: bool = False) -> int:
    registry.ensure_residents()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.verbose = verbose
    print(f"The Pit: http://{host}:{port}/  (api: /api/leaderboard /api/games /api/agents /api/runs /api/status; Ctrl-C to stop)",
          flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
