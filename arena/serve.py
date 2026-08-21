"""The Pit dashboard + hub service: live web UI, JSON API, and (with --hub) the continuous ladder.

    uv run python -m arena serve                       # dashboard only, http://127.0.0.1:8765
    uv run python -m arena serve --hub                 # dashboard + continuous ladder matchmaker
    uv run python -m arena serve --hub --host 0.0.0.0  # reachable over the tailnet

Stdlib `http.server` (ThreadingHTTPServer), vanilla JS, no build step. Everything is read on
demand from data/arena/ (sqlite + registry.json + progress/*.json), so the dashboard can also be
started without the hub while a CLI match runs. Modelled on the PTCG arena's dashboard.py /
arena_api.py / arena_worker.py, much simplified.

GET  /api/leaderboard[?engine=&include_retired=1]   BT/Elo table + by-class win rates + h2h + champions
GET  /api/games[?limit&agent&run&flagged=1]         most recent first
GET  /api/agents /api/runs /api/status /api/hub     registry, live+recent runs, engine info, hub state
GET  /api/leaderboard.md                            the markdown export
GET  /api/leases                                    live pause leases (see arena/leases.py)
POST /api/control/pause | resume                    durable pause flag (survives restarts)
     resume is refused with 409 + the lease list while any pause lease is live;
     {"force":true,"actor":"who"} overrides and breaks (releases) those leases, logged.
POST /api/control/lease  {"op":"acquire","owner":"engineer-b","purpose":"expB-303 gates","ttl_s":3600}
                         {"op":"renew","lease_id":"..."} | {"op":"release","lease_id":"...","resume":false}
                         {"op":"list"}
POST /api/control/config      {"workers":14,"games_per_pairing":4,"max_opponents":5}
POST /api/control/bench       {"agent":"v0a","undo":false}      out of the ladder, still rated
POST /api/control/retire      {"agent":"v0a","undo":false}      out of ladder AND rating fit
POST /api/control/champion    {"team":"A","agent":"v1"}
POST /api/control/register    {"name","team","git_ref","path under agents/"}  -> registers BENCHED
No auth: bind 127.0.0.1 or trust the tailnet. Registration is deliberately restricted to
committed files under agents/ at an origin ref (registering schedules code execution here);
see docs/pit.md "Security".
"""
from __future__ import annotations

import base64
import json
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

from . import engine as eng
from . import hub as hubmod
from . import leaderboard, leases, registry, store
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
    for r in rows:  # trim long text fields for the wire
        for k in ("error0", "error1", "harness_error"):
            if r.get(k):
                r[k] = r[k][-300:]
    return rows


def api_agents() -> dict:
    reg = registry.ensure_residents()
    agents = registry.list_agents(reg)
    for a in agents:
        a["state"] = "retired" if a.get("retired") else ("active" if registry.is_active(a) else
                                                         ("ghost" if a["kind"] == "ghost" else "benched"))
    return {"agents": agents, "champions": reg.get("champions", {})}


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
                d["status"] = "stale"   # the process died without finishing
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
            "hub": hubmod.HUB.status(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def api_leases() -> dict:
    return {"leases": leases.active(), "paused": hubmod.is_paused()}


def control(action: str, body: dict) -> dict:
    if action == "pause":
        return {"paused": hubmod.set_paused(True), **api_leases()}
    if action == "resume":
        # raises LeaseHeld -> 409 unless force + actor (see arena/leases.py)
        paused = hubmod.set_paused(False, force=bool(body.get("force")), actor=body.get("actor", ""))
        return {"paused": paused, **api_leases()}
    if action == "lease":
        op = body.get("op", "list")
        if op == "list":
            return api_leases()
        if op == "acquire":
            rec = hubmod.acquire_lease(body.get("owner", ""), body.get("purpose", ""),
                                       ttl_s=int(body.get("ttl_s") or leases.DEFAULT_TTL_S),
                                       pause=body.get("pause", True) is not False)
            return {"lease": rec, "hub": hubmod.HUB.status(), **api_leases()}
        if op == "renew":
            return {"lease": leases.renew(body["lease_id"], body.get("ttl_s")), **api_leases()}
        if op == "release":
            rec = hubmod.release_lease(body["lease_id"], resume=bool(body.get("resume")))
            return {"lease": rec, **api_leases()}
        raise ValueError(f"unknown lease op {op!r} (acquire|renew|release|list)")
    if action == "config":
        return {"config": hubmod.set_config(**body)}
    if action in ("bench", "retire"):
        undo = bool(body.get("undo"))
        if action == "bench":
            a = registry.set_flag(body["agent"], "active", undo)
        else:
            a = registry.set_retired(body["agent"], not undo)
        return {"agent": a}
    if action == "activate":
        return {"agent": registry.set_flag(body["agent"], "active", True)}
    if action == "champion":
        return {"champion": registry.set_champion(body["team"], body["agent"])}
    if action == "register":
        # Locked down per the Auditor (2026-08-19): registering IS scheduling code for execution
        # on this host (the framework execs the file, and the hub anchors new agents within
        # minutes). The API therefore only accepts files under the repo's agents/ tree at a git
        # ref that exists on origin -- i.e. code that went through a commit + push. Raw local
        # paths / inline code require `"allow_unsafe_register": true` in data/arena/hub.json
        # (default off), and every API-registered agent starts BENCHED (explicit activation).
        name, team = body["name"], body.get("team", "resident")
        notes = body.get("notes", "")
        unsafe_ok = bool(hubmod.config().get("allow_unsafe_register"))
        rel = (body.get("path") or "").replace("\\", "/").strip("/")
        if body.get("git_ref"):
            if not rel:
                raise ValueError("register needs path (repo-relative, under agents/)")
            norm = PurePosixPath(rel)
            if norm.parts[:1] != ("agents",) or ".." in norm.parts:
                raise ValueError(f"path must be under the repo's agents/ tree, got {rel!r}")
            a = registry.register_git_ref(body["git_ref"], rel, name, team, notes)
        elif body.get("code_b64") or body.get("path"):
            if not unsafe_ok:
                raise ValueError("only git_ref registration is allowed over the API "
                                 "(set allow_unsafe_register in data/arena/hub.json to override)")
            if body.get("code_b64"):
                with tempfile.TemporaryDirectory() as td:
                    f = Path(td) / f"{name}.py"
                    f.write_bytes(base64.b64decode(body["code_b64"]))
                    a = registry.register_file(f, name, team, notes or "registered via API (inline code)")
            else:
                a = registry.register_file(body["path"], name, team, notes)
        else:
            raise ValueError("register needs git_ref + path (agents/...)")
        a = registry.set_flag(a["id"], "active", False)   # benched until someone activates it
        return {"agent": a}
    raise ValueError(f"unknown control action {action!r}")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>The Pit — Kaggler internal arena</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1117;--panel:#161a23;--line:#262c3a;--fg:#e6e8ee;--dim:#8b93a7;--acc:#7bd88f;--warn:#f3c969;--bad:#ff6b6b;--a:#6cb6ff;--b:#ff9f6c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;gap:14px;align-items:baseline;padding:12px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2;flex-wrap:wrap}
h1{font-size:18px;margin:0;letter-spacing:.08em}h2{font-size:13px;margin:0 0 8px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em}
.pill{padding:2px 8px;border-radius:10px;font-size:12px;border:1px solid var(--line);color:var(--dim)}
.pill.live{border-color:var(--acc);color:var(--acc)}.pill.paused{border-color:var(--warn);color:var(--warn)}.pill.bad{border-color:var(--bad);color:var(--bad)}
main{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 20px}section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto}
section.wide{grid-column:1/-1}table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:4px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}th{color:var(--dim);font-weight:normal;position:sticky;top:0;background:var(--panel)}
.A{color:var(--a)}.B{color:var(--b)}.res{color:var(--dim)}.ok{color:var(--acc)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;background:var(--acc)}
.mono{color:var(--dim);font-size:12px}.champ{font-weight:bold}.small{font-size:11px;color:var(--dim)}
td.h2h{font-size:11px}.id{max-width:240px;overflow:hidden;text-overflow:ellipsis;display:inline-block;vertical-align:bottom}
button{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:3px 10px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}button.warn:hover{border-color:var(--warn)}button.bad:hover{border-color:var(--bad)}
input,select{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:3px 6px;font:inherit;font-size:12px}
input[type=number]{width:56px}form.reg{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:8px}
a{color:var(--a)}footer{padding:8px 20px;color:var(--dim);font-size:12px}
#toast{position:fixed;bottom:14px;right:14px;background:var(--panel);border:1px solid var(--acc);border-radius:6px;padding:8px 14px;display:none;z-index:9}
.linklike{cursor:pointer;text-decoration:underline dotted}
</style></head><body>
<header><h1>THE PIT</h1><span id="hubpill" class="pill">…</span><span id="leasepill" class="pill" style="display:none"></span>
 <button id="btnPause" class="warn">pause</button><button id="btnResume">resume</button>
 <label class="small">workers <input id="cfgWorkers" type="number" min="0" max="32"></label>
 <label class="small">games/pairing <input id="cfgGames" type="number" min="2" max="32" step="2"></label>
 <label class="small">opponents <input id="cfgOpp" type="number" min="1" max="12"></label>
 <button id="btnCfg">apply</button>
 <span id="engine" class="mono"></span><span id="champs" class="mono"></span><span id="clock" class="mono" style="margin-left:auto"></span></header>
<main>
 <section class="wide"><h2>Leaderboard <span class="small">(BT = Bradley-Terry over all stored games, 1000+400·log10 strength; Elo sequential, diagnostic; win% counts ties as half; p99/max = worst game; gates: 0 crash, 0 timeout, p99 ≤ 150 ms — click an agent for detail)</span></h2><div id="lb"></div></section>
 <section><h2>Ladder / running</h2><div id="runs"></div></section>
 <section><h2>Recent games <span class="small">(latest 40; red = flagged)</span> <span id="gamefilter" class="small"></span></h2><div id="games"></div></section>
 <section class="wide" id="detailSec" style="display:none"><h2>Agent detail — <span id="detailName"></span> <button id="btnCloseDetail">close</button></h2><div id="detail"></div></section>
 <section class="wide"><h2>Head-to-head <span class="small">(row vs column: win% [Wilson 95% CI] n)</span></h2><div id="h2h"></div></section>
 <section class="wide"><h2>Agents</h2><div id="agents"></div>
  <form class="reg" id="regForm">
   <b class="small">REGISTER:</b>
   <input id="regName" placeholder="name (e.g. v1)" required>
   <select id="regTeam"><option>A</option><option>B</option><option>resident</option></select>
   <input id="regRef" placeholder="git ref (branch/sha) on origin" size="22" required>
   <input id="regPath" placeholder="repo path under agents/, e.g. agents/team_a/v1.py" size="34" required>
   <button type="submit">register</button>
   <span class="small">registers <b>benched</b> — activate it to enter the ladder</span>
  </form></section>
</main>
<footer>API: <a href="/api/leaderboard">/api/leaderboard</a> · <a href="/api/games?limit=50">/api/games</a> · <a href="/api/agents">/api/agents</a> · <a href="/api/runs">/api/runs</a> · <a href="/api/status">/api/status</a> · <a href="/api/hub">/api/hub</a> · <a href="/api/leaderboard.md">/api/leaderboard.md</a> — POST /api/control/* to script it · refreshes every 3 s</footer>
<div id="toast"></div>
<script>
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>(100*x).toFixed(1);const teamCls=t=>t==='A'?'A':t==='B'?'B':'res';const short=id=>id.length>30?id.slice(0,27)+'…':id;
let detailAgent=null;
async function j(u){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error(u+' '+r.status);return r.json();}
async function post(u,body){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const d=await r.json();if(!r.ok)throw new Error(d.error||r.status);return d;}
function toast(m,bad){const t=$('toast');t.textContent=m;t.style.borderColor=bad?'var(--bad)':'var(--acc)';t.style.display='block';setTimeout(()=>t.style.display='none',3500);}
async function ctl(a,body){try{await post('/api/control/'+a,body);toast(a+' ok');tick();}catch(e){toast(a+' failed: '+e.message,1);}}
function lbTable(lb){const ch=new Set(Object.values(lb.champions||{}));let h='<table><tr><th>#</th><th class="l">agent</th><th>team</th><th>BT</th><th>Elo</th><th>games</th><th>win%</th><th>vs builtin</th><th>vs team A</th><th>vs team B</th><th>vs ghost</th><th>crash</th><th>t/o</th><th>p99 ms</th><th>max ms</th><th>bank</th></tr>';
 lb.table.forEach((r,i)=>{const bc=k=>{const c=r.by_class[k];return c?`${pct(c.rate)} <span class="small">n=${c.n}</span>`:'·'};
  const crash=r.crashes?'bad':'ok',to=r.timeouts?'bad':'ok',p99=r.p99_ms>150?'bad':'ok';
  h+=`<tr><td>${i+1}</td><td class="l ${ch.has(r.agent)?'champ':''}"><span class="id linklike" onclick="openDetail('${esc(r.agent)}')" title="${esc(r.agent)}">${esc(r.agent)}</span>${ch.has(r.agent)?' ★':''}</td><td class="${teamCls(r.team)}">${esc(r.team)}</td><td><b>${r.bt.toFixed(0)}</b></td><td>${r.elo.toFixed(0)}</td><td>${r.games}</td><td>${pct(r.win_rate)}</td><td>${bc('builtin')}</td><td>${bc('team A')}</td><td>${bc('team B')}</td><td>${bc('ghost')}</td><td class="${crash}">${r.crashes}</td><td class="${to}">${r.timeouts}</td><td class="${p99}">${r.p99_ms.toFixed(1)}</td><td>${r.max_ms.toFixed(0)}</td><td>${r.money_mean.toFixed(0)}±${r.money_std.toFixed(0)}</td></tr>`;});
 return h+'</table>';}
function h2hTable(lb){const ids=lb.table.map(r=>r.agent);if(!ids.length)return '<span class="small">no games yet</span>';let h='<table><tr><th class="l"></th>'+ids.map(a=>`<th title="${esc(a)}">${esc(short(a.split('@')[0]))}</th>`).join('')+'</tr>';
 ids.forEach(a=>{h+=`<tr><td class="l" title="${esc(a)}">${esc(short(a))}</td>`;ids.forEach(b=>{if(a===b){h+='<td class="h2h">—</td>';return;}const c=(lb.h2h[a]||{})[b];h+=c?`<td class="h2h ${c.ci_lo>0.5?'ok':c.ci_hi<0.5?'bad':''}">${(100*c.rate).toFixed(0)}% [${(100*c.ci_lo).toFixed(0)}–${(100*c.ci_hi).toFixed(0)}] n=${c.n}</td>`:'<td class="h2h">·</td>';});h+='</tr>';});return h+'</table>';}
function runsBlock(rs,hub){let h='';
 if(hub&&(hub.leases||[]).length){h+='<div class="small warn">pause lease held — resume is refused until it is released or lapses:</div>';
  hub.leases.forEach(l=>{h+=`<div class="small warn">· <b>${esc(l.owner)}</b> — ${esc(l.purpose)} · ${Math.round(l.expires_in_s/60)} min left · held ${Math.round(l.held_for_s/60)} min · <span class="mono">${esc(l.id)}</span> <span class="res">(release: <span class="mono">arena lease release ${esc(l.id)}</span>)</span></div>`;});}
 if(hub){const r=hub.round;h+=`<div class="small">hub: <b class="${hub.paused?'warn':hub.running?'ok':'res'}">${hub.paused?'PAUSED':(hub.running?hub.phase:'not running (dashboard-only)')}</b> · rounds ${hub.rounds_played} · cfg ${hub.config.workers||'auto'}w / ${hub.config.games_per_pairing}g / ${hub.config.max_opponents}opp${hub.last_error?` · <span class="bad">${esc(hub.last_error)}</span>`:''}</div>`;
  if(r)h+=`<div class="small">round ${esc(r.run_id)}: anchor <b>${esc(short(r.anchor))}</b> (${r.anchor_games} ladder games) vs ${r.opponents.map(short).map(esc).join(', ')} · seeds ${esc(r.seeds)}</div>`;}
 if(!rs.live.length)h+='<div class="small">no match playing right now</div>';
 rs.live.forEach(r=>{const p=r.total?r.done/r.total:0;h+=`<div><b>${esc(r.mode)}</b> ${esc(r.run_id)} <span class="small">${esc(r.host)} · ${r.workers} workers · seeds ${esc(r.seeds)}</span><div class="bar"><i style="width:${(100*p).toFixed(1)}%"></i></div><div class="small">${r.done}/${r.total} games (${r.total-r.done} queued) · ${esc(r.agents.map(short).join(', '))}${r.last?` · last: ${esc(short(r.last.p0))} ${r.last.money0.toFixed(0)} – ${r.last.money1.toFixed(0)} ${esc(short(r.last.p1))}${r.last.flag?' <span class=bad>flag</span>':''}`:''}</div></div>`;});
 if(rs.recent.length){h+='<table><tr><th class="l">recent run</th><th>mode</th><th>games</th><th>wall s</th><th class="l">agents</th></tr>';rs.recent.slice(0,8).forEach(r=>{h+=`<tr><td class="l">${esc(r.run_id)} <span class="small ${r.status==='stale'?'warn':''}">${esc(r.status)}</span></td><td>${esc(r.mode)}</td><td>${r.done}/${r.total}</td><td>${r.wall_s??''}</td><td class="l small">${esc(r.agents.map(short).join(', '))}</td></tr>`;});h+='</table>';}
 return h;}
function gamesTable(gs){if(!gs.length)return '<span class="small">none</span>';let h='<table><tr><th class="l">when</th><th class="l">p0</th><th>bank</th><th>bank</th><th class="l">p1</th><th>seed</th><th>mode</th><th>p99 ms</th></tr>';
 gs.forEach(g=>{const flag=g.crash0||g.crash1||g.timeout0||g.timeout1||g.harness_error;const w0=g.result0===1,w1=g.result0===0;
  h+=`<tr class="${flag?'bad':''}"><td class="l small">${esc(g.created_at.slice(5,16).replace('T',' '))}</td><td class="l ${w0?'ok':''}" title="${esc(g.p0)}">${esc(short(g.p0))}</td><td>${g.money0.toFixed(0)}</td><td>${g.money1.toFixed(0)}</td><td class="l ${w1?'ok':''}" title="${esc(g.p1)}">${esc(short(g.p1))}</td><td>${g.seed}</td><td>${esc(g.mode)}</td><td>${Math.max(g.p99_0,g.p99_1).toFixed(1)}</td></tr>`;});return h+'</table>';}
function agentsTable(a){let h='<table><tr><th class="l">id</th><th>team</th><th>kind</th><th>state</th><th>engine</th><th class="l">registered / source</th><th class="l">actions</th></tr>';
 a.agents.forEach(x=>{const st=x.state;const stc=st==='active'?'ok':st==='retired'?'bad':st==='ghost'?'res':'warn';
  let act='';
  if(x.kind!=='ghost'){
   if(st==='active')act+=`<button class="warn" onclick="ctl('bench',{agent:'${esc(x.id)}'})">bench</button> `;
   if(st==='benched')act+=`<button onclick="ctl('activate',{agent:'${esc(x.id)}'})">activate</button> `;
   if(st!=='retired')act+=`<button class="bad" onclick="ctl('retire',{agent:'${esc(x.id)}'})">retire</button> `;
   else act+=`<button onclick="ctl('retire',{agent:'${esc(x.id)}',undo:true})">un-retire</button> `;
   if(x.team==='A'||x.team==='B')act+=`<button onclick="ctl('champion',{team:'${esc(x.team)}',agent:'${esc(x.id)}'})">champ ${esc(x.team)}</button> `;
   act+=`<button onclick="ctl('champion',{team:'org',agent:'${esc(x.id)}'})">champ org</button>`;}
  h+=`<tr><td class="l linklike" onclick="openDetail('${esc(x.id)}')">${esc(x.id)}</td><td class="${teamCls(x.team)}">${esc(x.team)}</td><td>${esc(x.kind)}</td><td class="${stc}">${esc(st)}</td><td>${esc(x.engine)}</td><td class="l small">${esc(x.registered_at)} — ${esc(x.source)}${x.notes?' — '+esc(x.notes):''}${x.ghost?' — '+esc(x.ghost.limitation):''}</td><td class="l">${act}</td></tr>`;});return h+'</table>';}
async function openDetail(id){detailAgent=id;$('detailSec').style.display='';$('detailName').textContent=id;
 try{const [lb,gs]=await Promise.all([j('/api/leaderboard'),j('/api/games?limit=30&agent='+encodeURIComponent(id))]);
  const row=(lb.h2h[id]||{});let h='<table><tr><th class="l">vs</th><th>n</th><th>W-L-T</th><th>win%</th><th>95% CI</th></tr>';
  Object.entries(row).sort((a,b)=>b[1].n-a[1].n).forEach(([b,c])=>{h+=`<tr><td class="l">${esc(b)}</td><td>${c.n}</td><td>${c.wins}-${c.losses}-${c.ties}</td><td class="${c.ci_lo>0.5?'ok':c.ci_hi<0.5?'bad':''}">${pct(c.rate)}</td><td>[${pct(c.ci_lo)}–${pct(c.ci_hi)}]</td></tr>`;});
  h+='</table><h2 style="margin-top:10px">recent games</h2>'+gamesTable(gs);$('detail').innerHTML=h;window.scrollTo({top:$('detailSec').offsetTop-60,behavior:'smooth'});}catch(e){$('detail').innerHTML='<span class=bad>'+esc(e.message)+'</span>';}}
$('btnCloseDetail').onclick=()=>{detailAgent=null;$('detailSec').style.display='none';};
$('btnPause').onclick=()=>ctl('pause');$('btnResume').onclick=()=>ctl('resume');
$('btnCfg').onclick=()=>ctl('config',{workers:+$('cfgWorkers').value,games_per_pairing:+$('cfgGames').value,max_opponents:+$('cfgOpp').value});
$('regForm').onsubmit=e=>{e.preventDefault();const b={name:$('regName').value.trim(),team:$('regTeam').value,
 git_ref:$('regRef').value.trim(),path:$('regPath').value.trim()};
 if(!b.git_ref||!b.path.startsWith('agents/')){toast('need a git ref and a path under agents/',1);return;}
 ctl('register',b);};
let cfgTouched=false;['cfgWorkers','cfgGames','cfgOpp'].forEach(id=>$(id).addEventListener('input',()=>cfgTouched=true));
async function tick(){try{const [lb,rs,ag,st]=await Promise.all([j('/api/leaderboard'),j('/api/runs'),j('/api/agents'),j('/api/status')]);
 const gs=await j('/api/games?limit=40');
 $('lb').innerHTML=lbTable(lb);$('h2h').innerHTML=h2hTable(lb);$('runs').innerHTML=runsBlock(rs,st.hub);$('games').innerHTML=gamesTable(gs);$('agents').innerHTML=agentsTable(ag);
 const hub=st.hub,pill=$('hubpill');
 pill.textContent=hub.paused?'LADDER PAUSED':hub.running?('LADDER '+hub.phase.toUpperCase()):(rs.live.length?'MATCH RUNNING':'IDLE');
 pill.className='pill '+(hub.paused?'paused':(hub.running&&hub.phase==='playing')||rs.live.length?'live':'');
 const lz=hub.leases||[],lp=$('leasepill');lp.style.display=lz.length?'':'none';lp.className='pill warn';
 lp.textContent=lz.length?('LEASED · '+lz.map(l=>`${l.owner}: ${l.purpose} (${Math.round(l.expires_in_s/60)}m)`).join(' · ')):'';
 if(!cfgTouched){$('cfgWorkers').value=hub.config.workers;$('cfgGames').value=hub.config.games_per_pairing;$('cfgOpp').value=hub.config.max_opponents;}
 $('engine').textContent=`engine ${st.engine} · arena ${st.arena_sha} · ${st.n_games} games`;
 $('champs').textContent='champions: '+(Object.entries(lb.champions||{}).map(([t,c])=>`${t}=${short(c)}`).join('  ')||'none');
 $('clock').textContent='updated '+new Date().toLocaleTimeString();
 if(detailAgent)openDetail(detailAgent);}catch(e){const p=$('hubpill');p.textContent='ERROR '+e.message;p.className='pill bad';}}
tick();setInterval(tick,3000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ThePit/2.0"

    def log_message(self, fmt, *args):
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
                return self._json(api_games(int(q.get("limit", 50)), q.get("agent"), q.get("run"),
                                            q.get("flagged") in ("1", "true")))
            if u.path == "/api/agents":
                return self._json(api_agents())
            if u.path == "/api/runs":
                return self._json(api_runs())
            if u.path == "/api/status":
                return self._json(api_status())
            if u.path == "/api/hub":
                return self._json(hubmod.HUB.status())
            if u.path == "/api/leases":
                return self._json(api_leases())
            if u.path == "/api/leaderboard.md":
                lb = api_leaderboard(q.get("engine"), False)
                return self._send(HTTPStatus.OK, leaderboard.render_markdown(lb).encode("utf-8"),
                                  "text/markdown; charset=utf-8")
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # never take the server down for one bad request
            return self._json({"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if u.path.startswith("/api/control/"):
                return self._json(control(u.path.rsplit("/", 1)[1], body))
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except leases.LeaseHeld as exc:
            return self._json({"error": str(exc), "leases": exc.leases}, HTTPStatus.CONFLICT)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            return self._json({"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def serve(host: str = "127.0.0.1", port: int = 8765, verbose: bool = False, hub: bool = False) -> int:
    registry.ensure_residents()
    if hub:
        hubmod.HUB.start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.verbose = verbose
    print(f"The Pit: http://{host}:{port}/  hub={'on' if hub else 'off (dashboard only)'}  "
          f"(api under /api/, Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        hubmod.HUB.stop()
        httpd.server_close()
    return 0
