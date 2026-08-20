"""Match scheduler: paired seeds with side swap, process-parallel, crash-isolated.

Modes
  screen       64 episodes (32 seed pairs x 2 seats) on tuning seeds 0..    -- cheap signal
  confirm      400 episodes (200 pairs) on held-out seeds 1000..            -- accept/reject
  round-robin  every registered agent vs every other, `--n` episodes per pair

(The continuous ladder is NOT here: it lives in arena/hub.py (`arena serve --hub`), the single
implementation, on its own seed band LADDER_SEED_BASE=2000 -- one ladder, one seed authority.)

Every game is one job in a worker pool (multiprocessing Processes owned here, not a
concurrent.futures pool: if a worker dies hard -- os._exit, segfault, OOM -- its job is recorded
as a harness crash and a fresh worker is started, the run continues). Each worker imports the
engine once. A game that exceeds --game-timeout seconds kills its worker the same way.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import multiprocessing as mp
import os
import socket
import time
import uuid
from pathlib import Path

from . import engine as eng
from . import registry, store
from .paths import CONFIRM_N, CONFIRM_SEED_BASE, PROGRESS_DIR, SCREEN_N, SCREEN_SEED_BASE
from .runner import worker_main


def default_workers() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


def game_id(p0: str, p1: str, seed: int, engine: str) -> str:
    return hashlib.sha1(f"{p0}|{p1}|{seed}|{engine}".encode()).hexdigest()[:16]


def parse_seeds(spec: str | None, mode: str, n: int) -> list[int]:
    """`screen` / `confirm` (named sets), `a-b` (inclusive), `a,b,c`, or None (by mode).
    Returns exactly n//2 seeds (each is played on both seats) unless an explicit list is longer/shorter."""
    pairs = max(1, n // 2)
    if spec in (None, "", "auto"):
        spec = "confirm" if mode == "confirm" else "screen"
    if spec == "screen":
        return list(range(SCREEN_SEED_BASE, SCREEN_SEED_BASE + pairs))
    if spec == "confirm":
        return list(range(CONFIRM_SEED_BASE, CONFIRM_SEED_BASE + pairs))
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def default_n(mode: str) -> int:
    return CONFIRM_N if mode == "confirm" else SCREEN_N


def _pair_jobs(a: dict, b: dict, seeds: list[int], steps: int, common: dict) -> list[dict]:
    """Paired seeds with side swap: seed s as (a,b) and as (b,a). Ghost seeds are forced."""
    ghosts = [x for x in (a, b) if x["kind"] == "ghost"]
    if ghosts:
        gs = {int(g["ghost"]["seed"]) for g in ghosts}
        if len(gs) > 1:
            raise ValueError(f"two ghosts with different replay seeds cannot meet: {a['id']} vs {b['id']}")
        for g in ghosts:
            if g["engine"] != eng.engine_version():
                raise ValueError(f"ghost {g['id']} was recorded on engine {g['engine']}, local is {eng.engine_version()}")
        seeds = sorted(gs)
    jobs = []
    engine = eng.engine_version()
    for s in seeds:
        for p0, p1 in ((a, b), (b, a)):
            if p0["id"] == p1["id"] and (p0, p1) == (b, a):
                continue  # self-play: one game per seed is all there is
            jobs.append({**common, "game_id": game_id(p0["id"], p1["id"], s, engine),
                         "p0": p0, "p1": p1, "seed": s, "steps": steps})
    return jobs


def plan_jobs(mode: str, agents: list[dict], seeds: list[int], steps: int, common: dict) -> list[dict]:
    if mode in ("screen", "confirm"):
        if len(agents) < 2:
            raise ValueError("screen/confirm need two agents: arena match A B ...")
        a = agents[0]
        jobs = []
        for b in agents[1:]:
            jobs += _pair_jobs(a, b, seeds, steps, {**common, "focus": a["id"]})
        return jobs
    if mode == "round-robin":
        jobs = []
        for a, b in itertools.combinations(agents, 2):
            jobs += _pair_jobs(a, b, seeds, steps, common)
        return jobs
    raise ValueError(f"unknown mode {mode!r}")


class Pool:
    """Tiny crash-tolerant process pool: jobs in, rows out, dead workers replaced."""

    def __init__(self, workers: int, game_timeout: float):
        self.ctx = mp.get_context("spawn")
        self.job_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.n = workers
        self.game_timeout = game_timeout
        self.procs: dict[int, mp.Process] = {}
        self.busy: dict[int, tuple[dict, float]] = {}
        self.ready: set[int] = set()
        self.killed: set[int] = set()
        self.next_idx = 0

    def _spawn(self):
        idx = self.next_idx
        self.next_idx += 1
        p = self.ctx.Process(target=worker_main, args=(self.job_q, self.result_q, idx), daemon=True)
        p.start()
        self.procs[idx] = p
        return idx

    def run(self, jobs: list[dict], on_row=None) -> list[dict]:
        pending = list(jobs)
        rows: list[dict] = []
        total = len(jobs)
        for _ in range(min(self.n, max(1, total))):
            self._spawn()
        # All jobs go on the shared queue up front; a worker announces ("start", idx, game_id)
        # when it picks one, so a dead worker's in-flight game can be attributed and recorded.
        for j in pending:
            self.job_q.put(j)
        done_ids: set[str] = set()
        by_id = {j["game_id"]: j for j in jobs}
        last_progress = time.monotonic()
        while len(done_ids) < total:
            # dead-worker sweep
            for idx, p in list(self.procs.items()):
                if not p.is_alive():
                    del self.procs[idx]
                    self.ready.discard(idx)
                    if idx in self.busy:
                        job, _ = self.busy.pop(idx)
                        if job["game_id"] not in done_ids:
                            why = (f"game exceeded --game-timeout {self.game_timeout:.0f}s; worker killed"
                                   if idx in self.killed else f"worker {idx} died (exit {p.exitcode}) during game")
                            row = _harness_crash_row(job, why)
                            if idx in self.killed:
                                row["timeout0"] = row["timeout1"] = 1
                            done_ids.add(row["game_id"]); rows.append(row)
                            if on_row:
                                on_row(row)
                    if len(done_ids) < total and len(self.procs) < self.n:
                        self._spawn()
            # game-timeout sweep
            now = time.monotonic()
            for idx, (job, t0) in list(self.busy.items()):
                if now - t0 > self.game_timeout:
                    p = self.procs.get(idx)
                    if p is not None and idx not in self.killed:
                        self.killed.add(idx)
                        p.terminate()
            try:
                kind, idx, payload = self.result_q.get(timeout=1.0)
            except Exception:
                # Nobody is playing and nothing arrives: the remaining jobs were lost with a worker
                # that died between dequeue and its "start" message. Record them and stop.
                idle_all = all(i in self.ready for i in self.procs)
                if not self.busy and idle_all and time.monotonic() - last_progress > 30:
                    for gid, job in by_id.items():
                        if gid not in done_ids:
                            row = _harness_crash_row(job, "job lost: no worker reported it")
                            done_ids.add(gid); rows.append(row)
                            if on_row:
                                on_row(row)
                continue
            last_progress = time.monotonic()
            if kind == "ready":
                self.ready.add(idx)
            elif kind == "start":
                self.busy[idx] = (by_id[payload], time.monotonic())
            elif kind == "row":
                self.busy.pop(idx, None)
                row = payload
                if "p0" not in row:  # worker-level failure wrapper
                    row = _harness_crash_row(by_id[row["game_id"]], row.get("harness_error", "worker failure"))
                if row["game_id"] not in done_ids:
                    done_ids.add(row["game_id"]); rows.append(row)
                    if on_row:
                        on_row(row)
            elif kind == "dead":
                pass  # the liveness sweep handles it
        for _ in range(len(self.procs)):
            self.job_q.put(None)
        for p in self.procs.values():
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        return rows


def _harness_crash_row(job: dict, msg: str) -> dict:
    return {
        "game_id": job["game_id"], "run_id": job.get("run_id", ""), "mode": job.get("mode", ""),
        "p0": job["p0"]["id"], "p1": job["p1"]["id"], "seed": job["seed"], "steps": job["steps"],
        "engine": eng.engine_version(), "arena_sha": job.get("arena_sha", ""), "host": job.get("host", ""),
        "money0": 0.0, "money1": 0.0, "status0": "HARNESS_CRASH", "status1": "HARNESS_CRASH",
        "result0": 0.5, "crash0": 1, "crash1": 1, "timeout0": 0, "timeout1": 0,
        "p50_0": 0.0, "p99_0": 0.0, "max_0": 0.0, "p50_1": 0.0, "p99_1": 0.0, "max_1": 0.0,
        "over_budget0": 0, "over_budget1": 0, "error0": "", "error1": "", "harness_error": msg,
        "steps_played": 0, "wall_s": 0.0, "replay_path": "", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_match(agent_specs: list[str], mode: str = "screen", n: int | None = None, seeds: str | None = None,
              workers: int | None = None, steps: int = 720, keep_replay: str = "crash", db_path: Path | None = None,
              game_timeout: float = 900.0, quiet: bool = False, run_id: str | None = None) -> dict:
    """Run a match and store rows. Returns {run_id, rows, wall_s, jobs}."""
    reg = registry.ensure_residents()
    agents = [registry.resolve(s, reg) for s in agent_specs]
    n = n or default_n(mode)
    workers = workers or default_workers()
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    host = socket.gethostname()
    common = {"run_id": run_id, "mode": mode, "keep_replay": keep_replay, "arena_sha": eng.arena_git_sha(), "host": host}

    if mode == "round-robin" and len(agents) < 2:
        # all registered agents except retired ones and ghosts (a ghost's seed is fixed: name it explicitly)
        agents = [a for a in registry.list_agents(reg) if not a.get("retired") and a["kind"] != "ghost"]
    if mode == "ladder":
        raise ValueError("the ladder runs in the hub only: `arena serve --hub` (see docs/pit.md)")

    seed_list = parse_seeds(seeds, mode, n)
    jobs = plan_jobs(mode, agents, seed_list, steps, common)
    return _execute(jobs, mode, agents, seed_list, workers, db_path, game_timeout, quiet, run_id, host)


def _write_progress(run_id: str, payload: dict) -> None:
    """Publish live run state for `arena serve` (best effort; never fails a run)."""
    try:
        PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS_DIR / f"{run_id}.json.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(PROGRESS_DIR / f"{run_id}.json")
    except Exception:
        pass


def _execute(jobs, mode, agents, seed_list, workers, db_path, game_timeout, quiet, run_id, host) -> dict:
    seed_list = sorted({j["seed"] for j in jobs}) or list(seed_list)   # ghosts force their own seed
    con = store.connect(db_path)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    prog = {"run_id": run_id, "mode": mode, "status": "running", "agents": [a["id"] for a in agents],
            "total": len(jobs), "done": 0, "workers": min(workers, len(jobs)), "host": host, "started_at": started,
            "seeds": f"{seed_list[0]}-{seed_list[-1]}" if seed_list else "", "last": None, "updated_at": started}
    _write_progress(run_id, prog)
    if not quiet:
        print(f"arena run {run_id}: mode={mode} games={len(jobs)} workers={min(workers, len(jobs))} "
              f"engine={eng.engine_version()} arena={eng.arena_git_sha()} host={host}")
        print(f"  agents: {', '.join(a['id'] for a in agents)}")
        print(f"  seeds : {seed_list[0]}..{seed_list[-1]} ({len(seed_list)}) x 2 seats" if seed_list else "  seeds: ghost-forced")
    done = 0

    def on_row(row):
        nonlocal done
        done += 1
        store.upsert_rows(con, [row])
        prog.update(done=done, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    last={"p0": row["p0"], "p1": row["p1"], "seed": row["seed"], "result0": row["result0"],
                          "money0": row["money0"], "money1": row["money1"],
                          "flag": bool(row["crash0"] or row["crash1"] or row["timeout0"] or row["timeout1"] or row["harness_error"])})
        if done % 5 == 0 or done == len(jobs):
            _write_progress(run_id, prog)
        if not quiet and (done % 10 == 0 or done == len(jobs)):
            print(f"  {done}/{len(jobs)} games  ({time.perf_counter() - t0:.0f}s)", flush=True)

    rows = Pool(min(workers, len(jobs)), game_timeout).run(jobs, on_row) if jobs else []
    wall = time.perf_counter() - t0
    store.record_run(con, {
        "run_id": run_id, "mode": mode, "agents": ",".join(a["id"] for a in agents), "n_games": len(rows),
        "seeds": f"{seed_list[0]}-{seed_list[-1]}" if seed_list else "", "engine": eng.engine_version(),
        "arena_sha": eng.arena_git_sha(), "host": host, "workers": workers, "wall_s": round(wall, 1),
        "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    con.close()
    prog.update(status="done", done=len(rows), wall_s=round(wall, 1), finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_progress(run_id, prog)
    return {"run_id": run_id, "rows": rows, "wall_s": wall, "jobs": jobs, "agents": agents, "mode": mode}
