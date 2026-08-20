"""Play one arena game inside a worker process and return a flat result row.

File agents run behind `isolation.SubprocessAgent` -- each in its own interpreter, exactly
one agent per process, like Kaggle's sandbox (see isolation.py for the 2026-08-19
sys.modules-collision bug this prevents). The framework's `Agent` wrapper still catches
exceptions (status ERROR) and measures per-turn wall time (`duration`), which is what we
store; a subprocess agent's duration now includes its (small) JSON-IPC overhead.
"""
from __future__ import annotations

import json
import statistics
import time
import traceback
from pathlib import Path

from . import engine as eng
from .paths import AGENTS_DIR, ENV_NAME, REPLAYS_DIR

BAD_STATUSES = ("ERROR", "TIMEOUT", "INVALID")
ACT_TIMEOUT_S = 1.0   # the competition's per-turn budget; any turn above it is flagged


def _agent_for(entry: dict):
    kind = entry["kind"]
    if kind == "builtin":
        return entry["entry"]
    if kind == "file":
        from .isolation import SubprocessAgent
        return SubprocessAgent(AGENTS_DIR / entry["entry"], name=entry["id"])
    if kind == "ghost":
        from .ghost import load_ghost
        return load_ghost(entry)
    raise ValueError(f"unknown agent kind {kind!r}")


def _pct(ds: list[float], q: float) -> float:
    if not ds:
        return 0.0
    return ds[min(len(ds) - 1, int(len(ds) * q))]


def _timing(durs: list[float]) -> dict:
    ds = sorted(durs)
    return {
        "n_turns": len(ds),
        "p50_ms": round(1000 * _pct(ds, 0.50), 3),
        "p99_ms": round(1000 * _pct(ds, 0.99), 3),
        "max_ms": round(1000 * ds[-1], 3) if ds else 0.0,
        "mean_ms": round(1000 * statistics.fmean(ds), 3) if ds else 0.0,
        "over_budget": sum(1 for d in ds if d > ACT_TIMEOUT_S),
    }


def play_game(job: dict) -> dict:
    """job: game_id, p0, p1 (registry entries), seed, steps, keep_replay ('none'|'crash'|'loss'),
    focus (agent id whose loss triggers a kept replay), mode, run_id, configuration (optional)."""
    t0 = time.perf_counter()
    row = {
        "game_id": job["game_id"], "run_id": job.get("run_id", ""), "mode": job.get("mode", ""),
        "p0": job["p0"]["id"], "p1": job["p1"]["id"], "seed": int(job["seed"]), "steps": int(job["steps"]),
        "engine": eng.engine_version(), "arena_sha": job.get("arena_sha", ""), "host": job.get("host", ""),
        "money0": 0.0, "money1": 0.0, "status0": "HARNESS_ERROR", "status1": "HARNESS_ERROR",
        "result0": 0.5, "crash0": 0, "crash1": 0, "timeout0": 0, "timeout1": 0,
        "p50_0": 0.0, "p99_0": 0.0, "max_0": 0.0, "p50_1": 0.0, "p99_1": 0.0, "max_1": 0.0,
        "over_budget0": 0, "over_budget1": 0, "error0": "", "error1": "", "harness_error": "",
        "steps_played": 0, "wall_s": 0.0, "replay_path": "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    env = None
    agents: list = []
    try:
        ke = eng.import_engine()
        cfg = dict(job.get("configuration") or {})
        cfg.update({"episodeSteps": int(job["steps"]), "seed": int(job["seed"])})
        env = ke.make(ENV_NAME, configuration=cfg, debug=False)
        agents = [_agent_for(job["p0"]), _agent_for(job["p1"])]
        env.run(agents)
        final = env.steps[-1]
        for i in (0, 1):
            row[f"money{i}"] = float(final[i].observation.farms[i]["money"])
            row[f"status{i}"] = str(final[i].status)
            durs = [lg[i]["duration"] for lg in env.logs if len(lg) > i and isinstance(lg[i], dict) and "duration" in lg[i]]
            t = _timing(durs)
            row[f"p50_{i}"], row[f"p99_{i}"], row[f"max_{i}"], row[f"over_budget{i}"] = t["p50_ms"], t["p99_ms"], t["max_ms"], t["over_budget"]
            errs = [lg[i].get("stderr", "") for lg in env.logs if len(lg) > i and isinstance(lg[i], dict)]
            tb = next((e for e in errs if e and "Traceback" in e), "")
            # Any ERROR/INVALID status or traceback counts as a crash; TIMEOUT status or a turn
            # over the 1 s budget counts as a timeout (locally the framework only times out after
            # the 60 s overage bank is spent, so the budget flag is the one that bites).
            any_bad = any(str(s[i].status) in BAD_STATUSES for s in env.steps)
            row[f"crash{i}"] = int(bool(tb) or row[f"status{i}"] in ("ERROR", "INVALID") or (any_bad and row[f"status{i}"] != "TIMEOUT"))
            row[f"timeout{i}"] = int(row[f"status{i}"] == "TIMEOUT" or t["over_budget"] > 0)
            row[f"error{i}"] = tb[-600:]
        row["steps_played"] = len(env.steps)
        bad0 = row["crash0"] or row["timeout0"] or row["status0"] != "DONE"
        bad1 = row["crash1"] or row["timeout1"] or row["status1"] != "DONE"
        if bad0 and not bad1:
            row["result0"] = 0.0
        elif bad1 and not bad0:
            row["result0"] = 1.0
        elif bad0 and bad1:
            row["result0"] = 0.5
        else:
            row["result0"] = 1.0 if row["money0"] > row["money1"] else (0.0 if row["money0"] < row["money1"] else 0.5)
    except Exception:
        row["harness_error"] = traceback.format_exc()[-1000:]
    finally:
        for a in agents:                       # terminate subprocess agents, every game
            close = getattr(a, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    row["wall_s"] = round(time.perf_counter() - t0, 2)

    keep = job.get("keep_replay", "crash")
    focus = job.get("focus")
    crashed = bool(row["crash0"] or row["crash1"] or row["timeout0"] or row["timeout1"] or row["harness_error"])
    focus_lost = focus is not None and (
        (focus == row["p0"] and row["result0"] == 0.0) or (focus == row["p1"] and row["result0"] == 1.0))
    want = (keep == "all") or (keep in ("crash", "loss") and crashed) or (keep == "loss" and focus_lost)
    if want and env is not None:
        try:
            REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
            tag = "crash" if crashed else "loss"
            out = REPLAYS_DIR / f"{tag}_{row['game_id'][:12]}_seed{row['seed']}.json"
            out.write_text(json.dumps(env.toJSON()), encoding="utf-8")
            row["replay_path"] = str(out)
        except Exception as exc:  # never fail a game because the replay could not be written
            row["harness_error"] = (row["harness_error"] + f"\nreplay write failed: {exc!r}")[-1000:]
    return row


def worker_main(job_q, result_q, worker_idx: int) -> None:
    """Worker process loop: warm the engine once, then play jobs until a None sentinel."""
    try:
        eng.import_engine()
    except Exception:
        result_q.put(("dead", worker_idx, traceback.format_exc()))
        return
    result_q.put(("ready", worker_idx, None))
    while True:
        job = job_q.get()
        if job is None:
            break
        result_q.put(("start", worker_idx, job["game_id"]))
        try:
            row = play_game(job)
        except BaseException:  # noqa: BLE001 -- a worker must never die silently
            row = {"game_id": job["game_id"], "harness_error": traceback.format_exc()[-1000:]}
        result_q.put(("row", worker_idx, row))
