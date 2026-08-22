"""Process pool for rollouts and harness-side evaluation.

Deliberately much simpler than `arena/scheduler.py`: the arena's pool is crash-tolerant
because an *evaluation* must record a crash rather than lose the run. A rollout that crashes
is a bug we want to see immediately, so here a worker exception propagates.

Policies are rebuilt inside each worker from `.npz` paths (weights are ~760 KB and load in
milliseconds); nothing large crosses the pickle boundary except the recorded trajectories.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

_ENGINE_READY = False


def _init():
    global _ENGINE_READY
    if not _ENGINE_READY:
        from . import harness  # noqa: F401  (imports + quiets kaggle_environments)
        _ENGINE_READY = True


def _run_eval(job):
    from . import harness as H
    _init()
    t0 = time.perf_counter()
    # `temperature` turns this into the SAMPLED read K8' needs. The opponent stays greedy either
    # way, so the sampled and greedy cells differ in exactly one thing -- which is what makes the
    # sampled-vs-greedy gap attributable to our own policy rather than to the matchup.
    temp = job.get("temperature")
    if temp is None:
        actor = H.make_policy(job["actor"], safety=job.get("safety", 3), greedy=True)
    else:
        actor = H.make_policy(job["actor"], safety=job.get("safety", 3), greedy=False,
                              temperature=float(temp), seed=int(job.get("rng_seed", 0)),
                              record=False)
    opp = H.make_policy(job["opponent"], safety=job.get("safety", 3), greedy=True)
    us, them, n = H.play(job["seed"], job["seat"], actor, opp, steps=job.get("steps", 720))
    return {"seed": job["seed"], "seat": job["seat"], "us": us, "them": them,
            "steps": n, "wall_s": time.perf_counter() - t0,
            "temperature": temp,
            "opponent": job.get("opponent_id", str(job["opponent"]))}


def _run_rollout(job):
    from . import harness as H
    _init()
    t0 = time.perf_counter()
    turns, urows, us, them, ep_lines, ep_aux = H.rollout(
        job["seed"], job["seat"], job["actor"], job["opponent"], job["rng_seed"],
        temperature=job.get("temperature", 1.0), safety=job.get("safety", 3),
        steps=job.get("steps", 720), record_units=job.get("record_units", False),
        unit_subsample=job.get("unit_subsample", 1.0),
        unit_temperature=job.get("unit_temperature"),
        record_lines=job.get("record_lines", False),
        aux_names=job.get("aux_names", ()))
    return {"turns": turns, "unit_rows": urows, "R": us, "R_opp": them,
            "lines": ep_lines, "aux": ep_aux,
            "seed": job["seed"], "seat": job["seat"],
            "opponent_id": job.get("opponent_id", ""), "wall_s": time.perf_counter() - t0}


def run(jobs, workers, kind="eval", chunksize=1):
    fn = _run_rollout if kind == "rollout" else _run_eval
    if workers <= 1:
        return [fn(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init) as pool:
        return pool.map(fn, jobs, chunksize=chunksize)


def default_workers(reserve=2):
    return max(1, (os.cpu_count() or 4) - reserve)
