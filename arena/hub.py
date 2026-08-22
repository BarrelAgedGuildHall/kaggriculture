"""The Pit hub: the continuous ladder matchmaker that runs alongside the dashboard.

Modelled on the PTCG arena's 24/7 worker (arena_worker.py): the ladder plays continuously,
**least-played-first** — each round the active agent with the fewest ladder games anchors a
star match against the most rating-similar active opponents, on fresh paired seeds with side
swap. New agents (0 games) jump to the front and get rated fast. Results are appended to the
sqlite store as each game finishes; the dashboard's BT/Elo fit reads them live.

Control state lives in data/arena/ so it survives restarts and is shared with the CLI:
  paused          flag file  — hub idles while it exists (durable pause, PTCG-style)
  leases.json     pause leases — while any is live the ladder may not be resumed (see leases.py)
  hub.json        config     — {"workers": N, "games_per_pairing": G, "max_opponents": K}
  registry.json   active/benched/retired flags decide who plays (arena bench/retire or the UI)

The hub never edits old rows and never runs two rounds at once. Round size is small
(max_opponents pairings x games_per_pairing episodes), so pause/config changes take effect
within a round (~1-3 min).
"""
from __future__ import annotations

import json
import threading
import time
import uuid

from . import engine as eng
from . import leases, ratings, registry, store
from .paths import ARENA_DIR, LADDER_SEED_BASE, ensure_dirs
from .scheduler import run_match

PAUSE_FLAG = ARENA_DIR / "paused"
HUB_CONFIG = ARENA_DIR / "hub.json"
DEFAULTS = {"workers": 0, "games_per_pairing": 4, "max_opponents": 5}


def config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(HUB_CONFIG.read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg


def set_config(**kw) -> dict:
    ensure_dirs()
    cfg = config()
    cfg.update({k: int(v) for k, v in kw.items() if k in DEFAULTS and v is not None})
    HUB_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def is_paused() -> bool:
    return PAUSE_FLAG.exists()


def set_paused(paused: bool, *, force: bool = False, actor: str = "") -> bool:
    """Set the durable pause flag. Resuming is refused while a pause lease is live.

    The check lives here rather than in the HTTP layer so that every path -- the API, the
    dashboard button, the CLI and any script importing the module -- obeys the same rule.
    `force=True` needs an `actor` and *breaks* the leases it overrides, so the state stays
    honest: nobody is left believing they hold exclusivity while the ladder plays.
    """
    ensure_dirs()
    if paused:
        if not PAUSE_FLAG.exists():     # keep the original timestamp: "paused since" is evidence
            PAUSE_FLAG.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    else:
        if force:
            # break_all is the only path that may pass an unreadable lease state, and it
            # quarantines the file under the actor's name when it does.
            leases.break_all(actor, reason="force-resume")
        else:
            # raises LeaseHeld, or LeaseStateUnreadable (a LeaseHeld) if the state is unknown
            held = leases.active(action="resume")
            if held:
                raise leases.LeaseHeld(held, action="resume")
        PAUSE_FLAG.unlink(missing_ok=True)
    return is_paused()


def acquire_lease(owner: str, purpose: str, ttl_s: int = leases.DEFAULT_TTL_S,
                  pause: bool = True, shared: bool = False) -> dict:
    """Take a pause lease and (by default) pause the ladder in the same step.

    The ladder finishes the round it is already playing, so the caller must wait for
    `status()['phase'] == 'paused'` before it starts measuring anything time-sensitive.

    Exclusive unless `shared` -- a second holder raises `leases.LeaseConflict` (HTTP 409) and the
    ladder is left exactly as it was, because the pause only happens after the lease is granted.
    """
    rec = leases.acquire(owner, purpose, ttl_s, shared=shared)
    if pause:
        set_paused(True)
    return rec


def release_lease(lease_id: str, resume: bool = False) -> dict:
    """Release a lease. Does NOT resume unless asked -- and then only if no lease remains."""
    rec = leases.release(lease_id)
    if resume and not leases.status()["lease_held"]:
        set_paused(False)
    return rec


def reconcile_pause() -> bool:
    """A live lease means "the ladder must not play" -- re-assert the pause if it does not hold.

    The hub loop used to check the pause flag only, so a lease held over an un-paused ladder (a
    lease taken with `pause=False`, a flag deleted by hand, two arena dirs) had nothing reconciling
    the two: the leaseholder believed it had exclusivity while the hub dealt games underneath it.
    Returns True when it had to pause. An unreadable lease file counts as held (leases.status()).
    """
    held = leases.status()
    if not held["lease_held"] or is_paused():
        return False
    leases.log_event("reconciled-pause", {
        "reason": "lease held while the ladder was not paused",
        "lease_ids": [l["id"] for l in held["leases"]],
        "state_error": held["lease_state_error"]})
    set_paused(True)
    return True


def _ladder_counts_and_seed() -> tuple[dict, int]:
    con = store.connect()
    counts: dict[str, int] = {}
    max_seed = LADDER_SEED_BASE - 1
    for g in con.execute("SELECT p0, p1, seed FROM games WHERE mode = 'ladder'"):
        counts[g["p0"]] = counts.get(g["p0"], 0) + 1
        counts[g["p1"]] = counts.get(g["p1"], 0) + 1
        max_seed = max(max_seed, int(g["seed"]))
    games = store.all_games(con)
    con.close()
    return {"counts": counts, "games": games}, max_seed + 1


def plan_round() -> dict | None:
    """Pick the anchor and its opponents. None when fewer than 2 active agents."""
    reg = registry.ensure_residents()
    field = [a for a in registry.list_agents(reg) if registry.is_active(a)]
    if len(field) < 2:
        return None
    state, next_seed = _ladder_counts_and_seed()
    counts = state["counts"]
    anchor = min(field, key=lambda a: (counts.get(a["id"], 0), a["registered_at"]))
    bt = ratings.bradley_terry(state["games"]) if state["games"] else {}
    a_r = bt.get(anchor["id"], 0.0)
    others = sorted((b for b in field if b["id"] != anchor["id"]),
                    key=lambda b: (abs(bt.get(b["id"], 0.0) - a_r), counts.get(b["id"], 0)))
    cfg = config()
    opponents = others[: max(1, cfg["max_opponents"])]
    return {"anchor": anchor, "opponents": opponents, "next_seed": next_seed, "cfg": cfg,
            "anchor_games": counts.get(anchor["id"], 0)}


class Hub:
    """Background matchmaker thread. `status()` is read by the dashboard/API."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = {"running": False, "phase": "stopped", "round": None, "rounds_played": 0,
                       "last_error": "", "started_at": None}
        self._lock = threading.Lock()

    def status(self) -> dict:
        with self._lock:
            st = dict(self._state)
        st["paused"] = is_paused()
        st["config"] = config()
        # status() must never raise on a bad lease file -- but it must not report "free" either:
        # leases.status() reads an unreadable state as HELD and carries the fault forward.
        st.update(leases.status())
        return st

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="pit-hub", daemon=True)
        self._thread.start()
        self._set(running=True, phase="starting", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def stop(self):
        self._stop.set()
        self._set(running=False, phase="stopped")

    def _loop(self):
        while not self._stop.is_set():
            try:
                if reconcile_pause():
                    self._set(phase="paused", round=None)
                if is_paused():
                    self._set(phase="paused", round=None)
                    self._stop.wait(3.0)
                    continue
                plan = plan_round()
                if plan is None:
                    self._set(phase="idle (need >=2 active agents)", round=None)
                    self._stop.wait(5.0)
                    continue
                cfg = plan["cfg"]
                pairs = max(1, cfg["games_per_pairing"] // 2)
                seeds = ",".join(str(plan["next_seed"] + i) for i in range(pairs))
                run_id = "ladder-" + time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
                desc = {"run_id": run_id, "anchor": plan["anchor"]["id"], "anchor_games": plan["anchor_games"],
                        "opponents": [o["id"] for o in plan["opponents"]], "seeds": seeds}
                self._set(phase="playing", round=desc)
                run_match([plan["anchor"]["id"]] + [o["id"] for o in plan["opponents"]],
                          mode="screen", n=pairs * 2, seeds=seeds, workers=cfg["workers"] or None,
                          quiet=True, run_id=run_id)
                # rows are stored with mode='screen' by run_match; retag this run as ladder
                con = store.connect()
                con.execute("UPDATE games SET mode = 'ladder' WHERE run_id = ?", (run_id,))
                con.execute("UPDATE runs SET mode = 'ladder' WHERE run_id = ?", (run_id,))
                con.commit()
                con.close()
                with self._lock:
                    self._state["rounds_played"] += 1
                    self._state["round"] = None
            except Exception as exc:  # the hub must survive anything a round throws
                self._set(phase="error", last_error=repr(exc)[:500])
                self._stop.wait(10.0)


HUB = Hub()
