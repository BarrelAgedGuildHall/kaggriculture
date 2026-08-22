"""Pause leases: mutual exclusion between a manual run and the hub ladder.

The durable pause (`data/arena/paused`) is a bare flag file that anyone may delete, so two
operators sharing the Pit host cannot coordinate: on 2026-08-21 the ladder was resumed by a
second actor underneath a live timing battery and contaminated its arms (foreign load inflates
turn times, and `arena gate` takes p99/max as the MAX over all stored games -- one contaminated
arm poisons the timing clause permanently).

A **lease** fixes that with the smallest possible amount of state: while any lease is live,
`hub.set_paused(False)` refuses (`LeaseHeld`), so `/api/control/resume`, the dashboard button and
any future CLI path all refuse together. Leases are:

  named    -- owner + purpose, so the refusal says who is running what and where to ask;
  durable  -- data/arena/leases.json, survives a hub restart like the pause flag itself;
  bounded  -- an explicit TTL, so a crashed run cannot wedge the ladder forever. Renew is
              explicit: nothing auto-renews, an abandoned lease always lapses;
  breakable-- `set_paused(False, force=True, actor=...)` breaks them, but it must name an actor
              and it *releases* the leases it breaks (an override that left the leases standing
              would run the ladder while their holder still believed they had exclusivity).

Every acquire/renew/release/expiry/force-break is appended to `data/arena/lease-log.jsonl` --
that log is the evidence for voiding (or not voiding) a contaminated arm after the fact.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid

from .paths import ARENA_DIR, ensure_dirs

LEASES_PATH = ARENA_DIR / "leases.json"
LEASE_LOG = ARENA_DIR / "lease-log.jsonl"

DEFAULT_TTL_S = 3600            # 1 h: long enough for a gate battery arm, short enough to lapse
MAX_TTL_S = 6 * 3600            # a run that needs more than this renews, deliberately
MIN_TTL_S = 1

# The whole file is a read-modify-write, and the hub serves on a ThreadingHTTPServer, so two
# requests could interleave between _load and _save and drop a lease. One process-wide lock plus a
# unique temp name closes that window. Across processes (hub + a CLI on the same host) the write is
# still last-writer-wins -- atomic per write thanks to os.replace, but not serialised. That is
# acceptable for now because a lost *acquire* fails safe (the caller sees its lease missing in the
# verify step the runbook requires) while a lost *release* only leaves the ladder paused.
_LOCK = threading.RLock()


class LeaseHeld(RuntimeError):
    """Raised when an operation would disturb a run that holds a live lease."""

    def __init__(self, leases: list[dict], action: str = "resume"):
        self.leases = leases
        super().__init__(describe(leases, action))


class LeaseStateUnreadable(LeaseHeld):
    """The lease file exists but cannot be read.

    This used to be swallowed (`except Exception: state = {}`), which is the worst possible
    answer: a truncated or half-written leases.json reported "no leases held", so resume went from
    409 to 200 and the ladder started underneath a live measurement -- silently, with no log row.
    Unknown state is now treated as **held**: every mutation refuses, `lease_held` reads true, and
    the only way through is a deliberate, attributed force (which quarantines the file and says so
    in the log).

    It subclasses LeaseHeld so every existing refusal path -- HTTP 409, the CLI, `hub.set_paused`
    -- already handles it.
    """

    def __init__(self, path, exc: Exception, action: str = "resume"):
        self.leases = []
        self.path = str(path)
        RuntimeError.__init__(
            self, f"{action} refused: the lease state at {path} is unreadable ({exc.__class__.__name__}: {exc}). "
                  "Unknown lease state is treated as held. Inspect the file; if it is genuinely "
                  "stale, override with a force-resume naming an actor "
                  '(POST /api/control/resume {"force":true,"actor":"<who>"}), which quarantines it '
                  "and writes the break to the lease log.")


def _epoch() -> float:                       # patched in tests to travel in time
    return time.time()


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch))


def _slug(s: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in s.strip().lower()]
    return ("".join(keep).strip("-") or "run")[:24]


def _load(action: str = "read") -> dict:
    """Read the lease state. A missing file means "no leases"; an unreadable one means "unknown",
    which is treated as held (LeaseStateUnreadable) -- never as empty."""
    try:
        raw = LEASES_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"leases": {}}
    except OSError as exc:
        log_event("state-unreadable", {"path": str(LEASES_PATH), "error": f"{exc.__class__.__name__}: {exc}"})
        raise LeaseStateUnreadable(LEASES_PATH, exc, action) from exc
    try:
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise ValueError(f"lease state must be a JSON object, got {type(state).__name__}")
        leases = state.setdefault("leases", {})
        if not isinstance(leases, dict):
            raise ValueError(f"'leases' must be an object, got {type(leases).__name__}")
        for lid, rec in leases.items():
            if not isinstance(rec, dict) or "expires_at" not in rec or "acquired_at" not in rec:
                raise ValueError(f"lease {lid!r} is missing its timestamps")
    except Exception as exc:
        log_event("state-unreadable", {"path": str(LEASES_PATH), "bytes": len(raw),
                                       "error": f"{exc.__class__.__name__}: {exc}"})
        raise LeaseStateUnreadable(LEASES_PATH, exc, action) from exc
    return state


def _save(state: dict) -> None:
    ensure_dirs()
    # unique temp name: a fixed one is a second race between two writers (F3)
    tmp = LEASES_PATH.with_name(f"{LEASES_PATH.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, LEASES_PATH)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def quarantine(actor: str, reason: str = "force") -> str | None:
    """Move an unreadable lease file aside so the ladder can be recovered deliberately.

    Only the force path calls this, it always names an actor, and it *keeps* the bad file (renamed)
    -- deleting it would turn "unknown state" back into "no leases", which is the failure this
    whole change exists to remove.
    """
    if not LEASES_PATH.exists():
        return None
    dest = LEASES_PATH.with_name(f"{LEASES_PATH.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(LEASES_PATH, dest)
    log_event("state-quarantined", {"actor": actor, "reason": reason, "moved_to": str(dest)})
    return str(dest)


def log_event(event: str, detail: dict) -> None:
    ensure_dirs()
    rec = {"at": _iso(_epoch()), "event": event, **detail}
    with LEASE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _view(rec: dict, now: float) -> dict:
    out = dict(rec)
    out["expires_in_s"] = max(0, int(round(rec["expires_at"] - now)))
    out["held_for_s"] = max(0, int(round(now - rec["acquired_at"])))
    return out


def active(now: float | None = None, action: str = "read") -> list[dict]:
    """Live leases, oldest first. Expired ones are pruned (and logged) as a side effect.

    Raises LeaseStateUnreadable if the state cannot be read -- callers that must not fail (status
    displays) use `status()` instead, which reports the fault while still reading as held."""
    now = _epoch() if now is None else now
    with _LOCK:
        state = _load(action)
        live: dict[str, dict] = {}
        expired: list[tuple[str, dict]] = []
        for lid, rec in state["leases"].items():
            if rec.get("expires_at", 0) > now:
                live[lid] = rec
            else:
                expired.append((lid, rec))
        if expired:
            state["leases"] = live
            _save(state)
    for lid, rec in expired:
        log_event("expired", {"lease_id": lid, "owner": rec.get("owner"), "purpose": rec.get("purpose"),
                              "ttl_s": rec.get("ttl_s")})
    return [_view(r, now) for _, r in sorted(live.items(), key=lambda kv: kv[1].get("acquired_at", 0))]


def status() -> dict:
    """Lease state for display: never raises, and an unreadable file still reads as HELD.

    `lease_held` is the one-glance answer, so it must never say "free" for a reason as weak as
    "I could not tell"."""
    try:
        live = active()
        return {"leases": live, "lease_held": bool(live), "lease_state_error": ""}
    except LeaseStateUnreadable as exc:
        return {"leases": [], "lease_held": True, "lease_state_error": str(exc)}


def describe(leases: list[dict], action: str = "resume") -> str:
    if not leases:
        return f"{action}: no lease held"
    parts = [f"{l['id']} (owner {l['owner']}, {l['purpose']!r}, "
             f"{max(0, int(l.get('expires_in_s', 0))) // 60}m left)" for l in leases]
    return (f"{action} refused: the ladder is leased by " + "; ".join(parts) +
            ". Ask the owner, wait for the lease to lapse, or release it "
            "(`arena lease release <id>`); override with force + actor if it is stale.")


def acquire(owner: str, purpose: str, ttl_s: int = DEFAULT_TTL_S, lease_id: str | None = None,
            host: str | None = None) -> dict:
    """Take a named lease. Concurrent leases are allowed -- they all have to be released."""
    owner, purpose = (owner or "").strip(), (purpose or "").strip()
    if not owner:
        raise ValueError("a lease needs an owner (the operative or person running it)")
    if not purpose:
        raise ValueError("a lease needs a purpose (what is being measured, e.g. 'expB-303 gate battery')")
    ttl_s = max(MIN_TTL_S, min(MAX_TTL_S, int(ttl_s)))
    now = _epoch()
    with _LOCK:
        state = _load("acquire")
        lid = lease_id or f"{_slug(owner)}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{uuid.uuid4().hex[:4]}"
        if lid in state["leases"]:
            raise ValueError(f"lease {lid!r} already exists")
        rec = {"id": lid, "owner": owner, "purpose": purpose, "ttl_s": ttl_s,
               "host": host or socket.gethostname(), "pid": os.getpid(),
               "acquired_at": now, "acquired_at_iso": _iso(now),
               "expires_at": now + ttl_s, "expires_at_iso": _iso(now + ttl_s), "renewals": 0}
        state["leases"][lid] = rec
        _save(state)
    log_event("acquire", {"lease_id": lid, "owner": owner, "purpose": purpose, "ttl_s": ttl_s,
                          "host": rec["host"]})
    return _view(rec, now)


def get(lease_id: str) -> dict:
    now = _epoch()
    with _LOCK:
        rec = _load("get")["leases"].get(lease_id)
    if rec is None:
        raise KeyError(f"no such lease: {lease_id!r}")
    if rec["expires_at"] <= now:
        raise KeyError(f"lease {lease_id!r} expired at {rec['expires_at_iso']}")
    return _view(rec, now)


def renew(lease_id: str, ttl_s: int | None = None) -> dict:
    """Extend a live lease from now. An expired lease cannot be renewed -- take a new one."""
    now = _epoch()
    with _LOCK:
        state = _load("renew")
        rec = state["leases"].get(lease_id)
        if rec is None:
            raise KeyError(f"no such lease: {lease_id!r}")
        if rec["expires_at"] <= now:
            raise KeyError(f"lease {lease_id!r} expired at {rec['expires_at_iso']} -- acquire a new one")
        rec["ttl_s"] = max(MIN_TTL_S, min(MAX_TTL_S, int(ttl_s if ttl_s is not None else rec["ttl_s"])))
        rec["expires_at"] = now + rec["ttl_s"]
        rec["expires_at_iso"] = _iso(rec["expires_at"])
        rec["renewals"] = int(rec.get("renewals", 0)) + 1
        rec["renewed_at_iso"] = _iso(now)
        _save(state)
    log_event("renew", {"lease_id": lease_id, "owner": rec["owner"], "ttl_s": rec["ttl_s"],
                        "renewals": rec["renewals"]})
    return _view(rec, now)


def release(lease_id: str, reason: str = "released") -> dict:
    now = _epoch()
    with _LOCK:
        state = _load("release")
        rec = state["leases"].pop(lease_id, None)
        if rec is None:
            raise KeyError(f"no such lease: {lease_id!r}")
        _save(state)
    log_event("release", {"lease_id": lease_id, "owner": rec.get("owner"),
                          "purpose": rec.get("purpose"), "reason": reason,
                          "held_for_s": int(round(now - rec.get("acquired_at", now)))})
    return _view(rec, now)


def break_all(actor: str, reason: str = "force") -> list[dict]:
    """Release every live lease on behalf of `actor` (the force-resume path). Always logged."""
    actor = (actor or "").strip()
    if not actor:
        raise ValueError("breaking a lease requires an actor (who is overriding, for the log)")
    try:
        live = active(action="force-resume")
    except LeaseStateUnreadable:
        # The only path allowed past an unreadable lease file, and it is deliberate, attributed
        # and logged: keep the bad file (renamed) and start from a clean, empty state.
        quarantine(actor, reason)
        return []
    broken = []
    for l in live:
        rec = release(l["id"], reason=f"{reason} by {actor}")
        broken.append(rec)
    if broken:
        log_event("force-break", {"actor": actor, "reason": reason,
                                  "lease_ids": [b["id"] for b in broken]})
    return broken
