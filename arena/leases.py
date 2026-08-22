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
  exclusive-- one holder at a time (`shared=True` opts out), see below;
  breakable-- `set_paused(False, force=True, actor=...)` breaks them, but it must name an actor
              and it *releases* the leases it breaks (an override that left the leases standing
              would run the ladder while their holder still believed they had exclusivity).

**Exclusivity (2026-08-22).** The first version allowed concurrent leases and only ever blocked
the *ladder*, which is not the thing a measured run collides with. On 2026-08-21 21:08-22:00 MDT
`engineer-b` held a lease for the expB-304a gate battery and `modeler` took a second one 15
minutes later and started its own battery on the same 16-core host: one arm ran at load 36.67 and
recorded 16 spurious timeouts, p99 68.7 ms and a 2647 ms worst turn for an agent that measures
p99 7.2 ms quiesced -- and the arena scores an over-budget turn as a loss regardless of bank. So:

  * `acquire` is **exclusive by default** -- a second acquire raises `LeaseConflict` (HTTP 409)
    naming the holder. `shared=True` opts in to coexistence, and two shared leases may coexist,
    but a shared lease still cannot be taken over an exclusive one;
  * `guard_run` makes `arena match` / `arena gate` **refuse to start** while somebody else holds
    a lease. A lease that does not exclude the thing it exists to protect against is a warning
    light, not a lock.
  * and, since the guard only closed one direction (Auditor L-1), a run that declares no lease
    now **takes one** (`hub.lease_for_run`, released on every exit path) instead of proceeding
    silently. Two leaseless runs colliding was the same incident with the actors swapped, and it
    was the easy default because nothing ever prompted for a lease. `--no-lease` is the opt-out
    and is logged as `run-unleased`.

Anything that takes a lease must release it on **every** exit path, including Ctrl-C: use
`leases.held(...)` in Python or `scripts/lease.sh` (an EXIT trap) in a shell script. A lease left
behind by a dead process holds the ladder down and refuses everybody else's runs until its TTL
lapses.

Every acquire/renew/release/expiry/force-break/refusal is appended to
`data/arena/lease-log.jsonl` -- that log is the evidence for voiding (or not voiding) a
contaminated arm after the fact.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
import uuid

from .paths import ARENA_DIR, ensure_dirs

LEASES_PATH = ARENA_DIR / "leases.json"
LEASE_LOG = ARENA_DIR / "lease-log.jsonl"
LOCK_PATH = ARENA_DIR / "leases.lock"

DEFAULT_TTL_S = 3600            # 1 h: long enough for a gate battery arm, short enough to lapse
MAX_TTL_S = 6 * 3600            # a run that needs more than this renews, deliberately
MIN_TTL_S = 1

LOCK_WAIT_S = 5.0               # how long a mutation waits for the cross-process lock
LOCK_STALE_S = 20.0             # a lock file older than this belonged to a dead process

# Env fallbacks so a battery script exports once instead of threading flags through every command.
LEASE_ENV = "KAGGRI_LEASE"      # the lease id *I* hold -- my own lease is not a foreign lease
HUB_ENV = "KAGGRI_PIT_HUB"      # also check this hub's leases before starting a run
OWNER_ENV = "KAGGRI_LEASE_OWNER"  # who to name as owner on a lease taken for you (see default_owner)

# The whole file is a read-modify-write, and the hub serves on a ThreadingHTTPServer, so two
# requests could interleave between _load and _save and drop a lease. `_LOCK` closes that window
# between threads. Between *processes* (the hub and a CLI on the Pit host, or two operatives'
# shells) it used to be last-writer-wins, which was tolerable while leases were advisory and is
# not tolerable now that a lease is a lock: two `acquire` calls could each read "no leases" and
# both write themselves in. `_state_lock` adds an O_EXCL lock file around every mutation.
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


class LeaseLockTimeout(LeaseStateUnreadable):
    """The cross-process lock could not be taken, so the state could not be mutated safely.

    Subclasses LeaseStateUnreadable (and so LeaseHeld) on purpose: "I could not establish the
    lease state" must read as HELD everywhere, never as free.
    """

    def __init__(self, path, waited: float, action: str = "read"):
        LeaseStateUnreadable.__init__(
            self, path, TimeoutError(f"another process held the lease lock for {waited:.1f}s"), action)


class LockLost(LeaseStateUnreadable):
    """This process was inside the state lock when the lock was broken as stale.

    Subclasses LeaseStateUnreadable (and so LeaseHeld) for the usual reason: "I could not
    establish the lease state" must read as HELD everywhere. The mutation was refused, not
    half-applied -- the write never happened.
    """

    def __init__(self, path, action: str, held: str, found: str | None):
        self.leases = []
        self.path = str(path)
        RuntimeError.__init__(
            self, f"{action} refused: this process held the lease lock {held} but the lock file "
                  f"now holds {found!r} -- it was broken as stale (more than {LOCK_STALE_S:.0f}s "
                  f"inside a lease read-modify-write) and somebody else owns it. The write was "
                  f"NOT applied. Retry; if this repeats, the host is stalling badly enough that "
                  f"any timing measured on it is suspect (see `lock-lost` in lease-log.jsonl).")


class LeaseConflict(LeaseHeld):
    """A lease was requested while an incompatible one is live (the exclusivity rule).

    Exclusive is the default, so the ordinary case is "somebody already holds the Pit". The
    message has to be enough to act on without a second command: who, what for, how long left,
    and the exact release invocation.
    """

    def __init__(self, held: list[dict], shared: bool = False):
        self.leases = held
        parts = [f"{l['id']} (owner {l['owner']}, {l['purpose']!r}, "
                 f"{'shared' if l.get('shared') else 'exclusive'}, "
                 f"{max(0, int(l.get('expires_in_s', 0))) // 60}m left)" for l in held]
        want = "a shared lease" if shared else "an exclusive lease"
        RuntimeError.__init__(
            self, f"acquire refused: cannot take {want} while the Pit is leased by "
                  + "; ".join(parts) + ". A lease is exclusive so that two batteries cannot share "
                  "the host and poison each other's timing. Wait for it to lapse, ask the owner, "
                  f"or release it (`arena lease release {held[0]['id']}`). If your hold genuinely "
                  "does not measure anything, both sides must use `--shared`.")


class LeaseCheckFailed(LeaseHeld):
    """A lease source (a remote hub) could not be read, so a run must not start."""

    def __init__(self, where, exc: Exception, action: str = "run"):
        self.leases = []
        RuntimeError.__init__(
            self, f"{action} refused: the lease state at {where} could not be read "
                  f"({exc.__class__.__name__}: {exc}). Unknown lease state is treated as held -- "
                  "fix the source, or drop the remote check if it is not the host you run on.")


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
    _refuse_if_lock_lost("save")
    # unique temp name: a fixed one is a second race between two writers (F3)
    tmp = LEASES_PATH.with_name(f"{LEASES_PATH.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, LEASES_PATH)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


_HELD = threading.local()       # the lock token this thread is currently inside, if any


def _refuse_if_lock_lost(action: str) -> None:
    """Refuse a write whose lock was broken out from under it.

    Breaking a stale lock lets a second holder in while the first is still (very slowly) inside;
    no timeout-based lock can avoid that, and 20 s inside a small JSON read-modify-write needs a
    pathological stall. What it CAN avoid is the slow holder's write landing on top of the new
    holder's state -- which is the part that actually corrupts a lease file. So the last thing
    before every write is: is this still my lock? If not, the mutation fails closed
    (LeaseLockTimeout is a LeaseHeld: "I could not establish the state" reads as HELD, never as
    free) and `lock-lost` records that it happened.
    """
    token = getattr(_HELD, "token", None)
    if token is None:
        return                                   # not inside _state_lock (a pure read path)
    found = _lock_holder()
    if found == token:
        return
    log_event("lock-lost", {"action": action, "pid": os.getpid(), "held_token": token,
                            "found": found,
                            "note": "the lock was broken as stale while this mutation was inside "
                                    "it; the write was refused rather than applied over the new "
                                    "holder's state"})
    raise LockLost(LOCK_PATH, action, token, found)


def _lock_token(action: str) -> str:
    """The identity written into a lock file, first line, one line.

    Host and pid make it readable in a forensic dump; the uuid is what makes it *unique* -- a pid
    is reused, and "pid 4242 acquire" from a dead process must never be mistaken for our own lock.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _lock_holder() -> str | None:
    """The token in the current lock file, or None if there is no lock file."""
    try:
        return LOCK_PATH.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None


def _lock_is_stale() -> bool:
    """True (and the lock is broken) when the lock file outlived any plausible holder.

    Deliberately uses the real clock, not `_epoch`: tests travel in time to expire *leases*, and
    that must not make the lock look abandoned. A crashed holder would otherwise wedge every
    mutation forever, which is the failure mode a lock file is famous for.

    The break is a **rename**, not an unlink (Auditor L-2, 2026-08-22): `os.replace` to a unique
    name is atomic, so when two processes decide the same lock is stale exactly one of them owns
    the break and the loser simply retries. Unlinking let both "break" it and then both create,
    and the evidence of the break disappeared with the file.
    """
    try:
        age = time.time() - os.path.getmtime(LOCK_PATH)
    except OSError:
        return True                                  # it went away on its own: retry immediately
    if age <= LOCK_STALE_S:
        return False
    holder = _lock_holder() or "?"
    dest = LOCK_PATH.with_name(f"{LOCK_PATH.name}.broken-{time.strftime('%Y%m%d-%H%M%S')}-"
                               f"{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        os.replace(LOCK_PATH, dest)
    except OSError:
        return True                                  # somebody else broke it first: retry
    with contextlib.suppress(OSError):
        dest.unlink()
    log_event("lock-broken", {"held_by": holder, "age_s": round(age, 1), "breaker_pid": os.getpid()})
    return True


@contextlib.contextmanager
def _state_lock(action: str = "mutate"):
    """Serialise a lease read-modify-write across threads AND processes.

    Order is always `_LOCK` then the lock file, and no critical section nests inside another, so
    there is nothing to deadlock on. Only mutations take it -- `status()`/`active()` reads stay
    lock-free so a stale lock cannot wedge the dashboard's poll (a *jammed* one still fails the
    prune closed, via LeaseLockTimeout -- see `active()`).

    **Release is identity-checked** (Auditor L-2, 2026-08-22). The first version unlinked
    unconditionally, so a critical section that outlived LOCK_STALE_S had its lock broken, a
    second process took its own, and the first then deleted the *second's* lock on the way out --
    letting a third in while the second was still mutating (observed A-in, B-in, A-out, B-out).
    Now the lock file carries a unique token and we only remove the file that still carries ours;
    finding somebody else's is logged as `lock-lost`, which is the forensic marker that a
    mutation ran outside the lock and any lease written around it is suspect.

    The residual window -- read our token, and be broken before the unlink lands -- is microseconds
    against a 20 s staleness threshold, i.e. strictly smaller than the window it replaces, and it
    is bounded by the same `lock-lost` evidence on the next holder's way out.
    """
    with _LOCK:
        ensure_dirs()
        token = _lock_token(action)
        t0 = time.monotonic()
        while True:
            try:
                fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                waited = time.monotonic() - t0
                if _lock_is_stale():
                    continue
                if waited > LOCK_WAIT_S:
                    raise LeaseLockTimeout(LOCK_PATH, waited, action)
                time.sleep(0.01)
        previous = getattr(_HELD, "token", None)
        try:
            try:
                os.write(fd, f"{token}\npid {os.getpid()} {action} "
                             f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
            finally:
                os.close(fd)
            _HELD.token = token
            yield
        finally:
            _HELD.token = previous
            _release_lock(token, action)


def _release_lock(token: str, action: str) -> None:
    """Drop the lock file, but only if it is still the one we created."""
    holder = _lock_holder()
    if holder == token:
        LOCK_PATH.unlink(missing_ok=True)
        return
    log_event("lock-lost", {"action": action, "pid": os.getpid(), "held_token": token,
                            "found": holder,
                            "note": "our lock was broken as stale while we were inside it; the "
                                    "file now belongs to somebody else and was left alone"})


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
    out["shared"] = bool(rec.get("shared"))
    out["expires_in_s"] = max(0, int(round(rec["expires_at"] - now)))
    out["held_for_s"] = max(0, int(round(now - rec["acquired_at"])))
    return out


def _partition(state: dict, now: float) -> tuple[dict, list[tuple[str, dict]]]:
    """Split the stored leases into live and expired. Pure -- no I/O, no logging."""
    live: dict[str, dict] = {}
    expired: list[tuple[str, dict]] = []
    for lid, rec in state["leases"].items():
        if rec.get("expires_at", 0) > now:
            live[lid] = rec
        else:
            expired.append((lid, rec))
    return live, expired


def _sorted_views(live: dict, now: float) -> list[dict]:
    return [_view(r, now) for _, r in sorted(live.items(), key=lambda kv: kv[1].get("acquired_at", 0))]


def _log_expired(expired: list[tuple[str, dict]]) -> None:
    for lid, rec in expired:
        log_event("expired", {"lease_id": lid, "owner": rec.get("owner"), "purpose": rec.get("purpose"),
                              "ttl_s": rec.get("ttl_s")})


def active(now: float | None = None, action: str = "read") -> list[dict]:
    """Live leases, oldest first. Expired ones are pruned (and logged) as a side effect.

    The read itself is lock-free -- only the prune takes the cross-process lock, so a stale lock
    file can never wedge the dashboard's status poll, and a prune that loses a race is harmless
    (the next reader prunes again).

    Raises LeaseStateUnreadable if the state cannot be read -- callers that must not fail (status
    displays) use `status()` instead, which reports the fault while still reading as held."""
    now = _epoch() if now is None else now
    with _LOCK:
        live, expired = _partition(_load(action), now)
    if expired:
        with _state_lock("prune"):
            state = _load(action)
            live, expired = _partition(state, now)
            state["leases"] = live
            _save(state)
        _log_expired(expired)
    return _sorted_views(live, now)


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
             f"{max(0, int(l.get('expires_in_s', 0))) // 60}m left"
             + (f", on {l['hub']}" if l.get("hub") else "") + ")" for l in leases]
    if action in ("resume", "force-resume"):
        tail = ("Ask the owner, wait for the lease to lapse, or release it "
                "(`arena lease release <id>`); override with force + actor if it is stale.")
    else:
        tail = (f"Starting now would share the host with that run and poison both sides' timing. "
                f"Wait for it to lapse or ask the owner; pass `--lease {leases[0]['id']}` if it is "
                "in fact yours, or `--ignore-lease` (logged) if this run measures nothing.")
    return f"{action} refused: the Pit is leased by " + "; ".join(parts) + ". " + tail


def _blockers(live: dict, now: float, shared: bool) -> list[dict]:
    """Which live leases refuse a new `shared`/exclusive one.

    An exclusive request is blocked by ANY live lease; a shared request only by an exclusive one,
    so two deliberate `--shared` holds coexist while nothing can slip in beside a measured run.
    """
    blocking = [r for r in live.values() if not (shared and r.get("shared"))]
    return _sorted_views({r["id"]: r for r in blocking}, now)


def acquire(owner: str, purpose: str, ttl_s: int = DEFAULT_TTL_S, lease_id: str | None = None,
            host: str | None = None, shared: bool = False) -> dict:
    """Take a named lease. **Exclusive by default**: a second acquire raises LeaseConflict.

    `shared=True` is the opt-out for a hold that measures nothing (an ingestion window, a long
    unmeasured training run that only wants the ladder quiet). Shared leases coexist with each
    other but never with an exclusive one -- so a shared holder cannot appear beside a battery
    and cannot keep a battery from starting once it releases.
    """
    owner, purpose = (owner or "").strip(), (purpose or "").strip()
    if not owner:
        raise ValueError("a lease needs an owner (the operative or person running it)")
    if not purpose:
        raise ValueError("a lease needs a purpose (what is being measured, e.g. 'expB-303 gate battery')")
    ttl_s = max(MIN_TTL_S, min(MAX_TTL_S, int(ttl_s)))
    now = _epoch()
    # The check and the write share one critical section, across processes as well as threads:
    # two shells that each read "no leases" and then each wrote themselves in is precisely the
    # race this is here to lose.
    with _state_lock("acquire"):
        state = _load("acquire")
        live, expired = _partition(state, now)
        blocked = _blockers(live, now, shared)
        if blocked:
            log_event("acquire-refused", {"owner": owner, "purpose": purpose, "shared": shared,
                                          "held_by": [b["id"] for b in blocked]})
            raise LeaseConflict(blocked, shared)
        lid = lease_id or f"{_slug(owner)}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{uuid.uuid4().hex[:4]}"
        if lid in live:
            raise ValueError(f"lease {lid!r} already exists")
        rec = {"id": lid, "owner": owner, "purpose": purpose, "ttl_s": ttl_s, "shared": bool(shared),
               "host": host or socket.gethostname(), "pid": os.getpid(),
               "acquired_at": now, "acquired_at_iso": _iso(now),
               "expires_at": now + ttl_s, "expires_at_iso": _iso(now + ttl_s), "renewals": 0}
        live[lid] = rec
        state["leases"] = live
        _save(state)
    _log_expired(expired)
    log_event("acquire", {"lease_id": lid, "owner": owner, "purpose": purpose, "ttl_s": ttl_s,
                          "shared": bool(shared), "host": rec["host"]})
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
    with _state_lock("renew"):
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
    with _state_lock("release"):
        state = _load("release")
        rec = state["leases"].pop(lease_id, None)
        if rec is None:
            raise KeyError(f"no such lease: {lease_id!r}")
        _save(state)
    log_event("release", {"lease_id": lease_id, "owner": rec.get("owner"),
                          "purpose": rec.get("purpose"), "reason": reason,
                          "held_for_s": int(round(now - rec.get("acquired_at", now)))})
    return _view(rec, now)


@contextlib.contextmanager
def held(owner: str, purpose: str, ttl_s: int = DEFAULT_TTL_S, shared: bool = False,
         reason: str = "run finished"):
    """Hold a lease for the duration of a block, and release it however the block ends.

    This is the Python half of the EXIT-trap convention (`scripts/lease.sh` is the shell half):
    every long-running thing that takes a lease must release it on **every** exit path, including
    an exception and a Ctrl-C, or the next operative finds the Pit leased by a process that died
    an hour ago and has to wait out the TTL.

        with leases.held("engineer-b", "expB-303 gate battery", ttl_s=3600) as lease:
            os.environ[leases.LEASE_ENV] = lease["id"]     # child `arena match`es inherit it
            ...

    Releasing a lease that is already gone (it lapsed, or somebody force-broke it) is not an
    error -- it is logged and swallowed, because raising in a finally would hide whatever the
    block was actually doing.
    """
    rec = acquire(owner, purpose, ttl_s=ttl_s, shared=shared)
    try:
        yield rec
    finally:
        try:
            release(rec["id"], reason=reason)
        except KeyError:
            log_event("release-noop", {"lease_id": rec["id"], "owner": owner,
                                       "reason": "already gone at release (lapsed or force-broken)"})


def default_owner() -> str:
    """Who to name as the owner when nobody said (`KAGGRI_LEASE_OWNER`, else user@host).

    A lease with a meaningless owner is only half a lease -- the whole point of the refusal
    message is that it tells the next operative who to ask -- so operatives should export
    KAGGRI_LEASE_OWNER (their operative id) on the Pit host. user@host is the honest fallback.
    """
    named = (os.environ.get(OWNER_ENV) or "").strip()
    if named:
        return named
    try:
        import getpass
        user = getpass.getuser()
    except Exception:                               # noqa: BLE001 - no name is not a failure
        user = "unknown"
    return f"{user}@{socket.gethostname()}"


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


# --------------------------------------------------------------- the run-start guard (2026-08-22)

def hub_leases(hub_url: str, action: str = "run") -> list[dict]:
    """Live leases on a remote hub, tagged with where they came from.

    Fails CLOSED: a hub we cannot reach, or one reporting an unreadable lease file, is not
    evidence that the host is free. The remote check is opt-in for exactly that reason -- turn it
    on for the host you actually run games on.
    """
    import json as _json
    import urllib.request
    url = hub_url.rstrip("/") + "/api/leases"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            out = _json.loads(resp.read())
    except Exception as exc:                    # noqa: BLE001 - any failure is "state unknown"
        raise LeaseCheckFailed(url, exc, action) from exc
    if out.get("lease_state_error"):
        raise LeaseCheckFailed(url, RuntimeError(out["lease_state_error"]), action)
    return [dict(l, hub=hub_url) for l in out.get("leases", [])]


def guard_run(action: str, lease_id: str | None = None, hub_url: str | None = None,
              ignore: bool = False, detail: str = "") -> list[dict]:
    """Refuse to start a measured run while somebody *else* holds the Pit.

    This is the half of the lock the first version was missing. `acquire` now stops a second
    lease, but the 2026-08-21 contamination did not need a second lease -- it only needed a second
    `arena match`, and nothing checked. So `arena match` and `arena gate` call this at run start.

    It is only the *refusal* half. Refusing under a foreign lease and then running leaseless
    leaves the mirror hole open (Auditor L-1), so the CLI follows a clean guard with
    `hub.lease_for_run` unless the caller declared a lease or opted out -- see `cli._run_lease`.

      lease_id  the lease you hold (CLI `--lease`, env KAGGRI_LEASE): yours is not foreign
      hub_url   also consult a remote hub (CLI `--lease-hub`, env KAGGRI_PIT_HUB) for the case
                where the games run somewhere other than this arena dir
      ignore    the deliberate override; it is written to the lease log, not swallowed

    The hub ladder is exempt by design: it calls `scheduler.run_match` directly and is already
    held off by the pause flag that `hub.reconcile_pause()` re-asserts.
    """
    lease_id = (lease_id or os.environ.get(LEASE_ENV, "")).strip() or None
    hub_url = (hub_url or os.environ.get(HUB_ENV, "")).strip() or None
    seen = list(active(action=action))          # raises LeaseStateUnreadable -> refuse, fail closed
    if hub_url:
        seen += hub_leases(hub_url, action)
    foreign = [l for l in seen if l.get("id") != lease_id]
    if not foreign:
        return seen
    if ignore:
        log_event("guard-ignored", {"action": action, "detail": detail, "pid": os.getpid(),
                                    "lease_ids": [l["id"] for l in foreign],
                                    "owners": [l.get("owner") for l in foreign]})
        return seen
    log_event("guard-refused", {"action": action, "detail": detail, "pid": os.getpid(),
                                "declared_lease": lease_id,
                                "lease_ids": [l["id"] for l in foreign],
                                "owners": [l.get("owner") for l in foreign]})
    raise LeaseHeld(foreign, action=action)
