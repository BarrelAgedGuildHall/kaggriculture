"""Millrace M0 — chassis: entry point, sanitiser, harm-ordered fallback, bank latch.

Hard constraints honoured here (spec section 0):
  - global try/except -> a safe action; the agent never raises on the ladder
  - PLANT sanitiser: if a turn's PLANT requests for one crop exceed seeds held,
    the engine drops EVERY PLANT for that crop that turn -> cap emitted PLANTs
  - market order hygiene: <= 10 orders/turn, SELLs before BUYs, quantities
    clamped to shed contents / int-coerced
  - stdlib only, no unbounded loops

TIMING MODEL (ruling-04 section 1, read from kaggle-environments 1.32.7 source —
`agent.py:220`, `core.py:631-632`, `docs/engine/kaggriculture.json`):

    actTimeout = 1 s per turn
    remainingOverageTime = 60 s per agent per EPISODE (kaggriculture overrides
                           the framework schema default of 12)
    if duration - actTimeout > observation.remainingOverageTime:
        action = DeadlineExceeded()   # -> status TIMEOUT -> reward None

The runner NEVER interrupts a turn.  It lets the turn finish and then bills the
excess against a per-episode bank.  So the real constraint is a BANK, not a
per-turn cliff: `sum_t max(0, d_t - actTimeout) <= 60 s`.  Two consequences,
which are the whole shape of this module:

  1. A post-hoc "the turn was slow, throw the finished plan away" branch is
     strictly dominated.  The wall-clock is already spent and `sanitise` has
     already run; emitting the plan costs zero additional time.  It converted a
     slow turn into a slow turn AND a lost turn, so it is gone (C-1/R-26).
  2. Our safety valve now has the same shape as the engine's: a latch on the
     bank (`remainingOverageTime`, which is in the observation and readable at
     runtime) rather than a threshold on one turn (C-3/R-28).

`deadline_s` survives only as an ADVISORY number handed to the planner and
reported in the latency histogram.  It is `deadline_frac * actTimeout`, never an
absolute-seconds constant (C-4/R-29; section 6.4 as amended bars absolute
dollars, days AND seconds).

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import atexit
import json
import os
import time
import traceback

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
# animals never SELL from the shed via this sanitiser guard (they are placed,
# not sold, in every Millrace plan); crops with seeds live in private.seeds
ANIMAL_NAMES = ("GOOSE", "COW", "SHEEP")

SAFE = {"farmer": ["PASS"], "hands": [], "market": []}

# Vendored engine table (kaggle-environments 1.32.7 / docs/engine/kaggriculture.json).
# Used ONLY when the loader hands us no configuration; the two-argument agent
# signature reads the live `actTimeout` in both Kaggle's loader and the Pit's
# child harness.  tests/test_team_b_chassis.py pins this against the installed
# engine, so an engine bump breaks a test instead of silently mis-sizing us.
ACT_TIMEOUT_S = 1.0
OVERAGE_BANK_S = 60.0

# The planner's advisory budget, as a fraction of the engine's per-turn figure.
DEADLINE_FRAC = 0.60

# PANIC latch (C-3): three cumulative planner exceptions, or the engine's own
# overage bank down to a quarter of what we observed on turn 0.
PANIC_EXCEPTIONS = 3
PANIC_BANK_FRAC = 0.25


def _safe_copy():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def act_timeout_of(config):
    """The engine's per-turn figure, live from `configuration` when the loader
    gives us one, else the vendored table."""
    try:
        v = float(config["actTimeout"])
    except Exception:
        return ACT_TIMEOUT_S
    return v if v > 0.0 else ACT_TIMEOUT_S


def overage_bank_of(obs):
    """`remainingOverageTime` from the observation, or None when absent (some
    harnesses build a bare observation)."""
    try:
        v = obs["remainingOverageTime"]
    except Exception:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fallback_action(obs, latched=False):
    """Harm-ordered planner-less fallback (C-3 / R-28).

    TRANSIENT (`latched=False`, i.e. a single planner exception): unit actions
    only — DROP if carrying on a shed tile, else PASS — and **no market orders
    at all**.  One bad turn must not do anything irreversible: the old
    unconditional "SELL every product in the shed" dumped the whole shed at spot
    into our own price impact, on a MELON curve that is quadratic above target.

    LATCHED (PANIC, for the remainder of the episode): the same unit actions
    PLUS the SELL-the-shed market leg, because once the planner is out of the
    game simple selling is the only revenue path left.
    """
    out = _safe_copy()
    try:
        me = obs["farms"][obs["player"]]
        priv = obs["private"]
        units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
        invs = priv["inventories"]
        acts = []
        for i, pos in enumerate(units):
            carry = sum((invs[i] if i < len(invs) else {}).values())
            acts.append(["DROP"] if (carry > 0 and pos in SHED_TILES) else ["PASS"])
        orders = []
        if latched:
            for item, n in priv["shed"].items():
                if n > 0 and item not in ANIMAL_NAMES and len(orders) < 10:
                    orders.append(["SELL", item, int(n)])
        out = {"farmer": acts[0], "hands": acts[1:], "market": orders}
    except Exception:
        out = _safe_copy()
    return out


def sanitise(obs, action):
    """Enforce engine-facing invariants on a planned action dict."""
    if not isinstance(action, dict):
        return _safe_copy()
    me = obs["farms"][obs["player"]]
    priv = obs["private"]
    n_hands = len(me["hands"])

    def _norm(ua):
        return list(ua) if (isinstance(ua, list) and ua) else ["PASS"]

    farmer = _norm(action.get("farmer"))
    hands = [_norm(h) for h in (action.get("hands") or [])][:n_hands]
    while len(hands) < n_hands:
        hands.append(["PASS"])
    unit_acts = [farmer] + hands

    # ---- PLANT cap: > seeds held for one crop voids ALL PLANTs of that crop
    seeds = dict(priv.get("seeds", {}))
    plant_seen = {}
    for ua in unit_acts:
        if isinstance(ua, list) and ua and ua[0] == "PLANT" and len(ua) > 1:
            crop = ua[1]
            plant_seen[crop] = plant_seen.get(crop, 0) + 1
            if plant_seen[crop] > seeds.get(crop, 0):
                ua[:] = ["PASS"]

    # ---- market: SELLs first (same-turn proceeds fund purchases), <= 10,
    #      SELL quantities clamped to shed, all quantities int
    raw = [o for o in (action.get("market") or []) if isinstance(o, list) and o]
    shed = dict(priv.get("shed", {}))
    sells, others = [], []
    for o in raw:
        if o[0] == "SELL" and len(o) >= 3:
            n = min(int(o[2]), int(shed.get(o[1], 0)))
            if n > 0:
                sells.append(["SELL", o[1], n])
                shed[o[1]] = shed.get(o[1], 0) - n
        else:
            if len(o) >= 3:
                try:
                    o = [o[0], o[1], int(o[2])]
                except (TypeError, ValueError):
                    continue
            others.append(o)
    market = (sells + others)[:10]

    return {"farmer": unit_acts[0], "hands": unit_acts[1:], "market": market}


def _install_diag_dump(diag, timings):
    """`MILLRACE_DIAG=<dir>` -> dump `diag` + raw per-turn timings as JSON at
    interpreter exit.  This is the only way to read the counters back out of a
    Pit game, where every agent runs in its own child process
    (`arena/isolation.py`) and only its action crosses the pipe.  Unset on
    Kaggle, so it costs one `os.environ.get` per episode and nothing else.
    """
    out_dir = os.environ.get("MILLRACE_DIAG")
    if not out_dir:
        return

    def _dump():
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "millrace-diag-%d-%d.json" % (os.getpid(), id(diag)))
            payload = dict(diag)
            payload["timings"] = [round(t, 9) for t in timings]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception:
            pass

    atexit.register(_dump)


def make_chassis(plan_fn, state_factory=dict, deadline_frac=DEADLINE_FRAC):
    """Wrap `plan_fn(obs, state, deadline) -> action` into a safe agent.

    The returned agent takes `(obs, config=None)`.  Both real loaders slice the
    call on `co_argcount` (`kaggle_environments/agent.py:152,171`; the Pit's
    child harness at `arena/agent_child.py`), so a two-argument agent receives
    the live `configuration` and a one-argument agent does not — which is why
    `act_timeout_of` still has a vendored fallback.

    Diagnostics exposed on the returned function, beside `.timings`:
      `.plan_exceptions`  cumulative planner exceptions (C-2/R-27)
      `.first_exc`        repr + traceback tail of the first one
      `.diag`             the live counter dict (bank, latch, act_timeout, ...)

    `MILLRACE_STRICT=1` in the environment re-raises instead of masking.  It is
    absent on Kaggle (where an unmasked crash is `status ERROR`, a certain loss)
    and set by the test suite, `run_episode.py` and every Pit harness
    invocation — which is how "any run with `plan_exceptions > 0` FAILS" is
    enforced through the arena without the arena needing to know about us: a
    re-raise reaches `agent_child` as an error and the arena scores it a crash.
    """
    state = state_factory()
    timings = []
    diag = {
        "plan_exceptions": 0,
        "first_exc": None,
        "panic": False,
        "panic_turn": None,
        "panic_reason": "",
        "turns": 0,
        "act_timeout": None,
        "deadline_s": None,
        "overage_bank_start": None,
        "overage_bank_min": None,
    }

    def _latch(reason):
        if not diag["panic"]:
            diag["panic"] = True
            diag["panic_turn"] = diag["turns"]
            diag["panic_reason"] = reason
            agent.panic = True

    def agent(obs, config=None):
        t0 = time.perf_counter()
        diag["turns"] += 1

        act_timeout = act_timeout_of(config)
        deadline_s = deadline_frac * act_timeout
        diag["act_timeout"] = act_timeout
        diag["deadline_s"] = deadline_s

        bank = overage_bank_of(obs)
        if bank is not None:
            if diag["overage_bank_start"] is None:
                diag["overage_bank_start"] = bank
            if diag["overage_bank_min"] is None or bank < diag["overage_bank_min"]:
                diag["overage_bank_min"] = bank
            start = diag["overage_bank_start"]
            if start > 0.0 and bank < PANIC_BANK_FRAC * start:
                _latch("bank")

        # PANIC: skip the planner entirely.  Unlike the deleted post-hoc
        # discard, this genuinely saves the wall-clock it is protecting.
        if diag["panic"]:
            out = fallback_action(obs, latched=True)
            timings.append(time.perf_counter() - t0)
            return out

        try:
            fb = fallback_action(obs, latched=False)
        except Exception:
            fb = _safe_copy()
        try:
            out = plan_fn(obs, state, t0 + deadline_s)
            out = sanitise(obs, out)
            # NO post-hoc deadline discard (C-1/R-26): the plan is complete and
            # sanitised, the time is already spent, and emitting it is free.
        except Exception as exc:
            if os.environ.get("MILLRACE_STRICT"):
                raise
            diag["plan_exceptions"] += 1
            if diag["first_exc"] is None:
                tail = "".join(traceback.format_exc().splitlines(True)[-6:])
                diag["first_exc"] = "turn %d: %r\n%s" % (diag["turns"], exc, tail)
            agent.plan_exceptions = diag["plan_exceptions"]
            agent.first_exc = diag["first_exc"]
            if diag["plan_exceptions"] >= PANIC_EXCEPTIONS:
                _latch("exceptions")
            out = fb
        timings.append(time.perf_counter() - t0)
        return out

    agent.timings = timings
    agent.state = state  # read-only diagnostics hook (feed_audit / debug tools)
    agent.diag = diag
    agent.plan_exceptions = 0
    agent.first_exc = None
    agent.panic = False
    _install_diag_dump(diag, timings)
    return agent
