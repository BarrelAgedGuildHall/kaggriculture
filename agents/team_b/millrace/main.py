"""Millrace M0 — chassis: entry point, per-turn deadline, sanitiser, fallback.

Hard constraints honoured here (spec section 0):
  - global try/except -> {"farmer": ["PASS"], "hands": [], "market": []}
  - hard per-turn deadline (~250 ms) with a precomputed fallback action
  - PLANT sanitiser: if a turn's PLANT requests for one crop exceed seeds held,
    the engine drops EVERY PLANT for that crop that turn -> cap emitted PLANTs
  - market order hygiene: <= 10 orders/turn, SELLs before BUYs, quantities
    clamped to shed contents / int-coerced
  - stdlib only, no unbounded loops

`make_chassis(plan_fn)` wraps any planner `plan_fn(obs, state, deadline) ->
action dict` into a safe agent. The planner may be sloppy; the chassis is not.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import time

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
# animals never SELL from the shed via this sanitiser guard (they are placed,
# not sold, in every Millrace plan); crops with seeds live in private.seeds
ANIMAL_NAMES = ("GOOSE", "COW", "SHEEP")

SAFE = {"farmer": ["PASS"], "hands": [], "market": []}


def _safe_copy():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def fallback_action(obs):
    """Precomputed no-planner fallback: DROP if carrying on a shed tile, else
    PASS; market = sell shed contents (SELL only, <= 10 orders)."""
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


def make_chassis(plan_fn, state_factory=dict, deadline_s=0.25):
    """Wrap `plan_fn(obs, state, deadline) -> action` into a safe agent.

    The returned agent exposes `.timings` (per-turn planning seconds) so the
    census/timing tools can report p99 without patching the engine.
    """
    state = state_factory()
    timings = []

    def agent(obs):
        t0 = time.perf_counter()
        try:
            fb = fallback_action(obs)
        except Exception:
            fb = _safe_copy()
        try:
            out = plan_fn(obs, state, t0 + deadline_s)
            out = sanitise(obs, out)
            if time.perf_counter() - t0 > deadline_s:
                out = fb  # blew the budget: emit the precomputed safe action
        except Exception:
            out = fb
        timings.append(time.perf_counter() - t0)
        return out

    agent.timings = timings
    agent.state = state  # read-only diagnostics hook (feed_audit / debug tools)
    return agent
