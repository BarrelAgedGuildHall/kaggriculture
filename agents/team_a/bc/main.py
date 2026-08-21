"""Kaggriculture submission entry point -- Team A ensemble behaviour-cloning agent.

kaggle-environments loads an agent file by exec'ing its raw source, so `__file__` is NOT
defined here. It does append the file's directory to sys.path, which is how `features`,
`net`, `policy` and `fallback` resolve from inside the tarball; `policy.__file__` then gives
us the weights directory.

The policy object is built once at import and reused -- rebuilding it per turn would re-read
the weight files 720 times and blow the 1 s/turn budget.

Degraded mode (audit finding NB-2): if the weights cannot be loaded, or the loaded policy
raises during play, we do NOT emit PASS for the rest of the episode -- that is a guaranteed
loss dressed up as resilience. We switch, once and loudly, to the vendored weights-free v0
heuristic in `fallback.py`, which plays a real (if weaker) episode. PASS survives only as the
last resort for an observation so degenerate that even v0 cannot parse it.
"""
from __future__ import annotations

import os
import sys

try:
    import policy as _policy
except ImportError:  # not on sys.path (direct exec from an unusual cwd)
    for _cand in ("/kaggle_simulations/agent",
                  os.path.join(os.getcwd(), "agents", "team_a", "bc"),
                  os.getcwd()):
        if os.path.isfile(os.path.join(_cand, "policy.py")):
            sys.path.insert(0, _cand)
            break
    import policy as _policy

import fallback as _fallback

# 0 = pure clone, 1 = + endgame drop/liquidate, 2 = + keep-alive invariants,
# 3 = + rule-based selling. 3 is the shipped default (exp-005); lower levels exist so
# the contribution of each layer stays measurable.
_SAFE = int(os.environ.get("BC_SAFETY", "3"))

_DEGRADED = False        # latched: from here on the episode is played by the fallback
_LOAD_TRIED = 1          # loading costs ~300 ms; retry it exactly once, never per turn
_ACT_ERRORS = 0
_ACT_ERROR_LIMIT = 3     # a one-off act() failure costs one turn, not the episode


def _degrade(reason, exc):
    """Latch degraded mode and say so once, loudly, on stderr."""
    global _DEGRADED
    if not _DEGRADED:
        _DEGRADED = True
        print("TEAM_A_BC_DEGRADED reason=%s error=%s:%s -- playing the vendored v0 fallback"
              % (reason, type(exc).__name__, exc), file=sys.stderr, flush=True)


try:
    _POLICY = _policy.load_default(safety=_SAFE)
except Exception as _exc:            # noqa: BLE001 -- import must not kill the episode
    _POLICY = None
    _LOAD_TRIED = 0
    _IMPORT_ERROR = _exc


def agent(obs, config=None):
    global _POLICY, _LOAD_TRIED, _ACT_ERRORS
    if _POLICY is None and not _LOAD_TRIED:
        # One retry: a transient filesystem hiccup at import is worth a second attempt,
        # 720 of them are not.
        _LOAD_TRIED = 1
        try:
            _POLICY = _policy.load_default(safety=_SAFE)
        except Exception as exc:     # noqa: BLE001
            _degrade("weights-load", exc)
    if _POLICY is not None and not _DEGRADED:
        try:
            return _POLICY.act(obs)
        except Exception as exc:     # noqa: BLE001
            _ACT_ERRORS += 1
            if _ACT_ERRORS >= _ACT_ERROR_LIMIT:
                _degrade("policy-act", exc)
            else:
                print("TEAM_A_BC_ACT_ERROR n=%d error=%s:%s -- this turn plays the fallback"
                      % (_ACT_ERRORS, type(exc).__name__, exc), file=sys.stderr, flush=True)
    try:
        return _fallback.agent(obs, config)
    except Exception:                # noqa: BLE001 -- absolute last resort
        # Never crash the episode: the arena scores a crashed agent as a loss.
        return {"farmer": ["PASS"], "hands": [], "market": []}
