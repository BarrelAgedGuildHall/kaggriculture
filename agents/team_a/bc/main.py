"""Kaggriculture submission entry point -- Team A ensemble behaviour-cloning agent.

kaggle-environments loads an agent file by exec'ing its raw source, so `__file__` is NOT
defined here. It does append the file's directory to sys.path, which is how `features`,
`net` and `policy` resolve from inside the tarball; `policy.__file__` then gives us the
weights directory.

The policy object is built once on first call and reused -- rebuilding it per turn would
re-read the weight files 720 times and blow the 1 s/turn budget.
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

# 0 = pure clone, 1 = + endgame drop/liquidate, 2 = + keep-alive invariants,
# 3 = + rule-based selling. 3 is the shipped default (exp-005); lower levels exist so
# the contribution of each layer stays measurable.
_SAFE = int(os.environ.get("BC_SAFETY", "3"))

# Build the policy at IMPORT time, not on the first act. Loading three ensemble members costs
# ~300 ms; done lazily that lands entirely inside turn 0 and was the only thing pushing the
# worst measured turn over the arena's 400 ms ceiling (p99 stays ~20 ms). Both the real
# kaggle-environments loader and arena/agent_child.py import the module before they start
# timing turns, so paying it here is both cheaper and a more honest measurement of the
# per-turn decision cost. Failure is deferred to the first call rather than killing the
# import, so a weights problem still surfaces as a playable fallback instead of a crash.
try:
    _POLICY = _policy.load_default(safety=_SAFE)
except Exception:
    _POLICY = None


def agent(obs, config=None):
    global _POLICY
    if _POLICY is None:
        _POLICY = _policy.load_default(safety=_SAFE)
    try:
        return _POLICY.act(obs)
    except Exception:
        # Never crash the episode: the arena scores a crashed agent as a loss.
        return {"farmer": ["PASS"], "hands": [], "market": []}
