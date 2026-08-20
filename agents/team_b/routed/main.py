"""Frozen expB-201 Team B champion (millrace:routed_mixed) — Pit registration snapshot.

This directory exists so the expB-201 champion can be registered in the arena
as a self-contained directory agent (the loader execs main.py flat, dir on
sys.path).  chassis/layout/routing/routed_probe/probe_core are byte-identical
copies of their agents/team_b/millrace counterparts at the feedchain packaging
commit (asserted by tests/test_team_b_millrace.py); the policy is the frozen
expB-201 final arm — iter1 constants, Mixed probe economy from main@31ff526,
M3a siting + M3b routing.  FROZEN: never edit these files; if the millrace
copies diverge later, drop the identity test, not this snapshot.

Engine: kaggle-environments 1.32.7.
"""

# EAGER flat import: the loader appends this dir to sys.path only WHILE
# exec'ing this file (popped before the first agent call), and execs into a
# plain dict with no __name__/__package__ (relative imports raise KeyError).
from routed_probe import make_routed_agent

_AGENT = make_routed_agent()


def agent(obs):
    return _AGENT(obs)
