"""Millrace — Team B submission entry point (build: millrace:feedchain).

Kaggle (and arena/agent_child.py, which mimics it) loads a directory agent by
exec'ing main.py with the directory APPENDED to sys.path and no package
context, then calls the last callable defined here.  Two packaging rules
follow (docs/engine/AGENTS.md; PR #11 audit finding on loader parity):

  - every module this agent needs lives in THIS directory under flat,
    collision-unlikely names (`chassis`, `feedchain`, `layout`, `market`,
    `routing`, `service`, `probe_core`) — `probe_core` is vendored from
    agents/team_b/probes/ (byte-identity enforced by
    tests/test_team_b_millrace.py);
  - main.py itself stays a thin entry: because the directory is *appended*,
    a heavyweight module here named like anything earlier on sys.path would
    silently lose the import race.

The agent: M0 chassis (deadline + sanitiser + fallback) + M4 service
scheduler + M6 wheat sizing + M5a lambda on the routed Mixed economy —
Team B champion per expB-213 (G1 95.3%/93.8% vs the expB-201 arm).

Engine: kaggle-environments 1.32.7.
"""

# EAGER flat import: kaggle_environments.get_last_callable appends this dir to
# sys.path only WHILE exec'ing this file (it pops it before the first agent
# call), so all sibling imports must happen now, at module-exec time.  Flat
# (not relative) because the loader execs into a plain dict with no
# __name__/__package__, where a relative import raises KeyError.  This entry
# is never imported as a package module (tools/tests use millrace.feedchain).
from feedchain import make_feedchain_agent

_AGENT = make_feedchain_agent()


def agent(obs):
    return _AGENT(obs)
