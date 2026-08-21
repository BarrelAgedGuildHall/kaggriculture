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

# expB-219/220 (stock-pool saturation): the shipped build carries the melon
# race rule — structural replant window + MR-per-action stopping rule (no
# tile-count constant). FEEDCHAIN_DEFAULTS stay at the e8a76b6 behaviour so
# `millrace:feedchain` remains the frozen A/B control lineage.
#
# expB-303 (strawberry): 16 tiles of STRAWBERRY, the game's richest crop and a
# corner Team B held at zero while Team A realised ~$241/unit on it. The
# acreage is a FIXED constant and therefore a deliberate, logged §0 violation
# (issue #16 list): the section-0-clean "mr" rule was measured and lost to
# fixed-16 by 13pp head-to-head, so the constant ships until expB-304's
# dispatch-queue pricing can derive the acreage. Sweep (128 ep, seeds 0-63,
# vs Champion[B]): 8 tiles 74.2%, 16 tiles 76.6%, 24 tiles 69.5%.
_AGENT = make_feedchain_agent({"melon_replant": True, "melon_rule": "mr",
                               "strawberry_tiles": 16})


def agent(obs):
    return _AGENT(obs)
