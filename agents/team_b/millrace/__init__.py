"""Millrace — Team B competition agent (v0 build).

Spec: workshop/deliverables/2026-08-19-team-b-millrace-v0-spec/millrace-v0-spec.md
Engine pin: kaggle-environments 1.32.7.

Modules:
    main.py         submission entry point: `agent(obs)` -> feedchain build
    chassis.py      M0  chassis: deadline, sanitiser, fallback (was main.py)
    market.py       M1  runtime price/drain identification (no hard-coded curve)
    layout.py       M3a siting: contiguous rings from quadrant shed-access anchors
    routing.py      M3b persistent cluster assignment + target-locked routing
    service.py      M4  service scheduler: keep-alive feed/water, FeedRouter
    feedchain.py    the champion arm: M4 + M6 sizing + M5a on the routed economy
    routed_probe.py expB-201 arm (frozen A/B control; excluded from submissions)
    probe_core.py   VENDORED copy of agents/team_b/probes/probe_core.py so the
                    directory is self-contained under the Kaggle loader
                    (byte-identity enforced by tests/test_team_b_millrace.py)

All intra-agent imports are dual-mode (package try / flat except) so the same
files work both as the package `millrace` (repo tools/tests) and exec'd flat by
the Kaggle/arena loader (dir appended to sys.path, no package context).
"""
