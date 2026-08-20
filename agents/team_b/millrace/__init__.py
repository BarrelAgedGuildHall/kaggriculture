"""Millrace — Team B competition agent (v0 build).

Spec: workshop/deliverables/2026-08-19-team-b-millrace-v0-spec/millrace-v0-spec.md
Engine pin: kaggle-environments 1.32.7.

Modules (build order M0 -> M1 -> M3a/M3b first; expB-201 gates the rest):
    main.py         M0  chassis: entry, deadline, sanitiser, fallback
    market.py       M1  runtime price/drain identification (no hard-coded curve)
    layout.py       M3a siting: contiguous rings from quadrant shed-access anchors
    routing.py      M3b persistent cluster assignment + target-locked routing
    routed_probe.py     expB-201 arm: frozen Mixed probe economics + M3a/M3b
"""
