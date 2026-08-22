"""Shared pytest setup.

`MILLRACE_STRICT=1` for the whole suite (ruling-04 C-2/R-27): Team B's chassis
masks planner exceptions so that a crash on the Kaggle ladder is not
`status ERROR` (a certain loss), but that mask must never be silent inside a
measurement. Under STRICT the chassis re-raises, so a swallowed exception is a
red test instead of a full episode of PASS with a plausible-looking bank.

The two tests that exercise the mask itself delete the variable with
`monkeypatch.delenv`.
"""
import os

os.environ.setdefault("MILLRACE_STRICT", "1")
