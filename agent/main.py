"""Kaggler's Kaggriculture agent -- submission entry point.

Kaggle imports this file and calls `agent(obs)` once per turn (720 turns, 1 s each).
Keep this file self-contained: a single-file submission is `main.py` alone; a multi-file
submission is a tar.gz with `main.py` at the root (see docs/engine/AGENTS.md).

This is the placeholder baseline: it never breaks the rules and never spends money
(farmer PASS, no hands, no market orders), so it finishes with exactly startingMoney.
The Modeler replaces the policy; keep the `agent(obs)` signature and return shape.
"""


def agent(obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}
