"""Marks `agents/` as a *regular* package. Do not delete -- this file is load-bearing.

Without it `agents` is a namespace package, and Python only falls back to a namespace
portion when no regular module of that name exists anywhere on `sys.path`. `kaggle_environments`
puts its bundled environment directories on `sys.path`, and one of them contains
`lux_ai_s3/agents.py` -- a regular module, which therefore wins.

In a normal process nothing notices, because our `agents` package is imported before the
engine. It bites in the arena's *spawn* workers: the child inherits the parent's polluted
`sys.path` and re-imports the driver from scratch, so `from agents.team_a...` resolves to
Lux AI's `agents.py`, which raises `ImportError: attempted relative import with no known
parent package`. Symptom seen 2026-08-21: every worker died during `__main__` fixup, the
scheduler sat at 0/16 games with no live workers and 0.5% CPU -- a silent hang, not a crash.
"""
