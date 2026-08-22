"""Agent-spec resolution shared by the measurement tools in `tools/`.

One spec grammar for every team, so a census run can put a Team A and a Team B build on the
same page:

    pass | random | starter                 built-in engine agents
    path/to/agent.py                        a file exposing `agent(obs)`
    path/to/dir                             a directory with `main.py` (Team A's packaged form)
    probe:<PresetName> | probe:{json}       Team B probe core (agents/team_b/probes/probe_core.py)
    millrace:routed_mixed | millrace:feedchain | millrace:feedchain{json} | millrace:{json}

Repo-relative paths are resolved against the repo root, so a spec reads the same from any
working directory.

The probe:/millrace: branches mirror `agents/team_b/tools/instrument.py`, which keeps its own
copy: that file is Team B's episode tracer and is not shared tooling. This module is the
shared one, and it is the only loader the tools in `tools/` use.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BUILTINS = ("pass", "random", "starter")


def _team_b_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath("agents", "team_b", *parts))


def load_agent(spec: str):
    """Resolve a spec to a callable `agent(obs)` (or `agent(obs, config)` -- see `arity_wrap`)."""
    if spec.startswith("probe:"):
        sys.path.insert(0, _team_b_path("probes"))
        from probe_core import PRESETS, make_agent  # noqa: PLC0415

        arg = spec[len("probe:"):]
        return make_agent(json.loads(arg) if arg.startswith("{") else PRESETS[arg])
    if spec.startswith("millrace:"):
        sys.path.insert(0, _team_b_path())
        from millrace.feedchain import make_feedchain_agent  # noqa: PLC0415
        from millrace.routed_probe import make_routed_agent  # noqa: PLC0415

        arg = spec[len("millrace:"):]
        if arg == "routed_mixed":
            return make_routed_agent()
        if arg == "feedchain":
            return make_feedchain_agent()
        if arg.startswith("feedchain{"):
            return make_feedchain_agent(json.loads(arg[len("feedchain"):]))
        if arg.startswith("{"):
            return make_routed_agent(json.loads(arg))
        raise ValueError(f"unknown millrace spec: {spec!r}")
    if spec in BUILTINS:
        import kaggle_environments.envs.kaggriculture.kaggriculture as K  # noqa: PLC0415

        return {"pass": K.pass_agent, "random": K.random_agent, "starter": K.starter_agent}[spec]

    p = Path(spec)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if p.is_dir():
        p = p / "main.py"
    if not p.exists():
        raise FileNotFoundError(f"agent spec {spec!r} resolved to {p}, which does not exist")
    sys.path.insert(0, str(p.parent))
    mod_spec = importlib.util.spec_from_file_location(f"agent_{p.stem}_{abs(hash(str(p)))}", p)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod.agent


def arity_wrap(fn, config):
    """Match the kaggle loader: a 2-arg agent callable receives (obs, config)."""
    try:
        n = fn.__code__.co_argcount
    except AttributeError:
        return fn
    if n >= 2:
        return lambda obs: fn(obs, config)
    return fn


def curve_value(K, market_inv, item, n) -> float:
    """Revenue of selling n units of `item` into the current market, walking the sell curve."""
    inv = market_inv.get(item, 10000)
    total = 0.0
    for _ in range(int(n)):
        p = K.market_price(item, inv)
        total += p
        if p > 1:
            inv += 1
    return total
