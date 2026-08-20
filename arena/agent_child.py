"""Run ONE file agent in its own interpreter (stdlib only; see isolation.py for why).

Mimics kaggle_environments' loader exactly: exec the entry file with its directory on
sys.path and take the last callable defined, passing (observation, configuration) trimmed
to the callable's co_argcount, both wrapped in a dict-with-attribute-access view (the
framework's Struct). Protocol: one JSON line {"obs":..., "config":...} per turn on stdin,
one JSON line {"action":..., "duration":...} | {"error": traceback} back on the ORIGINAL
stdout file descriptor; the agent's own prints are redirected to stderr so they can never
corrupt the protocol stream.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback


class Struct(dict):
    """Minimal stand-in for kaggle_environments.utils.Struct: dict + attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def structify(obj):
    if isinstance(obj, list):
        return [structify(v) for v in obj]
    if isinstance(obj, dict):
        return Struct((k, structify(v)) for k, v in obj.items())
    return obj


def _json_default(obj):
    item = getattr(obj, "item", None)   # numpy scalars -> native python
    if callable(item):
        return item()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"action contains a non-JSON-serializable {type(obj).__name__}")


def main() -> int:
    entry = os.path.abspath(sys.argv[1])
    # Keep a private handle on the real stdout for the protocol, then point fd 1 at stderr
    # so anything the agent prints lands in the (parent-provided) stderr log instead.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(payload: dict) -> None:
        try:
            line = json.dumps(payload, default=_json_default)
        except Exception:
            line = json.dumps({"error": "reply not JSON-serializable:\n" + traceback.format_exc()})
        proto.write(line + "\n")
        proto.flush()

    try:
        sys.path.insert(0, os.path.dirname(entry))
        with open(entry, encoding="utf-8") as f:
            src = f.read()
        env: dict = {}
        exec(compile(src, entry, "exec"), env)  # noqa: S102 -- this IS the sandbox runner
        callables = [v for v in env.values() if callable(v)]
        if not callables:
            raise RuntimeError(f"no callable defined in {entry}")
        agent = callables[-1]
    except BaseException:
        send({"ready": False, "error": traceback.format_exc()})
        return 1
    nargs = getattr(getattr(agent, "__code__", None), "co_argcount", 2) or 2
    send({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("exit"):
            break
        args = [structify(msg.get("obs")), structify(msg.get("config"))][:nargs]
        t0 = time.perf_counter()
        try:
            action = agent(*args)
            send({"action": action, "duration": round(time.perf_counter() - t0, 6)})
        except BaseException:
            send({"error": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
