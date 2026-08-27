"""Synthetic trace-fixture agent: emits PASS, then raises on turn three."""


def agent(obs):
    if int(obs.get("step", -1)) == 3:
        raise RuntimeError("synthetic trace fault at turn 3")
    return {"farmer": ["PASS"], "hands": [], "market": []}

