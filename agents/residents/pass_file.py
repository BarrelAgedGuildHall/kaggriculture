"""Resident: PASS agent (never acts; the floor of the scale). Registered as pass-file."""


def agent(obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}
