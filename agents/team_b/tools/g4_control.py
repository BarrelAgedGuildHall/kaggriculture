"""G4 hardcoded-price CONTROL agent (evaluation-only; never a submission).

The G4 robustness assay (spec section 2) compares Millrace — which reads
prices from `obs.market` every turn — against a control that does exactly what
spec section 0 rule 1 forbids: it carries a frozen replica of the 1.32.7
`MARKET_PARAMS` table and REPLACES every price the policy sees with the
replica's computed price for the live inventory. Under the pinned engine the
replica is exact, so control == Millrace bit for bit (this equality is the
divergence self-check for the assay). Under a perturbed engine
(`configuration.marketParams`) the control keeps trusting its stale table
while Millrace reads the real quotes.

The replica below is DELIBERATELY hardcoded (not imported from the engine):
importing would silently re-sync it on an engine bump, which is precisely the
failure mode the control exists to model.
"""
from __future__ import annotations

import math

PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MARKET_I0 = 10000

# frozen copy of kaggle-environments 1.32.7 MARKET_PARAMS — see module docstring
REPLICA_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def replica_price(item, inventory):
    p = REPLICA_PARAMS[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def make_control_agent(params=None):
    """routed_mixed with every observed price replaced by the replica's."""
    from millrace.routed_probe import make_routed_agent
    inner = make_routed_agent(params)

    def agent(obs):
        m = obs["market"]
        inv = {p: m["inventory"].get(p, 0) for p in REPLICA_PARAMS}
        fake = {p: replica_price(p, inv[p]) for p in REPLICA_PARAMS}
        obs2 = dict(obs)
        obs2["market"] = {"inventory": inv, "prices": fake}
        return inner(obs2)

    return agent
