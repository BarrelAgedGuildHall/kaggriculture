"""Invariant tests for the Millrace M0 chassis, M1 market model and M3b router."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))

from millrace.main import fallback_action, make_chassis, sanitise  # noqa: E402
from millrace.market import MarketModel  # noqa: E402
from millrace.routing import Router  # noqa: E402


def _obs(n_hands=2, seeds=None, shed=None):
    return {
        "player": 0,
        "day": 1, "hour": 3,
        "farms": [{
            "farmer": [4, 4],
            "hands": [[1, 1]] * n_hands,
            "tiles": [[None] * 10 for _ in range(10)],
            "unlocked_quadrants": ["NW"],
            "money": 1000.0,
            "hires_today": n_hands,
        }, {}],
        "private": {
            "shed": shed or {"WHEAT": 5, "EGG": 3},
            "seeds": seeds or {"WHEAT": 1, "CARROT": 0, "MELON": 0},
            "inventories": [{} for _ in range(n_hands + 1)],
        },
        "market": {"prices": {"EGG": 30}, "inventory": {"EGG": 100}},
        "town": {"unlocked_shops": []},
    }


# ---------------------------------------------------------------- sanitiser
def test_sanitise_plant_cap_voids_excess_plants():
    obs = _obs(n_hands=2, seeds={"WHEAT": 1})
    act = {"farmer": ["PLANT", "WHEAT"],
           "hands": [["PLANT", "WHEAT"], ["PASS"]],
           "market": []}
    out = sanitise(obs, act)
    plants = [a for a in [out["farmer"]] + out["hands"] if a[0] == "PLANT"]
    assert len(plants) == 1  # second PLANT (over the 1 seed held) stripped


def test_sanitise_sells_before_buys_and_cap_10():
    obs = _obs(shed={"EGG": 50, "WHEAT": 5})
    market = [["BUY_PRODUCT", "WHEAT", 3]] + [["SELL", "EGG", 2]] * 4
    out = sanitise(obs, {"farmer": ["PASS"], "hands": [], "market": market})
    kinds = [o[0] for o in out["market"]]
    assert kinds.index("BUY_PRODUCT") > max(i for i, k in enumerate(kinds) if k == "SELL")
    # over-long lists are truncated at 10, SELLs surviving preferentially
    market12 = [["BUY_PRODUCT", "WHEAT", 3]] + [["SELL", "EGG", 2]] * 12
    out12 = sanitise(obs, {"farmer": ["PASS"], "hands": [], "market": market12})
    assert len(out12["market"]) == 10
    assert all(o[0] == "SELL" for o in out12["market"])


def test_sanitise_clamps_sell_to_shed():
    obs = _obs(shed={"EGG": 3})
    out = sanitise(obs, {"farmer": ["PASS"], "hands": [],
                         "market": [["SELL", "EGG", 99]]})
    assert out["market"] == [["SELL", "EGG", 3]]


def test_sanitise_pads_and_truncates_hands():
    obs = _obs(n_hands=2)
    out = sanitise(obs, {"farmer": ["PASS"], "hands": [["PASS"]] * 5, "market": []})
    assert len(out["hands"]) == 2


def test_chassis_swallows_planner_exceptions():
    agent = make_chassis(lambda obs, st, dl: 1 / 0)
    out = agent(_obs())
    assert out["farmer"] == ["PASS"] or out["farmer"] == ["DROP"]
    assert isinstance(out["market"], list)


def test_fallback_sells_shed_only():
    out = fallback_action(_obs(shed={"EGG": 4, "GOOSE": 2}))
    assert ["SELL", "EGG", 4] in out["market"]
    assert all(o[1] != "GOOSE" for o in out["market"])


# ---------------------------------------------------------------- market M1
def test_market_marginal_revenue_walks_measured_slope():
    m = MarketModel()
    # feed synthetic samples: price falls 2 per unit of inventory
    obs = _obs()
    for inv, price in [(100, 50), (101, 48), (102, 46), (103, 44)]:
        obs["market"] = {"inventory": {"EGG": inv}, "prices": {"EGG": price}}
        m.observe(obs)
    assert m.quote("EGG") == 44
    assert abs(m.dPdI("EGG") + 2.0) < 1e-9
    # 3 units: 44 + 42 + 40
    assert abs(m.marginal_revenue("EGG", 3) - 126.0) < 1e-9


def test_market_drain_prior_from_shops():
    m = MarketModel()
    obs = _obs()
    obs["town"] = {"unlocked_shops": ["PET_CAFE", "BAKERY"]}
    assert m.drain_per_day("CARROT", obs) == 13  # centre 1 + single-product 12
    assert m.drain_per_day("EGG", obs) == 7      # centre 1 + multi-product 6
    assert m.drain_per_day("FERTILIZER", obs) == 0


# ---------------------------------------------------------------- router M3b
def _ctx(invs, shed, quads=("NW",)):
    from millrace.layout import ANCHORS, dist

    def step_toward(pos, tgt):
        x, y = pos
        tx, ty = tgt
        if abs(tx - x) >= abs(ty - y) and tx != x:
            return "EAST" if tx > x else "WEST"
        if ty != y:
            return "SOUTH" if ty > y else "NORTH"
        return "EAST" if tx > x else "WEST" if tx < x else "PASS"

    return {
        "invs": invs, "shed": shed, "quads": list(quads),
        "step_toward": step_toward,
        "shed_tiles": [(4, 4), (5, 4), (4, 5), (5, 5)],
        "nearest_shed": lambda pos: min([(4, 4), (5, 4), (4, 5), (5, 5)],
                                        key=lambda t: dist(pos, t)),
        "requirement": lambda act: {"FEED": "WHEAT"}.get(act[0]),
        "batch": {"WHEAT": 8},
        "claimed": set(),
    }


def test_router_locks_target_across_turns():
    r = Router()
    units = [(0, 0)]
    tasks = [(6, (3, 0), ["CARE"], ("C", 3, 0)), (6, (0, 3), ["CARE"], ("C", 0, 3))]
    actions = [["PASS"]]
    r.plan(0, units, [False], actions, tasks, _ctx([{}], {}))
    first = actions[0]
    assert first[0] in ("EAST", "SOUTH")
    key0 = r.locks[0]["key"]
    # unit moved one step; both tasks still pending -> keeps the SAME target
    units2 = [(1, 0) if first[0] == "EAST" else (0, 1)]
    actions2 = [["PASS"]]
    r.plan(0, units2, [False], actions2, tasks, _ctx([{}], {}))
    assert r.locks[0]["key"] == key0


def test_router_urgent_preempts_locked_care():
    r = Router()
    units = [(0, 0)]
    care = [(6, (3, 0), ["CARE"], ("C", 3, 0))]
    actions = [["PASS"]]
    r.plan(0, units, [False], actions, care, _ctx([{}], {}))
    assert r.locks[0]["key"] == ("C", 3, 0)
    # a FEED (prio 1, keep-alive band) appears: takes over
    tasks = care + [(1, (0, 2), ["FEED"], ("F", 0, 2))]
    actions2 = [["PASS"]]
    r.plan(0, units, [False], actions2, tasks, _ctx([{"WHEAT": 2}], {}))
    assert r.locks[0]["key"] == ("F", 0, 2)


def test_router_routes_via_shed_for_missing_requirement():
    r = Router()
    units = [(0, 0)]
    tasks = [(1, (2, 0), ["FEED"], ("F", 2, 0))]
    actions = [["PASS"]]
    # no wheat carried, wheat in shed -> walk toward shed, not the task
    r.plan(0, units, [False], actions, tasks, _ctx([{}], {"WHEAT": 10}))
    assert actions[0][0] in ("EAST", "SOUTH")  # heading to (4,4), not (2,0) straight
    # standing ON a shed tile -> batched PICKUP
    r2 = Router()
    actions2 = [["PASS"]]
    r2.plan(0, [(4, 4)], [False], actions2, tasks, _ctx([{}], {"WHEAT": 10}))
    assert actions2[0] == ["PICKUP", "WHEAT", 8]


def test_router_rebinds_on_new_day():
    r = Router()
    tasks = [(6, (1, 1), ["CARE"], ("C", 1, 1))]
    r.plan(0, [(0, 0)], [False], [["PASS"]], tasks, _ctx([{}], {}))
    assert r.locks
    r.plan(1, [(0, 0)], [False], [["PASS"]], tasks, _ctx([{}], {}))
    assert r.day == 1
