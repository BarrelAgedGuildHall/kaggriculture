"""Invariant tests for the Millrace M0 chassis, M1 market model and M3b router."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))

from millrace.chassis import fallback_action, make_chassis, sanitise  # noqa: E402
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


# ------------------------------------------------- market M1 v2 estimators
def _feed_day(m, day, inv_series, orders_series=None, shops=()):
    """Feed one day of synthetic EGG observations (24 turns)."""
    obs = _obs()
    for h in range(24):
        obs = dict(obs)
        obs["day"], obs["hour"] = day, h
        obs["market"] = {"inventory": {"EGG": inv_series[h]}, "prices": {"EGG": 40}}
        obs["town"] = {"unlocked_shops": list(shops)}
        m.observe(obs, (orders_series[h] if orders_series else None))


def test_market_drain_daily_balance_tick_aligned():
    m = MarketModel()
    # drain of 2 lands at obs steps s where (s-1) % 4 == 0 (engine consumes at
    # engine-step s-1); no trades. 6 ticks/day -> drain 12/day.
    inv = []
    cur = 10000
    for day in range(2):
        for h in range(24):
            s = day * 24 + h
            if s > 0 and (s - 1) % 4 == 0:
                cur -= 2
            inv.append(cur)
    _feed_day(m, 0, inv[:24])
    _feed_day(m, 1, inv[24:])
    est, n = m.drain_live("EGG")
    assert m.tick_ok
    assert n == 1
    assert abs(est - 12.0) < 1e-6


def test_market_drain_sees_through_our_sales():
    m = MarketModel()
    # we sell 1 EGG every turn; drain 2 on tick turns. Net dI: +1 off-tick,
    # -1 on tick. Balance must still find drain = 12/day.
    inv, cur = [], 10000
    orders = []
    for day in range(2):
        for h in range(24):
            s = day * 24 + h
            if s > 0:
                cur += 1  # our sale from last turn
                if (s - 1) % 4 == 0:
                    cur -= 2
            inv.append(cur)
            orders.append([["SELL", "EGG", 1]])
    _feed_day(m, 0, inv[:24], orders[:24])
    _feed_day(m, 1, inv[24:], orders[24:])
    est, n = m.drain_live("EGG")
    assert m.tick_ok
    assert abs(est - 12.0) < 1e-6


def test_market_tick_selfcheck_flips_on_offtick_decrement():
    m = MarketModel()
    # EGG (non-buyable) decrement on an off-tick turn with no sale of ours
    inv = [10000] * 24
    for h in range(3, 24):
        inv[h] = 9999  # decrement lands at h=3, an off-tick obs step
    _feed_day(m, 0, inv)
    assert not m.tick_ok


def test_market_shop_unlock_resets_drain_window():
    m = MarketModel()
    inv = [10000] * 24
    _feed_day(m, 0, inv)
    _feed_day(m, 1, inv)  # closes day 0 -> EGG has a sample
    assert m.drain_live("EGG")[1] >= 1
    obs = dict(_obs())
    obs["day"], obs["hour"] = 1, 5
    obs["market"] = {"inventory": {"EGG": 10000}, "prices": {"EGG": 40}}
    obs["town"] = {"unlocked_shops": ["BAKERY"]}  # sells EGG -> stale window
    m.observe(obs)
    assert m.drain_live("EGG") == (None, 0) or m.drain_live("EGG")[1] == 0


def test_market_price_table_interpolates_observed_curve():
    m = MarketModel()
    obs = _obs()
    for inv, price in [(100, 50), (104, 42)]:
        obs = dict(obs)
        obs["market"] = {"inventory": {"EGG": inv}, "prices": {"EGG": price}}
        m.observe(obs)
    assert abs(m.price_at("EGG", 102) - 46.0) < 1e-9   # interpolated
    assert m.price_at("EGG", 100) == 50.0              # exact


# ------------------------------------------------- service M4 scheduler
def _animal(fed=False, cared=False, unfed_days=0, fert=False, yld=0, kind="GOOSE"):
    return {"kind": "COOP" if kind == "GOOSE" else "PASTURE", "animal": kind,
            "fed_today": fed, "cared_today": cared, "consecutive_unfed": unfed_days,
            "fertilizer_available": fert, "yield_units": yld, "placed_day": 0}


def test_service_feed_rationing_criticals_always_eat():
    from millrace.service import animal_tasks, URGENT_FEED
    animals = [((0, 0), _animal(unfed_days=1)),   # critical
               ((1, 0), _animal()), ((2, 0), _animal())]  # routine
    P = {"care": True}
    tasks = animal_tasks(animals, wheat_stock=2, P=P, hour=5, last_day_soon=False)
    feeds = [t for t in tasks if t[2] == ["FEED"]]
    # wheat 2, criticals 1 -> 1 critical + 1 routine task, second routine starves
    assert len(feeds) == 2
    assert feeds[0][0] == URGENT_FEED or any(t[0] == URGENT_FEED for t in feeds)


def test_service_care_only_when_fed():
    from millrace.service import animal_tasks
    animals = [((0, 0), _animal(fed=True)), ((1, 0), _animal(fed=False))]
    tasks = animal_tasks(animals, wheat_stock=0, P={"care": True}, hour=5, last_day_soon=False)
    cares = [t for t in tasks if t[2] == ["CARE"]]
    assert len(cares) == 1
    assert cares[0][1] == (0, 0)  # only the fed animal gets CARE


def test_feedrouter_skips_unsuppliable_tasks():
    from millrace.service import FeedRouter
    r = FeedRouter()
    units = [(2, 2)]
    busy = [False]
    actions = [["PASS"]]
    # FEED requires WHEAT; the unit carries none and the shed is empty ->
    # the task must not be picked (the base router churned a claim on it)
    tasks = [(1, (1, 1), ["FEED"], ("F", 1, 1))]
    ctx = _ctx([{}], {"WHEAT": 0})
    r.plan(0, units, busy, actions, tasks, ctx)
    assert actions[0] == ["PASS"] and not busy[0]


def test_feedrouter_borrows_urgent_cross_cluster():
    from millrace.service import FeedRouter
    r = FeedRouter()
    units = [(2, 2)]  # NW
    busy = [False]
    actions = [["PASS"]]
    # home (NW) has only routine work it cannot supply; NE has an urgent FEED
    # and the unit carries wheat -> it must leave home
    tasks = [(0.25, (7, 1), ["FEED"], ("F", 7, 1))]
    ctx = _ctx([{"WHEAT": 3}], {"WHEAT": 0}, quads=("NW", "NE"))
    r.home = {0: "NW"}
    r.day = 0
    r.plan(0, units, busy, actions, tasks, ctx)
    assert busy[0] and actions[0] != ["PASS"]


# ------------------------------------------------- submission packaging
MILLRACE_DIR = REPO_ROOT / "agents" / "team_b" / "millrace"
ROUTED_DIR = REPO_ROOT / "agents" / "team_b" / "routed"


def test_vendored_probe_core_is_byte_identical():
    canonical = (REPO_ROOT / "agents" / "team_b" / "probes" / "probe_core.py").read_bytes()
    assert (MILLRACE_DIR / "probe_core.py").read_bytes() == canonical


def test_frozen_routed_snapshot_matches_millrace_modules():
    # The expB-201 Pit snapshot is byte-identical to its millrace counterparts.
    # If millrace/* diverge later (post-freeze refactors), drop this test and
    # leave agents/team_b/routed/ untouched -- it is a frozen champion.
    for name in ("chassis.py", "layout.py", "routing.py", "routed_probe.py", "probe_core.py"):
        assert (ROUTED_DIR / name).read_bytes() == (MILLRACE_DIR / name).read_bytes(), name


def _flat_exec_agent(agent_dir):
    """Mimic kaggle_environments.get_last_callable exactly: main.py exec'd into
    a PLAIN dict (no __name__/__package__) with the dir appended to sys.path
    only during the exec (popped before the agent is ever called), then the
    last callable is the agent."""
    src = (agent_dir / "main.py").read_text(encoding="utf-8")
    before = set(sys.modules)
    try:
        ns = {}
        sys.path.append(str(agent_dir))
        try:
            exec(compile(src, str(agent_dir / "main.py"), "exec"), ns)
        finally:
            sys.path.remove(str(agent_dir))  # the loader pops before calling
        fn = [v for v in ns.values() if callable(v)][-1]
        assert fn.__name__ == "agent"
        out = fn(_obs())  # dir no longer importable: imports must have been eager
        assert isinstance(out, dict) and "farmer" in out and "market" in out
        return out
    finally:
        for m in set(sys.modules) - before:
            del sys.modules[m]


def test_kaggle_loader_flat_exec_millrace_entry():
    _flat_exec_agent(MILLRACE_DIR)


def test_kaggle_loader_flat_exec_routed_entry():
    _flat_exec_agent(ROUTED_DIR)


# ------------------------------------------------- expB-219/220 melon race rule

def test_shipped_entry_carries_melon_race_rule():
    # the submission entry constructs the expB-220 build; FEEDCHAIN_DEFAULTS
    # stay at the e8a76b6 behaviour (frozen A/B control lineage)
    src = (MILLRACE_DIR / "main.py").read_text(encoding="utf-8")
    assert '"melon_replant": True' in src and '"melon_rule": "mr"' in src
    from millrace.feedchain import FEEDCHAIN_DEFAULTS
    assert FEEDCHAIN_DEFAULTS["melon_replant"] is False
    assert FEEDCHAIN_DEFAULTS["melon_rule"] == "fixed"


def test_melon_mr_rule_prices_tiles_from_market():
    # a rich observed melon quote must open acreage; the rule output is a
    # runtime quantity, never the melon_tiles constant
    from millrace.feedchain import make_feedchain_agent
    ag = make_feedchain_agent({"melon_replant": True, "melon_rule": "mr"})
    obs = _obs(seeds={"WHEAT": 1, "CARROT": 0, "MELON": 2})
    obs["market"] = {"prices": {"EGG": 30, "MELON": 240, "WHEAT": 25},
                    "inventory": {"EGG": 100, "MELON": 5, "WHEAT": 50}}
    out = ag(obs)
    assert isinstance(out, dict) and "market" in out
    st = ag.state["st"]
    assert st["want_melon"] > 6  # rich curve: beyond the frozen control's 6


def test_melon_mr_rule_closes_on_dead_pool():
    # a floored melon quote must close acreage growth (replant stops when the
    # marginal cycle prices below lambda)
    from millrace.feedchain import make_feedchain_agent
    ag = make_feedchain_agent({"melon_replant": True, "melon_rule": "mr"})
    # realistic crew: lambda = fib(11)/prod_rate ~ $14/action, not the ~$0.3
    # of a 2-hand toy farm (the rule is a ratio, so the test must price both
    # sides realistically)
    obs = _obs(n_hands=10, seeds={"WHEAT": 1, "CARROT": 0, "MELON": 2})
    obs["market"] = {"prices": {"EGG": 30, "MELON": 1, "WHEAT": 25},
                    "inventory": {"EGG": 100, "MELON": 400, "WHEAT": 50}}
    ag(obs)
    assert ag.state["st"]["want_melon"] == 0
