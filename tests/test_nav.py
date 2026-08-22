"""Acceptance suite for `nav` (S-1), spec §6.

The two tests that matter most, and why they are built the way they are:

* **The terminal identity (T1)** is asserted on a REAL episode replayed through the library:
  every non-cash line must be exactly zero at snapshot 719, the first index at which no
  action can be taken. It is worth something only because the companion static check (T1b)
  proves there is no `if last_day:` branch in any mark -- otherwise the identity would be
  true by construction of the wrong kind, and unfalsifiable.
* **The engine-bump tripwire (T4)** is the standing guard on plan §1 row 16. It has three
  parts: nav re-derives PERTURBED curves it was never given; the tripwire stays quiet while
  it does (no crying wolf); and a curve outside the five shape families makes it FIRE. A
  tripwire that has never been seen to fire is not a tripwire.

Run: `uv run pytest -q tests/test_nav.py`
Single-command engine-bump check: `uv run python scripts/nav_tripwire.py`
"""
from __future__ import annotations

import ast
import copy
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

from _engine import ENV_NAME, import_engine  # noqa: E402

import nav  # noqa: E402
from nav import curves as navcurves  # noqa: E402
from nav.rules import TERMINAL_INDEX, TURNS_PER_DAY  # noqa: E402

NAV_DIR = REPO / "nav"
RUNTIME_MODULES = [NAV_DIR / name for name in nav.RUNTIME_MODULES]
PERTURB = REPO / "agents" / "team_b" / "tools" / "g4_perturb.json"
STARTER = "starter"


def engine():
    return import_engine().envs.kaggriculture.kaggriculture


def _perturb_params() -> dict:
    return {k: v for k, v in json.loads(PERTURB.read_text(encoding="utf-8")).items()
            if not k.startswith("_")}


def _play(agents, seed: int, params: dict | None = None, steps: int = 720):
    ke = import_engine()
    cfg = {"episodeSteps": steps, "seed": seed}
    if params:
        cfg["marketParams"] = params
    env = ke.make(ENV_NAME, configuration=cfg, debug=False)
    env.run(agents)
    return env


@pytest.fixture(scope="module")
def episode():
    """One ordinary episode, both seats' observations. `starter` actually farms, so the
    terminal identity has something to be non-trivially zero about."""
    return _play([STARTER, STARTER], seed=7)


@pytest.fixture(scope="module")
def perturbed_episode():
    """The same, with the g4 perturbation applied through the engine's own marketParams
    override: amplitudes +-30% and MELON's glut shape swapped sq -> linear."""
    return _play([STARTER, STARTER], seed=7, params=_perturb_params())


def observations(env, seat: int):
    return [env.steps[i][seat].observation for i in range(len(env.steps))]


# ------------------------------------------------------------------ rulebook (bump tripwire c)
def test_rules_match_the_engine():
    """`nav.rules` is inlined because a submission cannot import the engine. If a release
    changes a table, this fails loudly instead of silently re-pricing the balance sheet."""
    K = engine()
    assert dict(nav.rules.CROPS) == K.CROPS
    assert dict(nav.rules.ANIMALS) == K.ANIMALS
    assert {k: list(v) for k, v in nav.rules.SHOPS.items()} == K.SHOPS
    assert list(nav.rules.PRODUCTS) == K.PRODUCTS
    assert list(nav.rules.TOWN_CENTER_PRODUCTS) == K.TOWN_CENTER_PRODUCTS
    assert nav.rules.PRICE_FLOOR == K.PRICE_FLOOR
    assert nav.rules.MARKET_I0_FALLBACK == K.MARKET_I0
    assert nav.rules.HINGE_GAIN == K.HINGE_GAIN
    assert nav.rules.MAX_SHOP_INSTANCES == K.MAX_SHOP_INSTANCES
    assert nav.rules.ENGINE_VERSION == import_engine().__version__


def test_shape_matches_the_engine():
    K = engine()
    for func in nav.rules.SHAPES:
        for x in (0, 1, 7, 100, 399, 400, 401, 1000):
            for T in (100, 200, 450):
                assert nav.shape(func, x, T) == pytest.approx(K._shape(func, x, T), rel=1e-12)


def test_shape_integral_matches_numeric_quadrature():
    """The closed forms in `shape_integral` are the whole point of the O(1) marginal
    revenue; check each against a fine Riemann sum."""
    for func in nav.rules.SHAPES:
        for T in (100.0, 300.0):
            for x in (5.0, 137.0, 640.0):
                n = 4000
                h = x / n
                approx = sum(nav.shape(func, (i + 0.5) * h, T) for i in range(n)) * h
                assert nav.shape_integral(func, x, T) == pytest.approx(approx, rel=2e-4)


# ---------------------------------------------------------------------- the closed-form integral
def _walk_revenue(K, product, inv, n, params=None):
    """The reference per-unit walk (`instrument.curve_value`), i.e. what the engine does."""
    total = 0
    for _ in range(int(n)):
        p = K.market_price(product, inv, params)
        total += p
        if p > 1:
            inv += 1
    return total


def test_closed_form_revenue_matches_the_engine_walk():
    """T4's revenue gate, on exact curves: the closed form replaces the walk, so it has to
    agree with it. Measured worst case 0.26% at n = 900, 0.15% at n <= 60."""
    K = engine()
    cs = nav.CurveSet.from_params(K.MARKET_PARAMS)
    worst = 0.0
    for product in K.PRODUCTS:
        for inv in (9000, 9800, 10000, 10200, 11000, 12000, 20000):
            for n in (1, 2, 5, 20, 60, 200, 900):
                true = _walk_revenue(K, product, inv, n)
                got = cs[product].marginal_revenue(inv, n)
                worst = max(worst, abs(got - true) / max(1.0, true))
    assert worst < 0.01, f"closed-form revenue drifted from the engine walk: {worst:.4%}"


def test_marginal_cost_follows_the_engine_quote():
    """BUY_PRODUCT quotes at `inventory - 1` and then decrements (the probe was one unit
    short of that; the engine wins)."""
    K = engine()
    cs = nav.CurveSet.from_params(K.MARKET_PARAMS)
    for product in ("WHEAT", "FERTILIZER"):
        for inv0 in (9500, 10000, 10500):
            for n in (1, 5, 30, 120):
                inv, true = inv0, 0
                for _ in range(n):
                    true += K.market_price(product, inv - 1)
                    inv -= 1
                got = cs[product].marginal_cost(inv0, n)
                assert got == pytest.approx(true, rel=0.01)


def test_floored_pool_units_do_not_move_the_market():
    """`_commit_unit`: a sale at $1 does not increase supply, so an over-sold pool pays $1 a
    unit forever. Row 12 falls out of that rather than out of a constant."""
    K = engine()
    cs = nav.CurveSet.from_params(K.MARKET_PARAMS)
    wool = cs["WOOL"]
    floored_inv = 10000 + 4000
    assert wool.quote(floored_inv) == 1
    assert wool.priced_units(floored_inv, 50) == 0
    assert wool.marginal_revenue(floored_inv, 50) == pytest.approx(50.0)
    assert wool.marginal_revenue(floored_inv, 50) == pytest.approx(
        _walk_revenue(K, "WOOL", floored_inv, 50))


# ------------------------------------------------------------------------- curve fitting (T4a)
def _fit_over_episode(env, seat=0, upto=None):
    n = nav.Nav()
    for obs in observations(env, seat)[:upto]:
        n.observe(obs)
    return n


def _quote_errors(navobj, K, params=None):
    resolved = K._resolve_market_params(params) if params else None
    errs = {}
    for product in K.PRODUCTS:
        curve = navobj.curves[product]
        seen = curve._observed
        if len(seen) < 5:
            continue
        lo, hi = min(seen), max(seen)
        e = [abs(curve.quote(inv) - K.market_price(product, inv, resolved))
             for inv in range(lo, hi + 1) if inv not in seen]
        if e:
            errs[product] = e
    return errs


def test_curve_fit_recovers_the_engine_curve(episode):
    """`nav` is never given the params: base and I0 come from step 0 and everything else is
    fitted from observed (inventory, price) pairs."""
    K = engine()
    navobj = _fit_over_episode(episode)
    assert navobj.curves["WHEAT"].base == K.MARKET_PARAMS["WHEAT"]["base"]
    assert navobj.curves["WHEAT"].i0 == K.MARKET_I0
    errs = _quote_errors(navobj, K)
    assert errs, "the episode moved no pool -- nothing was fitted"
    for product, e in errs.items():
        assert statistics.median(e) <= 1, f"{product}: median |quote - true| = {statistics.median(e)}"


def test_curve_agnostic_under_perturbed_params(perturbed_episode):
    """T4, the standing engine-bump assay. The engine's curves are moved +-30% and MELON's
    glut shape is swapped; `nav` is given NO params and must re-derive them from observation
    alone by day 10 -- and must not cry wolf while doing it."""
    K = engine()
    params = _perturb_params()
    navobj = _fit_over_episode(perturbed_episode, upto=10 * TURNS_PER_DAY)
    errs = _quote_errors(navobj, K, params)
    assert errs, "the perturbed episode moved no pool -- nothing was fitted"
    for product, e in errs.items():
        assert statistics.median(e) <= 1, f"{product}: median |quote - true| = {statistics.median(e)}"
    assert navobj.curves.tripwire() == {}, "the tripwire fired on a curve nav re-derived fine"

    # ... and the revenue gate on the re-derived curves, against the true walk. Only where
    # the pool has actually been observed: a side with no evidence is not a fitted curve, it
    # is the documented fallback (price_at -> base, mark tagged provisional), and asserting
    # accuracy there would be asserting that nav guesses well.
    resolved = K._resolve_market_params(params)
    full = _fit_over_episode(perturbed_episode)          # by day 10 only two pools have moved 60+
    worst, checked = 0.0, 0
    for product, curve in full.curves.curves.items():
        lo, hi = min(curve._observed), max(curve._observed)
        if hi - lo < 60:
            continue
        true = _walk_revenue(K, product, lo, 60, resolved)
        worst = max(worst, abs(curve.marginal_revenue(lo, 60) - true) / max(1.0, true))
        checked += 1
    assert checked >= 3, "not enough pools moved far enough to gate the revenue"
    assert worst < 0.03, f"marginal_revenue on re-derived curves is off by {worst:.2%}"


def test_an_unobserved_side_is_a_fallback_and_says_so(perturbed_episode):
    """The pre-registered fallback ladder (spec §3): a curve with no evidence on the side a
    mark needs returns `base` and the mark is flagged `provisional` -- it never silently
    becomes a guess. Ten days of a farming episode never push a pool into glut, so the glut
    side is exactly that case."""
    navobj = _fit_over_episode(perturbed_episode, upto=10 * TURNS_PER_DAY)
    fertilizer = navobj.curves["FERTILIZER"]
    assert fertilizer.confidence == 0.0
    assert fertilizer.price_at(fertilizer.i0 + 500) == float(fertilizer.base)
    sheet = navobj.sheet(variant="drain")
    assert isinstance(sheet.diagnostics["curve_confidence"], float)


def test_tripwire_stays_quiet_on_ordinary_episodes(episode):
    navobj = _fit_over_episode(episode)
    assert navobj.curves.tripwire() == {}


def test_tripwire_fires_on_an_unrepresentable_curve():
    """The negative control, and the reason the tripwire is worth having: a market whose
    curve is NOT one of the five shapes must be noticed, not silently fitted.

    A tripwire that has never been observed to fire is decoration.
    """
    import math

    base = 25
    cs = nav.CurveSet()
    fired_at = None
    for k in range(200):
        inv = 10000 - 3 * k
        d = 10000 - inv
        price = max(1, int(round(base + 14 * math.sin(d / 37.0) + 0.02 * d)))
        market = {"inventory": {p: (inv if p == "WHEAT" else 10000) for p in nav.rules.PRODUCTS},
                  "prices": {p: (price if p == "WHEAT" else 25) for p in nav.rules.PRODUCTS}}
        cs.observe(market, step=k)
        if fired_at is None and "WHEAT" in cs.tripwire():
            fired_at = k
    assert fired_at is not None, "an oscillating, unfittable curve did not trip the wire"
    assert fired_at < 60, f"the tripwire took {fired_at} points to notice"
    assert cs.tripwire()["WHEAT"] > 0.05


def test_tripwire_ignores_a_curve_it_can_represent():
    import math

    base, cs = 25, nav.CurveSet()
    for k in range(200):
        inv = 10000 - 3 * k
        price = max(1, int(round(base + (0.8 * base / math.sqrt(400)) * math.sqrt(10000 - inv))))
        market = {"inventory": {p: (inv if p == "WHEAT" else 10000) for p in nav.rules.PRODUCTS},
                  "prices": {p: (price if p == "WHEAT" else 25) for p in nav.rules.PRODUCTS}}
        cs.observe(market, step=k)
        assert "WHEAT" not in cs.tripwire()


def test_confidence_is_zero_before_any_evidence():
    cs = nav.CurveSet()
    market = {"inventory": {p: 10000 for p in nav.rules.PRODUCTS},
              "prices": {p: 25 for p in nav.rules.PRODUCTS}}
    cs.observe(market, step=0)
    assert cs["WHEAT"].confidence == 0.0
    assert cs["WHEAT"].price_at(9990) == 25.0        # falls back to base, never guesses


# ----------------------------------------------------------------- terminal identity (T1/T1b)
@pytest.mark.parametrize("variant", ["spot", "walk", "drain"])
@pytest.mark.parametrize("lam_opp", [15.0, 50.0])
def test_terminal_identity_on_a_real_episode(episode, variant, lam_opp):
    """Phi(719) == 0 exactly, on both seats, marked by the shipped library on a real replay.

    Not a tolerance: every line goes to zero because H = 0 starves the same formula that
    values it at H = 700.
    """
    lam = nav.LambdaPair(opp=lam_opp)
    for seat in (0, 1):
        navobj = _fit_over_episode(episode, seat)
        assert navobj.step == TERMINAL_INDEX
        sheet = navobj.sheet(variant=variant, lam=lam)
        assert sheet.step == TERMINAL_INDEX
        assert sheet.cash > 0, "the episode banked nothing -- pick a livelier agent"
        for line in sheet.lines:
            assert line.value == 0.0, f"seat {seat} line {line.name} = {line.value} at the bell"
        assert sheet.phi == 0.0
        assert sheet.nav == sheet.cash
        assert navobj.phi_rel(variant=variant, lam=lam) == 0.0


def test_phi_is_not_trivially_zero(episode):
    """The guard on T1: a mark that is always zero would pass the terminal identity. Phi has
    to be positive while the farm still owns something."""
    navobj = nav.Nav()
    seen_positive = 0
    for i, obs in enumerate(observations(episode, 0)):
        navobj.observe(obs)
        if i in (5 * TURNS_PER_DAY, 10 * TURNS_PER_DAY, 15 * TURNS_PER_DAY, 20 * TURNS_PER_DAY):
            if navobj.sheet().phi > 0:
                seen_positive += 1
    assert seen_positive >= 3, "Phi was ~zero mid-episode: the marks are not marking anything"


def test_no_terminal_special_casing():
    """T1b, the static half of T1: no mark may branch on 'is it the last day'.

    Feasibility comparisons (`actions > H`, `harvest_day > LAST_DAY`) are the formula and are
    allowed; an equality against the terminal day/step/tick is a hand-written terminal case
    and is not.
    """
    tree = ast.parse((NAV_DIR / "sheet.py").read_text(encoding="utf-8"))
    banned_names = {"LAST_DAY", "TERMINAL_INDEX", "t_ticks", "last_day"}
    offences = []
    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,))]:
        for node in ast.walk(func):
            if not isinstance(node, ast.If):
                continue
            for cmp_node in [c for c in ast.walk(node.test) if isinstance(c, ast.Compare)]:
                if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in cmp_node.ops):
                    continue
                text = ast.dump(cmp_node)
                if any(name in text for name in banned_names) or "29" in [
                        str(getattr(c, "value", "")) for c in ast.walk(cmp_node)]:
                    offences.append(f"{func.name}: {ast.unparse(node.test)}")
    assert not offences, "terminal special-casing found in nav/sheet.py: " + "; ".join(offences)


# ------------------------------------------------------------------------------ purity (T6)
def test_marks_do_not_mutate_their_inputs(episode):
    """T6. The liquidation pool is the one thing a mark may move, and it lives in the
    context, not in the caller's state."""
    navobj = nav.Nav()
    obs_list = observations(episode, 0)
    for obs in obs_list[:15 * TURNS_PER_DAY]:
        navobj.observe(obs)
    obs = obs_list[15 * TURNS_PER_DAY - 1]
    before = copy.deepcopy(json.loads(json.dumps(obs, default=dict)))
    first = navobj.sheet().as_dict()
    after = json.loads(json.dumps(obs, default=dict))
    assert after == before, "marking mutated the observation"
    second = navobj.sheet().as_dict()
    assert first == second, "marking the same state twice gave different numbers"


def test_replay_reproduces_the_online_object(episode):
    """T5: the offline reconstruction is the online object, so an offline table and the
    agent's own reading are the same number."""
    obs_list = observations(episode, 0)
    online = nav.Nav()
    for obs in obs_list:
        online.observe(obs)
    offline = nav.Nav.replay(obs_list)
    assert offline.sheet().as_dict() == online.sheet().as_dict()
    assert offline.diagnostics()["curves"] == online.diagnostics()["curves"]


# ------------------------------------------------------------------------ opponent books (row 13)
def test_opponent_sheet_equals_their_own_public_lines(episode):
    """Phi_rel's foundation: `obs.farms` is public for both seats, so the opponent's crops,
    animals, structures and land mark to exactly what their own sheet says -- computed from
    their seat or ours. (`spot` isolates this from the order the shared pool is walked in.)"""
    us, them = nav.Nav(), nav.Nav()
    obs_us, obs_them = observations(episode, 0), observations(episode, 1)
    for i in range(12 * TURNS_PER_DAY):
        us.observe(obs_us[i])
        them.observe(obs_them[i])
    ours_view = us.opponent_sheet(variant="spot")
    theirs = them.sheet(variant="spot")
    assert ours_view.seat == 1 and theirs.seat == 1
    assert ours_view.cash == theirs.cash
    for name in ("crops", "animals", "fertilizer", "feed", "structures", "land"):
        assert ours_view.line(name).value == pytest.approx(theirs.line(name).value), name
    assert ours_view.line("shed") is None, "their shed is not visible and must not be marked"
    assert ours_view.diagnostics["unmarkable_lines"] == "shed,carried,seeds"


def test_phi_rel_is_the_difference_of_the_two_potentials(episode):
    navobj = nav.Nav()
    for obs in observations(episode, 0)[:12 * TURNS_PER_DAY]:
        navobj.observe(obs)
    expected = navobj.sheet().phi - navobj.opponent_sheet().phi
    assert navobj.phi_rel() == pytest.approx(expected)


def test_opponent_net_flow_is_exact_when_own_flow_is_given():
    """The inference is `their_flow = delta_inventory + town_drain - our_flow`; the town
    drain is exact from obs.town, and our own flow is the caller's to supply."""
    from nav.book import DemandBook
    from nav.opponent import OpponentBook

    book = DemandBook()
    ob = OpponentBook(book)
    farms = [{"money": 100.0}, {"money": 200.0}]
    town = {"unlocked_shops": ["YARN_STORE"]}          # 2 WOOL every 4 steps
    inv = {p: 10000 for p in nav.rules.PRODUCTS}

    book.observe(town, 100)
    ob.observe({"inventory": inv, "prices": inv}, farms, seat=0, step=100)
    # steps 100..103 fire one shop tick (at 100 -> already past; at 104 -> not yet): step 100
    # itself has not been processed when we observe index 100, so ticks at 100..103 count.
    nxt = dict(inv)
    nxt["WOOL"] = inv["WOOL"] + 7 - 2                  # 7 units sold into it, 2 drained by the shop
    book.observe(town, 104)
    ob.observe({"inventory": nxt, "prices": nxt}, farms, seat=0, step=104, own_flow={"WOOL": 3})
    assert ob.net_sales("WOOL") == pytest.approx(4.0)  # 7 total - 3 ours
    assert ob.provisional is False
    assert ob.cash() == 200.0

    ob2 = OpponentBook(book)
    ob2.observe({"inventory": inv, "prices": inv}, farms, seat=0, step=100)
    ob2.observe({"inventory": nxt, "prices": nxt}, farms, seat=0, step=104)
    assert ob2.provisional is True, "a step without own_flow must be flagged, not guessed"
    assert ob2.unmarkable_residual()["missing_lines"] == ("shed", "carried", "seeds")


# ---------------------------------------------------------------------------- the demand book
def test_drain_per_day_matches_the_validated_probe():
    """Lifted from probe_core `_drain_per_day`, which is already validated -- so assert
    equality with it rather than re-deriving the number."""
    sys.path.insert(0, str(REPO / "agents" / "team_b" / "probes"))
    import probe_core

    from nav.book import DemandBook

    for shops in ([], ["YARN_STORE"], ["BAKERY", "BAKERY"], ["PET_CAFE", "PIZZA_SHOP", "FARMERS_MARKET"]):
        book = DemandBook()
        book.observe({"unlocked_shops": shops}, step=0)
        for product in nav.rules.PRODUCTS:
            assert book.drain_per_day(product) == probe_core._drain_per_day(
                {"unlocked_shops": shops}, product), (shops, product)


def test_town_drain_matches_the_engine_over_an_episode(episode):
    """The exact half of the demand book, checked against what the engine actually removed:
    with no market orders in play the pool moves by the town drain and nothing else."""
    from nav.book import DemandBook

    K = engine()
    env = _play(["pass", "pass"], seed=3, steps=200)
    book = DemandBook()
    steps = [env.steps[i][0].observation for i in range(len(env.steps))]
    for i, obs in enumerate(steps):
        book.observe(obs["town"], i)
    for product in K.PRODUCTS:
        start, end = steps[0]["market"]["inventory"][product], steps[-1]["market"]["inventory"][product]
        drained = start - end
        # the unlock set grows during the episode, so re-walk it step by step
        expected = 0.0
        for i in range(len(steps) - 1):
            b = DemandBook()
            b.observe(steps[i]["town"], i)
            expected += b.drain_over_steps(product, i, i)
        assert drained == pytest.approx(expected), product


def test_unlock_posterior_is_the_engine_draw():
    from nav.book import DemandBook

    book = DemandBook()
    book.observe({"unlocked_shops": ["BAKERY"]}, step=0)
    post = book.unlock_posterior(day=0)
    assert set(post) == set(nav.rules.SHOPS)
    assert 0.0 < post["YARN_STORE"] < 1.0
    late = DemandBook()
    late.observe({"unlocked_shops": ["BAKERY"]}, step=28 * TURNS_PER_DAY)
    assert late.remaining_draws() == 0
    assert late.unlock_posterior()["YARN_STORE"] == 0.0


# ---------------------------------------------------------------------------------- lambda
def test_lambda_is_explicit_and_moves_the_sheet(episode):
    navobj = _fit_over_episode(episode, upto=12 * TURNS_PER_DAY)
    cheap = navobj.sheet(lam=nav.LambdaPair(opp=15.0)).phi
    dear = navobj.sheet(lam=nav.LambdaPair(opp=50.0)).phi
    assert dear < cheap, "charging more per action must shrink the potential"
    assert navobj.sheet().lam == nav.LAMBDA_CRITIC


def test_live_lambda_falls_back_until_its_window_is_full():
    live = nav.LiveLambda(wage=15.0, fallback_opp=15.0, window_days=3)
    assert live.at(0) == nav.LambdaPair(opp=15.0, wage=15.0)
    for day in range(3):
        live.record(day, gross=1000.0, cogs=200.0, productive_actions=20.0, idle_fraction=0.0)
    assert live.at(72).opp == pytest.approx(40.0)
    live2 = nav.LiveLambda(wage=15.0)
    live2.record(0, gross=10.0, cogs=9.0, productive_actions=20.0)
    assert live2.at(0).opp == 15.0, "an unfed estimator returns the documented constant"


# ------------------------------------------------------------------- diagnostics, not lines
def test_covenant_and_intangible_never_enter_nav(episode):
    """Spec §3.8 / row 2: both are reported and neither is netted into NAV."""
    navobj = _fit_over_episode(episode, upto=20 * TURNS_PER_DAY)
    sheet = navobj.sheet()
    assert "covenant_charge" in sheet.diagnostics
    assert sheet.nav == pytest.approx(sheet.cash + sum(l.value for l in sheet.lines))
    assert all(l.name != "covenant" and l.name != "intangible" for l in sheet.lines)


def test_sheet_as_dict_is_json_safe_and_versioned(episode):
    navobj = _fit_over_episode(episode, upto=8 * TURNS_PER_DAY)
    payload = navobj.sheet().as_dict()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["schema_version"] == nav.SCHEMA_VERSION
    assert round_tripped["lam"] == {"opp": 15.0, "wage": 15.0}
    assert set(round_tripped) >= {"step", "day", "seat", "variant", "cash", "nav", "phi", "lines"}


# ------------------------------------------------------------------------- isolation (T9)
def test_runtime_half_imports_nothing_but_the_stdlib():
    """T9 + the vendoring precondition: no numpy, no engine, nothing from agents/ or arena/.
    `nav.analysis` is exempt and is never imported by `nav/__init__.py`."""
    banned = ("numpy", "pandas", "scipy", "torch", "kaggle_environments", "agents", "arena",
              "tools", "probe_core", "instrument")
    for path in RUNTIME_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                assert root not in banned, f"{path.name} imports {name}"
    tree = ast.parse((NAV_DIR / "__init__.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("analysis" in name for name in imported), \
        "nav/__init__.py must not import the offline half"
    assert "analysis" not in sys.modules.get("nav").__dict__, \
        "importing nav must not drag numpy in"


def test_analysis_is_the_only_module_that_may_use_numpy():
    assert "import numpy" in (NAV_DIR / "analysis.py").read_text(encoding="utf-8")
    assert "analysis.py" not in nav.RUNTIME_MODULES


# ---------------------------------------------------------------------- vendorability (T8)
def test_vendoring_produces_an_importable_bundle():
    """The bundle is what Kaggle sees: `main.py` at the root, `nav/` beside it, and nothing
    on `sys.path` but that directory."""
    import tarfile

    with tempfile.TemporaryDirectory(prefix="nav-vendor-") as tmp:
        tmp = Path(tmp)
        res = subprocess.run(
            [sys.executable, str(SCRIPTS / "make_submission.py"), "--agent-dir", "agents/team_b/routed",
             "--vendor", "nav", "--dist-dir", str(tmp / "dist")],
            cwd=str(REPO), text=True, capture_output=True)
        assert res.returncode == 0, res.stdout + res.stderr
        artefact = tmp / "dist" / "submission.tar.gz"
        assert artefact.exists()
        with tarfile.open(artefact) as tar:
            names = tar.getnames()
            tar.extractall(tmp / "bundle", filter="data")
        assert "main.py" in names
        assert "nav/__init__.py" in names and "nav/sheet.py" in names
        assert "nav/analysis.py" not in names, "the offline half must never be vendored"
        assert artefact.stat().st_size < 200 * 1024

        check = tmp / "check.py"
        check.write_text(
            "import sys; sys.path.insert(0, sys.argv[1]); import nav;"
            "print(nav.SCHEMA_VERSION, nav.source_hash(), len(nav.CurveSet().curves))",
            encoding="utf-8")
        out = subprocess.run([sys.executable, "-I", str(check), str(tmp / "bundle")],
                             cwd=str(tmp), text=True, capture_output=True)
        assert out.returncode == 0, out.stdout + out.stderr
        assert out.stdout.split()[0] == str(nav.SCHEMA_VERSION)
        assert out.stdout.split()[2] == "9"


# ------------------------------------------------------------------------ runtime budget (T7)
def test_runtime_budget(episode):
    """Spec §7: `observe` every turn, `sheet` on demand, both far below the Pit's timing gate
    (p99 150 ms) so adding nav can never be why an agent times out. The assertion is loose on
    purpose -- this runs on whatever the CI box is doing -- but the numbers are printed."""
    import time

    navobj = nav.Nav()
    obs_ms, sheet_ms = [], []
    for i, obs in enumerate(observations(episode, 0)):
        t = time.perf_counter()
        navobj.observe(obs)
        obs_ms.append(1000 * (time.perf_counter() - t))
        if i % 24 == 0:
            t = time.perf_counter()
            navobj.sheet()
            navobj.phi_rel()
            sheet_ms.append(1000 * (time.perf_counter() - t))
    obs_ms.sort()
    sheet_ms.sort()
    print(f"\nnav observe: p50 {obs_ms[len(obs_ms) // 2]:.3f} ms  p99 {obs_ms[int(0.99 * len(obs_ms))]:.3f} ms  "
          f"max {obs_ms[-1]:.3f} ms")
    print(f"nav sheet+phi_rel: p50 {sheet_ms[len(sheet_ms) // 2]:.3f} ms  "
          f"p99 {sheet_ms[int(0.99 * len(sheet_ms))]:.3f} ms  max {sheet_ms[-1]:.3f} ms")
    assert obs_ms[int(0.99 * len(obs_ms))] < 5.0
    assert sheet_ms[int(0.99 * len(sheet_ms))] < 25.0
