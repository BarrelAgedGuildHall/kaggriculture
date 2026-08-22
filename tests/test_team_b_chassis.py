"""expB-306c — the bank-shaped chassis (ruling-04 section 1.3, C-1..C-5).

What each group pins:

  1. C-1  the post-hoc deadline discard is GONE: a slow planner's completed plan
          is emitted, because the wall-clock is already spent and emitting is free.
  2. C-2  the crash mask is loud: counters, MILLRACE_STRICT re-raise.
  3. C-3  harm ordering: the transient fallback places NO market orders; the
          PANIC latch fires on 3 exceptions and on a depleted overage bank, and
          in PANIC the planner is never called at all.
  4. C-4  `deadline_s = 0.60 * actTimeout` read from `configuration`, with the
          vendored table pinned against the installed engine so a bump breaks a
          test rather than silently mis-sizing us.
  5. R-32 the behaviour-identity argument's mechanical premise: `feedchain._plan`
          never READS the deadline it is handed, so changing the deadline's
          value cannot change any action. Asserted statically over the AST, so
          it cannot rot silently.

Engine: kaggle-environments 1.32.7.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agents" / "team_b"))

from millrace import chassis as CH  # noqa: E402

MILLRACE_DIR = REPO_ROOT / "agents" / "team_b" / "millrace"


def _obs(n_hands=2, seeds=None, shed=None, bank=None):
    o = {
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
            "seeds": seeds or {"WHEAT": 1},
            "inventories": [{} for _ in range(n_hands + 1)],
        },
        "market": {"prices": {"EGG": 30}, "inventory": {"EGG": 100}},
        "town": {"unlocked_shops": []},
    }
    if bank is not None:
        o["remainingOverageTime"] = bank
    return o


def _plan_ok(obs, st, dl):
    return {"farmer": ["MOVE", "N"], "hands": [["PASS"], ["PASS"]], "market": []}


def _plan_boom(obs, st, dl):
    raise RuntimeError("planner exploded")


# ------------------------------------------------------------------ C-1 no discard
def test_a_slow_plan_is_emitted_not_discarded():
    """The whole of R-26 in one test: the engine never interrupts a turn and
    bills the excess against a 60 s bank, so a completed plan costs nothing
    extra to emit. A guard that can only lose is not a guard."""
    import time

    def slow(obs, st, dl):
        time.sleep(0.30)          # 300 ms: 1.2x the OLD 250 ms discard threshold
        return _plan_ok(obs, st, dl)

    agent = CH.make_chassis(slow)
    out = agent(_obs(), {"actTimeout": 1})
    assert out["farmer"] == ["MOVE", "N"], "a slow but complete plan must be emitted"
    assert agent.timings[0] > 0.25
    assert agent.plan_exceptions == 0
    assert agent.panic is False


def test_the_chassis_has_no_wall_clock_branch_left():
    """Static: no comparison of elapsed time against the deadline survives.
    `time` is used only to measure, never to decide."""
    src = (MILLRACE_DIR / "chassis.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            seg = ast.get_source_segment(src, node) or ""
            assert "perf_counter" not in seg, f"wall-clock decision survives: {seg!r}"


# ------------------------------------------------------------------ C-2 loud mask
def test_plan_exceptions_are_counted_and_the_first_is_recorded(monkeypatch):
    monkeypatch.delenv("MILLRACE_STRICT", raising=False)
    agent = CH.make_chassis(_plan_boom)
    agent(_obs())
    assert agent.plan_exceptions == 1
    assert "planner exploded" in agent.first_exc
    assert agent.diag["plan_exceptions"] == 1


def test_millrace_strict_re_raises_instead_of_masking(monkeypatch):
    monkeypatch.setenv("MILLRACE_STRICT", "1")
    agent = CH.make_chassis(_plan_boom)
    with pytest.raises(RuntimeError, match="planner exploded"):
        agent(_obs())


def test_the_mask_still_exists_when_strict_is_absent(monkeypatch):
    """REJECTED in section 1.4: deleting the mask. An unmasked crash on the
    ladder is `status ERROR`, which is a certain loss."""
    monkeypatch.delenv("MILLRACE_STRICT", raising=False)
    agent = CH.make_chassis(_plan_boom)
    out = agent(_obs())
    assert out["farmer"] in (["PASS"], ["DROP"])


# ------------------------------------------------------------------ C-3 harm order
def test_transient_fallback_places_no_market_orders():
    """One bad turn must not force-liquidate the shed at spot into our own
    price impact, on a MELON curve that is quadratic above target."""
    out = CH.fallback_action(_obs(shed={"MELON": 40, "EGG": 9}), latched=False)
    assert out["market"] == []
    assert out["farmer"] in (["PASS"], ["DROP"])


def test_a_single_planner_exception_emits_no_market_orders(monkeypatch):
    monkeypatch.delenv("MILLRACE_STRICT", raising=False)
    agent = CH.make_chassis(_plan_boom)
    out = agent(_obs(shed={"MELON": 40}))
    assert out["market"] == []


def test_latch_fires_on_three_exceptions_then_sells_the_shed(monkeypatch):
    monkeypatch.delenv("MILLRACE_STRICT", raising=False)
    agent = CH.make_chassis(_plan_boom)
    for _ in range(2):
        assert agent(_obs(shed={"EGG": 4}))["market"] == []
    assert agent.panic is False
    agent(_obs(shed={"EGG": 4}))                     # third exception -> latch
    assert agent.panic is True
    assert agent.diag["panic_reason"] == "exceptions"
    out = agent(_obs(shed={"EGG": 4}))               # next turn: latched
    assert ["SELL", "EGG", 4] in out["market"]


def test_latch_fires_when_the_engine_bank_falls_below_a_quarter():
    """Our safety valve now has the same shape as the engine's: a bank, read
    from `remainingOverageTime`, not a per-turn cliff."""
    calls = []

    def counting_plan(obs, st, dl):
        calls.append(1)
        return _plan_ok(obs, st, dl)

    agent = CH.make_chassis(counting_plan)
    agent(_obs(bank=60.0), {"actTimeout": 1})
    assert agent.panic is False and len(calls) == 1
    agent(_obs(bank=40.0), {"actTimeout": 1})
    assert agent.panic is False and len(calls) == 2
    out = agent(_obs(bank=14.0, shed={"EGG": 2}), {"actTimeout": 1})   # < 0.25 * 60
    assert agent.panic is True
    assert agent.diag["panic_reason"] == "bank"
    # PANIC skips the planner entirely -- which, unlike the deleted discard,
    # genuinely saves the wall-clock it is trying to protect.
    assert len(calls) == 2
    assert ["SELL", "EGG", 2] in out["market"]
    assert agent.diag["overage_bank_start"] == 60.0
    assert agent.diag["overage_bank_min"] == 14.0


def test_a_missing_overage_bank_never_latches():
    agent = CH.make_chassis(_plan_ok)
    for _ in range(5):
        agent(_obs())            # no remainingOverageTime key at all
    assert agent.panic is False
    assert agent.diag["overage_bank_start"] is None


# ------------------------------------------------------------------ C-4 the ratio
def test_deadline_is_a_ratio_of_the_live_act_timeout():
    seen = []
    agent = CH.make_chassis(lambda obs, st, dl: seen.append(dl) or _plan_ok(obs, st, dl))
    agent(_obs(), {"actTimeout": 7})
    assert agent.diag["act_timeout"] == 7.0
    assert agent.diag["deadline_s"] == pytest.approx(4.2)
    agent(_obs(), None)                       # loader gave us nothing
    assert agent.diag["act_timeout"] == CH.ACT_TIMEOUT_S
    assert agent.diag["deadline_s"] == pytest.approx(0.6)


def test_no_absolute_seconds_constant_survives_outside_the_engine_table():
    """Section 6.4 as amended bars absolute dollars, days AND seconds. The only
    absolute-time numbers left are the vendored engine table itself."""
    src = (MILLRACE_DIR / "chassis.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    allowed = {"ACT_TIMEOUT_S", "OVERAGE_BANK_S"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & {"DEADLINE_FRAC", "PANIC_BANK_FRAC"}:
                assert 0.0 < node.value.value <= 1.0, names
            for n in names:
                if n.endswith("_S"):
                    assert n in allowed, f"absolute-seconds constant {n} outside the engine table"


def test_vendored_engine_timing_table_matches_the_installed_engine():
    """`ACT_TIMEOUT_S` / `OVERAGE_BANK_S` are vendored because a one-argument
    loader gives us no configuration. Pin them, so an engine bump breaks a test
    instead of silently mis-sizing every turn."""
    import json
    spec = json.loads((REPO_ROOT / "docs" / "engine" / "kaggriculture.json").read_text(encoding="utf-8"))
    assert CH.ACT_TIMEOUT_S == float(spec["configuration"]["actTimeout"])
    import kaggle_environments
    env = kaggle_environments.make("kaggriculture", debug=False)
    assert float(env.configuration["actTimeout"]) == CH.ACT_TIMEOUT_S
    assert float(env.state[0].observation["remainingOverageTime"]) == CH.OVERAGE_BANK_S


# ------------------------------------------------------- R-32 identity premise
def test_the_planner_never_reads_the_deadline_it_is_handed():
    """The mechanical core of the behaviour-identity re-gate: `_plan` binds
    `deadline` and never loads it, so changing its VALUE (0.25 s -> 0.60 x
    actTimeout) cannot change a single emitted action. If a future planner
    starts reading it, this test fails and the identity argument must be
    re-earned rather than assumed."""
    src = (MILLRACE_DIR / "feedchain.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_plan")
    assert "deadline" in [a.arg for a in fn.args.args]
    loads = [n for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "deadline" and isinstance(n.ctx, ast.Load)]
    assert loads == [], f"_plan now reads its deadline at {[n.lineno for n in loads]}"


def test_the_entry_point_takes_two_arguments():
    """Both loaders slice on `co_argcount` (kaggle_environments/agent.py:152,
    arena/agent_child.py), so this signature is what delivers `configuration`."""
    src = (MILLRACE_DIR / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert fns and fns[-1].name == "agent"
    assert [a.arg for a in fns[-1].args.args] == ["obs", "config"]
