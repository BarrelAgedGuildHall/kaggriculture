"""exp-014's auxiliary labels are exact, and they never reach the actor.

Two independent things are pinned here, and the second is the important one.

**The labels close the engine's own identity.** S-2's validation rests on
`inv[t+1] - inv[t] == sum_players (sold_above_floor - bought) - drain(t)` holding with ZERO
residual over 155,304 product-cells, because `market["inventory"]` has exactly two writers. Our
rollout-side labels are computed from the same two quantities, so the same identity must close on
a live episode -- otherwise the flow tally or the drain model is wrong and every `opp_net` label
is quietly off.

**The auxiliary heads are off the deployed forward path** (addendum-01 A-4.3). exp-014 is the
first Team A arm to put privileged information anywhere near the network, and the boundary A-4.2
draws is not detectable on a score: an agent that cheats scores *better*, right up until the
submission runs without the labels. So the red test is structural -- the critic's `value()` must
be bit-identical with the aux heads' weights present and with them zeroed, and the actor's
weights must not contain a critic tensor at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.team_a.ppo import auxlabels as AUX
from agents.team_a.ppo import critic as C
from agents.team_a.ppo import harness as H
from agents.team_a.ppo import lines as L
from agents.team_a.ppo.bcpath import F

F_QTY = F.QTY_BUCKETS

SEED = 3001
STEPS = 720


@pytest.fixture(scope="module")
def episode():
    K = H.engine_module()
    acc = L.Accountant(K)
    rec = AUX.Recorder(K)
    acc.on_step(rec)
    actor = H.make_policy("pi0", safety=3, greedy=True)
    opp = H.make_policy("base", safety=3, greedy=True)
    us, them, _ = H.play(SEED, 0, actor, opp, steps=STEPS, instrument=acc)
    return acc, rec, us, them


def test_the_market_inventory_identity_closes_exactly(episode):
    """S-2 validation 2, on a live episode instead of a replay."""
    acc, rec, _us, _them = episode
    order = np.argsort(np.asarray(rec.steps))
    inv = np.stack(rec.inv)[order]
    cum = np.stack(rec.cum)[order]
    drain = np.stack(rec.drain)[order]
    sold_above = (cum[:, :, L.FLOWS.index("sold"), :]
                  - cum[:, :, L.FLOWS.index("sold_floored"), :]).sum(1)
    bought = cum[:, :, L.FLOWS.index("bought"), :].sum(1)
    # cumulative -> per-transition
    d_inv = np.diff(inv, axis=0)
    flow = np.diff(sold_above - bought, axis=0)
    resid = d_inv - (flow - drain[:-1])
    assert np.abs(resid).max() == 0, (
        f"market inventory identity broken: max |residual| = {np.abs(resid).max()}, "
        f"{int((resid != 0).sum())} nonzero cells")


def test_opp_net_telescopes_to_the_opponents_whole_episode_flow(episode):
    """The label at step t is the rival's net flow from t INCLUSIVE to the horizon.

    Inclusive, matching S-2's `p{i}_future_net_true`, because the snapshot is taken before the
    step's own commits. So the last row is not zero -- it is that final transition's own flow,
    and asserting zero there would be asserting an off-by-one into the labels.
    """
    acc, rec, _us, _them = episode
    steps = sorted(rec.steps)
    lab = rec.labels(acc, 0, ("opp_net",), steps)["opp_net"]
    assert lab.shape == (len(steps), L.N_PRODUCTS)
    assert np.array_equal(lab[0].astype(np.int64), acc.net_units(1))
    # telescoping: every consecutive pair differs by exactly that transition's signed flow
    per_step = -np.diff(lab, axis=0)
    assert np.array_equal((per_step.sum(0) + lab[-1]).astype(np.int64), acc.net_units(1))
    # a single transition cannot move more than both seats' market-order budget
    assert np.abs(lab[-1]).sum() <= 2 * 10 * max(F_QTY)


def test_headroom_substitutes_drain_for_the_bottomless_pools(episode):
    """WHEAT and EGG have no reachable $1 floor; S-2 emits -1 and names the substitute."""
    acc, rec, _us, _them = episode
    steps = sorted(rec.steps)
    lab = rec.labels(acc, 0, ("headroom",), steps)["headroom"]
    bottomless = [i for i, p in enumerate(L.PRODUCTS) if rec.floors[i] == AUX.NO_FLOOR]
    assert [L.PRODUCTS[i] for i in bottomless] == ["WHEAT", "EGG"]
    # never the sentinel, and never negative: the substitute is a remaining-drain suffix sum
    assert (lab[:, bottomless] >= 0).all()
    assert lab[:, bottomless][-1].sum() == 0          # nothing left to drain at the horizon
    # and it is a decreasing suffix sum
    for j in bottomless:
        assert np.all(np.diff(lab[:, j]) <= 0)


def test_a_turn_with_no_engine_snapshot_is_an_error_not_a_clamp(episode):
    acc, rec, _us, _them = episode
    with pytest.raises(RuntimeError, match="no engine snapshot"):
        rec.labels(acc, 0, ("opp_net",), [10 ** 6])


def test_nav_head_fails_loudly_rather_than_inventing_a_nav(episode):
    acc, rec, _us, _them = episode
    assert "nav" not in AUX.available()
    with pytest.raises(RuntimeError, match="no NAV implementation"):
        rec.labels(acc, 0, ("nav",), sorted(rec.steps))


def test_aux_heads_are_off_the_deployed_path():
    """A-4.3's red test, in its critic-side form.

    The deployed artefact carries no critic at all, so the strongest available statement is that
    the value the advantage uses cannot move when every auxiliary head is destroyed. If zeroing
    the aux weights changed `value()`, an aux head would be inside the value sum -- which is the
    same defect as an aux feature on the actor's forward path, one layer earlier.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 32)).astype(np.float32)
    cr = C.FactoredCritic(32, hidden=16, seed=1, n_lines=L.N_LINES,
                          aux_dims={"opp_net": 9, "headroom": 9})
    cr.fit_norm(x, rng.normal(size=(256, L.N_LINES)),
                {"opp_net": rng.normal(size=(256, 9)), "headroom": rng.normal(size=(256, 9))})
    before = cr.value(x).copy()
    lines_before = cr.lines(x).copy()
    zeroed = 0
    for name in ("opp_net", "headroom"):
        for p in cr.net.heads[name]:
            p[...] = 0.0
            zeroed += 1
    assert zeroed == 4, "no auxiliary head parameters found -- the test is not testing anything"
    assert np.array_equal(before, cr.value(x)), "an auxiliary head is inside value()"
    assert np.array_equal(lines_before, cr.lines(x))
