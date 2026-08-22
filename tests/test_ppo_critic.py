"""exp-014's factored critic: the properties the one-variable comparison rests on.

Three of these are not style checks, they are the claims the experiment makes:
  * the advantage still sees ONE scalar, and that scalar is the sum of the lines;
  * an auxiliary head cannot leak into the value (so a badly scaled aux head can cost us
    trunk capacity but can never move the policy gradient);
  * the analytic gradient of the value loss is the finite-difference gradient, which is the
    same standard `test_ppo_update.py` holds the policy heads to.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.team_a.ppo.critic import FactoredCritic, ScalarCritic, make_critic
from agents.team_a.ppo.lines import LINES, N_LINES


def _fixture(seed=0, n=64, d=12):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    tl = rng.standard_normal((n, N_LINES)) * np.array([3.0, 9.0, 5.0, 1.0, 2.0, 4.0, 0.5])
    return x, tl


def test_lines_are_a_partition_with_a_stable_order():
    assert len(set(LINES)) == N_LINES == 7
    # the order is serialised into every exp-014 row; a silent reorder would make two runs
    # incomparable while every number still looked plausible
    assert LINES[0] == "sell_staple" and LINES[-1] == "cost_labour"


def test_value_is_exactly_the_sum_of_the_lines():
    x, tl = _fixture()
    c = FactoredCritic(x.shape[1], hidden=16, seed=1)
    c.fit_norm(x, tl)
    assert np.allclose(c.value(x), c.lines(x).sum(1), atol=0, rtol=0)


def test_aux_heads_cannot_enter_the_value():
    x, tl = _fixture()
    c = FactoredCritic(x.shape[1], hidden=16, seed=2, aux_dims={"opp_net": 9, "headroom": 9})
    c.fit_norm(x, tl)
    before = c.value(x).copy()
    for k in ("opp_net", "headroom"):
        W, b = c.net.heads[k]
        W += 100.0
        b += 100.0
    assert np.array_equal(c.value(x), before)


def test_value_loss_gradient_matches_finite_differences():
    x, tl = _fixture(seed=3, n=32, d=8)
    c = FactoredCritic(x.shape[1], hidden=8, seed=4)
    c.fit_norm(x, tl)
    t = (tl - c.line_mu) / c.line_sd
    xn = (x - c.mu) / c.sd

    def loss():
        pred = c.net.forward(xn)["v"]
        return float(((pred - t) ** 2).mean(0).sum())

    out, cache = c.net.forward(xn, cache=True)
    diff = out["v"] - t
    grads = c.net.backward(cache, {"v": (2.0 * diff / t.shape[0]).astype(np.float32)})
    params = c.net.params()

    rng = np.random.default_rng(5)
    checked = 0
    for p, g in zip(params, grads):
        flat_p, flat_g = p.reshape(-1), np.asarray(g).reshape(-1)
        for i in rng.choice(flat_p.size, size=min(4, flat_p.size), replace=False):
            h = 1e-3 * max(1.0, abs(float(flat_p[i])))
            old = float(flat_p[i])
            flat_p[i] = old + h
            lp = loss()
            flat_p[i] = old - h
            lm = loss()
            flat_p[i] = old
            num = (lp - lm) / (2 * h)
            assert abs(num - float(flat_g[i])) <= 2e-4 * max(1.0, abs(num)), (
                f"analytic {float(flat_g[i]):.6g} vs numeric {num:.6g}")
            checked += 1
    assert checked >= 12


def test_a_partially_labelled_aux_batch_is_legal():
    """The S-2 corpus covers some rows and not others; a missing head must be a no-op."""
    x, tl = _fixture(seed=6)
    c = FactoredCritic(x.shape[1], hidden=16, seed=7, aux_dims={"opp_net": 9, "headroom": 9})
    c.fit_norm(x, tl, aux={"opp_net": np.zeros((len(x), 9))})
    before = [p.copy() for p in c.net.heads["headroom"]]
    c.update(x, target_lines=tl, aux={"opp_net": np.zeros((len(x), 9))})
    # `headroom` had no target, so its own head matrices must not have moved
    for a, b in zip(c.net.heads["headroom"], before):
        assert np.array_equal(a, b)


def test_fitting_lines_drives_the_pooled_scalar_down():
    x, tl = _fixture(seed=8, n=256, d=10)
    c = FactoredCritic(x.shape[1], hidden=32, lr=3e-3, seed=9)
    c.fit_norm(x, tl)
    first = c.update(x, target_lines=tl)["critic_loss"]
    for _ in range(200):
        c.update(x, target_lines=tl)
    last = c.update(x, target_lines=tl)["critic_loss"]
    assert last < first
    ev = c.explained_var(x, tl)
    assert len(ev["per_line"]) == N_LINES
    assert ev["pooled"] > 0.0


def test_factored_critic_degrades_to_a_scalar_when_lines_are_absent():
    x, _tl = _fixture(seed=10)
    tgt = np.linspace(-1.0, 1.0, len(x))
    c = FactoredCritic(x.shape[1], hidden=16, seed=11)
    c.fit_norm(x, np.repeat(tgt[:, None] / N_LINES, N_LINES, axis=1))
    out = c.update(x, target=tgt)
    assert np.isfinite(out["critic_loss"])


def test_scalar_critic_still_behaves_and_the_factory_dispatches():
    x, _tl = _fixture(seed=12)
    tgt = np.linspace(-1.0, 1.0, len(x))
    c = make_critic("scalar", x.shape[1], hidden=16, seed=13)
    assert isinstance(c, ScalarCritic) and c.kind == "scalar"
    c.fit_norm(x)
    first = c.update(x, tgt)["critic_loss"]
    for _ in range(100):
        c.update(x, tgt)
    assert c.update(x, tgt)["critic_loss"] < first
    assert make_critic("factored", x.shape[1], hidden=8).kind == "factored"
    with pytest.raises(ValueError):
        make_critic("nope", 4)
