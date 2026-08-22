"""`--dump-batch` (C-4): a rollout batch frozen to disk, in the critic's input contract.

The point of the file is that somebody else can pick it up and reproduce what the critic saw
without having the trainer. So the properties pinned here are the ones a consumer relies on:

  * the arrays that ARE the contract (`x`, `target`, `line_targets`, `aux_*`) survive the round
    trip with their shapes and their row alignment intact;
  * enough rides along to recompute the targets -- the per-episode `(seed, seat, opponent, R,
    R_opp)` columns and `ep_of_row`, without which the leave-one-out baseline cannot be checked;
  * the manifest names the arm. `alpha`, the critic kind and the reward scale are not
    decoration: numbers do not carry across a reward change (R-4.3), and a batch that does not
    say which reward produced it will eventually be compared with one from another.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # so this module also runs on its own, not only in a full suite

from agents.team_a.ppo.lines import N_LINES  # noqa: E402
from agents.team_a.ppo.train import (  # noqa: E402
    ACTION_KEYS,
    BATCH_FORMAT,
    LOGP_KEYS,
    REWARD_SCALE,
    loo_baseline,
    read_batch,
    write_batch,
)

N_EP, PER_EP, DIM, N_ORDERS = 6, 5, 12, 4
N_ROWS = N_EP * PER_EP


def _fixture(seed=0, with_lines=True, with_aux=True):
    rng = np.random.default_rng(seed)
    b = {
        "x": rng.standard_normal((N_ROWS, DIM)).astype(np.float32),
        "order_mask": (rng.random((N_ROWS, N_ORDERS)) > 0.3).astype(np.float32),
        "fire": rng.integers(0, 2, (N_ROWS, N_ORDERS)).astype(np.int32),
        "qty_idx": rng.integers(0, 18, (N_ROWS, N_ORDERS)).astype(np.int32),
        "hire_idx": rng.integers(0, 3, N_ROWS),
        "exec_fire": rng.integers(0, 2, (N_ROWS, N_ORDERS)).astype(np.int32),
        "exec_qty": rng.integers(0, 18, (N_ROWS, N_ORDERS)).astype(np.int32),
        "exec_hire": rng.integers(0, 3, N_ROWS),
        "logp_fire": rng.standard_normal((N_ROWS, N_ORDERS)),
        "logp_qty": rng.standard_normal((N_ROWS, N_ORDERS)),
        "logp_hire": rng.standard_normal(N_ROWS),
    }
    ep_of_row = np.repeat(np.arange(N_EP), PER_EP).astype(np.int32)
    results = [{"R": 40_000.0 + 1000 * i, "R_opp": 30_000.0 + 500 * i,
                "seed": 5000 + (i // 2), "seat": i % 2, "opponent_id": ["pi0", "base"][i % 2]}
               for i in range(N_EP)]
    R = np.array([r["R"] for r in results])
    base, _ = loo_baseline(results)
    target_row = ((R - base) / REWARD_SCALE)[ep_of_row]
    lines = np.stack([np.linspace(-1, 1, N_LINES) * (i + 1) for i in range(N_EP)])[ep_of_row] \
        if with_lines else None
    aux = {"opp_net": rng.standard_normal((N_ROWS, 9)).astype(np.float32)} if with_aux else None
    return b, ep_of_row, target_row, lines, aux, results


META = {"iteration": 3, "alpha": 0.0, "critic": "factored", "phase": "P1", "temperature": 1.0}


def test_round_trip_preserves_the_critic_contract(tmp_path):
    b, ep, target, lines, aux, results = _fixture()
    v = np.linspace(-0.2, 0.2, N_ROWS)
    manifest = write_batch(tmp_path / "batch.npz", b, ep, target, lines, aux, results,
                           meta=META, values=v, advantage=target - v)
    arrays, back = read_batch(tmp_path / "batch.npz")

    np.testing.assert_array_equal(arrays["x"], b["x"])
    np.testing.assert_allclose(arrays["target"], target)
    np.testing.assert_allclose(arrays["line_targets"], lines)
    np.testing.assert_allclose(arrays["aux_opp_net"], aux["opp_net"])
    np.testing.assert_allclose(arrays["value"], v)
    np.testing.assert_allclose(arrays["advantage_raw"], target - v, atol=1e-12)
    for k in ACTION_KEYS + LOGP_KEYS:
        np.testing.assert_array_equal(arrays[k], b[k])

    # everything the critic and the policy loss index by row must agree on the row count
    per_row = ["x", "order_mask", "target", "line_targets", "aux_opp_net", "value",
               "advantage_raw", "ep_of_row", *ACTION_KEYS, *LOGP_KEYS]
    assert {len(arrays[k]) for k in per_row} == {N_ROWS}
    assert arrays["line_targets"].shape == (N_ROWS, N_LINES)
    assert back == manifest and back["batch_format"] == BATCH_FORMAT
    assert back["reward_scale"] == REWARD_SCALE and back["n_rows"] == N_ROWS
    assert back["alpha"] == 0.0 and back["critic"] == "factored" and back["iteration"] == 3
    assert back["aux_names"] == ["opp_net"] and len(back["line_names"]) == N_LINES
    assert json.loads((tmp_path / "batch.json").read_text(encoding="utf-8")) == manifest


def test_the_episode_columns_let_the_baseline_be_recomputed(tmp_path):
    b, ep, target, lines, aux, results = _fixture()
    write_batch(tmp_path / "b.npz", b, ep, target, lines, aux, results, meta=META)
    a, m = read_batch(tmp_path / "b.npz")
    rebuilt = [{"R": R, "R_opp": Ro, "seed": int(s), "seat": int(st), "opponent_id": str(o)}
               for R, Ro, s, st, o in zip(a["ep_R"], a["ep_R_opp"], a["ep_seed"], a["ep_seat"],
                                          a["ep_opponent"])]
    base, _ = loo_baseline(rebuilt)
    np.testing.assert_allclose(((a["ep_R"] - base) / m["reward_scale"])[a["ep_of_row"]],
                               a["target"])


def test_line_targets_sum_to_the_scalar_target(tmp_path):
    """The identity exp-014 rests on: the factored critic supervises a decomposition of exactly
    the quantity the scalar critic sees. A dump that broke row alignment would break it here."""
    b, ep, _, lines, aux, results = _fixture()
    target = lines.sum(axis=1)
    write_batch(tmp_path / "b.npz", b, ep, target, lines, aux, results, meta=META)
    a, _ = read_batch(tmp_path / "b.npz")
    np.testing.assert_allclose(a["line_targets"].sum(axis=1), a["target"], atol=1e-9)


def test_a_scalar_arm_dumps_without_lines_or_aux(tmp_path):
    b, ep, target, _, _, results = _fixture(with_lines=False, with_aux=False)
    m = write_batch(tmp_path / "b.npz", b, ep, target, None, None, results,
                    meta={**META, "critic": "scalar"})
    a, back = read_batch(tmp_path / "b.npz")
    assert "line_targets" not in a and not any(k.startswith("aux_") for k in a)
    assert back["line_names"] == [] and back["aux_names"] == [] and m["n_rows"] == N_ROWS


def test_the_flag_exists_on_the_driver():
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "ppo_train.py"), "--help"],
                         capture_output=True, text=True, cwd=REPO, timeout=180)
    assert out.returncode == 0, out.stderr[-800:]
    assert "--dump-batch" in out.stdout and "--dump-batch-iters" in out.stdout


def test_read_batch_refuses_pickled_arrays(tmp_path):
    """`read_batch` loads with allow_pickle=False. A batch file is data, and a .npz that
    unpickles on load is an arbitrary-code-execution path through a shared results directory."""
    np.save(tmp_path / "evil.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        read_batch(tmp_path / "evil.npy")
