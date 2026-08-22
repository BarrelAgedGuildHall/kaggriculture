"""K7'-b and K8' reproduce the ruling's own published arithmetic.

These two statistics decide whether the alpha pilot escalates, and until now neither had ever
been executed -- the exp-013 halt read computed them by hand. Pinning them against numbers the
ruling already published (R-6.1's table and A-3.3's MDE table) is the only check available that
does not simply re-assert the implementation against itself.

The A-3.3 correction is the thing most worth pinning. R-6.1 read a $5,588 margin decline against
a borrowed Millrace sigma of $2.0-5.8k and reported "1.0-2.8 SD, fires on any SD in the band". At
Team A's own sigma_pair ~ $16k that reading inverts: 0.35 per-episode SD (misses) versus 2.79 SE
of the paired mean at n = 64 (fires decisively). R-6.1's conclusion survived; its stated reason
did not. So the unit is the test.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.ppo_interim_read import k7_prime, k8_prime

# R-6.1's halt-read table.
PI0_OURS, PI0_THEM = 65_684.0, 68_310.0          # margin -2,626  = -4.00% of own bank
CAND_OURS, CAND_THEM = 68_602.0, 76_816.0        # margin -8,214  = -11.97% of own bank
SIGMA = 16_000.0                                 # A-3.2: Team A's own same-family paired sigma
N_PAIRS = 64                                     # A-3.3: raised from 32


def _rows(mean_ours, mean_them, sigma, n=N_PAIRS, seed=0, corr=0.0):
    """Boards with an exact mean and a controlled paired-margin SD."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    e = (e - e.mean()) / e.std(ddof=1)            # exactly mean 0, SD 1
    common = rng.normal(size=n)
    common = (common - common.mean()) / common.std(ddof=1)
    out = []
    for i in range(n):
        d = sigma * e[i]
        base = corr * 5_000.0 * common[i]
        out.append({"seed": 3032 + i, "seat": i % 2,
                    "us": mean_ours + base + d / 2.0,
                    "them": mean_them + base - d / 2.0})
    return out


def test_k7_prime_reproduces_the_R61_table():
    cand = _rows(CAND_OURS, CAND_THEM, SIGMA, seed=1)
    ref = _rows(PI0_OURS, PI0_THEM, SIGMA, seed=2)
    k = k7_prime(cand, ref)

    assert k["n_pairs"] == 64
    assert k["iter0"]["margin"] == pytest.approx(-2626.0, abs=1.0)
    assert k["iter0"]["d_over_bank_pct"] == pytest.approx(-4.00, abs=0.02)
    assert k["candidate"]["margin"] == pytest.approx(-8214.0, abs=1.0)
    assert k["candidate"]["d_over_bank_pct"] == pytest.approx(-11.97, abs=0.02)
    assert k["margin_decline"] == pytest.approx(5588.0, abs=1.0)

    # R-6.2's required decomposition
    assert k["decomposition"]["d_ours"] == pytest.approx(2918.0, abs=1.0)
    assert k["decomposition"]["d_theirs"] == pytest.approx(8506.0, abs=1.0)

    # A-3.3's MDE table: sigma 16k over sqrt(64) -> SE 2,000; 5,588 / 2,000 = 2.79
    assert k["sigma_pair"] == pytest.approx(SIGMA, rel=0.01)
    assert k["se_paired_mean"] == pytest.approx(2000.0, rel=0.01)
    assert k["decline_in_se"] == pytest.approx(2.79, abs=0.02)
    assert k["mde_at_1645se"] == pytest.approx(3290.0, rel=0.01)
    assert k["money_up"] and k["fires"]


def test_the_per_episode_SD_reading_would_have_missed():
    """The A-3.3 inversion, stated as a test so it cannot quietly come back.

    $5,588 is 0.35 of a $16k per-episode SD. Anything that computes the trigger in per-episode
    SDs makes K7' untriggerable at the effect size that matters ($4-6k), which is the opposite
    of the power defect K7' was created to fix.
    """
    cand = _rows(CAND_OURS, CAND_THEM, SIGMA, seed=1)
    ref = _rows(PI0_OURS, PI0_THEM, SIGMA, seed=2)
    k = k7_prime(cand, ref)
    assert k["margin_decline"] / k["sigma_pair"] == pytest.approx(0.35, abs=0.02)
    assert k["decline_in_se"] > 1.645 > k["margin_decline"] / k["sigma_pair"]


def test_k7_prime_does_not_fire_when_our_money_did_not_rise():
    """Both arms of the trigger are required: `money up` AND `d/bank` down."""
    cand = _rows(PI0_OURS - 3_000.0, CAND_THEM, SIGMA, seed=1)
    ref = _rows(PI0_OURS, PI0_THEM, SIGMA, seed=2)
    k = k7_prime(cand, ref)
    assert k["margin_decline"] > k["threshold_se"] * k["se_paired_mean"]
    assert not k["money_up"] and not k["fires"]


def test_k7_prime_sigma_comes_from_the_probe_not_a_constant():
    """A-3.2's standing rule: a gate in SDs computes its SD from the matchup it evaluates."""
    tight = k7_prime(_rows(CAND_OURS, CAND_THEM, 4_000.0, seed=1),
                     _rows(PI0_OURS, PI0_THEM, 4_000.0, seed=2))
    wide = k7_prime(_rows(CAND_OURS, CAND_THEM, 32_000.0, seed=1),
                    _rows(PI0_OURS, PI0_THEM, 32_000.0, seed=2))
    assert tight["sigma_pair"] == pytest.approx(4_000.0, rel=0.01)
    assert wide["sigma_pair"] == pytest.approx(32_000.0, rel=0.01)
    assert tight["decline_in_se"] > wide["decline_in_se"]
    assert tight["fires"] and not wide["fires"]


def test_k8_prime_admission_uses_D3s_measured_numbers():
    """D-3: pi0 sampled at T=2 banks 26,682 against its own greedy 66,466 -- a 59.9% shortfall.

    That is the measurement that produced A-1.4's admission check, so it is the natural fixture:
    the check must REFUSE that arm and admit the T=1 case where the gap is a few percent.
    """
    greedy = _rows(66_466.0, 60_000.0, SIGMA, n=32, seed=3)
    sampled_t2 = _rows(26_682.0, 60_000.0, SIGMA, n=32, seed=4)
    k = k8_prime(greedy, sampled_t2, 2.0)
    assert k["money_gap_pct"] == pytest.approx(-59.9, abs=0.1)
    assert not k["admission_pass"] and k["trails_by_more_than_limit"]

    sampled_t1 = _rows(61_480.0, 60_000.0, SIGMA, n=32, seed=5)   # -7.5%, D-3's T=1 cell
    k1 = k8_prime(greedy, sampled_t1, 1.0)
    assert k1["money_gap_pct"] == pytest.approx(-7.5, abs=0.1)
    assert k1["admission_pass"]


def test_k8_prime_reports_the_sign_flip_a_money_gap_hides():
    """A-1.4's addition. D-3's sharper failure was the two series moving in OPPOSITE directions
    on d/bank, which a money gap within the 25% limit does not show at all."""
    greedy = _rows(66_000.0, 69_000.0, SIGMA, n=32, seed=6)       # d/bank negative
    sampled = _rows(60_000.0, 55_000.0, SIGMA, n=32, seed=7)      # d/bank positive
    k = k8_prime(greedy, sampled, 1.0)
    assert k["admission_pass"], "money gap alone is inside the limit -- that is the point"
    assert not k["same_sign"]
    assert k["greedy_d_over_bank_pct"] < 0 < k["sampled_d_over_bank_pct"]
