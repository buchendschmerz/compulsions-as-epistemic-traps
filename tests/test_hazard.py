"""Analytic core: the rank-one theorem and its constructive converse."""

import numpy as np
import pytest

from epistemic_trap.hazard import (fisher_information, fisher_information_policy,
                                   grad_lam, identifiability_summary,
                                   identify_from_two_k, lam, loglik,
                                   ridge_q_given_p)

INTERIOR = [(0.1, 0.2), (0.3, 0.8), (0.55, 0.45), (0.9, 0.95), (0.02, 0.99)]


@pytest.mark.parametrize("q,p", INTERIOR)
@pytest.mark.parametrize("k", [0, 1, 2, 5, 10])
def test_lambda_matches_explicit_product(q, p, k):
    assert lam(q, p, k) == pytest.approx(q * np.prod([1 - p] * k) if k else q)


@pytest.mark.parametrize("q,p", INTERIOR)
@pytest.mark.parametrize("k", [1, 3, 7])
def test_gradient_matches_finite_difference(q, p, k):
    h = 1e-6
    g = grad_lam(q, p, k)
    dq = (lam(q + h, p, k) - lam(q - h, p, k)) / (2 * h)
    dp = (lam(q, p + h, k) - lam(q, p - h, k)) / (2 * h)
    assert g[0] == pytest.approx(dq, rel=1e-5)
    assert g[1] == pytest.approx(dp, rel=1e-5)


@pytest.mark.parametrize("q,p", INTERIOR)
@pytest.mark.parametrize("k", [0, 1, 2, 4, 8])
def test_fixed_k_information_is_rank_one_and_singular(q, p, k):
    """The central theorem: every fixed-k design has a singular information matrix."""
    info = fisher_information(q, p, k)
    scale = np.trace(info)
    assert abs(np.linalg.det(info)) < 1e-9 * max(scale ** 2, 1.0)
    assert np.linalg.matrix_rank(info, tol=1e-10 * max(scale, 1.0)) <= 1


# Corners where lambda_k underflows towards zero are *practically* singular even
# with two conditions; the full-rank claim is asserted away from those corners.
NON_EXTREME = [(0.1, 0.2), (0.3, 0.8), (0.55, 0.45)]


@pytest.mark.parametrize("q,p", NON_EXTREME)
@pytest.mark.parametrize("k1,k2", [(0, 1), (0, 3), (1, 2), (2, 6)])
def test_two_distinct_k_restore_full_rank(q, p, k1, k2):
    info = fisher_information_policy(q, p, [k1, k2], [0.5, 0.5], n_episodes=100)
    s = identifiability_summary(info)
    assert s["det"] > 0
    assert np.isfinite(s["cond"]) and s["cond"] < 1e10


def test_extreme_corners_are_practically_singular_even_with_two_conditions():
    """Formal identifiability is not practical identifiability.

    When the agent repeats so often that harm essentially never occurs, the
    information matrix is formally non-singular but numerically indistinguishable
    from rank one -- the finite-sample face of the same trap.
    """
    info = fisher_information_policy(0.02, 0.99, [2, 6], [0.5, 0.5], n_episodes=100)
    assert identifiability_summary(info)["cond"] > 1e10


@pytest.mark.parametrize("q,p", INTERIOR)
def test_repeating_the_same_k_never_helps(q, p):
    """A policy with all its mass on one k stays singular however many episodes."""
    for n in (1, 10, 10_000):
        info = fisher_information_policy(q, p, [4], [1.0], n_episodes=n)
        assert abs(np.linalg.det(info)) < 1e-9 * max(np.trace(info) ** 2, 1.0)


@pytest.mark.parametrize("q,p", INTERIOR)
@pytest.mark.parametrize("k", [1, 3, 6])
def test_loglikelihood_is_flat_along_the_ridge(q, p, k):
    lam0 = lam(q, p, k)
    p_grid = np.linspace(0.01, 0.99, 60)
    q_grid = ridge_q_given_p(lam0, p_grid, k)
    ok = np.isfinite(q_grid)
    ll = loglik(300 * lam0, 300, q_grid[ok], p_grid[ok], k)
    assert np.ptp(ll) < 1e-8


@pytest.mark.parametrize("q,p", INTERIOR)
@pytest.mark.parametrize("k", [1, 4])
def test_ridge_slope_is_positive(q, p, k):
    """H3 is a geometric fact: dq/dp = k q /(1-p) >= 0."""
    lam0 = lam(q, p, k)
    ps = np.linspace(0.05, 0.9, 200)
    qs = ridge_q_given_p(lam0, ps, k)
    ok = np.isfinite(qs)
    assert np.all(np.diff(qs[ok]) >= -1e-12)


@pytest.mark.parametrize("q,p", INTERIOR)
@pytest.mark.parametrize("k1,k2", [(0, 1), (0, 4), (2, 5)])
def test_two_condition_recovery_is_exact(q, p, k1, k2):
    q_hat, p_hat = identify_from_two_k(lam(q, p, k1), k1, lam(q, p, k2), k2)
    assert q_hat == pytest.approx(q, abs=1e-9)
    assert p_hat == pytest.approx(p, abs=1e-9)


def test_response_prevention_alone_carries_no_information_about_p():
    """k = 0 identifies q exactly and p not at all -- the correction to the brief."""
    info = fisher_information(0.3, 0.8, 0)
    assert info[1, 1] == pytest.approx(0.0)
    assert info[0, 1] == pytest.approx(0.0)
    assert info[0, 0] > 0


def test_policy_probabilities_are_normalised():
    a = fisher_information_policy(0.3, 0.8, [0, 2], [1.0, 1.0], n_episodes=1)
    b = fisher_information_policy(0.3, 0.8, [0, 2], [0.5, 0.5], n_episodes=1)
    assert np.allclose(a, b)
