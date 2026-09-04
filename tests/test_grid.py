"""Grid posterior: correctness against analytic Bayes and information identities."""

import numpy as np
import pytest
from scipy.stats import beta as beta_dist

from epistemic_trap.grid import ParameterGrid, binary_entropy


def test_uniform_prior_has_zero_differential_entropy():
    g = ParameterGrid.from_beta_prior(1, 1, 1, 1, n=101)
    assert g.entropy_joint() == pytest.approx(0.0, abs=1e-9)
    assert g.entropy_q() == pytest.approx(0.0, abs=1e-9)
    assert g.entropy_p() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("a,b", [(2, 5), (8, 2), (1.5, 13.5), (3, 7)])
def test_beta_prior_moments_and_entropy(a, b):
    g = ParameterGrid.from_beta_prior(a, b, a, b, n=401)
    mq, mp = g.mean()
    assert mq == pytest.approx(a / (a + b), abs=2e-3)
    assert mp == pytest.approx(a / (a + b), abs=2e-3)
    assert g.entropy_p() == pytest.approx(beta_dist(a, b).entropy(), abs=2e-2)


def test_prior_is_independent_across_q_and_p():
    g = ParameterGrid.from_beta_prior(3, 7, 8, 2, n=201)
    assert g.corr_qp() == pytest.approx(0.0, abs=1e-9)


def test_k0_harm_updates_reproduce_analytic_beta_binomial():
    """At k = 0 the harm outcome is Bernoulli(q), so q must follow conjugate Bayes."""
    a, b, n_obs, y = 3.0, 7.0, 40, 11
    g = ParameterGrid.from_beta_prior(a, b, 8, 2, n=801)
    g.update_harm(0, y, n_obs=n_obs)
    post = beta_dist(a + y, b + n_obs - y)
    assert g.mean()[0] == pytest.approx(post.mean(), abs=2e-3)
    assert np.sqrt(g.cov()[0, 0]) == pytest.approx(post.std(), abs=2e-3)


def test_k0_harm_leaves_the_p_marginal_untouched():
    """Response prevention is informative about q and about nothing else."""
    g = ParameterGrid.from_beta_prior(3, 7, 8, 2, n=201)
    before = g.marginal_p().copy()
    g.update_harm(0, 1, n_obs=25)
    assert np.allclose(g.marginal_p(), before, atol=1e-12)


def test_fixed_k_data_induces_positive_q_p_coupling():
    """H3, as a property of the update rule rather than of any simulation."""
    g = ParameterGrid.from_beta_prior(3, 7, 3, 7, n=201)
    for _ in range(200):
        g.update_harm(4, 0)
    assert g.corr_qp() > 0.3


def test_observable_composite_sharpens_while_parameters_do_not():
    """The signature of the trap: lambda_k becomes known, (q, p) does not."""
    g = ParameterGrid.from_beta_prior(3, 7, 3, 7, n=201)
    sd_lam0, sd_p0 = g.sd_lambda(4), np.sqrt(g.cov()[1, 1])
    rng = np.random.default_rng(0)
    lam_true = 0.3 * 0.2 ** 4
    for _ in range(3000):
        g.update_harm(4, int(rng.random() < lam_true))
    sd_lam1, sd_p1 = g.sd_lambda(4), np.sqrt(g.cov()[1, 1])
    assert sd_lam1 / sd_lam0 < 0.15          # the composite collapses
    assert sd_p1 / sd_p0 > 0.5               # the parameter does not


def test_updates_are_order_invariant():
    g1 = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    g2 = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    g1.update_harm(2, 1); g1.update_harm(5, 0); g1.update_check(0, True, 0.8, 0.8)
    g2.update_check(0, True, 0.8, 0.8); g2.update_harm(5, 0); g2.update_harm(2, 1)
    assert np.allclose(g1.posterior, g2.posterior, atol=1e-12)


def test_batched_update_equals_repeated_single_updates():
    g1 = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    g2 = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    g1.update_harm(3, 4, n_obs=10)
    for _ in range(4):
        g2.update_harm(3, 1)
    for _ in range(6):
        g2.update_harm(3, 0)
    assert np.allclose(g1.posterior, g2.posterior, atol=1e-12)


def test_perfect_check_at_zero_equals_a_harm_observation_at_k0():
    """A noiseless baseline probe is exactly a k = 0 outcome observation."""
    g1 = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    g2 = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    g1.update_check(0, True, 1.0, 1.0)
    g2.update_harm(0, 1)
    assert np.allclose(g1.posterior, g2.posterior, atol=1e-12)


def test_uninformative_check_is_a_no_op():
    g = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    before = g.posterior.copy()
    g.update_check(2, True, 0.5, 0.5)
    assert np.allclose(g.posterior, before, atol=1e-12)


def test_flatten_identity_and_limit():
    g = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=101)
    for _ in range(50):
        g.update_harm(1, 0)
    ref = g.posterior.copy()
    g.flatten(1.0)
    assert np.allclose(g.posterior, ref, atol=1e-12)
    g.flatten(0.0)
    assert np.allclose(g.posterior, np.full_like(ref, 1.0 / ref.size), atol=1e-12)


def test_eig_matches_brute_force_mutual_information():
    g = ParameterGrid.from_beta_prior(3, 7, 4, 6, n=151)
    for k in (0, 1, 3, 6):
        post, lk = g.posterior, g.lam_grid(k)
        marg = float((post * lk).sum())
        brute = binary_entropy(marg) - float((post * binary_entropy(lk)).sum())
        assert g.eig_harm(k) == pytest.approx(brute, rel=1e-12)


def test_eig_is_non_negative_and_zero_for_a_degenerate_design():
    g = ParameterGrid.from_beta_prior(3, 7, 4, 6, n=101)
    assert all(g.eig_harm(k) >= -1e-15 for k in range(8))
    # A check with chance-level sensitivity carries no information at all.
    assert g.eig_check(2, 0.5, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_targeted_eig_of_response_prevention_is_about_q_not_p():
    g = ParameterGrid.from_beta_prior(3, 7, 4, 6, n=151)
    assert g.eig_harm_targeted(0, target="q") > 1e-3
    assert g.eig_harm_targeted(0, target="p") == pytest.approx(0.0, abs=1e-12)


def test_grid_resolution_convergence():
    """The posterior mean is stable across grid resolutions."""
    means = []
    for n in (101, 201, 401):
        g = ParameterGrid.from_beta_prior(3, 7, 2, 8, n=n)
        for _ in range(100):
            g.update_harm(3, 0)
        g.update_harm(3, 1, n_obs=1)
        means.append(g.mean())
    means = np.array(means)
    assert np.abs(means[-1] - means[0]).max() < 5e-3
