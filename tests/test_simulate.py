"""Generative simulation, policies, and end-to-end reproducibility."""

import numpy as np
import pytest

from epistemic_trap.config import PRIORS
from epistemic_trap.grid import ParameterGrid
from epistemic_trap.policies import (FixedKPolicy, GreedyPolicy, RandomizedPolicy,
                                     ScheduledPolicy, action_utilities,
                                     expected_cost)
from epistemic_trap.simulate import (AgentSpec, World, policy_empirical_distribution,
                                     run_agent)


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_episode_harm_frequency_matches_the_analytic_hazard(k):
    w = World(q=0.4, p=0.6)
    rng = np.random.default_rng(0)
    n = 200_000
    harms = np.mean([w.run_episode(k, rng)[0] for _ in range(n)])
    expected = 0.4 * 0.4 ** k
    assert harms == pytest.approx(expected, abs=4 / np.sqrt(n))


def test_check_reports_with_the_specified_sensitivity():
    w = World(q=1.0, p=0.0, d_dirty=0.75, d_clean=0.9)
    rng = np.random.default_rng(1)
    obs = [w.run_episode(0, rng, check_mode="baseline")[1] for _ in range(50_000)]
    assert np.mean(obs) == pytest.approx(0.75, abs=0.01)
    w_clean = World(q=0.0, p=0.5, d_dirty=0.75, d_clean=0.9)
    obs = [w_clean.run_episode(0, rng, check_mode="baseline")[1] for _ in range(50_000)]
    assert np.mean(obs) == pytest.approx(0.10, abs=0.01)


def test_expected_cost_increases_with_wash_cost_and_decreases_with_hazard():
    g = ParameterGrid.from_beta_prior(**PRIORS["accurate"])
    ks = np.arange(6)
    cheap = expected_cost(g, ks, cost=dict(harm=20.0, wash=0.05, check=0.1))
    dear = expected_cost(g, ks, cost=dict(harm=20.0, wash=1.00, check=0.1))
    assert np.argmin(cheap) >= np.argmin(dear)


def test_more_doubt_buys_more_repetitions():
    """The Rigoux mechanism, recovered inside the Bayesian agent."""
    ks = np.arange(11)
    picks = []
    for name in ("accurate", "mild_doubt", "severe_doubt"):
        g = ParameterGrid.from_beta_prior(**PRIORS[name])
        picks.append(int(np.argmax(action_utilities(g, ks))))
    assert picks[0] < picks[1] < picks[2]


def test_uncertainty_aversion_never_reduces_repetitions():
    g = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    ks = np.arange(11)
    k0 = int(np.argmin(expected_cost(g, ks, kappa=0.0)))
    k1 = int(np.argmin(expected_cost(g, ks, kappa=2.0)))
    assert k1 >= k0


def test_information_bonus_drives_the_choice_towards_the_best_experiment():
    """As tau grows the agent stops minimising harm and starts designing experiments."""
    g = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    ks = np.arange(11)
    greedy = int(np.argmax(action_utilities(g, ks, tau=0.0)))
    curious = int(np.argmax(action_utilities(g, ks, tau=1e4, info_target="p")))
    best_experiment = int(np.argmax([g.eig_harm_targeted(k, "p") for k in ks]))
    assert curious == best_experiment
    assert curious != greedy


def test_response_prevention_alone_is_uninformative_about_efficacy():
    """EIG about p is exactly zero at k = 0, for every belief.

    This is the formal correction to the intuition that response prevention by
    itself restores identifiability: it is a perfect probe of the baseline hazard
    and a null probe of one's own efficacy.
    """
    for name in PRIORS:
        g = ParameterGrid.from_beta_prior(**PRIORS[name])
        assert g.eig_harm_targeted(0, "p") == pytest.approx(0.0, abs=1e-12)
        assert g.eig_harm_targeted(0, "q") > 1e-3
        assert g.eig_harm_targeted(0, "q") == max(
            g.eig_harm_targeted(k, "q") for k in range(11))


def test_policies_return_valid_distributions():
    g = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    ks = np.arange(11)
    for pol in (GreedyPolicy(k_values=ks), FixedKPolicy(k_values=ks, k=3),
                RandomizedPolicy(k_values=ks, probs=np.arange(11.0) + 1)):
        d = pol.k_distribution(g)
        assert d.shape == ks.shape
        assert np.all(d >= 0) and d.sum() == pytest.approx(1.0)


def test_fixed_policy_is_degenerate_and_scheduled_policy_follows_its_schedule():
    ks = np.arange(11)
    g = ParameterGrid.from_beta_prior(**PRIORS["accurate"])
    rng = np.random.default_rng(0)
    assert all(FixedKPolicy(k_values=ks, k=4).sample(g, rng, t=t) == 4 for t in range(20))
    sched = ScheduledPolicy(k_values=ks, schedule=lambda t: t % 4)
    assert [sched.sample(g, rng, t=t) for t in range(6)] == [0, 1, 2, 3, 0, 1]


def test_empirical_policy_distribution():
    d = policy_empirical_distribution([0, 0, 2, 2, 2, 5])
    assert d[0] == pytest.approx(2 / 6) and d[2] == pytest.approx(3 / 6)
    assert d.sum() == pytest.approx(1.0)


def test_run_agent_is_reproducible_given_a_seed():
    w = World()
    spec = AgentSpec(prior=PRIORS["mild_doubt"], policy=GreedyPolicy())
    a = run_agent(w, spec, 60, seed=5, log_every=20)
    b = run_agent(w, AgentSpec(prior=PRIORS["mild_doubt"], policy=GreedyPolicy()),
                  60, seed=5, log_every=20)
    assert np.allclose(a.mean_p.values, b.mean_p.values)
    assert np.array_equal(a.attrs["k_history"], b.attrs["k_history"])


def test_fixed_k_agent_realises_a_singular_design_and_variable_k_does_not():
    w = World()
    ks = np.arange(11)
    fixed = run_agent(w, AgentSpec(prior=PRIORS["severe_doubt"],
                                   policy=FixedKPolicy(k_values=ks, k=4)),
                      200, seed=2, log_every=100)
    varied = run_agent(w, AgentSpec(prior=PRIORS["severe_doubt"],
                                    policy=RandomizedPolicy(
                                        k_values=ks,
                                        probs=np.r_[np.ones(4), np.zeros(7)])),
                       200, seed=2, log_every=100)
    assert bool(fixed.iloc[-1].fim_singular) is True
    assert bool(varied.iloc[-1].fim_singular) is False


def test_variable_k_recovers_efficacy_while_fixed_k_does_not():
    """End-to-end statement of the central claim."""
    w = World(q=0.3, p=0.8)
    ks = np.arange(11)
    fixed = run_agent(w, AgentSpec(prior=PRIORS["severe_doubt"],
                                   policy=FixedKPolicy(k_values=ks, k=5)),
                      1500, seed=4, log_every=1500)
    varied = run_agent(w, AgentSpec(prior=PRIORS["severe_doubt"],
                                    policy=RandomizedPolicy(
                                        k_values=ks,
                                        probs=np.r_[np.ones(4), np.zeros(7)])),
                       1500, seed=4, log_every=1500)
    assert abs(varied.iloc[-1].mean_p - 0.8) < 0.12
    assert abs(fixed.iloc[-1].mean_p - 0.8) > 0.30
    assert fixed.iloc[-1].corr_qp > varied.iloc[-1].corr_qp


# --------------------------------------------------------------- state uncertainty
def test_commit_state_prob_without_a_check_is_the_predictive_hazard():
    g = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    from epistemic_trap.simulate import commit_state_prob
    for k in (0, 2, 5):
        assert commit_state_prob(g, k, None, None, 0.8, 0.8) == pytest.approx(
            g.mean_lambda(k))


def test_a_perfect_reassurance_check_resolves_the_state_completely():
    g = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    from epistemic_trap.simulate import commit_state_prob
    assert commit_state_prob(g, 3, "reassurance", 1, 1.0, 1.0) == pytest.approx(1.0)
    assert commit_state_prob(g, 3, "reassurance", 0, 1.0, 1.0) == pytest.approx(0.0)


def test_a_perfect_baseline_check_propagates_through_the_washes():
    """A clean baseline probe implies safety; a dirty one leaves (1-p)^k risk."""
    g = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    from epistemic_trap.simulate import commit_state_prob
    assert commit_state_prob(g, 4, "baseline", 0, 1.0, 1.0) == pytest.approx(0.0)
    post = g.posterior
    expected = float((post * g.Q * (1 - g.P) ** 4).sum() / (post * g.Q).sum())
    assert commit_state_prob(g, 4, "baseline", 1, 1.0, 1.0) == pytest.approx(expected)


def test_an_uninformative_check_leaves_state_belief_unchanged():
    g = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    from epistemic_trap.simulate import commit_state_prob
    for obs in (0, 1):
        assert commit_state_prob(g, 3, "reassurance", obs, 0.5, 0.5) == pytest.approx(
            g.mean_lambda(3))


def test_run_agent_can_be_resumed_without_changing_the_trajectory():
    """Block-structured protocols must be identical to one continuous run."""
    w = World()
    ks = np.arange(11)
    rng_a = np.random.default_rng(99)
    spec = AgentSpec(prior=PRIORS["mild_doubt"], policy=FixedKPolicy(k_values=ks, k=3))
    whole = run_agent(w, spec, 80, seed=None, rng=rng_a, log_every=80)

    rng_b = np.random.default_rng(99)
    spec2 = AgentSpec(prior=PRIORS["mild_doubt"], policy=FixedKPolicy(k_values=ks, k=3))
    first = run_agent(w, spec2, 40, seed=None, rng=rng_b, log_every=40)
    second = run_agent(w, spec2, 40, seed=None, rng=first.attrs["rng"],
                       grid=first.attrs["grid"], t_offset=first.attrs["t_end"],
                       k_history=first.attrs["k_history"],
                       harm_history=first.attrs["harm_history"], log_every=40)
    assert second.iloc[-1].mean_p == pytest.approx(whole.iloc[-1].mean_p, abs=1e-12)
    assert second.iloc[-1].episode == whole.iloc[-1].episode
