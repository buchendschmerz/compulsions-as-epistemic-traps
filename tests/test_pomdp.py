"""Rigoux et al. (2024) POMDP replication: solver and belief-update correctness."""

import numpy as np
import pytest

from epistemic_trap.pomdp import (CHECK, COOK, WASH, PomdpParams, belief_predict,
                                  check_posterior, compulsive_episodes,
                                  immediate_reward, sample_params,
                                  simulate_sequence, solve_pomdp)


def test_transition_table_matches_the_paper():
    prm = PomdpParams(p_soiling=0.3, p_success=0.8)
    assert belief_predict(0.0, COOK, prm) == pytest.approx(0.3)   # clean -> soiled
    assert belief_predict(1.0, COOK, prm) == pytest.approx(1.0)   # dirty stays dirty
    assert belief_predict(1.0, WASH, prm) == pytest.approx(0.2)   # 1 - p_SUCCESS
    assert belief_predict(0.0, WASH, prm) == pytest.approx(0.0)   # clean stays clean
    for b in (0.0, 0.37, 1.0):
        assert belief_predict(b, CHECK, prm) == pytest.approx(b)  # check moves nothing


def test_check_update_is_bayes_rule():
    prm = PomdpParams(d_dirty=0.9, d_clean=0.7)
    b = 0.4
    p_d, b_if_d, b_if_c = check_posterior(b, prm)
    assert p_d == pytest.approx(0.9 * 0.4 + 0.3 * 0.6)
    assert b_if_d == pytest.approx(0.9 * 0.4 / p_d)
    assert b_if_c == pytest.approx(0.1 * 0.4 / (1 - p_d))
    # the two posteriors must average back to the prior
    assert p_d * b_if_d + (1 - p_d) * b_if_c == pytest.approx(b)


def test_perfect_check_resolves_the_state():
    prm = PomdpParams(d_dirty=1.0, d_clean=1.0)
    _, b_if_d, b_if_c = check_posterior(0.4, prm)
    assert b_if_d == pytest.approx(1.0)
    assert b_if_c == pytest.approx(0.0)


def test_reward_table_matches_the_paper():
    prm = PomdpParams(c_sick=0.7, c_wash=0.2, c_check=0.1)
    r = immediate_reward(np.array([0.0, 1.0]), prm)
    assert r[COOK][0] == pytest.approx(0.0)     # reward_DINNER fixed at zero
    assert r[COOK][1] == pytest.approx(-0.7)
    assert np.allclose(r[WASH], -0.2)
    assert np.allclose(r[CHECK], -0.1)


def test_value_iteration_converges_and_is_monotone_in_belief():
    prm = PomdpParams(c_sick=0.6, c_wash=0.25, c_check=0.15, gamma=0.9)
    b, V, pol = solve_pomdp(prm, n_belief=401)
    assert np.all(np.diff(V) <= 1e-6)           # dirtier belief is never better
    assert pol[0] == COOK                        # cook when certainly clean
    assert V.max() <= 1e-9                       # all outcomes are costs


def test_value_iteration_is_resolution_stable():
    prm = PomdpParams(c_sick=0.6, c_wash=0.25, c_check=0.15, gamma=0.9)
    _, V1, _ = solve_pomdp(prm, n_belief=201)
    b2, V2, _ = solve_pomdp(prm, n_belief=801)
    assert np.max(np.abs(np.interp(b2, np.linspace(0, 1, 201), V1) - V2)) < 5e-3


def test_zero_discount_gives_the_myopic_policy():
    prm = PomdpParams(c_sick=0.9, c_wash=0.05, c_check=0.05, gamma=0.0)
    b, V, pol = solve_pomdp(prm, n_belief=201)
    # myopic: cook while -c_sick*b exceeds -min(c_wash, c_check)
    cutoff = min(prm.c_wash, prm.c_check) / prm.c_sick
    assert np.all(pol[b < cutoff - 0.02] == COOK)


def test_compulsive_episode_definition():
    # cook-wash-check-cook is not compulsive; a repeat of either action is
    assert compulsive_episodes([COOK, WASH, CHECK, COOK])["n_episodes"] == 0
    assert compulsive_episodes([COOK, WASH, WASH, COOK])["n_episodes"] == 1
    assert compulsive_episodes([COOK, CHECK, CHECK, COOK])["n_episodes"] == 1
    assert compulsive_episodes([COOK, CHECK, WASH, CHECK, COOK])["n_episodes"] == 1
    r = compulsive_episodes([COOK, WASH, WASH, COOK, CHECK, CHECK, CHECK, COOK])
    assert r["n_episodes"] == 2 and r["max_len"] == 3 and r["n_cook"] == 3


def test_sampled_parameters_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(200):
        prm = sample_params(rng)
        assert 0 <= prm.p_soiling <= 1 and 0 <= prm.p_success <= 1
        assert 0.5 <= prm.d_dirty <= 1 and 0.5 <= prm.d_clean <= 1
        assert prm.c_sick + prm.c_wash + prm.c_check == pytest.approx(1.0)


def test_simulation_is_reproducible():
    prm = PomdpParams()
    a1, s1 = simulate_sequence(prm, prm, 200, np.random.default_rng(3))
    a2, s2 = simulate_sequence(prm, prm, 200, np.random.default_rng(3))
    assert np.array_equal(a1, a2) and np.array_equal(s1, s2)


def test_underestimating_efficacy_never_reduces_preventive_effort():
    """The Rigoux result, as a property of the solved policy.

    An agent that believes washing works less well should not wash *less*.
    """
    world = PomdpParams(p_soiling=0.35, p_success=0.85, c_sick=0.6,
                        c_wash=0.25, c_check=0.15, gamma=0.9)
    rng = np.random.default_rng(11)
    fracs = []
    for belief_p in (0.85, 0.45, 0.15):
        agent = PomdpParams(**{**world.__dict__, "p_success": belief_p})
        actions, _ = simulate_sequence(world, agent, 3000, rng)
        fracs.append(compulsive_episodes(actions)["frac_noncook"])
    assert fracs[0] <= fracs[1] + 1e-9 <= fracs[2] + 1e-9
