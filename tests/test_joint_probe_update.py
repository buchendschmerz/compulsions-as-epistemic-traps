"""Joint evidence must respect the shared latent state within an episode."""

import numpy as np
import pytest

from epistemic_trap.config import PRIORS
from epistemic_trap.grid import ParameterGrid


@pytest.mark.parametrize("k", [0, 1, 4])
@pytest.mark.parametrize("harm", [0, 1])
@pytest.mark.parametrize("report", [0, 1])
def test_baseline_joint_update_matches_latent_state_enumeration(k, harm, report):
    grid = ParameterGrid.from_beta_prior(**PRIORS["severe_doubt"], n=31)
    prior = grid.posterior.copy()
    likelihood = np.zeros_like(prior)
    for initial in (0, 1):
        initial_prob = grid.Q if initial else 1.0 - grid.Q
        report_dirty_prob = 0.8 if initial else 0.2
        report_prob = report_dirty_prob if report else 1.0 - report_dirty_prob
        harm_prob = (1.0 - grid.P) ** k if initial else np.zeros_like(grid.P)
        outcome_prob = harm_prob if harm else 1.0 - harm_prob
        likelihood += initial_prob * report_prob * outcome_prob
    expected = prior * likelihood
    expected /= expected.sum()
    grid.update_episode(k, harm, "baseline", report)
    np.testing.assert_allclose(grid.posterior, expected, atol=1e-14)


@pytest.mark.parametrize("harm", [0, 1])
@pytest.mark.parametrize("report", [0, 1])
def test_post_action_probe_adds_no_parameter_evidence_after_terminal_outcome(harm, report):
    grid = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    outcome_only = grid.copy().update_harm(3, harm)
    grid.update_episode(3, harm, "reassurance", report)
    np.testing.assert_allclose(grid.posterior, outcome_only.posterior, atol=1e-14)


@pytest.mark.parametrize("harm", [0, 1])
def test_uninformative_baseline_probe_reduces_to_outcome_only(harm):
    grid = ParameterGrid.from_beta_prior(**PRIORS["mild_doubt"])
    outcome_only = grid.copy().update_harm(2, harm)
    grid.update_episode(2, harm, "baseline", 1, 0.5, 0.5)
    np.testing.assert_allclose(grid.posterior, outcome_only.posterior, atol=1e-14)
