r"""Episode-level generative simulation of the Bayesian safety-behaviour agent.

Generative model of one episode
-------------------------------
1. The episode starts in the hazardous state with probability ``q`` (world).
2. The agent commits to ``k`` preventive repetitions drawn from its policy.
3. Each repetition removes the hazard with probability ``p`` (world).
4. Optionally the agent performs a *check*: a noisy observation of the state
   either **before** any repetition (``check_mode="baseline"``) or **after** all
   of them (``check_mode="reassurance"``).
5. The agent commits to the consequential action; harm occurs iff the hazard is
   still present.
6. The agent updates its grid posterior over ``(q, p)`` on the harm outcome and
   on the check, if any.

The agent's belief updating is exactly Bayesian and its credit assignment is
exact.  Anything pathological that emerges is therefore a property of the
*design* the policy generates, not of the learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import K_MAX, N_GRID, WORLD
from .grid import ParameterGrid, binary_entropy
from .hazard import fisher_information_policy, identifiability_summary

SINGULAR_COND = 1e10  # condition number above which a realised design is singular

__all__ = ["World", "AgentSpec", "run_agent", "policy_empirical_distribution",
           "commit_state_prob", "SINGULAR_COND"]


@dataclass
class World:
    """Objective environment parameters."""

    q: float = WORLD["q"]
    p: float = WORLD["p"]
    d_dirty: float = WORLD["d_dirty"]
    d_clean: float = WORLD["d_clean"]

    def run_episode(self, k, rng, check_mode=None):
        """Return ``(harm, check_obs, m_check)`` for one episode with ``k`` repetitions."""
        state_dirty = rng.random() < self.q
        check_obs, m_check = None, None
        if check_mode == "baseline":
            check_obs, m_check = self._observe(state_dirty, rng), 0
        for _ in range(int(k)):
            if state_dirty and rng.random() < self.p:
                state_dirty = False
        if check_mode == "reassurance":
            check_obs, m_check = self._observe(state_dirty, rng), int(k)
        return int(state_dirty), check_obs, m_check

    def _observe(self, state_dirty, rng):
        """Noisy check: reports 'dirty' with sensitivity/specificity ``d_*``."""
        if state_dirty:
            return int(rng.random() < self.d_dirty)
        return int(rng.random() >= self.d_clean)


@dataclass
class AgentSpec:
    """Everything that defines one simulated individual."""

    prior: dict
    policy: object
    check_mode: str = None          # None | "baseline" | "reassurance"
    gamma_forget: float = 1.0       # 1.0 = no forgetting
    n_grid: int = N_GRID
    label: str = "agent"
    meta: dict = field(default_factory=dict)

    def new_grid(self):
        return ParameterGrid.from_beta_prior(n=self.n_grid, **self.prior)


def commit_state_prob(grid, k, check_mode, obs, d_dirty, d_clean):
    r"""Agent's probability that the hazard is present at the moment of commitment.

    This is *state* uncertainty -- "am I safe right now?" -- as opposed to the
    *structural* uncertainty about ``(q, p)`` carried by the parameter posterior.
    Keeping the two apart is what makes the reassurance paradox (H5) measurable:
    a check performed after the safety action collapses the former while leaving
    the latter almost untouched.

    ``check_mode="baseline"`` probes the state *before* the preventive actions, so
    its evidence must be propagated through ``k`` repetitions before commitment.
    """
    post = grid.posterior
    if check_mode is None or obs is None:
        return grid.mean_lambda(k)
    lik_dirty = d_dirty if obs else 1.0 - d_dirty
    lik_clean = (1.0 - d_clean) if obs else d_clean
    if check_mode == "reassurance":
        lam_k = grid.lam_grid(k)
        num = float((post * lam_k * lik_dirty).sum())
        den = num + float((post * (1.0 - lam_k) * lik_clean).sum())
        return num / den if den > 0 else grid.mean_lambda(k)
    if check_mode == "baseline":
        w_dirty = post * grid.Q * lik_dirty
        w_clean = post * (1.0 - grid.Q) * lik_clean
        den = float(w_dirty.sum() + w_clean.sum())
        if den <= 0:
            return grid.mean_lambda(k)
        return float((w_dirty * (1.0 - grid.P) ** k).sum()) / den
    raise ValueError(f"unknown check_mode {check_mode!r}")


def policy_empirical_distribution(k_history, k_values=None):
    """Empirical distribution over ``k`` actually realised by an agent."""
    k_values = np.arange(K_MAX + 1) if k_values is None else np.asarray(k_values)
    counts = np.array([(np.asarray(k_history) == k).sum() for k in k_values], dtype=float)
    total = counts.sum()
    return counts / total if total else counts


def run_agent(world, spec, n_episodes, seed, log_every=1, k_ref_mode="modal",
              record_fisher=True, grid=None, rng=None, t_offset=0,
              k_history=None, harm_history=None):
    """Simulate one agent for ``n_episodes`` and return a per-episode log.

    Parameters
    ----------
    k_ref_mode : {"modal", "current"}
        Which ``k`` to use as the reference when reporting the entropy of the
        observable composite ``lambda_k``.  ``"modal"`` uses the agent's most
        frequent ``k`` so far -- the quantity the agent actually predicts well.

    Returns
    -------
    pandas.DataFrame
        One row per logged episode, carrying posterior moments, entropies,
        the realised action, and the identifiability of the *realised* design.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    grid = spec.new_grid() if grid is None else grid
    rows = []
    k_history = [] if k_history is None else list(k_history)
    harm_history = [] if harm_history is None else list(harm_history)
    true_world = np.array([world.q, world.p])

    for t in range(t_offset, t_offset + n_episodes):
        k = spec.policy.sample(grid, rng, t=t)
        harm, check_obs, m_check = world.run_episode(k, rng, check_mode=spec.check_mode)
        k_history.append(k)
        harm_history.append(harm)

        # State uncertainty is read off the belief the agent holds when it commits,
        # i.e. after any check it just performed but before the outcome arrives.
        b_pre = grid.mean_lambda(k)
        b_commit = commit_state_prob(grid, k, spec.check_mode, check_obs,
                                     world.d_dirty, world.d_clean)

        # -- Bayesian update: harm outcome first, then the check (order irrelevant)
        grid.update_harm(k, harm)
        if check_obs is not None:
            grid.update_check(m_check, bool(check_obs),
                              world.d_dirty, world.d_clean)
        grid.flatten(spec.gamma_forget)

        if (t + 1) % log_every == 0 or t == n_episodes - 1:
            k_ref = int(np.bincount(k_history).argmax()) if k_ref_mode == "modal" else k
            row = dict(episode=t + 1, label=spec.label, k=k, harm=harm,
                       check_obs=-1 if check_obs is None else int(check_obs),
                       H_state_pre=float(binary_entropy(b_pre)),
                       H_state_commit=float(binary_entropy(b_commit)),
                       p_hazard_commit=float(b_commit))
            if hasattr(spec.policy, "responsibility"):
                row["responsibility"] = float(spec.policy.responsibility)
            row.update(grid.summary(k_ref=k_ref, truth=(world.q, world.p)))
            row["err_p"] = row["mean_p"] - world.p
            row["err_q"] = row["mean_q"] - world.q
            row["mean_k_sofar"] = float(np.mean(k_history))
            row["var_k_sofar"] = float(np.var(k_history))
            row["harm_rate_sofar"] = float(np.mean(harm_history))
            row["eig_current_k"] = grid.eig_harm(k)
            if record_fisher:
                emp = policy_empirical_distribution(k_history)
                info = fisher_information_policy(true_world[0], true_world[1],
                                                 np.arange(K_MAX + 1), emp,
                                                 n_episodes=len(k_history))
                summ = identifiability_summary(info)
                # Declare singularity on the (scale-free) condition number: the
                # determinant of a rank-one matrix is only zero up to floating
                # point, so a raw det test is not reliable.
                singular = bool(np.count_nonzero(emp) < 2
                                or not np.isfinite(summ["cond"])
                                or summ["cond"] > SINGULAR_COND)
                row["fim_singular"] = singular
                row["fim_det"] = 0.0 if singular else float(summ["det"])
                row["fim_log10_det"] = -np.inf if singular else float(summ["log10_det"])
                row["fim_cond"] = float(summ["cond"])
                row["fim_eig_min"] = float(summ["eig_min"])
                row["k_support"] = int(np.count_nonzero(emp))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["grid"] = grid
    df.attrs["rng"] = rng
    df.attrs["k_history"] = np.array(k_history)
    df.attrs["harm_history"] = np.array(harm_history)
    df.attrs["t_end"] = t_offset + n_episodes
    return df
