r"""Replication of the Rigoux, Stephan & Petzschner (2024) POMDP handwashing model.

States ``S = {clean, dirty}``, actions ``A = {cook, wash, check}``, observations
``O = {clean, dirty}``.

Transitions (their Table 3)::

    T(dirty | cook,  clean) = p_SOILING        T(dirty | cook,  dirty) = 1
    T(clean | wash,  dirty) = p_SUCCESS        T(clean | wash,  clean) = 1
    T(. | check, .) = identity

Rewards (their Table 4), with ``reward_DINNER`` fixed to 0 and the three costs
normalised to sum to one::

    R(cook, clean) = 0     R(cook, dirty) = -cost_SICKNESS
    R(wash, .)     = -cost_WASH
    R(check, .)    = -cost_CHECK

Observations (their Table 5) are at chance for cook and wash and carry
sensitivity ``p_DETECT_DIRTY`` / specificity ``p_DETECT_CLEAN`` for check.

Because the state space has two elements, the belief simplex is the interval
``[0, 1]`` and the value function can be solved by discounted value iteration on a
fine belief grid with linear interpolation -- an exact-in-the-limit substitute for
the ``pomdp-solve`` incremental-pruning solver used in the original paper.

Following the original, each simulated individual carries *two* parameter sets:
objective **world** parameters that generate transitions and observations, and
subjective **agent** parameters (primed in the paper) that generate the policy and
the belief update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PomdpParams", "solve_pomdp", "simulate_sequence",
           "compulsive_episodes", "sample_params", "PARAM_NAMES", "ACTION_NAMES",
           "COOK", "WASH", "CHECK"]

PARAM_NAMES = ["p_soiling", "p_success", "d_dirty", "d_clean",
               "c_sick", "c_wash", "c_check", "gamma"]

COOK, WASH, CHECK = 0, 1, 2
ACTION_NAMES = ["cook", "wash", "check"]


@dataclass
class PomdpParams:
    """One full parameter set (either the world's or an agent's beliefs)."""

    p_soiling: float = 0.30
    p_success: float = 0.80
    d_dirty: float = 0.80
    d_clean: float = 0.80
    c_sick: float = 0.70
    c_wash: float = 0.20
    c_check: float = 0.10
    gamma: float = 0.90

    def as_array(self):
        return np.array([getattr(self, n) for n in PARAM_NAMES])


def sample_params(rng, gamma_range=(0.60, 0.98)):
    """Sample a valid parameter set the way the original paper does.

    Transition probabilities are uniform on ``[0, 1]``; detection probabilities
    are uniform on ``[0.5, 1]`` (state inference is symmetric about chance); the
    three costs are uniform on the unit 2-simplex.
    """
    c = rng.dirichlet(np.ones(3))
    return PomdpParams(
        p_soiling=float(rng.uniform(0, 1)),
        p_success=float(rng.uniform(0, 1)),
        d_dirty=float(rng.uniform(0.5, 1)),
        d_clean=float(rng.uniform(0.5, 1)),
        c_sick=float(c[0]), c_wash=float(c[1]), c_check=float(c[2]),
        gamma=float(rng.uniform(*gamma_range)),
    )


# --------------------------------------------------------------------- dynamics
def belief_predict(b, action, prm):
    """Belief about ``dirty`` after the transition induced by ``action``."""
    if action == COOK:
        return b + (1.0 - b) * prm.p_soiling
    if action == WASH:
        return b * (1.0 - prm.p_success)
    return b  # check does not move the state


def check_posterior(b, prm):
    """Belief update after each possible check observation.

    Returns ``(p_obs_dirty, b_given_dirty, b_given_clean)``.
    """
    p_dirty = prm.d_dirty * b + (1.0 - prm.d_clean) * (1.0 - b)
    p_dirty = np.clip(p_dirty, 1e-12, 1.0 - 1e-12)
    b_if_dirty = prm.d_dirty * b / p_dirty
    b_if_clean = (1.0 - prm.d_dirty) * b / (1.0 - p_dirty)
    return p_dirty, np.clip(b_if_dirty, 0, 1), np.clip(b_if_clean, 0, 1)


def immediate_reward(b, prm):
    """Expected immediate reward of each action at belief ``b`` (dirty prob.)."""
    r_cook = -prm.c_sick * b                      # reward_DINNER fixed at 0
    r_wash = -prm.c_wash * np.ones_like(np.asarray(b, dtype=float))
    r_check = -prm.c_check * np.ones_like(np.asarray(b, dtype=float))
    return np.stack([r_cook, r_wash, r_check], axis=0)


# ----------------------------------------------------------------------- solver
def solve_pomdp(prm, n_belief=401, tol=1e-8, max_iter=3000):
    """Discounted value iteration on a discretised belief interval.

    Returns
    -------
    grid : ndarray
        Belief grid (probability that the hands are dirty).
    V : ndarray
        Optimal value at each grid point.
    policy : ndarray
        Index of the optimal action at each grid point.
    """
    b = np.linspace(0.0, 1.0, n_belief)
    V = np.zeros(n_belief)
    R = immediate_reward(b, prm)

    b_cook = belief_predict(b, COOK, prm)
    b_wash = belief_predict(b, WASH, prm)
    p_dirty, b_chk_d, b_chk_c = check_posterior(b, prm)

    for _ in range(max_iter):
        interp = lambda x: np.interp(x, b, V)  # noqa: E731
        q = np.empty((3, n_belief))
        q[COOK] = R[COOK] + prm.gamma * interp(b_cook)
        q[WASH] = R[WASH] + prm.gamma * interp(b_wash)
        q[CHECK] = (R[CHECK] + prm.gamma *
                    (p_dirty * interp(b_chk_d) + (1.0 - p_dirty) * interp(b_chk_c)))
        V_new = q.max(axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    policy = q.argmax(axis=0)
    return b, V, policy


def _act(policy_grid, belief_grid, b):
    """Look up the optimal action at belief ``b``."""
    return int(policy_grid[int(np.argmin(np.abs(belief_grid - b)))])


# -------------------------------------------------------------------- simulation
def simulate_sequence(world, agent, n_steps, rng, n_belief=401, solved=None):
    """Generate an action sequence for an agent acting under subjective beliefs.

    Mirrors the pseudo-algorithm of the original paper: the *agent* parameters
    determine the policy and the belief update, the *world* parameters determine
    the realised state transitions and observations.
    """
    belief_grid, _, policy_grid = solved if solved is not None else solve_pomdp(agent, n_belief)
    state_dirty = rng.random() < world.p_soiling
    b = 0.5
    actions = np.empty(n_steps, dtype=int)
    states = np.empty(n_steps, dtype=int)

    for t in range(n_steps):
        a = _act(policy_grid, belief_grid, b)
        actions[t] = a
        states[t] = int(state_dirty)

        if a == COOK:
            if not state_dirty and rng.random() < world.p_soiling:
                state_dirty = True
            b = belief_predict(b, COOK, agent)
        elif a == WASH:
            if state_dirty and rng.random() < world.p_success:
                state_dirty = False
            b = belief_predict(b, WASH, agent)
        else:  # CHECK -- the state does not change, but an observation arrives
            if state_dirty:
                obs_dirty = rng.random() < world.d_dirty
            else:
                obs_dirty = rng.random() >= world.d_clean
            p_d, b_if_d, b_if_c = check_posterior(b, agent)
            b = b_if_d if obs_dirty else b_if_c
        b = float(np.clip(b, 0.0, 1.0))
    return actions, states


def compulsive_episodes(actions):
    """Episodes of consecutive non-cook actions containing a repeated action.

    The original definition: a run of non-cook actions counts as a compulsive
    episode when either wash or check occurs at least twice in it.

    Returns
    -------
    dict
        ``n_episodes`` (compulsive runs), ``max_len``, ``mean_len``,
        ``n_runs`` (all non-cook runs), and ``frac_noncook``.
    """
    runs, cur = [], []
    for a in actions:
        if a == COOK:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(a)
    if cur:
        runs.append(cur)

    comp = [r for r in runs
            if r.count(WASH) >= 2 or r.count(CHECK) >= 2]
    return dict(
        n_episodes=len(comp),
        n_runs=len(runs),
        max_len=max((len(r) for r in comp), default=0),
        mean_len=float(np.mean([len(r) for r in comp])) if comp else 0.0,
        frac_noncook=float(np.mean(np.asarray(actions) != COOK)),
        n_cook=int(np.sum(np.asarray(actions) == COOK)),
    )
