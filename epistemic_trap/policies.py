r"""Policies: how a belief about ``(q, p)`` becomes a number of preventive repetitions.

A policy is the object that turns a posterior into an *experimental design*.  The
central claim of the project lives here: a policy that is optimal for immediate
harm minimisation concentrates all of its mass on a single ``k``, and a
single-``k`` design is exactly the design under which ``q`` and ``p`` are
non-identifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import BETA_CHOICE, COST, K_MAX, KAPPA_RISK, TAU_INFO

__all__ = [
    "expected_cost",
    "action_utilities",
    "Policy",
    "GreedyPolicy",
    "FixedKPolicy",
    "RandomizedPolicy",
    "ScheduledPolicy",
    "QuotaPolicy",
]


def expected_cost(grid, ks, cost=None, kappa=KAPPA_RISK):
    r"""Subjective expected cost of committing to ``k`` preventive repetitions.

    .. math::
        C(k) = c_{\mathrm{harm}}
               \left(\mathbb E[\lambda_k] + \kappa\,\mathrm{SD}[\lambda_k]\right)
             + c_{\mathrm{wash}}\, k .

    ``kappa > 0`` makes the agent uncertainty-averse: it pays not only for the
    expected hazard but for its *uncertainty* about the hazard.  This is the
    formal handle on intolerance of uncertainty, and it is what closes the
    self-maintaining loop, because parameter uncertainty then directly buys extra
    repetitions, which in turn suppress the evidence that would reduce it.
    """
    cost = COST if cost is None else cost
    out = np.empty(len(ks))
    for i, k in enumerate(ks):
        m = grid.mean_lambda(k)
        s = grid.sd_lambda(k) if kappa else 0.0
        out[i] = cost["harm"] * (m + kappa * s) + cost["wash"] * k
    return out


def action_utilities(grid, ks, cost=None, kappa=KAPPA_RISK, tau=TAU_INFO,
                     info_target="joint"):
    r"""Utility of each ``k``: negative expected cost plus an epistemic bonus.

    .. math:: U(k) = -C(k) + \tau\,\mathrm{EIG}(k).

    ``tau`` is the *information-seeking weight*.  ``tau = 0`` is the purely
    safety-driven agent of the main simulations; ``tau > 0`` is the agent that is
    willing to pay for causal knowledge, and it is the third axis of the
    supplementary phase diagram.
    """
    u = -expected_cost(grid, ks, cost=cost, kappa=kappa)
    if tau:
        if info_target == "joint":
            eig = np.array([grid.eig_harm(k) for k in ks])
        else:
            eig = np.array([grid.eig_harm_targeted(k, target=info_target) for k in ks])
        u = u + tau * eig
    return u


@dataclass
class Policy:
    """Base class. Subclasses implement :meth:`k_distribution`."""

    k_values: np.ndarray = field(default_factory=lambda: np.arange(K_MAX + 1))

    def k_distribution(self, grid, t=0):
        raise NotImplementedError

    def sample(self, grid, rng, t=0):
        probs = self.k_distribution(grid, t=t)
        return int(rng.choice(self.k_values, p=probs))

    @property
    def name(self):
        return type(self).__name__


@dataclass
class GreedyPolicy(Policy):
    r"""Softmax over subjectively optimal ``k``.

    With a large ``beta`` this is the harm-minimising agent: it puts essentially
    all of its probability mass on ``argmin_k C(k)``.  Its design matrix is
    rank one, and so its Fisher information about ``(q, p)`` is singular.
    """

    beta: float = BETA_CHOICE
    kappa: float = KAPPA_RISK
    tau: float = TAU_INFO
    cost: dict = None
    info_target: str = "joint"

    def k_distribution(self, grid, t=0):
        u = action_utilities(grid, self.k_values, cost=self.cost,
                             kappa=self.kappa, tau=self.tau,
                             info_target=self.info_target)
        z = self.beta * (u - u.max())
        w = np.exp(z)
        return w / w.sum()


@dataclass
class FixedKPolicy(Policy):
    """Always performs the same number of repetitions (pure response prevention
    when ``k = 0``)."""

    k: int = 0

    def k_distribution(self, grid, t=0):
        probs = np.zeros(len(self.k_values))
        probs[int(np.where(self.k_values == self.k)[0][0])] = 1.0
        return probs


@dataclass
class RandomizedPolicy(Policy):
    """Draws ``k`` from a fixed distribution -- a deliberately *exciting* design."""

    probs: np.ndarray = None

    def k_distribution(self, grid, t=0):
        if self.probs is None:
            return np.full(len(self.k_values), 1.0 / len(self.k_values))
        p = np.asarray(self.probs, dtype=float)
        return p / p.sum()


@dataclass
class ScheduledPolicy(Policy):
    """Graded exposure: ``k`` follows a prescribed schedule ``k(t)``.

    Used for graded ERP, where the support of the policy is widened over time
    rather than collapsed to a single value.
    """

    schedule: callable = None

    def k_distribution(self, grid, t=0):
        k = int(self.schedule(t))
        probs = np.zeros(len(self.k_values))
        probs[int(np.where(self.k_values == k)[0][0])] = 1.0
        return probs

    def sample(self, grid, rng, t=0):
        return int(self.schedule(t))


@dataclass
class QuotaPolicy(Policy):
    r"""Greedy safety behaviour with a standing quota of response prevention.

    With probability ``epsilon`` the agent performs ``k_probe`` repetitions
    (``0`` by default) regardless of what its beliefs recommend; otherwise it acts
    greedily.  Behaviourally this is a small, bounded amount of extra risk.
    Epistemically it is the difference between a rank-one design and a full-rank
    one, so it is the minimal intervention that keeps the agent's own efficacy
    identifiable.
    """

    beta: float = BETA_CHOICE
    kappa: float = KAPPA_RISK
    tau: float = TAU_INFO
    cost: dict = None
    epsilon: float = 0.25
    k_probe: int = 0

    def k_distribution(self, grid, t=0):
        u = action_utilities(grid, self.k_values, cost=self.cost,
                             kappa=self.kappa, tau=self.tau)
        z = self.beta * (u - u.max())
        w = np.exp(z)
        w = w / w.sum()
        probe = np.zeros(len(self.k_values))
        probe[int(np.where(self.k_values == self.k_probe)[0][0])] = 1.0
        return (1.0 - self.epsilon) * w + self.epsilon * probe
