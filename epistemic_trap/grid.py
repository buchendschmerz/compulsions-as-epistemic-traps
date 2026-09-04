r"""Grid-based Bayesian posterior over the latent causal parameters ``(q, p)``.

The brief mandates grid inference rather than MCMC: the parameter space is a
two-dimensional unit square, so an :math:`N\times N` grid gives exact,
deterministic, perfectly reproducible posteriors at negligible cost.

All densities are stored as *log* weights and normalised lazily, which keeps the
update numerically stable even after thousands of episodes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .config import N_GRID
from .hazard import lam

EPS = 1e-300
LOGEPS = -700.0

__all__ = ["ParameterGrid", "binary_entropy"]


def binary_entropy(r):
    """Entropy (nats) of a Bernoulli(``r``) variable, safe at the boundaries."""
    r = np.clip(np.asarray(r, dtype=float), 1e-15, 1.0 - 1e-15)
    return -(r * np.log(r) + (1.0 - r) * np.log1p(-r))


def _cell_centres(n):
    """Midpoint grid on the open unit interval; avoids the degenerate 0/1 edges."""
    edges = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


class ParameterGrid:
    r"""Posterior over :math:`(q, p)` on an ``n x n`` midpoint grid.

    Axis 0 indexes ``q`` (base hazard rate), axis 1 indexes ``p`` (efficacy of a
    single preventive action).
    """

    def __init__(self, n=N_GRID, log_w=None):
        self.n = int(n)
        self.q_vals = _cell_centres(self.n)
        self.p_vals = _cell_centres(self.n)
        self.dq = 1.0 / self.n
        self.dp = 1.0 / self.n
        self.Q, self.P = np.meshgrid(self.q_vals, self.p_vals, indexing="ij")
        self.log_w = np.zeros((self.n, self.n)) if log_w is None else np.array(log_w)

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_beta_prior(cls, a_q, b_q, a_p, b_p, n=N_GRID):
        """Independent ``Beta(a_q, b_q) x Beta(a_p, b_p)`` prior, discretised."""
        g = cls(n=n)
        lq = (a_q - 1.0) * np.log(g.q_vals) + (b_q - 1.0) * np.log1p(-g.q_vals)
        lp = (a_p - 1.0) * np.log(g.p_vals) + (b_p - 1.0) * np.log1p(-g.p_vals)
        g.log_w = lq[:, None] + lp[None, :]
        g.normalise()
        return g

    def copy(self):
        return ParameterGrid(n=self.n, log_w=self.log_w.copy())

    # ------------------------------------------------------------- normalisation
    def normalise(self):
        self.log_w = self.log_w - logsumexp(self.log_w)
        return self

    @property
    def posterior(self):
        """Normalised probability mass on each grid cell (sums to one)."""
        return np.exp(self.log_w - logsumexp(self.log_w))

    @property
    def density(self):
        """Posterior *density* (mass divided by cell area)."""
        return self.posterior / (self.dq * self.dp)

    # ------------------------------------------------------------------- updates
    def _add_loglik(self, ll):
        self.log_w = self.log_w + np.clip(ll, LOGEPS, None)
        self.normalise()
        return self

    def lam_grid(self, k):
        r"""``lambda_k = q (1-p)^k`` evaluated on every grid cell."""
        return lam(self.Q, self.P, k)

    def update_harm(self, k, y, n_obs=1):
        """Condition on ``y`` harm events out of ``n_obs`` episodes run at ``k``.

        This is the *only* channel through which a purely outcome-observing agent
        learns, and it sees ``(q, p)`` exclusively through the scalar
        ``lambda_k`` -- the origin of the ridge.
        """
        lk = np.clip(self.lam_grid(k), 1e-15, 1.0 - 1e-15)
        ll = y * np.log(lk) + (n_obs - y) * np.log1p(-lk)
        return self._add_loglik(ll)

    def update_check(self, m, obs_dirty, d_dirty, d_clean):
        r"""Condition on a *check* performed after ``m`` preventive actions.

        The checked state is hazardous with probability ``lambda_m``; the check
        reports it with sensitivity ``d_dirty`` and specificity ``d_clean``.  A
        check at ``m = 0`` is a genuine baseline probe of ``q``; a check at the
        agent's own operating ``m = k`` is informative only about the same
        composite ``lambda_k`` the harm outcome already reveals -- this is the
        formal content of the reassurance paradox.
        """
        lm = np.clip(self.lam_grid(m), 1e-15, 1.0 - 1e-15)
        p_report_dirty = d_dirty * lm + (1.0 - d_clean) * (1.0 - lm)
        p_report_dirty = np.clip(p_report_dirty, 1e-15, 1.0 - 1e-15)
        ll = np.log(p_report_dirty) if obs_dirty else np.log1p(-p_report_dirty)
        return self._add_loglik(ll)

    def flatten(self, gamma):
        r"""Power-discount the posterior, :math:`p \propto p^{\gamma}`.

        Models a learner that gradually forgets, which is what allows it to track
        a non-stationary hazard in the hysteresis simulation.  ``gamma = 1``
        disables forgetting.
        """
        if gamma != 1.0:
            self.log_w = gamma * self.log_w
            self.normalise()
        return self

    # ------------------------------------------------------------------- moments
    def marginal_q(self):
        return self.posterior.sum(axis=1)

    def marginal_p(self):
        return self.posterior.sum(axis=0)

    def mean(self):
        post = self.posterior
        return float((post * self.Q).sum()), float((post * self.P).sum())

    def cov(self):
        post = self.posterior
        mq, mp = self.mean()
        vq = float((post * (self.Q - mq) ** 2).sum())
        vp = float((post * (self.P - mp) ** 2).sum())
        cqp = float((post * (self.Q - mq) * (self.P - mp)).sum())
        return np.array([[vq, cqp], [cqp, vp]])

    def corr_qp(self):
        c = self.cov()
        denom = np.sqrt(c[0, 0] * c[1, 1])
        return float(c[0, 1] / denom) if denom > 0 else 0.0

    # ------------------------------------------------------------------ entropies
    @staticmethod
    def _diff_entropy(mass, cell):
        """Differential entropy (nats) of a piecewise-constant density."""
        m = mass[mass > 0]
        return float(-(m * (np.log(m) - np.log(cell))).sum())

    def entropy_joint(self):
        return self._diff_entropy(self.posterior, self.dq * self.dp)

    def entropy_q(self):
        return self._diff_entropy(self.marginal_q(), self.dq)

    def entropy_p(self):
        return self._diff_entropy(self.marginal_p(), self.dp)

    def entropy_lambda(self, k, n_bins=1001):
        r"""Differential entropy of the *observable composite* ``lambda_k``.

        The signature prediction of the model is
        :math:`H(\lambda_k)\downarrow` while :math:`H(q,p)` stays large: the agent
        can predict what will happen to it with great precision and still not know
        why.  Computed on a fixed ``[0, 1]`` bin grid so it is comparable with
        ``entropy_q`` / ``entropy_p``.

        .. warning::
           For very small :math:`\lambda_k` (a compulsive agent at large ``k``)
           the whole posterior falls inside the first bin and this saturates at
           :math:`-\log(n_{\mathrm{bins}})`.  The entropy *reduction* computed
           from it is then a **lower bound** on the information actually gained.
           Use :meth:`sd_lambda` for a scale-free comparison in that regime.
        """
        vals = self.lam_grid(k).ravel()
        mass = self.posterior.ravel()
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        hist, _ = np.histogram(vals, bins=edges, weights=mass)
        return self._diff_entropy(hist, 1.0 / n_bins)

    def sd_lambda(self, k):
        post = self.posterior
        vals = self.lam_grid(k)
        m = float((post * vals).sum())
        return float(np.sqrt((post * (vals - m) ** 2).sum()))

    def mean_lambda(self, k):
        return float((self.posterior * self.lam_grid(k)).sum())

    # -------------------------------------------------- Bayesian experiment design
    def eig_harm(self, k):
        r"""Expected information gain of one harm observation run at ``k``.

        :math:`\mathrm{EIG}(k) = I((q,p); Y \mid k, D)
        = H[\bar\lambda_k] - \mathbb E_{\theta}\,H[\lambda_k(\theta)]`,
        the standard Bayesian experimental-design utility.
        """
        post = self.posterior
        lk = self.lam_grid(k)
        marg = float((post * lk).sum())
        return float(binary_entropy(marg) - (post * binary_entropy(lk)).sum())

    def eig_check(self, m, d_dirty, d_clean):
        """Expected information gain of a check performed after ``m`` actions."""
        post = self.posterior
        lm = self.lam_grid(m)
        r = d_dirty * lm + (1.0 - d_clean) * (1.0 - lm)
        marg = float((post * r).sum())
        return float(binary_entropy(marg) - (post * binary_entropy(r)).sum())

    def eig_harm_targeted(self, k, target="p", n_bins=201):
        """EIG of a harm observation at ``k`` about a single marginal parameter."""
        post = self.posterior
        lk = self.lam_grid(k)
        marg = float((post * lk).sum())
        out = 0.0
        for y, py in ((1, marg), (0, 1.0 - marg)):
            if py <= 0:
                continue
            like = lk if y == 1 else 1.0 - lk
            new = post * like
            new = new / new.sum()
            axis = 1 if target == "q" else 0
            m_new = new.sum(axis=axis)
            m_old = post.sum(axis=axis)
            cell = self.dq if target == "q" else self.dp
            out += py * (self._diff_entropy(m_old, cell) - self._diff_entropy(m_new, cell))
        return float(out)

    # ---------------------------------------------------------------- accuracy
    def _cell_index(self, q, p):
        i = int(np.clip(np.searchsorted(self.q_vals, q), 0, self.n - 1))
        j = int(np.clip(np.searchsorted(self.p_vals, p), 0, self.n - 1))
        return i, j

    def log_density_at(self, q, p):
        """Log posterior *density* at a specific parameter value.

        Entropy alone rewards confident wrongness: an agent can become very
        certain of a false ``(q, p)`` and look like it has "learned".  The log
        density at the true parameter is a proper scoring rule and separates
        knowledge from mere confidence.
        """
        i, j = self._cell_index(q, p)
        return float(np.log(max(self.density[i, j], 1e-300)))

    def mass_within(self, q, p, radius=0.05):
        """Posterior mass inside a box of half-width ``radius`` around ``(q, p)``."""
        mask = (np.abs(self.Q - q) <= radius) & (np.abs(self.P - p) <= radius)
        return float(self.posterior[mask].sum())

    # --------------------------------------------------------------- diagnostics
    def summary(self, k_ref=None, truth=None):
        """One-row dictionary of every posterior statistic used downstream."""
        mq, mp = self.mean()
        c = self.cov()
        out = dict(
            mean_q=mq, mean_p=mp,
            sd_q=float(np.sqrt(c[0, 0])), sd_p=float(np.sqrt(c[1, 1])),
            corr_qp=self.corr_qp(),
            H_q=self.entropy_q(), H_p=self.entropy_p(), H_joint=self.entropy_joint(),
        )
        if k_ref is not None:
            out.update(
                k_ref=k_ref,
                mean_lambda=self.mean_lambda(k_ref),
                sd_lambda=self.sd_lambda(k_ref),
                H_lambda=self.entropy_lambda(k_ref),
            )
        if truth is not None:
            q_t, p_t = truth
            out.update(
                logdens_true=self.log_density_at(q_t, p_t),
                mass_near_true=self.mass_within(q_t, p_t, radius=0.05),
            )
        return out
