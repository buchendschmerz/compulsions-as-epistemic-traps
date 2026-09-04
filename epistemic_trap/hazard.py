r"""Analytic core of the epistemic-trap model.

One episode:

* the episode starts in the hazardous state with probability ``q`` (base rate),
* the agent performs ``k`` preventive actions, each of which removes the hazard
  with probability ``p`` (efficacy),
* harm occurs iff the hazard is still present when the agent commits to the
  consequential action.

Hence the only quantity the agent gets to observe is the composite

.. math:: \lambda_k = q (1-p)^k .

Every statement in this module is about the geometry of that map.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lam",
    "grad_lam",
    "ridge_q_given_p",
    "loglik",
    "fisher_information",
    "fisher_information_policy",
    "identifiability_summary",
    "identify_from_two_k",
]

EPS = 1e-12


def lam(q, p, k):
    r"""Probability of harm, :math:`\lambda_k = q(1-p)^k`.

    Broadcasting over ``q``, ``p`` and ``k`` is supported.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    k = np.asarray(k, dtype=float)
    return q * (1.0 - p) ** k


def grad_lam(q, p, k):
    r"""Gradient :math:`\nabla_{(q,p)} \lambda_k`.

    .. math::
        \frac{\partial \lambda_k}{\partial q} = (1-p)^k, \qquad
        \frac{\partial \lambda_k}{\partial p} = -k q (1-p)^{k-1}.

    Returns
    -------
    ndarray
        Array with a trailing axis of length 2, ordered ``(d/dq, d/dp)``.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    k = np.asarray(k, dtype=float)
    one_minus_p = np.clip(1.0 - p, EPS, 1.0)
    dq = one_minus_p ** k
    dp = -k * q * one_minus_p ** (k - 1.0)
    return np.stack(np.broadcast_arrays(dq, dp), axis=-1)


def ridge_q_given_p(lambda0, p, k):
    r"""The iso-likelihood ridge: :math:`q = \lambda_0 / (1-p)^k`.

    Values of ``q`` outside the unit interval are returned as ``nan`` because the
    ridge leaves the parameter simplex there.
    """
    p = np.asarray(p, dtype=float)
    one_minus_p = np.clip(1.0 - p, EPS, 1.0)
    q = np.asarray(lambda0, dtype=float) / one_minus_p ** k
    return np.where(q <= 1.0, q, np.nan)


def loglik(y, n, q, p, k):
    """Binomial log-likelihood of ``y`` harms in ``n`` episodes run at ``k``."""
    lk = np.clip(lam(q, p, k), EPS, 1.0 - EPS)
    return y * np.log(lk) + (n - y) * np.log1p(-lk)


def fisher_information(q, p, k):
    r"""Fisher information of a single Bernoulli harm observation at fixed ``k``.

    .. math::
        \mathcal I_k = \frac{\nabla\lambda_k \nabla\lambda_k^{\top}}
                            {\lambda_k (1-\lambda_k)} .

    Being an outer product of a single vector, this matrix has rank one at every
    interior parameter value, so ``det I_k == 0`` exactly.  Structural
    non-identifiability under a fixed-``k`` policy is therefore a theorem, not a
    numerical artefact.

    Returns
    -------
    ndarray
        ``(..., 2, 2)`` array.
    """
    g = grad_lam(q, p, k)
    lk = np.clip(lam(q, p, k), EPS, 1.0 - EPS)
    denom = lk * (1.0 - lk)
    outer = g[..., :, None] * g[..., None, :]
    return outer / denom[..., None, None]


def fisher_information_policy(q, p, k_values, k_probs, n_episodes=1.0):
    r"""Fisher information accumulated by a policy :math:`\pi` over ``k``.

    .. math:: \mathcal I_\pi = n \sum_k \pi(k)\, \mathcal I_k .

    A policy whose support is a single ``k`` yields a singular matrix; support on
    two or more distinct ``k`` yields (generically) a full-rank matrix.  It is the
    *diversity* of the policy, not the magnitude of ``k``, that buys
    identifiability.
    """
    k_values = np.asarray(k_values, dtype=float)
    k_probs = np.asarray(k_probs, dtype=float)
    if k_probs.sum() <= 0:
        raise ValueError("policy has zero total mass")
    k_probs = k_probs / k_probs.sum()
    total = np.zeros(np.broadcast(np.asarray(q), np.asarray(p)).shape + (2, 2))
    for kv, pr in zip(k_values, k_probs):
        if pr == 0.0:
            continue
        total = total + pr * fisher_information(q, p, kv)
    return n_episodes * total


def identifiability_summary(info):
    """Determinant, eigenvalues and condition number of a Fisher information matrix.

    Returns
    -------
    dict
        ``det``, ``eig_max``, ``eig_min``, ``log10_det`` and ``cond``.
        Following Rothenberg (1971), a non-singular information matrix
        corresponds to local identifiability, so ``det`` is the natural scalar
        identifiability index.
    """
    info = np.asarray(info, dtype=float)
    det = np.linalg.det(info)
    eig = np.linalg.eigvalsh(info)
    eig_min = eig[..., 0]
    eig_max = eig[..., -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(eig_min > 0, eig_max / eig_min, np.inf)
        log10_det = np.where(det > 0, np.log10(np.maximum(det, EPS)), -np.inf)
    return dict(det=det, eig_min=eig_min, eig_max=eig_max,
                cond=cond, log10_det=log10_det)


def identify_from_two_k(lambda_k1, k1, lambda_k2, k2):
    r"""Recover ``(q, p)`` from two harm rates observed at two distinct ``k``.

    .. math::
        \frac{\lambda_{k_1}}{\lambda_{k_2}} = (1-p)^{k_1-k_2}
        \quad\Longrightarrow\quad
        p = 1 - \left(\lambda_{k_1}/\lambda_{k_2}\right)^{1/(k_1-k_2)},
        \qquad q = \lambda_{k_1} (1-p)^{-k_1}.

    This is the constructive counterpart of the rank-one result: two distinct
    experimental conditions are necessary *and* sufficient for point
    identification.  Note in particular that response prevention alone
    (:math:`k=0`) identifies ``q`` but carries **no** information about ``p`` --
    it is the *contrast* between conditions that restores identifiability.
    """
    if k1 == k2:
        raise ValueError("k1 and k2 must differ for identification")
    ratio = np.asarray(lambda_k1, dtype=float) / np.asarray(lambda_k2, dtype=float)
    one_minus_p = ratio ** (1.0 / (k1 - k2))
    p = 1.0 - one_minus_p
    q = np.asarray(lambda_k1, dtype=float) / np.clip(one_minus_p, EPS, None) ** k1
    return q, p
