#!/usr/bin/env python3
r"""Phase 1 -- policy-induced structural non-identifiability of ``(q, p)``.

Three results, in increasing order of importance:

**(i) The rank-one theorem.**  Under any policy that always performs the same
number ``k`` of preventive repetitions, a terminal harm/no-harm observation has
Fisher information

.. math::
    \mathcal I_k = \frac{\nabla\lambda_k\,\nabla\lambda_k^{\top}}
                        {\lambda_k(1-\lambda_k)},
    \qquad \lambda_k = q(1-p)^k,

an outer product of a single vector, hence ``det I_k = 0`` at every interior
parameter value.  ``(q, p)`` is therefore *structurally* non-identifiable: no
amount of data collected under a fixed-``k`` policy can separate baseline hazard
from preventive efficacy.  Verified symbolically with SymPy.

**(ii) The direction of the trap.**  The null direction of ``I_k`` is the tangent
to the iso-likelihood ridge ``q = \lambda/(1-p)^k``, whose slope is
``dq/dp = k q / (1-p) \ge 0``.  The ridge is *positively* sloped, so the residual
posterior dependence between ``q`` and ``p`` is predicted to be positive, and its
strength grows with ``k``.  This is hypothesis H3, derived rather than assumed.

**(iii) It is policy diversity, not repetition count, that buys identifiability.**
For a policy ``pi`` over ``k``, ``I_pi = n \sum_k \pi(k) I_k`` is full rank as soon
as ``pi`` has support on two distinct values of ``k``.  Response prevention alone
(``k = 0``) identifies ``q`` but supplies *no* information about ``p``: it is the
*contrast* between conditions, not the absence of the safety action, that restores
identifiability.

Outputs
-------
results/phase1_symbolic.txt          machine-checked symbolic derivation
results/phase1_ridge.csv             iso-likelihood ridge coordinates
results/phase1_policy_identifiability.csv   det/cond of I_pi for candidate policies
results/phase1_pair_grid.csv         det of I_pi for every two-point policy (k1,k2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epistemic_trap.config import K_MAX, MASTER_SEED, RESULTS, WORLD
from epistemic_trap.hazard import (fisher_information, fisher_information_policy,
                                   identifiability_summary, identify_from_two_k,
                                   lam, loglik, ridge_q_given_p)
from epistemic_trap.plotstyle import C, panel_label, save, use_style

Q_TRUE, P_TRUE = WORLD["q"], WORLD["p"]
SING_TOL = 1e10  # condition-number threshold above which a design is called singular


# ------------------------------------------------------------------- (i) symbolic
def symbolic_proof():
    """Machine-checked derivation of the rank-one / zero-determinant result."""
    import sympy as sp

    q, p, k, n = sp.symbols("q p k n", positive=True)
    lam_s = q * (1 - p) ** k
    grad = sp.Matrix([sp.diff(lam_s, q), sp.diff(lam_s, p)])
    info = (grad * grad.T) / (lam_s * (1 - lam_s))
    det = sp.simplify(sp.det(info))
    rank = info.rank()

    # null direction and ridge slope
    null_vec = sp.simplify(info.nullspace()[0]) if info.nullspace() else None
    lam0 = sp.symbols("lambda_0", positive=True)
    q_ridge = lam0 * (1 - p) ** (-k)
    ridge_slope = sp.simplify(sp.diff(q_ridge, p).subs(lam0, q * (1 - p) ** k))

    # two distinct k restore full rank
    k1, k2 = sp.symbols("k_1 k_2", positive=True)
    def info_at(kk):
        l = q * (1 - p) ** kk
        g = sp.Matrix([sp.diff(l, q), sp.diff(l, p)])
        return (g * g.T) / (l * (1 - l))
    det_two = sp.simplify(sp.det(sp.Rational(1, 2) * (info_at(k1) + info_at(k2))))
    det_two_eq = sp.simplify(det_two.subs(k2, k1))

    # constructive recovery from two conditions
    lam1, lam2 = sp.symbols("lambda_1 lambda_2", positive=True)
    p_hat = 1 - (lam1 / lam2) ** (1 / (k1 - k2))
    p_check = sp.simplify(p_hat.subs({lam1: q * (1 - p) ** k1,
                                      lam2: q * (1 - p) ** k2}))

    lines = [
        "Symbolic derivation (SymPy) -- policy-induced non-identifiability",
        "=" * 70,
        f"lambda_k                    = {sp.srepr(lam_s) and sp.pretty(lam_s, use_unicode=False)}",
        f"grad lambda_k               = {sp.pretty(grad.T, use_unicode=False)}",
        "",
        "(i) Fisher information of one Bernoulli harm observation at fixed k:",
        f"    rank(I_k)               = {rank}",
        f"    det(I_k)                = {det}          <-- identically zero",
        f"    null direction of I_k   = {sp.pretty(null_vec.T, use_unicode=False) if null_vec is not None else 'n/a'}",
        "",
        "(ii) Iso-likelihood ridge q = lambda_0 (1-p)^(-k):",
        f"    dq/dp along the ridge   = {sp.simplify(ridge_slope)}",
        "    -> non-negative for all k >= 0, so the residual (q,p) dependence is",
        "       POSITIVE and increases with k  [hypothesis H3].",
        "",
        "(iii) A two-point policy pi = {k_1, k_2} with equal mass:",
        f"    det(I_pi)               = {sp.simplify(det_two)}",
        f"    det(I_pi) at k_1 = k_2  = {det_two_eq}          <-- degenerate only if k_1 = k_2",
        "",
        "(iv) Constructive identification from two conditions:",
        "    p_hat = 1 - (lambda_1/lambda_2)^(1/(k_1-k_2))",
        f"    substituting the true lambdas gives p_hat = {p_check}",
        "",
        "Conclusion: non-identifiability under a repetitive safety policy is a",
        "theorem about the design, not a numerical artefact of the simulation.",
    ]
    text = "\n".join(lines)
    (RESULTS / "phase1_symbolic.txt").write_text(text)
    return text, dict(det=det, rank=rank)


# ------------------------------------------------------------- (ii) numeric ridge
def ridge_table(k, n_obs=400, n_points=400):
    """Iso-likelihood ridge and the profile log-likelihood along it."""
    p_grid = np.linspace(0.001, 0.999, n_points)
    lam0 = lam(Q_TRUE, P_TRUE, k)
    q_grid = ridge_q_given_p(lam0, p_grid, k)
    y = n_obs * lam0
    ll = loglik(y, n_obs, q_grid, p_grid, k)
    slope = k * q_grid / (1.0 - p_grid)
    return pd.DataFrame(dict(k=k, p=p_grid, q_ridge=q_grid,
                             loglik=ll, ridge_slope=slope, lambda0=lam0))


# --------------------------------------------------- (iii) policies and their FIM
def candidate_policies():
    """The policy families compared in Figure 2C."""
    ks = np.arange(K_MAX + 1)

    def onehot(k):
        v = np.zeros(len(ks)); v[k] = 1.0; return v

    def uniform_on(vals):
        v = np.zeros(len(ks)); v[list(vals)] = 1.0; return v / v.sum()

    pols = {
        "fixed k=0 (pure response prevention)": onehot(0),
        "fixed k=1": onehot(1),
        "fixed k=3": onehot(3),
        "fixed k=6 (compulsive)": onehot(6),
        "compulsive, jittered (k in {5,6,7})": uniform_on([5, 6, 7]),
        "k in {0,1}": uniform_on([0, 1]),
        "k in {0,1,2,3}": uniform_on([0, 1, 2, 3]),
        "graded ERP (k in 0..6)": uniform_on(range(7)),
        "k in {0,6} (baseline + compulsive)": uniform_on([0, 6]),
    }
    return ks, pols


def policy_table(n_episodes=500):
    ks, pols = candidate_policies()
    rows = []
    for name, probs in pols.items():
        info = fisher_information_policy(Q_TRUE, P_TRUE, ks, probs,
                                         n_episodes=n_episodes)
        s = identifiability_summary(info)
        support = np.count_nonzero(probs)
        mean_k = float((ks * probs).sum())
        var_k = float((ks ** 2 * probs).sum() - mean_k ** 2)
        # A determinant of exactly zero is only recoverable up to floating point,
        # so declare singularity on the condition number, which is scale-free.
        singular = bool(support < 2 or not np.isfinite(s["cond"]) or s["cond"] > SING_TOL)
        det = 0.0 if singular else float(s["det"])
        log10_det = -np.inf if singular else float(s["log10_det"])
        cov = np.linalg.inv(info) if not singular else None
        rows.append(dict(policy=name, support=support, mean_k=mean_k, var_k=var_k,
                         singular=singular, det=det, log10_det=log10_det,
                         eig_min=float(s["eig_min"]), eig_max=float(s["eig_max"]),
                         cond=float(s["cond"]),
                         se_p=np.inf if singular else float(np.sqrt(cov[1, 1])),
                         se_q=np.inf if singular else float(np.sqrt(cov[0, 0]))))
    return pd.DataFrame(rows)


def pair_grid(n_episodes=500):
    """det(I) for every equal-weight two-point policy ``{k1, k2}``."""
    ks = np.arange(K_MAX + 1)
    out = np.full((len(ks), len(ks)), np.nan)
    for i, k1 in enumerate(ks):
        for j, k2 in enumerate(ks):
            probs = np.zeros(len(ks))
            probs[i] += 0.5
            probs[j] += 0.5
            info = fisher_information_policy(Q_TRUE, P_TRUE, ks, probs,
                                             n_episodes=n_episodes)
            out[i, j] = float(np.linalg.det(info))
    df = pd.DataFrame(out, index=[f"k1={k}" for k in ks],
                      columns=[f"k2={k}" for k in ks])
    return ks, out, df


# ----------------------------------------------------------------------- figure
def make_figure(ridges, poltab, ks_pair, pairmat):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    use_style()
    fig = plt.figure(figsize=(9.6, 6.2))
    gs = fig.add_gridspec(2, 3)

    # ---- A: log-likelihood surface with the ridge, fixed k
    ax = fig.add_subplot(gs[0, 0])
    k_show, n_obs = 4, 400
    qq, pp = np.meshgrid(np.linspace(0.005, 0.995, 300),
                         np.linspace(0.005, 0.995, 300), indexing="ij")
    lam0 = lam(Q_TRUE, P_TRUE, k_show)
    ll = loglik(n_obs * lam0, n_obs, qq, pp, k_show)
    ll = ll - ll.max()
    levels = [-30, -20, -12, -6, -3, -1, -0.3]
    cf = ax.contourf(pp, qq, ll, levels=levels + [0], cmap="Blues_r", extend="min")
    ax.contour(pp, qq, ll, levels=[-3, -1], colors=C["ridge"], linewidths=[0.7, 1.1])
    ridge = ridges[k_show]
    ax.plot(ridge.p, ridge.q_ridge, color=C["compulsive"], lw=2.0,
            label=r"ridge $q=\lambda_k(1-p)^{-k}$")
    ax.plot([P_TRUE], [Q_TRUE], "*", ms=11, color=C["accent"],
            markeredgecolor="k", markeredgewidth=0.5, label="truth")
    ax.set_xlabel("preventive efficacy $p$")
    ax.set_ylabel("baseline hazard $q$")
    ax.set_title(f"Likelihood ridge at fixed $k={k_show}$\n({n_obs} episodes)")
    ax.legend(loc="upper left")
    fig.colorbar(cf, ax=ax, label="relative log-likelihood", pad=0.02,
                 fraction=0.046, shrink=0.92)
    panel_label(ax, "A")

    # ---- B: the ridge steepens with k
    ax = fig.add_subplot(gs[0, 1])
    for k in [1, 2, 4, 6, 8]:
        r = ridges[k]
        ax.plot(r.p, r.q_ridge, lw=1.5, label=f"$k={k}$",
                color=plt.cm.viridis(k / 9))
    ax.plot([P_TRUE], [Q_TRUE], "*", ms=11, color=C["accent"],
            markeredgecolor="k", markeredgewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("preventive efficacy $p$")
    ax.set_ylabel("baseline hazard $q$")
    ax.set_title("Ridge slope $dq/dp = kq/(1-p)$\ngrows with repetition")
    ax.legend(ncol=2, loc="upper left")
    panel_label(ax, "B")

    # ---- C: rank of the information matrix
    ax = fig.add_subplot(gs[0, 2])
    kk = np.arange(0, K_MAX + 1)
    dets = [float(np.linalg.det(fisher_information(Q_TRUE, P_TRUE, k) * 500))
            for k in kk]
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(kk, dets, "o-", color=C["compulsive"], ms=5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("repetitions $k$ (fixed-$k$ policy)")
    ax.set_ylabel(r"$\det \mathcal{I}_k$")
    ax.set_title("Every fixed-$k$ design is singular\n"
                 r"($\det \mathcal{I}_k \equiv 0$, rank 1)")
    ax.text(0.5, 0.62, "rank-one for every $k$\n(SymPy-verified)",
            transform=ax.transAxes, ha="center", color=C["neutral"])
    ax.text(0.5, 0.20, r"one repetition count $\Rightarrow$ one" "\n"
            r"scalar $\lambda_k$ $\Rightarrow$ one direction of" "\n"
            r"information in a two-parameter space",
            transform=ax.transAxes, ha="center", fontsize=7.0,
            color=C["compulsive"])
    panel_label(ax, "C")

    # ---- D: identifiability by policy
    ax = fig.add_subplot(gs[1, :2])
    tab = poltab.sort_values("log10_det")
    y = np.arange(len(tab))
    vals = tab.log10_det.values.copy()
    singular = tab.singular.values
    finite = vals[np.isfinite(vals)]
    floor = float(np.nanmin(finite)) - 2.2
    stub = 1.05  # singular designs get a visible hatched stub, not a zero-width bar
    vals_plot = np.where(singular, floor + stub, vals)
    colors = [C["compulsive"] if s else C["healthy"] for s in singular]
    bars = ax.barh(y, vals_plot - floor, left=floor, color=colors, height=0.62)
    for b, s in zip(bars, singular):
        if s:
            b.set_hatch("///")
            b.set_edgecolor("white")
            b.set_linewidth(0.0)
    for yi, s in enumerate(singular):
        if s:
            ax.text(floor + stub + 0.12, yi,
                    r"$\det \mathcal{I}_\pi = 0$ — non-identifiable",
                    va="center", fontsize=7.2, color=C["compulsive"])
    ax.axvline(floor + stub, color=C["neutral"], lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(tab.policy)
    ax.set_xlabel(r"$\log_{10}\det \mathcal{I}_\pi$  (500 episodes)")
    ax.set_title("Identifiability is set by the diversity of the policy, "
                 "not by how much safety behaviour it performs")
    ax.set_xlim(floor, float(finite.max()) + 1.0)
    ticks = [tk for tk in np.arange(-4, 8, 2) if tk > floor + stub]
    ax.set_xticks(ticks)
    panel_label(ax, "D", dx=-0.30)

    # ---- E: two-point policy grid
    ax = fig.add_subplot(gs[1, 2])
    m = pairmat.copy()
    m[m <= m[m > 0].min() * 1e-6 if (m > 0).any() else 0] = np.nan
    np.fill_diagonal(m, np.nan)
    im = ax.imshow(m, origin="lower", cmap="magma", norm=LogNorm(),
                   extent=[-0.5, K_MAX + 0.5, -0.5, K_MAX + 0.5])
    ax.plot([-0.5, K_MAX + 0.5], [-0.5, K_MAX + 0.5], color=C["healthy"], lw=1.2)
    ax.annotate("$k_1=k_2$: singular", xy=(K_MAX * 0.35, K_MAX * 0.35),
                xytext=(K_MAX * 0.30, K_MAX * 0.82), fontsize=7.2, color="white",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=C["healthy"], ec="none"),
                arrowprops=dict(arrowstyle="-|>", color=C["healthy"], lw=1.0))
    ax.set_xlabel("$k_2$")
    ax.set_ylabel("$k_1$")
    ax.set_title(r"$\det\mathcal{I}$ for two-point policies")
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046,
                 label=r"$\det\mathcal{I}_\pi$")
    panel_label(ax, "E")

    return save(fig, "fig2_nonidentifiability")


# ------------------------------------------------------------------------- main
def main():
    np.random.default_rng(MASTER_SEED)
    text, sym = symbolic_proof()
    print(text)
    assert sym["det"] == 0, "symbolic determinant should be identically zero"
    assert sym["rank"] == 1

    ridges = {k: ridge_table(k) for k in range(0, K_MAX + 1)}
    pd.concat(ridges.values()).to_csv(RESULTS / "phase1_ridge.csv", index=False)

    poltab = policy_table()
    poltab.to_csv(RESULTS / "phase1_policy_identifiability.csv", index=False)
    print("\n" + poltab.round(3).to_string(index=False))

    ks_pair, pairmat, pairdf = pair_grid()
    pairdf.to_csv(RESULTS / "phase1_pair_grid.csv")

    # numerical confirmation of the constructive recovery
    l1, l2 = lam(Q_TRUE, P_TRUE, 0), lam(Q_TRUE, P_TRUE, 3)
    q_hat, p_hat = identify_from_two_k(l1, 0, l2, 3)
    print(f"\n[phase1] recovery from k in {{0,3}}: q={q_hat:.6f} (true {Q_TRUE}), "
          f"p={p_hat:.6f} (true {P_TRUE})")
    assert abs(q_hat - Q_TRUE) < 1e-9 and abs(p_hat - P_TRUE) < 1e-9

    print("[phase1] tables complete; manuscript Figure 2 is assembled by "
          "make_submission_figures.py")


if __name__ == "__main__":
    main()
