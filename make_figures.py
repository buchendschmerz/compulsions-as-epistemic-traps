#!/usr/bin/env python3
"""Assemble the publication figures from the saved phase results.

Usage
-----
    python3 make_figures.py            # all figures that have data available
    python3 make_figures.py 1 3 6      # only these

Figure 2 is produced by ``phase1_nonidentifiability.py`` and Figure S1 by
``phase0_replication.py``, because both are cheap to regenerate alongside their
analyses.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from epistemic_trap.config import RESULTS, WORLD
from epistemic_trap.plotstyle import (C, CONDITION_COLORS, CONDITION_LABELS,
                                      panel_label, save, use_style)

Q_TRUE, P_TRUE = WORLD["q"], WORLD["p"]


def _need(*names):
    missing = [n for n in names if not (RESULTS / n).exists()]
    if missing:
        print(f"  [skip] missing {missing}")
    return not missing


# =============================================================== Figure 1
def fig1_concept():
    """Conceptual lineage: static point beliefs -> a closed causal-learning loop."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    def box(ax, xy, w, h, text, fc, ec=None, fs=8.0):
        ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.025",
                                    linewidth=1.1, facecolor=fc,
                                    edgecolor=ec or fc, zorder=2))
        ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs,
                zorder=3, linespacing=1.4)

    def arrow(ax, a, b, color="black", rad=0.0, lw=1.4, ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=lw, color=color, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=1, shrinkB=1, zorder=1))

    # ------------------------------------------------ A: the model we extend
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(-0.20, 1.16); ax.axis("off")
    ax.text(0.5, 1.14, "Rigoux et al. (2024)", ha="center", va="top",
            fontsize=9.5, fontweight="bold")
    ax.text(0.5, 1.075, "beliefs about contingencies are fixed point estimates",
            ha="center", va="top", fontsize=8.0, color=C["neutral"])

    ys = [0.86, 0.62, 0.38, 0.13]
    box(ax, (0.44, ys[0]), 0.68, 0.155,
        "agent parameters\n"
        "$p'_{\\mathrm{SUCCESS}},\\ p'_{\\mathrm{SOILING}},\\ \\ldots$\n"
        "(static point estimates)", "#E8EEF4", C["healthy"])
    box(ax, (0.44, ys[1]), 0.56, 0.115, "POMDP policy $\\pi^{*}(b)$",
        "#F3F3F3", C["neutral"])
    box(ax, (0.44, ys[2]), 0.56, 0.115, "actions: cook / wash / check",
        "#F3F3F3", C["neutral"])
    box(ax, (0.44, ys[3]), 0.56, 0.115, "state belief $b=p(\\mathrm{dirty})$",
        "#F3F3F3", C["neutral"])
    for a_y, b_y in zip(ys[:-1], ys[1:]):
        arrow(ax, (0.44, a_y - 0.082), (0.44, b_y + 0.062))
    arrow(ax, (0.80, ys[3]), (0.80, ys[1] - 0.058), color=C["neutral"], rad=-0.55)
    ax.text(0.93, 0.38, "state belief feeds back", fontsize=6.8, rotation=270,
            va="center", ha="center", color=C["neutral"])
    ax.annotate("nothing returns to the top box:\n"
                "contingency beliefs never change",
                xy=(0.155, ys[0] - 0.072), xytext=(0.40, -0.15),
                fontsize=7.4, color=C["compulsive"], ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C["compulsive"], lw=1.0,
                                ls=(0, (3, 2)),
                                connectionstyle="arc3,rad=-0.45"))
    panel_label(ax, "A", dx=-0.02, dy=1.16)

    # ------------------------------------------------ B: the closed loop
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(-0.20, 1.16); ax.axis("off")
    ax.text(0.5, 1.14, "This paper", ha="center", va="top", fontsize=9.5,
            fontweight="bold")
    ax.text(0.5, 1.075, "the policy decides which contrasts enter the data",
            ha="center", va="top", fontsize=8.0, color=C["neutral"])

    nodes = {
        "belief": ((0.50, 0.90), 0.56, 0.13, "posterior $P(q,p\\mid D)$",
                   "#E8EEF4", C["healthy"]),
        "policy": ((0.845, 0.52), 0.29, 0.155,
                   "policy:\nhow many\nrepetitions $k$?", "#F7E4E2",
                   C["compulsive"]),
        "world": ((0.50, 0.115), 0.52, 0.13,
                  "world  $\\lambda_k=q(1-p)^{k}$", "#F3F3F3", C["neutral"]),
        "obs": ((0.155, 0.52), 0.29, 0.155,
                "observation\n$Y\\sim\\mathrm{Bern}(\\lambda_k)$",
                "#E9F1EE", C["erp"]),
    }
    for xy, w, h, label, fc, ec in nodes.values():
        box(ax, xy, w, h, label, fc, ec, fs=7.8)

    arrow(ax, (0.75, 0.855), (0.895, 0.61), color=C["compulsive"], rad=0.30)
    arrow(ax, (0.845, 0.435), (0.68, 0.165), color=C["compulsive"], rad=0.30)
    arrow(ax, (0.32, 0.165), (0.155, 0.435), color=C["neutral"], rad=0.30)
    arrow(ax, (0.155, 0.605), (0.28, 0.855), color=C["healthy"], rad=0.30)

    ax.add_patch(FancyBboxPatch((0.345, 0.365), 0.31, 0.30,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                facecolor="#FFFFFF", edgecolor=C["compulsive"],
                                zorder=3,
                                linewidth=1.0, linestyle=(0, (4, 2))))
    ax.text(0.50, 0.515, "the trap\n\n"
            "if $k$ never varies,\n"
            r"$\det\mathcal{I}_k = 0$" "\n"
            "and $(q,p)$ cannot\nbe separated",
            ha="center", va="center", fontsize=7.2, color=C["compulsive"],
            zorder=4, linespacing=1.45)
    ax.text(0.50, -0.14,
            "belief $\\rightarrow$ policy $\\rightarrow$ which contrasts are "
            "observed $\\rightarrow$ belief",
            ha="center", va="center", fontsize=7.4, color=C["neutral"])
    panel_label(ax, "B", dx=-0.02, dy=1.16)

    return save(fig, "fig1_concept")


# =============================================================== Figure 3
def fig3_trajectories():
    """Posterior trajectories and final joint posteriors."""
    if not _need("phase2_trajectories.csv", "phase2_posteriors.npz",
                 "phase2_k_sweep_means.csv"):
        return None
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    use_style()
    traj = pd.read_csv(RESULTS / "phase2_trajectories.csv")
    post = np.load(RESULTS / "phase2_posteriors.npz")
    sweep = pd.read_csv(RESULTS / "phase2_k_sweep_means.csv")

    arms = [("severe_doubt/fixed-k", C["compulsive"], "compulsive: fixed $k$"),
            ("severe_doubt/variable-k", C["healthy"], "variable $k$"),
            ("accurate/fixed-k", C["mild"], "accurate prior, fixed $k$")]

    fig = plt.figure(figsize=(9.8, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])

    def band(ax, col, ylabel, hline=None, title=None):
        for cond, colr, lab in arms:
            sub = traj[traj.condition == cond]
            g = sub.groupby("episode")[col]
            m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
            ax.fill_between(m.index, lo, hi, color=colr, alpha=0.15, linewidth=0)
            ax.plot(m.index, m.values, color=colr, label=lab)
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.9, ls=":")
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

    ax = fig.add_subplot(gs[0, 0])
    band(ax, "mean_p", r"$\mathbb{E}[p\mid D]$", hline=P_TRUE,
         title="Belief in one's own efficacy")
    ax.text(traj.episode.max() * 0.98, P_TRUE + 0.03, "true $p$", ha="right",
            fontsize=7.2)
    ax.legend(loc="lower right")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    band(ax, "mean_q", r"$\mathbb{E}[q\mid D]$", hline=Q_TRUE,
         title="Belief about the baseline hazard")
    ax.text(traj.episode.max() * 0.98, Q_TRUE + 0.02, "true $q$", ha="right",
            fontsize=7.2)
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    for cond, colr, lab in arms:
        sub = traj[traj.condition == cond]
        ax.plot(sub.groupby("episode").dH_lambda.mean(), color=colr, lw=1.6,
                label=f"{lab}: $\\lambda_k$")
        ax.plot(sub.groupby("episode").dH_p.mean(), color=colr, lw=1.2, ls="--",
                label=f"{lab}: $p$")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("episode")
    ax.set_ylabel("information gained (nats)")
    ax.set_title("What gets learned: the composite,\nnot the parameters")
    ax.legend(fontsize=6.2, ncol=1, loc="upper left")
    panel_label(ax, "C")

    # ---- joint posteriors
    cmap = LinearSegmentedColormap.from_list(
        "trap", ["#FFFFFF", "#CFE0EC", "#5D8FB3", "#1F3F58"])
    for j, (cond, colr, lab) in enumerate(arms[:2]):
        ax = fig.add_subplot(gs[1, j])
        # axis 0 of the stored posterior is q, axis 1 is p
        P = post[cond]
        ax.imshow(P, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cmap)
        ks_used = sorted(traj[traj.condition == cond].k.value_counts()
                         .head(4).index.tolist())
        n_ep = int(traj[traj.condition == cond].episode.max())
        pg = np.linspace(0.005, 0.995, 400)
        for i, k_r in enumerate(ks_used):
            lam0 = Q_TRUE * (1 - P_TRUE) ** k_r
            qg = lam0 / (1 - pg) ** k_r
            ok = qg <= 1
            ax.plot(pg[ok], qg[ok], color=C["compulsive"],
                    lw=1.5 if i == 0 else 0.8,
                    ls="--", alpha=1.0 if i == 0 else 0.45,
                    label=(f"iso-$\\lambda_k$ ridge through the truth, "
                           f"$k\\in${ks_used}" if i == 0 else None))
        # The band the data can actually resolve: with N episodes, any lambda
        # below roughly 3/N is indistinguishable from zero.
        k_main = ks_used[0]
        qg = (3.0 / n_ep) / (1 - pg) ** k_main
        ok = qg <= 1
        ax.plot(pg[ok], qg[ok], color=C["ridge"], lw=1.1, ls=":",
                label=r"resolution limit $\lambda_k = 3/N$")
        ax.plot([P_TRUE], [Q_TRUE], "*", ms=12, color=C["accent"],
                markeredgecolor="k", markeredgewidth=0.5, label="truth")
        ax.set_xlabel("preventive efficacy $p$")
        ax.set_ylabel("baseline hazard $q$")
        ax.set_title(f"Final posterior: {lab}")
        ax.legend(loc="upper left", fontsize=6.8)
        panel_label(ax, "DE"[j])

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(sweep.k_fixed, sweep.dH_p, "o-", color=C["compulsive"], ms=4,
            label=r"about efficacy $p$")
    ax.plot(sweep.k_fixed, sweep.corr_qp, "s--", color=C["healthy"], ms=4,
            label=r"posterior $\mathrm{corr}(q,p)$")
    kbest = int(sweep.loc[sweep.dH_p.idxmax(), "k_fixed"])
    ax.axvline(kbest, color=C["accent"], lw=1.2, ls=":")
    ax.annotate(f"most informative\nrepetition count $k={kbest}$",
                xy=(kbest, sweep.dH_p.max()),
                xytext=(kbest + 0.4, sweep.dH_p.max() * 1.02),
                fontsize=7.0, color=C["neutral"], ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.8, color=C["neutral"]))
    ax.set_xlabel("repetitions $k$ (design held fixed)")
    ax.set_ylabel("nats gained about $p$  /  correlation")
    ax.set_title("Never acting and always acting\nare equally uninformative")
    ax.legend(loc="center right", fontsize=7.0)
    panel_label(ax, "F")

    return save(fig, "fig3_posterior_trajectories")




# =============================================================== Figure 4
def fig4_phase_diagram():
    """Where the trap lives: doubt x harm cost, and the axis that escapes it."""
    if not _need("phase4_phase_diagram.csv", "phase4_tau_diagram.csv"):
        return None
    import matplotlib.pyplot as plt

    use_style()
    pdg = pd.read_csv(RESULTS / "phase4_phase_diagram.csv")
    tdg = pd.read_csv(RESULTS / "phase4_tau_diagram.csv")

    def grid_of(df, x, y, v):
        g = df.groupby([y, x])[v].mean().unstack()
        return g.columns.values, g.index.values, g.values

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.2))

    # A: how much safety behaviour
    ax = axes[0]
    xs, ys, Z = grid_of(pdg, "mean_p_prior", "harm_cost", "mean_k")
    im = ax.pcolormesh(xs, ys, Z, cmap="magma", shading="nearest")
    ax.set_yscale("log")
    ax.set_xlabel(r"initial belief in efficacy  $\mathbb{E}_0[p]$")
    ax.set_ylabel("cost of harm")
    ax.set_title("Amount of safety behaviour\n(mean $k$)")
    ax.text(0.04, 0.06, "black: never acts", transform=ax.transAxes,
            fontsize=6.6, color="white", va="bottom")
    fig.colorbar(im, ax=ax, pad=0.02)
    panel_label(ax, "A")

    # B: how wrong the agent stays
    ax = axes[1]
    xs, ys, Z = grid_of(pdg, "mean_p_prior", "harm_cost", "abs_err_p")
    im = ax.pcolormesh(xs, ys, Z, cmap="viridis", shading="nearest")
    cs = ax.contour(xs, ys, Z, levels=[0.15, 0.30], colors="white",
                    linewidths=[1.0, 1.6])
    ax.clabel(cs, fmt=lambda v: f"{v:.2f}", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"initial belief in efficacy  $\mathbb{E}_0[p]$")
    ax.set_ylabel("cost of harm")
    ax.set_title("Residual error in efficacy belief\n"
                 r"$|\mathbb{E}[p\mid D]-p^\ast|$ after 800 episodes")
    fig.colorbar(im, ax=ax, pad=0.02)
    panel_label(ax, "B")

    # C: the escape axis
    ax = axes[2]
    sub = tdg.copy()
    sub["tau_plot"] = sub.tau.replace(0.0, sub[sub.tau > 0].tau.min() / 3)
    xs, ys, Z = grid_of(sub, "mean_p_prior", "tau_plot", "abs_err_p")
    im = ax.pcolormesh(xs, ys, Z, cmap="viridis", shading="nearest")
    ax.set_yscale("log")
    ax.set_xlabel(r"initial belief in efficacy  $\mathbb{E}_0[p]$")
    ax.set_ylabel(r"information-seeking weight  $\tau$")
    ticks = [ys[0], ys[len(ys) // 2], ys[-1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0 (harm-\nminimising)", f"{ys[len(ys)//2]:.0f}",
                        f"{ys[-1]:.0f}"])
    ax.set_title("Paying a little for knowledge\ndissolves the trap")
    fig.colorbar(im, ax=ax, pad=0.02)
    panel_label(ax, "C")

    # D: the unifying U -- too little safety behaviour and too much are both
    # uninformative about one's own efficacy
    ax = axes[3]
    kb = (pdg.assign(kbin=pdg.mean_k.round())
          .groupby("kbin").agg(err=("abs_err_p", "mean"),
                               sd=("abs_err_p", "std"),
                               harm=("harm_rate", "mean"),
                               n=("abs_err_p", "size")))
    kb = kb[kb.n >= 5]
    ax.scatter(pdg.mean_k, pdg.abs_err_p, s=9, color=C["neutral"], alpha=0.18,
               edgecolors="none")
    ax.errorbar(kb.index, kb.err, yerr=kb.sd, fmt="o-", ms=5, capsize=3,
                color=C["compulsive"], zorder=3, label="mean $\\pm$ SD")
    kstar = float(kb.err.idxmin())
    ax.axvline(kstar, color=C["accent"], ls=":", lw=1.2)
    ax.annotate("never act:\nnothing learned\nabout $p$", xy=(0.0, kb.err.iloc[0]),
                xytext=(0.55, kb.err.iloc[0] * 0.72), fontsize=6.6,
                color=C["neutral"], ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C["neutral"]))
    ax.annotate("always act:\nno outcome\never surprises", xy=(kb.index.max(),
                kb.err.iloc[-1]), xytext=(kb.index.max() - 2.6, kb.err.iloc[-1] + 0.13),
                fontsize=6.6, color=C["neutral"], ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C["neutral"]))
    ax2 = ax.twinx()
    ax2.plot(kb.index, 100 * kb.harm, "s--", ms=4, color=C["erp"], lw=1.2)
    ax2.set_ylabel("harm rate (%)", color=C["erp"])
    ax2.tick_params(axis="y", colors=C["erp"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C["erp"])
    ax.set_xlabel("realised mean repetitions $k$")
    ax.set_ylabel(r"$|\mathbb{E}[p\mid D]-p^\ast|$")
    ax.set_title("The trap has two arms:\ntoo little and too much are both blind")
    ax.legend(loc="upper center", fontsize=6.8)
    panel_label(ax, "D")

    return save(fig, "fig4_phase_diagram")


# =============================================================== Figure 5
def fig5_hysteresis():
    """Path dependence: what happens when the danger goes away."""
    if not _need("phase4_hysteresis_loops.csv", "phase4_loop_areas.csv",
                 "phase4_step_removal.csv", "phase4_step_summary.csv"):
        return None
    import matplotlib.pyplot as plt

    use_style()
    loops = pd.read_csv(RESULTS / "phase4_hysteresis_loops.csv")
    areas = pd.read_csv(RESULTS / "phase4_loop_areas.csv")
    step = pd.read_csv(RESULTS / "phase4_step_removal.csv")

    arm_col = {"locked": C["compulsive"], "quota": C["graded"],
               "curious": C["healthy"], "checker": C["reassurance"]}
    arm_lab = {"locked": "locked design (deterministic)",
               "quota": "25% response-prevention quota",
               "curious": "information-seeking",
               "checker": "checks instead of varying"}
    arm_short = {"locked": "locked\ndesign", "quota": "25% RP\nquota",
                 "curious": "information\nseeking", "checker": "checks, no\nvariation"}

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.2))

    # A/B: the loops themselves
    for j, arm in enumerate(["locked", "quota"]):
        ax = axes[j]
        sub = loops[loops.arm == arm]
        for limb, ls, mk in (("up", "-", "o"), ("down", "--", "s")):
            g = sub[sub.limb == limb].groupby("q_bin").mean_k
            m, sd = g.mean(), g.std()
            ax.plot(m.index, m.values, ls, marker=mk, ms=3.5,
                    color=arm_col[arm], alpha=1.0 if limb == "up" else 0.6,
                    label=f"hazard {'rising' if limb == 'up' else 'falling'}")
            ax.fill_between(m.index, m - sd, m + sd, color=arm_col[arm],
                            alpha=0.12, linewidth=0)
        area = areas[areas.arm == arm].loop_area_k
        ax.set_xlabel("objective hazard $q$")
        ax.set_ylabel("repetitions $k$")
        ax.set_title(arm_lab[arm])
        ax.legend(loc="lower right")
        ax.text(0.03, 0.97,
                f"loop area = {area.mean():.2f} $\\pm$ {area.std():.2f}",
                transform=ax.transAxes, va="top", fontsize=6.8,
                color=C["neutral"])
        panel_label(ax, "AB"[j])

    # C: belief calibration on the descending limb
    ax = axes[2]
    order = ["locked", "quota", "curious", "checker"]
    m = areas.groupby("arm").calib_err_down.mean().reindex(order)
    sd = areas.groupby("arm").calib_err_down.std().reindex(order)
    ax.bar(np.arange(len(order)), m.values, yerr=sd.values, capsize=3,
           color=[arm_col[a] for a in order], width=0.62)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([arm_short[a] for a in order], fontsize=6.8)
    ax.set_ylabel(r"$\overline{|\mathbb{E}[q\mid D]-q|}$ (falling limb)")
    ax.set_title("Miscalibration while the\nworld is getting safer")
    up = areas.groupby("arm").calib_err_up.mean().reindex(order)
    ax.plot(np.arange(len(order)), up.values, "k_", ms=16, mew=1.4,
            label="rising limb", zorder=4)
    ax.legend(loc="lower right", fontsize=6.8)
    panel_label(ax, "C")

    # D: hazard removed outright
    ax = axes[3]
    rem = step[step.limb == "removed"]
    for arm in order:
        sub = rem[rem.arm == arm]
        g = sub.groupby("episode").k.mean().rolling(50, min_periods=10).mean()
        x = np.arange(len(g))
        ax.plot(x, g.values, color=arm_col[arm], label=arm_lab[arm], lw=1.4)
    ax.set_xlabel("episodes since the hazard was removed")
    ax.set_ylabel("repetitions $k$")
    ax.set_title("Protecting against a danger\nthat no longer exists")
    ax.legend(loc="upper right", fontsize=6.4)
    panel_label(ax, "D")

    return save(fig, "fig5_hysteresis")


# =============================================================== Figure 6
def fig6_interventions():
    """Interventions: which of them restores knowledge of control?"""
    if not _need("phase3_trajectories.csv", "phase3_summary.csv"):
        return None
    import matplotlib.pyplot as plt

    use_style()
    traj = pd.read_csv(RESULTS / "phase3_trajectories.csv")
    summ = pd.read_csv(RESULTS / "phase3_summary.csv")

    cols = dict(CONDITION_COLORS)
    cols.update(locked_k=C["ridge"], baseline_check=C["accent"],
                erp_naive="#8FA6B2")
    labs = dict(CONDITION_LABELS)
    labs.update(locked_k="Locked $k$ (no diversification)",
                baseline_check="Check BEFORE acting",
                erp_naive="Response prevention, no history",
                reassurance="Reassurance (check AFTER acting)")
    order = ["free_choice", "locked_k", "reassurance", "baseline_check",
             "erp_k0", "graded_erp", "variable_k"]
    # hand-tuned label offsets so the clustered untreated arms stay legible
    offs_d = {"free_choice": (6, 6), "locked_k": (34, -9), "reassurance": (6, 4),
              "baseline_check": (6, 2), "erp_k0": (6, -10),
              "graded_erp": (6, 4), "variable_k": (6, 4)}
    offs_e = {"free_choice": (6, 7), "locked_k": (34, -9), "reassurance": (6, -11),
              "baseline_check": (6, 4), "erp_k0": (-6, -14),
              "graded_erp": (6, 4), "variable_k": (6, -10)}

    n_pre = int(traj[traj.block == "pre"].episode.max())
    n_tx = int(traj[traj.block == "tx"].episode.max())

    fig = plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(2, 3)

    def shade(ax):
        ax.axvspan(n_pre, n_tx, color="#000000", alpha=0.045, linewidth=0)
        ax.text((n_pre + n_tx) / 2, ax.get_ylim()[1], " treatment", fontsize=7,
                va="top", ha="center", color=C["neutral"])

    ax = fig.add_subplot(gs[0, 0])
    for c in order:
        sub = traj[traj.condition == c]
        ax.plot(sub.groupby("episode").mean_p.mean(), color=cols[c], label=labs[c])
    ax.axhline(P_TRUE, color="k", lw=0.9, ls=":")
    ax.set_xlabel("episode"); ax.set_ylabel(r"$\mathbb{E}[p\mid D]$")
    ax.set_title("Belief in one's own efficacy")
    shade(ax)
    ax.legend(fontsize=6.3, loc="upper left")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    for c in order:
        sub = traj[traj.condition == c]
        ax.plot(sub.groupby("episode").logdens_true.mean(), color=cols[c])
    ax.set_xlabel("episode")
    ax.set_ylabel(r"$\log P(q^\ast,p^\ast\mid D)$")
    ax.set_title("Knowledge of the truth\n(a proper score, not mere confidence)")
    shade(ax)
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    for c in order:
        sub = traj[traj.condition == c]
        ax.plot(sub.groupby("episode").k.mean().rolling(25, min_periods=5).mean(),
                color=cols[c])
    ax.set_xlabel("episode"); ax.set_ylabel("repetitions $k$")
    ax.set_title("Behaviour")
    shade(ax)
    panel_label(ax, "C")

    # D: the dissociation
    ax = fig.add_subplot(gs[1, 0])
    tx = summ[summ.block == "tx"].groupby("condition")
    pre = summ[summ.block == "pre"].groupby("condition")
    state = (tx.d_H_state.mean()).reindex(order)
    struct = (tx.logdens_true.mean() - pre.logdens_true.mean()).reindex(order)
    ax.scatter(state.values, struct.values, s=70,
               c=[cols[c] for c in order], edgecolors="k", linewidths=0.5, zorder=3)
    for c, x, y in zip(order, state.values, struct.values):
        ax.annotate(labs[c], (x, y), fontsize=6.3, xytext=offs_d[c],
                    textcoords="offset points")
    ax.set_xlabel(r'state uncertainty removed  ("am I safe now?", nats)')
    ax.set_ylabel("structural knowledge gained (nats)")
    ax.set_title("The reassurance paradox, quantified")
    panel_label(ax, "D")

    # E: knowledge gained vs harm paid
    ax = fig.add_subplot(gs[1, 1])
    harm = tx.harm_rate.mean().reindex(order)
    ax.scatter(harm.values, struct.values, s=70, c=[cols[c] for c in order],
               edgecolors="k", linewidths=0.5, zorder=3)
    for c, x, y in zip(order, harm.values, struct.values):
        ha = "right" if offs_e[c][0] < 0 else "left"
        ax.annotate(labs[c], (x, y), fontsize=6.3, xytext=offs_e[c],
                    textcoords="offset points", ha=ha)
    ax.set_xlabel("harm experienced during treatment")
    ax.set_ylabel("structural knowledge gained (nats)")
    ax.set_title("The price of finding out")
    panel_label(ax, "E")

    # F: follow-up behaviour relative to baseline
    ax = fig.add_subplot(gs[1, 2])
    post = summ[summ.block == "post"].groupby("condition").mean_k.mean()
    base = summ[summ.block == "pre"].groupby("condition").mean_k.mean()
    delta = (post - base).reindex(order)
    ax.barh(np.arange(len(order)), delta.values,
            color=[cols[c] for c in order], height=0.62)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([labs[c] for c in order], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel(r"change in repetitions at follow-up  ($k_{post}-k_{pre}$)")
    ax.set_title("What is left after treatment stops")
    panel_label(ax, "F", dx=-0.55)

    return save(fig, "fig6_interventions")


# =============================================================== Figure 7
def fig7_predictions():
    """Falsifiable predictions for a human experiment."""
    if not _need("phase5_variance_matched.csv", "phase5_population.csv",
                 "phase5_regression.csv"):
        return None
    import matplotlib.pyplot as plt

    use_style()
    vm = pd.read_csv(RESULTS / "phase5_variance_matched.csv")
    pop = pd.read_csv(RESULTS / "phase5_population.csv")
    reg = pd.read_csv(RESULTS / "phase5_regression.csv")

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.2))

    # A: matched mean exposure, varying variability
    ax = axes[0]
    g = vm.groupby(["design", "var_k"]).sd_p.agg(["mean", "std"]).reset_index()
    g = g.sort_values("var_k")
    ax.errorbar(g.var_k, g["mean"], yerr=g["std"], fmt="o", ms=6, capsize=3,
                color=C["compulsive"], zorder=3)
    b, a = np.polyfit(g.var_k, g["mean"], 1)
    xs = np.linspace(g.var_k.min(), g.var_k.max(), 20)
    ax.plot(xs, a + b * xs, ls="--", lw=1.1, color=C["neutral"], zorder=1)
    for _, r in g.iterrows():
        ax.annotate(r["design"], (r.var_k, r["mean"]), fontsize=6.4,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(r"variance of the design, $\mathrm{Var}(k)$")
    ax.set_ylabel(r"posterior SD of efficacy, $\mathrm{SD}[p\mid D]$")
    ax.set_title(r"Matched exposure ($\mathbb{E}[k]=3$):"
                 "\nvariability, not amount, buys knowledge")
    panel_label(ax, "A")

    # B: the cost of that knowledge
    ax = axes[1]
    g2 = vm.groupby(["design", "var_k"])[["harm_rate", "abs_err_p"]].mean().reset_index()
    g2 = g2.sort_values("var_k")
    ax.plot(g2.harm_rate, g2.abs_err_p, "o", ms=6, color=C["healthy"])
    for _, r in g2.iterrows():
        ax.annotate(r["design"], (r.harm_rate, r.abs_err_p), fontsize=6.4,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("harm rate")
    ax.set_ylabel(r"$|\mathbb{E}[p\mid D]-p^\ast|$")
    ax.set_title("Every bit of self-knowledge\nis paid for in risk")
    panel_label(ax, "B")

    # C: population-level individual differences, stratified by severity
    ax = axes[2]
    sc = ax.scatter(pop.var_k, pop.sd_p, c=pop.severity, cmap="magma", s=13,
                    edgecolors="none", alpha=0.75)
    strata = pd.qcut(pop.severity, 3, labels=["low", "mid", "high"])
    for lab, ls in zip(["low", "mid", "high"], ["-", "--", ":"]):
        g3 = pop[strata == lab]
        b, a = np.polyfit(g3.var_k, g3.sd_p, 1)
        xs = np.linspace(g3.var_k.min(), g3.var_k.max(), 20)
        r = np.corrcoef(g3.var_k, g3.sd_p)[0, 1]
        ax.plot(xs, a + b * xs, ls, lw=1.5, color=C["ridge"],
                label=f"{lab} severity, $r={r:+.2f}$")
    ax.set_xlabel(r"behavioural variability $\mathrm{Var}(k)$")
    ax.set_ylabel(r"$\mathrm{SD}[p\mid D]$")
    ax.set_title("Matched on severity, those who vary\nknow their own efficacy better")
    ax.legend(loc="upper right", fontsize=6.2)
    fig.colorbar(sc, ax=ax, label="severity (mean $k$)", pad=0.02)
    panel_label(ax, "C")

    # D: nested model comparison, plus the stratified (non-circular) check
    ax = axes[3]
    labels = [r"$\mathbb{E}_0[p]$" + "\n(where they started)",
              r"$+$ severity", r"$+\ \mathrm{Var}(k)$",
              r"$+\ \mathrm{EIG}(\pi)$"]
    colr = [C["neutral"], C["mild"], C["graded"], C["healthy"]]
    ax.bar(np.arange(len(reg)), reg.adj_r2.values, color=colr, width=0.62)
    for i, (r2, pv) in enumerate(zip(reg.adj_r2, reg.p_vs_previous)):
        if not np.isfinite(pv):
            txt = f"{r2:.2f}"
        elif pv < 1e-6:
            txt = f"{r2:.2f}\n$p<10^{{-6}}$"
        elif pv < 0.05:
            txt = f"{r2:.2f}\n$p={pv:.3f}$"
        else:
            txt = f"{r2:.2f}\nn.s."
        ax.text(i, r2 + 0.02, txt, ha="center", fontsize=6.4)
    ax.set_xticks(np.arange(len(reg)))
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel(r"adjusted $R^2$ for $|\mathbb{E}[p\mid D]-p^\ast|$")
    ax.set_title("What predicts how accurate\nsomeone ends up about themselves")
    if (RESULTS / "phase5_stratified.csv").exists():
        st = pd.read_csv(RESULTS / "phase5_stratified.csv")
        ax.text(0.03, 0.97,
                "within severity strata:\n"
                + "\n".join(f"  $r(\\mathrm{{Var}}(k),\\mathrm{{SD}}[p])"
                            f"={r.corr_vark_sdp:+.2f}$  ($\\bar k$={r.mean_severity:.1f})"
                            for _, r in st.iterrows()),
                transform=ax.transAxes, va="top", ha="left", fontsize=6.0,
                color=C["neutral"])
    panel_label(ax, "D")

    return save(fig, "fig7_predictions")




def figS2_uncertainty_diagram():
    """Supplementary: the brief's axes -- initial uncertainty against harm cost."""
    if not _need("phase4_uncertainty_diagram.csv"):
        return None
    import matplotlib.pyplot as plt

    use_style()
    d = pd.read_csv(RESULTS / "phase4_uncertainty_diagram.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))

    g = d.groupby(["harm_cost", "concentration"]).abs_err_p.mean().unstack()
    ax = axes[0]
    im = ax.pcolormesh(g.columns.values, g.index.values, g.values,
                       cmap="viridis", shading="nearest")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("prior concentration  (low = uncertain)")
    ax.set_ylabel("cost of harm")
    ax.set_title(r"Residual error $|\mathbb{E}[p\mid D]-p^\ast|$"
                 "\n(prior mean held at 0.35)")
    fig.colorbar(im, ax=ax, pad=0.02)
    panel_label(ax, "A")

    ax = axes[1]
    g2 = d.groupby(["harm_cost", "concentration"]).mean_k.mean().unstack()
    im = ax.pcolormesh(g2.columns.values, g2.index.values, g2.values,
                       cmap="magma", shading="nearest")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("prior concentration  (low = uncertain)")
    ax.set_ylabel("cost of harm")
    ax.set_title("Amount of safety behaviour\n(mean $k$)")
    fig.colorbar(im, ax=ax, pad=0.02)
    panel_label(ax, "B")

    ax = axes[2]
    agg = d.groupby("concentration").agg(z=("z_p", "mean"), zs=("z_p", "std"),
                                         e=("abs_err_p", "mean"),
                                         es=("abs_err_p", "std"))
    ax.errorbar(agg.index, agg.z, yerr=agg.zs, fmt="o-", ms=5, capsize=3,
                color=C["compulsive"], label=r"miscalibration $z_p$")
    ax.axhline(2.0, color=C["neutral"], ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("prior concentration  (low = uncertain)")
    ax.set_ylabel(r"$|\mathbb{E}[p\mid D]-p^\ast| / \mathrm{SD}[p\mid D]$")
    ax.set_title("It is confident wrongness, not uncertainty,\n"
                 "that the trap converts into behaviour")
    ax2 = ax.twinx()
    ax2.plot(agg.index, agg.e, "s--", ms=4, color=C["healthy"], lw=1.2)
    ax2.set_ylabel(r"$|\mathbb{E}[p\mid D]-p^\ast|$", color=C["healthy"])
    ax2.tick_params(axis="y", colors=C["healthy"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C["healthy"])
    ax.legend(loc="upper left")
    panel_label(ax, "C")

    return save(fig, "fig_S2_uncertainty_diagram")


REGISTRY = {1: fig1_concept, 3: fig3_trajectories,
            4: fig4_phase_diagram, 5: fig5_hysteresis,
            6: fig6_interventions, 7: fig7_predictions,
            8: figS2_uncertainty_diagram}


def main():
    wanted = [int(a) for a in sys.argv[1:]] or sorted(REGISTRY)
    for k in wanted:
        fn = REGISTRY.get(k)
        if fn is None:
            print(f"[fig{k}] no such figure")
            continue
        print(f"[fig{k}] {fn.__doc__.splitlines()[0]}")
        out = fn()
        if out:
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
