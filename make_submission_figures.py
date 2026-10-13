#!/usr/bin/env python3
"""Create concise, journal-ready figures for the DOCX/PDF manuscript.

The original figures remain untouched. These versions reduce panel count,
standardise the visual language, and are sized for a single-column manuscript.
"""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("OCD_RESULTS_DIR", ROOT / "results"))
OUT = Path(os.environ.get("OCD_MAIN_FIGURES_DIR", ROOT / "paper" / "figures_submission"))
OUT.mkdir(parents=True, exist_ok=True)

INK = "#18242F"
MUTED = "#68727D"
GRID = "#DCE3E8"
BLUE = "#2878B5"
BLUE_LIGHT = "#DCECF6"
RED = "#C44E52"
RED_LIGHT = "#F7E3E1"
GOLD = "#D99E2B"
TEAL = "#3A8F81"
PURPLE = "#6C5B9D"
BG = "#FFFFFF"


def read_result_csv(name, **kwargs):
    path = RESULTS / name
    compressed = path.with_suffix(path.suffix + ".gz")
    return pd.read_csv(compressed if compressed.exists() else path, **kwargs)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel(ax, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.0,
            fontweight="bold", va="top", ha="left", color=INK, zorder=20,
            clip_on=False)


def finish(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=360)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def box(ax, xy, width, height, text, face, edge, fontsize=8.2):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.25, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=INK)
    return patch


def arrow(ax, start, end, color=INK, rad=0.0, dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.35, color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3,
    )
    ax.add_patch(patch)


def figure1_concept() -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel(ax, "A", x=0.01)
    ax.set_title("Static-belief account", color=INK, pad=8, fontweight="bold")
    box(ax, (0.12, 0.73), 0.76, 0.14,
        "Subjective world model\n(point estimates)", BLUE_LIGHT, BLUE)
    box(ax, (0.18, 0.44), 0.64, 0.13,
        "Policy selects safety actions", "#F3F5F7", MUTED)
    box(ax, (0.18, 0.16), 0.64, 0.13,
        "State belief and outcomes", "#F3F5F7", MUTED)
    arrow(ax, (0.5, 0.73), (0.5, 0.57))
    arrow(ax, (0.5, 0.44), (0.5, 0.29))
    # State belief feeds back to the policy, but contingency beliefs remain fixed.
    arrow(ax, (0.66, 0.29), (0.66, 0.44), color=MUTED)
    ax.text(0.50, 0.04, "Contingency beliefs do not update",
            ha="center", color=RED, fontsize=8.2, fontweight="bold")

    ax = axes[1]
    panel(ax, "B", x=0.01)
    ax.set_title("Policy-induced epistemic trap", color=INK, pad=8,
                 fontweight="bold")
    box(ax, (0.18, 0.72), 0.64, 0.16,
        "Posterior belief over\nhazard q and efficacy p", BLUE_LIGHT, BLUE,
        fontsize=7.7)
    box(ax, (0.64, 0.43), 0.30, 0.13,
        "Policy\nrepetitions k", RED_LIGHT, RED)
    box(ax, (0.18, 0.15), 0.64, 0.12,
        "Outcome:  Y ~ Bernoulli[q(1-p)ᵏ]", "#F3F5F7", MUTED)
    box(ax, (0.03, 0.43), 0.33, 0.13,
        "Bayesian update", "#E4F2EE", TEAL)
    arrow(ax, (0.72, 0.74), (0.79, 0.56), color=RED)
    arrow(ax, (0.79, 0.43), (0.66, 0.27), color=RED)
    arrow(ax, (0.36, 0.27), (0.20, 0.43), color=TEAL)
    arrow(ax, (0.20, 0.56), (0.30, 0.74), color=BLUE)
    ax.text(0.50, 0.04,
            "If k never varies, q and p cannot be separated",
            ha="center", color=RED, fontsize=8.2, fontweight="bold")

    fig.suptitle("Safety behaviour controls risk - and controls what can be learned",
                 y=0.99, fontsize=11.2, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.84, wspace=0.18)
    finish(fig, "figure_1_epistemic_trap")


def figure2_identifiability() -> None:
    style()
    fig = plt.figure(figsize=(7.1, 3.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.04], wspace=0.58)

    ax = fig.add_subplot(gs[0, 0])
    ax_a = ax
    p = np.linspace(0.01, 0.93, 240)
    q_true, p_true = 0.30, 0.80
    for k, color in zip([1, 2, 4, 6], [PURPLE, BLUE, TEAL, GOLD]):
        lam = q_true * (1 - p_true) ** k
        q = lam / np.maximum((1 - p) ** k, 1e-12)
        q[q > 1] = np.nan
        ax.plot(p, q, color=color, lw=1.8, label=f"k = {k}")
    ax.scatter([p_true], [q_true], s=48, marker="*", color=GOLD,
               edgecolor=INK, linewidth=0.6, zorder=5)
    ax.set(xlim=(0, 0.95), ylim=(0, 1), xlabel="Preventive efficacy, p",
           ylabel="Baseline hazard, q", title="One repetition count gives\none likelihood ridge")
    ax.legend(frameon=True, facecolor=BG, edgecolor="none", framealpha=0.96,
              borderpad=0.25, ncol=2, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    ax_b = ax
    policies = pd.read_csv(RESULTS / "phase1_policy_identifiability.csv")
    labels = {
        "fixed k=0 (pure response prevention)": "Fixed: k=0",
        "fixed k=1": "Fixed: k=1",
        "fixed k=3": "Fixed: k=3",
        "fixed k=6 (compulsive)": "Fixed: k=6",
        "compulsive, jittered (k in {5,6,7})": "Jittered: 5-7",
        "k in {0,6} (baseline + compulsive)": "Contrast: {0,6}",
        "graded ERP (k in 0..6)": "Graded: 0-6",
        "k in {0,1}": "Contrast: {0,1}",
    }
    order = [
        "fixed k=0 (pure response prevention)", "fixed k=1", "fixed k=3",
        "fixed k=6 (compulsive)", "compulsive, jittered (k in {5,6,7})",
        "k in {0,6} (baseline + compulsive)", "graded ERP (k in 0..6)",
        "k in {0,1}",
    ]
    d = policies.set_index("policy").loc[order].copy()
    finite = np.isfinite(d.log10_det.to_numpy())
    singular_x = -2.55
    values = np.where(finite, d.log10_det.to_numpy(), singular_x)
    y = np.arange(len(d))
    ax.axvspan(-2.82, -2.24, color=RED_LIGHT, zorder=0)
    ax.scatter(values[finite], y[finite], s=38, color=BLUE, marker="o",
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(values[~finite], y[~finite], s=42, color=RED, marker="x",
               linewidth=1.5, zorder=3)
    ax.axvline(-2.18, color=GRID, lw=0.9)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.invert_yaxis()
    ax.set_xlim(-2.82, 6.15)
    ax.set_xticks([singular_x, -1, 1, 3, 5],
                  ["rank 1\n(det = 0)", "-1", "1", "3", "5"])
    ax.set_xlabel("log₁₀ det information (full-rank designs)")
    ax.set_title("Diversity, not repetition,\nrestores identifiability")
    ax.tick_params(axis="y", labelsize=6.8, pad=3)
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.75)

    ax = fig.add_subplot(gs[0, 2])
    ax_c = ax
    pair = pd.read_csv(RESULTS / "phase1_pair_grid.csv", index_col=0).to_numpy(float)
    pair = np.maximum(pair, 1e-6)
    im = ax.imshow(pair, origin="lower", cmap="magma", norm=LogNorm(vmin=1e-6, vmax=pair.max()))
    ax.plot([-0.5, 10.5], [-0.5, 10.5], color="white", lw=1.1, ls="--")
    ax.set(xticks=[0, 2, 4, 6, 8, 10], yticks=[0, 2, 4, 6, 8, 10],
           xlabel="Second repetition count, k₂", ylabel="First repetition count, k₁",
           title="Any two distinct counts\nidentify both parameters")
    ax.set_anchor("N")
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.03)
    cbar.set_label("det information")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.79)
    for label, axis in zip("ABC", (ax_a, ax_b, ax_c)):
        fig.text(axis.get_position().x0, 0.965, label, fontsize=10.0,
                 fontweight="bold", va="top", ha="left", color=INK)
    finish(fig, "figure_2_identifiability")


def trajectory_band(ax, data, condition, field, color, label):
    d = data[data.condition == condition]
    grouped = d.groupby("episode")[field]
    x = grouped.mean().index.to_numpy()
    mean = grouped.mean().to_numpy()
    lo = grouped.quantile(0.10).to_numpy()
    hi = grouped.quantile(0.90).to_numpy()
    ax.plot(x, mean, color=color, lw=1.8, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)


def figure3_learning() -> None:
    style()
    traj = pd.read_csv(RESULTS / "phase2_trajectories.csv")
    post = np.load(RESULTS / "phase2_posteriors.npz")
    fig = plt.figure(figsize=(7.1, 4.85))
    gs = fig.add_gridspec(2, 2, hspace=0.66, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    trajectory_band(ax, traj, "severe_doubt/fixed-k", "mean_p", RED,
                    "Fixed k = 6")
    trajectory_band(ax, traj, "severe_doubt/variable-k", "mean_p", BLUE,
                    "Variable k in {0,1,2,3}")
    ax.axhline(0.80, color=INK, ls=":", lw=1.0)
    ax.set(xlabel="Episode", ylabel="Mean efficacy belief, E[p | D]",
           title="Belief in personal efficacy")
    ax.legend(frameon=False, loc="lower right")
    panel(ax, "A", x=0.01, y=1.22)

    ax = fig.add_subplot(gs[0, 1])
    trajectory_band(ax, traj, "severe_doubt/fixed-k", "mean_q", RED,
                    "Fixed k = 6")
    trajectory_band(ax, traj, "severe_doubt/variable-k", "mean_q", BLUE,
                    "Variable k in {0,1,2,3}")
    ax.axhline(0.30, color=INK, ls=":", lw=1.0)
    ax.set(xlabel="Episode", ylabel="Mean hazard belief, E[q | D]",
           title="Belief about baseline hazard")
    panel(ax, "B", x=0.01, y=1.22)

    extent = [0, 1, 0, 1]
    for index, (key, title, label) in enumerate([
        ("severe_doubt/fixed-k", "Fixed policy: posterior remains a ridge", "C"),
        ("severe_doubt/variable-k", "Variable policy: posterior recovers truth", "D"),
    ]):
        ax = fig.add_subplot(gs[1, index])
        density = post[key]
        ax.imshow(density, origin="lower", extent=extent, cmap="Blues",
                  aspect="auto", interpolation="nearest")
        ax.scatter([0.80], [0.30], s=55, marker="*", color=GOLD,
                   edgecolor=INK, linewidth=0.6, zorder=5)
        ax.set(xlabel="Preventive efficacy, p", ylabel="Baseline hazard, q",
               title=title, xlim=(0, 1), ylim=(0, 1))
        panel(ax, label, x=0.01, y=1.22)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.91)
    finish(fig, "figure_3_learning")


def figure4_interventions() -> None:
    style()
    traj = read_result_csv("phase3_trajectories.csv")
    summary = pd.read_csv(RESULTS / "phase3_summary.csv")
    fig = plt.figure(figsize=(7.1, 5.25))
    gs = fig.add_gridspec(2, 2, hspace=0.88, wspace=0.43)
    # Use a colour-blind-friendly categorical palette so similar interventions
    # remain distinguishable in print and on projectors.
    colors = {
        "locked_k": "#4D4D4D",
        "reassurance": "#E69F00",
        "baseline_check": "#0072B2",
        "erp_k0": "#009E73",
        "graded_erp": "#CC79A7",
        "variable_k": "#D55E00",
    }
    labels = {
        "locked_k": "Locked k",
        "reassurance": "Post-action probe",
        "baseline_check": "Pre-action probe",
        "erp_k0": "Response prevention",
        "graded_erp": "Graded ERP",
        "variable_k": "Variable-k policy",
    }
    conditions = list(colors)
    markers = {
        "locked_k": "o",
        "reassurance": "s",
        "baseline_check": "^",
        "erp_k0": "D",
        "graded_erp": "P",
        "variable_k": "X",
    }
    linestyles = {
        "locked_k": "-",
        "reassurance": "--",
        "baseline_check": "-.",
        "erp_k0": ":",
        "graded_erp": (0, (3, 1, 1, 1)),
        "variable_k": (0, (6, 2)),
    }

    ax = fig.add_subplot(gs[0, 0])
    for condition in conditions:
        d = traj[traj.condition == condition]
        g = d.groupby("episode").mean_p.mean()
        ax.plot(g.index, g.values, lw=1.55, color=colors[condition],
                linestyle=linestyles[condition], label=labels[condition])
    ax.axvspan(400, 1000, color="#EEF1F3", zorder=-1)
    ax.axhline(0.80, color=INK, ls=":", lw=0.9)
    ax.set(xlabel="Episode", ylabel="Mean efficacy belief, E[p | D]",
           title="Efficacy belief during intervention")
    panel(ax, "A", x=-0.06)

    means = summary.groupby(["condition", "block"]).mean(numeric_only=True)
    ax = fig.add_subplot(gs[0, 1])
    for condition in conditions:
        pre = means.loc[(condition, "pre")]
        tx = means.loc[(condition, "tx")]
        gain = tx.logdens_true - pre.logdens_true
        ax.scatter(tx.harm_rate, gain, s=52, color=colors[condition],
                   marker=markers[condition], edgecolor=INK, linewidth=0.55,
                   zorder=3)
    ax.set(xlabel="Harm rate during treatment",
           ylabel="Knowledge gain at truth",
           title="Knowledge gain versus harm")
    panel(ax, "B", x=-0.06)

    ax = fig.add_subplot(gs[1, 0])
    for condition in conditions:
        pre = means.loc[(condition, "pre")]
        tx = means.loc[(condition, "tx")]
        gain = tx.logdens_true - pre.logdens_true
        state = tx.d_H_state
        ax.scatter(state, gain, s=52, color=colors[condition],
                   marker=markers[condition], edgecolor=INK, linewidth=0.55,
                   zorder=3)
    ax.axvline(0, color=GRID, lw=0.9)
    ax.set(xlabel="State uncertainty removed (nats)",
           ylabel="Gain in structural knowledge",
           title="State versus structural learning")
    reass = means.loc[("reassurance", "tx")]
    reass_pre = means.loc[("reassurance", "pre")]
    ax.annotate("Post-action probe", (reass.d_H_state,
                reass.logdens_true - reass_pre.logdens_true),
                xytext=(-72, 10), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": colors["reassurance"]},
                fontsize=6.8, color=colors["reassurance"])
    panel(ax, "C", x=-0.06)

    ax = fig.add_subplot(gs[1, 1])
    change = []
    for condition in conditions:
        pre = means.loc[(condition, "pre")].mean_k
        post_mean = means.loc[(condition, "post")].mean_k
        change.append(post_mean - pre)
    y = np.arange(len(conditions))
    ax.barh(y, change, color=[colors[c] for c in conditions], height=0.63)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y, [labels[c] for c in conditions])
    ax.invert_yaxis()
    ax.set(xlabel="Change in repetitions at follow-up",
           title="Change in repetitions at follow-up")
    panel(ax, "D", x=-0.06)

    handles = [
        Line2D([0], [0], color=colors[condition], lw=1.8,
               linestyle=linestyles[condition], marker=markers[condition],
               markeredgecolor=INK, markeredgewidth=0.45, markersize=5,
               label=labels[condition])
        for condition in conditions
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=7.0,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.45)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.91)
    finish(fig, "figure_4_interventions")


def figure5_predictions() -> None:
    style()
    vm = pd.read_csv(RESULTS / "phase5_variance_matched.csv")
    order = ["fixed {3}", "{2,4}", "{1,5}", "uniform 0..6"]
    display = ["Fixed {3}", "{2,4}", "{1,5}", "Uniform 0-6"]
    variances = [0, 1, 4, 4]
    colors = ["#4D4D4D", "#0072B2", "#D55E00", "#009E73"]
    schedules = np.zeros((4, 7))
    schedules[0, 3] = 1.0
    schedules[1, [2, 4]] = 0.5
    schedules[2, [1, 5]] = 0.5
    schedules[3, :] = 1 / 7

    fig = plt.figure(figsize=(7.1, 5.15))
    gs = fig.add_gridspec(2, 2, hspace=0.88, wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(schedules, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(schedules.shape[0]):
        for col in range(schedules.shape[1]):
            value = schedules[row, col]
            if value > 0:
                label = "1" if value == 1 else ("1/2" if value == 0.5 else "1/7")
                ax.text(col, row, label, ha="center", va="center",
                        fontsize=7.0, color="white" if value >= 0.5 else INK,
                        fontweight="bold")
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(4),
                  [f"{name}  (Var={variance})"
                   for name, variance in zip(display, variances)])
    ax.set(xlabel="Assigned action count, k",
           title="Four schedules with E[k] = 3")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel(ax, "A")

    grouped = vm[vm.design.isin(order)].groupby("design")
    means_sd = grouped.sd_p.mean().reindex(order)
    errors_sd = grouped.sd_p.std().reindex(order)
    means_err = grouped.abs_err_p.mean().reindex(order)
    errors_err = grouped.abs_err_p.std().reindex(order)
    y = np.arange(4)

    ax = fig.add_subplot(gs[0, 1])
    for i in range(4):
        ax.errorbar(means_sd.iloc[i], i, xerr=errors_sd.iloc[i], fmt="o",
                    color=colors[i], ecolor=colors[i], capsize=3, ms=5.5,
                    markeredgecolor=INK, markeredgewidth=0.4, zorder=3)
    ax.set_yticks(y, display)
    ax.invert_yaxis()
    ax.set(xlabel="Posterior SD of efficacy, p",
           title="Uncertainty after matched exposure")
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.8)
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    for i in range(4):
        ax.errorbar(means_err.iloc[i], i, xerr=errors_err.iloc[i], fmt="o",
                    color=colors[i], ecolor=colors[i], capsize=3, ms=5.5,
                    markeredgecolor=INK, markeredgewidth=0.4, zorder=3)
    ax.set_yticks(y, display)
    ax.invert_yaxis()
    ax.set(xlabel="Absolute error in final efficacy estimate",
           title="Primary simulated outcome")
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.8)
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.02, 0.59), 0.24, 0.25, "Randomize\nschedule", BLUE_LIGHT, BLUE,
        fontsize=7.5)
    box(ax, (0.36, 0.59), 0.30, 0.25, "Estimate q, p,\nand confidence\nby block",
        "#E4F2EE", TEAL, fontsize=6.8)
    box(ax, (0.75, 0.59), 0.23, 0.25, "Final efficacy\nerror\n(primary)", RED_LIGHT, RED,
        fontsize=7.0)
    arrow(ax, (0.26, 0.715), (0.37, 0.715), color=INK)
    arrow(ax, (0.66, 0.715), (0.75, 0.715), color=INK)
    ax.text(0.02, 0.39, "Primary contrast", fontsize=7.0, color=MUTED,
            fontweight="bold")
    ax.text(0.02, 0.29, "Fixed {3} vs informative schedules",
            fontsize=7.4, color=INK)
    ax.text(0.02, 0.17, "Same-variance diagnostic", fontsize=7.0,
            color=MUTED, fontweight="bold")
    ax.text(0.02, 0.09, "{1,5} vs uniform 0-6", fontsize=6.8, color=INK)
    ax.text(0.02, 0.01, "Optional free-choice transfer is secondary",
            fontsize=6.5, color=MUTED)
    ax.set_title("Preregistered measurement sequence")
    panel(ax, "D")

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.09, top=0.92)
    finish(fig, "figure_5_predictions_boundary")


def main() -> None:
    figure1_concept()
    figure2_identifiability()
    figure3_learning()
    figure4_interventions()
    figure5_predictions()
    print(f"Wrote submission figures to {OUT}")


if __name__ == "__main__":
    main()
