#!/usr/bin/env python3
"""Create concise, journal-ready figures for the DOCX/PDF manuscript.

The original figures remain untouched. These versions reduce panel count,
standardise the visual language, and are sized for a single-column manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from epistemic_trap.config import result_path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures_submission"
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


def panel(ax, label: str) -> None:
    ax.text(0.015, 0.985, label, transform=ax.transAxes, fontsize=10.0,
            fontweight="bold", va="top", ha="left", color=INK, zorder=20,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5,
                  "alpha": 0.88})


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
    panel(ax, "A")
    ax.set_title("Static-belief account", color=INK, pad=8, fontweight="bold")
    box(ax, (0.12, 0.73), 0.76, 0.14,
        "Subjective world model\n(point estimates)", BLUE_LIGHT, BLUE)
    box(ax, (0.18, 0.44), 0.64, 0.13,
        "Policy selects safety actions", "#F3F5F7", MUTED)
    box(ax, (0.18, 0.16), 0.64, 0.13,
        "State belief and outcomes", "#F3F5F7", MUTED)
    arrow(ax, (0.5, 0.73), (0.5, 0.57))
    arrow(ax, (0.5, 0.44), (0.5, 0.29))
    arrow(ax, (0.82, 0.225), (0.82, 0.505), color=MUTED, rad=-0.22)
    ax.text(0.50, 0.04, "Contingency beliefs do not update",
            ha="center", color=RED, fontsize=8.2, fontweight="bold")

    ax = axes[1]
    panel(ax, "B")
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
    arrow(ax, (0.36, 0.21), (0.20, 0.43), color=TEAL)
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
    fig = plt.figure(figsize=(7.1, 3.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.46)

    ax = fig.add_subplot(gs[0, 0])
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
    ax.legend(frameon=False, ncol=2, loc="upper left")
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    policies = pd.read_csv(RESULTS / "phase1_policy_identifiability.csv")
    labels = {
        "fixed k=0 (pure response prevention)": "Fixed k = 0",
        "fixed k=1": "Fixed k = 1",
        "fixed k=3": "Fixed k = 3",
        "fixed k=6 (compulsive)": "Fixed k = 6",
        "compulsive, jittered (k in {5,6,7})": "Jittered {5,6,7}",
        "k in {0,6} (baseline + compulsive)": "Contrast {0,6}",
        "graded ERP (k in 0..6)": "Graded 0...6",
        "k in {0,1}": "Contrast {0,1}",
    }
    order = [
        "fixed k=0 (pure response prevention)", "fixed k=1", "fixed k=3",
        "fixed k=6 (compulsive)", "compulsive, jittered (k in {5,6,7})",
        "k in {0,6} (baseline + compulsive)", "graded ERP (k in 0..6)",
        "k in {0,1}",
    ]
    d = policies.set_index("policy").loc[order].copy()
    values = d.log10_det.replace(-np.inf, -3.0).to_numpy()
    colors = [RED if singular else BLUE for singular in d.singular]
    y = np.arange(len(d))
    ax.barh(y, values, color=colors, height=0.64, edgecolor="white")
    ax.axvline(-3, color=MUTED, ls=":", lw=0.9)
    ax.set_yticks(y, [labels[item] for item in order])
    ax.invert_yaxis()
    ax.set_xlabel("log₁₀ det(Fisher information)")
    ax.set_title("Diversity, not repetition,\nrestores identifiability")
    ax.text(-2.92, 3.6, "singular", color=RED, fontsize=7.2, va="center")
    panel(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    pair = pd.read_csv(RESULTS / "phase1_pair_grid.csv", index_col=0).to_numpy(float)
    pair = np.maximum(pair, 1e-6)
    im = ax.imshow(pair, origin="lower", cmap="magma", norm=LogNorm(vmin=1e-6, vmax=pair.max()))
    ax.plot([-0.5, 10.5], [-0.5, 10.5], color="white", lw=1.1, ls="--")
    ax.set(xticks=[0, 2, 4, 6, 8, 10], yticks=[0, 2, 4, 6, 8, 10],
           xlabel="Second repetition count, k₂", ylabel="First repetition count, k₁",
           title="Any two distinct counts\nidentify both parameters")
    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.03)
    cbar.set_label("det information")
    panel(ax, "C")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.79)
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
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    trajectory_band(ax, traj, "severe_doubt/fixed-k", "mean_q", RED,
                    "Fixed k = 6")
    trajectory_band(ax, traj, "severe_doubt/variable-k", "mean_q", BLUE,
                    "Variable k in {0,1,2,3}")
    ax.axhline(0.30, color=INK, ls=":", lw=1.0)
    ax.set(xlabel="Episode", ylabel="Mean hazard belief, E[q | D]",
           title="Belief about baseline hazard")
    panel(ax, "B")

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
        panel(ax, label)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.91)
    finish(fig, "figure_3_learning")


def figure4_interventions() -> None:
    style()
    traj = pd.read_csv(result_path("phase3_trajectories.csv"))
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
        "reassurance": "Check after acting",
        "baseline_check": "Check before acting",
        "erp_k0": "Response prevention",
        "graded_erp": "Graded ERP",
        "variable_k": "Variable-k policy",
    }

    ax = fig.add_subplot(gs[0, 0])
    for condition in ["locked_k", "reassurance", "graded_erp", "variable_k"]:
        d = traj[traj.condition == condition]
        g = d.groupby("episode").mean_p.mean()
        ax.plot(g.index, g.values, lw=1.45, color=colors[condition], label=labels[condition])
    ax.axvspan(400, 1000, color="#EEF1F3", zorder=-1)
    ax.axhline(0.80, color=INK, ls=":", lw=0.9)
    ax.set(xlabel="Episode", ylabel="Mean efficacy belief, E[p | D]",
           title="Efficacy belief during intervention")
    panel(ax, "A")

    means = summary.groupby(["condition", "block"]).mean(numeric_only=True)
    conditions = list(colors)
    ax = fig.add_subplot(gs[0, 1])
    for condition in conditions:
        pre = means.loc[(condition, "pre")]
        tx = means.loc[(condition, "tx")]
        gain = tx.logdens_true - pre.logdens_true
        ax.scatter(tx.harm_rate, gain, s=46, color=colors[condition],
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.set(xlabel="Harm rate during treatment",
           ylabel="Knowledge gain at truth",
           title="Knowledge gained versus harm experienced")
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    for condition in conditions:
        pre = means.loc[(condition, "pre")]
        tx = means.loc[(condition, "tx")]
        gain = tx.logdens_true - pre.logdens_true
        state = tx.d_H_state
        ax.scatter(state, gain, s=46, color=colors[condition],
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(0, color=GRID, lw=0.9)
    ax.set(xlabel="State uncertainty removed (nats)",
           ylabel="Gain in structural knowledge",
           title="State uncertainty versus structural knowledge")
    reass = means.loc[("reassurance", "tx")]
    reass_pre = means.loc[("reassurance", "pre")]
    ax.annotate("Check after acting", (reass.d_H_state,
                reass.logdens_true - reass_pre.logdens_true),
                xytext=(-72, 10), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": colors["reassurance"]},
                fontsize=6.8, color=colors["reassurance"])
    panel(ax, "C")

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
    panel(ax, "D")

    handles = [
        Line2D([0], [0], color=colors[condition], lw=2.2, label=labels[condition])
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
    pop = pd.read_csv(RESULTS / "phase5_population.csv")
    reg = pd.read_csv(RESULTS / "phase5_regression.csv")
    resp = pd.read_csv(RESULTS / "phase6_responsibility_summary.csv")
    fig = plt.figure(figsize=(7.1, 5.05))
    gs = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.40)

    ax = fig.add_subplot(gs[0, 0])
    g = vm.groupby(["design", "var_k"]).sd_p.agg(["mean", "std"]).reset_index().sort_values("var_k")
    ax.errorbar(g.var_k, g["mean"], yerr=g["std"], fmt="o", color=BLUE,
                ecolor=BLUE, capsize=3, ms=5)
    label_offsets = {
        "fixed {3}": (7, -16),
        "{2,4}": (7, 7),
        "{1,5}": (7, 7),
        "uniform 0..6": (7, -16),
        "{0,6}": (7, 7),
    }
    for _, row in g.iterrows():
        ax.annotate(
            row.design,
            (row.var_k, row["mean"]),
            xytext=label_offsets.get(row.design, (7, 7)),
            textcoords="offset points",
            fontsize=6.6,
        )
    ax.set(xlabel="Variance of repetition count, Var(k)",
           ylabel="Posterior SD of efficacy",
           title="Matched exposure: policy variance and precision")
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    strata = pd.qcut(pop.severity, 3, labels=["Low", "Mid", "High"])
    for label, color, marker in zip(["Low", "Mid", "High"], [BLUE, PURPLE, GOLD], ["o", "s", "^"]):
        d = pop[strata == label]
        ax.scatter(d.var_k, d.sd_p, s=12, alpha=0.35, color=color, marker=marker,
                   edgecolor="none")
        slope, intercept = np.polyfit(d.var_k, d.sd_p, 1)
        xs = np.linspace(d.var_k.min(), d.var_k.max(), 30)
        corr = np.corrcoef(d.var_k, d.sd_p)[0, 1]
        ax.plot(xs, intercept + slope * xs, color=color, lw=1.5,
                label=f"{label} severity, r = {corr:.2f}")
    ax.set(xlabel="Behavioural variability, Var(k)", ylabel="Posterior SD of efficacy",
           title="Policy variance within severity strata")
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    labels = ["Initial belief", "+ severity", "+ Var(k)", "+ policy information"]
    bars = ax.bar(np.arange(len(reg)), reg.adj_r2, color=[MUTED, GOLD, PURPLE, BLUE], width=0.66)
    ax.set_xticks(np.arange(len(reg)), labels, rotation=16, ha="right")
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Adjusted R²")
    ax.set_title("Nested prediction of final efficacy error")
    for bar, value in zip(bars, reg.adj_r2):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.015, f"{value:.2f}",
                ha="center", fontsize=7.2)
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(resp.strength, resp.persistence, "o-", color=RED, lw=1.8, ms=5)
    ax.axhline(0.80, color=INK, ls="--", lw=0.9)
    ax.fill_between(resp.strength, 0.80, 1.02, color=RED_LIGHT, alpha=0.7)
    closed = resp[resp.behavioural_closure.astype(bool)]
    if not closed.empty:
        row = closed.iloc[0]
        ax.annotate(f"Loop closes at α = {row.strength:g}",
                    (row.strength, row.persistence), xytext=(-86, -43),
                    textcoords="offset points", arrowprops={"arrowstyle": "->", "color": RED},
                    fontsize=7.2, color=RED,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.92, "pad": 1.2})
    ax.set(xlabel="Responsibility-feedback strength, α",
           ylabel="Late / early repetition ratio", ylim=(0.25, 1.03),
           title="Responsibility feedback and behavioural persistence")
    panel(ax, "D")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.91)
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
