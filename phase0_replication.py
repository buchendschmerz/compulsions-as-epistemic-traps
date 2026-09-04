#!/usr/bin/env python3
"""Phase 0 -- replication of Rigoux, Stephan & Petzschner (2024).

Reproduces the paper's central result: among agents acting optimally under
idiosyncratic *point* beliefs in worlds where compulsion is not optimal, the one
belief that separates compulsive from non-compulsive agents is an
underestimation of the effectiveness of the preventive action
(``Delta p_SUCCESS < 0``); no other belief distortion separates the groups.

This establishes the baseline our Bayesian extension departs from: in the
original model ``p_SUCCESS`` is a static point estimate, so the model can say
that compulsive agents *hold* a pessimistic belief but not why that belief is
never corrected.

Outputs
-------
results/phase0_worlds.csv        selected worlds
results/phase0_agents.csv        per-agent belief distortions and severity
results/phase0_group_stats.csv   group comparison (Cohen's d + bootstrap p)
figures/fig_S1_replication.pdf   Fig 3/5 equivalents
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from epistemic_trap.config import MASTER_SEED, RESULTS
from epistemic_trap.plotstyle import C, panel_label, save, use_style
from epistemic_trap.pomdp import (PARAM_NAMES, compulsive_episodes,
                                  sample_params, simulate_sequence, solve_pomdp)

PRETTY = {
    "p_soiling": r"$\Delta p_{\mathrm{SOILING}}$",
    "p_success": r"$\Delta p_{\mathrm{SUCCESS}}$",
    "d_dirty": r"$\Delta p_{\mathrm{DETECT,DIRTY}}$",
    "d_clean": r"$\Delta p_{\mathrm{DETECT,CLEAN}}$",
    "c_sick": r"$\Delta \mathrm{cost}_{\mathrm{SICKNESS}}$",
    "c_wash": r"$\Delta \mathrm{cost}_{\mathrm{WASH}}$",
    "c_check": r"$\Delta \mathrm{cost}_{\mathrm{CHECK}}$",
    "gamma": r"$\Delta \gamma$",
}


# --------------------------------------------------------------------- selection
def select_worlds(n_worlds, rng, n_steps, n_belief, max_tries=200_000):
    """Worlds whose *optimal* policy uses all three actions and is not compulsive.

    This is the paper's world-selection criterion: it removes environments in
    which repeating a preventive action is simply the rational thing to do, so
    that any compulsion observed later is attributable to belief, not to reality.
    """
    worlds, tries = [], 0
    while len(worlds) < n_worlds and tries < max_tries:
        tries += 1
        w = sample_params(rng)
        b, V, pol = solve_pomdp(w, n_belief=n_belief)
        if set(pol.tolist()) != {0, 1, 2}:
            continue
        actions, _ = simulate_sequence(w, w, n_steps, rng, solved=(b, V, pol))
        ce = compulsive_episodes(actions)
        if ce["n_episodes"] == 0 and ce["frac_noncook"] < 0.9:
            worlds.append(w)
    if len(worlds) < n_worlds:
        raise RuntimeError(f"only found {len(worlds)}/{n_worlds} valid worlds")
    return worlds


MIN_COOK = 5  # an agent must actually act on the world for "episodes" to be defined


def simulate_agents(world, n_per_group, rng, n_steps, n_belief, max_tries=4000):
    """Collect ``n_per_group`` compulsive and non-compulsive agents for one world.

    Deviation from the original, documented in the manuscript: agents whose
    subjectively optimal policy never selects the consequential action (they stall
    forever on the cheapest action) are excluded rather than counted as
    compulsive.  For such agents a "compulsive episode" -- a run of preventive
    actions *between* consequential actions -- is not defined, and including them
    would let a degenerate corner of the cost simplex dominate the comparison.
    """
    comp, noncomp, abstain, tries = [], [], 0, 0
    w_arr = world.as_array()
    while (len(comp) < n_per_group or len(noncomp) < n_per_group) and tries < max_tries:
        tries += 1
        agent = sample_params(rng)
        b, V, pol = solve_pomdp(agent, n_belief=n_belief)
        actions, _ = simulate_sequence(world, agent, n_steps, rng, solved=(b, V, pol))
        ce = compulsive_episodes(actions)
        rec = dict(zip([f"d_{n}" for n in PARAM_NAMES], agent.as_array() - w_arr))
        rec.update({f"agent_{n}": v for n, v in zip(PARAM_NAMES, agent.as_array())})
        rec.update(n_comp_episodes=ce["n_episodes"], max_len=ce["max_len"],
                   mean_len=ce["mean_len"], frac_noncook=ce["frac_noncook"],
                   n_cook=ce["n_cook"])
        if ce["n_cook"] < MIN_COOK:
            abstain += 1
            continue
        if ce["n_episodes"] > 0:
            rec["group"] = "compulsive"
            if len(comp) < n_per_group:
                comp.append(rec)
        else:
            rec["group"] = "non_compulsive"
            if len(noncomp) < n_per_group:
                noncomp.append(rec)
    return comp + noncomp, abstain


# ------------------------------------------------------------------- statistics
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
                 / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


def cluster_bootstrap_p(agents, col, rng, n_boot=5000):
    """Two-sided cluster bootstrap on the group difference, clustered by world.

    The unit of replication is the *world*, not the agent: the 60 agents drawn
    inside one world share its parameters, so an agent-level test treats
    correlated observations as independent and returns a vanishing p-value for
    every parameter.  We therefore form the per-world group difference

        d_w = mean(compulsive in w) - mean(non-compulsive in w),

    resample worlds with replacement, and ask how often the resampled mean of
    ``d_w`` crosses zero.  The resulting p-value is driven by between-world
    variability, which is what the original paper's bootstrap was protecting.
    """
    per_world = []
    for w, sub in agents.groupby("world"):
        x = sub.loc[sub.group == "compulsive", col]
        y = sub.loc[sub.group == "non_compulsive", col]
        if len(x) and len(y):
            per_world.append(x.mean() - y.mean())
    d = np.asarray(per_world, dtype=float)
    if len(d) < 2:
        return np.nan
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    frac_neg = float(np.mean(boots <= 0))
    frac_pos = float(np.mean(boots >= 0))
    return float(min(1.0, 2 * min(frac_neg, frac_pos) + 1 / n_boot))


def bootstrap_ci_mean(x, rng, n_boot=2000, alpha=0.05):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = np.asarray(x)[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def severity_stats(agents):
    """Rank correlations between efficacy doubt and compulsion severity.

    Episode lengths are heavy-tailed (a handful of agents produce runs hundreds of
    actions long), so Pearson correlations here are dominated by the tail. We
    report Spearman, and the same quantity computed within worlds.
    """
    from scipy import stats as sps
    sub = agents[agents.group == "compulsive"].copy()
    rows = []
    for m in ("max_len", "mean_len", "n_comp_episodes", "frac_noncook"):
        rho, pv = sps.spearmanr(sub.d_p_success, sub[m])
        dev_x = sub.groupby("world").d_p_success.transform(lambda s: s - s.mean())
        dev_y = sub.groupby("world")[m].transform(lambda s: s.rank() - s.rank().mean())
        rows.append(dict(measure=m, spearman_rho=float(rho), spearman_p=float(pv),
                         within_world_r=float(np.corrcoef(dev_x, dev_y)[0, 1])))
    return pd.DataFrame(rows)


def compute_stats(agents, rng):
    """Group comparison for every belief parameter."""
    rows = []
    for name in PARAM_NAMES:
        x = agents.loc[agents.group == "compulsive", f"d_{name}"].values
        y = agents.loc[agents.group == "non_compulsive", f"d_{name}"].values
        lo_c, hi_c = bootstrap_ci_mean(x, rng)
        rows.append(dict(param=name, mean_compulsive=float(np.mean(x)),
                         sd_compulsive=float(np.std(x, ddof=1)),
                         ci_lo_compulsive=lo_c, ci_hi_compulsive=hi_c,
                         mean_noncompulsive=float(np.mean(y)),
                         sd_noncompulsive=float(np.std(y, ddof=1)),
                         cohens_d=cohens_d(x, y),
                         n_worlds=int(agents.world.nunique()),
                         boot_p=cluster_bootstrap_p(agents, f"d_{name}", rng)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- figure
def make_figure(agents, stats):
    import matplotlib.pyplot as plt

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))

    # A: belief distortions by group
    ax = axes[0]
    names = list(PRETTY)
    pos = np.arange(len(names))
    for j, (grp, col) in enumerate([("non_compulsive", C["healthy"]),
                                    ("compulsive", C["compulsive"])]):
        sub = agents[agents.group == grp]
        m = [sub[f"d_{n}"].mean() for n in names]
        lo = [np.quantile(sub[f"d_{n}"], 0.25) for n in names]
        hi = [np.quantile(sub[f"d_{n}"], 0.75) for n in names]
        off = (j - 0.5) * 0.3
        ax.errorbar(np.array(m), pos + off,
                    xerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)],
                    fmt="o", ms=4, lw=1.2, color=col, capsize=2,
                    label=grp.replace("_", "-"))
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_yticks(pos)
    ax.set_yticklabels([PRETTY[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel(r"belief distortion  (agent $-$ world)")
    ax.set_title("Belief distortions by group\n(mean, with interquartile range)")
    ax.legend(loc="lower left")
    panel_label(ax, "A", dx=-0.42)

    # B: effect sizes
    ax = axes[1]
    d = stats.set_index("param").loc[names, "cohens_d"].values
    colors = [C["compulsive"] if n == "p_success" else C["neutral"] for n in names]
    ax.barh(pos, d, color=colors, height=0.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels([PRETTY[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's $d$ (compulsive $-$ non-compulsive)")
    ax.set_title("Effect sizes")
    panel_label(ax, "B", dx=-0.42)

    # C: severity gradient. Episode lengths are heavy-tailed, so we summarise with
    # medians and quantify with Spearman rank correlation rather than Pearson.
    ax = axes[2]
    from scipy import stats as sps
    sub = agents[agents.group == "compulsive"]
    bins = np.unique(np.quantile(sub.d_p_success, np.linspace(0, 1, 9)))
    idx = np.clip(np.digitize(sub.d_p_success, bins) - 1, 0, len(bins) - 2)
    xs = 0.5 * (bins[:-1] + bins[1:])
    med = [np.median(sub.max_len.values[idx == i]) for i in range(len(xs))]
    q1 = [np.quantile(sub.max_len.values[idx == i], 0.25) for i in range(len(xs))]
    q3 = [np.quantile(sub.max_len.values[idx == i], 0.75) for i in range(len(xs))]
    ax.scatter(sub.d_p_success, sub.max_len, s=4, alpha=0.14,
               color=C["compulsive"], edgecolors="none")
    ax.fill_between(xs, q1, q3, color=C["ridge"], alpha=0.15, linewidth=0)
    ax.plot(xs, med, "o-", color=C["ridge"], ms=4, label="median (IQR)")
    rho, pv = sps.spearmanr(sub.d_p_success, sub.max_len)
    rho2, _ = sps.spearmanr(sub.d_p_success, sub.frac_noncook)
    ax.set_xlabel(r"$\Delta p_{\mathrm{SUCCESS}}$")
    ax.set_ylabel("longest compulsive episode")
    ax.set_yscale("log")
    ax.set_title("More doubt, longer compulsive episodes")
    ax.text(0.03, 0.04,
            rf"Spearman $\rho={rho:+.2f}$ (episode length)" "\n"
            rf"$\rho={rho2:+.2f}$ (fraction of non-consequential actions)",
            transform=ax.transAxes, fontsize=6.8, color=C["neutral"], va="bottom")
    ax.legend(loc="upper right")
    panel_label(ax, "C", dx=-0.30)

    return save(fig, "fig_S1_replication")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=40)
    ap.add_argument("--n-per-group", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=600)
    ap.add_argument("--n-belief", type=int, default=201)
    ap.add_argument("--quick", action="store_true",
                    help="tiny run for smoke-testing the pipeline")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute statistics and the figure from saved agents")
    args = ap.parse_args()
    if args.quick:
        args.n_worlds, args.n_per_group, args.n_steps = 4, 10, 300

    rng = np.random.default_rng(MASTER_SEED)
    t0 = time.time()

    if args.stats_only:
        agents = pd.read_csv(RESULTS / "phase0_agents.csv")
        stats = compute_stats(agents, rng)
        stats.to_csv(RESULTS / "phase0_group_stats.csv", index=False)
        print(stats.round(4).to_string(index=False))
        sev = severity_stats(agents)
        sev.to_csv(RESULTS / "phase0_severity_stats.csv", index=False)
        print("\n[phase0] severity gradient among compulsive agents:")
        print(sev.round(4).to_string(index=False))
        print(f"[phase0] figure -> {make_figure(agents, stats)}")
        return

    print(f"[phase0] selecting {args.n_worlds} non-compulsive-optimal worlds ...")
    worlds = select_worlds(args.n_worlds, rng, args.n_steps, args.n_belief)
    wdf = pd.DataFrame([w.as_array() for w in worlds], columns=PARAM_NAMES)
    wdf.to_csv(RESULTS / "phase0_worlds.csv", index=False)
    print(f"[phase0]   done ({time.time() - t0:.1f}s)")

    records, n_abstain = [], 0
    for i, w in enumerate(worlds):
        recs, ab = simulate_agents(w, args.n_per_group, rng, args.n_steps, args.n_belief)
        n_abstain += ab
        for r in recs:
            r["world"] = i
        records += recs
        print(f"[phase0] world {i + 1}/{len(worlds)}: {len(recs)} agents "
              f"({time.time() - t0:.0f}s)", end="\r")
    agents = pd.DataFrame(records)
    agents.to_csv(RESULTS / "phase0_agents.csv", index=False)
    print(f"\n[phase0] {len(agents)} agents simulated "
          f"({n_abstain} never-cooking agents excluded)")

    stats = compute_stats(agents, rng)
    stats.to_csv(RESULTS / "phase0_group_stats.csv", index=False)
    sev = severity_stats(agents)
    sev.to_csv(RESULTS / "phase0_severity_stats.csv", index=False)
    print("\n[phase0] severity gradient among compulsive agents:")
    print(sev.round(4).to_string(index=False))
    print(stats.round(4).to_string(index=False))

    path = make_figure(agents, stats)
    print(f"[phase0] figure -> {path}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
