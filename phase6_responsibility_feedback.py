#!/usr/bin/env python3
r"""Phase 6 -- robustness: does responsibility feedback close the behavioural loop?

Phase 2 shows a limitation of the main model: the agent stays epistemically
wrong for thousands of episodes, but its repetition count still decays.  This
post-hoc robustness check adds one minimal psychological feedback that the main
model deliberately omitted: recent repetition raises the subjective cost of harm,
as if having acted repeatedly makes a bad outcome feel more personally
responsible.

This is not part of the central theorem.  It is a boundary-condition check: if
behavioural persistence appears only after such a term is added, the manuscript
should say that plainly.

Outputs
-------
results/phase6_responsibility_feedback.csv   logged trajectories
results/phase6_responsibility_summary.csv    per-strength summary
results/phase6_hypotheses.csv                numeric verdict
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from epistemic_trap.config import K_MAX, MASTER_SEED, N_GRID, PRIORS, RESULTS, WORLD
from epistemic_trap.policies import ResponsibilityFeedbackPolicy
from epistemic_trap.simulate import AgentSpec, World, run_agent

KS = np.arange(K_MAX + 1)


def _parse_strengths(text):
    return [float(x) for x in text.split(",") if x.strip()]


def run_sweep(world, strengths, n_agents, n_episodes, log_every, n_grid, decay,
              kappa):
    """Run the responsibility-feedback sweep."""
    traj, summary = [], []
    for si, strength in enumerate(strengths):
        for agent in range(n_agents):
            policy = ResponsibilityFeedbackPolicy(
                k_values=KS,
                beta=60.0,
                kappa=kappa,
                strength=float(strength),
                decay=decay,
            )
            spec = AgentSpec(
                prior=PRIORS["severe_doubt"],
                policy=policy,
                label=f"responsibility={strength:g}",
                n_grid=n_grid,
            )
            df = run_agent(
                world,
                spec,
                n_episodes,
                seed=MASTER_SEED + 160_000 + 1000 * si + agent,
                log_every=log_every,
                record_fisher=False,
            )
            df["strength"] = float(strength)
            df["agent"] = agent
            traj.append(df)

            k_hist = df.attrs["k_history"]
            early_n = max(50, n_episodes // 10)
            late_start = int(0.8 * n_episodes)
            final = df.iloc[-1]
            summary.append({
                "strength": float(strength),
                "agent": agent,
                "k_early": float(np.mean(k_hist[:early_n])),
                "k_late": float(np.mean(k_hist[late_start:])),
                "k_final": float(np.mean(k_hist[-early_n:])),
                "responsibility_final": float(policy.responsibility),
                "mean_p": float(final.mean_p),
                "mean_q": float(final.mean_q),
                "sd_p": float(final.sd_p),
                "abs_err_p": float(abs(final.err_p)),
                "harm_rate": float(final.harm_rate_sofar),
                "logdens_true": float(final.logdens_true),
            })
        print(f"[phase6] strength={strength:g} ({len(summary)} agents total)")
    return pd.concat(traj, ignore_index=True), pd.DataFrame(summary)


def summarise(agent_summary):
    """Aggregate the per-agent results and mark the first closed-loop condition."""
    agg = (agent_summary.groupby("strength")
           .agg(k_early=("k_early", "mean"),
                k_late=("k_late", "mean"),
                k_final=("k_final", "mean"),
                responsibility_final=("responsibility_final", "mean"),
                mean_p=("mean_p", "mean"),
                mean_q=("mean_q", "mean"),
                sd_p=("sd_p", "mean"),
                abs_err_p=("abs_err_p", "mean"),
                harm_rate=("harm_rate", "mean"),
                logdens_true=("logdens_true", "mean"))
           .reset_index())
    agg["persistence"] = agg.k_late / np.maximum(agg.k_early, 1e-9)
    agg["behavioural_closure"] = (agg.persistence >= 0.8) & (agg.k_late >= 4.0)
    return agg


def hypothesis_table(summary):
    """One explicit verdict for the manuscript-facing robustness check."""
    base = summary.loc[summary.strength == summary.strength.min()].iloc[0]
    closed = summary[summary.behavioural_closure]
    if len(closed):
        first = closed.sort_values("strength").iloc[0]
        supported = True
        value = float(first.persistence)
        comparison = float(base.persistence)
        note = (f"first closed-loop strength={first.strength:g}; "
                f"late k={first.k_late:.2f}")
    else:
        first = summary.sort_values("persistence", ascending=False).iloc[0]
        supported = False
        value = float(first.persistence)
        comparison = float(base.persistence)
        note = (f"no strength met the closure threshold; "
                f"best strength={first.strength:g}")
    return pd.DataFrame([{
        "hypothesis": "RESPONSIBILITY_FEEDBACK",
        "statement": "A responsibility feedback from recent repetition to subjective "
                     "harm cost can close the behavioural loop that uncertainty "
                     "aversion alone did not close.",
        "metric": "late/early repetition ratio, with late mean k >= 4",
        "value": value,
        "comparison": comparison,
        "supported": supported,
        "note": note,
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=8)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--strengths", default="0,2,4,8,16")
    ap.add_argument("--decay", type=float, default=0.98)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--n-grid", type=int, default=N_GRID)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_agents, args.n_episodes, args.log_every, args.n_grid = 3, 400, 20, 51

    t0 = time.time()
    world = World(**WORLD)
    strengths = _parse_strengths(args.strengths)
    traj, per_agent = run_sweep(world, strengths, args.n_agents, args.n_episodes,
                                args.log_every, args.n_grid, args.decay,
                                args.kappa)
    summary = summarise(per_agent)
    hyp = hypothesis_table(summary)

    traj.to_csv(RESULTS / "phase6_responsibility_feedback.csv", index=False)
    per_agent.to_csv(RESULTS / "phase6_responsibility_agents.csv", index=False)
    summary.to_csv(RESULTS / "phase6_responsibility_summary.csv", index=False)
    hyp.to_csv(RESULTS / "phase6_hypotheses.csv", index=False)

    print("\n[phase6] responsibility-feedback summary:")
    print(summary.round(3).to_string(index=False))
    print("\n[phase6] verdict:")
    print(hyp[["hypothesis", "value", "comparison", "supported", "note"]]
          .round(3).to_string(index=False))
    print(f"\n[phase6] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
