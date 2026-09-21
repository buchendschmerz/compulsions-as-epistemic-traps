#!/usr/bin/env python3
r"""Phase 5 -- matched-exposure predictions for a human experiment.

Four randomized action schedules hold mean exposure at ``E[k] = 3`` while
changing the support of the design. The ``{1,5}`` and uniform ``0..6`` schedules
also match raw variance, allowing informative support to be distinguished from
variability itself.

Outputs
-------
results/phase5_variance_matched.csv
results/phase5_variance_matched_means.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from epistemic_trap.config import K_MAX, MASTER_SEED, RESULTS, WORLD
from epistemic_trap.policies import RandomizedPolicy
from epistemic_trap.simulate import AgentSpec, World, run_agent

KS = np.arange(K_MAX + 1)


# ------------------------------------------------ prediction 1: matched exposure
def matched_mean_designs(mean_k=3):
    """Policies with identical ``E[k]`` and increasing ``Var(k)``."""
    designs = {}
    designs[f"fixed {{{mean_k}}}"] = {mean_k: 1.0}
    for spread in range(1, mean_k):
        lo, hi = mean_k - spread, mean_k + spread
        designs[f"{{{lo},{hi}}}"] = {lo: 0.5, hi: 0.5}
    designs[f"uniform 0..{2 * mean_k}"] = {k: 1.0 for k in range(2 * mean_k + 1)}
    out = {}
    for name, spec in designs.items():
        probs = np.zeros(len(KS))
        for k, w in spec.items():
            probs[k] = w
        probs = probs / probs.sum()
        m = float((KS * probs).sum())
        v = float((KS ** 2 * probs).sum() - m ** 2)
        assert abs(m - mean_k) < 1e-9, (name, m)
        out[name] = (probs, m, v)
    return out


def run_variance_matched(world, n_agents, n_episodes, mean_k=3):
    rows = []
    for design_index, (name, (probs, m, v)) in enumerate(
            matched_mean_designs(mean_k).items()):
        for a in range(n_agents):
            spec = AgentSpec(prior=dict(a_q=3.0, b_q=7.0, a_p=1.5, b_p=13.5),
                             policy=RandomizedPolicy(k_values=KS, probs=probs),
                             label=name)
            df = run_agent(world, spec, n_episodes,
                           seed=MASTER_SEED + 71_000 + 997 * design_index + a,
                           log_every=n_episodes)
            last = df.iloc[-1]
            rows.append(dict(design=name, agent=a, mean_k=m, var_k=v,
                             realised_mean_k=float(df.attrs["k_history"].mean()),
                             realised_var_k=float(df.attrs["k_history"].var()),
                             sd_p=float(last.sd_p), sd_q=float(last.sd_q),
                             mean_p=float(last.mean_p), abs_err_p=float(abs(last.err_p)),
                             H_p=float(last.H_p), corr_qp=float(last.corr_qp),
                             logdens_true=float(last.logdens_true),
                             harm_rate=float(last.harm_rate_sofar),
                             singular=bool(last.fim_singular)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=1200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reanalyse", action="store_true",
                    help="recompute statistics from saved simulation output")
    args = ap.parse_args()
    if args.quick:
        args.n_agents, args.n_episodes = 5, 300

    world = World(**WORLD)
    t0 = time.time()

    if args.reanalyse:
        vm = pd.read_csv(RESULTS / "phase5_variance_matched.csv")
    else:
        vm = run_variance_matched(world, args.n_agents, args.n_episodes)
        vm.to_csv(RESULTS / "phase5_variance_matched.csv", index=False)
    vm_agg = (vm.groupby(["design", "var_k"])
              .agg(sd_p=("sd_p", "mean"), abs_err_p=("abs_err_p", "mean"),
                   logdens_true=("logdens_true", "mean"),
                   harm_rate=("harm_rate", "mean"),
                   realised_mean_k=("realised_mean_k", "mean"))
              .reset_index().sort_values("var_k"))
    print("[phase5] Prediction 1 -- matched E[k]=3, increasing Var(k):")
    print(vm_agg.round(4).to_string(index=False))
    vm_agg.to_csv(RESULTS / "phase5_variance_matched_means.csv", index=False)

    print(f"\n[phase5] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
