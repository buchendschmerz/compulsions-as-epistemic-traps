#!/usr/bin/env python3
r"""Phase 2 -- Bayesian extension: ``p_SUCCESS`` and ``p_SOILING`` become learned.

Rigoux et al. treat the agent's beliefs about world contingencies as static point
estimates, and note themselves that separating *confidence* from *bias* requires a
Bayesian formulation.  Here the agent carries a full posterior
``P(q, p | data)`` on a 101x101 grid, updates it exactly, and chooses how many
preventive repetitions to perform from that posterior.

Everything about the learner is normative.  What varies is the *design* its own
policy generates.

Hypotheses addressed
--------------------
H1  Safety-behaviour censoring: agents that repeat more retain higher posterior
    entropy about ``p`` while their uncertainty about the *observable composite*
    ``lambda_k`` collapses.
H2  Self-maintaining compulsion: initial doubt raises ``k``, raising ``k``
    lowers the information yield per episode, which leaves the doubt uncorrected.
H3  Identifiability: ``q`` and ``p`` become strongly *positively* coupled in the
    posterior of repetitive agents, as predicted analytically in Phase 1.

Outputs
-------
results/phase2_trajectories.csv   posterior statistics over episodes
results/phase2_summary.csv        end-of-run summary per condition
results/phase2_hypotheses.csv     numeric verdicts on H1-H3
results/phase2_posteriors.npz     final joint posteriors for Figure 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from epistemic_trap.config import (K_MAX, MASTER_SEED, N_GRID, PRIORS,
                                   RESULTS, WORLD)
from epistemic_trap.grid import ParameterGrid
from epistemic_trap.policies import GreedyPolicy, RandomizedPolicy, action_utilities
from epistemic_trap.simulate import AgentSpec, World, run_agent

BETA_DETERMINISTIC = 60.0  # near-argmax: the harm-minimising agent


def initial_k_star(prior):
    """The ``k`` a greedy agent picks on its very first episode, from the prior."""
    g = ParameterGrid.from_beta_prior(**prior)
    u = action_utilities(g, np.arange(K_MAX + 1))
    return int(np.argmax(u))


def conditions():
    """The agent x policy grid simulated in Phase 2."""
    out = []
    for pname, prior in PRIORS.items():
        k0 = initial_k_star(prior)
        out.append(dict(condition=f"{pname}/closed-loop", prior_name=pname,
                        prior=prior, policy_kind="greedy", k_fixed=k0))
        out.append(dict(condition=f"{pname}/fixed-k", prior_name=pname,
                        prior=prior, policy_kind="fixed", k_fixed=k0))
    # A reference agent whose policy is deliberately diverse (the identifiable design)
    out.append(dict(condition="severe_doubt/variable-k", prior_name="severe_doubt",
                    prior=PRIORS["severe_doubt"], policy_kind="variable",
                    k_fixed=initial_k_star(PRIORS["severe_doubt"])))
    return out


def build_policy(cond):
    ks = np.arange(K_MAX + 1)
    if cond["policy_kind"] == "greedy":
        return GreedyPolicy(k_values=ks, beta=BETA_DETERMINISTIC)
    if cond["policy_kind"] == "fixed":
        from epistemic_trap.policies import FixedKPolicy
        return FixedKPolicy(k_values=ks, k=cond["k_fixed"])
    probs = np.zeros(len(ks))
    probs[:4] = 1.0  # k in {0,1,2,3}
    return RandomizedPolicy(k_values=ks, probs=probs)


def prior_reference(prior, k_ref):
    """Prior-side values of every logged statistic, for computing reductions."""
    g = ParameterGrid.from_beta_prior(**prior)
    return g.summary(k_ref=k_ref)


def k_sweep(world, n_agents, n_episodes, prior_name="accurate"):
    """Hold the prior fixed and vary only the design: fixed ``k`` from 0 to K_MAX.

    This isolates the effect of the *policy* on what can be learned.  Everything
    else -- prior, learner, environment -- is identical across arms, so any
    difference in posterior geometry is attributable to the design alone.
    """
    from epistemic_trap.policies import FixedKPolicy
    ks = np.arange(K_MAX + 1)
    prior = PRIORS[prior_name]
    rows = []
    for k in ks:
        for a in range(n_agents):
            spec = AgentSpec(prior=prior, policy=FixedKPolicy(k_values=ks, k=int(k)),
                             label=f"fixed k={k}")
            df = run_agent(world, spec, n_episodes,
                           seed=MASTER_SEED + 90_000 + 100 * int(k) + a,
                           log_every=n_episodes)
            r = df.iloc[-1].to_dict()
            r.update(k_fixed=int(k), agent=a, prior_name=prior_name)
            rows.append(r)
    return pd.DataFrame(rows)


def kappa_sweep(world, n_agents, n_episodes, kappas=(0.0, 0.5, 1.0, 2.0, 4.0),
                prior_name="severe_doubt", log_every=10):
    r"""Does the loop close? Uncertainty aversion and self-maintaining compulsion.

    A purely harm-minimising agent that starts out doubting itself is wrong for a
    very long time, but its *behaviour* drifts back down as its belief slowly
    improves.  H2 -- a genuinely self-maintaining compulsion -- needs one more
    ingredient, and the model supplies an independently motivated one:
    intolerance of uncertainty.  An agent that pays for the *variance* of the
    hazard as well as its mean (``kappa > 0``) repeats more; repeating more
    suppresses the evidence; the suppressed evidence keeps the variance high.
    That is the feedback loop, written out.
    """
    prior = PRIORS[prior_name]
    rows = []
    for kap in kappas:
        for a in range(n_agents):
            spec = AgentSpec(prior=prior,
                             policy=GreedyPolicy(k_values=np.arange(K_MAX + 1),
                                                 beta=BETA_DETERMINISTIC,
                                                 kappa=float(kap)),
                             label=f"kappa={kap}")
            df = run_agent(world, spec, n_episodes,
                           seed=MASTER_SEED + 95_000 + 300 * int(kap * 10) + a,
                           log_every=log_every)
            df["kappa"], df["agent"] = float(kap), a
            rows.append(df[["episode", "kappa", "agent", "k", "mean_p", "mean_q",
                            "sd_p", "H_p", "corr_qp", "err_p", "err_q",
                            "mean_k_sofar", "harm_rate_sofar", "logdens_true",
                            "fim_singular"]])
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=20)
    ap.add_argument("--n-episodes", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-kappa", action="store_true",
                    help="run only the uncertainty-aversion sweep")
    args = ap.parse_args()
    if args.quick:
        args.n_agents, args.n_episodes, args.log_every = 3, 300, 25

    world = World(**WORLD)
    conds = conditions()
    t0 = time.time()

    if args.only_kappa:
        ks = kappa_sweep(world, args.n_agents, args.n_episodes,
                         log_every=args.log_every)
        ks.to_csv(RESULTS / "phase2_kappa_sweep.csv", index=False)
        fin = ks[ks.episode == ks.episode.max()]
        early = ks[ks.episode <= max(50, ks.episode.min())]
        k0 = early.groupby("kappa").k.mean()
        late = ks[ks.episode >= 0.8 * ks.episode.max()].groupby("kappa").k.mean()
        agg = (fin.groupby("kappa")
               .agg(mean_k=("mean_k_sofar", "mean"), mean_p=("mean_p", "mean"),
                    err_p=("err_p", "mean"), sd_p=("sd_p", "mean"),
                    corr_qp=("corr_qp", "mean"), harm=("harm_rate_sofar", "mean"),
                    logdens_true=("logdens_true", "mean"))
               .reset_index())
        agg["k_early"] = agg.kappa.map(k0)
        agg["k_late"] = agg.kappa.map(late)
        # H2: a self-maintaining compulsion is one whose action rate does not decay
        agg["persistence"] = agg.k_late / agg.k_early
        agg.to_csv(RESULTS / "phase2_kappa_means.csv", index=False)
        print("[phase2] uncertainty aversion (severe-doubt prior):")
        print(agg.round(3).to_string(index=False))
        r = np.corrcoef(agg.kappa, agg.mean_k)[0, 1]
        r2 = np.corrcoef(agg.kappa, agg.err_p.abs())[0, 1]
        print(f"\n[phase2] corr(kappa, mean k) = {r:.3f};  "
              f"corr(kappa, |error in E[p]|) = {r2:.3f}")
        print(f"[phase2] done in {time.time() - t0:.1f}s")
        return

    traj, summary, posteriors = [], [], {}
    for ci, cond in enumerate(conds):
        ref = prior_reference(cond["prior"], cond["k_fixed"])
        for a in range(args.n_agents):
            seed = MASTER_SEED + 1000 * ci + a
            spec = AgentSpec(prior=cond["prior"], policy=build_policy(cond),
                             label=cond["condition"], n_grid=N_GRID)
            df = run_agent(world, spec, args.n_episodes, seed,
                           log_every=args.log_every)
            df["condition"] = cond["condition"]
            df["prior_name"] = cond["prior_name"]
            df["policy_kind"] = cond["policy_kind"]
            df["agent"] = a
            df["dH_p"] = ref["H_p"] - df["H_p"]           # nats of structural gain
            df["dH_q"] = ref["H_q"] - df["H_q"]
            df["dH_lambda"] = ref["H_lambda"] - df["H_lambda"]  # observable gain
            traj.append(df)
            last = df.iloc[-1].to_dict()
            last.update(prior_H_p=ref["H_p"], prior_H_lambda=ref["H_lambda"],
                        prior_H_q=ref["H_q"], seed=seed)
            summary.append(last)
            if a == 0:
                posteriors[cond["condition"]] = df.attrs["grid"].posterior
        print(f"[phase2] {cond['condition']:>28}  "
              f"k0={cond['k_fixed']}  ({time.time() - t0:.0f}s)")

    traj = pd.concat(traj, ignore_index=True)
    traj.to_csv(RESULTS / "phase2_trajectories.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(RESULTS / "phase2_summary.csv", index=False)
    np.savez_compressed(RESULTS / "phase2_posteriors.npz", **posteriors)

    agg = (summ.groupby("condition")
           .agg(mean_k=("mean_k_sofar", "mean"), var_k=("var_k_sofar", "mean"),
                mean_p=("mean_p", "mean"), sd_mean_p=("mean_p", "std"),
                mean_q=("mean_q", "mean"), err_p=("err_p", "mean"),
                err_q=("err_q", "mean"), sd_p=("sd_p", "mean"),
                corr_qp=("corr_qp", "mean"),
                dH_p=("dH_p", "mean"), dH_q=("dH_q", "mean"),
                dH_lambda=("dH_lambda", "mean"),
                harm_rate=("harm_rate_sofar", "mean"),
                fim_log10_det=("fim_log10_det", "mean"),
                frac_singular=("fim_singular", "mean"))
           .reset_index())
    print("\n" + agg.round(3).to_string(index=False))
    agg.to_csv(RESULTS / "phase2_condition_means.csv", index=False)

    # ------------------------------------------- controlled design sweep over k
    sweep = k_sweep(world, max(3, args.n_agents // 2), args.n_episodes)
    sweep.to_csv(RESULTS / "phase2_k_sweep.csv", index=False)
    sw = (sweep.groupby("k_fixed")
          .agg(corr_qp=("corr_qp", "mean"), sd_p=("sd_p", "mean"),
               H_p=("H_p", "mean"), H_lambda=("H_lambda", "mean"),
               mean_p=("mean_p", "mean"), mean_q=("mean_q", "mean"),
               err_p=("err_p", "mean"), harm_rate=("harm_rate_sofar", "mean"))
          .reset_index())
    print("\n[phase2] controlled k-sweep (prior held fixed):")
    print(sw.round(3).to_string(index=False))

    # ------------------------------------------------------------- H verdicts
    rows = []
    fx = summ[summ.policy_kind == "fixed"]
    var = summ[summ.policy_kind == "variable"]
    sev_fx = fx[fx.prior_name == "severe_doubt"]

    sw_pos = sw[sw.k_fixed >= 1]
    r_ck = np.corrcoef(sw_pos.k_fixed, sw_pos.corr_qp)[0, 1]
    r_hk = np.corrcoef(sw_pos.k_fixed, sw_pos.H_p)[0, 1]
    # Information actually acquired about efficacy, as a function of the design.
    ref_sweep = prior_reference(PRIORS["accurate"], 0)
    sw["dH_p"] = ref_sweep["H_p"] - sw["H_p"]
    k_star = int(sw.loc[sw.dH_p.idxmax(), "k_fixed"])
    dH_p_star = float(sw.dH_p.max())
    dH_p_tail = float(sw.loc[sw.k_fixed == sw.k_fixed.max(), "dH_p"].iloc[0])
    sw.to_csv(RESULTS / "phase2_k_sweep_means.csv", index=False)

    # H1: structural entropy stays high while observable entropy collapses
    rows.append(dict(
        hypothesis="H1",
        statement="Repetitive agents keep structural uncertainty about p while "
                  "uncertainty about the observable composite lambda_k collapses",
        metric="dH_lambda - dH_p (nats), severe-doubt fixed-k agents",
        value=float((sev_fx.dH_lambda - sev_fx.dH_p).mean()),
        comparison=float((var.dH_lambda - var.dH_p).mean()),
        supported=bool((sev_fx.dH_lambda - sev_fx.dH_p).mean()
                       > (var.dH_lambda - var.dH_p).mean())))
    # H1b: entropy about p as a function of how much safety behaviour is performed,
    # with the prior held constant so only the design varies
    rows.append(dict(hypothesis="H1b",
                     statement="With the prior held fixed, more repetitions leave "
                               "higher residual posterior entropy about p",
                     metric="corr(k, H[p]) over the controlled k-sweep (k>=1)",
                     value=float(r_hk), comparison=0.0, supported=bool(r_hk > 0)))
    # H2: doubt is not corrected -- terminal bias in E[p]
    rows.append(dict(hypothesis="H2",
                     statement="Initial doubt survives thousands of episodes under "
                               "a self-generated repetitive design",
                     metric="mean |E[p]-p_true| severe-doubt fixed-k vs variable-k",
                     value=float(sev_fx.err_p.abs().mean()),
                     comparison=float(var.err_p.abs().mean()),
                     supported=bool(sev_fx.err_p.abs().mean() > 0.20
                                    and sev_fx.err_p.abs().mean()
                                    > var.err_p.abs().mean())))
    # H3: positive q-p posterior coupling, growing with k
    rows.append(dict(hypothesis="H3",
                     statement="Wherever the design is informative at all, q and p "
                               "are POSITIVELY coupled in the posterior, as the "
                               "ridge geometry predicts; the coupling peaks at "
                               "small k and decays as the design stops yielding "
                               "information (prior held fixed)",
                     metric="max_k corr(q,p) in the sweep; corr(k, corr_qp) for k>=1",
                     value=float(sw.corr_qp.max()), comparison=float(r_ck),
                     supported=bool(sw.corr_qp.max() > 0.5
                                    and (sw.corr_qp >= -1e-9).all()
                                    and r_ck < 0)))
    rows.append(dict(hypothesis="H1c",
                     statement="Information about one's own efficacy is maximised at "
                               "an intermediate, SMALL number of repetitions and "
                               "vanishes both at k=0 (never act) and at large k "
                               "(always act): an epistemically optimal dose exists",
                     metric=f"argmax_k dH[p] = {k_star}; dH[p] at that k vs at k=K_MAX",
                     value=dH_p_star, comparison=dH_p_tail,
                     supported=bool(0 < k_star < sw.k_fixed.max()
                                    and dH_p_star > 3 * max(dH_p_tail, 1e-6))))
    rows.append(dict(hypothesis="H3c",
                     statement="Response prevention (k=0) leaves q and p exactly "
                               "uncoupled because it never probes p",
                     metric="corr(q,p) at k=0 in the controlled sweep",
                     value=float(sw.loc[sw.k_fixed == 0, "corr_qp"].mean()),
                     comparison=0.0,
                     supported=bool(abs(sw.loc[sw.k_fixed == 0,
                                               "corr_qp"].mean()) < 1e-6)))
    # H3b: the design itself is singular
    rows.append(dict(hypothesis="H3b",
                     statement="Fixed-k designs are singular; diverse designs are not",
                     metric="fraction of fixed-k agents with det(I)=0",
                     value=float(fx.fim_singular.mean()),
                     comparison=float(var.fim_singular.mean()),
                     supported=bool(fx.fim_singular.mean() > 0.99
                                    and var.fim_singular.mean() < 0.01)))
    # Safety-vs-knowledge trade-off
    rows.append(dict(hypothesis="TRADEOFF",
                     statement="The identifiable design costs harm: variable-k agents "
                               "learn p but experience more harm",
                     metric="harm rate, severe-doubt fixed-k vs variable-k",
                     value=float(sev_fx.harm_rate_sofar.mean()),
                     comparison=float(var.harm_rate_sofar.mean()),
                     supported=bool(var.harm_rate_sofar.mean()
                                    > sev_fx.harm_rate_sofar.mean())))
    hyp = pd.DataFrame(rows)
    hyp.to_csv(RESULTS / "phase2_hypotheses.csv", index=False)
    print("\n" + hyp[["hypothesis", "metric", "value", "comparison",
                      "supported"]].round(4).to_string(index=False))
    print(f"\n[phase2] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
