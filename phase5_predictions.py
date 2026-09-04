#!/usr/bin/env python3
r"""Phase 5 -- falsifiable predictions for a human experiment.

The theory's distinctive empirical claim is not "compulsive people repeat more".
It is that **holding the amount of safety behaviour constant, its variability
determines how much a person can learn about their own efficacy**.

Prediction 1 (design diversity at matched exposure)
    Across policies matched on ``E[k]`` but differing in ``Var(k)``, posterior
    precision about ``p`` increases with ``Var(k)``.  Two people who wash exactly
    as often, on average, end up with different knowledge of whether washing
    works, purely because one of them varies.

Prediction 2 (incremental explanatory value)
    In a simulated population, how *accurate* a person's belief about their own
    efficacy becomes is predicted not only by where they started -- the
    belief-bias account -- but additionally by how much information their own
    policy generated.  We deliberately target accuracy rather than severity: in
    this model decision noise raises both the variability and the mean of the
    action count, so a severity regression on ``Var(k)`` is partly mechanical.
    The non-circular version of the claim is the stratified one: *holding
    severity fixed*, individuals whose behaviour varies more end up with more
    precise beliefs about their own efficacy.

Prediction 3 (dissociation)
    Reassurance-like probes reduce state uncertainty without reducing structural
    uncertainty, so self-reported momentary confidence and causal knowledge should
    dissociate.  Quantified in Phase 3 and carried through here as an effect size.

Outputs
-------
results/phase5_variance_matched.csv   matched-mean, varying-variance designs
results/phase5_population.csv         simulated individual-differences population
results/phase5_regression.csv         nested model comparison
results/phase5_predictions.csv        the headline numbers for Figure 7
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from epistemic_trap.config import COST, K_MAX, MASTER_SEED, RESULTS, WORLD
from epistemic_trap.policies import GreedyPolicy, RandomizedPolicy
from epistemic_trap.simulate import AgentSpec, World, run_agent

KS = np.arange(K_MAX + 1)


# ------------------------------------------------ prediction 1: matched exposure
def matched_mean_designs(mean_k=3):
    """Policies with identical ``E[k]`` and increasing ``Var(k)``."""
    designs = {}
    designs[f"fixed {{{mean_k}}}"] = {mean_k: 1.0}
    for spread in range(1, mean_k + 1):
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
    for name, (probs, m, v) in matched_mean_designs(mean_k).items():
        for a in range(n_agents):
            spec = AgentSpec(prior=dict(a_q=3.0, b_q=7.0, a_p=1.5, b_p=13.5),
                             policy=RandomizedPolicy(k_values=KS, probs=probs),
                             label=name)
            df = run_agent(world, spec, n_episodes,
                           seed=MASTER_SEED + 71_000 + 137 * hash(name) % 9973 + a,
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


# --------------------------------------- prediction 2: individual differences
def simulate_population(world, n_individuals, n_episodes, rng):
    """A population with heterogeneous doubt, cost sensitivity and decision noise.

    Decision noise matters here for a substantive reason: it is the only source of
    behavioural variability in an otherwise deterministic harm-minimiser, so it is
    the model's proxy for how much unintended experimentation a person performs.
    """
    rows = []
    for i in range(n_individuals):
        mean_p_prior = float(rng.uniform(0.05, 0.95))
        harm_cost = float(np.exp(rng.uniform(np.log(4), np.log(60))))
        beta = float(np.exp(rng.uniform(np.log(1.0), np.log(60.0))))
        conc = 15.0
        prior = dict(a_q=3.0, b_q=7.0,
                     a_p=conc * mean_p_prior, b_p=conc * (1 - mean_p_prior))
        cost = dict(COST, harm=harm_cost)
        spec = AgentSpec(prior=prior,
                         policy=GreedyPolicy(k_values=KS, beta=beta, cost=cost),
                         label=f"ind{i}")
        df = run_agent(world, spec, n_episodes, seed=MASTER_SEED + 81_000 + i,
                       log_every=n_episodes)
        last = df.iloc[-1]
        ks = df.attrs["k_history"]
        tail = ks[len(ks) // 2:]
        grid = df.attrs["grid"]
        emp = np.array([(ks == k).mean() for k in KS])
        eig_policy = float(sum(w * grid.eig_harm(int(k))
                               for k, w in zip(KS, emp) if w > 0))
        rows.append(dict(
            individual=i, mean_p_prior=mean_p_prior, harm_cost=harm_cost, beta=beta,
            severity=float(tail.mean()),            # observable: how much they repeat
            var_k=float(tail.var()),                # observable: how much they vary
            mean_p=float(last.mean_p),              # latent belief (model-derived)
            sd_p=float(last.sd_p), H_p=float(last.H_p),
            corr_qp=float(last.corr_qp),
            eig_policy=eig_policy,                  # information the policy generates
            abs_err_p=float(abs(last.err_p)),
            logdens_true=float(last.logdens_true),
            harm_rate=float(last.harm_rate_sofar)))
    return pd.DataFrame(rows)


def _fit(pop, cols, target):
    X = np.column_stack([np.ones(len(pop))] + [pop[c].values for c in cols])
    y = pop[target].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    n, k = len(y), X.shape[1]
    return dict(predictors="+".join(cols), target=target, r2=r2,
                adj_r2=1 - (1 - r2) * (n - 1) / (n - k),
                n_params=k, ss_res=ss_res)


def nested_regression(pop, target="abs_err_p",
                      ladder=(("mean_p_prior",),
                              ("mean_p_prior", "severity"),
                              ("mean_p_prior", "severity", "var_k"),
                              ("mean_p_prior", "severity", "var_k", "eig_policy"))):
    """Nested models for how accurate the agent's efficacy belief becomes.

    The base model is the belief-bias account in its strongest form: where the
    person started.  Each further step adds something the present account says
    should matter -- how much they act, how much they vary, and how much
    information their policy actually generates.
    """
    from scipy import stats

    models = [_fit(pop, list(c), target) for c in ladder]
    out = pd.DataFrame(models)
    n = len(pop)
    f_stats, p_vals = [np.nan], [np.nan]
    for a, b in zip(models[:-1], models[1:]):
        df1 = b["n_params"] - a["n_params"]
        df2 = n - b["n_params"]
        f = ((a["ss_res"] - b["ss_res"]) / df1) / (b["ss_res"] / df2)
        f_stats.append(f)
        p_vals.append(float(1 - stats.f.cdf(f, df1, df2)))
    out["F_vs_previous"] = f_stats
    out["p_vs_previous"] = p_vals
    return out


def severity_stratified(pop, n_strata=3):
    """Within strata matched on severity, does variability predict precision?

    This is the non-circular form of Prediction 2.  Decision noise inflates both
    the mean and the variance of the action count, so an unstratified regression
    of severity on ``Var(k)`` would be partly mechanical; conditioning on
    severity removes that path.
    """
    pop = pop.copy()
    pop["stratum"] = pd.qcut(pop.severity, n_strata,
                             labels=[f"s{i+1}" for i in range(n_strata)])
    rows = []
    for name, g in pop.groupby("stratum", observed=True):
        rows.append(dict(
            stratum=str(name), n=len(g), mean_severity=float(g.severity.mean()),
            corr_vark_sdp=float(np.corrcoef(g.var_k, g.sd_p)[0, 1]),
            corr_vark_abserr=float(np.corrcoef(g.var_k, g.abs_err_p)[0, 1]),
            corr_eig_sdp=float(np.corrcoef(g.eig_policy, g.sd_p)[0, 1])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=30)
    ap.add_argument("--n-episodes", type=int, default=1200)
    ap.add_argument("--n-individuals", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reanalyse", action="store_true",
                    help="recompute statistics from saved simulation output")
    args = ap.parse_args()
    if args.quick:
        args.n_agents, args.n_episodes, args.n_individuals = 5, 300, 40

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

    if args.reanalyse:
        pop = pd.read_csv(RESULTS / "phase5_population.csv")
    else:
        rng = np.random.default_rng(MASTER_SEED + 5)
        pop = simulate_population(world, args.n_individuals, args.n_episodes, rng)
        pop.to_csv(RESULTS / "phase5_population.csv", index=False)
    reg = nested_regression(pop)
    reg.to_csv(RESULTS / "phase5_regression.csv", index=False)
    print(f"\n[phase5] Prediction 2 -- accuracy of the efficacy belief (n={len(pop)}):")
    print(reg.round(4).to_string(index=False))

    sev_reg = nested_regression(
        pop, target="severity",
        ladder=(("mean_p_prior",), ("mean_p_prior", "harm_cost"),
                ("mean_p_prior", "harm_cost", "sd_p"),
                ("mean_p_prior", "harm_cost", "sd_p", "eig_policy")))
    sev_reg.to_csv(RESULTS / "phase5_severity_regression.csv", index=False)
    print("\n[phase5] (secondary) severity regression:")
    print(sev_reg.round(4).to_string(index=False))

    strat = severity_stratified(pop)
    strat.to_csv(RESULTS / "phase5_stratified.csv", index=False)
    print("\n[phase5] Prediction 2 (non-circular): within severity strata,")
    print(strat.round(3).to_string(index=False))

    r_var = np.corrcoef(vm.var_k, vm.sd_p)[0, 1]
    r_pop = np.corrcoef(pop.var_k, pop.sd_p)[0, 1]
    preds = pd.DataFrame([
        dict(prediction="P1", statement="At matched mean exposure, posterior "
                                        "precision about efficacy rises with the "
                                        "variance of the exposure",
             metric="corr(Var(k), SD[p|D]) across matched designs",
             value=float(r_var), supported=bool(r_var < 0)),
        dict(prediction="P1b", statement="The matched fixed-k design is the only "
                                         "singular one",
             metric="fraction singular: fixed vs varied designs",
             value=float(vm[vm.var_k == 0].singular.mean()
                         - vm[vm.var_k > 0].singular.mean()),
             supported=bool(vm[vm.var_k == 0].singular.mean() > 0.99
                            and vm[vm.var_k > 0].singular.mean() < 0.01)),
        dict(prediction="P2", statement="Policy-induced information explains how "
                                        "accurate a person's efficacy belief "
                                        "becomes, beyond where they started",
             metric="delta adj-R2 from adding Var(k) and policy EIG "
                    "(target: |E[p]-p*|)",
             value=float(reg.adj_r2.iloc[-1] - reg.adj_r2.iloc[0]),
             supported=bool(reg.p_vs_previous.iloc[-1] < 0.05
                            and reg.adj_r2.iloc[-1] > reg.adj_r2.iloc[0])),
        dict(prediction="P2b", statement="Non-circular form: WITHIN strata matched "
                                         "on severity, individuals whose behaviour "
                                         "varies more hold more precise beliefs "
                                         "about their own efficacy",
             metric="corr(Var(k), SD[p|D]) within every severity stratum",
             value=float(strat.corr_vark_sdp.max()),
             supported=bool((strat.corr_vark_sdp < 0).all())),
        dict(prediction="P2c", statement="Unstratified, variability also tracks "
                                         "precision, but this is partly mechanical",
             metric="corr(Var(k), SD[p|D]) in the whole population",
             value=float(r_pop), supported=bool(r_pop < 0)),
    ])
    preds.to_csv(RESULTS / "phase5_predictions.csv", index=False)
    print("\n" + preds[["prediction", "metric", "value", "supported"]]
          .round(4).to_string(index=False))
    print(f"\n[phase5] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
