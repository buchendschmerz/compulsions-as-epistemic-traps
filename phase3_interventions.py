#!/usr/bin/env python3
r"""Phase 3 -- interventions: what actually restores identifiability?

Every agent is identical up to the intervention: the same severe-doubt prior, the
same environment, the same exact-Bayesian learner, and the same *pre-treatment*
block of free choice, during which it settles into a repetitive safety policy.
Only the treatment block differs.

Conditions
----------
``free_choice``     continue choosing ``k`` to minimise expected harm (control).
``locked_k``        the pre-treatment ``k`` is held fixed throughout: the pure
                    no-diversification control, and the only condition that keeps
                    the realised design singular.
``reassurance``     free choice, plus a check *after* the preventive actions --
                    the formal analogue of reassurance seeking: it resolves "am I
                    safe now?" and re-probes the same composite ``lambda_k`` the
                    harm outcome already reports.
``baseline_check``  free choice, plus a check *before* the preventive actions --
                    same behavioural cost, but a genuine probe of the baseline
                    hazard ``q``.  A novel, clinically actionable contrast.
``erp_k0``          pure response prevention: ``k = 0`` throughout treatment.
``graded_erp``      the number of repetitions is stepped down through
                    ``k_pre, ..., 1, 0`` and then cycled, so the treatment block
                    contains several distinct experimental conditions.
``variable_k``      ``k`` drawn uniformly from ``{0, 1, 2, 3}`` -- maximal design
                    diversity at a comparable mean action rate.
``erp_naive``       response prevention with **no** pre-treatment block, which
                    isolates the fact that ``k = 0`` on its own carries exactly
                    zero information about one's own efficacy.

A finding worth stating up front: response prevention applied *after* a period of
repetitive behaviour does inform the agent about ``p``, but only *indirectly*.
The compulsive block pins down the composite ``lambda_{k_pre}``; the
response-prevention block pins down ``q``; together they are a two-point design
and ``p`` follows.  ERP is therefore not a probe of efficacy, it is the second
half of a contrast -- and it is worthless as a first half, which ``erp_naive``
demonstrates.

Each run has three blocks: pre-treatment free choice, treatment, and a
free-choice follow-up that tests whether the gains persist.

Outputs
-------
results/phase3_trajectories.csv.gz per-episode logs for every condition (gzip)
results/phase3_summary.csv        end-of-block summaries
results/phase3_blocks.csv         block means per condition
results/phase3_hypotheses.csv     numeric verdicts on H4 and H5
"""

from __future__ import annotations

import argparse
import gzip
import io
import time

import numpy as np
import pandas as pd

from epistemic_trap.config import K_MAX, MASTER_SEED, PRIORS, RESULTS, WORLD
from epistemic_trap.grid import ParameterGrid
from epistemic_trap.policies import (FixedKPolicy, GreedyPolicy, RandomizedPolicy,
                                     ScheduledPolicy)
from epistemic_trap.simulate import AgentSpec, World, run_agent

KS = np.arange(K_MAX + 1)
BETA_DETERMINISTIC = 60.0
PRIOR = PRIORS["severe_doubt"]

CONDITIONS = ["free_choice", "locked_k", "reassurance", "baseline_check",
              "erp_k0", "graded_erp", "variable_k", "erp_naive"]

NO_PRETREATMENT = {"erp_naive"}  # conditions that start from the naive prior


def write_compressed_csv(frame: pd.DataFrame, path) -> None:
    """Write a deterministic gzip CSV so the raw trajectory file stays small."""
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False)


def greedy():
    return GreedyPolicy(k_values=KS, beta=BETA_DETERMINISTIC)


def treatment_spec(condition, k_pre):
    """Policy and check mode used during the treatment block."""
    if condition == "free_choice":
        return greedy(), None
    if condition == "locked_k":
        return FixedKPolicy(k_values=KS, k=k_pre), None
    if condition == "erp_naive":
        return FixedKPolicy(k_values=KS, k=0), None
    if condition == "reassurance":
        return greedy(), "reassurance"
    if condition == "baseline_check":
        return greedy(), "baseline"
    if condition == "erp_k0":
        return FixedKPolicy(k_values=KS, k=0), None
    if condition == "graded_erp":
        ladder = list(range(max(k_pre, 1), -1, -1))  # k_pre, ..., 1, 0
        return ScheduledPolicy(k_values=KS,
                               schedule=lambda t, L=ladder: L[t % len(L)]), None
    if condition == "variable_k":
        probs = np.zeros(len(KS))
        probs[:4] = 1.0
        return RandomizedPolicy(k_values=KS, probs=probs), None
    raise ValueError(condition)


def run_protocol(world, condition, seed, n_pre, n_tx, n_post):
    """Pre-treatment free choice, then the intervention, then free-choice follow-up."""
    rng = np.random.default_rng(seed)
    if condition in NO_PRETREATMENT:
        # No compulsive history at all: the whole run is response prevention, so
        # the p-marginal has nothing but k = 0 evidence to work with.
        spec = AgentSpec(prior=PRIOR, policy=FixedKPolicy(k_values=KS, k=0),
                         label=condition)
    else:
        spec = AgentSpec(prior=PRIOR, policy=greedy(), label=condition)

    pre = run_agent(world, spec, n_pre, seed=None, rng=rng, log_every=1)
    pre["block"] = "pre"
    k_pre = int(np.bincount(pre.attrs["k_history"]).argmax())

    pol, check_mode = treatment_spec(condition, k_pre)
    spec_tx = AgentSpec(prior=PRIOR, policy=pol, check_mode=check_mode, label=condition)
    tx = run_agent(world, spec_tx, n_tx, seed=None, rng=pre.attrs["rng"],
                   grid=pre.attrs["grid"], t_offset=0,
                   k_history=pre.attrs["k_history"],
                   harm_history=pre.attrs["harm_history"], log_every=1)
    tx["block"] = "tx"
    tx["episode"] = np.arange(n_pre + 1, n_pre + n_tx + 1)

    spec_post = AgentSpec(prior=PRIOR, policy=greedy(), label=condition)
    post = run_agent(world, spec_post, n_post, seed=None, rng=tx.attrs["rng"],
                     grid=tx.attrs["grid"], t_offset=0,
                     k_history=tx.attrs["k_history"],
                     harm_history=tx.attrs["harm_history"], log_every=1)
    post["block"] = "post"
    post["episode"] = np.arange(n_pre + n_tx + 1, n_pre + n_tx + n_post + 1)

    final_grid = post.attrs["grid"]
    for f in (pre, tx, post):
        f.attrs = {}          # pandas compares .attrs when concatenating
    df = pd.concat([pre, tx, post], ignore_index=True)
    df["condition"] = condition
    df["k_pre"] = k_pre
    df.attrs["grid"] = final_grid
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=25)
    ap.add_argument("--n-pre", type=int, default=400)
    ap.add_argument("--n-tx", type=int, default=600)
    ap.add_argument("--n-post", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_agents, args.n_pre, args.n_tx, args.n_post = 3, 60, 90, 60

    world = World(**WORLD)
    prior_grid = ParameterGrid.from_beta_prior(**PRIOR)
    prior_ref = prior_grid.summary(k_ref=0)
    t0 = time.time()

    frames, summaries = [], []
    for ci, cond in enumerate(CONDITIONS):
        for a in range(args.n_agents):
            df = run_protocol(world, cond, MASTER_SEED + 7000 * ci + a,
                              args.n_pre, args.n_tx, args.n_post)
            df["agent"] = a
            df["dH_p"] = prior_ref["H_p"] - df["H_p"]
            df["dH_q"] = prior_ref["H_q"] - df["H_q"]
            frames.append(df[["episode", "block", "condition", "agent", "k", "harm",
                              "mean_p", "mean_q", "sd_p", "sd_q", "corr_qp",
                              "H_p", "H_q", "H_joint", "dH_p", "dH_q",
                              "H_state_pre", "H_state_commit", "p_hazard_commit",
                              "err_p", "err_q", "fim_singular", "fim_log10_det",
                              "k_support", "eig_current_k", "k_pre",
                              "logdens_true", "mass_near_true"]])
            for blk, sub in df.groupby("block"):
                last = sub.iloc[-1]
                summaries.append(dict(
                    condition=cond, agent=a, block=blk,
                    mean_k=float(sub.k.mean()), var_k=float(sub.k.var()),
                    harm_rate=float(sub.harm.mean()),
                    mean_p=float(last.mean_p), mean_q=float(last.mean_q),
                    err_p=float(last.err_p), err_q=float(last.err_q),
                    sd_p=float(last.sd_p), corr_qp=float(last.corr_qp),
                    H_p=float(last.H_p), H_q=float(last.H_q),
                    dH_p=float(prior_ref["H_p"] - last.H_p),
                    dH_q=float(prior_ref["H_q"] - last.H_q),
                    z_p=float(abs(last.err_p) / max(last.sd_p, 1e-12)),
                    z_q=float(abs(last.err_q) / max(last.sd_q, 1e-12)),
                    d_H_state=float((sub.H_state_pre - sub.H_state_commit).mean()),
                    H_state_commit=float(sub.H_state_commit.mean()),
                    H_state_pre=float(sub.H_state_pre.mean()),
                    logdens_true=float(last.logdens_true),
                    mass_near_true=float(last.mass_near_true),
                    fim_singular=bool(last.fim_singular),
                    k_support=int(last.k_support)))
        print(f"[phase3] {cond:>15}  ({time.time() - t0:.0f}s)")

    traj = pd.concat(frames, ignore_index=True)
    write_compressed_csv(traj, RESULTS / "phase3_trajectories.csv.gz")
    summ = pd.DataFrame(summaries)
    summ.to_csv(RESULTS / "phase3_summary.csv", index=False)

    blocks = (summ.groupby(["condition", "block"]).mean(numeric_only=True)
              .reset_index().drop(columns=["agent"]))
    blocks.to_csv(RESULTS / "phase3_blocks.csv", index=False)
    order = {"pre": 0, "tx": 1, "post": 2}
    show = blocks.assign(o=blocks.block.map(order)).sort_values(["condition", "o"])
    print("\n" + show.drop(columns="o").round(3).to_string(index=False))

    # ------------------------------------------------------------- H verdicts
    tx = summ[summ.block == "tx"].set_index(["condition", "agent"])
    post = summ[summ.block == "post"].set_index(["condition", "agent"])
    pre = summ[summ.block == "pre"].set_index(["condition", "agent"])

    def g(frame, cond, col):
        return frame.xs(cond, level="condition")[col]

    def gain(cond, col):
        """Change in a statistic from the end of pre-treatment to the end of treatment."""
        return float(g(tx, cond, col).mean() - g(pre, cond, col).mean())

    rows = []
    rows.append(dict(
        hypothesis="H4a",
        statement="Response prevention raises the agent's own predicted risk during "
                  "treatment: the short-term cost of the intervention is real, and "
                  "the model produces it without any fear-extinction machinery",
        metric="mean P(hazard at commitment) during tx: erp_k0 vs free_choice",
        value=float(traj[(traj.condition == "erp_k0") & (traj.block == "tx")]
                    .p_hazard_commit.mean()),
        comparison=float(traj[(traj.condition == "free_choice") & (traj.block == "tx")]
                         .p_hazard_commit.mean()),
        supported=None))
    rows.append(dict(
        hypothesis="H4b",
        statement="Response prevention is NOT a probe of one's own efficacy: with no "
                  "compulsive history to contrast against, k=0 moves the p-marginal "
                  "by exactly nothing, however long it is applied",
        metric="|E[p] at end of treatment - prior E[p]|: erp_naive vs erp_k0",
        value=float(abs(g(tx, "erp_naive", "mean_p").mean() - prior_ref["mean_p"])),
        comparison=float(abs(g(tx, "erp_k0", "mean_p").mean() - prior_ref["mean_p"])),
        supported=None))
    rows.append(dict(
        hypothesis="H4b2",
        statement="ERP after a compulsive block works INDIRECTLY: the earlier "
                  "fixed-k data fixes the composite lambda_k, k=0 fixes q, and the "
                  "two together determine p. ERP is the second half of a contrast",
        metric="entropy of p removed during tx: erp_naive vs erp_k0",
        value=gain("erp_naive", "dH_p"),
        comparison=gain("erp_k0", "dH_p"),
        supported=None))
    rows.append(dict(
        hypothesis="H4c",
        statement="What restores knowledge is the CONTRAST between distinct values "
                  "of k. Locking k at its pre-treatment value -- identical action "
                  "count, zero diversity -- leaves the agent stuck",
        metric="gain in log posterior density at the truth: best diversified "
               "condition vs locked_k",
        value=max(gain(c, "logdens_true")
                  for c in ("erp_k0", "graded_erp", "variable_k")),
        comparison=gain("locked_k", "logdens_true"),
        supported=None))
    rows.append(dict(
        hypothesis="H4d",
        statement="Confidence is not knowledge: the untreated agent ends up "
                  "confidently wrong, its posterior placing the true baseline "
                  "hazard many posterior standard deviations away",
        metric="posterior z-score of the true q at follow-up: free_choice vs "
               "variable_k",
        value=float(g(post, "free_choice", "z_q").mean()),
        comparison=float(g(post, "variable_k", "z_q").mean()),
        supported=None))
    rows.append(dict(
        hypothesis="H5a",
        statement="Reassurance does what it is for: it removes STATE uncertainty "
                  "('am I safe right now?') at the moment of commitment",
        metric="state entropy removed by the check during tx: reassurance vs "
               "free_choice",
        value=float(g(tx, "reassurance", "d_H_state").mean()),
        comparison=float(g(tx, "free_choice", "d_H_state").mean()),
        supported=None))
    rows.append(dict(
        hypothesis="H5b",
        statement="Reassurance leaves STRUCTURAL knowledge essentially where it "
                  "found it, unlike diversified exposure",
        metric="gain in log posterior density at the truth: reassurance vs variable_k",
        value=gain("reassurance", "logdens_true"),
        comparison=gain("variable_k", "logdens_true"),
        supported=None))
    rows.append(dict(
        hypothesis="H5c",
        statement="Checking BEFORE acting is diagnostic where checking AFTER acting "
                  "is not: the same behaviour, moved to a different epistemic "
                  "position, becomes informative",
        metric="gain in log posterior density at the truth: baseline_check vs "
               "reassurance",
        value=gain("baseline_check", "logdens_true"),
        comparison=gain("reassurance", "logdens_true"),
        supported=None))
    rows.append(dict(
        hypothesis="DURABILITY",
        statement="Knowledge acquired during treatment persists into free-choice "
                  "follow-up",
        metric="|E[p]-p_true| at follow-up: variable_k vs free_choice",
        value=float(g(post, "variable_k", "err_p").abs().mean()),
        comparison=float(g(post, "free_choice", "err_p").abs().mean()),
        supported=None))
    rows.append(dict(
        hypothesis="UNDERPROTECTION",
        statement="The trap does not only produce too much safety behaviour. The "
                  "untreated agent settles into a badly underestimated baseline "
                  "hazard and therefore acts LESS than the truth warrants; every "
                  "condition that corrects q ends up acting more",
        metric="mean k at follow-up: best diversified condition vs free_choice",
        value=float(max(g(post, c, "mean_k").mean()
                        for c in ("erp_k0", "graded_erp", "variable_k"))),
        comparison=float(g(post, "free_choice", "mean_k").mean()),
        supported=None))
    rows.append(dict(
        hypothesis="CONFIDENTLY_WRONG",
        statement="The untreated agent is not merely uncertain about the baseline "
                  "hazard, it is confidently wrong about it",
        metric="posterior z-score of the true q at follow-up: free_choice vs "
               "variable_k",
        value=float(g(post, "free_choice", "z_q").mean()),
        comparison=float(g(post, "variable_k", "z_q").mean()),
        supported=None))

    hyp = pd.DataFrame(rows)
    hyp["supported"] = [
        rows[0]["value"] > rows[0]["comparison"],          # H4a
        rows[1]["value"] < 1e-9 < rows[1]["comparison"],   # H4b
        rows[2]["value"] < 1e-9 < rows[2]["comparison"],   # H4b2
        rows[3]["value"] > rows[3]["comparison"],          # H4c
        rows[4]["value"] > rows[4]["comparison"],          # H4d
        rows[5]["value"] > rows[5]["comparison"],          # H5a
        rows[6]["value"] < rows[6]["comparison"],          # H5b
        rows[7]["value"] > rows[7]["comparison"],          # H5c
        rows[8]["value"] < rows[8]["comparison"],          # DURABILITY
        rows[9]["value"] > rows[9]["comparison"],          # UNDERPROTECTION
        rows[10]["value"] > 3 * rows[10]["comparison"],    # CONFIDENTLY_WRONG
    ]
    hyp.to_csv(RESULTS / "phase3_hypotheses.csv", index=False)
    print("\n" + hyp[["hypothesis", "metric", "value", "comparison",
                      "supported"]].round(4).to_string(index=False))
    print(f"\n[phase3] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
