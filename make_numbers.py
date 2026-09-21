#!/usr/bin/env python3
"""Emit ``paper/numbers.tex``: every number quoted in the manuscript, as a macro.

Writing results into the text by hand is how a paper drifts away from its own
data.  Each macro here is computed from the saved result files, so recompiling the
manuscript after re-running a phase updates the prose automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epistemic_trap.config import RESULTS, ROOT, WORLD

OUT = ROOT / "paper" / "numbers.tex"
MACROS = {}


def put(name, value, fmt="{:.2f}"):
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def have(name):
    return (RESULTS / name).exists()


# ------------------------------------------------------------------ world
put("worldQ", WORLD["q"], "{:.2f}")
put("worldP", WORLD["p"], "{:.2f}")

# ------------------------------------------------------------------ phase 1
if have("phase1_policy_identifiability.csv"):
    p1 = pd.read_csv(RESULTS / "phase1_policy_identifiability.csv")
    p1 = p1.set_index("policy")
    put("detBestTwoPoint", p1.loc["k in {0,1}", "log10_det"], "{:.2f}")
    put("detGraded", p1.loc["graded ERP (k in 0..6)", "log10_det"], "{:.2f}")
    put("detJittered", p1.loc["compulsive, jittered (k in {5,6,7})", "log10_det"],
        "{:.2f}")
    put("seBestTwoPoint", p1.loc["k in {0,1}", "se_p"], "{:.3f}")
    put("seJittered", p1.loc["compulsive, jittered (k in {5,6,7})", "se_p"], "{:.2f}")
    put("nSingularPolicies", int(p1.singular.sum()), "{:d}")

# ------------------------------------------------------------------ phase 2
if have("phase2_condition_means.csv"):
    c = pd.read_csv(RESULTS / "phase2_condition_means.csv").set_index("condition")
    put("bayesEpisodes", 2000, "{:d}")
    for key, cond in [("Fix", "severe_doubt/fixed-k"),
                      ("Var", "severe_doubt/variable-k"),
                      ("Free", "severe_doubt/closed-loop"),
                      ("Acc", "accurate/fixed-k")]:
        if cond in c.index:
            put(f"bayes{key}MeanP", c.loc[cond, "mean_p"], "{:.2f}")
            put(f"bayes{key}MeanQ", c.loc[cond, "mean_q"], "{:.2f}")
            put(f"bayes{key}Corr", c.loc[cond, "corr_qp"], "{:.2f}")
            put(f"bayes{key}Harm", 100 * c.loc[cond, "harm_rate"], "{:.1f}")
            put(f"bayes{key}dHlam", c.loc[cond, "dH_lambda"], "{:.2f}")
            put(f"bayes{key}dHp", c.loc[cond, "dH_p"], "{:.2f}")
            put(f"bayes{key}MeanK", c.loc[cond, "mean_k"], "{:.2f}")

# The entropy of lambda_k saturates on a fixed [0,1] bin grid when lambda_k is
# tiny, so we also report the scale-free collapse in its posterior SD.
if have("phase2_summary.csv"):
    from epistemic_trap.grid import ParameterGrid
    from epistemic_trap.config import PRIORS
    s2 = pd.read_csv(RESULTS / "phase2_summary.csv")
    for key, cond, prior in [("Fix", "severe_doubt/fixed-k", "severe_doubt"),
                             ("Var", "severe_doubt/variable-k", "severe_doubt"),
                             ("Free", "severe_doubt/closed-loop", "severe_doubt")]:
        sub = s2[s2.condition == cond]
        if len(sub) == 0:
            continue
        k_ref = int(sub.k_ref.mode().iloc[0])
        g0 = ParameterGrid.from_beta_prior(**PRIORS[prior])
        sd0 = g0.sd_lambda(k_ref)
        sd1 = float(sub.sd_lambda.mean())
        put(f"bayes{key}SdLamRatio", sd0 / max(sd1, 1e-300), "{:.0f}")
        put(f"bayes{key}SdPRatio",
            float(np.sqrt(g0.cov()[1, 1]) / max(sub.sd_p.mean(), 1e-300)), "{:.2f}")
        put(f"bayes{key}Kref", k_ref, "{:d}")

if have("phase2_k_sweep_means.csv"):
    sw = pd.read_csv(RESULTS / "phase2_k_sweep_means.csv")
    put("sweepBestK", int(sw.loc[sw.dH_p.idxmax(), "k_fixed"]), "{:d}")
    put("sweepBestdHp", sw.dH_p.max(), "{:.2f}")
    put("sweepTaildHp", float(sw.dH_p.iloc[-1]), "{:.2f}")
    put("sweepMaxCorr", sw.corr_qp.max(), "{:.2f}")
    put("sweepCorrAtZero", sw.loc[sw.k_fixed == 0, "corr_qp"].iloc[0], "{:.3f}")
    put("sweepKmax", int(sw.k_fixed.max()), "{:d}")

if have("phase2_kappa_means.csv") and "persistence" in pd.read_csv(
        RESULTS / "phase2_kappa_means.csv").columns:
    kp = pd.read_csv(RESULTS / "phase2_kappa_means.csv")
    put("kappaMax", kp.kappa.max(), "{:.1f}")
    put("kappaZeroK", kp.loc[kp.kappa == 0, "mean_k"].iloc[0], "{:.2f}")
    put("kappaMaxK", kp.loc[kp.kappa == kp.kappa.max(), "mean_k"].iloc[0], "{:.2f}")
    put("kappaZeroErrP", abs(kp.loc[kp.kappa == 0, "err_p"].iloc[0]), "{:.2f}")
    put("kappaMaxErrP", abs(kp.loc[kp.kappa == kp.kappa.max(), "err_p"].iloc[0]),
        "{:.2f}")
    put("kappaCorrK", float(np.corrcoef(kp.kappa, kp.mean_k)[0, 1]), "{:.3f}")
    put("kappaCorrErr", float(np.corrcoef(kp.kappa, kp.err_p.abs())[0, 1]), "{:.3f}")
    put("kappaZeroEarlyK", kp.loc[kp.kappa == 0, "k_early"].iloc[0], "{:.1f}")
    put("kappaMaxEarlyK", kp.loc[kp.kappa == kp.kappa.max(), "k_early"].iloc[0],
        "{:.1f}")
    put("kappaZeroLateK", kp.loc[kp.kappa == 0, "k_late"].iloc[0], "{:.1f}")
    put("kappaMaxLateK", kp.loc[kp.kappa == kp.kappa.max(), "k_late"].iloc[0],
        "{:.1f}")

# ------------------------------------------------------------------ phase 3
if have("phase3_hypotheses.csv") and have("phase3_blocks.csv"):
    h = pd.read_csv(RESULTS / "phase3_hypotheses.csv").set_index("hypothesis")
    b = pd.read_csv(RESULTS / "phase3_blocks.csv")
    put("nHypSupported", int(h.supported.sum()), "{:d}")
    put("nHypTotal", len(h), "{:d}")
    put("erpRiskDuring", h.loc["H4a", "value"], "{:.3f}")
    put("freeRiskDuring", h.loc["H4a", "comparison"], "{:.3f}")
    put("erpNaiveShiftP", h.loc["H4b", "value"], "{:.3f}")
    put("erpAfterShiftP", h.loc["H4b", "comparison"], "{:.2f}")
    put("bestDiversifiedGain", h.loc["H4c", "value"], "{:.1f}")
    put("lockedGain", h.loc["H4c", "comparison"], "{:.1f}")
    put("zqFree", h.loc["H4d", "value"], "{:.1f}")
    put("zqVariable", h.loc["H4d", "comparison"], "{:.1f}")
    put("reassuranceState", h.loc["H5a", "value"], "{:.3f}")
    put("reassuranceGain", h.loc["H5b", "value"], "{:.1f}")
    put("variableGain", h.loc["H5b", "comparison"], "{:.1f}")
    put("baselineCheckGain", h.loc["H5c", "value"], "{:.1f}")
    put("durVariable", h.loc["DURABILITY", "value"], "{:.2f}")
    put("durFree", h.loc["DURABILITY", "comparison"], "{:.2f}")
    if "UNDERPROTECTION" in h.index:
        put("postKtreated", h.loc["UNDERPROTECTION", "value"], "{:.2f}")
        put("postKuntreated", h.loc["UNDERPROTECTION", "comparison"], "{:.2f}")
    post_b = b[b.block == "post"].set_index("condition")
    pre_b = b[b.block == "pre"].set_index("condition")
    put("preMeanK", pre_b.mean_k.mean(), "{:.2f}")
    put("preMeanP", pre_b.mean_p.mean(), "{:.2f}")
    put("preMeanQ", pre_b.mean_q.mean(), "{:.2f}")
    put("preZq", pre_b.z_q.mean(), "{:.1f}")
    put("postFreeQ", post_b.loc["free_choice", "mean_q"], "{:.2f}")
    put("postVarQ", post_b.loc["variable_k", "mean_q"], "{:.2f}")
    put("postErpK", post_b.loc["erp_k0", "mean_k"], "{:.2f}")
    put("txErpHarmRate", 100 * b[(b.block == "tx")].set_index(
        "condition").loc["erp_k0", "harm_rate"], "{:.0f}")
    tx = b[b.block == "tx"].set_index("condition")
    for key, cond in [("Erp", "erp_k0"), ("Var", "variable_k"),
                      ("Rea", "reassurance"), ("Base", "baseline_check"),
                      ("Grad", "graded_erp"), ("Lock", "locked_k")]:
        if cond in tx.index:
            put(f"tx{key}Harm", 100 * tx.loc[cond, "harm_rate"], "{:.1f}")
            put(f"tx{key}MeanP", tx.loc[cond, "mean_p"], "{:.2f}")

# ------------------------------------------------------------------ phase 4
if have("phase4_loop_areas.csv"):
    la = pd.read_csv(RESULTS / "phase4_loop_areas.csv").groupby("arm").mean(
        numeric_only=True)
    for arm in ("locked", "quota", "curious", "checker"):
        if arm in la.index:
            put(f"loop{arm.capitalize()}", la.loc[arm, "loop_area_k"], "{:.2f}")
            put(f"calib{arm.capitalize()}", la.loc[arm, "calib_err_up"], "{:.3f}")
            put(f"calib{arm.capitalize()}Down", la.loc[arm, "calib_err_down"],
                "{:.3f}")
if have("phase4_step_summary.csv"):
    st = pd.read_csv(RESULTS / "phase4_step_summary.csv").groupby("arm").mean(
        numeric_only=True)
    for arm in ("locked", "quota", "curious", "checker"):
        if arm in st.index:
            put(f"step{arm.capitalize()}K", st.loc[arm, "k_final"], "{:.2f}")
            put(f"step{arm.capitalize()}Q", st.loc[arm, "belief_q_final"], "{:.3f}")
            put(f"step{arm.capitalize()}Halve",
                st.loc[arm, "episodes_to_halve"], "{:.0f}")
            put(f"step{arm.capitalize()}P", st.loc[arm, "belief_p_final"], "{:.2f}")
            put(f"step{arm.capitalize()}Khaz", st.loc[arm, "k_hazard"], "{:.1f}")
if have("phase4_phase_diagram.csv"):
    pd4 = pd.read_csv(RESULTS / "phase4_phase_diagram.csv")
    lo = pd4[pd4.mean_p_prior <= pd4.mean_p_prior.min()]
    hi = pd4[pd4.mean_p_prior >= pd4.mean_p_prior.max()]
    put("phaseLowDoubtK", hi.mean_k.mean(), "{:.2f}")
    put("phaseHighDoubtK", lo.mean_k.mean(), "{:.2f}")
    put("phaseHighDoubtErr", lo.abs_err_p.mean(), "{:.2f}")
    put("phaseLowDoubtErr", hi.abs_err_p.mean(), "{:.2f}")
    # The phase diagram's real structure is a U in the realised action count:
    # too little safety behaviour and too much are both uninformative about p.
    kb = pd4.assign(kbin=pd4.mean_k.round()).groupby("kbin").agg(
        err=("abs_err_p", "mean"), harm=("harm_rate", "mean"), n=("abs_err_p", "size"))
    kb = kb[kb.n >= 5]
    put("phaseErrAtZeroK", kb.loc[0.0, "err"] if 0.0 in kb.index else np.nan, "{:.2f}")
    put("phaseBestK", float(kb.err.idxmin()), "{:.0f}")
    put("phaseBestErr", kb.err.min(), "{:.3f}")
    put("phaseWorstHighK", float(kb.index.max()), "{:.0f}")
    put("phaseErrHighK", kb.loc[kb.index.max(), "err"], "{:.2f}")
    put("phaseHarmAtZeroK",
        100 * (kb.loc[0.0, "harm"] if 0.0 in kb.index else np.nan), "{:.1f}")
    put("phaseHarmAtBestK", 100 * kb.loc[kb.err.idxmin(), "harm"], "{:.1f}")
    put("phaseCorrDoubtErr",
        float(np.corrcoef(pd4.mean_p_prior, pd4.abs_err_p)[0, 1]), "{:.2f}")
    hi = pd4[pd4.harm_cost >= 6]
    put("phaseCorrCostErrHigh",
        float(np.corrcoef(np.log(hi.harm_cost), hi.abs_err_p)[0, 1]), "{:.2f}")
    put("phaseCorrCostK",
        float(np.corrcoef(np.log(pd4.harm_cost), pd4.mean_k)[0, 1]), "{:.2f}")
if have("phase4_tau_diagram.csv"):
    td = pd.read_csv(RESULTS / "phase4_tau_diagram.csv")
    worst = td[td.mean_p_prior <= td.mean_p_prior.min()]
    put("tauZeroErr", worst[worst.tau == 0].abs_err_p.mean(), "{:.2f}")
    put("tauMaxErr", worst[worst.tau == worst.tau.max()].abs_err_p.mean(), "{:.2f}")
    put("tauMax", worst.tau.max(), "{:.0f}")
    # The doubtful third: the agents for whom the escape axis actually matters.
    lo = td[td.mean_p_prior <= td.mean_p_prior.quantile(0.34)]
    a0 = lo[lo.tau == 0]
    a1 = lo[lo.tau == lo.tau.max()]
    put("tauLoErrZero", a0.abs_err_p.mean(), "{:.2f}")
    put("tauLoErrMax", a1.abs_err_p.mean(), "{:.2f}")
    put("tauLoVarZero", a0.var_k.mean(), "{:.3f}")
    put("tauLoVarMax", a1.var_k.mean(), "{:.3f}")
    put("tauLoHarmZero", 100 * a0.harm_rate.mean(), "{:.1f}")
    put("tauLoHarmMax", 100 * a1.harm_rate.mean(), "{:.1f}")
    put("tauLoMeanPZero", a0.mean_p.mean(), "{:.2f}")
    put("tauLoMeanPMax", a1.mean_p.mean(), "{:.2f}")
    put("tauLoKZero", a0.mean_k.mean(), "{:.2f}")
    put("tauLoKMax", a1.mean_k.mean(), "{:.2f}")

if have("phase4_uncertainty_diagram.csv"):
    ud = pd.read_csv(RESULTS / "phase4_uncertainty_diagram.csv")
    agg = ud.groupby("concentration").agg(k=("mean_k", "mean"),
                                          e=("abs_err_p", "mean"),
                                          z=("z_p", "mean"))
    lo, hi = agg.index.min(), agg.index.max()
    put("uncLoConc", lo, "{:.0f}")
    put("uncHiConc", hi, "{:.0f}")
    put("uncLoK", agg.loc[lo, "k"], "{:.2f}")
    put("uncHiK", agg.loc[hi, "k"], "{:.2f}")
    put("uncLoErr", agg.loc[lo, "e"], "{:.2f}")
    put("uncHiErr", agg.loc[hi, "e"], "{:.2f}")
    put("uncLoZ", agg.loc[lo, "z"], "{:.1f}")
    put("uncHiZ", agg.loc[hi, "z"], "{:.1f}")
    put("uncCorrConcK",
        float(np.corrcoef(np.log(ud.concentration), ud.mean_k)[0, 1]), "{:.2f}")

# ------------------------------------------------------------------ phase 5
if have("phase5_variance_matched_means.csv"):
    vm = pd.read_csv(RESULTS / "phase5_variance_matched_means.csv")
    fixed = vm[vm.var_k == 0].iloc[0]
    best = vm.loc[vm.sd_p.idxmin()]
    put("vmFixedSdP", fixed.sd_p, "{:.3f}")
    put("vmBestSdP", best.sd_p, "{:.3f}")
    put("vmBestDesign", str(best.design))
    put("vmBestVarK", best.var_k, "{:.1f}")
    put("vmSdRatio", fixed.sd_p / best.sd_p, "{:.1f}")
    put("vmFixedHarm", 100 * fixed.harm_rate, "{:.1f}")
    put("vmBestHarm", 100 * best.harm_rate, "{:.1f}")
    if have("phase5_variance_matched.csv"):
        raw = pd.read_csv(RESULTS / "phase5_variance_matched.csv")
        put("vmCorr", float(np.corrcoef(raw.var_k, raw.sd_p)[0, 1]), "{:.2f}")

# ------------------------------------------------------------------ write
lines = ["% Auto-generated by make_numbers.py -- do not edit by hand.",
         "% Re-run `python3 make_numbers.py` after any phase is re-run.", ""]
for k, v in sorted(MACROS.items()):
    lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {len(MACROS)} macros to {OUT}")
missing = [n for n in ("phase1_policy_identifiability.csv",
                       "phase2_condition_means.csv", "phase3_hypotheses.csv",
                       "phase4_loop_areas.csv", "phase5_variance_matched_means.csv")
           if not have(n)]
if missing:
    print(f"  (not yet available: {missing})")
