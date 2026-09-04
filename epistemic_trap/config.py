"""Global configuration: seeds, grid resolution, and canonical parameter sets."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- reproducibility
MASTER_SEED = 20240620  # publication date of Rigoux et al. (2024)

# ---------------------------------------------------------------- grid resolution
N_GRID = 101  # points per parameter axis, as specified in the project brief
K_MAX = 10    # maximum number of preventive repetitions considered

# ---------------------------------------------------------------- world defaults
# "World" = objective generative parameters of the environment.
WORLD = dict(
    q=0.30,      # p_SOILING: probability the episode starts in the hazardous state
    p=0.80,      # p_SUCCESS: probability one preventive action removes the hazard
    d_dirty=0.80,  # p_DETECT_DIRTY: check sensitivity
    d_clean=0.80,  # p_DETECT_CLEAN: check specificity
)

# ---------------------------------------------------------------- agent defaults
# Beta priors over the two latent causal parameters.
PRIOR_ACCURATE = dict(a_q=3.0, b_q=7.0, a_p=8.0, b_p=2.0)          # E[q]=.30, E[p]=.80
PRIOR_MILD_DOUBT = dict(a_q=3.0, b_q=7.0, a_p=5.0, b_p=5.0)        # E[p]=.50
PRIOR_SEVERE_DOUBT = dict(a_q=3.0, b_q=7.0, a_p=1.5, b_p=13.5)     # E[p]=.10

PRIORS = {
    "accurate": PRIOR_ACCURATE,
    "mild_doubt": PRIOR_MILD_DOUBT,
    "severe_doubt": PRIOR_SEVERE_DOUBT,
}

# ---------------------------------------------------------------- decision costs
COST = dict(
    harm=20.0,    # cost_SICKNESS (magnitude)
    wash=0.25,    # cost_WASH per preventive repetition
    check=0.10,   # cost_CHECK
)

BETA_CHOICE = 12.0  # inverse temperature of the softmax over k
TAU_INFO = 0.0      # information-seeking weight (0 = purely harm-minimising)
KAPPA_RISK = 0.0    # uncertainty aversion: weight on the posterior SD of lambda_k
