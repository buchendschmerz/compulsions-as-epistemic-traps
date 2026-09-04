#!/usr/bin/env python3
r"""Phase 4 -- hysteresis and the geometry of the trap.

Part A: hysteresis
------------------
The objective hazard ``q`` is ramped slowly up and then back down.  Every arm uses
the same learner, the same environment and the same accurate prior; the arms
differ only in how much variation their policy retains.

Two things must be separated here, and the simulation separates them.  *Every*
tracking agent shows some path dependence, because belief updating lags a moving
world -- that part is not diagnostic.  What is diagnostic is that the locked
design also ends up systematically *miscalibrated*: its estimate of the hazard
stays high on the descending limb because the evidence that would bring it down
is exactly the evidence its own policy suppresses.  We therefore report both the
behavioural loop area and the belief calibration error, split by limb.

Part A2: step removal of the hazard
-----------------------------------
A sharper version: the hazard is removed outright.  How long does the agent go on
protecting itself against a danger that no longer exists?  This is also where a
second, independent trap appears: an agent that copes by *checking* rather than by
varying its actions runs into the false-positive floor of its own test, and
sustains a phantom hazard indefinitely.

Part B: phase diagram
---------------------
Where in parameter space does the trap live?  We sweep initial doubt about
efficacy against the cost of harm.  The prediction is a sharp boundary: the
higher the cost of being wrong, the less an agent can afford the behavioural
variation that would tell it whether it is wrong.

Part C: the escape axis
-----------------------
A third axis, the information-seeking weight ``tau``, shows what dissolves the
trap.  An agent willing to pay a little for causal knowledge escapes even at
harm costs that trap the purely safety-driven agent.

Outputs
-------
results/phase4_hysteresis.csv        per-episode ramp trajectories
results/phase4_hysteresis_loops.csv  binned up/down limbs and loop areas
results/phase4_phase_diagram.csv     doubt x harm-cost sweep
results/phase4_tau_diagram.csv       doubt x information-seeking sweep
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from epistemic_trap.config import (COST, K_MAX, MASTER_SEED, RESULTS,
                                   WORLD)
from epistemic_trap.grid import ParameterGrid
from epistemic_trap.policies import GreedyPolicy, QuotaPolicy
from epistemic_trap.simulate import World

KS = np.arange(K_MAX + 1)
P_TRUE = WORLD["p"]


def beta_from_mean(mean, concentration=15.0):
    """Beta parameters with a given mean and fixed concentration."""
    return dict(a_p=concentration * mean, b_p=concentration * (1.0 - mean))


def prior_with_doubt(mean_p, mean_q=0.30, concentration=15.0):
    return dict(a_q=concentration * mean_q, b_q=concentration * (1.0 - mean_q),
                **beta_from_mean(mean_p, concentration))


# ------------------------------------------------------------------ (A) ramp
def ramp_schedule(q_lo, q_hi, n_up, n_hold, n_down):
    """Slow up-ramp, plateau, slow down-ramp of the objective hazard."""
    up = np.linspace(q_lo, q_hi, n_up)
    hold = np.full(n_hold, q_hi)
    down = np.linspace(q_hi, q_lo, n_down)
    limb = (["up"] * n_up) + (["hold"] * n_hold) + (["down"] * n_down)
    return np.concatenate([up, hold, down]), np.array(limb)


def run_ramp(prior, beta, tau, gamma_forget, schedule, limbs, seed, cost=None,
             p_true=None, check_mode=None, epsilon=0.0, policy=None):
    """One agent living through a non-stationary hazard."""
    p_true = P_TRUE if p_true is None else p_true
    rng = np.random.default_rng(seed)
    grid = ParameterGrid.from_beta_prior(**prior)
    if policy is None:
        policy = (QuotaPolicy(k_values=KS, beta=beta, tau=tau, cost=cost,
                              epsilon=epsilon)
                  if epsilon > 0 else
                  GreedyPolicy(k_values=KS, beta=beta, tau=tau, cost=cost))
    rows = []
    for t, (q_t, limb) in enumerate(zip(schedule, limbs)):
        world = World(q=float(q_t), p=p_true)
        k = policy.sample(grid, rng, t=t)
        harm, obs, m_chk = world.run_episode(k, rng, check_mode=check_mode)
        grid.update_harm(k, harm)
        if obs is not None:
            grid.update_check(m_chk, bool(obs), world.d_dirty, world.d_clean)
        grid.flatten(gamma_forget)
        mq, mp = grid.mean()
        rows.append(dict(episode=t, q_world=float(q_t), limb=limb, k=k, harm=harm,
                         mean_q=mq, mean_p=mp, err_q=mq - float(q_t),
                         sd_p=float(np.sqrt(grid.cov()[1, 1])),
                         corr_qp=grid.corr_qp(), H_p=grid.entropy_p()))
    return pd.DataFrame(rows)


def hysteresis_loop(df, n_bins=12):
    """Binned mean ``k`` on the ascending and descending limbs, and the loop area."""
    edges = np.linspace(df.q_world.min(), df.q_world.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = []
    for limb in ("up", "down"):
        sub = df[df.limb == limb]
        idx = np.clip(np.digitize(sub.q_world, edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            m = idx == b
            if m.sum():
                out.append(dict(limb=limb, q_bin=centres[b],
                                mean_k=float(sub.k.values[m].mean()),
                                mean_p=float(sub.mean_p.values[m].mean()),
                                mean_q=float(sub.mean_q.values[m].mean()),
                                harm_rate=float(sub.harm.values[m].mean())))
    loop = pd.DataFrame(out)
    piv_k = loop.pivot(index="q_bin", columns="limb", values="mean_k").dropna()
    piv_b = loop.pivot(index="q_bin", columns="limb", values="mean_q").dropna()
    area_k = float(np.trapz(piv_k["down"] - piv_k["up"], piv_k.index))
    area_b = float(np.trapz(piv_b["down"] - piv_b["up"], piv_b.index))
    return loop, area_k, area_b


# ------------------------------------------------------- (B/C) parameter sweeps
def run_sweep_cell(mean_p_prior, harm_cost, tau, n_episodes, seed,
                   beta=60.0, gamma_forget=1.0, concentration=15.0):
    """Simulate one cell of a phase diagram and return its summary."""
    from epistemic_trap.simulate import AgentSpec, run_agent
    cost = dict(COST, harm=float(harm_cost))
    spec = AgentSpec(prior=prior_with_doubt(mean_p_prior,
                                            concentration=concentration),
                     policy=GreedyPolicy(k_values=KS, beta=beta, tau=tau, cost=cost),
                     gamma_forget=gamma_forget, label="sweep")
    world = World(**WORLD)
    df = run_agent(world, spec, n_episodes, seed=seed, log_every=n_episodes)
    last = df.iloc[-1]
    ks = df.attrs["k_history"]
    tail = ks[len(ks) // 2:]
    return dict(mean_p_prior=mean_p_prior, harm_cost=harm_cost, tau=tau,
                concentration=concentration,
                mean_k=float(tail.mean()), var_k=float(tail.var()),
                k_support=int(len(np.unique(tail))),
                mean_p=float(last.mean_p), err_p=float(last.err_p),
                abs_err_p=float(abs(last.err_p)),
                z_p=float(abs(last.err_p) / max(last.sd_p, 1e-12)),
                mean_q=float(last.mean_q), abs_err_q=float(abs(last.err_q)),
                logdens_true=float(last.logdens_true),
                corr_qp=float(last.corr_qp), H_p=float(last.H_p),
                harm_rate=float(last.harm_rate_sofar),
                singular=bool(last.fim_singular))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=12)
    ap.add_argument("--ramp-len", type=int, default=1200)
    ap.add_argument("--sweep-n", type=int, default=11)
    ap.add_argument("--sweep-seeds", type=int, default=4)
    ap.add_argument("--sweep-episodes", type=int, default=800)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="A,A2,B,C,D",
                    help="comma-separated subset of A,A2,B,C,D to run")
    args = ap.parse_args()
    parts = {s.strip() for s in args.parts.split(",")}
    if args.quick:
        args.n_agents, args.ramp_len = 2, 200
        args.sweep_n, args.sweep_seeds, args.sweep_episodes = 5, 1, 200

    t0 = time.time()
    schedule, limbs = ramp_schedule(0.05, 0.60, args.ramp_len, args.ramp_len // 4,
                                    args.ramp_len)

    # ------------------------------------------------------------------ (A)
    # p_true = 0.5: a partially effective preventive action, which gives the
    # number of repetitions a wide enough dynamic range for a ramp to be visible.
    prior_A = dict(a_q=3.0, b_q=7.0, a_p=7.5, b_p=7.5)
    cost_A = dict(COST, harm=20.0, wash=0.10)
    P_TRUE_A = 0.5
    arms = {
        "locked": dict(beta=60.0, tau=0.0, epsilon=0.0, check=None),
        "quota": dict(beta=60.0, tau=0.0, epsilon=0.25, check=None),
        "curious": dict(beta=60.0, tau=60.0, epsilon=0.0, check=None),
        "checker": dict(beta=60.0, tau=0.0, epsilon=0.0, check="baseline"),
    }
    ramp_rows, loop_rows, areas = [], [], []
    for name, cfg in (arms.items() if "A" in parts else []):
        for a in range(args.n_agents):
            df = run_ramp(prior_A, cfg["beta"], cfg["tau"], gamma_forget=0.998,
                          schedule=schedule, limbs=limbs,
                          seed=MASTER_SEED + 31_000 + 97 * a,
                          cost=cost_A, p_true=P_TRUE_A,
                          check_mode=cfg["check"], epsilon=cfg["epsilon"])
            df["arm"], df["agent"] = name, a
            ramp_rows.append(df)
            loop, area_k, area_b = hysteresis_loop(df)
            loop["arm"], loop["agent"] = name, a
            loop_rows.append(loop)
            up, down = df[df.limb == "up"], df[df.limb == "down"]
            k_range = float(np.ptp(loop[loop.limb == "up"].mean_k))
            areas.append(dict(
                arm=name, agent=a, loop_area_k=area_k, loop_area_belief=area_b,
                loop_area_k_norm=area_k / max(k_range, 1e-9),
                calib_err=float(np.abs(df.mean_q - df.q_world).mean()),
                calib_err_up=float(np.abs(up.mean_q - up.q_world).mean()),
                calib_err_down=float(np.abs(down.mean_q - down.q_world).mean()),
                mean_k=float(df.k.mean()), harm_rate=float(df.harm.mean()),
                final_mean_p=float(df.mean_p.iloc[-1])))
        print(f"[phase4A] {name:>9}  ({time.time() - t0:.0f}s)")

    if "A" in parts:
        ramp = pd.concat(ramp_rows, ignore_index=True)
        ramp.to_csv(RESULTS / "phase4_hysteresis.csv", index=False)
        loops = pd.concat(loop_rows, ignore_index=True)
        loops.to_csv(RESULTS / "phase4_hysteresis_loops.csv", index=False)
        area_df = pd.DataFrame(areas)
        area_df.to_csv(RESULTS / "phase4_loop_areas.csv", index=False)
        print("\n[phase4A] ramp summary (mean over agents):")
        print(area_df.groupby("arm")[["loop_area_k", "loop_area_k_norm", "calib_err",
                                      "calib_err_up", "calib_err_down", "mean_k",
                                      "harm_rate", "final_mean_p"]]
              .mean().round(4).to_string())

    # ----------------------------------------------------------------- (A2)
    n_high, n_zero = args.ramp_len // 2, args.ramp_len
    step = np.concatenate([np.full(n_high, 0.60), np.full(n_zero, 0.0)])
    step_limbs = np.array(["hazard"] * n_high + ["removed"] * n_zero)
    step_rows, step_summ = [], []
    for name, cfg in (arms.items() if "A2" in parts else []):
        for a in range(args.n_agents):
            # The agent has no change-point model, so it can only track a moving
            # world through exponential forgetting. Without it, every arm simply
            # averages the two regimes and the comparison is uninformative.
            df = run_ramp(prior_A, cfg["beta"], cfg["tau"], gamma_forget=0.998,
                          schedule=step, limbs=step_limbs,
                          seed=MASTER_SEED + 41_000 + 97 * a,
                          cost=cost_A, p_true=P_TRUE_A,
                          check_mode=cfg["check"], epsilon=cfg["epsilon"])
            df["arm"], df["agent"] = name, a
            step_rows.append(df)
            base = float(df[df.limb == "hazard"].k.tail(100).mean())
            after = df[df.limb == "removed"].reset_index(drop=True)
            roll = after.k.rolling(100, min_periods=100).mean().values
            hit = np.where(roll < 0.5 * base)[0]
            step_summ.append(dict(
                arm=name, agent=a, k_hazard=base,
                k_final=float(after.k.tail(200).mean()),
                episodes_to_halve=int(hit[0]) if len(hit) else n_zero,
                censored=bool(len(hit) == 0),
                belief_q_final=float(after.mean_q.iloc[-1]),
                belief_p_final=float(after.mean_p.iloc[-1])))
        print(f"[phase4A2] {name:>9}  ({time.time() - t0:.0f}s)")
    if "A2" in parts:
        pd.concat(step_rows, ignore_index=True).to_csv(
            RESULTS / "phase4_step_removal.csv", index=False)
        step_df = pd.DataFrame(step_summ)
        step_df.to_csv(RESULTS / "phase4_step_summary.csv", index=False)
        print("\n[phase4A2] hazard removed at mid-run "
              f"(horizon {n_zero} episodes):")
        print(step_df.groupby("arm")[["k_hazard", "k_final", "episodes_to_halve",
                                      "censored", "belief_q_final",
                                      "belief_p_final"]]
              .mean().round(3).to_string())

    # ------------------------------------------------------------------ (B)
    doubts = np.linspace(0.05, 0.95, args.sweep_n)
    harm_costs = np.geomspace(2.0, 80.0, args.sweep_n)
    rows = []
    for i, d in enumerate(doubts if "B" in parts else []):
        for h in harm_costs:
            for s in range(args.sweep_seeds):
                rows.append(run_sweep_cell(float(d), float(h), 0.0,
                                           args.sweep_episodes,
                                           MASTER_SEED + 51_000 + s))
        print(f"[phase4B] doubt {i + 1}/{len(doubts)}  ({time.time() - t0:.0f}s)",
              end="\r")
    if "B" in parts:
        pdiag = pd.DataFrame(rows)
        pdiag.to_csv(RESULTS / "phase4_phase_diagram.csv", index=False)
        print(f"\n[phase4B] {len(pdiag)} cells  ({time.time() - t0:.0f}s)")
    else:
        pdiag = pd.read_csv(RESULTS / "phase4_phase_diagram.csv")

    # ------------------------------------------------------------------ (C)
    taus = np.concatenate([[0.0], np.geomspace(1.0, 300.0, args.sweep_n - 1)])
    rows = []
    for i, d in enumerate(doubts if "C" in parts else []):
        for tau in taus:
            for s in range(args.sweep_seeds):
                rows.append(run_sweep_cell(float(d), COST["harm"], float(tau),
                                           args.sweep_episodes,
                                           MASTER_SEED + 61_000 + s))
        print(f"[phase4C] doubt {i + 1}/{len(doubts)}  ({time.time() - t0:.0f}s)",
              end="\r")
    if "C" in parts:
        tdiag = pd.DataFrame(rows)
        tdiag.to_csv(RESULTS / "phase4_tau_diagram.csv", index=False)

    # -------------------------------------------------- (D) the brief's axes
    # The brief asked for initial *uncertainty* against harm cost. Holding the
    # prior mean fixed and sweeping its concentration isolates confidence from
    # bias -- exactly the separation Rigoux et al. identify as needing a Bayesian
    # formulation.
    if "D" in parts:
        concentrations = np.geomspace(2.0, 200.0, args.sweep_n)
        rows = []
        for i, conc in enumerate(concentrations):
            for h in harm_costs:
                for s in range(args.sweep_seeds):
                    rows.append(run_sweep_cell(0.35, float(h), 0.0,
                                               args.sweep_episodes,
                                               MASTER_SEED + 71_000 + s,
                                               concentration=float(conc)))
            print(f"[phase4D] concentration {i + 1}/{len(concentrations)}  "
                  f"({time.time() - t0:.0f}s)", end="\r")
        udiag = pd.DataFrame(rows)
        udiag.to_csv(RESULTS / "phase4_uncertainty_diagram.csv", index=False)
        agg = (udiag.groupby("concentration")[["mean_k", "abs_err_p", "z_p",
                                               "harm_rate"]].mean())
        print("\n[phase4D] prior mean held at 0.35; sweeping prior concentration")
        print("          (low concentration = high initial uncertainty):")
        print(agg.round(3).to_string())

    summary = (pdiag.groupby(["mean_p_prior", "harm_cost"])
               .agg(mean_k=("mean_k", "mean"), abs_err_p=("abs_err_p", "mean"),
                    z_p=("z_p", "mean"), harm_rate=("harm_rate", "mean"))
               .reset_index())
    print("\n[phase4B] trap severity by initial doubt (averaged over harm cost):")
    print(summary.groupby("mean_p_prior")[["mean_k", "abs_err_p", "z_p"]]
          .mean().round(3).to_string())
    print(f"\n[phase4] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
