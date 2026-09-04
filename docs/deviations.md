# Where this implementation departs from the project brief

Each departure below was forced by something the analysis or the simulations
showed. They are listed so that the manuscript can state them rather than quietly
absorb them.

---

## 1. "ERP restores identifiability" is false as stated; the corrected version is stronger

**Brief (§3.5, RQ3, H4):** Response Prevention should be formalised as an
intervention that "restores the identifiability of `q` and `p`".

**What the mathematics says.** Pure response prevention means `k = 0`, and

```
P(harm | k=0) = q
```

does not depend on `p` at all. The expected information gain about `p` at `k = 0`
is exactly zero for every prior — verified numerically in
`tests/test_simulate.py::test_response_prevention_alone_is_uninformative_about_efficacy`.
Response prevention is a *perfect* probe of the baseline hazard and a *null*
probe of one's own efficacy.

**Corrected claim, which the simulations support.** What restores identifiability
is the **contrast between distinct repetition counts**. Response prevention works
in practice because the patient arrives with a history of repetitive episodes:
those episodes pin down the composite `λ_k`, the response-prevention episodes pin
down `q`, and the two together determine `p`. Phase 3 demonstrates this directly
by running response prevention with and without a prior repetitive block: with no
history, the efficacy belief moves by exactly nothing.

This is a better result than the original claim, because it converts "exposure
works" into a statement about experimental design with testable consequences
(graded protocols passing through several `k` should beat protocols that jump to
zero).

## 2. H3 as written is non-monotone; the reformulation is more interesting

**Brief (H3):** "In compulsive agents `q` and `p` are strongly correlated in the
posterior and cannot be separately estimated."

**What the controlled sweep shows** (`results/phase2_k_sweep_means.csv`, prior
held fixed, only `k` varying): posterior correlation is *not* monotone in `k`. It
is exactly 0 at `k = 0` (the likelihood never touches `p`), peaks near `k = 1–2`,
and decays towards 0 at large `k`.

The reason is that the two ends of the range are two *different* failures:

- at small `k`, the data are informative but only about a combination, so the
  posterior concentrates along a ridge — strong positive coupling;
- at large `k`, harm essentially never occurs, so the data are uninformative
  about anything and the posterior simply stays at the prior — no coupling,
  because nothing was learned.

The manuscript states both regimes. The analytic prediction of
Corollary 1 (ridge slope `dq/dp = kq/(1−p) ≥ 0`) survives intact: wherever the
design is informative at all, the coupling is positive.

## 3. Entropy is the wrong scoring rule on its own

Entropy reduction rewards confident wrongness. The untreated agent in Phase 3
becomes markedly *more certain* about the baseline hazard while its posterior
places the true value more than five posterior standard deviations away. All
knowledge comparisons therefore use `log P(q*, p* | D)` — a proper scoring rule —
alongside entropy, and the manuscript reports the dissociation between the two as
a result rather than hiding it.

## 4. A purely harm-minimising agent is not permanently compulsive

**Brief (H2):** small initial doubt → more washing → less evidence → doubt
uncorrected → more washing.

The epistemic half of this loop holds strongly: the agent's belief about its own
efficacy remains grossly wrong for thousands of episodes. The *behavioural* half
does not hold for a pure harm minimiser — as its efficacy estimate slowly creeps
up, its repetition count drifts back down.

Behavioural persistence needs one more ingredient, and the model supplies an
independently motivated one: **intolerance of uncertainty** (`κ > 0` in the cost
function), which makes the agent pay for the *variance* of the hazard as well as
its mean. Then uncertainty buys repetitions, repetitions suppress evidence, and
the suppressed evidence keeps the uncertainty high. `phase2 --only-kappa` sweeps
`κ` and shows repetition count rising monotonically and ceasing to decay.

## 5. Hysteresis is real but must be reported carefully

Ramping the hazard up and down produces a behavioural loop in *every* arm,
because any belief updater lags a moving world, and the loop areas do not separate
the arms (0.36 locked vs 0.38 quota, with SDs around 0.2). A loop by itself is
therefore not evidence for the trap, and the manuscript says so rather than
presenting loop area as if it were diagnostic.

What discriminates is **calibration on the descending limb specifically**
(0.172 ± 0.012 locked vs 0.115 ± 0.016 quota, against ~0.10 for both on the
ascending limb) and the step-removal experiment: after the hazard is set to zero,
the locked agent needs 1153 episodes to halve its repetition rate and still
believes the hazard to be 0.323, while an information-seeking agent needs 219.

## 6. A "phantom hazard" result that did not survive its own robustness check

An early version of the step-removal experiment ran without forgetting
(`γ = 1`). Under that setting an agent that copes by *checking* appeared to
sustain a phantom hazard indefinitely — the false-positive floor of its own test.
It looked like an attractive second finding.

It was an artefact. With `γ = 1` the agent has no way to represent a changing
world at all, so *every* arm simply averaged the pre-change and post-change
regimes and none of them recovered; the checker was not special. Re-running with
exponential forgetting (`γ = 0.998`), which is the only mechanism this agent has
for tracking change, the checker recovers normally (`E[q] = 0.027` in a
hazard-free world, against `0.323` for the locked design).

The claim was removed from the manuscript. It is recorded here because the
robustness check is the interesting part: a result that depends on the agent
being unable to represent change in the first place is not a result about
checking.

## 7. Phase 0 replication is partial, and says so

The direction and significance of the `Δp_SUCCESS` effect replicate. Two things
differ from the original: our world-selection criterion favours worlds with
effective washing, which shifts both groups' `Δp_SUCCESS` negative, and we find
group differences in cost parameters of comparable or larger effect size than the
original reported. We attribute this to the solver (grid value iteration rather
than `pomdp-solve`) and the world sample, and we flag it in the manuscript rather
than presenting a cleaner replication than we have.

We also exclude agents whose subjectively optimal policy never selects the
consequential action. For such agents a "compulsive episode" — a run of
preventive actions *between* consequential actions — is undefined, and including
them let a degenerate corner of the cost simplex dominate the comparison.

## 8. Mutual review by two tools, replaced by a test suite

The brief specified that Codex and Claude Code review each other's numerics. Only
one tool was available, so the intent is served by 176 automated tests that check
the numerically delicate parts against closed forms: the grid posterior against
analytic Beta–Binomial updates, expected information gain against brute-force
mutual information, the Fisher information against finite differences, and the
POMDP tables against the published tables. `tests/` is the artefact that replaces
the review step.

## 9. The trap has two arms, and the phase diagram shows it

The brief's phase diagram was expected to show residual doubt growing with the
cost of harm. It does not, and what it shows instead is better. Pooled over the
whole sweep, residual error in the efficacy belief is a **U-shaped** function of
the realised repetition count: agents that settle at `k = 0` are as ignorant of
their own efficacy as agents driven to `k ≈ 6`, and the minimum sits at a small
interior value. The cost of harm decides *which* kind of blindness an agent gets,
not how blind it is.

This is the same interior optimum that the controlled design sweep produces
(Fig. 3F), now arising from agents that were never told what to do, so the two
results corroborate each other.

## 10. The untreated agent under-protects

Setting up Phase 3 we expected the untreated arm to remain the most compulsive.
It does not. By the end of the pre-treatment block the untreated agent has
resolved its non-identifiability *in the wrong direction*: it concludes that the
world is safe (`E[q] ≈ 0.07` against a true `0.30`, roughly seven posterior
standard deviations away) rather than that its actions work. It therefore acts
*less* than the truth warrants, and every condition that supplies a genuine probe
of `q` raises the follow-up repetition rate.

We report this rather than tuning it away, because it sharpens the central claim.
A policy-induced rank deficiency does not have a characteristic behavioural
direction — the ridge admits both readings. What it reliably produces is a wrong
model held with unwarranted confidence.

## 11. Prediction 2 had to be made non-circular

In the simulated population, decision noise raises both the mean and the variance
of the action count, so regressing severity on `Var(k)` is partly mechanical. Two
changes fix it:

- the primary regression now targets the **accuracy** of the efficacy belief,
  `|E[p] − p*|`, with "where the agent started" as the base model;
- the headline claim is tested **within strata matched on severity**, where
  variability still predicts posterior precision (r between −0.25 and −0.29 in
  every stratum).

We also report a null: raw `Var(k)` adds nothing to the accuracy regression once
severity is controlled; it is the expected information the realised policy
generated that carries the effect. Variability matters, but only variability
*where outcomes still occur*.

## 12. Entropy of the observable composite saturates

`H(λ_k)` is computed on a fixed `[0,1]` bin grid so that it is directly
comparable with `H(p)` and `H(q)`. For a compulsive agent at large `k`, `λ_k` is
of order `10⁻⁵` and the whole posterior falls inside the first bin, so the entropy
saturates and the reported *reduction* is a lower bound. The manuscript therefore
reports the scale-free posterior-SD ratio alongside the nats, and the docstring in
`grid.py` carries the warning.

## 13. Responsibility feedback closes behaviour, but by adding a new assumption

The Phase 2 limitation remains important: uncertainty aversion alone raises
repetition and worsens knowledge, but it does not keep behaviour high over the
simulated horizon. `phase6_responsibility_feedback.py` asks the follow-up question
left in the draft: what if recent repetition increases the subjective cost of a
bad outcome, as if harm after repeated safety action feels more personally
responsible?

With that extra feedback term, the behavioural loop can close. This is a
boundary-condition result, not a rescue of the original H2. The central theorem is
unchanged: fixed designs are still non-identifying. What changes is the policy
layer, because repeated action now makes future harm more costly, which buys yet
more repetition. The manuscript should therefore present this as evidence that
persistent behaviour needs an additional psychological ingredient, not as
evidence that non-identifiability alone is sufficient for chronic compulsion.
