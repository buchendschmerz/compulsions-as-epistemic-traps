# Compulsions as Epistemic Traps
### Repetition obscures causal control — an extension of Rigoux, Stephan & Petzschner (2024)

**Central claim.** Even with exactly Bayesian belief updating and exact credit
assignment, a repetitive safety policy makes two latent causal quantities — the
baseline hazard `q` and the efficacy of one's own preventive action `p` —
*structurally non-identifiable*, because behaviour decides which counterfactual
contrasts enter the data.

The observable quantity is the composite

```
P(harm | k) = λ_k = q · (1 − p)^k
```

and the Fisher information of a terminal harm/no-harm observation at fixed `k` is
an outer product of a single vector: rank one, `det I_k ≡ 0`. Non-identifiability
under a repetitive policy is therefore a **theorem about the design**, not a
numerical finding.

---

## Repository layout

```
epistemic_trap/            library
  hazard.py                λ_k, gradients, Fisher information, identifiability
  grid.py                  grid-based Bayesian posterior over (q, p)
  policies.py              belief → number of repetitions k
  simulate.py              episode generative model, agent runner
  pomdp.py                 Rigoux et al. (2024) POMDP replication
  plotstyle.py             shared publication style
  config.py                seeds, grid resolution, canonical parameters

phase0_replication.py      Rigoux replication            → Fig S1
phase1_nonidentifiability.py  the theorem + geometry     → Fig 2
phase2_bayesian_pomdp.py   Bayesian agent, H1–H3         → Fig 3 data
phase3_interventions.py    8 intervention arms, H4–H5    → Fig 6 data
phase4_hysteresis_phasediagram.py  ramp, step, phase diagrams → Figs 4, 5 data
phase5_predictions.py      human-experiment predictions  → Fig 7 data
phase6_responsibility_feedback.py  post-hoc behavioural-loop robustness check
make_figures.py            assembles Figs 1, 3, 4, 5, 6, 7 and Fig S2
make_numbers.py            emits paper/numbers.tex from the result files
tests/                     177 tests (analytic identities, conjugate Bayes, EIG)
                           ruff-clean (F, E9)

results/                   every intermediate table (CSV/NPZ)
figures/                   PDF (for LaTeX) and PNG (for viewing)
paper/paper_draft.tex      manuscript draft
paper/references.bib       bibliography (Crossref-audited; see results/)
docs/deviations.md         where this departs from the original project brief
The publisher-upload package is prepared separately in the parent project's
`output/submission/` directory; this release contains code, data and figures.
```

## Running everything

```bash
python3 -m pip install numpy scipy matplotlib pandas sympy pytest pypdf
python3 run_all.py
```

Or phase by phase (approximate runtimes on a laptop):

```bash
python3 -m pytest tests/ -q                                            # ~6 s
python3 phase0_replication.py --n-worlds 60 --n-per-group 60           # ~70 s
python3 phase1_nonidentifiability.py                                   # ~10 s
python3 phase2_bayesian_pomdp.py --n-agents 20 --n-episodes 2000       # ~4 min
python3 phase2_bayesian_pomdp.py --only-kappa --n-agents 12 --n-episodes 2000
python3 phase3_interventions.py --n-agents 25                          # ~15 min
python3 phase4_hysteresis_phasediagram.py --n-agents 16 --sweep-n 11   # ~55 min
python3 phase5_predictions.py --n-individuals 300                      # ~10 min
python3 phase6_responsibility_feedback.py --n-agents 8                 # ~2 min
python3 make_figures.py
python3 make_numbers.py
```

Every script takes `--quick` for a smoke test. All randomness is seeded from
`epistemic_trap.config.MASTER_SEED`; inference is grid-based, so results are
bit-reproducible.

## Main results

| # | Result | Where |
|---|--------|-------|
| 1 | `det I_k ≡ 0` for every fixed `k` and every interior `(q,p)`; SymPy-verified | `phase1`, Fig 2 |
| 2 | Identifiability is set by the **support** of the policy over `k`, not by how much safety behaviour it performs; the best two-point design is `{0,1}`, not `{0, k_compulsive}` | `phase1`, Fig 2D–E |
| 3 | A repetitive Bayesian agent becomes precise about `λ_k` while learning nothing (or unlearning) about `p`; after 2000 episodes it still believes its actions work half as well as they do | `phase2`, Fig 3 |
| 4 | Information about one's own efficacy is **exactly zero at k = 0**, peaks at a small interior `k`, and vanishes again at large `k` | `phase2`, Fig 3F |
| 5 | Response prevention from a naive start moves the efficacy belief by exactly nothing; it works only as the second half of a contrast with earlier repetitive episodes | `phase3` |
| 6 | Reassurance (checking *after* acting) removes state uncertainty and almost no structural uncertainty; the same check moved *before* acting is strongly diagnostic | `phase3`, Fig 6D |
| 7 | The trap's second face: the untreated agent resolves the ambiguity the *wrong* way, concludes the world is safe (`E[q]≈0.07` vs a true `0.30`), and under-protects; every condition that probes `q` raises the follow-up repetition rate | `phase3`, Fig 6F |
| 8 | Behavioural persistence (a truly self-maintaining compulsion) requires uncertainty aversion `κ > 0`; without it the belief stays wrong but the behaviour decays | `phase2 --only-kappa` |
| 9 | Residual error is **U-shaped** in the realised repetition count: never acting and always acting are equally blind. The cost of harm decides which kind of blindness you get | `phase4`, Fig 4D |
| 10 | At matched mean exposure, `Var(k)` predicts posterior precision about `p` — the sharpest human prediction; it survives stratification on severity | `phase5`, Fig 7 |
| 11 | Uncertainty aversion (`κ>0`) raises repetition and worsens the efficacy belief, but does **not** make the compulsion behaviourally self-sustaining in this model — reported as a null | `phase2 --only-kappa` |
| 12 | A post-hoc responsibility feedback from recent repetition to subjective harm cost **can** close the behavioural loop, but this works by adding a new psychological assumption rather than by changing the identifiability theorem | `phase6` |

Hypothesis verdicts are written to `results/phase2_hypotheses.csv`,
`results/phase3_hypotheses.csv`, `results/phase5_predictions.csv`, and
`results/phase6_hypotheses.csv` with the metric, the comparison value, and a
boolean.

## Reproducibility notes

- Bayesian inference is on a 101×101 midpoint grid; `tests/test_grid.py` checks
  it against analytic Beta–Binomial posteriors and against grid refinement.
- The POMDP replication uses discounted value iteration on a discretised belief
  interval instead of `pomdp-solve`; `tests/test_pomdp.py` checks the transition,
  observation and reward tables against the published tables and checks
  resolution stability.
- Knowledge is scored by `log P(q*, p* | D)`, not by entropy. Entropy alone
  rewards confident wrongness, and one of the findings is that an untreated agent
  becomes confident and wrong.

## Submission status

- [x] Crossref audit of the 25 source references completed; decisions are in
      `results/reference_audit_crossref_decisions.md`.
- [x] Author, affiliation, funding, contribution, competing-interests and AI
      disclosure statements prepared in the separate manuscript package.
- [x] PLOS submission DOCX/PDF, cover letter and TIFF figure files prepared
      outside this code-and-data release.
- [ ] Add the corresponding author's ORCID and the public repository URL to
      the manuscript metadata before upload.
- [ ] Request PLOS fee assistance at submission; no external funding is
      available for the APC.

The LaTeX source is retained for reproducibility, but the intended submission
format is the DOCX/PDF package in `output/submission/`.
