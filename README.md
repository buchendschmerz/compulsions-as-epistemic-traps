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
  plotstyle.py             shared publication style
  config.py                seeds, grid resolution, canonical parameters

phase1_nonidentifiability.py  the theorem + geometry     → Fig 2
phase2_bayesian_pomdp.py   Bayesian agent, H1–H3         → Fig 3 data
phase3_interventions.py    intervention contrasts        → Fig 4 data
phase4_hysteresis_phasediagram.py  ramp, step, phase diagrams → Figs S1–S3 data
phase5_predictions.py      matched-exposure experiment   → Fig 5 data
make_submission_figures.py  assembles the five main figures
make_figures.py            assembles the three supporting figures
make_numbers.py            emits paper/numbers.tex from the result files
tests/                     analytic identities, conjugate Bayes and EIG checks

results/                   release tables (CSV/NPZ)
figures/                   supporting figures in PDF and PNG
paper/figures_submission/  final main figures in PDF and PNG
paper/paper_draft.tex      manuscript source
paper/references.bib       bibliography (Crossref-audited; see results/)
docs/deviations.md         where this departs from the original project brief
```

## Running everything

```bash
python3 -m pip install numpy scipy matplotlib pandas sympy pytest pypdf
python3 run_all.py
```

Or phase by phase (approximate runtimes on a laptop):

```bash
python3 -m pytest tests/ -q                                            # ~6 s
python3 phase1_nonidentifiability.py                                   # ~10 s
python3 phase2_bayesian_pomdp.py --n-agents 20 --n-episodes 2000       # ~4 min
python3 phase2_bayesian_pomdp.py --only-kappa --n-agents 12 --n-episodes 2000
python3 phase3_interventions.py --n-agents 25                          # ~15 min
python3 phase4_hysteresis_phasediagram.py --n-agents 16 --sweep-n 11   # ~55 min
python3 phase5_predictions.py
python3 make_figures.py 4 5 8
python3 make_submission_figures.py
python3 make_numbers.py
```

Every simulation script takes `--quick` for a smoke test. All randomness is seeded from
`epistemic_trap.config.MASTER_SEED`; inference is grid-based, so runs are
deterministic within a fixed software environment.

## Main results

| # | Result | Where |
|---|--------|-------|
| 1 | `det I_k ≡ 0` for every fixed `k` and every interior `(q,p)`; SymPy-verified | `phase1`, Fig 2 |
| 2 | Identifiability is set by the **support** of the policy over `k`, not by how much safety behaviour it performs; the best two-point design is `{0,1}`, not `{0, k_compulsive}` | `phase1`, Fig 2B–C |
| 3 | A repetitive Bayesian agent becomes precise about `λ_k` while learning nothing (or unlearning) about `p`; after 2000 episodes it still believes its actions work half as well as they do | `phase2`, Fig 3 |
| 4 | Information about one's own efficacy is **exactly zero at k = 0**, peaks at a small interior `k`, and vanishes again at large `k` | `phase2` fixed-design sweep |
| 5 | Response prevention from a naive start moves the efficacy belief by exactly nothing; it works only as the second half of a contrast with earlier repetitive episodes | `phase3` |
| 6 | A post-action state probe removes state uncertainty and almost no structural uncertainty; the same probe moved before acting is strongly diagnostic | `phase3`, Fig 4C |
| 7 | The trap's second face: the untreated agent resolves the ambiguity the *wrong* way, concludes the world is safe (`E[q]≈0.07` vs a true `0.30`), and under-protects; every condition that probes `q` raises the follow-up repetition rate | `phase3`, Fig 4D |
| 8 | Uncertainty aversion `κ > 0` raises repetition and worsens efficacy learning, but does not sustain repetition over the simulated horizon | `phase2 --only-kappa` |
| 9 | Residual error is **U-shaped** in the realised repetition count: never acting and always acting are equally blind. The cost of harm decides which kind of blindness you get | `phase4`, Fig S1 |
| 10 | Four randomized schedules can match mean exposure while separating raw variance from expected information; fixed `{3}`, `{2,4}`, `{1,5}`, and uniform `0..6` define the proposed human test | `phase5`, Fig 5 |

Hypothesis verdicts are written to `results/phase2_hypotheses.csv`
and `results/phase3_hypotheses.csv` with the metric, the comparison value, and a
boolean. Phase 5 supplies simulated outcomes for the proposed experiment; it is
not presented as human evidence.

## Reproducibility notes

- Bayesian inference is on a 101×101 midpoint grid; `tests/test_grid.py` checks
  it against analytic Beta–Binomial posteriors and against grid refinement.
- Knowledge is scored by `log P(q*, p* | D)`, not by entropy. Entropy alone
  rewards confident wrongness, and one of the findings is that an untreated agent
  becomes confident and wrong.

## Submission status

- [x] Crossref audit of the 25 source references completed; decisions are in
      `results/reference_audit_crossref_decisions.md`.
- [x] Author, affiliation, funding, contribution, competing-interests and AI
      disclosure statements prepared in the separate publisher upload package.
- [x] PLOS submission DOCX/PDF, clean bioRxiv PDF, cover letter and TIFF figure
      files prepared in that upload package.
- [x] Corresponding-author ORCID and public repository URL added to the manuscript and cover letter.
- [ ] Request PLOS fee assistance at submission; no external funding is
      available for the APC.

The LaTeX source is retained for reproducibility. Publisher upload files are
maintained separately so this public release contains only manuscript-related
source, code, results and figures.
