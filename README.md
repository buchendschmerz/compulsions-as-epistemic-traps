# Compulsions as epistemic traps

Computational Brain & Behavior submission snapshot, 13 October 2026.

Dai Miyazaki; Graduate School of Humanities and Sociology, The University of Tokyo; dmyzk29@gmail.com; https://orcid.org/0009-0000-0829-4435

Title: Compulsions as epistemic traps: repetition obscures causal control

All observations are synthetic. This snapshot corrects the joint likelihood of state probes and terminal outcomes; see results/cbb_revision_20261013/CORRECTION_PROVENANCE.json. Historical replication and unused POMDP code are excluded. The study does not establish persistent compulsive behaviour or clinical efficacy.

## Reproduce

Create a Python environment and install requirements.txt. Run `python -m pytest tests -q`. Verify the snapshot with `shasum -a 256 -c SHA256SUMS`. The large trajectory file is gzip-compressed; pandas reads it transparently. GZIP_PAYLOAD_SHA256.json records its decompressed SHA256.

To rerun simulations, set `OCD_RESULTS_DIR=results/cbb_revision_20261013` in the environment. Use the full phase1--phase5 commands and seed settings documented in Online Resource 1; the quick modes are smoke tests, not manuscript replications. For phase2 use 20 agents, 2000 episodes and --log-every 10; its --only-kappa run uses 12 agents and the same horizon. Phase3 uses --n-agents 25. Phase4 uses --n-agents 16 --sweep-n 11 --sweep-seeds 3 --parts A,A2,B,C,D. Phase5 uses its full defaults.

Render figures with `python tools/make_cbb_figures.py`. Set `OCD_RESULTS_DIR=results/cbb_revision_20261013` and `OCD_NUMBERS_OUTPUT=paper/cbb_revision/numbers.tex`, then run `python make_numbers.py`. Build Word files with `python build_cbb_submission.py`. The paper_draft.tex file is the preserved input template; the current prose edits are in build_cbb_submission.py, not a standalone current TeX manuscript. A local Word/LibreOffice installation is needed for PDF export; no TeX compiler is required.

Earlier repository commits contain a historical version, not this corrected snapshot. No publication acceptance is claimed.
