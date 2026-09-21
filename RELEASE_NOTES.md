# Public Release Notes

This directory is the reproducibility release for the PLOS Computational
Biology manuscript. It contains only the current phase 1-5 source code, tests,
generated CSV/NPZ results, five main figures, three supporting figures,
manuscript source and reference audit. Superseded exploratory phases, copied
reference PDFs, publisher upload files, render images and run logs are omitted.

The large phase-3 per-episode log is distributed as the deterministic gzip
file `results/phase3_trajectories.csv.gz` so the release remains below GitHub's
100 MB per-file limit. The analysis and figure scripts read it transparently.

After cloning, verify the publishable files with `shasum -a 256 -c SHA256SUMS`.

The public repository is
`https://github.com/buchendschmerz/compulsions-as-epistemic-traps`, and the
author's ORCID is recorded in `CITATION.cff`. No archive DOI has been minted;
none is claimed in the manuscript. If a persistent archive is created later,
its DOI should be added to both the citation metadata and the manuscript.

The five concise manuscript figures were regenerated for the final DOCX/PDF.
Figure 4 displays the same six conditions in every panel using colour, marker
and line-style redundancy. Figure 5 compares the four current matched-exposure
schedules. Its random streams use stable integer design indices rather than
Python's process-dependent string hash, so reruns are deterministic.

The release should be tested with:

```bash
python3 -m pytest tests/ -q
python3 phase5_predictions.py --reanalyse
python3 make_figures.py 4 5 8
python3 make_submission_figures.py
python3 make_numbers.py
```
