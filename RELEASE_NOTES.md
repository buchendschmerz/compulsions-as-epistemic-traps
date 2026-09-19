# Public Release Notes

This directory is a clean release candidate for the PLOS Computational
Biology submission. It contains the source code, tests, generated CSV/NPZ
results, figure sources, manuscript source and reference audit. The copied
reference PDFs, Word/PDF deliverables, render images and run logs are not part
of the release candidate.

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
Figure 4 now displays the same six conditions in every panel using colour,
marker and line-style redundancy. Figure 5 contains the three pre-specified
human-data predictions; the post-hoc responsibility-feedback analysis remains
available as an explicitly exploratory sensitivity analysis and is not used as
evidence in the main manuscript.

The release should be tested with:

```bash
python3 -m pytest tests/ -q
python3 make_numbers.py
```
