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

Before publication, replace the repository placeholder in `CITATION.cff`,
publish this directory in a public repository, and archive the release with
Zenodo or another service that supplies a persistent identifier. The DOI and
public URL must then be copied into the manuscript's data-and-code statement.

The release should be tested with:

```bash
python3 -m pytest tests/ -q
python3 make_numbers.py
```
