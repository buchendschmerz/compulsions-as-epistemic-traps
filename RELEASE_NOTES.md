# Public Release Notes

This directory is a clean release candidate for the PLOS Computational
Biology submission. It contains the source code, tests, generated CSV/NPZ
results, figure sources, manuscript source and reference audit. The copied
reference PDFs, Word/PDF deliverables, render images and run logs are not part
of the release candidate.

Before publication, replace the repository placeholder in `CITATION.cff`,
publish this directory in a public repository, and archive the release with
Zenodo or another service that supplies a persistent identifier. The DOI and
public URL must then be copied into the manuscript's data-and-code statement.

The release should be tested with:

```bash
python3 -m pytest tests/ -q
python3 make_numbers.py
```
