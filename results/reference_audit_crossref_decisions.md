# Crossref reference audit decisions

Audit date: 2026-09-02  
Scope: the 25 bibliography entries marked `VERIFY` in the pre-audit file  
External data sent: titles only  
Local-only matching signals: author family names and publication year

## Decision summary

- 15 exact Crossref records: metadata retained or corrected and DOI added.
- 2 Crossref records with title variants: accepted after local author/year checks.
- 2 records absent from Crossref: retained as canonical ICML/JMLR venue records without a DOI.
- 6 entries without a credible Crossref match: removed from the manuscript and bibliography.

## Entry-level decisions

| Key | Decision | Crossref outcome |
|---|---|---|
| `fradkin2020latent` | Retain; add DOI | Exact title, authors, and year |
| `sakai2022eligibility` | Retain; add DOI | Exact title, authors, and year |
| `geana2026local` | Remove | No credible record |
| `vaghi2017compulsivity` | Retain; correct pages; add DOI | Exact record |
| `seow2020transdiagnostic` | Retain; add issue and DOI | Exact record |
| `sharp2023transition` | Retain; complete volume, issue, pages, and DOI | Exact journal record |
| `gillan2014enhanced` | Retain; add DOI | Exact record |
| `rachman2002checking` | Retain; add DOI | Exact record |
| `vandenhout2003repeated` | Retain; add DOI | Exact record |
| `vandenhout2019meta` | Remove | Query returned the 2003 article, not a 2019 meta-analysis |
| `abbasi2023checking` | Correct title, authors, journal, article number, and DOI | Author/year match to a longer Crossref title |
| `dar2022underconfidence` | Remove | No credible record |
| `champion2022reassurance` | Remove | No credible record |
| `lovibond2009safety` | Retain; correct author given names; add DOI | Exact record |
| `craske2014inhibitory` | Retain; add DOI | Exact record |
| `vervliet2015low` | Retain; add DOI | Exact record |
| `elsner2022expectancy` | Remove | No credible record |
| `lakkaraju2017selective` | Use Crossref title and DOI | Authors/year match; Crossref uses a shorter title |
| `chaney2018algorithmic` | Retain; add DOI | Exact record |
| `moldovan2012safe` | Retain without DOI | No matching Crossref record; ICML venue metadata retained |
| `haggard2017sense` | Retain; add DOI | Exact record |
| `moore2016agency` | Retain; add DOI | Exact record |
| `gentsch2012agency` | Retain; add DOI | Exact record |
| `giuliani2021agency` | Remove | No credible record |
| `he2008active` | Retain without DOI | No matching Crossref record; JMLR venue metadata retained |

The raw Crossref results remain in `reference_audit_crossref.csv` and
`reference_audit_crossref.json` so that every candidate and score can be reviewed.
