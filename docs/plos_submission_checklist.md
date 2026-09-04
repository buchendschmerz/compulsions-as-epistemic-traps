# PLOS Computational Biology Submission Checklist

## Current package

- Manuscript: `output/submission/plos_compulsions_as_epistemic_traps_submission.docx` and `.pdf`
- Cover letter: `output/submission/plos_cover_letter.docx` and `.pdf`
- Main figures: `output/submission/figures/Fig1.tif` through `Fig5.tif`
- Supporting figures: `output/submission/figures/S1_Fig.tif` through `S4_Fig.tif`
- Reproduction source: `output/public_release/`

The manuscript uses the PLOS title limit, five keywords, a 189-word Author
Summary, CRediT roles for the sole author, an ethics statement, the required
unfunded statement, a competing-interests statement, and an AI-use disclosure.
It is double-spaced with continuous line numbers and page numbers.

## Two required metadata actions

- [ ] Add Dai Miyazaki's ORCID to the title page and the submission system.
- [ ] Publish `output/public_release/` at a public URL and add that URL to the
      Data and code availability statement. Create a free Zenodo archive from
      the public release and add its DOI before final submission if possible.

The current DOCX intentionally names both missing fields rather than silently
inventing an identifier or a repository URL.

## Lowest-cost route

PLOS Computational Biology lists a Research Article APC of USD 3,165, payable
on acceptance. Because this work has no funding, request PLOS fee assistance
in the submission comments at the same time as submission. The target budget
is therefore USD 0; retain USD 3,165 as the contingency only if assistance is
not granted. Do not purchase editing, figure or repository services: the
source package is reproducible locally, GitHub public repositories are free,
and Zenodo archiving is free.

## Order of operations

1. Create or confirm the ORCID and register the institutional email if the
   publisher requests institutional verification.
2. Publish the prepared code/data release, then create the archival DOI.
3. Replace the two explicit placeholders in the PLOS DOCX/PDF and cover
   letter, and re-render the final PDF once more.
4. Upload the manuscript PDF/DOCX, cover letter, five main figures, four
   supporting figures, and the requested metadata in the PLOS system.
5. Paste the fee-assistance request into the submission comments and keep the
   confirmation email with the manuscript record.
6. If the paper is sent for review, prepare a response matrix during the
   review period. PLOS asks for major revisions within 60 days and minor
   revisions within 30 days.

## Exact finalization command

After obtaining the two URLs, run these commands from the project root. A
GitHub URL is sufficient for code availability; a Zenodo DOI can be used
instead or added as the archival identifier.

```bash
PYTHON=/Users/dai2002/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
ORCID='https://orcid.org/0000-0000-0000-0000'
CODE_URL='https://github.com/REPLACE_WITH_PUBLIC_REPOSITORY'

"$PYTHON" build_plos_submission.py --orcid "$ORCID" --code-url "$CODE_URL"
"$PYTHON" build_plos_cover_letter_docx.py --orcid "$ORCID" --code-url "$CODE_URL"
"$PYTHON" /Users/dai2002/.codex/plugins/cache/openai-primary-runtime/documents/26.903.11726/skills/documents/render_docx.py \
  output/submission/plos_compulsions_as_epistemic_traps_submission.docx \
  --output_dir tmp/render_plos_submission_final --emit_pdf
"$PYTHON" /Users/dai2002/.codex/plugins/cache/openai-primary-runtime/documents/26.903.11726/skills/documents/render_docx.py \
  output/submission/plos_cover_letter.docx \
  --output_dir tmp/render_plos_cover_letter_final --emit_pdf
cp tmp/render_plos_submission_final/plos_compulsions_as_epistemic_traps_submission.pdf \
  output/submission/plos_compulsions_as_epistemic_traps_submission.pdf
cp tmp/render_plos_cover_letter_final/plos_cover_letter.pdf \
  output/submission/plos_cover_letter.pdf
```

Before uploading, search the final DOCX/PDF for `to be added` and `must be
inserted`; both should return no results. Keep the current placeholder version
until the real identifiers are available.

## Browser-side steps

1. Register or sign in at [ORCID](https://orcid.org/register), verify the
   email address, and copy the public 16-digit ORCID URL.
2. Create a public GitHub repository and push the already committed local
   release. From the project root, use
   `git -C output/public_release remote add origin <repo-url>` followed by
   `git -C output/public_release push -u origin main`. Do not upload the
   manuscript PDFs to the code repository unless desired; the code release is
   already complete.
3. Optionally connect the GitHub repository to
   [Zenodo](https://zenodo.org/), create a versioned release, and copy the DOI
   URL. This is free and gives the project a permanent archival identifier.
4. Open the [PLOS Computational Biology submission
   system](https://www.editorialmanager.com/pcbi/default.aspx), choose Research
   Article, and enter the title, author, affiliation, email and ORCID. Upload
   the final manuscript PDF, cover letter, five main TIFF figures and four
   supporting TIFF figures. Keep the DOCX ready if the system requests it.
5. In the data/code and funding questions, paste the public URL, state that
   no external funding is available, and submit the PLOS Publication Fee
   Assistance request during the same submission. Requests made after
   acceptance are not considered.
6. Review the generated submission PDF and all metadata on the confirmation
   page before clicking the final submit button. Save the manuscript ID and
   Fee Assistance confirmation email.

## Source pages

- PLOS submission guidelines: https://journals.plos.org/ploscompbiol/s/submission-guidelines
- PLOS fees and fee assistance: https://plos.org/fees/
- PLOS editorial and peer-review process: https://journals.plos.org/ploscompbiol/s/editorial-and-peer-review-process
- PLOS AI policy: https://journals.plos.org/ploscompbiol/s/ethical-publishing-practice
