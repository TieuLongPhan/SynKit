# Exact canonicalization of configured molecular stereographs

This is the integrated paper in SynKit. **`submission/` is the single final paper package.** It contains the manuscript, supporting information, cover letter, eleven figures, one editable source ZIP and its verification records.

- [Main manuscript](submission/manuscript.pdf)
- [Supporting information](submission/Additional_file_1.pdf)
- [Cover letter](submission/cover_letter.pdf)
- [Editable sources for all documents and figures](submission/sources.zip)
- [Figure files](submission/figures)
- [Package verification](submission/verification.json)

The authors are Tieu-Long Phan, Peter F. Stadler and Daniel Merkle. Funding follows the full Syndoku statement; Peter and Daniel contribute manuscript revision, supervision and funding acquisition. Competing interests remain explicitly pending. Preparing this package does not perform journal submission.

## Build from SynKit

```sh
python paper/stereo_graph/tools/build_submission.py
```

The command regenerates all 269 numerical claims from 37 checksummed reports, checks the 33 reviewed mathematical blocks, builds the documents and figures, then independently rebuilds them from the editable ZIP. It replaces only the generated `submission/` package after verification succeeds. Python with the SynKit research dependencies, TeX Live, `latexmk`, `pdfcrop` and Poppler tools is required.

## Source layout

| Location | Contents |
| --- | --- |
| `main.tex`, `technical_details.tex` | Article and technical appendix |
| `authors.tex`, `references.bib`, `cover_letter.tex` | Metadata, bibliography and letter source |
| `fig/` | Only graphics and styles used by the paper |
| `results/` | Current result manifest, numerical ledger, generated macros and review/source bindings |
| `tools/` | Build and source-verification commands |
| `submission/` | One current generated package |

Scientific code, tests, datasets and experiments are integrated into `synkit/`, `Test/` and `Experiment/Stereo/`. Frozen chemical selections, dependency locks and the compressed historical paper audit are in [PaperEvidence](../../Experiment/Stereo/Data/Revision/PaperEvidence/README.md). Earlier paper revisions and build products are removed from this active directory; their provenance is retained in that archive and Git history.

`main.tex` is the standalone journal manuscript: Introduction, Methods, Results, Discussion and Conclusions, followed by abbreviations, declarations and references. The theoretical development and validation protocols are nested under Methods. `technical_details.tex` is the standalone supporting information, with separate S-prefixed numbering. Both sources enable double spacing and line numbering directly. This structure was checked against the current [Methodology instructions](https://link.springer.com/journal/13321/submission-guidelines/methodology) and [general formatting instructions](https://link.springer.com/journal/13321/submission-guidelines) on 3 October 2026.

After a successful build, `main.pdf`, `technical_details.pdf` and `cover_letter.pdf` beside their sources link to the verified PDFs in `submission/`. There is one physical copy of each final PDF. Use the full build command above to establish both documents' cross-references.

See [review.md](review.md) for the scientific assessment and final preparation checklist. The local source rebuild uses TeX Live 2023; the journal's TeX Live 2021 server has not been tested.
