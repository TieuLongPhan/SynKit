# SynKit experiments

This directory contains reproducible studies and validation programs built on
the public SynKit APIs. Run commands are documented below or with each study.

## Experiment index

- [CRN](CRN/README.md): chemical reaction network validation, scaling, and the
  KEGG, formose, and BioModels case studies.
- [Lewis](Lewis/README.md): Lewis-state conversion and rule-replay studies.

Only CRN and Lewis are currently included in the public repository. Other
experiment directories remain local and ignored by Git until release.

Run the CRN experiments from the repository root:

```bash
python Experiment/CRN/run_all.py \
  --with-biomodels --offline --output-dir Experiment/CRN/results \
  --summary Experiment/CRN/results/summary.json
```

Add `--crnt4sbml-python /path/to/isolated/python` for the external comparison.
Use `--quick` for a small execution check. Individual experiments are in
`CRN/studies/`, shared benchmark code in `CRN/benchmarks/`, and report
verification and cache maintenance in `CRN/tools/`. See the
[CRN instructions](CRN/README.md) for complete workloads and report verification.

Generated outputs should remain reproducible from tracked code and documented
inputs. Large or restricted source datasets are not redistributed unless their
license permits it. Local READMEs record their sources and terms of use.
