# CRN experiments

Reproducible studies behind the **SynKit CRN** manuscript (`paper/synkit_crn`,
targeting *BMC Bioinformatics* as a Software article). Each study emits a JSON
evidence file with a declared schema and exits non-zero when a contract fails,
so the whole set is usable as a gate.

## Studies

| Study | Script | Schema | Network access |
| :--- | :--- | :--- | :--- |
| Validation against reference networks | `validation.py` | `synkit.crn-validation/1` | none |
| Scaling of the analysis stack | `scaling.py` | `synkit.crn-scaling/1` | none |
| KEGG metabolic modules | `kegg_case_study.py` | `synkit.crn-kegg/1` | none (cached) |
| Rule-generated formose network | `formose_case_study.py` | `synkit.crn-formose/1` | none |
| BioModels interoperability and scale | `biomodels.py` | `synkit.crn-biomodels/1` | first run only |

## Run everything

From the repository root:

```bash
python Experiment/CRN/run_all.py --output-dir results/ --summary results/summary.json
```

Add `--with-biomodels` to include the BioModels probe (downloads on first run),
or `--quick` to shrink the scaling sweep to a few seconds.

Individual studies take their own options:

```bash
python Experiment/CRN/validation.py --output results/crn-validation.json
python Experiment/CRN/scaling.py --sizes 50 100 200 400 --output results/crn-scaling.json
python Experiment/CRN/kegg_case_study.py --output results/crn-kegg.json
python Experiment/CRN/formose_case_study.py --output results/crn-formose.json
python Experiment/CRN/biomodels.py --offline --output results/crn-biomodels.json
```

## Render manuscript tables

```bash
python Experiment/CRN/paper_assets.py results/ --output-dir paper/synkit_crn/tables/
```

`paper_assets.py` rejects evidence with an unsupported schema or a failing
status, and emits only tabular rows so captions and labels stay in the
manuscript. Regenerate whenever the validation set, the cached inputs, or the
runtime environment changes.

## Inputs and provenance

- **Validation networks** and **scaling generators** live inside the package
  (`synkit.CRN.Benchmark`) because the validation set also runs in the automated
  test suite. The studies here drive them and record evidence.
- **KEGG modules** (M00001, M00307, M00009, M00004) are cached inside the package
  at `synkit/CRN/Benchmark/data/kegg_modules.json`. Refresh deliberately with
  `refresh_kegg_cache.py`, then re-run the test suite --- several tests assert
  specific metabolites and conservation laws, so a KEGG update can legitimately
  change the manuscript's numbers.
- **BioModels files** are cached under `data/biomodels/` on first run and are
  *not* redistributed with SynKit; BioModels content carries its own terms of
  use. Use `--offline` to require the cache.

## What the evidence does and does not support

Every verdict produced here is **structural**: it holds for all positive rate
constants and says nothing about behaviour at any particular parametrization.
Timings are operational measurements on the recorded machine and locate practical
ceilings rather than asymptotic complexity. The Angeli–De Leenheer–Sontag
persistence test is sufficient but not necessary, so a negative verdict does not
prove that a species can be driven to extinction. The BioModels sweep is a
hand-picked sample and supports no claim about coverage of the repository.

## Tests

The underlying library contracts are covered by `Test/CRN/Benchmark/`, which runs
the validation set, the scaling harness and the KEGG case study as unit tests.
