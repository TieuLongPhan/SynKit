# CRN experiments

Reproducible validation, scaling, biochemical case studies and Python/C++
comparisons using the public `synkit.CRN` APIs. Run commands from the repository
root. This directory contains the inputs and expected scientific outcomes.
Document layout and LaTeX table generation belong with the documents that
consume these results.

## Run and verify

Install the project and test dependencies with `python -m pip install -e '.[test]'`.
The default C++ backend needs a C++17 compiler and Boost headers (for example,
`libboost-dev` on Ubuntu). Compilation occurs when a study is invoked. Pass
`--native-library /path/to/library.so` to reuse an explicit build, or
`--native-build-dir /path/to/build` to choose the build directory.

```bash
# Reduced scaling and paired-timing workloads; all local controls still run.
python Experiment/CRN/run_all.py --quick --offline \
  --output-dir Experiment/CRN/results/quick \
  --summary Experiment/CRN/results/quick/summary.json
python Experiment/CRN/tools/check_reports.py Experiment/CRN/results/quick/summary.json

# Default workload, plus the optional BioModels and external comparisons.
python Experiment/CRN/run_all.py --with-biomodels --offline \
  --crnt4sbml-python /path/to/crnt4sbml/bin/python \
  --output-dir Experiment/CRN/results/full \
  --summary Experiment/CRN/results/full/summary.json
python Experiment/CRN/tools/check_reports.py Experiment/CRN/results/full/summary.json --require-all
```

Omit `--offline` to download missing BioModels files. The eleven expected XML
checksums are pinned in `data/biomodels_manifest.json`. Cached or downloaded
files with different checksums fail. XML files are excluded from distributions.
The optional CRNT4SBML comparison uses a separate Python 3.7 environment with
`environments/crnt4sbml-requirements.txt`. It must not share SynKit's interpreter.

| Study | Default workload | Inputs |
| --- | --- | --- |
| Validation | 12 reference networks and independent checks | Built-in fixtures |
| Scaling | Sizes 50, 100, 200, 400; three families; five tasks | Fixed-seed generators |
| KEGG | Four modules, explicit currency removal | `data/kegg_modules.json` |
| Formose | Expansion depths 1–4; depth-4 structural analysis | Local rules and seeds |
| Representation | Catalyst/currency controls; paired sizes 10, 14, 18, 20 with five repeats | Cached KEGG and generators |
| Supplementary | Conservation, symmetry, cofactor and formose route controls | `data/glycolysis_routes.json` and local rules |
| Glycolysis donors | Six prescribed-route inventory controls | `data/glycolysis_routes.json` |
| BioModels (optional) | Eleven curated inputs, including a reaction-free negative control | Local XML cache or download |
| CRNT4SBML (optional) | Shared structural quantities on the validation set | Isolated external interpreter |

`--backend python` selects reference implementations. The default run includes
seven studies. Both optional comparisons bring the total to nine. Native kernel
regressions run through `pytest` under `Test/CRN`, sharing one native build.
Reports use neutral study names and schemas:
`crn-representation.json`, `crn-supplementary.json`, and `crn-glycolysis-donors.json`,
each declaring `synkit.crn-<name>/1`.

The run summary (`synkit.crn-experiments/2`) records its workload profile,
omitted studies, environment, source hashes and each report's SHA-256. `complete`
covers the requested workload. `all_studies_included` records whether all nine
studies ran. A study exception is saved as a failure and remaining studies run.
The verifier rejects missing, changed, failed or incomplete reports.
`--require-all` also rejects quick profiles and omitted studies.

Scientific quantities and certificate outcomes have fixed expectations. Timings
are new measurements on the recorded machine and need not match historical
values. A quick run verifies execution on smaller workloads. Generated reports
and executed notebooks are ignored by Git.

## Benchmark tools

`benchmarks/scaling.py` supplies fixed-seed generators and isolated timing
workers. `studies/scaling.py` exposes sizes, families, tasks, repeats and budgets.
`benchmarks/backends.py` compares Python and C++ on identical inputs, alternating
order and retaining raw samples, completion status and output digests:

```bash
python Experiment/CRN/benchmarks/backends.py \
  --build-native /tmp/synkit-crn-native --repeats 3 --budget 5 \
  --tasks siphons traps p_semiflows t_semiflows \
  --modules M00004 --synthetic-sizes 10 20 \
  --output Experiment/CRN/results/backends.json
```

Use `--biomodels-dir Experiment/CRN/data/biomodels` to include locally cached
SBML inputs. This tool never downloads. `--modules M00001` selects one cached
KEGG module. An empty `--modules` selection permits a synthetic-only sweep.
Omit `--modules` to sweep all four modules. The Python structural enumerator can
reach its node cap on the larger currency-retaining networks. These cases remain
incomplete and receive no speedup estimate.
Incomplete measurements fail the report. A speedup is emitted only when every
repeat of both backends completes and their outputs agree. Preparation and
witness checks are included. Worker startup is recorded separately.

```bash
python -m pytest Test/CRN -q
python -m nbconvert --to notebook --execute \
  Experiment/CRN/notebooks/crn_glycolysis_case_study.ipynb \
  --output /tmp/crn-glycolysis-executed.ipynb --ExecutePreprocessor.timeout=180
```

The manually triggered `.github/workflows/crn-reproducibility.yml` runs the local
quick profile, verifies saved reports, runs CRN tests, executes the notebook and
uploads evidence. The optional external comparisons require their declared
inputs and environments and are run separately.

Regression test cases belong only in `Test/CRN`. Do not add parallel test
runners to this directory. Study checks describe scientific outcomes and report
completeness, and benchmark inputs are shared rather than copied into tests.

`studies/` contains report-producing analyses, `benchmarks/` contains fixtures
and timing code, and `tools/` contains verification and cache maintenance.
These experiments and their small input datasets ship in the source
distribution. The runtime wheel contains the public analysis package.
