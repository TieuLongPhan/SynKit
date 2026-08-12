# MTG validation experiment

This experiment exercises native rule composition, occurrence processes, and
mechanistic transition graph derivation with bounded deterministic fixtures.
The output schema is `synkit.mtg-validation/1`.

## Run the validation

From the repository root, run:

```bash
python Experiment/MTG/validation.py --output mtg-validation.json
```

The command exits with status zero only when every contract passes. The report
records the Python and NetworkX versions, input digest, search limits, timing
and memory observations, individual checks, and curated case-study summaries.

The fixture checks finite-graph construction and replay properties. Timing and
memory values are operational measurements; they do not establish chemical
kinetics, energetics, yield, or mechanism preference.

## Render tabular and LaTeX assets

A passing evidence file can be converted to stable paper inputs:

```bash
python Experiment/MTG/paper_assets.py mtg-validation.json \
  --latex-output mtg-results.tex \
  --timing-output mtg-timings.tsv
```

`paper_assets.py` rejects evidence with an unsupported schema or a failing
status. The generated files should be regenerated whenever the validation
fixture, search limits, or runtime environment changes.

## Tests

The lightweight contracts are covered by:

```bash
python -m pytest Test/Benchmark/test_mtg_validation.py \
  Test/Benchmark/test_mtg_paper_assets.py
```
