# Lewis benchmark experiments

This directory contains executable Lewis-structure benchmarks and retained
benchmark data. Unit and regression tests are stored under `Test`.

## Layout

- `Data` contains the shared input corpora and their provenance registry.
- `partial_expand`: minimal partial atom-mapping expansion and reproducible
  reaction-level runtime metadata for current LWG, RB1, RB2, and GM.
- `hydrogen_expand`: single-process comparison of legacy/new HExtend against
  executable hydrogen-extension Method A/B references.
- `rule_replay`: forward/backward graph-rule replay for the tuple and
  `typesGH` representations, with retained case evidence.
- `FLOWER`: the portable 425,517-reaction full-corpus replay payload and its
  resource-configurable bidirectional launcher.
- `mech_path`: reviewed MechanismBench partitions, corpus-reconstruction audit
  failures, and the evidence runner.

Shared dataset and serialization helpers live in `common.py`.

## Benchmark launcher

The launcher covers the three maintained benchmark suites below. The
`hydrogen_expand` and `FLOWER` directories provide separate workflows.

```console
./Experiment/Lewis/run_experiments.sh partial_expand
./Experiment/Lewis/run_experiments.sh rule_replay
./Experiment/Lewis/run_experiments.sh mech_path
./Experiment/Lewis/run_experiments.sh all
```

Results are written to a timestamped directory under the Git-ignored `Runs`
directory. A small validation run can use `--limit N --repetitions 1`.
`./Experiment/Lewis/run_experiments.sh --help` lists all options.
