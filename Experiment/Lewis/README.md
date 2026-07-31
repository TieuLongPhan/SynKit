# Lewis experiments

This package is the self-contained home for executable Lewis-structure
benchmarks and their retained data. Unit and regression tests remain under
`Test`.

- `Data` contains the shared input corpora and their provenance registry.
- `partial_expand`: minimal partial atom-mapping expansion and reproducible
  reaction-level runtime metadata for current LWG, RB1, RB2, and GM.
- `hydrogen_expand`: single-process comparison of legacy/new HExtend, with
  explicit GM/RB1/RB2 capability controls and Method A/B contract metadata.
- `rule_replay`: forward/backward graph-rule replay for the tuple and
  `typesGH` representations, with retained case evidence.
- `FLOWER`: the portable 425,517-reaction full-corpus replay payload and its
  resource-configurable bidirectional launcher.
- `mech_path`: reviewed MechanismBench partitions, corpus-reconstruction audit
  failures, and the evidence runner.

Shared dataset and serialization helpers live in `common.py`.

Run a new copy of one experiment, or all three, with:

```console
./Experiment/Lewis/run_experiments.sh partial_expand
./Experiment/Lewis/run_experiments.sh rule_replay
./Experiment/Lewis/run_experiments.sh mech_path
./Experiment/Lewis/run_experiments.sh all
```

By default, results go to a new timestamped directory under `Runs`, which is
ignored by Git. Use `--limit N --repetitions 1` for a quick pilot. Run
`./Experiment/Lewis/run_experiments.sh --help` for all options.
