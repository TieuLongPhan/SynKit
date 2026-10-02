# Synister method experiments

This directory retains search benchmarks, independent correctness oracles,
ablations, input selection, reproducibility tools, and method evaluation.
Library implementation belongs under `synkit/Chem/Mapper/`; library regression
tests live in `Test/Chem/Mapper/{api,chem,exact,graph,io,slap,studies}`.

## Retained experiments

| Programs | Purpose |
| --- | --- |
| `emission_benchmark.py`, `reward_frontier_benchmark.py`, `sparse_reward_diagnostic.py` | Measure output emission and experimental suffix/lower-bound strategies |
| `propagation_comparison.py`, `propagation_serial_timing.py`, `propagation_external_cohort.py` | Compare frozen search implementations on matched inputs |
| `compare_separator_spectrum_cohort.py`, `compare_suffix_orbit_cohort.py` | Compare separator and orbit-filtering experiments |
| `enumeration_benchmark.py`, `repeat_enumeration.py`, `binary_backend_benchmark.py`, `output_scaling.py` | Compare complete mapping sets, backends, repeat timings, and output size |
| `ablation_benchmark.py`, `isolated_ablation_benchmark.py`, `validate_ablations.py`, `validate_isolated_ablations.py` | Measure component contributions and verify interventions against literal sets |
| `seed_output_benchmark.py`, `seed_output_recovery.py`, `classify_seed_representatives.py` | Measure seed/output effects and resume frozen experiments without replacing outcomes |
| `worked_oracle.py`, `all_distance_oracle.py`, `global_milp.py`, `structural_oracle.py`, `hydrogen_oracle.py` | Independently check mappings, objective values, hydrogen handling, and ITS equality |
| `mapping_landscape.py`, `classify_enumeration.py`, `audit_two_sided_its.py`, `audit_search_contract.py`, `trace_exact_search.py` | Validate mapping/orbit counts, search contracts, and observed pruning decisions |
| `development.py`, `select_development.py`, `select_confirmation.py`, `select_rhea.py`, `confirmation_contract.py`, `replication_contract.py` | Select input cohorts and lock reproducible evaluation contracts |
| `run_binary_sensitivity.py`, `run_representation_sensitivity.py`, `replay_annotations.py` | Check objective/representation sensitivity and frozen method evaluation |

Companion `audit_*` programs independently check saved results. Workers isolate
searches and resource limits; `report_*`, `resource_report.py`,
`storage_report.py`, `closure_profile.py`, and `revision_diagnostics.py` retain
method-result accounting. `benchmark_artifacts.py` freezes source and environment
metadata. `benchmark_inputs.py` supplies shared input-only selection and the
checked worked-example record, without importing application or plotting code.

## Output and reproducibility

Historical tracked campaigns and frozen source snapshots live under
`benchmark_results/`, relocated from the repository root. Paths recorded inside
historical evidence describe the original execution environment.

Use new output directories under `runs/`. Preserve existing run records and
frozen source snapshots. Local environments and `runs/` are ignored by Git.
Manuscripts, datasets/evidence under `paper/`, and cleanup archives are also
local and ignored; supply the original inputs when reproducing dataset studies.

Paired propagation comparisons save durable `.solver.json` reports before
output publication. Separate validation processes save `.validation.json`, and
`.output.json` binds the map file by checksum. Minimum proof, complete shell
output, and completed validation are separate outcomes. The independent auditor
checks frozen sources, seed provenance, stage artifacts, map validity, and
comparable output sets. Focused replays accept `--cases`; `--solver-source`
selects a historical frozen `synkit/` tree.

The emission improvement is enabled in ordinary Python PABS. Reward-frontier
suffixes and early suffix-orbit pruning remain experimental and disabled by
default. Reviewed local outputs include `runs/emission_full_100_v1/`,
`runs/emission_capped_pilot_v2/`, and `runs/reward_frontier_development_v2/`.

## Validation

Run from the repository root in an environment with project dependencies:

```sh
python -m pytest Test/Chem/Mapper -q
python -m pytest Experiment/Synister/tests/test_benchmark_artifacts.py -q
```

The `synister-contracts` CI job in `.github/workflows/test-and-lint.yml` lists
portable enumeration/evaluation checks. The full experiment suite also includes
replays requiring original local evidence under `paper/synister/evidence/`:

```sh
Experiment/Synister/.venv-confirmation-reproduction/bin/python -m pytest Experiment/Synister/tests -q
```

The user entry point is `synkit.Chem.Mapper.map_reaction`; graph inputs use
`enumerate_pabs_mappings`. Both support `backend="python"` or `backend="cpp"`.
See the [mapping interface documentation](../../doc/chem.rst) for objectives,
hydrogen modes, unbalanced reactions, and backend limits. Build the optional
native implementation explicitly with:

```sh
python -m synkit.Chem.Mapper.exact.native_build --output-dir /tmp/synkit-native
```

## Local cleanup archive

Application studies (property stability and template transfer), manuscript
figures, presentation checks, and historical figure replay scripts were moved
to `paper/local_cleanup_2026-10-02/Experiment/Synister/`. Old root task/review
notes were archived alongside them. The archive's `manifest.json` records every
original path and SHA-256, including original versions of files whose shared
method helpers or tests were separated during cleanup. Restore archived files
to their original paths before using their original reproduction commands.
