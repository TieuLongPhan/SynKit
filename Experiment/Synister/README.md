# Synister method experiments

This directory retains search benchmarks, independent correctness oracles,
ablations, input selection, reproducibility tools, and method evaluation.
Paper-dependent checks live in `tests/test_historical_evidence.py` and are run
explicitly with local evidence; they are excluded from library CI.
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

Historical campaigns and frozen source snapshots may be retained locally under
`benchmark_results/`, relocated from the repository root. This directory is
ignored by Git. Paths recorded inside historical evidence describe the original
execution environment.

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

## Global mapping-shell campaigns

All Synister implementation and experiment code lives in SynKit. Run the
reference-blinded global experiment with an explicit frozen dataset:

```bash
python Experiment/Synister/run_global_shells.py \
  --dataset /path/to/flower_test_10000.csv.gz \
  --output Experiment/Synister/benchmark_results/synister_global_shells_v4 \
  --mode both \
  --workers 1 \
  --time-limit-per-shell 300 \
  --memory-limit-gib 6
```

The input CSV must contain `source_line`, `reaction_id`, and `mapped_reaction`.
The runner uses one numerical thread per worker. Parallel workers only compute;
the parent process is the sole writer of immutable gzip JSON case records. It
independently blinds the endpoint atom orders, runs either the unrestricted
shell at the held-out reference CD, the unrestricted globally minimal shell,
or both, and reveals the reference only for post-search evaluation. Timeouts,
caps, and errors are never counted as complete cases.

`--memory-limit-gib` is an address-space limit for each worker, not an aggregate
limit. On a 16-worker workstation, use the cgroup-based launcher to enforce an
64 GiB hard limit for the complete service while preventing it
from consuming swap:

```bash
Experiment/Synister/start_global_shells_16cpu.sh \
  /path/to/flower_test_10000.csv.gz \
  Experiment/Synister/benchmark_results/synister_global_shells_v4_60s_w16
```

The launcher uses 16 workers, 60 seconds per shell, a 4 GiB per-worker
address-space ceiling, `MemoryHigh=infinity`, `MemoryMax=64G`, and
`MemorySwapMax=0`. If a worker is killed by the cgroup, the pool stops instead
of silently converting the remaining campaign into error records. Existing
atomic case records remain resumable with the identical command.

Verify the case and manifest digests and regenerate censored,
manuscript-facing statistics with:

```bash
python Experiment/Synister/summarize_evidence.py \
  Experiment/Synister/benchmark_results/synister_global_shells_v4 \
  --output derived_findings.json
```

## Exact alternative-ITS application

Generate one deterministic AAM representative for every exact ITS class that
differs from a supplied mapped reference:

```bash
python Experiment/Synister/run_alternative_its.py \
  --dataset /path/to/flower_test_10000.csv.gz \
  --source-line 109 \
  --output Experiment/Synister/benchmark_results/alternative_its_line_109.json \
  --target minimal \
  --target reference \
  --target 10 \
  --seed-mode reference \
  --memory-limit-gib 6 \
  --time-limit 300
```

``reference`` uses the scalar CD of the mapped input. ``minimal`` proves and
enumerates the global optimum; any non-negative number requests that exact
global shell, whether or not it equals the reference CD. A reference seed is
an incumbent and ordering hint only. The output includes exact completeness,
stable ITS identifiers, representative mappings, and correspondences in the
input atom-map coordinates. These are reference-relative structural
alternatives, not claims of mechanistic invalidity.

## Mapping experiment figures

Regenerate the vector pilot and CD-spectrum panel directly from the frozen,
digest-verified evidence:

```bash
python Experiment/Synister/plot_figures.py
```

The plotting script validates the 100-case cohort identity, record 84:1, and
seed-invariant exact shell counts before writing
`paper/synister/figures/pilot_spectrum.pdf`. Its palette matches the TikZ
workflow and the shared `../Style` visual vocabulary.

The figure and gallery programs require the local `paper/synister/` inputs.
The gallery also requires a TeX installation and the optional PDF dependency:

```bash
python -m pip install -r Experiment/Synister/requirements.txt
```

## Local cleanup archive

Application studies (property stability and template transfer), manuscript
figures, presentation checks, and historical figure replay scripts were moved
to `paper/local_cleanup_2026-10-02/Experiment/Synister/`. Old root task/review
notes were archived alongside them. The archive's `manifest.json` records every
original path and SHA-256, including original versions of files whose shared
method helpers or tests were separated during cleanup. Restore archived files
to their original paths before using their original reproduction commands.
