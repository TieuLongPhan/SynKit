# Enumeration enhancement V12 — CPU-budgeted seed construction

Date: 2026-09-09. The corrected full-cohort rerun and fresh-process checks have finished.

## Why a second frozen run is necessary

The [V11 execution](ENUMERATION_ENHANCEMENT_V11_2026-09-09.md) implements exact
reference witnesses, both target modes, native certificate-only structure
classification and the zero-slack alternating-cycle specialization. Its full
cohort exposed a minimum-proof timeout on 17156_minimal. This failure is retained
as a measured regression; a successful retry is not substituted into that run.

The fragment heuristic used 25-millisecond wall-clock MCS slices and a shared
0.5-second wall-clock cover budget. An isolated experiment showed one wall-clock
attempt returning no initial fragment while the three CPU-clock attempts produced
the full fragment cover. All six seed attempts eventually reached cost 9 through
repair, so this experiment demonstrates timing sensitivity but does not prove
the precise seed path taken by the failed cohort case: that exception record
did not retain seed statistics.

## Change and correctness argument

Only the optional fragment heuristic changes from V11: its cooperative slice and
cover clocks now use time.process_time(). Its statistics explicitly identify the
budget clock as process_cpu. Scheduler descheduling no longer consumes these CPU
slices. This does not make the heuristic deterministic across processors or RDKit
versions.

The fragment cover supplies only a complete atom-compatible permutation. Its full
objective is recomputed before accepting any improvement. Its anchors are never
passed as restrictions to exact search. Therefore the clock affects which feasible
upper bound is found, not the domain of mappings or the validity of lower bounds.
Minimum mode still needs a complete optimality proof before enumerating its shell.
The outer absolute wall-clock deadline and strict output-completion audit remain
60 seconds. CPU-budgeted heuristic work cannot confer proof completeness or change
that acceptance threshold.

The native C++ implementation and portable library are unchanged from V11.
No stored dataset answer or reference mapping supplies a seed or pruning decision.
Pattern-cache use remains disabled.

## Validation and protocol

- 326 tests passed, one optional PuLP skip, one pre-existing historical provenance
  test excluded. The historical evidence was not rewritten.
- The changed clock's progress callback is tested independently of scheduler wall
  time; existing feasible-seed and complete-analysis tests pass.
- V11's unchanged native C++ passed all 98 UBSan tests.
- The corrected 17156 one-worker diagnostic completed search and full structure
  classification in 7.341 seconds. This is a diagnostic, not the 16-worker cohort.
- A new source snapshot is used for all 1,200 tasks, split into two disjoint
  600-task selections. There is no reused V11 result in this corrected cohort.
- Each case has 16 fresh workers, a 60-second wall deadline, 8,192-node slices,
  a cap of 1,000,000 retained worker-local double-orbit records, and 4 GiB per
  worker address space. Parent memory has no aggregate process-group limit.
- CPU partitions 0–15 and 16–31 allow two cases concurrently, up to 32 physical
  cores across the campaign. Case timing includes construction, worker startup,
  proof, enumeration, classification, formatting, JSON serialization and close.
  Imports and dataset loading precede each case clock. Close does not imply fsync.
- After either partition finishes, its freed CPUs run three fresh-process checks each
  of 17156_minimal, 30474_reference_cd and 22361_reference_cd. The other partition
  may still be running. No exclusive-host timing guarantee is asserted.
- All results are compared against complete saved outputs using exact labeled
  counts, rational coordinate frequencies and full class maps where available.
  The independently verified 13067 classification supplies its missing prior
  full-structure reference. Public IDs remain SHA-based labels; internal equality
  and reference witnesses use exact payloads.

## Results

| Mode | Tasks | Search complete | Structure complete | Both complete below 60 s |
|---|---:|---:|---:|---:|
| minimal | 751 | 750 | 750 | 750 |
| reference_cd | 449 | 449 | 449 | 449 |

Reference comparisons: 1199; observable differences: 0; classification regressions: 0. Pattern-cache hits: 0.

Comparison coverage: {"target_minimum_reference_fields": 1199, "labeled_solution_count": 1199, "exact_rational_coordinate_frequencies": 1199, "full_class_maps_in_labeled_units": 1199, "independent_13067_classification_check": 1}.

| Case | Repeat 1 full output | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|
| 17156 | 14.313 s | 14.139 s | 14.056 s |
| 30474 | 59.612 s | 58.363 s | 59.067 s |
| 22361 | 36.834 s | 35.849 s | 41.558 s |

Repeat audit failures: {}.

Incomplete/error records in the corrected cohort: [{"key": [17156, "minimal"], "error": {"type": "TimeoutError", "message": "minimum proof incomplete; no provisional shell enumerated"}, "reason": null}].

Exact comparison failures: [].

Two-partition cohort wall time: 36.33 minutes; summed per-case full-output times: 70.11 minutes. The latter overlap and are not campaign elapsed time.

Frozen Python/C++ source SHA-256: 575f7c02a8fc419822f36c401200adff0b93015ef7421d2f5a22810b10ae04ad.

Portable library SHA-256: 696b9e296531a1f590d57906dab96f978f06fefadd715973d7ddc8add13defa6.

Current package matches this older frozen source: False. [Full records and origins](../../benchmark_results/synister_enhancement_v12_20260909/cohort_1200), [machine-readable summary](../../benchmark_results/synister_enhancement_v12_20260909/final_summary.json).

The initial V11 cohort remains separately reported, including its minimum-proof timeout; none of its records were substituted into this corrected run. Runtime observations do not imply a guarantee under arbitrary host contention. The broader warm-start and packed-batch proposals remain undeployed; the explicit canonical-augmentation prototype remains separate from production.
