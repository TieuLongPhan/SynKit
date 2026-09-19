# Enumeration enhancement V14 — exact lower-bound witnesses

Date: 2026-09-09. The corrected full-cohort rerun has finished; fresh-process checks are reported separately in V15.

## Executed enhancements

This version combines the [V11 implementation](ENUMERATION_ENHANCEMENT_V11_2026-09-09.md)
with a direct native minimum-proof fast path and the bounded fragment-slice correction
tested in [V13](ENUMERATION_ENHANCEMENT_V13_2026-09-09.md).

- Exact packed reference witnesses replace hash-only membership decisions.
- The Python structure observer can explicitly use a native certificate-only backend.
  Canonical proof remains mandatory; only an unused group-order computation is skipped.
- Both minimum and reference-CD modes use the native fixed-target enumerator under one
  shared case deadline. Minimum mode must establish an optimum before publishing a shell.
- Zero-slack LAP filtering uses zero-reduced-cost strongly connected components instead
  of weighted all-pairs paths. Positive-slack calculations retain the existing algorithm.
- Per-case CPU, output bytes and memory high-water marks are recorded.
- A separate canonical-augmentation prototype passed explicit small-group coverage and
  uniqueness checks at every partial-matching depth. It is not production pruning.
- Parent-dual warm starts and fully contiguous batch transport remain undeployed ideas.
  This report does not claim every exploratory proposal was implemented.

## Native minimum proof without a heuristic seed

For each compatible atom pair, the existing atom-profile cost bounds its contribution
under every full mapping. Let C be that relaxation matrix, and let

    L = min over atom-compatible assignments pi of sum_i C[i, pi(i)].

Every full chemical distance D(m) is at least L. The new path spends at most a
one-second cooperative budget, including profile construction and native setup,
looking for a feasible mapping in the exact shell D(m)=L. It receives neither
the reference mapping nor its distance as a search target.

Before accepting a witness, the code verifies its bijectivity, atom compatibility
and full chemical distance. If D(m)=L, then L <= min D <= D(m)=L: the optimum is
proved. A stopped probe is not a complete shell. The subsequent native enumeration
must independently finish the entire optimum shell before the result is complete.

If the probe does not find a witness, it makes NO optimality claim. An empty shell,
a time limit or a lower bound that is not attained all fall back to the existing
reference-free seed plus exact assignment optimizer. The unchanged outer 60-second
wall deadline covers both paths. Probe and proof statistics are distinct in the
public result, and the probe retains its witness for replay.

This uses the existing checked profile relaxation and assignment solver. Native
input validation requires symmetric half-integer matrices within bounded numeric
ranges, so the relevant objective/profile values are exactly representable dyadics.
The lower-bound diagnostic for 17156 found a cost-9 witness in 0.338 seconds, visiting
228 nodes, without running the heuristic seed constructor.

## Seed fallback correction and preserved failures

The V11 and V12 full cohorts both completed 1,199 tasks and timed out on 17156_minimal.
A seed-only replay of the preceding 288 tasks reproduced a cost-25 seed: both tiny MCS
slices canceled before accepting a fragment. Starting the slice at the first progress
callback recovered a cost-9 seed after the same prefix.

The fallback fragment heuristic now uses an overall 0.5-second process-CPU cover budget,
including setup. Each 25 ms search slice begins at the first progress callback and is
capped by that overall deadline. The existing native MCS timeout and bounded polishing
iterations remain. Every candidate objective is recomputed; heuristic anchors never
restrict exact enumeration.

The V13 full-cohort attempt was stopped after 378 closed pilot records to integrate the
stronger proved minimum path before the final freeze. Its outputs are retained as an
interrupted pilot, not reported as a completed 1,200-task run. No earlier record is
substituted into this final cohort.

## Validation and uniform case protocol

- 330 tests passed, one optional PuLP skip and one pre-existing historical provenance
  test excluded. Historical evidence was not rewritten.
- New tests require the attained-bound path to work without invoking the heuristic.
  A regular-graph counterexample with bound zero but positive minimum verifies fallback.
  Raw-permutation minimum/count oracles and deadline failure tests also pass.
- The unchanged native C++ source passed all 98 UBSan tests.
- A new frozen source reruns all original 1,200 tasks: 751 minimum and 449 reference-CD.
  Two disjoint 600-task partitions use CPUs 0–15 and 16–31.
- Each case has 16 fresh workers, 8,192-node slices, a cap of 1,000,000 retained
  worker-local double-orbit records and a 4 GiB worker address-space limit. Parent
  memory is not subject to an aggregate process-group cap.
- The 60-second case wall deadline starts before construction; full-output timing
  includes proof, worker startup, search, classification, formatting, JSON serialization
  and file close. Imports/dataset loading precede that clock; close does not imply fsync.
  Cooperative overruns are retained and do not pass the strict acceptance gate.
- Pattern cache is disabled. Pure encoding memoization is not stored-answer reuse;
  all internal class/reference membership checks compare exact payloads.
- The first freed partition runs three fresh-process checks each of 17156_minimal,
  30474_reference_cd and 22361_reference_cd. Another partition may still be running.
  No exclusive-host timing claim is made.
- Completed results are compared against saved complete references in exact labeled
  units, rational coordinate frequencies and full class maps. The independently checked
  Python classification supplies the formerly missing 13067 structure reference.
  Saved outputs are audit inputs only, never inputs to enumeration.

## Results

| Mode | Tasks | Search complete | Structure complete | Both complete below 60 s |
|---|---:|---:|---:|---:|
| minimal | 751 | 751 | 751 | 751 |
| reference_cd | 449 | 449 | 449 | 448 |

Reference comparisons: 1200; observable differences: 0; classification regressions: 0. Pattern-cache hits: 0.

Comparison coverage: {"target_minimum_reference_fields": 1200, "labeled_solution_count": 1200, "exact_rational_coordinate_frequencies": 1200, "full_class_maps_in_labeled_units": 1200, "independent_13067_classification_check": 1}.

Fresh-process repetitions use the final formatter and are reported in [V15](ENUMERATION_ENHANCEMENT_V15_2026-09-09.md).

Incomplete/error records in the corrected cohort: [].

Exact comparison failures: [].

Completed searches exceeding the strict full-output budget: [{"source_line": 30474, "mode": "reference_cd", "end_to_end_wall_seconds": 60.513289794005686}].

Two-partition cohort wall time: 33.36 minutes; summed per-case full-output times: 65.00 minutes. The latter overlap and are not campaign elapsed time.

Frozen Python/C++ source SHA-256: 1a518db731120d9ecb90cd066132a8ae2f643156a239d31583964d186e5c3bf8.

Portable library SHA-256: 696b9e296531a1f590d57906dab96f978f06fefadd715973d7ddc8add13defa6.

Current package matches the final frozen source: False. [Full records and origins](../../benchmark_results/synister_enhancement_v14_20260909/cohort_1200), [machine-readable summary](../../benchmark_results/synister_enhancement_v14_20260909/final_summary.json).

The earlier V11 and V12 full cohorts and the interrupted V13 pilot remain separately reported; none of their records were substituted into this full run. Final formatter-only checks are reported separately in V15. Runtime observations do not imply a guarantee under arbitrary host contention. The broader warm-start and packed-batch proposals remain undeployed; the explicit canonical-augmentation prototype remains separate from production.
