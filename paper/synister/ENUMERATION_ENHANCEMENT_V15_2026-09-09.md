# Final enumeration enhancement — exact results and output cost

Date: 2026-09-09. Final validation is complete.

## What changed

The [V14 report](ENUMERATION_ENHANCEMENT_V14_2026-09-09.md) describes the search,
proof and classification improvements, with a full frozen 1,200-task rerun.
The final change only affects the optional JSON path: immutable ITS/template
class-count tuples can be serialized directly, avoiding hundreds of thousands
of temporary list allocations on the largest output.

GlobalShellAnalysisResult.as_dict(copy_sequences=False) selects this path.
The default remains mutable lists with the original behavior. Both lists and
tuples encode to identical JSON arrays. The benchmark exposes this explicitly as
--immutable-json; it is not silently substituted into the recorded cohort protocol.

No enumeration, pruning, canonical certificate, class count or weight changes
between V14 and this formatter-only version. Tests require byte-identical public
JSON and preserve default mutation independence. Every result from the completed
cohort is additionally checked for byte-identical JSON under both paths; those
saved outputs are audit data, not inputs to any search.

## Validation scopes

- V14 performed all 1,200 searches from scratch on one frozen implementation and
  one case configuration, in two disjoint CPU partitions. Its measured timings
  remain unchanged, including any full-output overrun.
- Final formatter checks start a fresh process for each case and execute the whole
  search/classification/output path. Three repetitions each cover 17156_minimal,
  30474_reference_cd and 22361_reference_cd.
- The whole 1,200-task search was NOT rerun after the formatter-only change.
  The all-cohort JSON-equivalence check proves output identity, not new runtimes.
- Pattern cache remains disabled, with 16 workers per case, a strict 60-second
  wall configuration, a 1,000,000 worker-local retained-class cap and 4 GiB address
  space per worker. The parent has no aggregate process-group limit.
- Compilation is portable C++17 -O3, with no PGO or native-architecture flags.
- Imports and dataset loading precede each case clock. Case construction, proof,
  worker startup, enumeration, classification, formatting, JSON serialization and
  file close are timed. Close does not imply fsync.
- A freed physical CPU partition runs the fresh checks while the other partition
  may still be finishing the cohort. No exclusive-host timing guarantee is claimed.
- 330 main regression tests passed before this output-only change; all 11 focused
  JSON/API/minimum tests passed afterward. The unchanged C++ passed 98 UBSan tests.
  One optional PuLP skip and one historical provenance exclusion are retained.

## Results

The full frozen cohort completed **1200/1,200 searches** and **1200/1,200 structure classifications**. Exact completed-result comparisons: 1200; observable mismatches: 0; classification regressions: 0.

Strict full-output completion below 60 seconds: **1199/1,200**. Recorded completed overruns: [{"source_line": 30474, "mode": "reference_cd", "end_to_end_wall_seconds": 60.513289794005686}]. These timings are not replaced by retries.

Final formatter, three fresh-process executions per case:

| Case | Repeat 1 | Repeat 2 | Repeat 3 | All search/classification/output below 60 s |
|---|---:|---:|---:|---|
| 17156 | 9.473 s | 9.296 s | 9.358 s | True |
| 30474 | 57.452 s | 56.281 s | 58.970 s | True |
| 22361 | 40.425 s | 34.239 s | 39.768 s | True |

Fresh-run observable mismatches: {}.

All **1,200 public result JSON payloads were byte-identical** under the default and immutable-sequence serializers. AST comparison confirms identical default reporting paths and byte-identical remaining Python/C++ implementation files between the cohort and final snapshots.

The two-partition cohort took 33.36 minutes. Summed per-case full-output time was 65.00 minutes and overlaps across partitions.

Cohort source SHA-256: 1a518db731120d9ecb90cd066132a8ae2f643156a239d31583964d186e5c3bf8.

Final source SHA-256: 564fe6e452a3904cf81a148218b81a1c43323bd933853fe2d9410b7a67cdef49.

Portable library SHA-256: 696b9e296531a1f590d57906dab96f978f06fefadd715973d7ddc8add13defa6.

[Complete cohort records and timing origins](../../benchmark_results/synister_enhancement_v14_20260909/cohort_1200), [final machine-readable summary](../../benchmark_results/synister_enhancement_v15_20260909/final_summary.json), [all-result serialization equivalence](../../benchmark_results/synister_enhancement_v15_20260909/serialization_equivalence.json).

Measured fresh-run success is not a universal runtime guarantee. The 50–55 second engineering margin and the deferred assignment warm-start/packed-batch proposals remain future performance gates if the reported ranges exceed that aim.
