# Exact enumeration enhancement V20

Date: 2026-09-09. All 1,200 original tasks have been rerun from scratch.

## Results

Search completion: **1200/1,200**. Structure completion:
**1200/1,200**. Both complete below 60 seconds:
**1200/1,200**.

Completed exact-result comparisons: **1200**; observable
mismatches: **0**. Comparisons include minimum/reference
fields, labeled counts, exact rational coordinate frequencies and full ITS/template
class maps in labeled units. The independent Python classification control for
line 13067 is also replayed against the new output.

| Mode | Tasks | Search complete | Structure complete | Complete below 60 s |
|---|---:|---:|---:|---:|
| minimal | 751 | 751 | 751 | 751 |
| reference_cd | 449 | 449 | 449 | 449 |

Incomplete outcomes: []

Full-output overruns: []

Slowest complete cohort cases:

| Line | Mode | Full output seconds |
|---|---|---:|
| 30474 | reference_cd | 53.213 |
| 22361 | reference_cd | 37.539 |
| 13071 | reference_cd | 14.172 |

The two-partition campaign took **33.76 minutes**.
Summed per-case full-output time was 64.66
minutes; case intervals overlap across the two partitions.

## Retained changes

- Reuse parent assignment potentials and matching hints only for larger residual
  problems. Repair every current row/column dual constraint, retain only finite
  disjoint tight matching edges, and finish the exact Hungarian solve. Small
  residuals retain the original fresh row-minimum reduction.
- Stop the preliminary singleton-domain test after finding its second allowed
  image. The complete assignment and forced-edge domain checks are unchanged.
- Construct the identical canonical leaf certificate from sorted present edges,
  and use exact factorial group orders for verified local twin factors.
- Reuse instance-local native scratch buffers and ctypes pointers. A bounded
  template palette cache binds full exact token tuples and reuses encoding tables.
  Published certificate bytes remain immutable.
- Read the fixed deadline once per worker slice instead of locking at each
  candidate. Every callback still checks it; the global mapping counter retains
  its shared lock.

The [mathematical review](../../benchmark_results/synister_enhancement_v20_20260909/MATHEMATICAL_REVIEW.md)
gives the invariant and exactness argument for each change. No mapping domain,
chemical objective, class equivalence, multiplicity or public identifier changed.
The broader contiguous result-batch redesign remains deferred; measurements
directed this change toward assignment and classification overhead.

## Correctness and technical checks

- 350 regression tests passed; one optional dependency test skipped.
- 105 native tests passed under undefined-behavior sanitization with recovery
  disabled.
- 200 independently solved assignment problems and 6,400 independently solved
  forced edges passed, including large-cost objectives above 50 million.
- New controls exercise extreme stale hints, forbidden and duplicate matching
  hints, mixed twin/non-twin symmetry, component exchange, exact palette identity,
  cache eviction and immutable output after scratch reuse.
- Existing exhaustive mapping/orbit, forced-edge, resumable frontier, deadline,
  global cap, typed-color and hash-collision controls pass.
- Ruff correctness checks and git diff --check passed.

The historical provenance test remains excluded because its preserved artifact
is bound to an earlier implementation. The [baseline check](../../benchmark_results/synister_enhancement_v20_20260909/preexisting_provenance_failure.json)
shows that V15 and current hashes are identical for that test's source scope and
both differ from the historical artifact. The old artifact was not rewritten.

## Fresh sentinel repetitions

Each invocation starts fresh workers and uses the same frozen source, library,
60-second policy and exact comparison criteria. These repetitions never replace
a cohort record.

| Line | Mode | Repeat | Full output seconds | Complete below 60 s |
|---|---|---:|---:|---|
| 30474 | reference_cd | 1 | 54.025 | True |
| 17156 | minimal | 1 | 9.713 | True |
| 22361 | reference_cd | 1 | 34.628 | True |
| 30474 | reference_cd | 2 | 54.493 | True |
| 17156 | minimal | 2 | 9.668 | True |
| 22361 | reference_cd | 2 | 34.660 | True |
| 30474 | reference_cd | 3 | 54.072 | True |
| 17156 | minimal | 3 | 9.547 | True |
| 22361 | reference_cd | 3 | 35.939 | True |

## Timing scope and limits

The clock covers case construction, proof, worker startup, enumeration,
classification, formatting, JSON serialization and file close. Imports and
dataset loading precede the clock; file close does not imply fsync. There are
16 workers per case, at most two concurrent cases on disjoint CPU partitions,
a 1,000,000 retained-worker-record cap, and 4 GiB address-space limit per worker.
The parent has no aggregate process-group memory cap. The native library uses
portable C++17 -O3 without PGO or architecture-specific flags. Pattern caching
remains disabled; exact encoding memoization is not stored-result reuse.

The host is shared. A contemporary 90-second diagnostic on line 30474 measured
the prior V15 baseline at 69.765 seconds and the intermediate V19 candidate at
67.528 seconds with exact parity. Neither is a V20 acceptance result. Earlier
historical V15 runs at 56-59 seconds do not establish current deadline compliance.
Microbenchmark improvements are not universal whole-case speed guarantees.

V16-V18 incomplete pilots and rejected prototypes are preserved separately.
They were not promoted as full-cohort runs. The recorded 60-second cohort
outcomes, including failures or overruns, are authoritative.

## Provenance

Source SHA-256: 52d0bb9f12fe408d312de7bd7891b7f9b71b65a8486d7ca8f5de20a66058aeaa

Library SHA-256: e44527c7f89131b3ccad970f3e0e5e1aaab4caaf2c65f9214c9aa936ab5648e6

Current package matches the frozen source: True.

[Machine-readable summary](../../benchmark_results/synister_enhancement_v20_20260909/final_summary.json),
[exact audits](../../benchmark_results/synister_enhancement_v20_20260909/audit_summary.json),
[original cohort outputs](../../benchmark_results/synister_enhancement_v20_20260909/cohort_1200).
