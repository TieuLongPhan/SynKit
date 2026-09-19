# Enumeration enhancement V11 — implementation and validation

Date: 2026-09-09. The frozen full-cohort rerun has finished.

## Implemented changes and mathematical obligations

### P0: exact reference witnesses

Reference presence is now tested using injective packed permutations and losslessly
compressed transported adjacency/property payloads. SHA digests remain output
identifiers and traversal diagnostics; they no longer decide these membership
queries. For n <= 256, a permutation uses a format marker and exactly n bytes.
Larger permutations use a different marker and little-endian uint64 entries.
Permutations entering the observer have already been validated by the search.
Lossless compression cannot merge two different uncompressed payloads: applying
the decompressor to an equal compressed value recovers the same payload.

The existing internal attributes named mapping_hashes and transport_hashes now
hold exact witnesses. This naming is retained to limit unrelated interface edits.
Native full-class deduplication continues to compare complete certificate payloads;
a public SHA class ID never establishes internal class equality. No dataset answers,
reference mappings or historical result files are inputs to enumeration.

### P1a: explicit native structure certificates

GlobalShellConfig.structure_native_library_path selects a native certificate
backend for the existing Python structure observer. This backend computes each
certificate and has no saved-class cache. The native search still proves a
canonical certificate; only the unused group-order calculation is omitted.
Incompleteness or unsupported graph structure produces an incomplete classification,
not a fabricated certificate. Group orders used for native orbit weights remain
mandatory and are not skipped.

The previously incomplete 13067_minimal classification was cross-checked against
the Python canonicalizer with a 5-second *per-certificate* allowance. Both produced
identical full ITS/template spectra. That independent check is correctness evidence,
not the uniform benchmark configuration. The native certificate observer completed
in 6.63 seconds versus 10.38 seconds for this independent Python check.

### P1b: shared minimum-proof and native shell API

The explicit native API accepts target_mode='minimal' or 'reference_cd'. Minimum
mode uses the existing exact assignment optimizer with _optimization_only=True,
no mapping cap and no reference mapping. A feasible heuristic mapping is only an
upper bound. Only a complete optimization result supplies the numeric shell target.
Failure to prove the minimum raises an explicit TimeoutError and publishes no
provisional minimum shell. The reference mapping is queried after enumeration;
minimum-mode search never receives its distance as a target.

The optimization pass and native fixed-target enumeration share an absolute
monotonic deadline. The benchmark starts it before reaction construction and
measures through JSON serialization/file close. Cooperative cancellation may exceed
the deadline; such outputs do not pass the strict acceptance gate. This is not an
OS-enforced hard process kill. Public target remains the legacy string 'minimal';
minimum_cost carries the proven number. A preliminary cohort was interrupted when
the comparison audit detected this reporting mismatch, and the final cohort was
restarted from a new immutable snapshot.

The first shared implementation uses Python for minimum proof and native code
for shell enumeration. It does not claim a new native minimum optimizer.

### P2: zero-slack alternating-cycle specialization

At each residual LAP, let L be the optimal assignment value, M the matching,
and u,v a feasible optimal dual. Reduced costs cbar(i,j)=c(i,j)-u(i)-v(j) are
nonnegative. Forcing row i onto the column matched to row j has assignment value

    L + cbar(i,M(j)) + d(j,i),

where d is the shortest-path distance in the alternating residual row graph.
If the remaining exact-search budget equals L, a forced edge can survive only
if both terms after L are zero. A zero edge i->j and a zero return path j->i
exist exactly when the endpoints lie in the same strongly connected component
of zero-reduced-cost edges. Consequently this case needs SCCs, not weighted
all-pairs shortest paths. The implementation returns zero within a component
and INF between components for this query only. Positive-slack calls retain the
existing exact shortest-path calculation.

The specialization runs only after the Hungarian matching/dual certificate.
It neither reuses a stale dual nor assumes a heuristic assignment is optimal.
Signed/half-integer input weights remain covered because residual costs are
nonnegative absolute differences. Production target and sentinel ranges keep
the integer comparisons below int32 overflow.

Existing incremental profile/cross arrays and undo rules are unchanged.
Parent-matching warm starts are not implemented in this execution: they need
a separate dual-repair implementation and measured benefit. The zero-slack
specialization is a smaller, proved reduction in assignment work.

### P3: witness packing and resource measurements

Native reference witnesses now carry exact one-byte-per-image permutations
on the supported domain rather than dense uint64 arrays or hash-only answers.
The benchmark records per-case aggregate coordinator plus child CPU time, output
bytes, wall time, and parent/child peak RSS. CPU time is not presented as elapsed
time. RSS fields are process high-water marks, not an aggregate memory cap.

Existing exact record transport already separates the shared palette from packed
certificate bytes. A new contiguous batch transport and time/byte-based scheduler
are not deployed without an equivalence and performance gate. Class commits,
counter reservations, prefix ownership and replay rules are unchanged.

### P4: independent canonical-augmentation prototype

An explicit finite-group prototype checks canonical construction paths for partial
bijections. Its objects are sets of compatible matching edges; R x P acts on their
two coordinates. For each parent it generates one extension per orbit of the
parent stabilizer. A child is accepted only if the newly added edge lies in the
invariant deletion orbit selected by its lexicographically minimum marked-object
certificate. The certificate here is computed by exhaustive group action.

Existence follows by deleting an edge in the selected orbit, transporting the
parent to its representative, and selecting the corresponding extension orbit.
For uniqueness, an isomorphism between two accepted children can be composed
with a child automorphism to align their selected deletion edges. Restriction
then gives an isomorphism between their parents and identifies their extension
orbits. Induction on matching size gives one child per isomorphism class.

The prototype checks both coverage and uniqueness against independent exhaustive
partial-bijection enumeration at EVERY depth, with edgeless, star/path, cycle/path
and typed examples. It has no asymmetric row restrictions and no distance pruning.
It does not prove correctness of combining this construction with the production
pruning rules, nor a speedup. It is not used by the benchmark.

## Validation and protocol

- Main regression suite: 325 passed, one optional PuLP skip, one pre-existing
  historical provenance test excluded. The historical evidence was not rewritten.
- Reporting correction: nine focused native API/minimum tests passed.
- Tests include raw permutation minima with deliberately worst-cost references,
  minimum-proof deadline failure, exact dense certificate comparisons, forced
  digest collisions, existing orbit-weight and cache-disabled differential oracles.
- UBSan: all 98 native tests passed in 14.71 seconds, with no undefined-behavior diagnostic.
- Portable C++17 -O3 build, no PGO and no -march=native.
- Final source: benchmark_results/synister_enhancement_v11_20260909/final_source.
- Original selection: 1,200 tasks, 751 minimum and 449 reference-CD tasks.
- Uniform configuration: frontier scheduler, 16 fresh workers per case,
  8,192-node slices, 1,000,000 retained worker-local double-orbit records,
  4 GiB address-space limit per worker, 60-second absolute case deadline.
- Pattern cache explicitly disabled. Pure encoding memoization is not a
  saved-answer cache and may persist within a coordinator.
- The cohort retained 113 closed sequential records on CPUs 0–15, then divided
  the remaining 1,087 tasks into disjoint batches of 544 and 543 on CPUs 0–15
  and 16–31. Each case uses 16 physical cores; campaign concurrency uses up to
  32 physical cores. All partitions use identical code and per-case settings.
  Source hashes and an explicit record-origin index validate the combined cohort.
- Separate hard repeats used CPUs 16–31 while the early sequential cohort used
  0–15. UBSan briefly used logical CPUs 32–35 (SMT siblings of 0–3). These jobs
  share a host and memory subsystem. No exclusive-host timing claim.
- Imports/dataset loading precede each case clock. Case construction, worker
  startup, proof, search, classification, final formatting and JSON file close
  are timed. File close does not imply fsync.
- Exact comparison normalizes class weights to labeled units and compares
  rational atom/bond frequencies, minima and reference-class presence. Traversal
  digests and literal selected-representative presence may differ by enumeration
  order; they are not used as equality requirements across different algorithms.


### Remaining performance work, informed by the completed repeats

In repeat 3, 30474 visited 27,897,762 new search nodes and replayed 124,687
prefix nodes (about 0.45% of their sum). It emitted 1,142,509 candidates and
retained 461,522 distinct full ITS classes. Coordinator merging took 3.725
seconds of measured coordinator wall time; worker startup took 1.029 seconds.
These components overlap worker execution and cannot be added as an independent
breakdown of the 54.629-second case wall time. Aggregate CPU time was 736.49
seconds. Parent peak RSS was about 2.17 GiB and the largest child high-water
mark about 0.47 GiB, not their sum.

These measurements make reduced assignment/canonicalization work a stronger
next candidate than merely reducing prefix replay. Warm starts need a repair
cost comparison because the current LAP already begins with feasible row/column
reductions. Unique construction paths must account for the cost of canonicalizing
partial states: fewer repeated leaves alone does not imply a faster engine.
No unvalidated alternative pruning rule is enabled in this run.

## Results

All three fresh-process full-output checks passed per hard case, with zero exact
observable differences against prior complete references:

| Case | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|
| 30474_reference_cd | 56.176 s | 56.723 s | 54.629 s |
| 22361_reference_cd | 38.563 s | 38.133 s | 36.919 s |

Every repeat includes JSON serialization/file close and used the strict 60-second
configuration. The engineering aim of <=55 seconds is not met consistently by
30474; all three observed repeats nevertheless finish below 60. This is measured
headroom, not a universal runtime guarantee.



## Completed 1,200-task cohort

| Mode | Tasks | Search complete | Structure complete | Both complete below 60 s |
|---|---:|---:|---:|---:|
| minimal | 751 | 750 | 750 | 750 |
| reference_cd | 449 | 449 | 449 | 449 |

Completed reference comparisons: 1199; observable mismatches: 0; classification regressions: 0. Pattern-cache hits: 0.

| Case | Full-output seconds | Search complete | Structure complete |
|---|---:|---|---|
| 13067_minimal | 13.780 | True | True |
| 28163_reference_cd | 8.143 | True | True |
| 30474_reference_cd | 58.600 | True | True |
| 22361_reference_cd | 38.426 | True | True |

The parallel continuation phase took 33.84 minutes; this excludes the already completed 113-case prefix. Summed per-case times are 70.50 minutes and overlap across the two CPU partitions.

Incomplete/error records: [{"key": [17156, "minimal"], "error": {"type": "TimeoutError", "message": "minimum proof incomplete; no provisional shell enumerated"}, "reason": null}].

Exact observable failures: [].

Frozen Python/C++ source SHA-256: a6ccee9ea1e19819487e5f74831a03681aa12ba86fa7a8878b40dc6937047bc6.

Portable library SHA-256: 696b9e296531a1f590d57906dab96f978f06fefadd715973d7ddc8add13defa6.

Current package matches the frozen implementation: False.

Full records, individual timing scopes and origins: [cohort artifacts](../../benchmark_results/synister_enhancement_v11_20260909/cohort_1200). Machine-readable audit: [final summary](../../benchmark_results/synister_enhancement_v11_20260909/final_summary.json).
