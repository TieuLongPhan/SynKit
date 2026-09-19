# Synister efficiency V8 — 8 September 2026

The optional native backend completes **one of the three remaining reference-CD shells**, including exact ITS and template classification. Cases **30474 and 22361 remain unresolved** within the 60-second budget. All existing V7 Python modules are byte-for-byte unchanged; this backend is selected explicitly.

This is a **different resource/output protocol**, not a new completion under the original single-search-worker, 100,000-product-representative protocol. Each case uses eight native workers, and the 100,000 cap counts retained worker-local **reactant/product double-orbit representatives**. Copies of the same class retained by different workers count separately against that shared cap. Reported solution counts are weighted back into the original product-orbit and labeled-mapping units. The original 1,200-task completion tally has not been updated or rerun.

| Source line | Result | Timed search/classification/merge | Analysis wall time | Product representatives | ITS classes |
|---|---|---:|---:|---:|---:|
| 28163 | Complete | 25.20 s | 29.31 s | 114,330 | 11,300 |
| 30474 | Timeout; partial counts | 69.89 s | 88.00 s | 2,111,800 | 70,962 |
| 22361 | Timeout; partial counts | 65.57 s | 75.71 s | 190,333 | 34,387 |

Counts on incomplete rows are lower bounds from fully proved observed orbits. Their held-out reference classes were not observed. Case 28163 has **11,280 template classes**, **5,619,548,160 labeled mappings**, and its held-out reference class is present. Its 114,330 product representatives already exceed the original 100,000 mapping cap.

The common deadline includes native search, exact per-candidate classification, worker-result transfer, deduplication, and frequency merging. Completion requires every shard to finish and the merged result to be ready before that deadline. Timeouts may take additional time to transfer and summarize partial results; the table exposes that overrun. Wall time also includes preparation, the reference-free seed, reference reveal, and public class-ID formatting, before JSON disk output.

Each worker has a 4 GiB address-space limit and is pinned to one of CPUs 0–7. The parent has no imposed address-space limit; the observed campaign peak was 1,346,776 KiB RSS. Regression checks ran separately on CPU 16, which is a distinct physical core. These timings must not be represented as single-worker speedups.

## Implementation and exactness

The new native_distance.cpp implements integer half-bond distance bounds, Hungarian assignment and forced-edge bounds, dynamic row selection, reversible residual profiles, and verified stabilizer pruning on both graph sides. Deterministic branching-prefix hashes partition the candidate search.

The native canonicalizer performs exact colored-graph refinement and individualization using verified automorphisms. Incomplete canonical or group-order proofs fail explicitly. Sparse exact keys are used for deduplication; public class identifiers retain the existing certificate encoding.

The orbit aggregator reconstructs exact product-orbit counts, labeled counts, and coordinate frequencies without expanding every mapping. For reactant group R, product group P, and full ITS stabilizer H, the product-orbit weight is |R|/|H| and the labeled weight is |R||P|/|H|. A coordinate orbit O receives weight × changed-coordinates-in-O / |O| at each coordinate. Every division is checked for integrality.

Worker results contain proved orbit-frequency totals, reducing parent-side merge work. The stream digest describes the compressed weighted records; it deliberately differs from the original expanded mapping stream and is labeled in backend metadata.

The explicit entry point is analyze_reference_blinded_native_shell in synkit/Chem/Mapper/native_analysis.py. Only the numeric reference CD reaches search; the mapping is revealed afterward. This Linux process backend requires compatible undirected half-integer weighted graphs, complete side-group proofs with the supported default budgets, and aligned symmetry/reaction-center properties. Unsupported settings fail explicitly.

## Validation and remaining limits

**266 tests passed**, with one optional skip and the previously known provenance test deselected. The 24 native/two-sided tests include exhaustive small mapping spaces, graph-atlas canonicalization and automorphism orders, colored graphs and loops, weighted reaction-center frequencies, exact ITS/template counts, shard partitioning, global retained-record limits, and public-API comparison.

All **46 previously completed reference-CD tasks** in the V7 verification selection also completed with the native backend under a separate 10-second, one-worker correctness run. Their counts, group orders, frequencies, reference-class checks, entropies, and public ITS/template class identifiers match exactly. The other 113 V7 completed tasks use the unchanged default path; they were not rerun in this native reference-CD audit.

A separate exploratory 180-second, 16-worker, 128-shard diagnostic with a one-million retained-record cap found **133,561 distinct ITS classes** for 30474 while still incomplete. Thus even a 100,000 distinct-class retention cap cannot represent its entire observed shell. This diagnostic used the earlier depth-12 sharding binary and is not a final-build benchmark completion. Both longer diagnostics remained incomplete. Simply adding static shards also left substantial work imbalance; further work needs better subtree scheduling and output handling for very large class spectra.

## Reproduction and provenance

Build explicitly; the command prints an immutable hashed library path and writes compiler/source/binary provenance beside it:

    python scripts/build_synister_native.py --output-dir /tmp/synkit-native

Then run scripts/benchmark_synister_native.py with these arguments:

- --selection benchmark_results/synister_efficiency_timeouts_v8_20260908/selection.json
- --source-root benchmark_results/synister_efficiency_timeouts_v8_20260908/verified_source
- --library followed by the printed library path
- --output /tmp/synister-native-review
- --seconds 60 --workers 8

Direct Python calls that spawn workers must run behind a main-module guard.

Artifacts are under benchmark_results/synister_efficiency_timeouts_v8_20260908/:

- verified_source/, build/, verified_run/: frozen source, immutable compiled kernel, full results and manifest.
- native_regression/, native_regression_audit.json: all 46 exact comparisons.
- default_source_audit.json, tests.log, summary.json: unchanged-default audit, test results, concise status.
- diagnostic_30474.json, diagnostic_22361.json: explicitly incomplete longer exploratory runs.

Final source hash (Python and C++): 96648cb1f66adff8c057c356433e6e2ef6d55892db01f02de85180d7ce13b441.

Native library SHA-256: 7efec1db19a29dfb65357dbb593542fcc7e3b5128d1880bed88fe2c232ba8abe.

Both final benchmark manifests confirm the frozen source stayed unchanged. The C++17 build produced no compiler warnings; Ruff and whitespace checks passed.
