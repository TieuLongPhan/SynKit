# V10 efficiency implementation and verification

Status: **all six fresh-process 60-second acceptance runs passed**.

This implements [the V10 plan](EFFICIENCY_PLAN_V10_60S_2026-09-08.md) in the optional native backend. The default Python implementation is unchanged: default_source_audit.json compares 425 modules with frozen V9.

## Measurements

Times include case construction, public result creation, JSON serialization, and file close. Development runs used a 600-second allowance to obtain complete spectra. These are individual observations on a shared host.

| Development build | 30474 (s) | 22361 (s) |
|---|---:|---:|
| Fresh V9 | 312.551 | 102.242 |
| Shared prefix replay | 115.997 | 66.828 |
| Pattern cache / assignment SCCs | 89.051 | 52.718 |
| Native template scans | 71.609 | 48.102 |
| Sparse certificates / native pairing | 65.603 | 40.216 |
| Uniform short slices (rejected) | 94.723 | 32.155 |
| Adaptive slices / lean encoding | 60.645 | 41.890 |
| Bytes transport | 61.069 | 40.712 |
| Shared setup / larger batches / compact JSON | 61.025 | 39.427 |
| Verified terminal completion | 63.143 | 40.366 |
| Assignment-certified completion | 62.155 | 37.085 |
| Profile-guided build | 60.195 | 44.319 |
| Reused buffers / contiguous neighbors | 59.729 | 37.833 |
| Shared palette transport | 58.035 | 40.481 |

Uniform 50 ms slicing increased overhead on 30474 and was replaced with adaptive slicing: short slices apply to batches with at most four requested prefixes; larger batches retain the node budget. CPU-specific compilation alone did not improve the measured result. Intermediate source copies, libraries, logs, and complete outputs are preserved.

The final runner supports compact JSON with identical fields and values. Earlier runs used indentation. Serialization is timed separately in every case_timings.json; formatting savings must not be attributed to mathematical search.

## Strict acceptance

Each task runs in a fresh Python process with 16 workers within CPUs 16–31, one BLAS/OpenMP thread, 4 GiB address space per worker, and a one-million retained-worker-record cap. The parent shares the same CPU set and has no separate address-space limit. No answer cache or process-warm class cache is supplied.

The 60-second deadline starts before case preparation. Passing additionally requires complete JSON output written and closed before 60 seconds. Physical-media persistence via fsync is not claimed.

| Fresh process | Full output seconds | Complete below 60 s |
|---|---:|---|
| acceptance_r1_22361 | 43.716 | True |
| acceptance_r1_30474 | 58.542 | True |
| acceptance_r2_22361 | 37.884 | True |
| acceptance_r2_30474 | 57.108 | True |
| acceptance_r3_22361 | 40.640 | True |
| acceptance_r3_30474 | 59.043 | True |

The cap counts distinct full ITS records within each worker, including copies in different workers. It does not count weighted product representatives. The API default remains 100,000, which cannot hold these full spectra.

## Exact observables

| Case | ITS classes | Template classes | Weighted product representatives | Labeled mappings |
|---|---:|---:|---:|---:|
| 30474 | 461,522 | 425,109 | 27,783,018 | 6,145,159,053,312 |
| 22361 | 125,162 | 124,962 | 642,926 | 987,534,336 |

The audit compares every public class identifier and multiplicity, all reaction-center frequencies, entropies, group orders, and reference-class checks with complete frozen V9. Discovery-order stream digests are excluded because scheduling changes their order. Totals alone do not establish correctness.

## Correctness of the implementation

1. **Assignment bounds.** Feasible integer dual reductions initialize Hungarian matching. Complementary zero-cost partial matching is augmented to an optimum. Nonnegative row minima and partial profile costs prune only when the remaining exact budget is exceeded. Within a strongly connected zero-reduced-cost component, every pair of vertices is connected at zero cost. Contracting these components therefore preserves shortest-path distances and forced-edge assignment bounds. An independent permutation oracle checks optimum costs, dual feasibility, strong duality, infeasibility, and every forced-edge optimum.

2. **Forced assignments.** A row with one remaining hard-domain image must use it in every feasible continuation. Skipping a relaxation there only postpones pruning. The usual reactant/product stabilizer and domain updates still run unless both groups already fix the chosen points, in which case those updates are inert. If all hard or assignment-certified domains are singleton and form a bijection fixed by both groups, the unique completion is checked directly against the exact objective. Requested prefixes are still checked against the deterministic forced sequence. Leaves require exact target equality.

3. **Canonical certificates.** Sparse neighbor signatures preserve the original refinement ordering. Equal-row twins allow the same first individualization branch without repeatedly refining cells that cannot split. For positive edge-color IDs, sparse pairs (-position, color) have the same lexicographic ordering as the dense adjacency certificate with implicit zero entries. Public IDs still hash the exact legacy certificate bytes.

4. **Complete group orders.** If every invariant root cell consists of true twins, its automorphism group is the product of the corresponding symmetric groups. Otherwise, joining the entire support of each verified generator gives disjoint components. Each generator acts within one component, so the generated group is the direct product of its restrictions. Each restriction uses the exact stabilizer chain. Overflow or incomplete proof fails closed.

5. **Witnessed duplicate cache.** Sparse ITS patterns are injective relative to the fixed reactant baseline. Entries represent retained classes or explicit images under verified reactant automorphisms. A hit therefore identifies an already-counted double orbit. Hash collisions use full vector equality; bounded eviction merely loses acceleration.

6. **Weights and frequencies.** For mapping m, H = Aut(A) intersect m-inverse Aut(B) m. The double-orbit weight is |Aut(A)| / |H|; labeled count multiplies it by |Aut(B)|. In coordinate orbit O containing k changed coordinates, each coordinate receives weight times k / |O|. Every division is checked for integrality. Repeated discovery adds zero; distinct ITS classes sharing a template add weights.

7. **Template scans and transport.** Native change flags preserve numeric equality and the existing isclose tolerance, separately from typed canonical colors. Context expansion and external-resource multisets preserve the public definitions. Template search omits unused final group-order calculation and optional seed construction but completes its canonical certificate. Cached complete edge fragments and copied SHA-256 prefix states preserve the legacy ID byte stream. Messages share exact palette bytes within a batch and retain the full numeric certificate and deterministic public ID. Their atomic tuple representation is injective, so digest collisions cannot merge distinct certificates.

8. **Scheduling and setup.** Prefix tries replay shared ancestors once. Node/time yields emit an exact disjoint cover of unfinished work. Shared validated setup, larger prefix batches, and adaptive yields change repeated computation and scheduling, not the problem. Completion requires all pending jobs, running jobs, classifications, and merges to finish.

Global class ownership, fully serialized search states, and parent-to-child dual reuse remain deferred. The runtime target was reached with the reductions described above.

## Validation and provenance

Focused tests cover exhaustive small graph oracles, weighted shells, assignment dual and forced-edge certificates, loops and typed colors, group overflow, component exchange, cache on/off equivalence, key collisions, and slow-callback resumption. Exact audits and regression records accompany the outputs.

Host-specific builds explicitly use GCC -march=native and -mtune=native. They are not claimed portable to CPUs lacking those features; the portable build remains the default. The immutable manifest records compiler version, flags, resolved CPU options, and source/library hashes. The final candidate also uses GCC profile-guided optimization, trained on bounded search and classification work from both cases. Stored profiles contain execution frequencies, not answers. Training is separate from every timed verification. The cache remains bounded at one million witnessed patterns or 64 MiB of integer payload per worker; allocator/container overhead is additional and remains subject to the unchanged 4 GiB worker limit. See [GCC instrumentation options](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html). See [GCC x86 options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html).

Validation status:

- 235 targeted tests passed against the actual optimized library with no exclusions. The broader mapper/canonicalizer suite passes 290 tests with one optional PuLP skip and the single previously documented frozen-evidence provenance exclusion (tests_full_excluding_known.log).
- Exact regression comparisons: 46/46.
- Control 28163: 5.881 seconds.
- Maximum acceptance time: 59.043 seconds.
- The observed maximum exceeds the 55-second engineering aim but meets the strict 60-second gate. Three repeats establish performance for these inputs, this build, and this host, not a guarantee under arbitrary contention.

The excluded test checks an older alternative-ITS evidence record against a source fingerprint. The covered source files and their fingerprint are identical in current code and frozen V9; both differ from that historical record. The original failure output and known_provenance_audit.json are retained, and the historical evidence is not rewritten.

Provenance:

- Library SHA-256: a2ea93819613f230f1a28b7f3d1515f82177eec80d0e33311caf45cc48ad6c13.
- C++ SHA-256: 5cae74248b3fee45ca1bff2c0a56ec781735756b7873d8d795259ebe6eb4d320.
- Frozen benchmark runner SHA-256: c25ade6e13cc279426ae4be6fd0e54264052ba48a0835fbbebed86cec90accb5.
- Complete build/profile metadata: benchmark_results/synister_efficiency_timeouts_v10_20260908/build/libsynkit_distance_4db32da77b34a661873c.json.

To reproduce a bounded two-case check from the repository root, choose a new empty output directory:

    taskset -c 16-31 /home/labhhc4/anaconda3/envs/synfrag/bin/python scripts/benchmark_synister_native.py --selection benchmark_results/synister_efficiency_timeouts_v10_20260908/selection.json --source-root benchmark_results/synister_efficiency_timeouts_v10_20260908/verified_source --library benchmark_results/synister_efficiency_timeouts_v10_20260908/build/libsynkit_distance_4db32da77b34a661873c.so --output /tmp/synister-v10-check --workers 16 --mapping-cap 1000000 --seconds 60 --wall-budget --compact-json

The immutable source/profile rebuild recipe is pgo_build_x.py in the artifact directory. Its optimize step uses the preserved profile data; the instrumentation/training steps are separate from verification. The ordinary build_synister_native.py command remains a portable, non-PGO build and is not the binary used for these timings.

Artifacts: benchmark_results/synister_efficiency_timeouts_v10_20260908/.

No new full 1,200-case or 10,000-case campaign is claimed. Historical campaign totals are not rewritten.
