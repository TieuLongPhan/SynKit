# V10 enhancement plan: complete cases 30474 and 22361 within 60 seconds

Date: 8 September 2026  
Status: proposed work; no V10 solver changes or new performance runs have been made.

The recommended sequence is to measure the complete execution path, move class-ID production into the existing worker pipeline, remove repeated leaf processing, and deduplicate exact ITS classes before expensive class-specific work. Optimize subtree replay and assignment bounds only where profiles justify them. Case 22361 needs approximately a twofold improvement; 30474 needs more than fivefold. Correctness can be established for the transformations below, but their combined speedup remains an experimental question.

## 1. Completion contract and baseline

“Complete below 60 seconds” will mean returning the entire exact reference-CD shell: weighted mapping counts, reaction-center frequencies, all ITS/template class counts and existing public class IDs, and post-search reference checks. The target distance is 12 in both cases. The task is exact-shell enumeration, not finding one mapping or proving that 12 is the global minimum.

Use two explicit clocks:

- Analysis wall: start before constructing the blinded case; stop after the complete public result is constructed, matching the existing benchmark's outer clock.
- End-to-end wall: the same start, through JSON serialization and output-file close. This is the proposed stricter acceptance clock. File close does not imply a storage-durability fsync; any durability requirement would need its own stated measurement.

One absolute 60-second deadline must cover case-dependent preparation, worker startup, search, classification, merging, reference checks and output production. Reaching a deadline with unfinished work produces an explicitly incomplete result. Compilation and general application installation remain outside per-case timing. No saved answer spectra, case-specific precomputed solutions or cross-run answer caches are allowed.

Use 16 worker processes, a fixed set of 16 physical cores shared with the coordinator, one numerical-library thread per process, the existing 4 GiB worker address-space limit, and the explicitly selected 1,000,000 worker-record cap. Hashing and classification must share these workers; another worker pool cannot silently increase CPU resources. Record coordinator and aggregate memory; the existing parent has no imposed address-space cap.

The current 100,000 default is insufficient for the full requested output: both cases have more than 100,000 distinct ITS classes. Therefore success under this plan means the stated 16-worker, one-million-cap protocol. It does not establish completion under the original 100,000 product-representative cap or the separate eight-worker protocol.

### Measured V9 evidence

| Quantity | 30474 | 22361 |
|---|---:|---:|
| Atoms | 87 | 96 |
| Analysis wall | 314.52 s | 121.53 s |
| Timed search/classification/transfer/merge | 244.10 s | 98.68 s |
| Wall outside that timed phase | 70.41 s | 22.85 s |
| Minimum analysis speedup needed for 60 s | 5.24× | 2.03× |
| Accepted candidate mappings | 1,142,509 | 257,650 |
| Distinct records summed across workers | 656,672 | 225,208 |
| Distinct ITS classes globally | 461,522 | 125,162 |
| Distinct template classes | 425,109 | 124,962 |
| Weighted product-orbit representatives | 27,783,018 | 642,926 |
| Labeled mappings represented | 6,145,159,053,312 | 987,534,336 |
| Visited nodes, including prefix replay | 40,301,662 | 9,828,302 |
| Completed batches | 14,830 | 4,505 |
| Maximum pending subtree count | 19,757 | 3,443 |
| Serialized result size | 103,931,885 bytes | 29,465,089 bytes |

These are complete runs with 16 workers and larger time allowances. The complete 30474 baseline is the completion_retry record; the earlier completion_run record for that case is incomplete.

The 70.41/22.85-second differences include several unseparated activities: input preparation, seed/group work, startup, post-search reference checks and public result construction. They are not isolated measurements of hashing. JSON serialization and disk writing happen after the recorded analysis wall time. The speedup requirements above are consequently lower bounds for the stricter end-to-end target.

The native enumerator did not individually enumerate 27.8 million leaves for 30474; orbit weights represent that many product orbits. Optimization estimates must use the actual candidate and class counts.

Historical runs used a shared host and differing affinities. Obtain an isolated matched baseline before attributing any speedup to code. Batch elapsed times overlap across workers and include preemption; their sum is not CPU time. Existing RSS fields are cumulative high-water marks, not reliable per-case aggregate memory measurements.

Evidence: [V9 report](EFFICIENCY_RESULTS_V9_2026-09-08.md), [extracted baseline with record hashes](EFFICIENCY_PLAN_V10_60S_BASELINE_2026-09-08.json), and the two original records:

- [30474 complete record](../../benchmark_results/synister_efficiency_timeouts_v9_20260908/completion_retry/30474_reference_cd.json)
- [22361 complete record](../../benchmark_results/synister_efficiency_timeouts_v9_20260908/completion_run/22361_reference_cd.json)

## 2. Mathematical invariants

### 2.1 Exact shell and admissible pruning

Let A and B be the symmetric bond-weight matrices, with zero diagonal, and let m be an element-compatible bijection. Preserve

$$
d(m)=\sum_{i<j}|A_{ij}-B_{m(i),m(j)}|,\qquad
\mathcal S_{12}=\{m:d(m)=12\}.
$$

The current native representation uses a=2A, b=2B and target Q=4d=48. All supported half-integer bond values are represented exactly. Changes must retain this scaling and checked integer arithmetic; floating tolerances cannot become a justification for pruning.

For a partial mapping, write C for its committed distance in quarter units. Any pruning bound L must satisfy

$$
L\leq\min_{\text{valid extensions}}(4d-C).
$$

Then C+L>48 safely rejects that subtree. Equality must remain searchable. A low-cost seed can guide traversal, but it cannot remove solutions in the requested shell or fix apparently unchanged atoms. The held-out reference mapping cannot guide search or pruning; only its numeric shell target is supplied.

### 2.2 Exact class identity and weights

Let R=Aut(A) and P=Aut(B), including all reported unary properties. Their action on mappings is

$$
(r,p)\cdot m=p\circ m\circ r^{-1}.
$$

For a mapping m, the stabilizer on reactant vertices is

$$
H_m=R\cap m^{-1}Pm.
$$

It is exactly the automorphism group of the full attributed ITS: node colors retain element and both unary-property values, and edge colors retain both bond weights. A complete canonical certificate therefore identifies a two-sided mapping orbit.

Orbit-stabilizer gives

$$
|[m]_{R\times P}|=\frac{|R||P|}{|H_m|}.
$$

The P action on bijections is free: p∘m=m implies p is the identity. Hence each distinct ITS class contributes

$$
w_m=\frac{|R|}{|H_m|}
$$

product-orbit representatives, and |P|w_m labeled mappings. Count this weight once per globally distinct full ITS certificate. Repeated discovery of the same ITS class does not add another weight. Different ITS classes sharing a template do contribute their separate weights to that template.

For a reactant coordinate orbit O, let k_O be the number of changed coordinates in O for one representative. Transitivity of R on O and invariance of the change set under H_m give the frequency increment for each coordinate:

$$
\Delta f_O=\frac{w_m k_O}{|O|}.
$$

All divisions must be exact. A missing group proof, nonintegral division or overflow prevents a complete result. Atom-frequency indicators must retain the current unary-change definition; template-center atoms also include changed-bond endpoints and are a different quantity.

These formulas justify global class deduplication and moving work between processes. They do not justify approximate graph equivalence.

### 2.3 Certificate and scheduling requirements

An internal compact encoding must be injective over the existing exact certificate: graph size, ordered typed node tokens, sorted edge positions and typed edge tokens must all be recoverable. Include explicit lengths, a format version and the reaction palette identity. Different worker-local color numbers cannot be compared without a shared dictionary.

A digest may select a bucket or worker. Exact certificate equality decides identity, including when digests collide. Preserve the current public SHA-256 identifiers byte for byte; those public identifiers are not a substitute for exact internal equality.

Every search split must partition a parent's unexplored subtree into disjoint children whose union is the parent remainder. Completion also requires every class-processing and output job to finish. An empty search queue alone is insufficient.

## 3. Phase A: profile the complete path

Add observability before selecting deeper algorithm changes.

Measure nonoverlapping coordinator phases and per-worker CPU categories: preparation and side-group proofs; prefix replay versus new search; profile-cost construction; Hungarian assignment; forced-edge/Floyd work; stabilizers; leaf matrix encoding; full ITS canonical search; automorphism-order calculation; template construction/canonicalization; certificate packing; IPC; exact merge; ID hashing; sorting; JSON serialization and writing.

Record native canonical search nodes and group failures, not just elapsed time. Sample hot-path timers rather than timing every primitive. Compare instrumentation on/off on identical inputs and report its overhead.

Add counters for duplicate candidates within a worker, duplicate ITS classes across workers, repeated template keys, transferred bytes, allocations, queue delay, worker idle time, and the number of distinct actual serialized ID prefixes. Measure fresh-process RSS and sample total live-process memory. Record source/binary hashes, CPU topology, affinity and competing load.

Run one matched complete baseline for each case with sufficient diagnostic allowance, then use short, recorded candidate streams for component microbenchmarks. Replay is allowed only for profiling; all final performance claims require a fresh complete search.

**Decision rule:** retain an optimization only when its end-to-end gain exceeds observed timing noise without violating exactness. For an isolated serial fraction f accelerated by s, the overall speedup is

$$
S=\frac{1}{(1-f)+f/s}.
$$

Apply this only to measured compatible time fractions; overlapping workers and pipeline stages need critical-path accounting. Search-only acceleration cannot meet the present 30474 wall target if its 70.41-second outside portion remains unchanged.

Deliverable: a phase table showing seconds, CPU seconds, invocation counts and bytes for both cases, with uncertainty and timing scopes.

## 4. Phase B: pipeline exact public-ID production

This is the first implementation priority because the current public result formatting occurs after enumeration.

1. Keep the existing worker pool alive for bounded ID jobs. Begin ITS ID computation when a new complete canonical key is established; begin template ID computation after exact template deduplication. Counts can accumulate independently while IDs are calculated.
2. Preserve the exact legacy dense certificate and its repr/escaping convention. The native backend already hashes it incrementally, so another “use incremental hashing” change is not new.
3. Cache the SHA-256 state after an actual repeated serialized node/header prefix. Clone it before appending that key's adjacency suffix. Bound the cache by memory and key it by the full prefix bytes and encoding version.
4. Batch token encoding and digest updates to reduce Python calls. Reuse a shared immutable token dictionary, and remove duplicate full-record dictionaries once their integrity checks have been retained in a smaller form.
5. Preserve deterministic public class ordering. Merge sorted output chunks where useful and measure serialization separately. Every class and its multiplicity must still appear in the result.

For prefix p and suffix q, copying the hash state after p and then updating with q computes exactly SHA256(p||q). This preserves the identifier without constructing the dense string. Python documents both update concatenation and hash-state cloning in its [hashlib API](https://docs.python.org/3.11/library/hashlib.html#hashlib.hash.copy).

Do not substitute SHA256(sparse_certificate), combine finished digests, or assume all node prefixes are identical. These change identifiers or use an unsupported assumption. Prefix reuse is worthwhile only if measured reuse exceeds cache and serialization costs.

**Gate:** byte-identical legacy IDs on every regression and adversarial escaping fixture; reduced total wall time with the same worker/core budget. Hashing a frozen stream is a microbenchmark, not a sub-60-second completion.

## 5. Phase C: process each exact class efficiently

### C1. Reuse leaf data and batch the native boundary

The current path creates a transported product matrix in observe, repeats it in commit, and builds the paired ITS matrix separately for full ITS and template canonicalization. It also returns to Python for every accepted candidate.

Introduce a native leaf context with reusable storage for the mapping, transported bonds, paired edge colors, unary colors, changed-coordinate indicators and template context. Derive these once per candidate and reuse them. Keep per-worker immutable reaction matrices and palettes prepared once; pass complete verified side-group data rather than proving the same immutable groups repeatedly during initialization.

Return bounded batches of compact certificates, mapping witnesses and required counters. Prefer contiguous owned buffers over nested Python edge tuples. Buffer reuse must wait for consumer acknowledgement; no queued record may refer to a mutable buffer that the search overwrites.

This removes repeated allocation and language crossings while preserving the same graph. Python callbacks are expensive within each process; there is no single GIL serializing all 16 worker processes.

### C2. Deduplicate globally before template and frequency work

Initially keep complete full-ITS canonicalization in search workers. After forming its exact compact key K, send a representative witness and proof data for K to a single logical owner.

A minimal first implementation can keep exact ownership in the coordinator, which already merges records, and dispatch class-processing batches through the same 16-worker pool. Distribute ownership by a stable hash only if profiling establishes that the coordinator is the bottleneck. Hash collisions remain exact-key buckets.

For a new K, create one pending class job. Repeated K values attach to that entry without scheduling another template/frequency job. Only commit its weight after complete H_m and class-processing proofs are available. Preserve raw candidate-mapping digest collection if the existing observed-reference-mapping field still needs it.

The measured worker-record/unique-class ratios are:

- 30474: 656,672 / 461,522 = 1.423.
- 22361: 225,208 / 125,162 = 1.799.

These are possible reductions in repeated per-class call counts, not total-runtime speedups. Early ownership cannot avoid the initial full canonicalization needed to identify K.

Use fair scheduling and bounded queues so class jobs cannot starve behind search jobs. A worker must not block waiting for another job in a fully occupied pool. Preserve compact per-search-worker membership and the existing one-million cap charge on that worker's first discovery; changing physical record layout must not silently change the benchmark cap unit.

### C3. Separate canonical labels from group-order work

The template call currently computes an automorphism group order that its caller discards. Add an explicit “canonical certificate only” native mode that skips the final stabilizer-chain order calculation for templates, while retaining the complete canonical search and verified automorphisms used for pruning it.

For full ITS classes, H_m remains essential. A later refinement may defer the final group-order calculation until global uniqueness is established, using complete canonical-search proof data or recomputing the order for that one canonical graph. Preserve proof completeness across the handoff.

The candidate/unique-ITS ratios are 2.476 and 2.059. They bound possible call-count savings for work that can move after global deduplication. They do not imply that the full canonical search itself disappears.

**Gate:** exact equality of class IDs, integer weights and coordinate frequencies; complete group proofs; bounded queue/memory growth; a measured gain after counting packing, transfer and deferred work.

## 6. Phase D: reduce search overhead where measured

### D1. Resume state without replaying from the root

V9's 14,830 and 4,505 batches use deterministic prefixes. The node counters include rebuilding those prefixes, so first measure the replay fraction.

Try slice/batch tuning before changing representation: larger slices reduce replay and IPC but can worsen tail imbalance. Use measured wall time and replay work, not the pending-queue count alone, to choose.

If replay remains material, add resumable native search contexts. A transferred context must include, or exactly reconstruct, the mapping and used columns; dynamic row order; row_min and permitted domains; cross costs and profile histograms; both current stabilizer groups; committed cost; pending DFS choices and undo state; and any branching/hash state still used. Transfer owned values and validated identifiers, not process-local pointers.

Continuing a worker-local context avoids replay. Work stealing requires an independently owned snapshot of an unvisited branch. Snapshots can be expensive; compare their copied bytes and group reconstruction cost with the replay they remove.

Correctness follows by induction: the snapshot reproduces the same remaining states and transition rules, and each transferred branch leaves the donor's pending stack exactly once. Never merge states merely because partial mappings or partial ITS graphs look alike; history-dependent allowed domains can differ.

Image-only prefixes also depend on deterministic row selection. A change to branching rules or adaptive caches must either retain reproducible replay under a versioned algorithm or use complete snapshots. Old V9 prefixes cannot be resumed under new branching rules without such a proof.

### D2. Reuse assignment certificates safely

The native solver already has typed profile lower bounds, Hungarian assignment, forced-edge bounds using shortest alternating paths, incremental histograms and dynamic row selection. Reimplementing those features is not an enhancement.

The proposed extension is dual warm starts and cheap certified rejection before an expensive assignment/Floyd pass.

For remaining atoms U and images V, define h_A(i,t,l) as the number of remaining type-t neighbors whose doubled bond value is at least level l; define h_B similarly. The existing profile discrepancy is

$$
P_{ij}=\sum_{t,l}\Delta_l
  |h_A(i,t,l)-h_B(j,t,l)|.
$$

For every compatible completion phi, sorted matching of the one-dimensional bond values minimizes the row's L1 discrepancy. Thus

$$
P_{i,\phi(i)}
\leq\sum_{k\in U}|a_{ik}-b_{\phi(i),\phi(k)}|.
$$

Let X_ij be the exact new cost of connecting i→j to already assigned atoms, in quarter units, and set c_ij=X_ij+P_ij on allowed edges. Summing X counts assigned-to-unassigned edges once. Summing the row discrepancies counts remaining unordered edges twice in doubled units, exactly the quarter-unit scale. Therefore the minimum assignment cost L over c is a valid lower bound on the remaining quarter-unit distance.

When costs or domains change, an old dual objective is not automatically valid. Retain column potentials v_j on surviving columns and repair each row:

$$
u_i=\min_{j:(i,j)\ {\rm allowed}}(c'_{ij}-v_j).
$$

An empty row proves infeasibility. Otherwise u_i+v_j≤c'_ij on every allowed edge, so

$$
D=\sum_i u_i+\sum_j v_j
$$

is a valid lower bound by weak duality. With reduced costs r_ij=c'_ij-u_i-v_j≥0, any assignment forced to use i→j costs at least D+r_ij. Hence C+D>48 rejects a node, and C+D+r_ij>48 safely excludes that edge.

For survivors, initially retain the existing exact Hungarian/Floyd calculation and branch-selection rules. Warm-start matching repair must establish primal feasibility, dual feasibility and complementary slackness before claiming an optimum. Do not reuse an old optimum merely because one row and column were removed.

Keep forbidden edges explicit and use checked wide integers. Add two bounds only when they charge disjoint objective contributions; otherwise take their maximum. The cheap dual screen may be weaker than the existing forced-edge bound and may add overhead, so it ships only if the complete-run profile shows a net gain.

**Gate:** compare every new bound and rejected edge against exhaustive valid completions on small states, including states with nontrivial symmetry domains. End-to-end speed matters more than a reduced Hungarian invocation count.

## 7. Conditional component optimization

If profiling shows many repeated disconnected full ITS components, cache their exact certificates and group proofs.

For a colored graph containing m_t copies of each connected component type C_t,

$$
|\operatorname{Aut}(G)|=
\prod_t |\operatorname{Aut}(C_t)|^{m_t}\,m_t!.
$$

The component-certificate multiset is an exact isomorphism invariant because graph isomorphisms permute connected components and restrict to component isomorphisms. Both the component multiplicities and their automorphism orders must be exact.

Apply this to the actual full ITS union graph for the current mapping, or to the actual boundary-colored template. Reactant and product components cannot simply be fixed or independently paired: legal mappings may connect them in the ITS.

This is conditional because sorting component certificates does not automatically reproduce the existing global canonical-label convention and public IDs. An initial use should supply verified automorphism generators or cached group proofs to the existing canonicalizer. A different internal certificate is possible only with a verified path back to identical legacy IDs, whose cost must be measured.

Individualization/refinement and automorphism pruning are established exact graph-canonicalization techniques; refinement colors alone are not complete isomorphism certificates. See McKay and Piperno, [Practical graph isomorphism, II](https://arxiv.org/abs/1301.1493). Replacing the engine with nauty/Traces is a separate experiment with label-compatibility and integration costs, not an assumed speedup.

## 8. Quantitative targets and priorities

Use the following 55-second engineering budget, leaving five seconds before the hard threshold. These are design targets, not predictions:

| Disjoint wall segment | Target |
|---|---:|
| Case preparation and worker readiness | ≤3 s |
| Overlapped search, exact class work and ID production | ≤42 s |
| Remaining class/output jobs, reference checks and final assembly | ≤8 s |
| JSON serialization and output-file close | ≤2 s |
| Total | ≤55 s |

For 30474, a 42-second main window requires about 27,203 accepted candidates/s and 10,989 unique ITS classes/s if candidate generation is unchanged. For 22361, the corresponding rates are about 6,135 candidates/s and 2,980 ITS classes/s. These are aggregate throughput requirements, not single-core targets or measured capacities.

The two spectra together contain 886,631 class entries for 30474 and 250,124 for 22361. Merely writing the current-size records in two seconds requires about 52.0 and 14.7 MB/s respectively; serialization and formatting must fit too. Enumerating and returning every class has an unavoidable output-size cost. The class count alone does not prove that 60 seconds is impossible.

Implement and validate B, C1 and C3 first. Then evaluate C2, where 22361's larger cross-worker duplication ratio makes it particularly relevant. Use D1/D2 only for a demonstrated remaining search bottleneck. Evolving the canonicalizer or partial-state symmetry quotient comes after these lower-risk changes and requires a separate preservation proof.

Do not multiply the individual duplication ratios and hypothetical native speedups: the affected work overlaps. If the measured critical path for 30474 still exceeds 60 seconds, report that result and its dominant cost. Neither exactness nor this plan establishes a universal runtime bound.

## 9. Implementation sequence and validation gates

| Work item | Likely files | Required evidence before proceeding |
|---|---|---|
| Phase clocks, shared deadline, resource manifest | scripts/benchmark_synister_native.py; synkit/Chem/Mapper/native_analysis.py | Complete matched baseline; separately measured serialization |
| Pipelined IDs and prefix-state reuse | exact/native_canonical.py; exact/orbit_aggregation.py; exact/native_frontier.py | Legacy ID byte equivalence; full wall improvement |
| Reused native leaf context and optional group-order pass | exact/native_distance.cpp; exact/native_candidates.py; exact/native_its.py | Same exact certificates and group orders; safe buffer lifetime |
| Exact global class ownership | exact/native_frontier.py; exact/orbit_aggregation.py | Once-only weighted commit; cap scope preserved; bounded pending work |
| Stateful continuation and dual warm starts | exact/native_distance.cpp; exact/native_candidates.py | Exhaustive continuation/pruning checks; measured net gain |
| Completion audit and result report | benchmark scripts; paper/synister | Both complete under the stated final protocol |

The “exact/” paths in this table are relative to synkit/Chem/Mapper/. Add new native exports or a versioned ABI without breaking existing callers. Keep the optional native entry point explicit and preserve the current default Python backend.

### Exactness checks

- Compare all element-compatible permutations for small weighted, attributed graphs against an independent enumeration oracle. Include half-integer bonds, zero and nonzero shell targets, disconnected and highly symmetric graphs, changing unary attributes, and template boundary colors.
- For every new pruning rule, compare its lower bound and each excluded assignment edge with exhaustive valid completions. Include restricted permitted domains and row_min conditions.
- Test every small continuation cut, tiny slices, interrupted transfers, duplicate delivery and reordered completion. Use 1, 2, 8 and 16 workers as appropriate. Preserve total outstanding work across search, queued children, in-flight messages and pending class/ID jobs; publish child work before retiring its parent credit.
- Force hash collisions and run different process hash seeds. Equality must still compare exact certificates. Check empty graphs, isolated atoms and typed tokens containing Unicode, quotes and backslashes against the legacy ID routine.
- Preserve V9's 46 previously complete reference-CD regressions and 28163 as a performance control. Run the existing relevant suite: the recorded baseline is 276 passes, one optional skip and one known provenance exclusion; report the actual new count and any exclusions.

For both large cases, compare complete ITS/template ID-to-count maps, coordinate frequencies, unions/intersections, group orders, weighted and labeled totals, entropies under the existing numerical policy, and reference-class checks with the frozen V9 outputs. Equality of total counts alone is insufficient.

Search counters and representative-stream digests may change under different traversal or proven pruning. Do not treat them as canonical outputs. Preserve the meaning of the “reference mapping observed” field: actual discovery, not inferred from class membership. Keep the reference mapping out of search inputs.

### Performance acceptance

Run each final candidate build three times per case, sequentially on the same controlled physical-core set, with fresh case-local state and complete output enabled. Alternate case order and retain every run, including failures; do not select the fastest run.

Pass only if all six runs are complete, all exact audits pass, and every end-to-end time is strictly below 60 seconds. Aim for a maximum of 55 seconds. Three repeats establish observed performance for this input/build/hardware setup, not a guarantee under arbitrary contention.

Record frozen source and native-library hashes, limits, affinity, timing scopes, class counts, memory and output bytes. Keep the old eight-worker bounded result and original campaign totals distinct. After these gates, rerun the relevant selected verification set under one uniform protocol before changing campaign claims.

## 10. Deliverables

1. The baseline and critical-path profile, with measured bottlenecks.
2. Small reviewable implementation changes, each carrying the applicable proof obligation and differential evidence.
3. A final report with complete exact comparisons and all repeated timing results.
4. A clear pass/fail statement for each case under the stated resources.

The present document and its baseline JSON are planning artifacts. They do not claim either remaining case has already reached 60 seconds.


## Implementation outcome

Completed on 8 September 2026. All six fresh-process acceptance runs produce complete, exact outputs below 60 seconds: maximum 59.043 seconds for 30474 and 43.716 seconds for 22361. The 55-second engineering aim was not reached for 30474. The chosen implementation, deferred work items, mathematical arguments, test exclusions, build/profile provenance, and reproduction command are recorded in [the V10 results report](EFFICIENCY_RESULTS_V10_2026-09-08.md). No original-protocol campaign totals were changed.
