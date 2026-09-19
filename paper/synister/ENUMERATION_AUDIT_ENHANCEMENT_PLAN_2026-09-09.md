# Enumeration audit and enhancement plan — 9 September 2026

This plan concerns the current Python enumeration path and the optional native
two-sided orbit path. Its first objective is correctness evidence independent
of cached classifications. Its performance objective is complete, correct
public output below 60 seconds, with useful margin, on the frozen timeout
cohort. Neither tests nor previous timings establish a universal runtime bound.

Audit artifacts are in
[benchmark_results/synister_enumeration_audit_20260909](../../benchmark_results/synister_enumeration_audit_20260909/).
The final measurements belong in the accompanying results report. This document
separates inspected invariants, two implemented corrections, and future work.

## 1. Exact problem and supported domain

Let A and B be the reactant and product bond matrices. A mapping m is a
bijection preserving atom type. The numeric objective is

    D(m) = (1/2) sum_ij |A_ij - B_m(i),m(j)|
         = sum_{i<j} |A_ij - B_m(i),m(j)|

for symmetric zero-diagonal inputs. A reference-CD task enumerates D(m)=T,
where only the scalar T may come from the held-out reference. A minimum task
first proves U=min_m D(m), then enumerates every optimizer at D(m)=U.

The native path now also requires ordinary atom-label equality to agree with
the exact typed label representation. It checks 1..256 vertices, equal type
multiplicities, symmetry,
zero diagonal, finite half-integer entries of absolute value at most 512,
at most 16 distinct bond values, and a nonnegative quarter-unit target at most
250,000. Unsupported inputs and incomplete side-group proofs raise errors.
The Python path has broader numerical/input semantics; it is not legitimate
to claim that the native implementation covers every Python input.

With a=2A, b=2B and t=4T, the native objective is
2 sum_{i<j}|a_ij-b_m(i),m(j)|. Every accepted branch and leaf uses integer
arithmetic. Quarter units permit the half-row relaxation to stay integral;
a shell target that no mapping attains must simply produce an empty shell.

## 2. Lower bounds, branching, and terminal closure

For an assigned set S and residual set U, let c be the already committed
quarter-unit cost on S. The cross cost for i->j is

    X_ij = 2 sum_{s in S} |a_is - b_j,m(s)|.

For each remaining neighbor type, sort the residual row entries of a and b.
Let P_ij be the sum of their sorted L1 distances. For any residual bijection f,
each row's sorted matching is no more costly than its matching induced by f.
Each residual unordered edge is counted in two rows, so

    c + min_assignment sum_i (X_i,f(i) + P_i,f(i))
      <= quarter-unit cost of every feasible completion.

Including the zero self-entry in both row multisets leaves their cumulative
difference unchanged. Signed bond values do not invalidate sorted L1 matching.
The cumulative-histogram implementation integrates the absolute difference of
counts over consecutive bond levels; it computes exactly the same sorted L1
distance. This is why the histogram shortcut is valid.

Hard incompatibilities receive a forbidden-edge sentinel. Nonnegative partial
profile costs and partial sums of row minima may reject a branch only after
they exceed the remaining target budget. Combining overlapping global bounds
requires their maximum; adding them can double-count bonds. The Python code
uses that maximum when combining its row-profile and bond-mass bounds.

The Hungarian solution supplies a matching pi and feasible duals u,v satisfying
u_i+v_j <= C_ij, with equality on matched edges. Define reduced costs
r_ij=C_ij-u_i-v_j. Relative to pi, forcing i->pi(k) closes an alternating cycle:
its extra minimum cost is r_i,pi(k) plus the shortest reduced-cost path k->i.
The other cycles of an alternative assignment have nonnegative cost and may
be omitted. This proves the forced-edge lower-bound test. Contracting strongly
connected components of zero-cost edges preserves these shortest distances.
Equality with the target must survive shell pruning.

Singleton assignment can safely delay bound computation. Closing an entire
residual assignment additionally requires singleton feasible domains, a
bijection, exact recomputation of its remaining cost, and both stabilizers
fixing all selected points. Prefix replay must consume the same deterministic
image sequence. These checks are present; tests exercise sliced and unsliced
search, full-length resumed prefixes, impossible costs and wrong prefixes.

**Correction implemented in this audit.** The assignment diagnostic accepts
64x64 matrices with finite entries through 1,000,000, but treated totals or
alternating paths at least 50,000,000 as infinite. Two counterexamples are a
constant 1,000,000 matrix and a diagonal/long-cycle matrix with a forced
64,000,000-cost cycle. Both failed on the V10 binary. The diagnostic now
distinguishes the actual 100,000,000 sentinel, and shortest-path propagation
retains large finite paths. Production accepted shell bounds never exceed
1,000,000 quarter units, so the counterexample did not establish a production
shell error. The new tests check the exact optimum, dual feasibility and all
forced-edge optima at this boundary.

## 3. Symmetry coverage and exact weights

**Input-contract correction implemented.** On two edgeless vertices labeled
[True, 1], ordinary compatibility permits two bijections, but the typed side
groups distinguish the labels. The full ITS key records the reactant atom
type, not a separate product atom type; both mappings can therefore share one
ITS key despite belonging to different double orbits. The previous weighted
path reported one labeled mapping instead of two. The same issue occurs with
[0.0, 0]. The native preparation wrapper now rejects equal labels with different
typed representations, rather than silently changing compatibility or public
ITS identifiers. Three rejection tests reproduce the missing validation.

This guard was added after the cohort snapshots were frozen. A separate check
constructed every one of the 1,077 reaction inputs and validated all 1,200
task instances: all atom labels are builtins.int and none is rejected.
An AST comparison confirms that removing only the new helper and its call
restores the frozen wrapper exactly; all other Python/C++ source files match.
Thus the counting computation on these inputs is unchanged. The recorded
cohort timings belong to the frozen wrapper and exclude the added guard's
small, unmeasured cost. Both snapshots and this scope audit are retained.

Let R=Aut(A) and P=Aut(B), preserving every reported atom property as well as
the weighted graph. They act on mappings by m -> p m r. Independent point
stabilizers remove equivalent choices. Earlier branch restrictions must remain
invariant under the current stabilizers: product exclusions cover entire
product orbits, while reactant row restrictions apply to the entire relevant
reactant orbit. Dynamic row ordering affects efficiency, not admissibility.

A coverage argument can be stated at a single node. Fix the chosen reactant
row x and its current R-stabilizer orbit Q. Among the images of Q in any
feasible completion, choose the smallest eligible product-orbit leader.
An R-stabilizer element moves its preimage to x; a P-stabilizer element moves
the image to that leader, while preserving the assigned prefix and objective.
That transformed completion belongs to the corresponding enumerated branch.
Branches with a larger chosen leader may therefore exclude lower product
orbits throughout Q. The additional numerical lower limit is consistent:
other allowed parent product orbits have larger leaders, and remaining members
of the chosen orbit exceed its leader. Child stabilizers preserve these
restrictions. Repeating the argument gives coverage inductively. A bounded
stabilizer routine that returns fewer verified generators only weakens future
pruning; it cannot strengthen these restrictions using an unverified witness.

This pruning is a coverage method; it does not prove that every final
double orbit is emitted exactly once. The implementation deliberately retains
full ITS certificates to remove duplicates. Claiming unique enumeration at
this stage, or removing this exact membership state without a new proof, would
be incorrect.

For one mapping, the full ITS automorphism group is

    H = R intersect m^(-1) P m.

The stabilizer of m in R x P has |H| elements, so its labeled double orbit has
|R||P|/|H| mappings. The action of P on bijections is free, hence the number
of product-orbit representatives contributed by one full ITS class is

    w = |R|/|H|.

The accumulator verifies both divisibilities before using w. Equal full ITS
keys contribute once. Different ITS classes with the same template contribute
separately to that template. Incomplete canonicalization or group-order proofs
cannot supply an exact weight.

For an R-orbit O of atom or bond coordinates, suppose a representative changes
k coordinates in O. Averaging over R gives w*k/|O| changes at each coordinate
in the product quotient. Changes are invariant under H, so this is integral;
the code checks the remainder rather than rounding. Multiplication by |P|
recovers labeled frequencies.

Canonicalization refines exact colors, individualizes unresolved cells, and
prunes only with verified automorphisms. Hash/refinement agreement alone is
insufficient. Exact equal-row twins permit the shortcut used in the code.
Group-order factorization joins the entire support of each generator, including
all its cycles; using graph components alone could miss component exchanges.

Additional audit tests construct independent double-orbit keys from raw
transported matrices under explicit reactant permutations. Twelve graph pairs
cover signed and half-integer weights, multiple atom types, disconnected
symmetries, randomized feasible seeds and several shells per pair. They check
coverage, class multiplicities, labeled counts and coordinate frequencies,
then compare complete cache-on/cache-off accumulation. Existing tests cover
generic canonical certificates, loops, group-order overflow, public output,
collisions, tiny frontier slices and global budget exits.

## 4. Cache inventory and what the cache-disabled experiment establishes

| State | Equality/evidence used | Scope and role |
| --- | --- | --- |
| Python structure cache | Exact colored component equality, or a verified bijective colored isomorphism; refinement hashes only select candidates | Bounded per-observer reuse of completed canonical codes |
| Native pattern cache | Full sparse deviation vector from a fixed A/A baseline; explicit verified R-images | Per-worker, per-case shortcut for an already retained ITS class |
| Full ITS membership dictionaries | Entire typed palette and numeric canonical certificate | Required exact duplicate suppression for the present enumeration method |
| Palette, token, SHA-prefix and transport memoization | Exact immutable function inputs; SHA states are copied before extension | Reuses encoding work, library handles or immutable byte objects |
| Compiler artifact reuse | Source/compiler/flags/build metadata and binary digest | Reuses executable code, not a task result |
| GCC profile-guided build used in V10 | Recorded compiler execution profiles | Case-trained optimization; useful timings require separate generalization evidence |

The sparse pattern stores every off-diagonal paired-bond change and every
unary-color change. The controlled ITS diagonal is always absent, and the
baseline is fixed, so baseline plus pattern is injective. The C++ set compares
the entire vector even if hashes collide. A transformed pattern is inserted
only through a generator checked against every baseline node color and bond.

A pattern is remembered only after its class has been retained, or after a
full canonical lookup confirms a duplicate of an already retained class.
In the frontier worker, shared-cap reservation occurs before remembering a
new pattern. Cache eviction removes opportunities to skip work; it cannot
add a class or change its weight. The cache has no disk persistence, reaction
identifier lookup, saved-answer file or held-out mapping input. Inspection of
the calculation path found no answer-table shortcut.

SYNKIT_NATIVE_PATTERN_CACHE=0 disables the optional pattern shortcut.
It does not disable full ITS duplicate suppression or encoding memoization.
The ablation must therefore be described as “pattern cache disabled,” not
“all caches removed.” Exact sets are part of the counting algorithm.

One strict-proof limitation remains in reporting: emitted-reference membership
uses SHA-256 sets, and public class IDs are SHA-256 digests. A digest alone is
collision resistant, not a mathematical equality proof. Full ITS aggregation
retains exact payloads, so digest collisions do not merge its internal counts.
The literal mapping-presence flag, and the Python transported-reference flag,
still rely on a cryptographic assumption. This is a reporting limitation,
separate from the inspected native count/weight logic. Existing public IDs
must remain compatible while any stricter presence representation is added.

## 5. Frozen rerun and comparison protocol

The original selection is 1,200 tasks across 1,077 reactions: 751 minimum tasks
and 449 reference-CD tasks. It contains 1,197 historical time-limit stops and
three historical mapping-limit stops. Preserve every task, its mode, dataset
hash and blinding seed; never replace a minimum task with a reference-CD task.

Three campaigns are required:

1. Current frozen Python source, all 1,200 tasks, eight task workers on physical
   CPUs 0–7, 60-second cooperative search limits and the existing 100,000
   mapping cap.
2. Audited native source, all 449 reference-CD tasks, fresh portable non-PGO
   build, pattern cache enabled, sixteen workers per sequential case.
3. The same native source and binary, all 449 reference-CD tasks, pattern cache
   disabled, the same search and retention limits.

Native runs have a case deadline beginning before construction, a 1,000,000
worker-local retained-class cap including duplicate copies across workers,
and 4 GiB address space per worker. Measure through public output, JSON
serialization and file close; count strict success only if complete and
below 60 seconds. Interpreter import and dataset loading precede that clock.

The Python timer is the historical cooperative search timer; its preparation
and finalization are measured but not bounded by that timer. The mapping caps
also have different units. Consequently these campaigns are separate result
rows, not a homogeneous 1,200-task native claim. No aggregate service memory
cap is applied this time; per-worker address-space limits are recorded.
Concurrent native campaigns use disjoint physical cores but share host memory
bandwidth. Their final manifests record the actual CPU sets.

Compare minima, targets, exact labeled counts, normalized rational coordinate
frequencies, and full ITS/template class maps wherever both classifications
are complete. Preserve incomplete-classification statuses. Search-order
stream digests, visited-node counts and literal emitted-reference membership
can differ when enumeration order or symmetry representatives change.
The class maps are published digest-ID/count maps; raw canonical certificates
are not serialized by this benchmark. Corpus agreement therefore complements
the independent raw-matrix oracles, rather than replacing them.
Cache differential checks additionally require the whole structure object,
class weights and entropy to agree for complete paired runs.

If a 60-second cache-disabled run truncates, it is not a valid full-output
differential. Complete a separately labeled diagnostic with a larger allowance
and compare it to frozen complete outputs. Such a diagnostic cannot count as
a 60-second recovery.

## 6. Enhancement sequence and acceptance gates

**P0 — Finish the audit evidence and reporting contract.** Retain the reproduced
numeric and mixed-type counterexamples, the input guard and the frozen corrected
binary. Complete the campaigns,
record every failure, and distinguish search completeness, structure
completeness and strict wall-time success. Preserve the known historical
provenance-test failure without rewriting its evidence. Add exact packed
mapping witnesses or an explicitly labeled collision-assumed presence mode
before advertising mathematical exactness for all reference-presence fields.
Measure the memory/transfer cost of that change separately.

**P1a — Complete structure classification without changing minimum search.**
The full Python rerun leaves one minimum task, 13067, with an incomplete ITS
classification. Add an explicit native canonicalization backend to the
existing Python structure observer. Keep the minimum proof, mapping stream,
unit weights and frequency accumulation unchanged. The canonical backend
must return the same dense certificate and public ID as the Python backend,
or return an incomplete status. A certificate-only call can omit an unused
group-order calculation; it must not omit canonical-search proof. Test exact
IDs and multiplicities on all prior complete structures, then run all minimum
tasks with the same structure budgets. This is a smaller integration step
than replacing the minimum search itself, and applies generically without a
case-ID exception.

**P1b — Make minimum and fixed-target modes share an explicit native contract.**
Add a proof phase returning a feasible incumbent U and certified frontier
lower bounds. A heuristic seed supplies only an upper bound after its full
objective is checked. Prove optimality when every remaining subtree has lower
bound at least U. If the root bound equals U, that equality is already the
optimum certificate. Enumerate the shell at U with the remaining case budget,
retaining equality branches. The result is complete only when both proof and
shell finish. Use one monotonic case deadline across seed, proof, enumeration,
classification and output. Keep minimum-mode reference information outside
both phases. Validate every small typed permutation instance and every one
of the 751 actual minimum tasks before integrating this backend into a
uniform full-cohort benchmark.

**P2 — Reduce assignment work without cached answers.** Maintain per-depth
residual profiles and cross-cost arrays with exact undo records. Reuse a
parent LAP matching/dual only after repairing dual feasibility for changed
costs and invalidated edges; recompute when repair is more expensive. A
feasible dual sum is an admissible bound even before optimization. Forced-edge
shortest-path filtering, however, requires an optimal matching and its valid
dual, so it must wait for that certificate. Instrument LAP calls, augmentations,
changed rows and prunes. Gate on identical oracle outputs and lower wall time
in the cache-disabled build, including signed weights and equality targets.

**P3 — Reduce classification and coordinator cost.** Keep a batch's typed
palette once, pack numeric records contiguously, and compute canonical
certificates/group proofs with reusable per-call scratch buffers. Bound
batches by estimated serialized bytes and observed classification time as
well as node count. Never yield halfway through committing an exact class:
a subtree or retained record must have one clear owner. Maintain deterministic
prefix replay, exact duplicate checks and cap reservation before publication.
Measure wall time, aggregate CPU time, bytes transferred, parent RSS and
worker RSS. Per-worker timings overlap and must not be added as elapsed time.

**P4 — Explore unique double-orbit generation as a separate algorithm.**
Represent a partial bijection as two fixed colored side graphs joined by
matching edges. Its automorphisms describe the coupled R x P action on that
partial mapping. A canonical construction path would choose an invariant
orbit of deletable matching edges and accept a child only when its new edge
belongs to that orbit, with one extension per parent-automorphism orbit.
This is a proposed application of the canonical construction-path method;
its uniqueness theorem depends on verified augmentation/reduction conditions.
See [McKay, Isomorph-free exhaustive generation, Theorem 1 and its conditions](https://users.cecs.anu.edu.au/~bdm/papers/orderly.pdf).
Prove those conditions for these partial mapping objects and test a small
prototype before combining it with distance pruning. The two side graphs require distinct side colors, and matching edges need a
distinct edge color, so an automorphism cannot reverse the reaction or confuse
a bond with a mapping edge. Object order is the number of matching edges.
Use intrinsic partial-state distance bounds in the prototype; the present
asymmetric row restrictions need a separate compatibility proof before reuse.
An augmented graph has 2n vertices, so the current 256-vertex canonical API
would need extension or a different coupled-group representation for n>128.
This may reduce repeated full canonicalization, but its added canonical tests
may cost more than they save. Keep the existing exact membership path until equivalence is established.

**P5 — Establish portable performance and a homogeneous corpus result.**
Use the measured pattern-cache-disabled configuration as the immediate portable
benchmark baseline. Cache-enabled and cache-disabled code perform the same
counting computation, but lookup, pattern expansion and memory traffic can
outweigh the canonicalization saved. The cache-disabled cohort completed
30474 below 60 seconds while the enabled run did not. Do not infer a universal
speedup from different CPU sets and concurrent host load; keep cache policy
explicit and test any adaptive policy on held-out cases.
Make the portable build the algorithmic baseline. Report optional PGO separately,
record training tasks, and evaluate on held-out task families before claiming
general performance. For 30474 and 22361, aim for at most 50–55 seconds to
provide margin, then require at least three fresh-process full-output runs
below 60 seconds with the declared memory and retention units. Apply the same
frozen configuration to all 1,200 tasks once both modes are supported. Publish
all timeouts/errors and source/binary hashes, not only recovered cases.

These steps provide mathematical obligations and measurable gates. They do
not promise that a new pruning rule, a larger cache or a compiler change will
finish every case in 60 seconds. Performance claims follow completed runs.


## Execution record (V11)

Implementation, proof obligations, validation and the homogeneous 1,200-task
rerun are recorded in [V11 execution report](ENUMERATION_ENHANCEMENT_V11_2026-09-09.md).
This execution implements exact witnesses, both shell modes, certificate-only
classification and a zero-slack assignment specialization. Parent-dual warm
starts and a fully packed record transport remain proposals, not deployed claims.
The canonical-augmentation prototype is deliberately separate from production.

The first execution exposed a timing-sensitive optional seed path. The generic
CPU-budget correction and a second complete frozen cohort are documented in
[V12 execution report](ENUMERATION_ENHANCEMENT_V12_2026-09-09.md). The first cohort's
timeout is preserved; the second cohort does not reuse its completed records.

The prefix replay subsequently showed that CPU accounting alone was insufficient:
the per-fragment allowance could expire during MCS initialization. The bounded
first-progress correction and final complete cohort are documented in
[V13 execution report](ENUMERATION_ENHANCEMENT_V13_2026-09-09.md).

A direct native lower-bound witness subsequently proved 17156's optimum without a
heuristic seed. The final bounded probe, its fallback proof, validation and fresh
full cohort are documented in [V14 execution report](ENUMERATION_ENHANCEMENT_V14_2026-09-09.md).

The final output-only step avoids copying immutable class-count tuples before JSON
serialization. V14 cohort timing and final formatter-specific fresh checks are kept
distinct in the [final report](ENUMERATION_ENHANCEMENT_V15_2026-09-09.md).
