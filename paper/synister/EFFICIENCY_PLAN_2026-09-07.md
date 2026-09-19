**Synister efficiency plan — 7 September 2026**

Improve completed exact shells at the existing 60-second search budget. Preserve
weighted chemical distance, the global search space, reference blinding, and
complete optimizer enumeration. The 120-second campaign supplies diagnostic
evidence; increasing its timeout is not part of this plan.

Initial implementation and timeout-only evaluation are now recorded in
[EFFICIENCY_RESULTS_2026-09-07.md](EFFICIENCY_RESULTS_2026-09-07.md). Symmetry fast
paths, early branch rejection, minimum-proof pruning, bounded seed repair and
basic stage diagnostics are implemented. The
[second-round report](EFFICIENCY_RESULTS_V2_2026-09-07.md) records deferred
stabilizers, bounded matrix-delta reuse, and a stronger profile assignment bound
on 32 development and 32 held-out timeout tasks. The sequence below also retains
proposals beyond these implemented changes. At the user’s request, performance cohorts now use timeout records
only; the earlier mixed cohort containing completed baseline cases is superseded.

The [third-round report](EFFICIENCY_RESULTS_V3_2026-09-07.md) adds bounded broad
seed descent and a profile bound conditioned on assigned atoms. That round completed
45 of the 64 evaluated timeout tasks, with 19 remaining tasks saved for targeted
diagnosis. The [fourth-round report](EFFICIENCY_RESULTS_V4_2026-09-08.md)
records completion of all 64 selected tasks, including those final 19, through
stronger seed construction, exact domain propagation, dynamic branching, and
bounded structure caching. The slowest final task took 28.79 seconds. The
configured search limit remains 60 seconds; no full 10,000-case rerun was made.

**Evidence and priorities**

The paired comparison uses source line and mode, rather than treating every
completed follow-up shell as newly recovered.

| Observation | Interpretation |
| --- | --- |
| 205 of 1,881 reference-CD timeouts completed at 120 seconds (10.9%). | Extra search time recovered relatively few cases. |
| 236 of 2,999 minimal-mode timeouts completed at 120 seconds (7.9%). | Minimum proof needs algorithmic improvement. |
| Of 2,763 remaining minimal-mode timeouts, 2,739 have no proven minimum; only 24 have a known minimum. | Prioritize the optimization pass before optimizing spectrum generation. |
| 1,400 remaining minimal-mode timeouts visited no complete assignment. | Branch ordering, initial feasible mappings, and bounds deserve attention. |
| The seed cost exceeds the held-out reference CD in 2,752 remaining minimal timeouts; the median excess is 33 CD units. | Initial feasible upper bounds are often weak. Reference CD is used here only for retrospective evaluation. |
| Every recorded timeout uses assignment branch-and-bound. | The existing edit-support backend cannot accelerate the campaign through a configuration switch. |
| Five reference-CD follow-up records have status `timeout` but reason `mapping_limit`. | Track output caps separately from elapsed-time failures. |

Three minimal-mode profiles used the synkit interpreter, a 3-second search
budget, and the current code including the timeout-unwinding fix:

| Source line / reaction | Atoms | Observed dominant work |
| --- | ---: | --- |
| 8297 / 6189:1 | 31 | Stabilizer construction: 2.50 of 3.05 seconds, about 82%. |
| 19599 / 14742:2 | 80 | Stabilizer construction: 2.67 of 3.10 seconds, about 86%. |
| 15041 / 11285:1 | 171 | Full-matrix cross-cost updates: 1.64 of 3.65 seconds, about 45%. |

These are diagnostic samples, not estimates of campaign-wide speedup. Python
profiling changes overhead and can change how much bounded symmetry discovery
finishes. Validate improvements with unprofiled paired runs. The full analysis
wall time also includes work outside the current enumeration timer.

Machine-readable metrics, profile summaries, and implementation provenance are
in [EFFICIENCY_DIAGNOSTICS_2026-09-07.json](EFFICIENCY_DIAGNOSTICS_2026-09-07.json).
Historical settings and counts are in
[SESSION_LOG_2026-09-07.md](SESSION_LOG_2026-09-07.md).

**Implementation sequence**

| Order | Change | Main files | Evidence to collect |
| --- | --- | --- | --- |
| 1 | Add stage timing and freeze a small comparison cohort. | `analysis.py`, `exact/distance.py`, benchmark harness | Seed, preprocessing, minimum proof, enumeration, observer and total wall time; incumbent cost, root bound, node counts and termination stage. |
| 2 | Reduce repeated symmetry and branch bookkeeping. | `exact/symmetry.py`, `exact/distance.py` | Stabilizer calls/cache hits, matrix-update calls, nodes per second, memory and exact output equality. |
| 3 | Make the minimum-proof pass search only for a strictly better mapping. | `exact/distance.py` | Time/nodes to prove the same minimum; full optimizer sets in the subsequent pass. |
| 4 | Improve the initial mapping and admissible bounds. | `analysis.py`, `exact/enumerate.py`, `exact/distance_bounds.py` | Incumbent quality, root bound, proof nodes and completed cases after charging seed cost. |
| 5 | Reduce large-molecule matrix work and repeated preprocessing. | `exact/distance.py`, `exact/distance_bounds.py`, `analysis.py` | Time per branch, allocation/memory cost, full wall time and larger-case completion. |

For step 2, first add an exact fast path when every generator already fixes the
chosen product atom: return the existing subgroup instead of reconstructing it.
Compute inverse orbit transporters once per orbit rather than inside every
generator iteration. Add a bounded per-case cache of checked stabilizers and
orbit data, retaining completion flags and accounting for cache memory.

Move the symmetry-rejection check ahead of residual-mass updates, matrix
updates, and next-stabilizer construction. Construct prefix tuples only where
certificates require them. Preserve certificate frontier/witness coverage and
mapping-limit completion behavior. Cache keys must identify the verified group
and stabilizer state; depth alone is insufficient.

For step 3, the current `_optimization_only` pass still uses the inclusive bound
needed for enumerating all minimizers. Once a feasible incumbent is available,
the proof pass can reject a branch whose admissible lower bound proves that it
cannot improve that incumbent. Specify conservative floating-point comparisons
before changing the bound checks. Keep the second pass inclusive so all equal
minima, counts, and spectra remain available. Retain the incumbent as a witness
even when the proof finishes without visiting another leaf. Reuse safe,
input-derived preprocessing across the two passes.

For step 4, audit how the first SLAP result becomes a complete mapping. Evaluate
a small deterministic set of reference-free candidates and bounded
element-preserving swap repair; existing repair helpers are starting points.
Measure the cost of repair and retain the best valid full mapping. A heuristic
candidate supplies an upper bound and ordering only; agreement among heuristic
maps does not prove that any atom assignment is fixed.

The current atom-profile domain filter is disabled during minimal search.
Evaluate its admissible use against the incumbent, and a stronger
element-compatible assignment bound based on incident-bond profiles. Develop
residual bounds without counting the same edge cost twice: combine overlapping
lower bounds with a maximum unless a disjoint-cost derivation justifies adding
them. Apply cheap bounds before expensive assignment solves. Start with static
ordering by safe domain size and constraint strength; dynamic ordering requires
separate changes to residual accounting and symmetry/certificate state.

For step 5, `update_cross_costs` currently recomputes a dense difference both on
push and pop. Compare a bounded stored delta with recomputation, and restrict
updates to entries needed by descendants with correct restoration on backtrack.
Precompute reactant element blocks and maintain active product blocks instead
of rebuilding both for every assignment bound. Sparse updates are a later
option, with the dense calculation retained as the correctness oracle.

Share adjacency, property vectors, reference-free seeds and verified product
symmetry where valid. Keep target-dependent state separate: results obtained
using the reference CD must not inform the blinded minimal-mode search. Charge
all extra seed/cache preparation in full-wall-time comparisons.

**Validation and adoption**

1. Start with the frozen 32-task development cohort of persistent 120-second
   timeouts (28 minimal, four reference-CD). Expand using only timeout records,
   stratifying minimum-proof failures, enumeration failures, atom count and
   symmetry size. Freeze a held-out timeout subset before further tuning and
   keep source reactions disjoint between development and held-out subsets.
   Track mapping-limit cases separately. Do not rerun the complete 10,000-case
   dataset. Report this selected cohort separately from population-level claims.
2. Use the same 60-second configuration, interpreter, input hashes, worker
   count, thread limits, output cap and memory policy for each paired comparison.
   Start serially to explain per-case gains; validate throughput with the
   existing 16-worker policy afterward. Use a fresh manifest/output directory
   for each implementation. Record full wall time, including preparation and
   finalization, so moving work outside the search timer cannot count as a win.
3. Require exact agreement on tiny exhaustive weighted and binary controls:
   minimum values, numeric shells, complete optimizer sets, symmetry expansion,
   counts and spectra. Cover zero-CD symmetric graphs, nontrivial minima,
   directed/weighted inputs, floating tolerances, fixed mappings, deadlines,
   mapping caps, and certificate replay. Keep reference-blinding tests.
4. Existing completed real cases must retain their mathematical outputs.
   Ordering changes may alter a stream digest, so compare normalized sets or
   aggregate mathematical results as appropriate, with a new implementation
   manifest. Independently validate newly completed cases where a reference
   solver can finish; agreement with a held-out mapping alone is insufficient.
5. Adopt changes that pass correctness and memory checks and show repeatable
   gains on held-out cases. Primary measure: additional complete shells within
   the same budget, separately by mode. Also report time to proven minimum,
   total wall time, peak memory, and failures of previously completed cases.
   Treat unfinished runs as censored; compare runtimes on paired completed
   cases and do not present timeout cutoffs as exact solution times.

Implement and measure one change at a time before combining winners. A useful
proposed milestone is recovering at least 10% of persistent minimal proof
failures in the held-out cohort at 60 seconds, with no correctness loss or
repeatable completion regression. This is a target, not a predicted speedup.

**Later work, conditional on remaining bottlenecks**

The current edit-support backend requires a numeric binary objective and
labeled output; this campaign uses weighted bonds and symmetry-aware output.
A compatible weighted edit-support solver with verified symmetry handling
would be a separate development project. Changing `binary` or fixing a
heuristic reaction exterior would change the experiment's scientific scope.

Consider a stronger stabilizer-chain implementation if the simpler symmetry
changes leave it dominant. Optimize or cache exact structure canonicalization
after minimum proof improves; only 24 of the present minimal timeouts are known
to have reached the enumeration stage. Decomposition needs an exact argument
for cross-component assignments and boundary costs. Per-case parallel search
is lower priority because the campaign already parallelizes reactions and
would need a fixed aggregate CPU budget.

The first implementation batch should contain stage diagnostics, symmetry
fast paths, early rejection of symmetry-pruned branches, and the isolated
strict-improvement proof pass. Run correctness checks and the development
cohort before proceeding to seed and bound changes.
