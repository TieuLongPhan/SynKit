**Synister timeout efficiency, second round — 7 September 2026**

Further improvements and the remaining timeout list are in the
[third-round report](EFFICIENCY_RESULTS_V3_2026-09-07.md). These second-round
measurements are preserved.

The second upgrade increased completions from 31 to 37 across 64 historical
timeout tasks at the unchanged 60-second search limit. The full 10,000-case
campaign was not rerun. The comparison starts from the bounded-seed-repair
version in the [first-round report](EFFICIENCY_RESULTS_2026-09-07.md).

| Cohort | Previous complete | New complete | Previous batch wall | New batch wall |
| --- | ---: | ---: | ---: | ---: |
| Development, 32 tasks | 15 | 19 | 181.5 s | 143.7 s |
| Held-out, 32 tasks | 16 | 18 | 175.3 s | 146.4 s |
| Total, 64 tasks | 31 | 37 | — | — |

All 31 shared completions had matching checked mathematical outputs; their
median paired task-wall speedup was 3.49 times. No previously completed task
became incomplete in these runs. There remain 27 timeouts: 22 minimal-mode
and five reference-CD tasks. These cohorts do not establish population-wide
recovery rates or speedups.

**Implementation**

- Defer child stabilizer construction until after timeout, bound, and leaf
  checks, avoiding group computation for rejected branches and leaves.
- Reuse cross-cost matrix deltas during recursive unwinding. A cumulative
  32 MiB retention budget bounds memory; deeper branches recompute on unwind.
- Strengthen the assignment lower bound using incident bond profiles. Prefix
  profile cost plus an element-blocked linear assignment on remaining atoms
  bounds the full objective. Combine it with the existing bound by a maximum,
  accounting for committed cost, rather than adding the two bounds.

The profile bound also applies to uncertified minimal searches. Certified
searches retain existing independently replayable bounds. Numeric domain
filtering remains specific to numeric targets. Statistics record stabilizer
calls, retained delta memory, and profile-bound calls/pruning.

**Controls and results**

Each selection has 28 minimal and four reference-CD tasks, all historically
still timed out at 120 seconds. Development uses the first-round selection.
The additional held-out source reactions are disjoint and were selected
deterministically and frozen before evaluating the new bound. Paired runs use
identical inputs, task selections, eight spawn workers, CPU IDs, one numerical
thread per worker, and 4 GiB per-worker address-space limits. Every benchmark
retained its initial source hash and produced zero error records.

New development completions: source lines 36545 and 34569, minimal mode,
36.54 and 57.13 seconds; 7190 and 25457, reference-CD, 5.51 and 4.59 seconds.
New held-out completions: 20691, minimal, 4.64 seconds; 14961, reference-CD,
51.20 seconds. The earlier borderline reference case 7190 now completes in
5.51 seconds; this does not establish the cause of its earlier timing discrepancy.

Summed task wall fell from 1,191.1 to 952.7 seconds in development and from
1,157.2 to 980.5 seconds in held-out data. These include censored timeouts and
measure resource use, not uncensored solution times. Peak worker lifetime RSS
rose from 201.5 to 240.3 MiB in development and 197.3 to 213.1 MiB in held-out
data. These are worker high-water marks. Full task wall includes preparation
and finalization and can exceed the cooperative 60-second search timer.

**Validation**

Mapper functional suite: 113 passed, one optional skip, one known pre-existing
provenance test excluded:
`test_frozen_alternative_its_case_is_bound_to_current_implementation`.
The frozen historical artifact already mismatched the pre-enhancement source
hash; its record was preserved. Selected lint and whitespace checks passed.

New tests cover delta-cache budgets and certificate equivalence, deferred
stabilizers, a positive minimum proved at the root, and profile-bound
admissibility against exhaustive typed bijections at every prefix for eight
small directed/undirected weighted examples, including decimal weights.

Paired output checks cover minima, reference-class presence, known labeled
counts, normalized atom/bond reaction-center frequencies when quotient
analysis is complete, and ITS/template labeled class counts when both
structural analyses are complete. All 31 shared completions matched.

Archived minimum certificates for 35488, 39584, and 41004 replayed successfully
with the current verifier and agreed with new minima. A new reference-CD
certificate for 7190 verified six representatives and 294,912 labeled mappings,
matching the benchmark count. This separate certificate-generation-and-replay
check took 644.9 seconds; it was outside the benchmark and is not a claim of
60-second verification.

**Artifacts**

Second-round data: `benchmark_results/synister_efficiency_timeouts_v2_20260907/`.

- `development_selection.json`, `heldout_selection.json`: frozen tasks.
- `branch_source/`, `profile_source/`: intermediate and final source snapshots.
- `branch_development/`: intermediate changes completed 17 of 32 tasks.
- `baseline_heldout/`, `profile_development/`, `profile_heldout/`: manifests,
  full per-task results and summaries. Development baseline is the first-round
  repaired-source run.
- `comparison.json`: paired output checks and performance metrics.
- `profile_bound_diagnostic.json`: bound and incumbent diagnostics.
- `certificate_checks/`: verification summary and new compressed certificate.

Use `scripts/benchmark_synister_timeouts.py` with these frozen selections and
source snapshots and a fresh output directory for repetitions. The current
package matches the final benchmark snapshot. Changes are local and uncommitted.
