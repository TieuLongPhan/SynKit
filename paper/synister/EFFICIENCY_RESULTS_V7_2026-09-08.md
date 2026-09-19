# Synister efficiency, round seven — 8 September 2026

Improved structure classification for the V6 recovered searches: **17/17
now have complete ITS/template classification**, compared with 9/17 in V6.
The same 17 exact searches complete; the three reference-CD searches at source
lines 28163, 30474 and 22361 remain incomplete. No additional one of these
three searches was solved in this round.

Across the full 162-task verification cohort, complete search plus structure
results increased from **124 to 159**. All 159 previously completed
searches remain complete and pass the mathematical regression audit, with no
loss of prior structure completeness in this run. Final run: 159 complete,
three timeouts, zero errors in 104.61 seconds.

| Source line | Mode | Exact search | Full task wall (s) | Structure complete |
| --- | --- | --- | ---: | --- |
| 27079 | minimal | complete | 6.18 | yes |
| 11365 | minimal | complete | 2.76 | yes |
| 26509 | reference_cd | complete | 4.85 | yes |
| 1346 | minimal | complete | 5.06 | yes |
| 34715 | reference_cd | complete | 1.54 | yes |
| 35277 | minimal | complete | 4.95 | yes |
| 28163 | reference_cd | timeout | 60.85 | no |
| 19926 | reference_cd | complete | 3.87 | yes |
| 30474 | reference_cd | timeout | 60.99 | no |
| 20274 | reference_cd | complete | 6.77 | yes |
| 17156 | minimal | complete | 20.32 | yes |
| 33737 | reference_cd | complete | 27.74 | yes |
| 13067 | minimal | complete | 7.94 | yes |
| 22691 | reference_cd | complete | 1.70 | yes |
| 24936 | minimal | complete | 2.04 | yes |
| 7124 | reference_cd | complete | 7.13 | yes |
| 32398 | minimal | complete | 2.60 | yes |
| 22361 | reference_cd | timeout | 60.75 | no |
| 747 | minimal | complete | 5.97 | yes |
| 6728 | reference_cd | complete | 35.44 | yes |

## Changes

The native exact canonicalizer now seeds swaps between identically colored
nonadjacent twin vertices. Candidates require identical full incoming and
outgoing neighborhoods, and each permutation is verified against the full
colored graph before it can prune a branch. Discovery uses the existing
search time budget and retains at most 256 initial generators; incomplete
searches still expose no canonical identity.

For larger stabilizer workloads, partition checks inspect cached moved
vertex pairs of immutable verified generators. Fixed vertices need no
partition comparison. Sibling branches reuse a stabilizer until a new
generator is registered. Both changes preserve the existing canonical
certificate and class identifier conventions.

The structure cache materializes disconnected components once for repeated
adjacency traversals instead of repeatedly traversing filtered graph views.
Cache hits still require exact color and edge preservation.

## Verification and controls

- 242 tests passed; one optional test skipped and the known historical frozen
  provenance test deselected. Ruff correctness checks and whitespace checks
  passed.
- Additional tests compare canonical codes to exhaustive enumeration for all
  graph-atlas graphs through five vertices, under relabeling and both normal
  and forced sparse stabilizer paths. Directed colored examples cover incoming
  edges, loops, and independently enumerated automorphism groups. A symmetric
  star control verifies reduced visited nodes without a timing assertion.
- The audit compares all 159 complete records from the same frozen V6
  verification run. It checks completion, minima, reference flags, labeled
  counts and normalized reaction-center frequencies. Labeled ITS/template
  class counts are compared where both analyses are complete, and loss of
  prior structure completeness is checked separately.
- The initial candidate improved classification but caused
  29867/reference-CD to time out: successful classification added enough work
  to exhaust the search timer. After materializing cached components, its
  targeted rerun completed in 47.11 seconds. The final full cohort verifies
  that the corrected candidate preserves its completion.
- A component-symmetry discovery experiment was not retained.
- The final source snapshot remained unchanged and matches the current package
  across all hashed Python files:
  26cd4821add322a4fb58b8fe72e84c3cda82fbcd504ef30886663ffe6df72ed8.

## Limits and next work

The 60-second cooperative search timer, 100,000-mapping cap, 0.25-second
structure-canonicalization budget, eight workers on CPUs 0–7, and 4 GiB
address-space limit per worker are unchanged. Full task wall includes seed
preprocessing and can exceed the cooperative search timer.

The three remaining shells still time out. V6 diagnostics with structure
classification disabled and a longer timer also reached the mapping cap;
those diagnostics are not V7 validation completions. Further exact
aggregation would need to preserve multiplicities, atom/bond-change
frequencies and held-out reference membership. Faster classification alone
does not establish completion of these large shells.

This version has not been rerun on all 1,200 expansion tasks. Timing and
classification completeness near budget boundaries may vary across runs.

## Artifacts

Round-seven artifacts are in
benchmark_results/synister_efficiency_timeouts_v7_20260908/:

- verified_source/, verified_run/, verification_selection.json;
- completion_summary.json, remaining_after_verification.json;
- regression_references.json, audit_regressions.py,
  verified_run/regression_audit.json;
- tests_verified.log, canonical_control.json and the 29867 profile;
- exploratory snapshots and runs, including the initial candidate's regression.

The authoritative final run is verified_run/; the earlier directory named
final_verification/ retains the pre-cache-fix candidate for review.
Changes remain local and uncommitted.
