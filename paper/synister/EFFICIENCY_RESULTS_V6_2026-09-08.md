# Synister timeout efficiency, round six — 8 September 2026

Recovered **17 of the 20 remaining exact-search timeout tasks** at the existing
60-second search limit and 100,000-mapping cap. Three reference-CD tasks remain
incomplete: 28163, 30474 and 22361. This does not complete all 20 tasks.

Final verification processed 162 tasks in 115.06 seconds: **159 complete,
three timeouts, zero errors**. This consists of the 20 targets plus 142 saved
completed regressions (the previous 96-case regression cohort and 46 V5
recoveries). All 142 regressions completed and passed the checked mathematical
invariants, with no structure-completeness regressions in this run.

Of the 17 recovered searches, nine also have complete ITS/template structure
classification; eight still have incomplete structure classification. All 17
have complete symmetry quotient analysis and known labeled counts. Search
completion must not be reported as full structure-classification completion.

| Source line | Mode | Exact search | Full task wall (s) | Structure complete |
| --- | --- | --- | ---: | --- |
| 27079 | minimal | complete | 4.96 | no |
| 11365 | minimal | complete | 2.62 | yes |
| 26509 | reference_cd | complete | 4.38 | no |
| 1346 | minimal | complete | 4.41 | no |
| 34715 | reference_cd | complete | 1.68 | yes |
| 35277 | minimal | complete | 4.28 | no |
| 28163 | reference_cd | timeout | 60.88 | no |
| 19926 | reference_cd | complete | 3.64 | no |
| 30474 | reference_cd | timeout | 61.04 | no |
| 20274 | reference_cd | complete | 6.83 | yes |
| 17156 | minimal | complete | 21.85 | no |
| 33737 | reference_cd | complete | 49.29 | yes |
| 13067 | minimal | complete | 7.15 | no |
| 22691 | reference_cd | complete | 1.91 | yes |
| 24936 | minimal | complete | 2.33 | yes |
| 7124 | reference_cd | complete | 7.30 | yes |
| 32398 | minimal | complete | 2.93 | yes |
| 22361 | reference_cd | timeout | 60.77 | no |
| 747 | minimal | complete | 4.71 | no |
| 6728 | reference_cd | complete | 37.14 | yes |

## Implementation

- Bound individual fragment MCS attempts to 25 ms within each existing
  0.5-second fragment cover. When the seed remains above the graph-only lower
  bound, try one reverse-reactant-order cover, restore the mapping and verify
  its full objective cost before acceptance. This adds at most one cover
  attempt; preprocessing is outside the cooperative exact-search timer.
- Compute half-integer row-profile distances using cumulative histograms with
  a sorted-distance fallback for general weights. Use prefix-conditioned
  assignment bounds and feasible-image row ordering during uncertified exact
  search. Forced-assignment bounds reject only provably infeasible branches;
  certificate search retains its existing ordering.
- Retain bounded, explicitly verified automorphism witnesses after incomplete
  canonical searches. Incomplete canonical results still provide no canonical
  code. Avoid redundant stabilizer operations for points already fixed by all
  generators, and verify matrix permutations using NumPy.
- Refine structure-cache candidates, retain up to 256 entries with LRU reuse,
  and try a fully checked color-preserving isomorphism before bounded VF2.
  Refinement hashes only filter candidates; they never establish equivalence.
  Canonical codes and public class identifiers retain their semantics.

For source line 13067, a reference-free typed bijection has cost 6 and attains
the recomputed row-profile assignment lower bound of 6. The witness is saved
in 13067_minimum_check.json; no reference mapping was used to build it.
This recomputation uses the same bound implementation, not a second independent
implementation of that bound.

## Remaining shells and diagnostic limits

All three remaining tasks time out in the final run with structure analysis
enabled. An exploratory frozen snapshot was also run with structure analysis
disabled and a 600-second search allowance, retaining the 100,000-mapping cap:

| Source line | Full task wall (s) | Representatives emitted | Product group order | Stop reason |
| --- | ---: | ---: | ---: | --- |
| 28163 | 427.06 | 100,000 | 49,152 | mapping_limit |
| 30474 | 584.95 | 100,000 | 221,184 | mapping_limit |
| 22361 | 280.99 | 100,000 | 1,536 | mapping_limit |

Each diagnostic reaches 100,000 representatives with search work remaining.
Those partial counts are not final shell sizes, and these runs are not
completions under the validation configuration. Longer time alone did not
resolve the retained mapping cap. Separate complete product-canonicalization
checks reproduce the same product group orders, so these caps cannot be
attributed merely to incomplete discovery of that configured product group.
Finishing these shells requires further enumeration/aggregation improvements
or a changed mapping budget, followed by validation with structure analysis
enabled. The current limits have been preserved.

## Validation and scope

- 238 tests passed across Test/Graph/Canon and Test/Chem/Mapper/module;
  one optional test skipped and the known frozen historical provenance test
  deselected. Correctness-focused Ruff checks and git diff --check passed.
- New tests compare dynamic/profile search against brute-force typed
  permutations for directed and undirected weighted graphs, including signed,
  half-integer and general float weights, fixed mappings and symmetry expansion.
  Additional tests check histogram bounds, verified partial automorphisms,
  canonical budget exits, fixed-point stabilizers and cache hash collisions.
- Regression checks compare completion, minima, reference-class flags, known
  labeled counts and normalized atom/bond-change frequencies. Labeled
  ITS/template class counts are compared where both analyses are complete;
  loss of prior structure completeness is checked separately.
- Final verification uses eight workers on CPUs 0–7, one numerical-library
  thread per worker and a 4 GiB address-space limit per worker. Cooperative
  search timeouts are not strict whole-task wall limits; full wall times above
  include preprocessing. The slowest recovered target took 49.30 seconds.
- The final frozen source remained unchanged during verification and matches
  the current package byte-for-byte across hashed Python files:
  d7c9c30ec6e525652c5fb5434aca30a72080847433e2fdaa7f1faa5d3c039e86.
- This version was not rerun across all 1,200 expansion tasks. Successes from
  different historical versions must not be described as one new full run.

## Artifacts

All round-six artifacts are under
benchmark_results/synister_efficiency_timeouts_v6_20260908/:

- final_source/, final_verification/, verification_selection.json;
- completion_summary.json, remaining_after_verification.json;
- regression_references.json, audit_regressions.py,
  final_verification/regression_audit.json;
- exploratory snapshots/runs, no_structure_size_diagnostic/,
  large_shell_symmetry_check.json, and 13067_minimum_check.json.

Changes remain local and uncommitted.
