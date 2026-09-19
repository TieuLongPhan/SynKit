**Synister timeout efficiency, third round — 7 September 2026**

This round targets the remaining timeouts from the
[second-round report](EFFICIENCY_RESULTS_V2_2026-09-07.md). It reuses the same
32 development and 32 validation tasks, each historically still timed out at
120 seconds. The artifact name `heldout` denotes the fixed validation cohort
from round two; it is reused here, not a newly unseen sample. The full
10,000-case campaign was not rerun. Search limits remain 60 seconds.

The subsequent [fourth round](EFFICIENCY_RESULTS_V4_2026-09-08.md) resolved all
19 remaining tasks, reaching 64/64 at the same 60-second limit. Results below
are the preserved third-round measurements.

**Measured results**

| Cohort | Previous complete | New complete | Previous batch wall | New batch wall |
| --- | ---: | ---: | ---: | ---: |
| Development, 32 tasks | 19 | 22 | 143.7 s | 124.5 s |
| Validation, 32 tasks | 18 | 23 | 146.4 s | 125.1 s |
| Total, 64 tasks | 37 | 45 | — | — |

Eight additional tasks completed: seven minimal-mode and one reference-CD.
All 37 previous completions were retained in these runs, with matching checked
mathematical outputs. Their median paired wall speedup was 1.06 times; the main
gain in this round is additional completed tasks, not a uniform speed increase.
Broader seed repair lowered starting costs on 19 of 56 minimal-mode tasks.

New development completions were source lines 40346 (11.71 seconds), 40575
(2.16 seconds), and 11232 (49.40 seconds), all minimal. Validation gains were
27970 (40.62 seconds), 39577 (30.00 seconds), 31215 (7.40 seconds), and 4723
(3.16 seconds) in minimal mode, plus 32156 (3.22 seconds) in reference-CD mode.

There remain 19 timeouts: 15 minimal and four reference-CD. Of these minimal
tasks, line 29763 now has a proven minimum of 10.5 but incomplete optimizer
enumeration; the other 14 still lack a proven minimum. The next diagnostic
selection contains only these remaining tasks, with stage, incumbent, and
bound-pruning records saved separately.

Summed observed task wall fell from 952.7 to 752.3 seconds in development and
980.5 to 723.6 seconds in validation. These include censored timeouts, so they
measure observed resource use rather than uncensored solution time. Peak worker
lifetime RSS increased from 240.3 to 281.0 MiB and from 213.1 to 255.3 MiB,
respectively. All runs had zero errors and unchanged package hashes; no seed
repair errors were recorded. Current source matches the final frozen snapshot.

**Implementation**

`exact/seed.py` batches objective deltas for same-element pair swaps, extending
seed repair beyond the previous small pool. It evaluates at most 64 descent
steps on a pool of at most 256 atoms, with batches of at most 256 pairs.
Changed rows and columns exclude their overlapping block, which is counted
once. The calculation supports asymmetric matrices, decimal/signed weights,
and diagonals. Every accepted swap also passes a freshly recomputed full-cost
check. Larger molecules select the pool by mismatch score. This heuristic
provides only a feasible incumbent and search ordering; it does not fix domains
or use the reference mapping. Existing bounded repair runs first, and optional
repair failures retain the latest verified feasible mapping.

The seed-only development experiment improved nine seeds but retained the same
19 completions. Examples: line 15041 fell from 210 to 126, and line 29763 from
51.5 to 10.5. Its batch wall fell from 143.7 to 130.5 seconds. Better seeds alone
did not resolve these remaining enumeration and proof bottlenecks.

`exact/distance_bounds.py` now computes a profile assignment lower bound on the
remaining subgraphs, conditioned on the assigned prefix. For each remaining
atom pair, exact cross costs to the prefix and the relaxed internal outgoing
row cost are combined before solving one element-compatible linear assignment.
This enforces a common assignment for the two contributions without counting
an edge twice. Adding committed prefix cost yields a bound on the full CD.

`exact/distance.py` uses this bound only after cheaper bounds survive, below the
root and with at most 64 atoms remaining, limiting cubic profile work. It takes
the maximum with existing lower bounds. Certified searches keep their previous
independently replayable bounds. Search statistics include conditioned-bound
calls and prunes.

**Validation**

The full mapper functional suite passed: 135 passed, one optional skip, and one
known pre-existing provenance test excluded. The excluded
`test_frozen_alternative_its_case_is_bound_to_current_implementation` already
mismatched its historical source hash before the efficiency work; that artifact
was preserved. Selected lint and whitespace checks passed.

New tests compare every batched swap delta against the full objective over all
permutations of small directed/undirected weighted matrices. They check typed
bijections, monotonic improvement, pool/step limits, invalid seeds, no compatible
swaps, and high-index atom images. Eight additional exhaustive
small-graph cases verify the conditioned lower bound against the best exact
completion at every typed prefix, including signed and decimal weights.

Paired output checks compare minima, reference-class presence, known labeled
solution counts, normalized atom/bond reaction-center frequencies when quotient
analysis is complete, and ITS/template labeled class counts when both structural
analyses are complete. Order-dependent representative stream hashes are not
mathematical invariants and are excluded from equality checks.

New minimum certificates for lines 40575 and 40346 were generated with the new
profile bounds disabled in certified search, then independently replayed.
Both prove minimum CD 1 and match benchmark labeled counts of 12 and 32,
respectively. The combined optimized-search/certification/replay checks took
2.1 and 9.4 seconds. No reference mapping was supplied as their seed.

**Artifacts and controls**

Artifacts are under `benchmark_results/synister_efficiency_timeouts_v3_20260907/`:

- `development_selection.json`, `heldout_selection.json`: unchanged selections.
- `seed_source/`, `seed_development/`: isolated seed-only experiment.
- `conditioned_source/`, `conditioned_development/`, `conditioned_heldout/`:
  final package snapshot and complete benchmark records/manifests.
- `compare_runs.py`, `comparison.json`: repeatable mathematical comparisons
  and performance metrics against round two.
- `seed_probe.py`, `seed_probe_development.json`: exploratory seed diagnostics.
- `certificate_checks/`: replay results and compressed minimum certificates.
- `remaining_selection.json`, `remaining_diagnostics.json`: the 19 outstanding
  tasks and their latest stage/incumbent/bound diagnostics.

Paired runs retain eight spawn workers on the same CPU IDs, one numerical
thread per worker, and a 4 GiB address-space limit per worker. Full task wall
includes preparation and finalization, and may exceed the cooperative search
timer. Worker RSS is a lifetime high-water mark. Small repeated cohorts do not
support a campaign-wide speedup or recovery estimate. Changes remain local and
uncommitted.
