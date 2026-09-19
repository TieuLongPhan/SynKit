**Synister timeout efficiency, fourth round — 8 September 2026**

This round continues the 19 remaining timeout tasks from the
[third-round report](EFFICIENCY_RESULTS_V3_2026-09-07.md). Evaluation stays within
the frozen 64-task cohort: 56 minimal and eight reference-CD tasks, all drawn
from historical 120-second timeouts. The full 10,000-case campaign was not rerun.
The configured exact search limit remains 60 seconds.

**Final measured results**

| Cohort | Previous complete | Final complete | Previous batch wall | Final batch wall |
| --- | ---: | ---: | ---: | ---: |
| Development, 32 tasks | 22 | 32 | 124.5 s | 30.7 s |
| Reused validation, 32 tasks | 23 | 32 | 125.1 s | 15.0 s |
| Total, 64 tasks | 45 | 64 | — | — |

All 19 remaining tasks completed: 15 minimal and four reference-CD. All 64
records have complete search, structure classification, and symmetry quotient
analysis, with known labeled counts. There were zero errors or completion
regressions. Checked mathematical outputs match all 45 previous completions;
their median paired wall speedup is 1.55 times. Some already short cases slowed,
so this is not a uniform per-case speedup.

The slowest final task was line 29763 at 28.79 seconds, including preparation
and finalization. Line 15041, with 171 atoms, completed in 3.91 seconds. Summed
observed task wall fell from 752.3 to 76.9 seconds in development and from
723.6 to 45.6 seconds in validation. These totals include censored baseline
timeouts and are not uncensored solution-time estimates. Peak worker lifetime
RSS was 281.2 MiB and 251.3 MiB, respectively (previously 281.0 and 255.3 MiB).

The final runs are `verified_development/` and `verified_heldout/`, both using
`dynamic_source/`. Current package contents match both frozen-run manifests:
`687605c9da025b8eb7c7e4efbb4156c5e4a39bc68bfc608cb7e04b4fab5caf3c`.
`completion_summary.json` records the final audit, and
`remaining_after_verification.json` contains no tasks. This finishes the
selected 64-task cohort; it does not establish completion of every historical
timeout in the larger dataset.

**Implementation**

Seed construction now uses a bounded, element-compatible quadratic-overlap
relaxation, with deterministic perturbations to escape pair-swap local minima.
Sparse matrix products and element-blocked assignments move groups of atoms
together. A graph-only profile bound stops optional refinement when attained.
The optional common-fragment cover checks alternative fragment orientations
before filling and polishing a complete typed mapping. Its anchors constrain
only that seed heuristic; global exact search still receives no fixed mapping.
Every accepted seed passes a fresh complete objective calculation, and failure
retains the latest verified feasible seed. Reference-CD searches also benefit
from this reference-free repair.

Work is bounded: four relaxed starts of at most 40 iterations, at most two
follow-up refinements and 24 perturbations, and at most 16 common fragments
under a shared cooperative 0.5-second budget. Fragment orientations use at most
8 reactant and 32 product embeddings per fragment. Relaxation and fragment
heuristics skip unsupported or oversized inputs. These are preprocessing
budgets; total task wall includes their cost and finalization.

The profile assignment bound now supplies a lower bound for every forced atom
pair using optimal-assignment alternating paths. This removes pairs that cannot
reach the shell or incumbent cost. The stronger filtering uses exact quarter-unit
arithmetic; arbitrary floats retain the prior individual bound.

When the global row-profile lower bound is attained, every row must attain its
individual bound. A zero-profile row must preserve its incident outgoing bonds
exactly. Reversible domain propagation enforces this condition, with both
directions checked where applicable. It is enabled only for half-integer input
weights, bounded exact arithmetic, and tolerance smaller than the cost quantum.
Removed-domain entries are retained as disjoint flat-index lists along the live
DFS path, keeping this extra restoration storage quadratic rather than cubic.

On that constrained search, dynamic branching chooses the row with the fewest
remaining compatible images, breaking ties by assigned neighbors and the
original priority. Reactant bond-mass bounds are updated reversibly alongside
product mass, preserving admissibility under the changing row order. The rule
is invariant under the verified product group. Certified searches retain their
existing bounds and ordering; the new propagation is not an unrecorded shortcut
in certificate replay. Redundant profile relaxations and unchanged stabilizer
constructions are avoided.

Structure classification uses a bounded per-analysis cache. Color/degree
summaries only locate candidates: a hit requires exact colored-graph
isomorphism, checked component by component. Identical labeled components can
be compared directly. Lookup exhaustion falls back to exact canonicalization.
The cache holds at most 32 graph snapshots, with graph-size/edge limits and a
shared 10 ms lookup budget. Vectorized bond-change/context construction avoids
repeated Python scans over absent edges. Internal transport fingerprints use
binary float matrices; public representative stream digests retain their
existing construction.

**Validation**

The final mapper module suite passed **176 tests**, with one optional skip and
one known provenance test deselected, in 2.67 seconds. Selected Ruff correctness
checks (`E4,E7,E9,F`) and `git diff --check` passed. The test command was:

```sh
/home/labhhc4/anaconda3/envs/synfrag/bin/pytest -q Test/Chem/Mapper/module -k 'not test_frozen_alternative_its_case_is_bound_to_current_implementation'
```

Regression comparisons check complete minima, reference-class presence, known
labeled counts, normalized atom/bond reaction-center frequencies when quotient
analysis is complete, and ITS/template labeled class counts when both structural
analyses are complete. Stream order is not used as a mathematical invariant.

New tests verify relaxed and fragment seed feasibility, non-worsening full
costs, local-minimum escape, work limits, and heuristic anchor handling. Forced
assignment bounds are checked against exhaustive typed assignments. Tight-row
propagation is compared with exhaustive complete mapping sets for directed and
undirected graphs, fixed assignments, and symmetry expansion; float and large
tolerance cases verify the conservative fallback. Cache tests include connected
non-isomorphic regular graphs with identical histograms, color differences,
component reordering, lookup exhaustion, and mutation protection. Vectorized
reaction-center counts are checked against scalar tolerance calculations.

New minimum certificates for source lines 6233 and 39461 independently replayed
successfully, matching minima 2 and 1 and labeled counts 96 and 192. The final
hard case, 29763, was also enumerated separately without a structure observer:
2,592 distinct representatives and 84,934,656 labeled mappings. Every collected
mapping was checked as a typed bijection with full objective 10.5, matching an
independently recomputed profile assignment lower bound. This establishes its
minimum independently of the seed heuristic; completeness comes from the exact
enumerator and its tested pruning rules.

One known pre-existing test remains excluded:
`test_frozen_alternative_its_case_is_bound_to_current_implementation`.
Its historical provenance hash already differed from the pre-enhancement
implementation. The frozen historical artifact is preserved.

**Controls and artifacts**

All runs use eight spawn workers on the same CPU IDs, one numerical-library
thread per worker, a 4 GiB address-space limit per worker, and frozen package
snapshots. Full task wall includes preparation/finalization and can exceed the
cooperative search timer. Worker RSS is a lifetime high-water mark. The reused
validation cohort is not a new independent population sample.

Artifacts are under `benchmark_results/synister_efficiency_timeouts_v4_20260907/`;
the directory reflects the start date of work, which continued on 8 September.
It includes frozen selections and source snapshots, full per-task records,
`compare_runs.py`, `comparison.json`, `certificate_checks/`, and
`29763_objective_checks.json`. Exploratory runs are retained transparently:
static minimum-domain ordering and the matrix-update tail shortcut caused
slowdowns and are not present in the final implementation. A targeted 58.7-second
success did not survive cohort load, prompting the dynamic-order improvement.
Final claims use the final full-cohort verification, not a union of successes
from different implementations. Changes remain local and uncommitted.
