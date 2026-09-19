**Synister timeout efficiency results — 7 September 2026**

Follow-up implementation and held-out evaluation are in the
[second-round report](EFFICIENCY_RESULTS_V2_2026-09-07.md). This report preserves
first-round measurements.

Implemented and evaluated an initial efficiency upgrade on 32 shell tasks drawn
exclusively from cases that still had status `timeout` in the 120-second
follow-up. The selection contains 28 minimal-mode tasks and four reference-CD
tasks. The original 10,000-case campaign was not rerun.

All three versions used a 60-second configured search limit, eight spawn
workers on the same eight CPU IDs, one numerical-library thread per worker,
and a 4 GiB per-worker address-space limit. Each version ran from its own frozen
source snapshot. The baseline already included the earlier timeout-unwinding
fix. The dataset and task selection were identical across versions.

| Version | Minimal complete / 28 | Reference-CD complete / 4 | Total complete / 32 | Batch wall time |
| --- | ---: | ---: | ---: | ---: |
| Saved baseline | 0 | 0 | 0 | 241.7 s |
| Symmetry and minimum-proof changes | 7 | 1 | 8 | 240.9 s |
| Above plus bounded seed repair | 15 | 0 | 15 | 181.5 s |

The final version closed 15 of the 28 persistent minimal-mode timeouts (53.6%).
Completed tasks took 0.47–48.64 seconds, with a median of 4.39 seconds. Summed
observed task wall time fell from 1,922.8 to 1,191.1 seconds. These totals include
censored runs and are resource-use measurements, not exact solution-time
estimates. No campaign-wide speedup or recovery rate is inferred from this
small development cohort.

All benchmark versions produced zero error records and retained their initial
source hashes. Maximum worker lifetime RSS in the final batch was about
201.5 MiB; this is a worker high-water mark rather than isolated per-case memory.
Preparation and finalization are included in task wall time. The search timer
is still cooperative, so full task wall time can exceed 60 seconds slightly.

**Changes**

- `exact/symmetry.py`: return the input generators when they already fix the
  selected point, preserving generator/work-budget behavior. Compute inverse
  orbit transports once instead of once per generator combination.
- `exact/distance.py`: reject symmetry-covered candidates before residual-mass
  and dense-matrix updates; build certificate prefixes only when needed.
- `exact/distance.py`: during the private minimum-proof pass, discard branches
  whose bound cannot improve an existing feasible mapping. This fast path is
  limited to integer/half-integer bond weights in a conservative exact
  floating-point range. Arbitrary weights retain tolerance-aware pruning, and
  the later shell pass retains all equal-cost minimizers.
- `analysis.py`: repair minimal-mode seeds with at most eight rounds of
  element-preserving swaps over at most 48 atoms. Recompute the complete cost
  before accepting a repair; retain the original seed if repair fails. The
  seed only supplies an upper bound and ordering. No reference mapping, reference
  CD, or heuristic fixed exterior restricts the minimal search.
- Preserve new stage/diagnostic statistics through hybrid backend selection:
  preprocessing/traversal time, proof work, incumbent cost, initial seed cost,
  and repair time/outcome.
- Add `scripts/benchmark_synister_timeouts.py`, which validates the selected
  timeout tasks and dataset checksum, imports an explicit source snapshot,
  records per-task results, and rejects reused output directories.

The seed-repair diagnostic improved 27 of 28 initial mappings in approximately
3.23 seconds total. Better seeds were then validated in the full exact searches;
seed improvement alone was not counted as solving a case.

**Correctness and limits**

The focused suite passed 44 tests. The broader mapper module suite passed 100,
with one existing optional test skipped and one archived-evidence binding test
excluded after diagnosing its pre-existing failure. The excluded test is
`test_frozen_alternative_its_case_is_bound_to_current_implementation`: its frozen
record's implementation hash already differs from the pre-enhancement snapshot.
The archived evidence was left intact; its hash was not rewritten to claim a
new execution. The initial unfiltered suite exposed the failure, and
`fixture_provenance_check.json` records the pre-existing mismatch.

New tests check exact tiny weighted/directed graphs against exhaustive
permutations, preservation of all streamed minima, sub-tolerance improvements
for arbitrary floating-point weights, fixed-point symmetry budgets, and seed
repair failure handling. The selected Python correctness lint rules and diff
whitespace checks passed.

Seven tasks completed with both optimization versions. They agreed on minima,
labeled counts, representative counts, reaction-center aggregates and exact
structure counts where both structure analyses finished. Eleven newly completed
minimal shells also matched labeled counts from historical completed numeric
shells at the same cost; this checks shell enumeration, not minimality by itself.

Three newly solved cases (source lines 35488, 39584 and 41004) were additionally
rerun through the single-pass certified enumeration path. Independent certificate
replay succeeded and the minimum values and counts matched the final benchmark.
Compressed certificates and their mapping witnesses are retained with the run.

Reference-CD performance remains unresolved. Source line 7190 completed in
55.1 seconds with the first optimization, then timed out with the final version.
Serial repeats on the same CPU reproduced completion at 56.3 seconds for the
first version and timeout near 60 seconds for the final version. Numeric seed
repair is disabled; the seed identity/cost and exact search code are checked
separately in the comparison artifact. A causal explanation has not been
established, and no claim of zero regression between those two versions is made.
The original saved baseline timed out on all 32 tasks.

The final batch leaves 12 minimal tasks in minimum proof, one minimal task in
shell enumeration, and four numeric-shell tasks incomplete. The next efficiency
work should target the remaining proof bounds and large-molecule branch costs,
while investigating the reference-CD timing difference. Subsequent performance
cohorts should remain restricted to timeout records, as requested.

**Reproduction and artifacts**

The local run directory is
[`benchmark_results/synister_efficiency_timeouts_20260907`](../../benchmark_results/synister_efficiency_timeouts_20260907).
It contains `selection.json`, the three source snapshots, `baseline/`,
`optimized/`, `repaired/`, `comparison.json`, seed diagnostics, serial numeric
repeats, and `certificate_crosschecks/`. Each run has a manifest and summary.

Run a fresh comparison using a new output directory:

```bash
/home/labhhc4/anaconda3/envs/synkit/bin/python scripts/benchmark_synister_timeouts.py \
  --selection benchmark_results/synister_efficiency_timeouts_20260907/selection.json \
  --source-root benchmark_results/synister_efficiency_timeouts_20260907/repaired_source \
  --output benchmark_results/synister_efficiency_timeouts_20260907/fresh_repeat \
  --seconds 60 --workers 8
```

Implementation changes are local and uncommitted. The benchmark jobs and
certificate checks described above have finished.
