**Synister timeout expansion and symmetry enhancement — 8 September 2026**

The requested expansion finished 1,200 historical timeout shell tasks (751
minimal, 449 reference-CD, across 1,077 reactions). The unchanged fourth-round
solver completed 1,134 exact searches; 66 timed out, with zero errors. Batch wall
was 1,002.6 seconds. The selection excludes reactions from the prior 64-task
cohort and is frozen with source, dataset, and selection hashes.

The candidate symmetry enhancement subsequently completed **46 of those 66
remaining searches**: 25 minimal and 21 reference-CD. Twenty still time out
(nine minimal, eleven reference-CD); seven are in minimum proof and thirteen
in shell enumeration. Search limits remain 60 seconds. Follow-up batch wall
was 218.1 seconds; recovered-task median wall was 4.30 seconds, maximum 55.61.

This is an exact-search improvement with an unresolved structure-classification
limitation. All 46 recovered results have complete symmetry quotient analysis
and known labeled counts, but only ten have complete ITS/template structure
classification. Do not equate these 46 search completions with 46 fully
classified results.

The implementation in synkit/Chem/Mapper/graph/automorphism.py recovers
verified transpositions of atoms with identical incoming/outgoing matrix rows
when canonicalization fails. For disconnected graphs, component canonicalizers
share the existing time/node budgets; internal generators and swaps between
exactly matched components remain compact. Every proposed permutation must
preserve the complete objective matrix and configured atom properties before
it can prune search. Exhausting discovery budgets retains only verified
subgroups. No search space, mapping cap, or exact-search time limit was relaxed.

**Validation and limitations**

- 181 mapper tests passed, one optional skip, one known historical provenance
  test deselected. Ruff correctness checks and whitespace checks passed.
- New tests compare component group orders against independent NetworkX
  isomorphism enumeration, exercise bond weights and atom properties, check
  directed incoming edges, budget exits and cache mutation, and compare
  exhaustive mappings, fixed subspaces, symmetry expansion and certificate replay.
- All 96 regression searches completed in both candidate runs. The set contains
  the prior 64 completions, 16 slowest expansion completions and 16 deterministic
  sampled completions. The checked minima, reference-class flags, known labeled
  counts and normalized reaction-center frequencies match; labeled ITS/template
  class counts match where both structural analyses are complete.
- Structure completeness regressed relative to the saved records on eleven
  tasks in the concurrent test and nine in the subsequent test run without the
  other SynKit batch. Therefore this candidate is **not established as fully
  regression-free**. A fresh 13-task control with the old frozen solver also
  produced search/structure incompletions; the comparison is retained in
  classification_control_comparison.json. Timing variability contributes,
  but these controls do not establish that the candidate has no effect.
- Other independent workloads were active on the host. The timeout follow-up
  used eight workers on CPUs 0–7; the initial regression batch used eight on
  8–15 and overlapped part of it. The second regression used eight on 0–7
  without another SynKit batch. Every worker retained a 4 GiB address-space
  limit and one numerical-library thread. Cooperative timeouts are not strict
  whole-task wall limits.
- The new implementation was not rerun on all 1,200 tasks. Do not report the
  combined historical successes as a single campaign under the new solver.

**Artifacts**

Expansion: benchmark_results/synister_efficiency_1200_timeouts_20260908/.
Enhancement: benchmark_results/synister_efficiency_timeout_enhancement_20260908/.

The enhancement directory includes the four-task exploratory twin fallback
probe, frozen component_source/ and final_source/, the 66-task follow-up,
both 96-task regression runs, the 13-task old-solver control, audit scripts,
completion_summary.json, and remaining_after_enhancement.json with 20 tasks.
The measured component snapshot and final source differ only in formatting
and docstrings; the summarizer verifies equal executable ASTs throughout the
package. The second regression uses the final package hash exactly.

Changes remain local and uncommitted. The next priority is structure-classification
reliability and the saved 20 remaining search tasks.
