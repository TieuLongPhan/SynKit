# Enumeration enhancement V13 — bounded useful fragment-search slices

Date: 2026-09-09. The cohort attempt was stopped after 378 closed pilot records to integrate the proved lower-bound witness path in V14.

## Implemented plan

The [V11 report](ENUMERATION_ENHANCEMENT_V11_2026-09-09.md) gives the implementation
and proof obligations for exact reference witnesses, minimum-proof plus native
shell enumeration, certificate-only structure classification, and the zero-slack
alternating-cycle specialization. It also records the independent canonical
augmentation prototype and the larger undeployed proposals.

V11 and [V12](ENUMERATION_ENHANCEMENT_V12_2026-09-09.md) each exposed a
17156_minimal timeout in a complete cohort. Changing the optional fragment budget
from wall time to CPU time was insufficient. Both cohorts and their failures
are retained separately.

## Reproduced cause and change

A seed-only replay of the preceding 288 tasks reproduced cost 25 on 17156,
with both forward and reverse MCS calls canceled without an accepted fragment.
This happens before enumeration and full-class caching are involved.

An instrumented replay changed the per-fragment slice to start at the first
MCS progress callback. It recovered a feasible cost-9 seed in 6.52 seconds after
the same prefix. Its first recorded callback still had only two atoms in the
partial MCS, and subsequent callbacks grew the fragment. This localizes the
sensitivity to the placement of the tiny search slice relative to initialization.

Production now retains two distinct bounds:

1. The whole fragment cover has the same 0.5-second process-CPU deadline, including
   graph construction and MCS initialization.
2. A fragment's 25-millisecond search allowance starts at its first progress
   callback and is capped by the remaining whole-cover deadline.

The callback is kept in a local variable throughout FindMCS. Native MCS's existing
one-second timeout remains. Subsequent seed polishing retains its bounded iteration
counts. All of this work is still charged to the outer 60-second case wall deadline.
No reaction identifier, known mapping, stored answer or target cost is used to
choose the policy.

This is a heuristic-budget change, not a new pruning rule. Every proposed seed is
a full atom-compatible permutation and its complete objective is recomputed.
Fragment anchors constrain only seed construction. Exact optimization must still
prove the minimum before native shell enumeration begins. Moving a heuristic
slice cannot turn an incumbent into an optimality certificate.

## Validation and full-cohort protocol

- 328 tests passed, one optional PuLP skip and one pre-existing historical
  provenance test excluded. The historical evidence was not rewritten.
- Regression tests cover a first callback delayed beyond 25 ms and ensure that
  such initialization does not extend the overall cover budget.
- The native C++ and portable binary are unchanged from V11's tested specialization;
  all 98 native UBSan tests passed on that binary source.
- The original 1,200 tasks are rerun from scratch on this final frozen source:
  751 minimum tasks and 449 reference-CD tasks. No earlier result is reused.
- Two disjoint 600-task partitions use physical CPUs 0–15 and 16–31. Each case has
  16 fresh workers, 8,192-node slices, a 1,000,000 worker-local retained-orbit
  record cap and a 4 GiB per-worker address-space limit. The parent has no aggregate
  process-group cap.
- Each 60-second wall deadline begins before case construction; measured full output
  includes proof, startup, search, classification, formatting, JSON serialization
  and file close. Imports/dataset loading precede that clock. Close does not imply
  fsync. Cooperative cancellation can overshoot, and overshoots are not accepted.
- Pattern cache is disabled; internal membership uses exact payloads. Pure encoding
  memoization remains explicitly distinct from answer reuse.
- After a partition finishes, its freed CPU set runs three fresh-process checks each
  of 17156_minimal, 30474_reference_cd and 22361_reference_cd. Another partition may
  still be running; there is no exclusive-host timing guarantee.
- Complete outputs are compared in exact labeled units, with rational coordinate
  frequencies and full class maps. The independent Python classification supplies
  the previously missing 13067 structure reference.

## Results

Interrupted pilot only. See the V14 report for the final frozen full-cohort run.
