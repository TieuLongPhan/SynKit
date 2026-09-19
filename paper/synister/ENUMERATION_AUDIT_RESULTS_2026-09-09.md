# Enumeration audit and full timeout rerun — 9 September 2026

The full original 1,200-task Python campaign finished with **1197/1,200 searches complete**, **0 errors**, and **1195 complete structure classifications**. Every completed search passed the available saved-output comparisons. The three Python timeouts remain 28163, 30474 and 22361.

Separate complete 449-task native campaigns used one freshly compiled portable binary, with the pattern cache enabled and disabled. They completed **448/449** and **449/449** searches respectively. These are separate protocols: the native API supports reference-CD mode, while the original cohort also contains 751 minimum tasks.

| Frozen campaign | Tasks processed | Search complete | Search + structure complete | Complete output below 60 s | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python, original search limits | 1200 | 1197 | 1195 | Not measured through file close | 0 |
| Portable native, pattern cache on | 449 | 448 | 448 | 448 | 0 |
| Portable native, pattern cache off | 449 | 449 | 449 | 449 | 0 |
| Cache-off hard-case diagnostic, 240 s allowance | 2 | 2 | 2 | Not an acceptance run | 0 |

Combining the separately run mode partitions gives **1200/1,200 complete searches** and **1199/1,200 complete structure classifications**: 751 minimum tasks use Python results and 449 reference-CD tasks use cache-disabled native results. This is coverage across two declared protocols, not one homogeneous run. The remaining incomplete structure classification is 13067_minimal.

## Audit findings

The inspected native pruning, canonicalization and weighted aggregation preserve exact counting under their documented integer-input and complete-group preconditions. No saved-answer table, reaction-ID special case or held-out mapping input was found in the calculation path. This is an engineering audit with proof arguments and independent tests, not formal verification of every program execution.

A numeric defect was reproduced and fixed in the assignment diagnostic: valid 64-million-cost assignments and alternating paths could be mistaken for infinity at the old 50-million threshold. The two new boundary tests fail on the previous V10 binary and pass on the corrected binary. Accepted production shell bounds are at most one million quarter units; these counterexamples did not show incorrect production shell counts.

A second correctness gap affected mixed-type atom labels such as [True, 1] or [0.0, 0]. Ordinary compatibility allows both bijections on two isolated vertices, but typed side groups and ITS keys do not describe the same equivalence relation. The previous native aggregation could report one labeled mapping instead of two. Native preparation now rejects equal labels with different typed representations; three new tests reproduce the missing validation.

The input guard was added after the running benchmark snapshots were frozen. All 1,200 cohort tasks (1,077 reaction inputs) were checked against it: all labels are builtins.int and none is rejected. Removing only this helper and its call makes the current wrapper AST identical to the frozen wrapper; every other Python/C++ file matches. Counting on the cohort inputs is unchanged. The cohort timings exclude the new guard, and validated_source retains the final guarded package separately from the timed native_source snapshot.

Cache hits are justified by exact colored isomorphism or a full sparse ITS pattern obtained through verified reactant automorphisms. A new pattern is remembered only after its full class is retained. Hashes select candidates; full native certificate payloads determine internal class equality. Fresh worker caches start empty, and eviction only causes additional canonicalization.

The cache-disabled campaign recorded **0 pattern hits**. All **448 mutually complete pairs** have identical reported class maps, multiplicities, reaction-center frequencies and entropy; the whole structure object also agrees. Incomplete 60-second records were not compared as complete answers. The two larger-allowance diagnostics supply separate complete-output checks for the hard cases.

Disabling the pattern cache does not remove exact duplicate-suppression sets or pure encoding memoization. The present candidate generator covers double orbits but can revisit them, so exact full-key membership remains necessary for counting each class once. This is not an answer cache.

Reference-presence reporting still has a strict mathematical limitation: some flags use SHA-256 membership sets. Those flags assume collision resistance. Public class identifiers are also digests, and corpus comparisons check the published ID/count maps rather than saved raw canonical certificates. Internal native class aggregation retains full certificates and does not merge distinct payloads solely on digest equality. The plan includes stronger exact reference-presence witnesses.

## Timeouts and difficult cases

| Source line | Cache on, 60 s budget | Cache off, 60 s budget | Cache off, 240 s diagnostic | Final guarded package, cache off, 60 s |
| --- | --- | --- | --- | --- |
| 28163 | complete, 7.817 s | complete, 6.481 s | — | — |
| 30474 | time_limit, 63.498 s | complete, 57.783 s | complete, 56.625 s | complete, 59.612 s |
| 22361 | complete, 44.658 s | complete, 41.948 s | complete, 41.705 s | complete, 40.518 s |

The cache-enabled portable run did not reproduce V10's below-60-second result for 30474. Disabling the pattern cache produced fresh complete results below 60 seconds without PGO, including the final guarded-package check. The earlier host-specific PGO measurements remain separate evidence. Both compiler and cache settings must accompany a timing claim; 30474 still has little timing margin. The longer diagnostic is correctness evidence only.

The final guarded-package checks run each hard case in a fresh benchmark process, using the same portable library with the pattern cache disabled. Their clocks include the new validation. All timed attempts are retained; there is no guarantee of a 60-second result on every repetition or machine.

All incomplete/error keys are retained in audit_summary.json. Any loss of structure completeness relative to the selected saved reference is listed separately in its classification_regressions field; no partial classification is promoted to complete.

## Validation and reproducibility

- **309 tests passed**, one optional PuLP-dependent skip, and one pre-existing frozen-provenance test excluded. Its historical evidence was not rewritten.
- **79 native tests passed** under undefined-behavior sanitization with recovery disabled.
- Twelve additional raw-matrix/permutation cases check signed and half-integer weights, atom types, seeds, orbit coverage, multiplicities, frequencies and cache equivalence. Adversarial cache tests reject invalid generators, verify empty fresh state, and exercise eviction.
- Independent assignment stress: 200 matrices and 6400 forced-edge solves, zero failures. Correctness-focused Ruff checks and git diff --check passed.
- Saved-output comparisons: Python 1197, native cache on 448, native cache off 449, diagnostics 2, final guarded cases 2; zero checked-observable differences.

The Python campaign took **279.476 seconds** overall. It used eight task workers on CPUs 0–7, the historical 60-second cooperative search limit and 100,000 mapping cap. Its source hash stayed unchanged. Search completion and full wall-time success remain different measurements.

Each native case used sixteen spawned workers, one numerical-library thread per worker, 4 GiB address space per worker, and a shared one-million retained worker-class cap including cross-worker copies. Its deadline starts before case construction; strict timing ends after full public output, JSON serialization and file close. Imports and dataset loading precede this timer. No parent address-space or aggregate service-memory cap was imposed.

The cache-on native campaign started on CPUs 16–31 while Python used CPUs 0–7. The cache-off campaign started on CPUs 0–15 after Python finished. The native campaigns used disjoint physical CPU sets but shared host memory bandwidth; ancillary validation also overlapped part of the campaigns. Workers and pattern caches are new per case; the coordinator stays alive across a cohort and may reuse bounded encoding state. The orchestration parent was stopped to remove its queued serial ablation; the active cache-on calculation continued, and the cache-off campaign was launched independently. Final record counts and source-unchanged manifests establish completion.

- Portable binary SHA-256: f0e92aa9cee511cecd98b1e04b2b474bde270e8c940b4dcb660aaa1bc73ae2c1
- Native C++ source SHA-256: c597b2480a49d8fb8ab2a330eeca1147d82f51f2c201eb170b37ab08be1c5bb8
- Frozen native Python/C++ tree SHA-256: 1370be55d37567ffa4243a2795be0395696c12b517fba382d325dafdd98455cd
- Final guarded Python/C++ tree SHA-256: e1d2c82bd772965e541d05d3e37e32d7cbc05cb1bb5640cc3ff4298ea41ac5dc
- Build flags: -std=c++17 -O3 -Wall -Wextra -fPIC -shared; no PGO or host-native flags.

The [detailed mathematical audit and enhancement plan](ENUMERATION_AUDIT_ENHANCEMENT_PLAN_2026-09-09.md) gives derivations, implementation locations, limitations and acceptance gates. Priorities are exact presence reporting, a native minimum-proof phase with a shared deadline, incremental assignment work, lower classification/transfer cost, and a separately proved unique double-orbit construction.

Artifacts: [machine-readable audit](../../benchmark_results/synister_enumeration_audit_20260909/audit_summary.json), [validation](../../benchmark_results/synister_enumeration_audit_20260909/validation_summary.json), [cache differential](../../benchmark_results/synister_enumeration_audit_20260909/cache_differential.json), [frozen sources and runners](../../benchmark_results/synister_enumeration_audit_20260909/), and per-task JSON/timings/manifests in each campaign directory. All changes remain local and uncommitted.
