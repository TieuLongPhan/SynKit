# COMPLETE — V20 verified on all 1,200 original tasks

All cohort, repeat, and audit processes exited zero. No solver remains running
for this task. Current package equals the frozen source.
- 1,200/1,200 complete search and structure results below 60 seconds.
- 1,200 exact comparisons, zero mismatches/incomplete/over-budget/missing records.
- Maximum cohort full-output time: 53.212953874 seconds.
- All 9 fresh sentinel runs pass; maximum 54.493440379 seconds.
- 350 regressions, 105 UBSan tests, 200 independent LAP checks and 6,400 forced
  edges pass. Optional skip and pre-existing provenance exclusion documented.
- final_summary.json, audit_summary.json and MATHEMATICAL_REVIEW.md are final.
- Report: paper/synister/ENUMERATION_ENHANCEMENT_V20_2026-09-09.md.
- Session log updated; no commit/reset/cleanup was performed.

Historical working notes below; their active-session instructions are superseded.

# Active task: finish V20 enhancement and all 1,200 tests

User authorized mathematically/technically sound enhancements and testing ALL
original 1,200 timeout tasks. No subagents. No goal created. Continue until full
cohort, audits, repeats and final report finish. Preserve all prior dirty work.

Final source frozen in this directory/source and current package matches it.
Do not edit production source while benchmark runs.
Library in library.txt is immutable V16/selective_fast_build, portable C++17 -O3.
Source/build binding verified, no compiler warnings, no PGO/native architecture.

ACTIVE TOOL SESSIONS:
- 30336 run_cohort.py: two fresh 600-task partitions, CPUs 0-15 and 16-31,
  16 workers per case, original 60s limit, mapping cap 1M, pattern cache disabled,
  compact immutable-sequence JSON. No earlier results substituted.
- 79879 run_repeats.py: waits for one finished partition, then three independent
  invocations of all three hard sentinels (9 fresh case executions).
Files: cohort_progress.log, cohort_part_{0,1}.log, repeats_progress.log,
hard_repeat_{1,2,3}.log. Full run expected 35-45 minutes.
Send commentary at least every 60 seconds; tool waits <=60 seconds.

Tools: bash -lc wrapper prevents recurring sandbox-init failures.
IMPORTANT: shell-quote the ENTIRE bash -lc inner script programmatically:
  const sh=s=>"'"+s.replace(/'/g,"'\\''")+"'";
Do not hand-wrap heredocs in single quotes: embedded apostrophes previously
broke a state-file write and stripped dictionary-key quotes in audit_results.py.
The V20 audit script was fixed and F821/E9 checks now pass; older diagnostic
audit script copies still have that unused bug. Benchmark/coordinator scripts
were checked and are fine.

Python/pytest/ruff: /home/labhhc4/anaconda3/envs/synfrag/bin/

CHANGES (3 production files plus test additions):
- native_distance.cpp: exact repaired parent assignment dual hints and only tight
  finite distinct matching hints; same Hungarian optimum and forced-edge bounds.
  Reuse only for residual dimension >=16; small matrices retain original cold
  row reduction. Parent metadata uses original coordinates.
- Cheap singleton guard stops after its second allowed image; later full exact
  domains unchanged.
- Sparse canonical leaf certificate has identical dense ordering; local exact
  equal-row twin factors use m!, mixed support components retain general solver.
- native_its.py: instance-local scratch/order/ctypes pointers, bounded exact
  template-palette encoding tables keyed by full token tuple, immutable output.
- native_frontier.py: read fixed deadline once per slice instead of locking at
  every candidate; mapping counter remains protected by its shared lock.
- Added arbitrary stale-hint/exhaustive controls, mixed twin-group controls and
  encoding cache/eviction/immutable-output checks.
- MATHEMATICAL_REVIEW.md contains proof arguments and evidence boundaries.

FINAL V20 VALIDATION DONE:
- tests.log: 350 passed, 1 optional dependency skip, 1 historical hash test
  deselected. preexisting_provenance_failure.json proves baseline V15 and current
  binding are equal and both differ from preserved historical record.
- ubsan_tests.log: 105 tests pass, -fsanitize=undefined -fno-sanitize-recover=all.
- assignment_stress.json/log: 200 independent SciPy LAP checks, 6,400 forced-edge
  checks, large-cost optima >50M, zero failures.
- Ruff selected correctness checks, git diff --check pass.
- changed_file_hashes.json records source/test hashes at freeze.

EXPERIMENT EVIDENCE:
V16/V17/V18 are diagnostics only; their 30474 60s pilots are incomplete.
A slower flat-refinement prototype was removed. Unconditional warm bookkeeping
was replaced by the size selector, retaining original fast cold initialization.
Full-candidate paired V19 90s diagnostic: V15 baseline 69.765s, V19 candidate
67.528s, both exact and class/frequency parity passes (paired_summary.json).
This is not V20 acceptance and not a general speed claim. Earlier historical
V15 runs were 56-59s, so current host conditions affect deadline compliance.

REMAINING:
1. Wait for all 1,200 new records and nine repeats. Coordinator gathers new files,
   validates unique selection/source coverage, then automatically invokes auditor.
2. When repeats finish, rerun audit_results.py to include them in audit_summary.
   It checks labeled counts, rational coordinate frequencies, complete full class
   maps in labeled units, minimum/reference fields, manifests, and JSON-close times.
3. If a 60s case is incomplete, run a separate final-V20 90s diagnostic after CPUs
   are free to check exact full output; never replace a 60s cohort result.
4. Verify current source still equals frozen source; check all bindings and counts.
5. Write final_summary.json and paper/synister/ENUMERATION_ENHANCEMENT_V20_2026-09-09.md
   with measured cohort/repeat outcomes, strict-under-60 counts, remaining failures,
   proof/test scope and limitations; append concise session log entry.
6. Give concise final response with ACTUAL results and report link. Do not claim
   all 1,200 strict passes if even one times out or overruns.

No resets, cleanup, commits, external campaigns, or changes to historical artifacts.

Additional session 83209 runs audit_progress.py, comparing each newly closed result once; see partial_audit.json. V20 audit script skips active repeats; final report generator write_final_report.py is ready and requires all three repeats.
