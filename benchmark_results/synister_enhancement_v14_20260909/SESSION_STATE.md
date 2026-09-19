# Active task checkpoint

User asks execute enumeration enhancement plan and rerun original 1,200 timeout tasks.
Do not spawn subagents. No goal created. Preserve all pre-existing dirty code.

Current production/final frozen version: V14.
Active final cohort: benchmark_results/synister_enhancement_v14_20260909.
- source/ is frozen current package.
- run_cohort.py is running (tool session 16835), two 600-task partitions, CPUs 0-15 and 16-31.
- run_repeats.py is running/waiting (tool session 22385); after first partition completes,
  use its freed CPUs for three fresh-process checks EACH of 17156 minimal, 30474 refCD,
  22361 refCD. Nine runs total.
- Watch cohort_progress.log, hard_repeats.log, cohort_1200/manifest.json.
- audit_results.py compares live completed outputs; audit_summary.json updated every 40 cases.
- When cohort and all nine repeat manifests have source_unchanged true, rerun audit_results.py,
  then write_final_report.py. That writes final_summary.json and final report.
- Final report: paper/synister/ENUMERATION_ENHANCEMENT_V14_2026-09-09.md.
- Need verify final coverage exactly1200, strict complete+structure below60 counts, all comparisons,
  repeat acceptance, source hashes equal current package. Do not report partial/failed cases as passes.
- Keep commentary updates about once a minute, waits <=60s.

Implemented production changes this task:
P0 exact packed permutation witnesses and losslessly compressed transport witnesses
(replacing hash-only reference membership). Native frontend workers use exact packed witnesses;
stream/public IDs remain hashes but don't decide internal equality.
P1a explicit structure_native_library_path config and certificate-only native structure backend,
with same dense canonical codes, skip unused group order only.
P1b native API supports both target modes and shares absolute deadline; output target remains
legacy string minimal for minimal mode. Native numeric shell handles both after optimum proved.
V14 adds _attain_native_profile_bound: up to1s probe of exact shell at certified atom-profile LAP
lower bound L, no heuristic/reference mapping. A fully checked feasible witness cost L proves min.
Then enumerate full shell separately. No witness -> original heuristic + Python exact optimizer.
Timeout exception retains diagnostic probe/seed info; runner saves it.
P2 native_distance.cpp zero-slack SCC specialization: only zero reduced-cost edges with endpoints
in same zero SCC can participate in budget-equal-LAP-optimum continuation. Positive slack unchanged.
P3 packed witnesses and per-case aggregate CPU measurement; broader warm starts/packed batching deferred.
P4 separate explicit-group canonical augmentation prototype passed all-depth brute force equivalence.

Seed refinements:
V12 changed fragment budgets to process CPU time; that alone did not fix cohort timeout.
V13 starts25ms fragment slice at FIRST progress callback, capped by same0.5s CPU cover deadline.
Keeps callback locally. Prefix replay proved recovery of cost9 after old seed path returned25.
Optional anchors remain only heuristic; full cost always checked. V14 keeps this for fallback/refCD.

Validation:
330 passed,1 optionalPuLP skip,1 pre-existing historical provenance test excluded (not rewritten).
tests.log in V14. Earlier unchanged native CPP passed98 UBSan tests in V11.
Ruff correctness selected checks and gitdiff--check passed before last freeze.
New tests include exact lower-bound attainment without calling heuristic and nonisomorphic regular
graphs with unattained lower bound0 requiring optimizer fallback. Existing raw permutation oracles pass.

Earlier evidence:
V11: full original1200 complete1199 / structures1199 / strict1199, sole failure17156 minimal proof.
V12: likewise1199, same17156. Both full runs preserved and final reports written.
V13: interrupted pilot after378 CLOSED records to integrate stronger proved bound path; not full1200.
Mark pilot honestly. Final V14 starts ALL1200 from scratch; no reused result records.
V11/V12 early diagnostics and repeats show2hard refCD cases ~55-57s and37-39s, all3 fresh repeats passed.
17156 fresh diag with good seed7.34s, but after289 seed calls oldCPU-only path returnedcost25 (bound9).
Instrumented first-progress slice prefix replay producedcost9 in6.52s.
A completely heuristic-free native shell probe atL9 found a witness in0.338s /228nodes, promptingV14.
Do not change current source while final run executes unless a real failure demands correction.

Portable library reused in library.txt (same CPP all V11-V14):
.../synister_enhancement_v11_20260909/build/libsynkit_distance_4ff96f9468c323f24d07.so
No PGO, no marchnative. Pattern cache env0.
Per-case16workers,1M retained-worker-double-orbit cap,4GiB AS perworker,
60s beforeconstruction throughJSONclose. Parent no aggregate limit,imports/datasetload excluded,
close != fsync. Two partitions up to32physicalcores, nonexclusive host.

Execution:
Use bash -lc wrapper for tools.exec_command (direct shell commands often bwrap failure).
Python /home/labhhc4/anaconda3/envs/synfrag/bin/python; pytest/ruff sameenv.
No further approval required. Do not mutate .git, reset, commit, clean.
