# Synister campaign session log — 2026-09-07

All times below use Asia/Ho_Chi_Minh (UTC+07:00).

## Outcome

Both requested campaigns finished. The original 60-second campaign processed
10,000 records; the 120-second follow-up processed 3,013 cases selected because
at least one original shell had status `timeout`. Both summaries report zero
remaining cases and zero error records. A processed case can still contain
timed-out shells; campaign completion does not imply exhaustive enumeration.

| Campaign | Mode | Complete | Timeout |
| --- | --- | ---: | ---: |
| Original, 60 seconds | reference_cd | 8,119 | 1,881 |
| Original, 60 seconds | minimal | 7,001 | 2,999 |
| Follow-up, 120 seconds | reference_cd | 1,337 | 1,676 |
| Follow-up, 120 seconds | minimal | 250 | 2,763 |

The follow-up reran both modes for each selected case, including a mode that
may already have completed at 60 seconds. Its complete counts are therefore
not counts of newly recovered results. Compare records by source line and
mode before calculating improvements across the two campaigns.

## Host and provenance

- Workspace: `/home/labhhc4/Documents/Workspace/Long/SynKit`.
- Branch used: `synister-16cpu-60s`, fetched at `5c940e8`.
- Python: `/home/labhhc4/anaconda3/envs/synkit/bin/python`.
- CPU: AMD Ryzen Threadripper 7970X, 32 physical cores / 64 logical CPUs.
- Companion repository: `../Synister`, branch `dev`, commit `900d40e`.
- Dataset: `../Synister/data/flower_test_10000_v252.csv.gz`.
- Dataset SHA-256: `e40647847169a7fc98af1aab44fa81c7a64deb70b77085ae3deb3925e39642b0`.

## Original campaign

- Output: `benchmark_results/synister_global_shells_v4_60s_w16`.
- Service: `synister-global-shells-v4-60s-w16.service`.
- Manifest: `41dc4ad2e06dd877ad1921f64d69ef5b85790e499c6d5ab83588c6953ea764fa`.
- Start: 2026-08-29 00:59:33, from the observed service timestamp.
- Finish: approximately 2026-08-29 07:10:48, from final log modification time.
- Wall duration: approximately 6 hours 11 minutes 15 seconds.
- 16 spawn workers; CPU quota 1600%; numerical library threads limited to one.
- Timeout: 60 seconds per shell; mode `both`.
- Per-worker address-space limit: 4 GiB.
- Aggregate memory policy was raised live from 8 GiB soft / 10 GiB hard to
  no soft limit / 64 GiB hard at the user's request. Swap remained disabled.

The scan of all 20,000 shell results found a maximum recorded shell elapsed
time of 86.334 seconds: reaction `11285:1`, source line 15041, mode `minimal`,
status `timeout`. The same case had the largest sum of shell elapsed times,
153.552 seconds (reference_cd: 67.218 seconds). The average sum per case was
35.479 seconds. These sums exclude any work outside the recorded shell timers.
Observed elapsed times exceed the configured search budget in some cases;
the timeout should not be described as a strict whole-case wall-clock cap.

## Follow-up campaign

- Output: `benchmark_results/synister_global_shells_v4_120s_timeouts_w16`.
- Service: `synister-global-shells-v4-120s-timeouts-w16.service`.
- Manifest: `ad525979d2c72d079b1289a741ce521738bbbaafdb40ff906d9f9489564c86b8`.
- Start: 2026-09-01 13:26:43, from the observed service timestamp.
- Finish: approximately 2026-09-01 23:44:28, from final log modification time.
- Wall duration: approximately 10 hours 17 minutes 45 seconds.
- Settings: 16 workers, mode `both`, 120 seconds per shell, 4 GiB address-space
  limit per worker, no aggregate soft limit, 64 GiB aggregate hard limit,
  no swap, stop-on-OOM policy.

The helper `scripts/queue_synister_timeout_followup.py` checked the completed
baseline summary and dataset checksum, validated atomic record hashes, selected
3,013 cases, and launched the follow-up using the synkit interpreter.
The baseline had already finished when the queue was launched on September 1.

Selection evidence is stored alongside the follow-up:

- `timeout_selection.json`: source manifest, selected source lines and checksum.
- `timeout_cases.csv.gz`: frozen selected dataset.
- `queue.log`: selection and launch events.
- `campaign.log`, `manifest.json`, `summary.json`, `cases/`: execution evidence.

## Local changes and validation

- Fixed two stale `completed` references in
  `scripts/run_synister_global_shells.py` to use `completed_this_run`.
  The first launch encountered a NameError after writing one record.
- Preserved that failed attempt in
  `benchmark_results/synister_global_shells_v4_60s_w16_startup_bug_20260829-005919`.
  The corrected campaign used a fresh manifest and output directory.
- Updated `scripts/start_synister_global_shells_16cpu.sh` and
  `paper/synister/RUN_LOG_16CPU_60S.md` to use the requested 64 GiB host policy.
- Added the independent follow-up helper without changing the running
  campaign's hashed implementation.
- Syntax compilation succeeded for the runner and helper. Launcher shell
  syntax and tracked diff whitespace checks passed during the session.
- Focused pytest execution could not run because pytest was absent from the
  synkit environment. No local pytest pass is claimed. The earlier handoff
  reported 94 tests passed on the source host.
- Both real campaigns subsequently produced final completion events and
  summaries with zero error records.

Changes are local and uncommitted. The result directories and helper were
untracked at log creation. Do not change hashed runner implementation or
manifest options when resuming an existing output directory. Any additional
timeout increase requires a separate campaign and has not been launched.


**Continuation completed — 8 September 2026**

The fourth efficiency round completed all 64 frozen timeout tasks (56 minimal,
eight reference-CD) at the unchanged 60-second search limit. Final verification
used `synister_efficiency_timeouts_v4_20260907/verified_development` and
`verified_heldout`: 32/32 each, zero errors, unchanged package hashes, complete
structure and symmetry quotient results. All 45 previous completions matched
the checked mathematical outputs; all 19 remaining tasks were recovered.
Slowest full task wall: 28.79 seconds. Tests: 176 passed, one optional skip,
one pre-existing provenance test excluded; two new minimum certificates replayed.
The 10,000-case campaign was not rerun. Full details and controls are in the
[fourth-round report](EFFICIENCY_RESULTS_V4_2026-09-08.md).


**Expansion and candidate enhancement — 8 September 2026**

The 1,200-task expansion completed 1,134 exact searches, with 66 timeouts and
zero errors. The candidate component/twin symmetry improvement recovered 46
of those 66 at the unchanged 60-second search limit; 20 remain. All 96
regression searches completed and matched checked mathematical outputs, but
structure-classification completeness remains an unresolved limitation.
See [the fifth-round report](EFFICIENCY_RESULTS_V5_2026-09-08.md) for the
classification caveat, controls, tests and frozen artifacts.


**Remaining timeout improvements — 8 September 2026**

V6 recovered 17/20 exact-search tasks at the retained 60-second search limit
and 100,000-mapping cap. Final verification: 159/162 complete, three timeouts,
zero errors; all 142 prior completed regressions pass the mathematical audit
without structure-completeness regressions. Nine of the 17 recovered tasks
also have complete structure classification; eight remain structurally
incomplete. The three remaining reference-CD tasks (28163, 30474, 22361)
reach the mapping cap in longer diagnostics with classification disabled,
which are not validation completions. Tests: 238 passed, one optional skip,
one known provenance test deselected. No new full 1,200-task rerun.
See [the sixth-round report](EFFICIENCY_RESULTS_V6_2026-09-08.md).


**Canonicalization and structure-cache improvements — 8 September 2026**

V7 preserves all 159 completed searches in the 162-task verification cohort. Complete structure classification increases from 124 to 159, including 17/17 of the V6 recovered searches (previously 9/17). The three large reference-CD shells remain timeouts at unchanged limits. All regression invariants pass; 242 tests passed, one optional skip and one known provenance test deselected. No new full 1,200-task rerun. See [the seventh-round report](EFFICIENCY_RESULTS_V7_2026-09-08.md).


**Optional native two-sided aggregation — 8 September 2026**

V8 completes case 28163, including exact ITS/template classification, in 25.20 seconds for the timed phase and 29.31 seconds analysis wall time. Cases 30474 and 22361 still time out. This uses a new explicit eight-worker protocol and a shared cap on retained double-orbit records; it is not a completion under the original product-representative cap. All 46 prior completed reference-CD cases match exact regression invariants. Tests: 266 passed, one optional skip, one known provenance test deselected. Existing V7 modules are unchanged. No full 1,200-task rerun or tally update. See [the eighth-round report](EFFICIENCY_RESULTS_V8_2026-09-08.md).


**Resumable native search and complete large-shell outputs — 8 September 2026**

V9 now has complete exact results for all three remaining reference-CD cases, with full ITS/template classification and held-out reference classes observed. The complete records use 16 workers, a one-million retained double-orbit record cap, and expanded time allowances: analysis wall times are 14.54 s (28163), 314.52 s (30474), and 121.53 s (22361). The final-build 60-second/eight-worker comparison still completes only 28163 (15.27 s wall time). The initial 300-second 30474 verification timed out under concurrent CPU load and is preserved; its complete result is in a separate 600-second-allowance retry. All three complete outputs agree with independently scheduled completed runs. All 46 prior reference-CD regression cases match exactly; 276 tests passed, one optional skip and one known provenance exclusion. Default V7 Python search modules remain unchanged. No full 1,200-task rerun or original-protocol tally update. See [the ninth-round report](EFFICIENCY_RESULTS_V9_2026-09-08.md).


**V10: both remaining cases below 60 seconds — 8 September 2026**

Implemented and verified the optional native efficiency changes. Three fresh-process, strict 60-second runs per case all completed with full JSON output: 30474 took 58.542, 57.108, and 59.043 seconds; 22361 took 43.716, 37.884, and 40.640 seconds. Every full class map, multiplicity, frequency, entropy, group order, and reference-class check matches frozen complete V9 output. The protocol uses 16 workers on physical CPUs 16–31, 4 GiB address space per worker, and the same one-million retained-worker-double-orbit-record cap. A recorded host-specific, profile-guided native build and compact JSON output are used.

All 46 prior reference-CD regressions match exactly and finish below 60 seconds; control 28163 matches exactly in 5.881 seconds. The broader mapper/canonicalizer suite has 290 passes, one optional skip, and the same frozen-evidence provenance exclusion documented previously. The excluded fingerprint is identical in current code and frozen V9, confirming that the mismatch predates V10. Default Python source is unchanged across 425 modules. No full 1,200-task or 10,000-task rerun and no original-protocol tally rewrite. The maximum observed time exceeds the 55-second engineering aim, while all six runs pass the strict 60-second requirement.

See [the V10 implementation and verification report](EFFICIENCY_RESULTS_V10_2026-09-08.md). Frozen source, native library, compiler profiles, all development runs, strict acceptance audits, and regression evidence are preserved in benchmark_results/synister_efficiency_timeouts_v10_20260908/.


**V20: all 1,200 original timeout tasks pass the 60-second gate — 9 September 2026**

Implemented exact repaired assignment warm starts for larger residuals, retained
fast fresh solves for small residuals, shortened singleton-domain checks, and
preserved canonical certificates through sparse construction and exact local twin
group factors. Native classification reuses bounded encoding tables and private
scratch buffers. Worker slices read the fixed deadline once; mapping-cap updates
remain locked.

A fresh frozen-source run completes all 1,200 searches and structure classifications
below 60 seconds, including JSON output: 751 minimal and 449 reference-CD tasks.
All 1,200 exact observable comparisons pass with zero mismatches, missing records,
incomplete cases or overruns. Maximum full-output time is 53.213 seconds on line
30474. All nine fresh hard-sentinel executions pass; line 30474 takes
54.025, 54.493 and 54.072 seconds. The full cohort took 33.76 minutes on two disjoint
16-CPU partitions. These are observed shared-host results, not universal guarantees.

Validation: 350 regression tests, 105 native undefined-behavior-sanitizer tests,
200 independent assignment problems and 6,400 forced-edge checks pass. One optional
dependency skip and the verified pre-existing historical-provenance exclusion
remain documented. The slower refinement prototype and incomplete development
pilots are preserved separately; none substitutes for a cohort result.

See [the V20 report and mathematical review](ENUMERATION_ENHANCEMENT_V20_2026-09-09.md).


**V21 targeted follow-up — 10 September 2026**

Full 10k run and recovery processed all 20,000 shells: 19,998 complete, one
minimum-proof error (21609) and one mapping-limit timeout (40548). Added bounded
native lower-shell minimum proof with exact cost-lattice skipping. Line 21609
now completes in 9.036–10.275 s across three fresh 60s runs. All 32 selected
previous minimum results match exactly; 124 tests pass. Line 40548 remains
unresolved: a diagnostic observes over 6 million distinct classes and a further
expanded-budget attempt hits MemoryError. Do not claim both tasks are fixed or
that a new full 10k run passed. Report: ENUMERATION_ENHANCEMENT_V21_2026-09-10.md.
Detailed artifacts: /tmp/synister_v21_20260910/. Production and frozen candidate
source match. No commit/reset/cleanup performed.


**V22: line 40548 fully resolved with disk-backed storage — 10 September 2026**

Complete reference-CD distance-17 result: 8,000,310 ITS classes, 7,997,022 template
classes, 31,791,010 weighted representatives and 1,017,312,320 labeled mappings.
Full output took 1,527.187 s. Both reference classes are observed. Coordinator
peak RSS 749.65 MiB; maximum child peak 343.35 MiB. Revised explicit protocol:
16 workers, unchanged 4 GiB worker address-space limit, 14,400-second allowance,
no cumulative record cap, full-certificate equality in SQLite, streamed class
tables. This is not an original 60s/1M-cap pass or a fresh full 10k rerun.
132 tests pass; six exact cross-backend controls and the complete prior partial
class subsets match. Independent candidate count and exact witness table both
contain 13,187,584 mappings. Export hashes/totals and SQLite quick_check pass.
Report: ENUMERATION_ENHANCEMENT_V22_2026-09-10.md. Authoritative large artifacts:
/tmp/synister_v22_20260910/revised/full/. Optional backend and CLI added; capped
backend unchanged. Frozen run source remains preserved; subsequent production
cleanup changes imports only. No commit/reset/cleanup performed.
