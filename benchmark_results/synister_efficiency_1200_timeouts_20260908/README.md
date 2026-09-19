Synister 1,200-timeout expansion — 8 September 2026

Launched as `synister-efficiency-1200-60s-20260908.service`.
Scope: 1,200 individual shell tasks across 1,077 reactions (751 minimal,
449 reference-CD). Selection excludes reaction source lines from the previous
64-task cohort. Of the selected historical timeout statuses, 1,197 have reason
time_limit and three have reason mapping_limit.

The solver and runner are frozen locally. The solver matches the verified
fourth-round hash. Controls: 60-second cooperative search limit, eight spawn
workers on CPU IDs 0–7, one numerical-library thread, 4 GiB address space per
worker, 40 GiB service memory cap, no service swap. Task wall includes
preparation and finalization, so it can exceed the search budget.

Validation before launch: 176 tests passed, one optional skip, one known
historical provenance test deselected; git diff --check passed. Every source
record digest and campaign identity was checked before selection.

Artifacts: selection.json documents selection and provenance; launch.json
records exact commands; source/ is the solver snapshot; campaign.log tracks
progress; run/ contains the manifest, per-task records, and summary.json once
finished. A processed timeout is not a solved task.

Status: systemctl --user status synister-efficiency-1200-60s-20260908.service
Progress: tail -f benchmark_results/synister_efficiency_1200_timeouts_20260908/campaign.log
