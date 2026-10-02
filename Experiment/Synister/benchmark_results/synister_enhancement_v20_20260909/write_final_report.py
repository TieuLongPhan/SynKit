"""Finalize only after the full cohort and all three fresh sentinel runs."""
import hashlib
import json
from pathlib import Path
import runpy

R = Path(__file__).resolve().parent
REPO = R.parent.parent
audit = json.loads((R / "audit_summary.json").read_text())
required = {"cohort_1200", "hard_repeat_1", "hard_repeat_2", "hard_repeat_3"}
assert required <= audit.keys(), ("unfinished runs", required - audit.keys())
cohort = audit["cohort_1200"]
assert cohort["processed"] == 1200 and cohort["missing_count"] == 0
assert all(not audit[name]["failures"] for name in required)
source_hash = runpy.run_path(str(R / "benchmark_synister_native.py"))["source_hash"]
current_hash, frozen_hash = source_hash(REPO), source_hash(R / "source")
assert current_hash == frozen_hash == cohort["source_sha256"]
library = Path((R / "library.txt").read_text().strip())
build = json.loads(library.with_suffix(".json").read_text())
assert build["source_sha256"] == hashlib.sha256((R / "source/synkit/Chem/Mapper/exact/native_distance.cpp").read_bytes()).hexdigest()
assert hashlib.sha256(library.read_bytes()).hexdigest() == cohort["library_sha256"]
manifest = json.loads((R / "cohort_1200/manifest.json").read_text())
timings = json.loads((R / "cohort_1200/case_timings.json").read_text())
by_mode = {}
for mode in ("minimal", "reference_cd"):
    selected = [t for t in timings if t["mode"] == mode]
    complete = structures = 0
    for timing in selected:
        name = "{}_{}.json".format(timing["source_line"], mode)
        result = json.loads((R / "cohort_1200" / name).read_text()).get("result", {})
        complete += bool(result.get("complete"))
        structures += bool(result.get("complete") and result.get("structure", {}).get("complete"))
    by_mode[mode] = dict(tasks=len(selected), search_complete=complete,
                        structure_complete=structures,
                        strict_complete=sum(t["complete_below_60_seconds"] for t in selected))
repeats = []
for repeat in range(1, 4):
    directory = R / f"hard_repeat_{repeat}"
    for timing in json.loads((directory / "case_timings.json").read_text()):
        name = "{}_{}.json".format(timing["source_line"], timing["mode"])
        doc = json.loads((directory / name).read_text())
        result = doc.get("result", {})
        repeats.append(dict(repeat=repeat, source_line=timing["source_line"], mode=timing["mode"],
                            seconds=timing["end_to_end_wall_seconds"],
                            search_complete=bool(result.get("complete")),
                            structure_complete=bool(result.get("structure", {}).get("complete")),
                            strict_complete=timing["complete_below_60_seconds"]))
assert len(repeats) == 9
summary = dict(cohort=cohort, by_mode=by_mode, repeats=repeats,
               source_sha256=frozen_hash, current_source_matches_frozen=True,
               library_sha256=cohort["library_sha256"],
               parallel_wall_seconds=manifest["parallel_phase_seconds"],
               validation=dict(regression_passed=350, optional_skipped=1,
                               preexisting_provenance_excluded=1, ubsan_passed=105,
                               assignment_matrices=200, independent_forced_edges=6400),
               completion_contract="Fresh original 1,200 tasks; no diagnostic or historical substitution")
(R / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

mode_rows = "\n".join("| {} | {} | {} | {} | {} |".format(mode, row["tasks"], row["search_complete"], row["structure_complete"], row["strict_complete"]) for mode, row in by_mode.items())
critical_rows = "\n".join("| {} | {} | {:.3f} |".format(t["source_line"], t["mode"], t["end_to_end_wall_seconds"]) for t in sorted(timings, key=lambda t:t["end_to_end_wall_seconds"], reverse=True)[:3])
repeat_rows = "\n".join("| {} | {} | {} | {:.3f} | {} |".format(row["source_line"], row["mode"], row["repeat"], row["seconds"], row["strict_complete"]) for row in repeats)
unresolved = json.dumps(cohort["incomplete"], ensure_ascii=True)
overruns = json.dumps(cohort["over_budget"], ensure_ascii=True)
report = f"""# Exact enumeration enhancement V20

Date: 2026-09-09. All 1,200 original tasks have been rerun from scratch.

## Results

Search completion: **{cohort["search_complete"]}/1,200**. Structure completion:
**{cohort["structure_complete"]}/1,200**. Both complete below 60 seconds:
**{cohort["strict_complete_below_60"]}/1,200**.

Completed exact-result comparisons: **{cohort["comparisons"]}**; observable
mismatches: **{len(cohort["failures"])}**. Comparisons include minimum/reference
fields, labeled counts, exact rational coordinate frequencies and full ITS/template
class maps in labeled units. The independent Python classification control for
line 13067 is also replayed against the new output.

| Mode | Tasks | Search complete | Structure complete | Complete below 60 s |
|---|---:|---:|---:|---:|
{mode_rows}

Incomplete outcomes: {unresolved}

Full-output overruns: {overruns}

Slowest complete cohort cases:

| Line | Mode | Full output seconds |
|---|---|---:|
{critical_rows}

The two-partition campaign took **{manifest["parallel_phase_seconds"]/60:.2f} minutes**.
Summed per-case full-output time was {cohort["total_full_output_seconds"]/60:.2f}
minutes; case intervals overlap across the two partitions.

## Retained changes

- Reuse parent assignment potentials and matching hints only for larger residual
  problems. Repair every current row/column dual constraint, retain only finite
  disjoint tight matching edges, and finish the exact Hungarian solve. Small
  residuals retain the original fresh row-minimum reduction.
- Stop the preliminary singleton-domain test after finding its second allowed
  image. The complete assignment and forced-edge domain checks are unchanged.
- Construct the identical canonical leaf certificate from sorted present edges,
  and use exact factorial group orders for verified local twin factors.
- Reuse instance-local native scratch buffers and ctypes pointers. A bounded
  template palette cache binds full exact token tuples and reuses encoding tables.
  Published certificate bytes remain immutable.
- Read the fixed deadline once per worker slice instead of locking at each
  candidate. Every callback still checks it; the global mapping counter retains
  its shared lock.

The [mathematical review](../../benchmark_results/synister_enhancement_v20_20260909/MATHEMATICAL_REVIEW.md)
gives the invariant and exactness argument for each change. No mapping domain,
chemical objective, class equivalence, multiplicity or public identifier changed.
The broader contiguous result-batch redesign remains deferred; measurements
directed this change toward assignment and classification overhead.

## Correctness and technical checks

- 350 regression tests passed; one optional dependency test skipped.
- 105 native tests passed under undefined-behavior sanitization with recovery
  disabled.
- 200 independently solved assignment problems and 6,400 independently solved
  forced edges passed, including large-cost objectives above 50 million.
- New controls exercise extreme stale hints, forbidden and duplicate matching
  hints, mixed twin/non-twin symmetry, component exchange, exact palette identity,
  cache eviction and immutable output after scratch reuse.
- Existing exhaustive mapping/orbit, forced-edge, resumable frontier, deadline,
  global cap, typed-color and hash-collision controls pass.
- Ruff correctness checks and git diff --check passed.

The historical provenance test remains excluded because its preserved artifact
is bound to an earlier implementation. The [baseline check](../../benchmark_results/synister_enhancement_v20_20260909/preexisting_provenance_failure.json)
shows that V15 and current hashes are identical for that test's source scope and
both differ from the historical artifact. The old artifact was not rewritten.

## Fresh sentinel repetitions

Each invocation starts fresh workers and uses the same frozen source, library,
60-second policy and exact comparison criteria. These repetitions never replace
a cohort record.

| Line | Mode | Repeat | Full output seconds | Complete below 60 s |
|---|---|---:|---:|---|
{repeat_rows}

## Timing scope and limits

The clock covers case construction, proof, worker startup, enumeration,
classification, formatting, JSON serialization and file close. Imports and
dataset loading precede the clock; file close does not imply fsync. There are
16 workers per case, at most two concurrent cases on disjoint CPU partitions,
a 1,000,000 retained-worker-record cap, and 4 GiB address-space limit per worker.
The parent has no aggregate process-group memory cap. The native library uses
portable C++17 -O3 without PGO or architecture-specific flags. Pattern caching
remains disabled; exact encoding memoization is not stored-result reuse.

The host is shared. A contemporary 90-second diagnostic on line 30474 measured
the prior V15 baseline at 69.765 seconds and the intermediate V19 candidate at
67.528 seconds with exact parity. Neither is a V20 acceptance result. Earlier
historical V15 runs at 56-59 seconds do not establish current deadline compliance.
Microbenchmark improvements are not universal whole-case speed guarantees.

V16-V18 incomplete pilots and rejected prototypes are preserved separately.
They were not promoted as full-cohort runs. The recorded 60-second cohort
outcomes, including failures or overruns, are authoritative.

## Provenance

Source SHA-256: {frozen_hash}

Library SHA-256: {cohort["library_sha256"]}

Current package matches the frozen source: True.

[Machine-readable summary](../../benchmark_results/synister_enhancement_v20_20260909/final_summary.json),
[exact audits](../../benchmark_results/synister_enhancement_v20_20260909/audit_summary.json),
[original cohort outputs](../../benchmark_results/synister_enhancement_v20_20260909/cohort_1200).
"""
(REPO / "paper/synister/ENUMERATION_ENHANCEMENT_V20_2026-09-09.md").write_text(report)
print(json.dumps(summary, indent=2))
