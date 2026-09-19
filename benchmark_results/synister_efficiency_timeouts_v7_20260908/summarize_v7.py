"""Summarize frozen V7 verification and compare it with the V6 run."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
V6 = ROOT.parent / "synister_efficiency_timeouts_v6_20260908"

def read(path):
    return json.loads(path.read_text())

def records(directory):
    result = {}
    for path in directory.glob("*_*.json"):
        record = read(path)
        if "result" in record:
            result[path.stem] = record
    return result

def main():
    summary = read(ROOT / "verified_run/summary.json")
    manifest = read(ROOT / "verified_run/manifest.json")
    audit = read(ROOT / "verified_run/regression_audit.json")
    assert summary["tasks"] == 162 and summary["complete"] == 159
    assert audit == dict(checked=159, expected=159, failures=[], classification_regressions=[])
    spec = importlib.util.spec_from_file_location(
        "benchmark", WORKSPACE / "scripts/benchmark_synister_timeouts.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    assert benchmark.implementation_hash(WORKSPACE) == manifest["implementation_sha256"]
    assert benchmark.implementation_hash(ROOT / "verified_source") == manifest["implementation_sha256"]
    current = records(ROOT / "verified_run")
    previous = records(V6 / "final_verification")
    target = read(ROOT / "selection.json")
    keys = [f'{task["source_line"]}_{task["mode"]}' for task in target["tasks"]]
    gains = [
        key for key, value in current.items()
        if value["result"]["complete"] and value["result"]["structure"]["complete"]
        and not previous[key]["result"]["structure"]["complete"]
    ]
    old_classified = sum(d["result"]["complete"] and d["result"]["structure"]["complete"] for d in previous.values())
    new_classified = sum(d["result"]["complete"] and d["result"]["structure"]["complete"] for d in current.values())
    target_classified = sum(current[key]["result"]["complete"] and current[key]["result"]["structure"]["complete"] for key in keys)
    rows = []
    for key in keys:
        d = current[key]
        x = d["result"]
        rows.append(dict(source_line=d["source_line"], mode=d["mode"], status=x["status"],
                         wall_seconds=d["wall_seconds"], structure_complete=x["structure"]["complete"]))
    report = dict(
        implementation_sha256=manifest["implementation_sha256"],
        current_package_matches_frozen_source=True,
        verification=summary, regression_audit=audit, target_results=rows,
        target_search_complete=sum(row["status"] == "complete" for row in rows),
        target_search_and_structure_complete=target_classified,
        cohort_structure_complete_before=old_classified,
        cohort_structure_complete_after=new_classified,
        classification_gains=gains,
        tests=dict(passed=242, optional_skipped=1, known_provenance_deselected=1),
        full_1200_rerun=False,
    )
    (ROOT / "completion_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    target["tasks"] = [task for task in target["tasks"]
                       if not current[f'{task["source_line"]}_{task["mode"]}']["result"]["complete"]]
    target["selection"] = "Three reference-CD tasks remain after V7 verification at unchanged limits."
    (ROOT / "remaining_after_verification.json").write_text(json.dumps(target, indent=2) + "\n")
    table = "\n".join(
        f'| {row["source_line"]} | {row["mode"]} | {row["status"]} | {row["wall_seconds"]:.2f} | {"yes" if row["structure_complete"] else "no"} |'
        for row in rows
    )
    text = f"""# Synister efficiency, round seven — 8 September 2026

Improved structure classification for the V6 recovered searches: **{target_classified}/17
now have complete ITS/template classification**, compared with 9/17 in V6.
The same 17 exact searches complete; the three reference-CD searches at source
lines 28163, 30474 and 22361 remain incomplete. No additional one of these
three searches was solved in this round.

Across the full 162-task verification cohort, complete search plus structure
results increased from **{old_classified} to {new_classified}**. All 159 previously completed
searches remain complete and pass the mathematical regression audit, with no
loss of prior structure completeness in this run. Final run: 159 complete,
three timeouts, zero errors in {summary["wall_seconds"]:.2f} seconds.

| Source line | Mode | Exact search | Full task wall (s) | Structure complete |
| --- | --- | --- | ---: | --- |
{table}

## Changes

The native exact canonicalizer now seeds swaps between identically colored
nonadjacent twin vertices. Candidates require identical full incoming and
outgoing neighborhoods, and each permutation is verified against the full
colored graph before it can prune a branch. Discovery uses the existing
search time budget and retains at most 256 initial generators; incomplete
searches still expose no canonical identity.

For larger stabilizer workloads, partition checks inspect cached moved
vertex pairs of immutable verified generators. Fixed vertices need no
partition comparison. Sibling branches reuse a stabilizer until a new
generator is registered. Both changes preserve the existing canonical
certificate and class identifier conventions.

The structure cache materializes disconnected components once for repeated
adjacency traversals instead of repeatedly traversing filtered graph views.
Cache hits still require exact color and edge preservation.

## Verification and controls

- 242 tests passed; one optional test skipped and the known historical frozen
  provenance test deselected. Ruff correctness checks and whitespace checks
  passed.
- Additional tests compare canonical codes to exhaustive enumeration for all
  graph-atlas graphs through five vertices, under relabeling and both normal
  and forced sparse stabilizer paths. Directed colored examples cover incoming
  edges, loops, and independently enumerated automorphism groups. A symmetric
  star control verifies reduced visited nodes without a timing assertion.
- The audit compares all 159 complete records from the same frozen V6
  verification run. It checks completion, minima, reference flags, labeled
  counts and normalized reaction-center frequencies. Labeled ITS/template
  class counts are compared where both analyses are complete, and loss of
  prior structure completeness is checked separately.
- The initial candidate improved classification but caused
  29867/reference-CD to time out: successful classification added enough work
  to exhaust the search timer. After materializing cached components, its
  targeted rerun completed in 47.11 seconds. The final full cohort verifies
  that the corrected candidate preserves its completion.
- A component-symmetry discovery experiment was not retained.
- The final source snapshot remained unchanged and matches the current package
  across all hashed Python files:
  {manifest["implementation_sha256"]}.

## Limits and next work

The 60-second cooperative search timer, 100,000-mapping cap, 0.25-second
structure-canonicalization budget, eight workers on CPUs 0–7, and 4 GiB
address-space limit per worker are unchanged. Full task wall includes seed
preprocessing and can exceed the cooperative search timer.

The three remaining shells still time out. V6 diagnostics with structure
classification disabled and a longer timer also reached the mapping cap;
those diagnostics are not V7 validation completions. Further exact
aggregation would need to preserve multiplicities, atom/bond-change
frequencies and held-out reference membership. Faster classification alone
does not establish completion of these large shells.

This version has not been rerun on all 1,200 expansion tasks. Timing and
classification completeness near budget boundaries may vary across runs.

## Artifacts

Round-seven artifacts are in
benchmark_results/synister_efficiency_timeouts_v7_20260908/:

- verified_source/, verified_run/, verification_selection.json;
- completion_summary.json, remaining_after_verification.json;
- regression_references.json, audit_regressions.py,
  verified_run/regression_audit.json;
- tests_verified.log, canonical_control.json and the 29867 profile;
- exploratory snapshots and runs, including the initial candidate's regression.

The authoritative final run is verified_run/; the earlier directory named
final_verification/ retains the pre-cache-fix candidate for review.
Changes remain local and uncommitted.
"""
    (WORKSPACE / "paper/synister/EFFICIENCY_RESULTS_V7_2026-09-08.md").write_text(text)
    log = WORKSPACE / "paper/synister/SESSION_LOG_2026-09-07.md"
    marker = "**Canonicalization and structure-cache improvements — 8 September 2026**"
    if marker not in log.read_text():
        with log.open("a") as stream:
            stream.write(f"\n\n{marker}\n\nV7 preserves all 159 completed searches in the 162-task verification cohort. "
                         f"Complete structure classification increases from {old_classified} to {new_classified}, "
                         f"including {target_classified}/17 of the V6 recovered searches (previously 9/17). "
                         "The three large reference-CD shells remain timeouts at unchanged limits. "
                         "All regression invariants pass; 242 tests passed, one optional skip and "
                         "one known provenance test deselected. No new full 1,200-task rerun. "
                         "See [the seventh-round report](EFFICIENCY_RESULTS_V7_2026-09-08.md).\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"target_results", "classification_gains"}}, indent=2))

if __name__ == "__main__":
    main()
