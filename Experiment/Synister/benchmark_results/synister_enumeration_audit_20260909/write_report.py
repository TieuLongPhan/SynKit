"""Write the completed audit report from recorded artifacts only."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
repo=R.parents[1]
s=json.loads((R/"audit_summary.json").read_text())
v=json.loads((R/"validation_summary.json").read_text())
assert all(s["runs"][name]["processed"]==count for name,count in (
    ("python_1200",1200),("native_449_cache_on",449),("native_449_cache_off",449),
    ("native_hard_cache_off_diagnostic",2),("guarded_60s_30474",1),("guarded_60s_22361",1)))
assert all(not r["observable_failures"] for r in s["runs"].values())
assert not s["cache_differential"]["failures"]
for name in ("native_449_cache_on","native_449_cache_off","native_hard_cache_off_diagnostic"):
    assert json.loads((R/name/"manifest.json").read_text())["source_unchanged"]
p=s["runs"]["python_1200"]
on=s["runs"]["native_449_cache_on"]
off=s["runs"]["native_449_cache_off"]
diag=s["runs"]["native_hard_cache_off_diagnostic"]
library=Path((R/"portable_library.txt").read_text().strip())
build=json.loads(library.with_suffix(".json").read_text())
native_manifest=json.loads((R/"native_449_cache_on/manifest.json").read_text())
python_summary=json.loads((R/"python_1200/summary.json").read_text())
base=f"../../benchmark_results/{R.name}"
lines=[
"# Enumeration audit and full timeout rerun — 9 September 2026","",
f"The full original 1,200-task Python campaign finished with **{p['complete']}/1,200 searches complete**, **{len(p['errors'])} errors**, and **{p['search_and_structure_complete']} complete structure classifications**. Every completed search passed the available saved-output comparisons. The three Python timeouts remain 28163, 30474 and 22361.",
"",
f"Separate complete 449-task native campaigns used one freshly compiled portable binary, with the pattern cache enabled and disabled. They completed **{on['complete']}/449** and **{off['complete']}/449** searches respectively. These are separate protocols: the native API supports reference-CD mode, while the original cohort also contains 751 minimum tasks.",
"",
"| Frozen campaign | Tasks processed | Search complete | Search + structure complete | Complete output below 60 s | Errors |",
"| --- | ---: | ---: | ---: | ---: | ---: |",
f"| Python, original search limits | 1200 | {p['complete']} | {p['search_and_structure_complete']} | Not measured through file close | {len(p['errors'])} |",
f"| Portable native, pattern cache on | 449 | {on['complete']} | {on['search_and_structure_complete']} | {on['strict_complete_below_60']} | {len(on['errors'])} |",
f"| Portable native, pattern cache off | 449 | {off['complete']} | {off['search_and_structure_complete']} | {off['strict_complete_below_60']} | {len(off['errors'])} |",
f"| Cache-off hard-case diagnostic, 240 s allowance | 2 | {diag['complete']} | {diag['search_and_structure_complete']} | Not an acceptance run | {len(diag['errors'])} |",
"",
f"Combining the separately run mode partitions gives **{s['combined_mode_partition_coverage']['search_complete']}/1,200 complete searches** and **{s['combined_mode_partition_coverage']['search_and_structure_complete']}/1,200 complete structure classifications**: 751 minimum tasks use Python results and 449 reference-CD tasks use cache-disabled native results. This is coverage across two declared protocols, not one homogeneous run. The remaining incomplete structure classification is 13067_minimal.",
"",
"## Audit findings","",
"The inspected native pruning, canonicalization and weighted aggregation preserve exact counting under their documented integer-input and complete-group preconditions. No saved-answer table, reaction-ID special case or held-out mapping input was found in the calculation path. This is an engineering audit with proof arguments and independent tests, not formal verification of every program execution.",
"",
"A numeric defect was reproduced and fixed in the assignment diagnostic: valid 64-million-cost assignments and alternating paths could be mistaken for infinity at the old 50-million threshold. The two new boundary tests fail on the previous V10 binary and pass on the corrected binary. Accepted production shell bounds are at most one million quarter units; these counterexamples did not show incorrect production shell counts.",
"",
"A second correctness gap affected mixed-type atom labels such as [True, 1] or [0.0, 0]. Ordinary compatibility allows both bijections on two isolated vertices, but typed side groups and ITS keys do not describe the same equivalence relation. The previous native aggregation could report one labeled mapping instead of two. Native preparation now rejects equal labels with different typed representations; three new tests reproduce the missing validation.",
"",
"The input guard was added after the running benchmark snapshots were frozen. All 1,200 cohort tasks (1,077 reaction inputs) were checked against it: all labels are builtins.int and none is rejected. Removing only this helper and its call makes the current wrapper AST identical to the frozen wrapper; every other Python/C++ file matches. Counting on the cohort inputs is unchanged. The cohort timings exclude the new guard, and validated_source retains the final guarded package separately from the timed native_source snapshot.",
"",
"Cache hits are justified by exact colored isomorphism or a full sparse ITS pattern obtained through verified reactant automorphisms. A new pattern is remembered only after its full class is retained. Hashes select candidates; full native certificate payloads determine internal class equality. Fresh worker caches start empty, and eviction only causes additional canonicalization.",
"",
f"The cache-disabled campaign recorded **{off['native_pattern_cache_hits']} pattern hits**. All **{s['cache_differential']['completed_pairs']} mutually complete pairs** have identical reported class maps, multiplicities, reaction-center frequencies and entropy; the whole structure object also agrees. Incomplete 60-second records were not compared as complete answers. The two larger-allowance diagnostics supply separate complete-output checks for the hard cases.",
"",
"Disabling the pattern cache does not remove exact duplicate-suppression sets or pure encoding memoization. The present candidate generator covers double orbits but can revisit them, so exact full-key membership remains necessary for counting each class once. This is not an answer cache.",
"",
"Reference-presence reporting still has a strict mathematical limitation: some flags use SHA-256 membership sets. Those flags assume collision resistance. Public class identifiers are also digests, and corpus comparisons check the published ID/count maps rather than saved raw canonical certificates. Internal native class aggregation retains full certificates and does not merge distinct payloads solely on digest equality. The plan includes stronger exact reference-presence witnesses.",
"",
"## Timeouts and difficult cases","",
"| Source line | Cache on, 60 s budget | Cache off, 60 s budget | Cache off, 240 s diagnostic | Final guarded package, cache off, 60 s |",
"| --- | --- | --- | --- | --- |",
]
for case in (28163,30474,22361):
    row=[str(case)]
    for name in ("native_449_cache_on","native_449_cache_off","native_hard_cache_off_diagnostic",f"guarded_60s_{case}"):
        path=R/name/f"{case}_reference_cd.json"
        if not path.exists():row.append("—");continue
        d=json.loads(path.read_text())
        timings=json.loads((R/name/"case_timings.json").read_text())
        wall=next(t["end_to_end_wall_seconds"] for t in timings if t["source_line"]==case)
        result=d.get("result",{})
        row.append(f"{'complete' if result.get('complete') else result.get('truncation_reason', 'error')}, {wall:.3f} s")
    lines.append("| "+" | ".join(row)+" |")
lines.extend([
"",
"The cache-enabled portable run did not reproduce V10's below-60-second result for 30474. Disabling the pattern cache produced fresh complete results below 60 seconds without PGO, including the final guarded-package check. The earlier host-specific PGO measurements remain separate evidence. Both compiler and cache settings must accompany a timing claim; 30474 still has little timing margin. The longer diagnostic is correctness evidence only.",
"",
"The final guarded-package checks run each hard case in a fresh benchmark process, using the same portable library with the pattern cache disabled. Their clocks include the new validation. All timed attempts are retained; there is no guarantee of a 60-second result on every repetition or machine.",
"",
"All incomplete/error keys are retained in audit_summary.json. Any loss of structure completeness relative to the selected saved reference is listed separately in its classification_regressions field; no partial classification is promoted to complete.",
"",
"## Validation and reproducibility","",
f"- **{v['tests_final']['passed']} tests passed**, one optional PuLP-dependent skip, and one pre-existing frozen-provenance test excluded. Its historical evidence was not rewritten.",
f"- **{v['native_ubsan']['passed']} native tests passed** under undefined-behavior sanitization with recovery disabled.",
"- Twelve additional raw-matrix/permutation cases check signed and half-integer weights, atom types, seeds, orbit coverage, multiplicities, frequencies and cache equivalence. Adversarial cache tests reject invalid generators, verify empty fresh state, and exercise eviction.",
f"- Independent assignment stress: {v['assignment_stress']['matrices_checked']} matrices and {v['assignment_stress']['forced_edges_checked']} forced-edge solves, zero failures. Correctness-focused Ruff checks and git diff --check passed.",
f"- Saved-output comparisons: Python {p['completed_exact_comparisons']}, native cache on {on['completed_exact_comparisons']}, native cache off {off['completed_exact_comparisons']}, diagnostics {diag['completed_exact_comparisons']}, final guarded cases {sum(s['runs'][name]['completed_exact_comparisons'] for name in ('guarded_60s_30474','guarded_60s_22361'))}; zero checked-observable differences.",
"",
f"The Python campaign took **{python_summary['wall_seconds']:.3f} seconds** overall. It used eight task workers on CPUs 0–7, the historical 60-second cooperative search limit and 100,000 mapping cap. Its source hash stayed unchanged. Search completion and full wall-time success remain different measurements.",
"",
"Each native case used sixteen spawned workers, one numerical-library thread per worker, 4 GiB address space per worker, and a shared one-million retained worker-class cap including cross-worker copies. Its deadline starts before case construction; strict timing ends after full public output, JSON serialization and file close. Imports and dataset loading precede this timer. No parent address-space or aggregate service-memory cap was imposed.",
"",
"The cache-on native campaign started on CPUs 16–31 while Python used CPUs 0–7. The cache-off campaign started on CPUs 0–15 after Python finished. The native campaigns used disjoint physical CPU sets but shared host memory bandwidth; ancillary validation also overlapped part of the campaigns. Workers and pattern caches are new per case; the coordinator stays alive across a cohort and may reuse bounded encoding state. The orchestration parent was stopped to remove its queued serial ablation; the active cache-on calculation continued, and the cache-off campaign was launched independently. Final record counts and source-unchanged manifests establish completion.",
"",
f"- Portable binary SHA-256: {build['library_sha256']}",
f"- Native C++ source SHA-256: {build['source_sha256']}",
f"- Frozen native Python/C++ tree SHA-256: {native_manifest['source_sha256_python_and_cpp']}",
f"- Final guarded Python/C++ tree SHA-256: {v['validated_source_sha256_python_and_cpp']}",
f"- Build flags: {' '.join(build['flags'])}; no PGO or host-native flags.",
"",
f"The [detailed mathematical audit and enhancement plan](ENUMERATION_AUDIT_ENHANCEMENT_PLAN_2026-09-09.md) gives derivations, implementation locations, limitations and acceptance gates. Priorities are exact presence reporting, a native minimum-proof phase with a shared deadline, incremental assignment work, lower classification/transfer cost, and a separately proved unique double-orbit construction.",
"",
f"Artifacts: [machine-readable audit]({base}/audit_summary.json), [validation]({base}/validation_summary.json), [cache differential]({base}/cache_differential.json), [frozen sources and runners]({base}/), and per-task JSON/timings/manifests in each campaign directory. All changes remain local and uncommitted.",
])
out=repo/"paper/synister/ENUMERATION_AUDIT_RESULTS_2026-09-09.md"
out.write_text("\n".join(lines)+"\n")
print(out)
