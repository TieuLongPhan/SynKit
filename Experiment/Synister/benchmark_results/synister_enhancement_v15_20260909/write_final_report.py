import json,hashlib,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent
B=R.parent
root=R.parents[1]
prior=B/"synister_enhancement_v14_20260909"
cohort_manifest=json.loads((prior/"cohort_1200/manifest.json").read_text())
assert cohort_manifest["source_unchanged"]
cohort=json.loads((prior/"audit_summary.json").read_text())["runs"]["cohort_1200"]
assert cohort["processed"]==1200 and cohort["missing_count"]==0
equivalence=json.loads((R/"serialization_equivalence.json").read_text())
assert equivalence["records_checked"]==1200 and equivalence["identical_json"]
scope=json.loads((R/"source_scope_audit.json").read_text())
assert scope["default_path_ast_identical"] and scope["all_other_python_cpp_files_identical"]
audit=json.loads((R/"audit_summary.json").read_text())["runs"]
repeats={}
for case in (17156,30474,22361):
    rows=[]
    for repeat in range(1,4):
        directory=R/f"hard_repeat_{repeat}_{case}"
        manifest=json.loads((directory/"manifest.json").read_text())
        assert manifest["source_unchanged"] and manifest["immutable_json_payload"]
        timing=json.loads((directory/"case_timings.json").read_text())[0]
        mode="minimal" if case==17156 else "reference_cd"
        record=json.loads((directory/f"{case}_{mode}.json").read_text())
        rows.append({"repeat":repeat,"full_output_seconds":timing["end_to_end_wall_seconds"],
          "search_complete":record.get("result",{}).get("complete",False),
          "structure_complete":record.get("result",{}).get("structure",{}).get("complete",False),
          "complete_below_60":timing["complete_below_60_seconds"],
          "error":record.get("error"),"audit":audit[directory.name]})
    repeats[str(case)]=rows
spec=importlib.util.spec_from_file_location("runner",R/"benchmark_synister_native.py")
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
source_hash=runner.source_hash(R/"source")
assert runner.source_hash(root)==source_hash
final={"cohort":cohort,"cohort_source_sha256":cohort_manifest["source_sha256_python_and_cpp"],
       "final_source_sha256":source_hash,"library_sha256":cohort_manifest["library_sha256"],
       "current_source_matches_final_snapshot":True,"serialization_equivalence_records":1200,
       "source_scope_audit":scope,"fresh_immutable_json_repeats":repeats,
       "cohort_parallel_wall_seconds":cohort_manifest["parallel_phase_seconds"],
       "warning":"Cohort timings use the original serializer; final formatter runtimes are fresh targeted checks only."}
(R/"final_summary.json").write_text(json.dumps(final,indent=2)+"\n")
p=root/"paper/synister/ENUMERATION_ENHANCEMENT_V15_2026-09-09.md"
report=p.read_text().replace("Final validation is running.","Final validation is complete.")
report=report.split("\n## Results\n")[0]+"\n## Results\n\n"
report+=f"The full frozen cohort completed **{cohort['search_complete']}/1,200 searches** and **{cohort['structure_complete']}/1,200 structure classifications**. Exact completed-result comparisons: {cohort['comparisons']}; observable mismatches: {len(cohort['failures'])}; classification regressions: {len(cohort['classification_regressions'])}.\n"
report+=f"\nStrict full-output completion below 60 seconds: **{cohort['strict_structure_complete_below_60']}/1,200**. Recorded completed overruns: "+json.dumps(cohort["completed_over_budget"])+". These timings are not replaced by retries.\n"
report+="\nFinal formatter, three fresh-process executions per case:\n\n| Case | Repeat 1 | Repeat 2 | Repeat 3 | All search/classification/output below 60 s |\n|---|---:|---:|---:|---|\n"
for case,rows in repeats.items():
    ok=all(x["complete_below_60"] and x["structure_complete"] for x in rows)
    report+=f"| {case} | {rows[0]['full_output_seconds']:.3f} s | {rows[1]['full_output_seconds']:.3f} s | {rows[2]['full_output_seconds']:.3f} s | {ok} |\n"
failures={f"{case}/{x['repeat']}":x["audit"]["failures"] for case,rows in repeats.items() for x in rows if x["audit"]["failures"]}
report+="\nFresh-run observable mismatches: "+json.dumps(failures)+".\n"
report+="\nAll **1,200 public result JSON payloads were byte-identical** under the default and immutable-sequence serializers. AST comparison confirms identical default reporting paths and byte-identical remaining Python/C++ implementation files between the cohort and final snapshots.\n"
report+=f"\nThe two-partition cohort took {cohort_manifest['parallel_phase_seconds']/60:.2f} minutes. Summed per-case full-output time was {cohort['total_full_output_seconds']/60:.2f} minutes and overlaps across partitions.\n"
report+=f"\nCohort source SHA-256: {cohort_manifest['source_sha256_python_and_cpp']}.\n\nFinal source SHA-256: {source_hash}.\n\nPortable library SHA-256: {cohort_manifest['library_sha256']}.\n"
report+="\n[Complete cohort records and timing origins](../../benchmark_results/synister_enhancement_v14_20260909/cohort_1200), [final machine-readable summary](../../benchmark_results/synister_enhancement_v15_20260909/final_summary.json), [all-result serialization equivalence](../../benchmark_results/synister_enhancement_v15_20260909/serialization_equivalence.json).\n"
report+="\nMeasured fresh-run success is not a universal runtime guarantee. The 50–55 second engineering margin and the deferred assignment warm-start/packed-batch proposals remain future performance gates if the reported ranges exceed that aim.\n"
p.write_text(report)
print(json.dumps(final,indent=2))
