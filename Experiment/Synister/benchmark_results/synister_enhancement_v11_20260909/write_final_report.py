"""Finalize report only from closed records and completed partition manifests."""
import json,hashlib,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent
root=R.parents[1]
m=json.loads((R/"cohort_1200/manifest.json").read_text())
assert m["execution_status"]=="complete" and m["source_unchanged"]
s=json.loads((R/"audit_summary.json").read_text())["runs"]["cohort_1200"]
assert s["processed"]==1200 and s["missing_count"]==0
spec=importlib.util.spec_from_file_location("runner",R/"benchmark_synister_native.py")
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
source_hash=runner.source_hash(R/"final_source")
assert source_hash==m["source_sha256_python_and_cpp"]
current_hash=runner.source_hash(root)
rows=[]
for p in (R/"cohort_1200").glob("*_*.json"):
    d=json.loads(p.read_text())
    if isinstance(d,dict) and "source_line" in d and "mode" in d: rows.append(d)
assert len(rows)==1200
timings={(t["source_line"],t["mode"]):t for t in json.loads((R/"cohort_1200/case_timings.json").read_text())}
by_mode={}
for mode in ("minimal","reference_cd"):
    ds=[d for d in rows if d["mode"]==mode]
    by_mode[mode]={
        "tasks":len(ds),
        "search_complete":sum(d.get("result",{}).get("complete",False) for d in ds),
        "structure_complete":sum(d.get("result",{}).get("structure",{}).get("complete",False) for d in ds),
        "strict_complete":sum(d.get("result",{}).get("structure",{}).get("complete",False)
            and timings[(d["source_line"],mode)]["complete_below_60_seconds"] for d in ds),
    }
hard=[]
for key in ((13067,"minimal"),(28163,"reference_cd"),(30474,"reference_cd"),(22361,"reference_cd")):
    d=next(d for d in rows if (d["source_line"],d["mode"])==key)
    hard.append({"case":f"{key[0]}_{key[1]}","complete":d.get("result",{}).get("complete",False),
        "structure_complete":d.get("result",{}).get("structure",{}).get("complete",False),
        "full_output_seconds":timings[key]["end_to_end_wall_seconds"]})
final={"cohort":s,"by_mode":by_mode,"hard_cases":hard,"source_sha256":source_hash,
       "current_source_matches_frozen":current_hash==source_hash,"library_sha256":m["library_sha256"],
       "cpp_sha256":hashlib.sha256((root/"synkit/Chem/Mapper/exact/native_distance.cpp").read_bytes()).hexdigest(),
       "parallel_phase_seconds":m["parallel_phase_seconds"]}
(R/"final_summary.json").write_text(json.dumps(final,indent=2)+"\n")
p=root/"paper/synister/ENUMERATION_ENHANCEMENT_V11_2026-09-09.md"
report=p.read_text().replace("Full cohort results are pending while the frozen run executes.","The frozen full-cohort rerun has finished.")
report=report.replace("Final cohort totals are pending. Machine-readable progress is in\nbenchmark_results/synister_enhancement_v11_20260909/audit_summary.json.","")
report+="\n## Completed 1,200-task cohort\n\n"
report+="| Mode | Tasks | Search complete | Structure complete | Both complete below 60 s |\n|---|---:|---:|---:|---:|\n"
for mode,x in by_mode.items():
    report+=f"| {mode} | {x['tasks']} | {x['search_complete']} | {x['structure_complete']} | {x['strict_complete']} |\n"
report+=f"\nCompleted reference comparisons: {s['comparisons']}; observable mismatches: {len(s['failures'])}; classification regressions: {len(s['classification_regressions'])}. Pattern-cache hits: {s['pattern_cache_hits']}.\n\n"
report+="| Case | Full-output seconds | Search complete | Structure complete |\n|---|---:|---|---|\n"
for x in hard:
    report+=f"| {x['case']} | {x['full_output_seconds']:.3f} | {x['complete']} | {x['structure_complete']} |\n"
report+=f"\nThe parallel continuation phase took {m['parallel_phase_seconds']/60:.2f} minutes; this excludes the already completed 113-case prefix. Summed per-case times are {s['total_full_output_seconds']/60:.2f} minutes and overlap across the two CPU partitions.\n"
report+="\nIncomplete/error records: "+json.dumps(s["incomplete"])+".\n"
report+="\nExact observable failures: "+json.dumps(s["failures"])+".\n"
report+=f"\nFrozen Python/C++ source SHA-256: {source_hash}.\n\nPortable library SHA-256: {m['library_sha256']}.\n\nCurrent package matches the frozen implementation: {current_hash==source_hash}.\n"
report+="\nFull records, individual timing scopes and origins: [cohort artifacts](../../benchmark_results/synister_enhancement_v11_20260909/cohort_1200). Machine-readable audit: [final summary](../../benchmark_results/synister_enhancement_v11_20260909/final_summary.json).\n"
p.write_text(report)
print(json.dumps(final,indent=2))
