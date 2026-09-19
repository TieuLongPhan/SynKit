import json,hashlib,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent
root=R.parents[1]
m=json.loads((R/"cohort_1200/manifest.json").read_text())
assert m["execution_status"]=="complete" and m["source_unchanged"]
audit=json.loads((R/"audit_summary.json").read_text())
s=audit["runs"]["cohort_1200"]
assert s["processed"]==1200 and s["missing_count"]==0
for repeat in range(1,4):
    for case in (17156,30474,22361):
        manifest=json.loads((R/f"hard_repeat_{repeat}_{case}/manifest.json").read_text())
        assert manifest["source_unchanged"]
spec=importlib.util.spec_from_file_location("runner",R/"benchmark_synister_native.py")
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
source_hash=runner.source_hash(R/"source")
assert source_hash==m["source_sha256_python_and_cpp"]
current_source_matches_frozen = runner.source_hash(root)==source_hash
rows=[]
for p in (R/"cohort_1200").glob("*_*.json"):
    d=json.loads(p.read_text())
    if isinstance(d,dict) and "source_line" in d and "mode" in d:rows.append(d)
assert len(rows)==1200
timings={(t["source_line"],t["mode"]):t for t in json.loads((R/"cohort_1200/case_timings.json").read_text())}
modes={}
for mode in ("minimal","reference_cd"):
    docs=[d for d in rows if d["mode"]==mode]
    modes[mode]={"tasks":len(docs),
        "search_complete":sum(d.get("result",{}).get("complete",False) for d in docs),
        "structure_complete":sum(d.get("result",{}).get("structure",{}).get("complete",False) for d in docs),
        "strict_complete":sum(d.get("result",{}).get("structure",{}).get("complete",False) and timings[(d["source_line"],mode)]["complete_below_60_seconds"] for d in docs)}
repeats={}
for case in (17156,30474,22361):
    repeats[str(case)]=[json.loads((R/f"hard_repeat_{repeat}_{case}/case_timings.json").read_text())[0] for repeat in range(1,4)]
final={"cohort":s,"by_mode":modes,"repeats":repeats,
       "source_sha256":source_hash,"current_source_matches_frozen":current_source_matches_frozen,
       "library_sha256":m["library_sha256"],"parallel_phase_seconds":m["parallel_phase_seconds"],
       "repeat_comparisons":{k:v for k,v in audit["runs"].items() if k.startswith("hard_repeat")}}
(R/"final_summary.json").write_text(json.dumps(final,indent=2)+"\n")
p=root/"paper/synister/ENUMERATION_ENHANCEMENT_V13_2026-09-09.md"
report=p.read_text().replace("Corrected full-cohort rerun is queued behind the earlier checks.","The corrected full-cohort rerun and fresh-process checks have finished.")
report=report.split("\n## Results\n")[0]+"\n## Results\n\n"
report+="| Mode | Tasks | Search complete | Structure complete | Both complete below 60 s |\n|---|---:|---:|---:|---:|\n"
for mode,x in modes.items():
    report+=f"| {mode} | {x['tasks']} | {x['search_complete']} | {x['structure_complete']} | {x['strict_complete']} |\n"
report+=f"\nReference comparisons: {s['comparisons']}; observable differences: {len(s['failures'])}; classification regressions: {len(s['classification_regressions'])}. Pattern-cache hits: {s['pattern_cache_hits']}.\n"
report+="\nComparison coverage: "+json.dumps(s["comparison_coverage"])+".\n"
report+="\n| Case | Repeat 1 full output | Repeat 2 | Repeat 3 |\n|---|---:|---:|---:|\n"
for case,ts in repeats.items():
    report+=f"| {case} | {ts[0]['end_to_end_wall_seconds']:.3f} s | {ts[1]['end_to_end_wall_seconds']:.3f} s | {ts[2]['end_to_end_wall_seconds']:.3f} s |\n"
report+="\nRepeat audit failures: "+json.dumps({k:v["failures"] for k,v in final["repeat_comparisons"].items() if v["failures"]})+".\n"
report+="\nIncomplete/error records in the corrected cohort: "+json.dumps(s["incomplete"])+".\n"
report+="\nExact comparison failures: "+json.dumps(s["failures"])+".\n"
report+=f"\nTwo-partition cohort wall time: {m['parallel_phase_seconds']/60:.2f} minutes; summed per-case full-output times: {s['total_full_output_seconds']/60:.2f} minutes. The latter overlap and are not campaign elapsed time.\n"
report+=f"\nFrozen Python/C++ source SHA-256: {source_hash}.\n\nPortable library SHA-256: {m['library_sha256']}.\n"
report+=f"\nCurrent package matches the final frozen source: {current_source_matches_frozen}. [Full records and origins](../../benchmark_results/synister_enhancement_v12_20260909/cohort_1200), [machine-readable summary](../../benchmark_results/synister_enhancement_v12_20260909/final_summary.json).\n"
report+="\nThe earlier V11 and V12 cohorts remain separately reported, including their minimum-proof timeouts; none of their records were substituted into this final run. Runtime observations do not imply a guarantee under arbitrary host contention. The broader warm-start and packed-batch proposals remain undeployed; the explicit canonical-augmentation prototype remains separate from production.\n"
p.write_text(report)
print(json.dumps(final,indent=2))
