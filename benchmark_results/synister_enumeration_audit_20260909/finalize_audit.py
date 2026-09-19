import time,json,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
names=("native_449_cache_on","native_449_cache_off","native_hard_cache_off_diagnostic")
while True:
    done=True
    for name in names:
        try:
            done &= bool(json.loads((R/name/"manifest.json").read_text()).get("source_unchanged"))
        except (FileNotFoundError,json.JSONDecodeError):
            done=False
    if done:break
    time.sleep(15)
python="/home/labhhc4/anaconda3/envs/synfrag/bin/python"
for name in ("audit_results.py","write_report.py"):
    result=subprocess.run([python,str(R/name)],check=True,capture_output=True,text=True)
    (R/(name.removesuffix(".py")+"_final.log")).write_text(result.stdout+result.stderr)
# Record source/binary integrity after all work; .pyc files are outside the source digest.
package=R/"native_source/synkit"
changes=[str(p.relative_to(package)) for p in package.rglob("*") if p.suffix in (".py",".cpp") and p.read_bytes()!=(R.parents[1]/"synkit"/p.relative_to(package)).read_bytes()]
summary=json.loads((R/"audit_summary.json").read_text())
summary["final_package_diff_from_native_snapshot"]=changes
guard=json.loads((R/"post_freeze_type_guard_audit.json").read_text())
assert changes==guard["only_post_freeze_changed_source_files"]
assert guard["tasks_validated"]==1200 and guard["guard_rejections"]==0
assert guard["computation_ast_unchanged_after_removing_guard"]
summary["post_freeze_type_guard_audit"]=guard
summary["all_requested_cohort_records_present"]=all(summary["runs"][name]["processed"]==count for name,count in (("python_1200",1200),("native_449_cache_on",449),("native_449_cache_off",449)))
assert summary["all_requested_cohort_records_present"]
(R/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("All campaigns and reports finished.",flush=True)
