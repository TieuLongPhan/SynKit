"""Finish the two difficult shells without the pattern shortcut (diagnostic only)."""
import os,time,json,subprocess,datetime
from pathlib import Path
r=Path(__file__).resolve().parent
selection=json.loads((r/"selection_449_reference_cd.json").read_text())
selection["tasks"]=[t for t in selection["tasks"] if t["source_line"] in (30474,22361)]
(r/"selection_hard_2.json").write_text(json.dumps(selection,indent=2)+"\n")
# Wait for the cache-on cohort to release its physical cores.
while True:
    manifest=json.loads((r/"native_449_cache_on/manifest.json").read_text())
    if manifest.get("source_unchanged"):
        break
    time.sleep(10)
library=(r/"portable_library.txt").read_text().strip()
command=["taskset","-c","16-31","/home/labhhc4/anaconda3/envs/synfrag/bin/python",str(r/"benchmark_synister_native.py"),"--selection",str(r/"selection_hard_2.json"),"--source-root",str(r/"native_source"),"--library",library,"--output",str(r/"native_hard_cache_off_diagnostic"),"--seconds","240","--workers","16","--mapping-cap","1000000","--wall-budget","--compact-json"]
env=dict(os.environ,SYNKIT_NATIVE_PATTERN_CACHE="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
env.pop("SYNKIT_NATIVE_PROFILE",None)
(r/"diagnostic_launch.json").write_text(json.dumps({"command":command,"started":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pattern_cache":"0","timing_eligibility":"240-second diagnostic; does not count as a 60-second recovery"},indent=2)+"\n")
with (r/"native_hard_cache_off_diagnostic.log").open("w") as log:
    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
