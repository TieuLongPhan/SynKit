import os,subprocess,json,shutil,datetime
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[1]
source=root/"native_source"
shutil.copytree(repo/"synkit",source/"synkit",ignore=shutil.ignore_patterns("__pycache__"))
library=Path((root/"portable_library.txt").read_text().strip())
runs=[]
for cache in ("1","0"):
    name="native_449_cache_"+("on" if cache=="1" else "off")
    command=["taskset","-c","16-31","/home/labhhc4/anaconda3/envs/synfrag/bin/python",str(root/"benchmark_synister_native.py"),"--selection",str(root/"selection_449_reference_cd.json"),"--source-root",str(source),"--library",str(library),"--output",str(root/name),"--seconds","60","--workers","16","--mapping-cap","1000000","--wall-budget","--compact-json"]
    env=dict(os.environ,SYNKIT_NATIVE_PATTERN_CACHE=cache,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    env.pop("SYNKIT_NATIVE_PROFILE",None)
    entry={"name":name,"command":command,"pattern_cache":cache,"started":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    runs.append(entry)
    (root/"native_launches.json").write_text(json.dumps(runs,indent=2)+"\n")
    with (root/(name+".log")).open("w") as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    entry.update(returncode=result.returncode,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root/"native_launches.json").write_text(json.dumps(runs,indent=2)+"\n")
    print(json.dumps(entry),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
