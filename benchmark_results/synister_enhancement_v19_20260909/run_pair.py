import os,subprocess,sys,json
from pathlib import Path
R=Path(__file__).resolve().parent
B=R.parent
arms=[("baseline",B/"synister_enhancement_v15_20260909/source",(B/"synister_enhancement_v15_20260909/library.txt").read_text().strip()),("candidate",R/"source",(R/"library.txt").read_text().strip())]
for name,source,library in arms:
 command=["taskset","-c","0-15",sys.executable,str(R/"benchmark_synister_native.py"),"--selection",str(R/"selection_30474.json"),"--source-root",str(source),"--library",library,"--output",str(R/("paired_90_"+name)),"--seconds","90","--workers","16","--mapping-cap","1000000","--wall-budget","--compact-json","--immutable-json"]
 with (R/("paired_90_"+name+".log")).open("w") as f:subprocess.run(command,env=dict(os.environ,SYNKIT_NATIVE_PATTERN_CACHE="0"),stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,"finished",flush=True)
