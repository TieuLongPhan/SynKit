"""Six independent process-level acceptance runs, with no warm solver cache."""
import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
PYTHON="/home/labhhc4/anaconda3/envs/synfrag/bin/python"
library=(R/"verified_library.txt").read_text().strip()
for trial in range(1,4):
    for case in ((30474,22361) if trial % 2 else (22361,30474)):
        name=f"acceptance_r{trial}_{case}"
        command=["taskset","-c","16-31",PYTHON,str(R/"verified_benchmark.py"),
                 "--selection",str(R/f"selection_{case}.json"),
                 "--source-root",str(R/"verified_source"),"--library",library,
                 "--output",str(R/name),"--workers","16","--mapping-cap","1000000",
                 "--seconds","60","--wall-budget","--compact-json"]
        with (R/f"{name}.log").open("w") as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        row=json.loads((R/name/"case_timings.json").read_text())[0]
        print(json.dumps({"run":name,**row}),flush=True)
