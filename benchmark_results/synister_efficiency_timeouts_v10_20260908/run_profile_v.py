import subprocess, os
os.environ["SYNKIT_NATIVE_PROFILE"]="1"
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.run(["taskset","-c","16-31","/home/labhhc4/anaconda3/envs/synfrag/bin/python","scripts/benchmark_synister_native.py","--selection",str(root/"selection_30474.json"),"--source-root",str(root/"stage_v_source"),"--library",(root/"stage_v_library.txt").read_text().strip(),"--output",str(root/"profile_v"),"--workers","16","--seconds","600","--compact-json","--mapping-cap","1000000"],check=True)
