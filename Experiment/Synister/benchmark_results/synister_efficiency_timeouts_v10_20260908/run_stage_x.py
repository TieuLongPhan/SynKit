import subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.run(["taskset","-c","16-31","/home/labhhc4/anaconda3/envs/synfrag/bin/python","scripts/benchmark_synister_native.py","--selection",str(root/"selection.json"),"--source-root",str(root/"stage_x_source"),"--library",(root/"stage_x_library.txt").read_text().strip(),"--output",str(root/"stage_x"),"--workers","16","--seconds","600","--compact-json","--mapping-cap","1000000"],check=True)
