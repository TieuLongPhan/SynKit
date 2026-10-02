import subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.run(["taskset","-c","16-31","/home/labhhc4/anaconda3/envs/synfrag/bin/python","scripts/benchmark_synister_native.py","--selection",str(root/"selection.json"),"--source-root",str(root/"stage_e_source"),"--library",(root/"stage_e_library.txt").read_text().strip(),"--output",str(root/"stage_e"),"--workers","16","--seconds","600","--mapping-cap","1000000"],check=True)
