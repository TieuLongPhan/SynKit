"""Sequential final control and 46-case regression verification."""
import subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
PYTHON="/home/labhhc4/anaconda3/envs/synfrag/bin/python"
library=(R/"verified_library.txt").read_text().strip()
for name,selection in (
    ("control",R/"control_selection.json"),
    ("native_regression",R.parent/"synister_efficiency_timeouts_v9_20260908/native_regression_selection.json"),
):
    command=["taskset","-c","16-31",PYTHON,str(R/"verified_benchmark.py"),
             "--selection",str(selection),"--source-root",str(R/"verified_source"),
             "--library",library,"--output",str(R/name),"--workers","16",
             "--mapping-cap","1000000","--seconds","60","--wall-budget","--compact-json"]
    subprocess.run(command,check=True)
