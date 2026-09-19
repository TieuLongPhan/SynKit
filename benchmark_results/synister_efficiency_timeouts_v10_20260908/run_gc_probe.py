# Diagnostic: disable cyclic collection in coordinator only, not in workers.
import gc,runpy,sys
from pathlib import Path
if __name__=="__main__":
    root=Path(__file__).resolve().parent
    gc.disable()
    sys.argv=["scripts/benchmark_synister_native.py","--selection",str(root/"selection.json"),"--source-root",str(root/"stage_c_source"),"--library",(root/"stage_c_library.txt").read_text().strip(),"--output",str(root/"gc_probe"),"--workers","16","--seconds","600","--mapping-cap","1000000"]
    runpy.run_path(sys.argv[0],run_name="__main__")
