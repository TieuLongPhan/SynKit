import csv,gzip,json,time,sys,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/"source"))
from synkit.Chem.Mapper import blinded_mapped_reaction_problem
from synkit.Chem.Mapper.analysis import _reference_free_slap_seed
import synkit.Chem.Mapper.exact.seed_fragments as fragments
base=json.loads((R/"selection_1200.json").read_text())
with gzip.open(base["dataset"],"rt") as f:
    row=next(d for d in csv.DictReader(f) if int(d["source_line"])==17156)
problem=blinded_mapped_reaction_problem(row["mapped_reaction"].rsplit("|",1)[0],heavy_only=True,blind_seed="synister-global-v1")

import os,hashlib
from threadpoolctl import threadpool_info
from synkit.Chem.Mapper.slap.lap import _adjacency_and_elements
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings

tasks=json.loads((R/"selection_part_1.json").read_text())["tasks"]
with gzip.open(base["dataset"],"rt") as f:
    rows={int(x["source_line"]):x for x in csv.DictReader(f)}
spec=importlib.util.spec_from_file_location(
    "synkit.Chem.Mapper.exact.seed_fragments_setup_experiment",
    R/"seed_fragments_setup_experiment.py")
variant=importlib.util.module_from_spec(spec);spec.loader.exec_module(variant)
# Reproduce the same original prefix, then apply only the variant to the target.
records=[]
for task in tasks:
    row=rows[task["source_line"]]
    problem=blinded_mapped_reaction_problem(row["mapped_reaction"].rsplit("|",1)[0],heavy_only=True,blind_seed="synister-global-v1")
    if (task["source_line"],task["mode"])==(17156,"minimal"):
        fragments.improve_fragment_seed=variant.improve_fragment_seed
    start=time.perf_counter()
    mapping,stats=_reference_free_slap_seed(problem.lgp,False,repair=True)
    records.append({"source_line":task["source_line"],"mode":task["mode"],"stats":stats})
    print(task["source_line"],stats.get("cost"),round(time.perf_counter()-start,3),flush=True)
    if (task["source_line"],task["mode"])==(17156,"minimal"):break
(R/"17156_setup_prefix.json").write_text(json.dumps(records,indent=2)+"\n")
print("PREFIX DONE",flush=True)
