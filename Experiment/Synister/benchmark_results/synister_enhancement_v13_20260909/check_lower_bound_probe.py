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


from synkit.Chem.Mapper.exact.distance_bounds import atom_profile_costs,blocked_assignment_extreme
from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates,NativeEnumerationStop
from synkit.Chem.Mapper.slap.lap import chemical_distance
a,er=_adjacency_and_elements(problem.lgp[0],False)
b,ep=_adjacency_and_elements(problem.lgp[1],False)
lower=blocked_assignment_extreme(atom_profile_costs(a,b,er,ep),range(len(er)),range(len(ep)),er,ep)
found=[]
def receive(mapping,cost):
    exact=chemical_distance(problem.lgp,mapping,binary=False)
    assert exact==lower
    found.append({"mapping":mapping,"cost":exact})
    raise NativeEnumerationStop("lower_bound_attained")
start=time.perf_counter()
result=enumerate_native_candidates(problem.lgp,lower,
    library_path=(R/"library.txt").read_text().strip(),time_limit_seconds=2,
    max_mappings=None,callback=receive,node_properties=("hcounts","charges"))
record={"lower":lower,"found":found,"search":result,"wall_seconds":time.perf_counter()-start}
(R/"lower_bound_probe_diagnostic.json").write_text(json.dumps(record,indent=2)+"\\n")
print(json.dumps(record),flush=True)
