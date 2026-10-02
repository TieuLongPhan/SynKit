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
start=time.perf_counter()
mapping,stats=_reference_free_slap_seed(problem.lgp,False,repair=True)
a,er=_adjacency_and_elements(problem.lgp[0],False)
b,ep=_adjacency_and_elements(problem.lgp[1],False)
record={"affinity":sorted(os.sched_getaffinity(0)),"threadpools":threadpool_info(),
        "stats":stats,"matrix_a_sha":hashlib.sha256(a.tobytes()).hexdigest(),
        "matrix_b_sha":hashlib.sha256(b.tobytes()).hexdigest()}
proof=enumerate_distance_mappings(problem.lgp,CD="minimal",binary=False,
       max_bijections=None,time_limit_seconds=10,symmetry_pruning=True,
       symmetry_node_properties=("hcounts","charges"),initial_mapping=mapping,
       max_mappings=None,collect_mappings=True,_optimization_only=True)
record["proof"]={"complete":proof.complete,"minimum_cost":proof.minimum_cost,
                 "nodes":proof.visited_nodes,"stats":proof.backend_statistics}
record["wall_seconds"]=time.perf_counter()-start
(R/"17156_affinity_diagnostic.json").write_text(json.dumps(record,indent=2)+"\\n")
print(json.dumps(record),flush=True)
