# Fixed-candidate microbenchmark only; never a completion result.
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import csv, gzip, json, time, sys, cProfile, pstats, ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
sys.path.insert(0,str(REPO))
from synkit.Chem.Mapper import blinded_mapped_reaction_problem
from synkit.Chem.Mapper.analysis import _BlindShellObserver, _property_vectors, _reference_free_slap_seed, GlobalShellConfig
from synkit.Chem.Mapper.slap.lap import _adjacency_and_elements, chemical_distance
from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates,prepare_native_candidates
from synkit.Chem.Mapper.exact.orbit_aggregation import OrbitAccumulator
selection=json.loads((ROOT/"selection_1200.json").read_text())
with gzip.open(selection["dataset"],"rt") as f:
    rows={int(r["source_line"]):r for r in csv.DictReader(f)}
old=REPO/"benchmark_results/synister_enhancement_v11_20260909/build/libsynkit_distance_4ff96f9468c323f24d07.so"
new=Path((ROOT/"warm_library.txt").read_text().strip())

summaries=[]
for case in (30474,22361,17156):
    p=blinded_mapped_reaction_problem(rows[case]["mapped_reaction"].rsplit("|",1)[0],heavy_only=True,blind_seed="synister-global-v1")
    seed,_=_reference_free_slap_seed(p.lgp,False,repair=True)
    target=chemical_distance(p.lgp,p.reference_mapping,binary=False)
    prepared={name:prepare_native_candidates(p.lgp,target,library_path=lib,initial_mapping=seed) for name,lib in (("cold",old),("warm",new))}
    previous=None
    for repeat in range(8):
        for name,lib in ((("cold",old),("warm",new)) if repeat%2==0 else (("warm",new),("cold",old))):
            start=time.perf_counter()
            result=enumerate_native_candidates(p.lgp,target,library_path=lib,max_mappings=4096,time_limit_seconds=30,_prepared=prepared[name])
            elapsed=time.perf_counter()-start
            mappings=result.pop("mappings")
            if previous is not None: assert mappings==previous
            previous=mappings
            row=dict(case=case,backend=name,repeat=repeat,seconds=elapsed,statistics=result)
            summaries.append(row)
            print(json.dumps(row),flush=True)
(ROOT/"warm_micro_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
