import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import csv,gzip,json,time
from pathlib import Path
from synkit.Chem.Mapper import blinded_mapped_reaction_problem
from synkit.Chem.Mapper.analysis import _BlindShellObserver,_property_vectors,_reference_free_slap_seed,GlobalShellConfig
from synkit.Chem.Mapper.slap.lap import _adjacency_and_elements,chemical_distance
from synkit.Chem.Mapper.exact.native_frontier import frontier_orbit_search as parallel_orbit_search

ROOT=Path(__file__).resolve().parent
def main():
    selection=json.loads((ROOT/"selection.json").read_text())
    with gzip.open(selection["dataset"],"rt") as f:
        rows={int(row["source_line"]):row for row in csv.DictReader(f)}
    for task in selection["tasks"]:
        started=time.perf_counter()
        problem=blinded_mapped_reaction_problem(rows[task["source_line"]]["mapped_reaction"].rsplit("|",1)[0],
                                              heavy_only=True,blind_seed="synister-global-v1")
        a,labels=_adjacency_and_elements(problem.lgp[0],False)
        b,_=_adjacency_and_elements(problem.lgp[1],False)
        config=GlobalShellConfig(time_limit_seconds=60, max_mappings=1000000)
        observer=_BlindShellObserver(a,b,labels,_property_vectors(problem.lgp,config.reaction_center_properties),config)
        seed,_=_reference_free_slap_seed(problem.lgp,False,repair=True)
        target=chemical_distance(problem.lgp,problem.reference_mapping,binary=False)
        result,merged=parallel_orbit_search(problem.lgp,target,config,seed,observer,
                                           library_path=ROOT/"build/libsynkit_distance_dff7b1ee2a89fc5a8d6a.so",workers=8)
        result["source_line"]=task["source_line"]
        result["reference_class_observed"]=merged.contains_reference(problem.reference_mapping)
        result["its_classes"]=len(observer.structure.its_counts)
        result["template_classes"]=len(observer.structure.template_counts)
        result["wall_seconds"]=time.perf_counter()-started
        (ROOT/f'direct_{task["source_line"]}.json').write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)
        del merged, observer
if __name__=="__main__":
    main()
