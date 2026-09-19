import csv, gzip, json, os, pickle, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parent

def run(index, task, row):
    os.sched_setaffinity(0, {index})
    from synkit.Chem.Mapper import blinded_mapped_reaction_problem
    from synkit.Chem.Mapper.analysis import _reference_free_slap_seed
    from synkit.Chem.Mapper.slap.lap import chemical_distance
    from synkit.Chem.Mapper.graph.automorphism import bounded_automorphism_permutations
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
    from synkit.Chem.Mapper.exact.symmetry import permutation_group_order
    problem = blinded_mapped_reaction_problem(row["mapped_reaction"].rsplit("|", 1)[0], heavy_only=True, blind_seed="synister-global-v1")
    scalar = chemical_distance(problem.lgp, problem.reference_mapping, binary=False)
    started = time.perf_counter()
    generators, complete = bounded_automorphism_permutations(problem.lgp[0], binary=False, node_properties=("hcounts", "charges"), limit=256, timeout_seconds=.25, max_search_nodes=10000)
    assert complete
    seed, _ = _reference_free_slap_seed(problem.lgp, False, repair=True)
    from synkit.Chem.Mapper.exact.native_candidates import enumerate_native_candidates
    from synkit.Chem.Mapper.analysis import _BlindShellObserver, _property_vectors, GlobalShellConfig
    from synkit.Chem.Mapper.slap.lap import _adjacency_and_elements
    from synkit.Chem.Mapper.exact.orbit_aggregation import OrbitAccumulator
    a,labels=_adjacency_and_elements(problem.lgp[0],False)
    b,_=_adjacency_and_elements(problem.lgp[1],False)
    properties=_property_vectors(problem.lgp,("hcounts","charges"))
    observer=_BlindShellObserver(a,b,labels,properties,GlobalShellConfig(time_limit_seconds=60))
    pg,pc=bounded_automorphism_permutations(problem.lgp[1],binary=False,node_properties=("hcounts","charges"),limit=256,timeout_seconds=.25,max_search_nodes=10000)
    assert pc
    aggregator=OrbitAccumulator(observer,generators[1:],permutation_group_order(generators[1:]),
                                permutation_group_order(pg[1:]),library_path=ROOT/"libsynkit_strong.so")
    result = enumerate_native_candidates(
        problem.lgp, scalar, library_path=ROOT/"libsynkit_strong.so",
        time_limit_seconds=60, max_mappings=100000, initial_mapping=seed, callback=aggregator.observe,
    )
    output = {key:value for key,value in result.items()
              if key not in {"mappings", "reactant_generators"}}
    output["weighted_product_representatives"]=observer.count
    output["labeled_mappings"]=observer.count*aggregator.product_order
    output["its_classes"]=len(observer.structure.its_counts)
    output["template_classes"]=len(observer.structure.template_counts)
    output["duplicates"]=aggregator.duplicates
    output["reference_class_observed"]=aggregator.contains_reference(problem.reference_mapping)
    output["source_line"] = task["source_line"]
    output["wall_seconds"] = time.perf_counter()-started
    stem = ROOT / ("strong_" + str(task["source_line"]))
    with gzip.open(str(stem)+".pkl.gz", "wb") as stream:
        pickle.dump(result["mappings"], stream)
    Path(str(stem)+".json").write_text(json.dumps(output, indent=2)+"\n")
    return output

if __name__ == "__main__":
    selection = json.loads((ROOT/"selection.json").read_text())
    with gzip.open(selection["dataset"],"rt") as stream:
        rows = {int(row["source_line"]):row for row in csv.DictReader(stream)}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, i, task, rows[task["source_line"]]) for i,task in enumerate(selection["tasks"])]
        for future in futures:
            print(json.dumps(future.result()), flush=True)
