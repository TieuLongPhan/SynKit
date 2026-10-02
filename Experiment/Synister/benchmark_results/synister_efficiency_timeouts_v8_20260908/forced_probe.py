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
    result = enumerate_distance_mappings(
        problem.lgp, CD=scalar, binary=False, max_bijections=None,
        time_limit_seconds=60, symmetry_pruning=True,
        symmetry_node_properties=("hcounts", "charges"), max_mappings=100000,
        initial_mapping=seed, compute_minimum_cost=False,
        _reactant_symmetry_generators=generators[1:],
    )
    output = dict(source_line=task["source_line"], complete=result.complete,
                  reason=result.truncation_reason, candidates=result.selected_mapping_count,
                  visited_nodes=result.visited_nodes, wall_seconds=time.perf_counter()-started,
                  reactant_group_order=permutation_group_order(generators[1:]),
                  product_group_order=result.symmetry_group_order,
                  statistics=result.backend_statistics)
    stem = ROOT / ("forced_" + str(task["source_line"]))
    with gzip.open(str(stem)+".pkl.gz", "wb") as stream:
        pickle.dump(result.mappings, stream)
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
