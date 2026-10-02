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
selection=json.loads((ROOT/"selection.json").read_text())
with gzip.open(selection["dataset"],"rt") as f:
    rows={int(r["source_line"]):r for r in csv.DictReader(f)}
old=REPO/"benchmark_results/synister_efficiency_timeouts_v9_20260908/build/libsynkit_distance_dff7b1ee2a89fc5a8d6a.so"
new=ROOT/"pgo/train.so"
summaries=[]
for case in (30474,22361):
    p=blinded_mapped_reaction_problem(rows[case]["mapped_reaction"].rsplit("|",1)[0],heavy_only=True,blind_seed="synister-global-v1")
    seed,_=_reference_free_slap_seed(p.lgp,False,repair=True)
    target=chemical_distance(p.lgp,p.reference_mapping,binary=False)
    prepared=prepare_native_candidates(p.lgp,target,library_path=old,initial_mapping=seed)
    result=enumerate_native_candidates(p.lgp,target,library_path=old,max_mappings=2048,time_limit_seconds=60,_prepared=prepared)
    mappings=result["mappings"]
    from collections import deque
    training_prepared=prepare_native_candidates(p.lgp,target,library_path=new,initial_mapping=seed)
    training_queue=deque([()])
    training_count=0
    while training_queue and training_count<4096:
        prefixes=[training_queue.popleft() for _ in range(min(64,len(training_queue)))]
        sample=enumerate_native_candidates(
            p.lgp,target,library_path=new,_prepared=training_prepared,
            max_mappings=4096-training_count,time_limit_seconds=60,
            prefixes=prefixes,slice_nodes=8192)
        training_count+=sample["candidate_count"]
        training_queue.extend(sample["frontier"])
    print("PGO search training",case,training_count,flush=True)
    a,labels=_adjacency_and_elements(p.lgp[0],False)
    b,_=_adjacency_and_elements(p.lgp[1],False)
    rg,pg,ro,po=prepared[4:8]
    previous=None
    for name,lib in (("v9_cpp",old),("v10_cpp",new)):
        observer=_BlindShellObserver(a,b,labels,_property_vectors(p.lgp,("hcounts","charges")),GlobalShellConfig(time_limit_seconds=60))
        acc=OrbitAccumulator(observer,rg[1:],ro,po,library_path=lib)
        library=ctypes.CDLL(str(lib))
        profile_native=hasattr(library,"synkit_canonical_profile_enable")
        if profile_native:library.synkit_canonical_profile_enable(1)
        profiler=cProfile.Profile()
        start=time.perf_counter()
        profiler.enable()
        for mapping in mappings:
            acc.observe(mapping,target)
        profiler.disable()
        elapsed=time.perf_counter()-start
        expected=(observer.count,dict(observer.structure.its_counts),dict(observer.structure.template_counts),dict(acc.atom_frequencies),dict(acc.bond_frequencies))
        if previous is not None:
            assert previous==expected
        previous=expected
        with (ROOT/f"pgo_train_{case}_{name}.txt").open("w") as f:
            pstats.Stats(profiler,stream=f).sort_stats("cumulative").print_stats(25)
        row={"source_line":case,"backend":name,"candidate_count":len(mappings),"seconds_with_profiler":elapsed,"classes":len(acc.seen),"weighted":observer.count}
        if profile_native:
            counters=(ctypes.c_int64*8)()
            library.synkit_canonical_profile_read(counters)
            row["native_phase_nanoseconds_or_calls"]=list(counters)
        print(json.dumps(row),flush=True)
        summaries.append(row)
(ROOT/"pgo_train_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
