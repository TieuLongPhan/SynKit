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
old=Path((ROOT/"library.txt").read_text().strip())
new=Path((ROOT/"twin_library.txt").read_text().strip())
import importlib.util
import synkit.Chem.Mapper.exact.orbit_aggregation as aggregation
spec=importlib.util.spec_from_file_location("synkit.Chem.Mapper.exact._native_its_before",ROOT/"native_its_buffers.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original_class=module.NativeITSCanonicalizer
updated_class=aggregation.NativeITSCanonicalizer
old=new
summaries=[]
for case in (30474,22361):
    p=blinded_mapped_reaction_problem(rows[case]["mapped_reaction"].rsplit("|",1)[0],heavy_only=True,blind_seed="synister-global-v1")
    seed,_=_reference_free_slap_seed(p.lgp,False,repair=True)
    target=chemical_distance(p.lgp,p.reference_mapping,binary=False)
    prepared=prepare_native_candidates(p.lgp,target,library_path=old,initial_mapping=seed)
    result=enumerate_native_candidates(p.lgp,target,library_path=old,max_mappings=2048,time_limit_seconds=60,_prepared=prepared)
    mappings=result["mappings"]
    a,labels=_adjacency_and_elements(p.lgp[0],False)
    b,_=_adjacency_and_elements(p.lgp[1],False)
    rg,pg,ro,po=prepared[4:8]
    previous=None
    for trial in range(12):
        name,lib=(("baseline",old),("palette",new))[trial % 2 if trial//2%2==0 else 1-trial%2]
        aggregation.NativeITSCanonicalizer=original_class if name=="baseline" else updated_class
        observer=_BlindShellObserver(a,b,labels,_property_vectors(p.lgp,("hcounts","charges")),GlobalShellConfig(time_limit_seconds=60))
        acc=OrbitAccumulator(observer,rg[1:],ro,po,library_path=lib)
        library=ctypes.CDLL(str(lib))
        profile_native=hasattr(library,"synkit_canonical_profile_enable")
        if profile_native:library.synkit_canonical_profile_enable(1)
        profiler=cProfile.Profile()
        start=time.perf_counter()
        # uninstrumented Python timing
        for mapping in mappings:
            acc.observe(mapping,target)
        # no cProfile
        elapsed=time.perf_counter()-start
        expected=(observer.count,dict(observer.structure.its_counts),dict(observer.structure.template_counts),dict(acc.atom_frequencies),dict(acc.bond_frequencies))
        if previous is not None:
            assert previous==expected
        previous=expected
        with (ROOT/f"palette_micro_{case}_{name}.txt").open("w") as f:
            f.write("Timing without cProfile\n")
        row={"trial":trial,"source_line":case,"backend":name,"candidate_count":len(mappings),"seconds_with_profiler":elapsed,"classes":len(acc.seen),"weighted":observer.count}
        if profile_native:
            counters=(ctypes.c_int64*8)()
            library.synkit_canonical_profile_read(counters)
            row["native_phase_nanoseconds_or_calls"]=list(counters)
        print(json.dumps(row),flush=True)
        summaries.append(row)
(ROOT/"palette_micro_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
