import csv,gzip,json,time,sys,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/"final_source"))
from synkit.Chem.Mapper import blinded_mapped_reaction_problem
from synkit.Chem.Mapper.analysis import _reference_free_slap_seed
import synkit.Chem.Mapper.exact.seed_fragments as fragments
base=json.loads((R/"selection_1200.json").read_text())
with gzip.open(base["dataset"],"rt") as f:
    row=next(d for d in csv.DictReader(f) if int(d["source_line"])==17156)
problem=blinded_mapped_reaction_problem(row["mapped_reaction"].rsplit("|",1)[0],heavy_only=True,blind_seed="synister-global-v1")
spec=importlib.util.spec_from_file_location("synkit.Chem.Mapper.exact.seed_fragments_cpu_experiment",R/"seed_fragments_cpu_experiment.py")
variant=importlib.util.module_from_spec(spec);spec.loader.exec_module(variant)
original=fragments.improve_fragment_seed
records=[]
for repeat in range(3):
    for clock,fn in (("wall",original),("cpu",variant.improve_fragment_seed)):
        fragments.improve_fragment_seed=fn
        start=time.perf_counter()
        mapping,stats=_reference_free_slap_seed(problem.lgp,False,repair=True)
        record={"repeat":repeat,"clock":clock,"wall_seconds":time.perf_counter()-start,"stats":stats}
        records.append(record)
        (R/"17156_seed_experiment.json").write_text(json.dumps(records,indent=2)+"\n")
        print(json.dumps(record),flush=True)
