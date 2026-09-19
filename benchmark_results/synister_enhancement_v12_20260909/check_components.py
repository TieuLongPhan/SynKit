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

import networkx as nx
a,er=_adjacency_and_elements(problem.lgp[0],False)
b,ep=_adjacency_and_elements(problem.lgp[1],False)
for matrix in (a,b):
    graph=nx.from_numpy_array(matrix)
    print(sorted([len(c) for c in nx.connected_components(graph)],reverse=True))
