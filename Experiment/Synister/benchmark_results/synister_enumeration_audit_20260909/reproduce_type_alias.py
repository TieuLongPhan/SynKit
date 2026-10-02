"""Reproduce the typed-label undercount against the frozen pre-guard wrapper."""
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/"native_source"))
import numpy as np
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from synkit.Chem.Mapper.exact.native_candidates import prepare_native_candidates,enumerate_native_candidates
from synkit.Chem.Mapper.exact.orbit_aggregation import OrbitAccumulator
from synkit.Chem.Mapper.analysis import GlobalShellConfig,_BlindShellObserver
lib=(R/"portable_library.txt").read_text().strip()
results=[]
for labels in ([True,1],[0.0,0]):
    pair=tuple(LabeledGraph({0:{},1:{}},labels) for _ in range(2))
    prepared=prepare_native_candidates(pair,0,library_path=lib,node_properties=())
    observer=_BlindShellObserver(np.zeros((2,2)),np.zeros((2,2)),labels,{},GlobalShellConfig(symmetry_node_properties=(),reaction_center_properties=()))
    acc=OrbitAccumulator(observer,prepared[4][1:],prepared[6],prepared[7],library_path=lib)
    outcome=enumerate_native_candidates(pair,0,library_path=lib,node_properties=(),callback=acc.observe)
    acc.finish()
    observed=observer.count*prepared[7]
    assert outcome["complete"] and observed==1
    results.append({"labels_repr":repr(labels),"expected_labeled_count_by_explicit_bijections":2,"previous_native_labeled_count":observed,"complete":outcome["complete"],"side_group_orders":list(prepared[6:8])})
(R/"typed_label_counterexamples.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results))
