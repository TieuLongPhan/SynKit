"""Independent endpoint/component ordering controls on bounded chemical inputs."""
import random

import pytest

from synkit.Chem.Mapper.identifiability import Endpoint, extract_label, parse_reaction
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.orbit_evaluation import SupportOrbitEvaluator


def reorder(endpoint,order):
    inverse={old:new for new,old in enumerate(order)}
    return Endpoint(*(tuple(values[i] for i in order) for values in
                      (endpoint.atomic_numbers,endpoint.charges,endpoint.hcounts)),
                    tuple(sorted((*sorted((inverse[i],inverse[j])),w) for i,j,w in endpoint.bonds)))


@pytest.mark.parametrize('reaction',[
    'CCO.O>>CC=O.O','CC.CC>>CCCC','[NH4+].[Cl-]>>N.Cl','CO.CO>>COC.O'])
def test_independent_atom_and_component_permutations_preserve_complete_labels_and_scores(reaction):
    r,p=parse_reaction(reaction)
    def solve(a,b):
        result=enumerate_distance_mappings([a.graph(),b.graph()],CD='minimal',binary=False,
                                          symmetry_pruning=False,time_limit_seconds=10)
        assert result.complete and result.status=='complete'
        return result
    original=solve(r,p)
    maps={tuple(m) for m in original.mappings}
    labels={extract_label(r,p,m) for m in maps}
    predictions=[min(maps),max(maps)]
    supports=[extract_label(r,p,m).changed_bonds for m in predictions]
    base=SupportOrbitEvaluator(r).paired_envelope(*supports,[x.changed_bonds for x in labels],labels_complete=True)
    rng=random.Random(20260920)
    for _ in range(5):
        left=list(range(len(r.atomic_numbers)));right=list(left)
        rng.shuffle(left);rng.shuffle(right)
        inverse_right={old:new for new,old in enumerate(right)}
        a,b=reorder(r,left),reorder(p,right)
        result=solve(a,b)
        assert result.cost==original.cost
        recovered=set()
        for m in result.mappings:
            restored=[None]*len(left)
            for new,old in enumerate(left): restored[old]=right[m[new]]
            recovered.add(tuple(restored))
        assert recovered==maps
        assert {extract_label(r,p,m) for m in recovered}==labels
        moved_predictions=[[inverse_right[m[old]] for old in left] for m in predictions]
        moved_supports=[extract_label(a,b,m).changed_bonds for m in moved_predictions]
        moved_labels=[extract_label(a,b,m).changed_bonds for m in result.mappings]
        score=SupportOrbitEvaluator(a).paired_envelope(*moved_supports,moved_labels,labels_complete=True)
        assert [w.difference for w in score]==[w.difference for w in base]
