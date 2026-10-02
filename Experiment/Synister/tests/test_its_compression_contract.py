"""ITS equivalence does not certify complete fixed-coordinate label exports."""

from itertools import permutations

import networkx as nx
import pytest

from synkit.Chem.Mapper.annotation_evaluation import analyze_annotations
from synkit.Chem.Mapper.evaluation import ExactBondEvaluator, bond_f1
from synkit.Chem.Mapper.identifiability import extract_label, parse_reaction
from synkit.Chem.Mapper.orbit_evaluation import SupportOrbitEvaluator


@pytest.mark.parametrize("evaluator", [ExactBondEvaluator, SupportOrbitEvaluator])
def test_single_its_class_loses_coordinates_but_full_orbit_recovers_scores(evaluator):
    # A structural toy, not a mechanistic claim: any two identical carbon
    # components can be joined. All six bijections have minimum heavy CD 1.
    r, p = parse_reaction("C.C.C>>CC.C")
    maps = list(permutations(range(3)))
    labels = [extract_label(r, p, mapping) for mapping in maps]
    assert {label.weighted_distance for label in labels} == {1}
    bonds = {label.changed_bonds for label in labels}
    assert len(bonds) == 3

    # Independently construct full paired-endpoint attributed ITS graphs.
    graphs = []
    for mapping in maps:
        graph = nx.Graph()
        inverse = {j: i for i, j in enumerate(mapping)}
        for i in range(3):
            graph.add_node(i, color=(r.atomic_numbers[i], r.charges[i],
                                    p.charges[mapping[i]], r.hcounts[i],
                                    p.hcounts[mapping[i]]))
        for j, k, order in p.bonds:
            graph.add_edge(inverse[j], inverse[k], color=(0, order))
        graphs.append(graph)
    assert all(nx.is_isomorphic(
        graphs[0], graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match("color", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("color", None),
    ) for graph in graphs)

    a, b = sorted(bonds, key=lambda x: sorted(x))[:2]
    assert {bond_f1(a, label) - bond_f1(b, label) for label in bonds} == {-1, 0, 1}
    scorer = evaluator(r)
    # An ITS representative is not a complete fixed-coordinate export.
    with pytest.raises(ValueError, match="complete"):
        scorer.paired_envelope(a, b, [a], labels_complete=False)
    with pytest.raises(ValueError, match="complete joint labels"):
        analyze_annotations(r, p, [maps[0]], maps[0], maps[1],
                            joint_labels_complete=False, minimum=1)
    # Explicit full-orbit handling, unlike raw coordinate scoring, removes
    # this symmetry-only variation. This is the plan's stated exception.
    lo, hi = scorer.paired_envelope(a, b, bonds, labels_complete=True)
    assert lo.difference == hi.difference == 0
    assert all(scorer.score(a, label).score == 1 for label in bonds)
