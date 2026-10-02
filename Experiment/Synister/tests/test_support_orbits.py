from itertools import permutations
import random

import pytest

from synkit.Chem.Mapper.identifiability import Endpoint, parse_reaction
from synkit.Chem.Mapper.evaluation import ExactBondEvaluator, IncompleteSymmetry, transport_bonds
from synkit.Chem.Mapper.orbit_evaluation import SupportOrbitEvaluator


def test_support_chain_matches_literal_full_groups():
    rng = random.Random(8819)
    for _ in range(30):
        n = rng.randrange(2, 7)
        r = Endpoint((6,) * n, (0,) * n, (0,) * n,
                     tuple((i, j, 2) for i in range(n) for j in range(i+1, n) if rng.random() < 0.3))
        edges = {(i, j): w for i, j, w in r.bonds}
        # Independent brute-force permutation authority, not VF2 enumeration.
        group = [g for g in permutations(range(n)) if all(
            edges.get((i, j), 0) == edges.get(tuple(sorted((g[i], g[j]))), 0)
            for i in range(n) for j in range(i+1, n))]
        sparse = SupportOrbitEvaluator(r)
        full = ExactBondEvaluator(r)
        labels = [frozenset((i, j) for i in range(n) for j in range(i+1, n) if rng.random() < 0.3)
                  for _ in range(3)]
        for y in labels:
            orbit = sparse.orbit(y)
            assert set(orbit) == {transport_bonds(y, g) for g in group}
            for image, witness in orbit.items():
                assert witness in group
                assert image == transport_bonds(y, witness)
            for prediction in labels:
                assert sparse.score(prediction, y).score == full.score(prediction, y).score
        a = sparse.paired_envelope(labels[0], labels[1], labels, labels_complete=True)
        b = full.paired_envelope(labels[0], labels[1], labels, labels_complete=True)
        assert [x.difference for x in a] == [x.difference for x in b]


def test_spectator_permutations_need_not_be_expanded():
    r, _ = parse_reaction("CO." + ".".join(["Cl"] * 12) + ">>CO." + ".".join(["Cl"] * 12))
    sparse = SupportOrbitEvaluator(r)
    orbit = sparse.orbit({(0, 1)})
    assert len(orbit) == 1
    assert sparse.isomorphism_calls == 0  # 12! spectator permutations irrelevant
    assert sparse.score({(0, 1)}, {(0, 1)}).score == 1


def test_label_orbits_move_between_identical_components():
    r, _ = parse_reaction("CC.CC.CC>>CC.CC.CC")
    sparse = SupportOrbitEvaluator(r)
    assert set(sparse.orbit({(0, 1)})) == {frozenset({(0, 1)}), frozenset({(2, 3)}), frozenset({(4, 5)})}
    # Formed cross-component bonds are in the metric universe too.
    assert len(sparse.orbit({(0, 2)})) == 12


def test_label_image_cap_is_not_an_automorphism_cap():
    r, _ = parse_reaction("CC.CC.CC>>CC.CC.CC")
    with pytest.raises(IncompleteSymmetry):
        SupportOrbitEvaluator(r, max_label_images=2).orbit({(0, 1)})
    for time in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SupportOrbitEvaluator(r, time_limit_seconds=time)


@pytest.mark.parametrize('backend',[ExactBondEvaluator,SupportOrbitEvaluator])
def test_identical_predictions_zero_difference_with_empty_and_nonempty_labels(backend):
    r,_=parse_reaction('CCCO.CC>>CCCO.CC')
    labels=[frozenset(),frozenset({(0,1)}),frozenset({(2,3)}),frozenset({(0,4)})]
    scorer=backend(r)
    for prediction in labels:
        lo,hi=scorer.paired_envelope(prediction,prediction,labels,labels_complete=True)
        assert lo.difference==hi.difference==0


@pytest.mark.parametrize('backend',[ExactBondEvaluator,SupportOrbitEvaluator])
def test_single_label_zero_width_does_not_imply_equal_method_scores(backend):
    r,_=parse_reaction('CCCO>>CCCO')
    scorer=backend(r)
    a,b=frozenset({(0,1)}),frozenset({(2,3)})
    for target in (frozenset(),a,b):
        lo,hi=scorer.paired_envelope(a,b,[target],labels_complete=True)
        assert lo.difference==hi.difference
    lo,hi=scorer.paired_envelope(a,b,[a],labels_complete=True)
    assert lo.difference==hi.difference==1
