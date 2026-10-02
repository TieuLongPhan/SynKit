from itertools import permutations
import pytest

from synkit.Chem.Mapper.annotation_evaluation import FeatureOrbitEvaluator, analyze_annotations, features
from synkit.Chem.Mapper.identifiability import Endpoint, extract_label, parse_reaction
from synkit.Chem.Mapper.evaluation import bond_f1, full_automorphisms
from Experiment.Synister.worked_oracle import REACTION, run
from Experiment.Synister.worker import perform


def test_lossy_bond_representative_is_refused_for_joint_metrics():
    # Both mappings have the same empty bond label, but distinct unary edits.
    # One representative per bond label cannot certify joint completeness.
    r=p=Endpoint((6,6),(0,1),(4,3),())
    candidates=[(0,1),(1,0)]
    labels=[extract_label(r,p,m) for m in candidates]
    assert len({x.changed_bonds for x in labels})==1
    assert len(set(labels))==2
    with pytest.raises(ValueError,match='complete joint labels'):
        analyze_annotations(r,p,[candidates[0]],*candidates,
                            joint_labels_complete=False,minimum=0)


def test_all_feature_transports_match_full_group():
    r, p = parse_reaction("CC.CC>>CC.CC")
    label = extract_label(r, p, (0, 2, 1, 3))
    for kind in ("bond", "atom", "typed", "joint"):
        evaluator = FeatureOrbitEvaluator(r)
        value = features(label, kind)
        expected = {evaluator._transport(value, g) for g in full_automorphisms(r)}
        assert set(evaluator.orbit(value)) == expected
        for candidate in expected:
            assert evaluator.score(candidate, value).score == max(bond_f1(candidate, x) for x in expected)


def test_joint_export_preserves_unary_distinctions_lost_by_bond_labels():
    r = p = Endpoint((6, 6), (0, 1), (4, 3), ())
    result = analyze_annotations(r, p, [(0, 1), (1, 0)], (0, 1), (1, 0),
                                 joint_labels_complete=True, minimum=0)
    assert result["structure"]["status"] == "complete"
    assert result["structure"]["its_classes"] == 2
    assert result["metrics"]["joint_exact"]["label_orbits"] == 2
    assert result["metrics"]["bond_f1"]["label_orbits"] == 1
    assert result["metrics"]["bond_f1"]["width"] == "0"
    assert result["metrics"]["atom_f1"]["width"] == "2"


def test_reference_outside_minimum_is_not_forced_into_envelope():
    r, p = parse_reaction("CO.CO>>CO.CO")
    minimum_maps = [(0, 1, 2, 3), (2, 3, 0, 1)]
    wrong = (0, 3, 2, 1)
    result = analyze_annotations(r, p, minimum_maps, minimum_maps[0], wrong,
                                 joint_labels_complete=True, minimum=0, reference_mapping=wrong)
    metric = result["metrics"]["bond_f1"]
    assert not result["reference"]["mapping_in_minimum"]
    assert not metric["reference_orbit_in_minimum"]
    assert metric["lower"]["difference"] == metric["upper"]["difference"] == "1"
    assert metric["reference_scores"] == {"a": "0", "b": "1"}


def test_full_its_admission_equals_minimum_cost_for_every_compatible_map():
    """Full endpoint-pair edge colors preserve CD, unlike coarse labels."""
    from synkit.Chem.Mapper.spectrum import exact_its_and_template_codes
    from synkit.Chem.Mapper.annotation_evaluation import _adjacency_and_elements

    r, p = parse_reaction("CO.CO>>CO.CO")
    rm, elements = _adjacency_and_elements(r.graph(), False)
    pm, _ = _adjacency_and_elements(p.graph(), False)
    properties = {name: (getattr(r, name), getattr(p, name))
                  for name in ("charges", "hcounts")}
    rows = []
    for mapping in permutations(range(len(r.atomic_numbers))):
        if any(r.atomic_numbers[i] != p.atomic_numbers[j] for i, j in enumerate(mapping)):
            continue
        code, _, reason = exact_its_and_template_codes(
            rm, pm, elements, properties, mapping, timeout_seconds=2)
        assert reason is None
        rows.append((extract_label(r, p, mapping).weighted_distance, repr(code)))
    minimum = min(cost for cost, _ in rows)
    admitted = {code for cost, code in rows if cost == minimum}
    assert len(rows) == 4
    assert any(cost > minimum for cost, _ in rows)
    for cost, code in rows:
        assert (code in admitted) == (cost == minimum)


def test_worked_joint_export_and_shared_policy_witnesses():
    exact = perform({"reaction": REACTION, "stage": "exact", "search_seconds": 5,
                     "export_joint_labels": True})
    assert exact["joint_labels_complete"]
    r, p = parse_reaction(REACTION)
    all_maps = run()["minimizing_maps"]
    result = analyze_annotations(r, p, [x["mapping"] for x in exact["joint_labels"]],
                                 all_maps[0], all_maps[-1], joint_labels_complete=True, minimum=6)
    assert result["structure"]["its_classes"] == 2
    assert result["policies"]["nearest_a"]["difference"] == "3/5"
    assert result["policies"]["nearest_b"]["difference"] == "-3/5"
    for policy in result["policies"].values():
        label = extract_label(r, p, policy["mapping"])
        scorer = FeatureOrbitEvaluator(r)
        a = scorer.score(features(extract_label(r, p, all_maps[0]), "bond"), features(label, "bond"))
        b = scorer.score(features(extract_label(r, p, all_maps[-1]), "bond"), features(label, "bond"))
        assert str(a.score-b.score) == policy["difference"]
