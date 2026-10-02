from copy import deepcopy

import pytest

from Experiment.Synister.audit_annotations import check_witness, label, tokens
from synkit.Chem.Mapper.annotation_evaluation import analyze_annotations
from synkit.Chem.Mapper.identifiability import Endpoint


def example():
    r = Endpoint((6, 6), (0, 1), (4, 3), ())
    maps = [(0, 1), (1, 0)]
    output = analyze_annotations(r, r, maps, *maps, joint_labels_complete=True, minimum=0)
    ys = {tokens(label(r, r, m), "atom") for m in maps}
    return r, maps, output, ys


def test_independent_feature_witness_checks_and_tampering():
    r, maps, output, ys = example()
    predictions = [tokens(label(r, r, m), "atom") for m in maps]
    witness = output["metrics"]["atom_f1"]["lower"]
    check_witness(r, predictions, ys, witness)
    bad = deepcopy(witness)
    bad["a_score"] = "1/7"
    with pytest.raises(AssertionError, match="score mismatch"):
        check_witness(r, predictions, ys, bad)
    bad = deepcopy(witness)
    bad["a_transporter"] = [1, 0]
    with pytest.raises(AssertionError, match="transporter"):
        check_witness(r, predictions, ys, bad)


def reordered(endpoint, order):
    inverse = {old: new for new, old in enumerate(order)}
    return Endpoint(*(tuple(getattr(endpoint, name)[i] for i in order)
                      for name in ("atomic_numbers", "charges", "hcounts")),
                    tuple(sorted((*sorted((inverse[i], inverse[j])), w)
                                 for i, j, w in endpoint.bonds)))


def test_independent_endpoint_reordering_preserves_policy_codes_and_scores():
    r, maps, original, _ = example()
    # Only the product is reversed: this is not a simultaneous relabeling.
    product_order = (1, 0)
    product = reordered(r, product_order)
    inverse = {old: new for new, old in enumerate(product_order)}
    changed = [tuple(inverse[m[i]] for i in range(2)) for m in maps]
    output = analyze_annotations(r, product, changed, *changed,
                                 joint_labels_complete=True, minimum=0)
    for name in original["metrics"]:
        for field in ("width", "label_orbits", "fixed_labels"):
            assert original["metrics"][name][field] == output["metrics"][name][field]
    for name in original["policies"]:
        for field in ("its_code", "a_score", "b_score", "difference"):
            assert original["policies"][name][field] == output["policies"][name][field]
