"""Independent small controls for the study's label and scoring contract."""

from fractions import Fraction

import pytest

from synkit.Chem.Mapper.identifiability import Endpoint, extract_label, parse_reaction
from synkit.Chem.Mapper.evaluation import (
    ExactBondEvaluator, IncompleteSymmetry, bond_f1, transport_bonds,
)


def test_strict_inventory_and_unmapped_retention():
    r, p = parse_reaction("[CH3:9]O>>CO")
    assert r == p
    assert len(r.atomic_numbers) == 2
    for reaction in ("CO>>C", "CO>O>CO", "[13CH3]O>>CO", "[H][H]>>[H][H]"):
        with pytest.raises(ValueError):
            parse_reaction(reaction)


def test_formed_bonds_and_unary_changes():
    r, p = parse_reaction("C.C>>CC")
    label = extract_label(r, p, (0, 1))
    assert label.changed_bonds == {(0, 1)}
    assert label.centre_atoms == {0, 1}
    assert label.weighted_distance == 1
    assert len(label.unary_changes) == 2
    r, p = parse_reaction("[NH4+]>>N")
    label = extract_label(r, p, (0,))
    assert not label.changed_bonds
    assert label.centre_atoms == {0}


def test_invalid_mapping():
    r, p = parse_reaction("CO>>CO")
    for mapping in ((0, 0), (1, 0), (0,), (0.0, 1.0)):
        with pytest.raises(ValueError):
            extract_label(r, p, mapping)


def test_symmetry_only_variation_disappears():
    r, _ = parse_reaction("CCC>>CCC")
    scorer = ExactBondEvaluator(r)
    left, right = {(0, 1)}, {(1, 2)}
    assert bond_f1(left, right) == 0
    assert scorer.score(left, right).score == 1
    assert scorer.orbit_key(left) == scorer.orbit_key(right)
    lo, hi = scorer.paired_envelope(left, right, [left, right], labels_complete=True)
    assert lo.difference == hi.difference == 0


def test_inequivalent_sites_and_attained_reversal():
    # Same local edge motif at inequivalent ends of the full attributed graph.
    r, _ = parse_reaction("CCCO>>CCCO")
    scorer = ExactBondEvaluator(r)
    a, b = {(0, 1)}, {(1, 2)}
    assert scorer.orbit_key(a) != scorer.orbit_key(b)
    lo, hi = scorer.paired_envelope(a, b, [a, b], labels_complete=True)
    assert (lo.difference, hi.difference) == (-1, 1)
    for witness in (lo, hi):
        assert witness.a.score == bond_f1(a, transport_bonds(witness.label, witness.a.transporter))
        assert witness.b.score == bond_f1(b, transport_bonds(witness.label, witness.b.transporter))
    identical = scorer.paired_envelope(a, a, [a, b], labels_complete=True)
    assert identical[0].difference == identical[1].difference == 0
    singleton = scorer.paired_envelope(a, b, [a], labels_complete=True)
    assert singleton[0].difference == singleton[1].difference == 1


def test_empty_and_fractional_f1():
    assert bond_f1([], []) == 1
    assert bond_f1([], [(0, 1)]) == 0
    assert bond_f1([(0, 1)], [(0, 1), (1, 2)]) == Fraction(2, 3)


def test_partial_results_are_not_exact():
    r, _ = parse_reaction("CCC>>CCC")
    with pytest.raises(IncompleteSymmetry):
        ExactBondEvaluator(r, max_automorphisms=1)
    scorer = ExactBondEvaluator(r)
    with pytest.raises(ValueError):
        scorer.paired_envelope([], [], [[]], labels_complete=False)
    with pytest.raises(ValueError):
        scorer.paired_envelope([], [], [], labels_complete=True)
    with pytest.raises(ValueError):
        scorer.score([(2, 0)], [])


def test_endpoint_rejects_unsupported_numerics():
    with pytest.raises(ValueError):
        Endpoint((6, 6), (0, 0), (0, 0), ((0, 1, 10**16),))


def test_worked_oracle_matches_exact_search_and_product_compression():
    from Experiment.Synister.worked_oracle import REACTION, run
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings

    oracle = run()
    r, p = parse_reaction(REACTION)
    literal_maps = {tuple(m) for m in oracle["minimizing_maps"]}
    expected_labels = {extract_label(r, p, m).changed_bonds for m in literal_maps}
    for compressed in (False, True):
        result = enumerate_distance_mappings(
            [r.graph(), p.graph()], CD="minimal", binary=False,
            symmetry_pruning=compressed, symmetry_node_properties=("hcounts", "charges"),
        )
        assert result.status == "complete"
        assert result.cost == 6
        actual_maps = {tuple(m) for m in result.mappings}
        assert {extract_label(r, p, m).changed_bonds for m in actual_maps} == expected_labels
        if not compressed:
            assert actual_maps == literal_maps
    assert oracle["fixed_bond_labels"] == oracle["bond_label_orbits"] == 2


def test_independent_coordinate_reordering():
    r, p = parse_reaction("CCCO>>CCCO")
    order = (3, 1, 0, 2)
    def permute(endpoint):
        inv = {old: new for new, old in enumerate(order)}
        return Endpoint(
            tuple(endpoint.atomic_numbers[i] for i in order),
            tuple(endpoint.charges[i] for i in order),
            tuple(endpoint.hcounts[i] for i in order),
            tuple(sorted((*sorted((inv[i], inv[j])), w) for i, j, w in endpoint.bonds)),
        )
    inverse = tuple(order.index(i) for i in range(4))
    a, b = {(0, 1)}, {(1, 2)}
    original = ExactBondEvaluator(r).paired_envelope(a, b, [a, b], labels_complete=True)
    a2, b2 = transport_bonds(a, inverse), transport_bonds(b, inverse)
    reordered = ExactBondEvaluator(permute(r)).paired_envelope(a2, b2, [a2, b2], labels_complete=True)
    assert [w.difference for w in original] == [w.difference for w in reordered]
