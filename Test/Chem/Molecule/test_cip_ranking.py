"""Independent CIP ranking kernel laws for stereo Sprint 31."""

from rdkit import Chem

from synkit.Chem.Molecule.cip_ranking import (
    CIPComparisonOutcome,
    CIPLigandEvidence,
    CIPNodeEvidence,
    CIPRanker,
    CIPSequenceRule,
    CIPSphereEvidence,
    CIPTermination,
)
from synkit.Graph.Stereo import TetrahedralStereo


def test_rule_1a_ranks_direct_atomic_number_with_a_witness() -> None:
    molecule = Chem.MolFromSmiles("C(F)(Cl)(Br)I")
    assert molecule is not None
    ranking = CIPRanker(molecule).rank(0, (1, 2, 3, 4))

    assert ranking.complete
    assert ranking.ordered_references == (4, 3, 2, 1)
    assert all(
        comparison.deciding_rule is CIPSequenceRule.RULE_1A_ATOMIC_NUMBER
        for comparison in ranking.comparisons
    )
    comparison = CIPRanker(molecule).compare(0, 1, 2)
    assert comparison.outcome is CIPComparisonOutcome.RIGHT_HIGHER
    assert comparison.depth == 1
    assert comparison.left_witness == (9,)
    assert comparison.right_witness == (17,)


def test_rule_1a_exhausts_outward_spheres_before_deciding() -> None:
    molecule = Chem.MolFromSmiles("C(C)(CC)(F)Cl")
    assert molecule is not None

    comparison = CIPRanker(molecule).compare(0, 1, 2)

    assert comparison.outcome is CIPComparisonOutcome.RIGHT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_1A_ATOMIC_NUMBER
    assert comparison.depth == 2
    assert comparison.left_witness == (1, 1, 1)
    assert comparison.right_witness == (6, 1, 1)


def test_multiple_bond_duplicate_nodes_decide_carbonyl_over_alcohol() -> None:
    molecule = Chem.MolFromSmiles("C(C=O)(CO)(F)Cl")
    assert molecule is not None

    comparison = CIPRanker(molecule).compare(0, 1, 3)
    carbonyl_nodes = comparison.left_evidence.spheres[1].nodes

    assert comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_1A_ATOMIC_NUMBER
    assert comparison.left_witness == (8, 8, 1)
    assert comparison.right_witness == (8, 1, 1)
    assert any(node.duplicate_kind == "multiple_bond" for node in carbonyl_nodes)


def test_multiple_bond_does_not_repeat_the_parent_core_in_next_sphere() -> None:
    molecule = Chem.MolFromSmiles("COP(C)(F)=O")
    assert molecule is not None

    comparison = CIPRanker(molecule).compare(2, 1, 5)

    assert comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_1A_ATOMIC_NUMBER
    assert comparison.left_witness == (6,)
    assert comparison.right_witness == (0,)


def test_simple_mancude_heterocycle_uses_mean_duplicate_atomic_number() -> None:
    molecule = Chem.MolFromSmiles("O[C@H](/C=N\\C)C1=NC=CC=C1")
    assert molecule is not None

    comparison = CIPRanker(molecule).compare(1, 2, 5)

    assert comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_1A_ATOMIC_NUMBER
    assert comparison.left_witness == (7, 7, 1)
    assert comparison.right_witness == (7, 6.5, 6)


def test_rule_1b_nearer_duplicate_node_has_priority() -> None:
    molecule = Chem.MolFromSmiles("C(F)(Cl)(Br)I")
    assert molecule is not None
    base = CIPNodeEvidence(1, 6, 12.011, 1, (0, 1))
    near = CIPNodeEvidence(2, 6, 12.011, 1, (0, 1, 1), True, 1, "ring_closure")
    far = CIPNodeEvidence(2, 6, 12.011, 1, (0, 1, 1), True, 3, "ring_closure")
    left = CIPLigandEvidence(
        0,
        1,
        (CIPSphereEvidence(1, (base,)), CIPSphereEvidence(2, (near,))),
        (),
        (),
        CIPTermination.EXHAUSTED,
    )
    right = CIPLigandEvidence(
        0,
        2,
        (CIPSphereEvidence(1, (base,)), CIPSphereEvidence(2, (far,))),
        (),
        (),
        CIPTermination.EXHAUSTED,
    )

    comparison = CIPRanker(molecule).compare(
        0,
        1,
        2,
        left_evidence=left,
        right_evidence=right,
    )

    assert comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_1B_DUPLICATE_DISTANCE
    assert comparison.depth == 2


def test_rule_2_uses_isotopic_mass_only_after_rule_1_ties() -> None:
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles("[C]([H])([2H])(F)Cl", parameters)
    assert molecule is not None

    comparison = CIPRanker(molecule).compare(0, 1, 2)

    assert comparison.outcome is CIPComparisonOutcome.RIGHT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_2_ISOTOPE_MASS
    assert comparison.depth == 1


def test_rule_2_uses_nuclide_mass_not_the_integer_mass_number() -> None:
    molecule = Chem.MolFromSmiles("C(O)([16OH])(F)Cl")
    assert molecule is not None

    comparison = CIPRanker(molecule).compare(0, 1, 2)

    assert comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_2_ISOTOPE_MASS
    assert comparison.left_witness is not None
    assert comparison.right_witness is not None
    assert comparison.left_witness > comparison.right_witness


def test_ring_digraph_terminates_and_records_closure_duplicates() -> None:
    molecule = Chem.MolFromSmiles("CC1CCCCC1")
    assert molecule is not None
    evidence = CIPRanker(molecule).build_evidence(0, 1)

    assert evidence.termination is CIPTermination.EXHAUSTED
    assert len(evidence.spheres) <= molecule.GetNumAtoms() + 1
    assert any(
        node.duplicate_kind == "ring_closure"
        for sphere in evidence.spheres
        for node in sphere.nodes
    )
    assert evidence.digest == CIPRanker(molecule).build_evidence(0, 1).digest


def test_atom_renumbering_preserves_the_rank_order() -> None:
    molecule = Chem.MolFromSmiles("C(C)(CC)(F)Cl")
    assert molecule is not None
    original = CIPRanker(molecule).rank(0, (1, 2, 4, 5))
    new_order = (5, 3, 1, 0, 4, 2)
    renumbered = Chem.RenumberAtoms(molecule, list(new_order))
    old_to_new = {old: new for new, old in enumerate(new_order)}
    remapped_refs = tuple(old_to_new[index] for index in (1, 2, 4, 5))
    observed = CIPRanker(renumbered).rank(old_to_new[0], remapped_refs)

    assert original.complete and observed.complete
    assert observed.ordered_references == tuple(
        old_to_new[index] for index in original.ordered_references
    )


def test_toolkit_cip_properties_are_not_inputs_to_ranking() -> None:
    molecule = Chem.MolFromSmiles("C(C)(CC)(F)Cl")
    assert molecule is not None
    before = CIPRanker(molecule).rank(0, (1, 2, 4, 5))
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp("_CIPCode", "R" if index % 2 else "S")
        atom.SetIntProp("_CIPRank", 10_000 - index)
    after = CIPRanker(molecule).rank(0, (1, 2, 4, 5))

    assert after == before
    assert after.digest == before.digest


def test_configured_stereogenic_units_are_witnessed_but_fail_closed() -> None:
    molecule = Chem.MolFromSmiles("C(C(F)Cl)(C(F)Cl)(Br)I")
    assert molecule is not None
    descriptors = (
        TetrahedralStereo((1, 0, 2, 3, "@H:1"), 1),
        TetrahedralStereo((4, 0, 5, 6, "@H:4"), -1),
    )

    comparison = CIPRanker(molecule, configured_descriptors=descriptors).compare(
        0, 1, 4
    )

    assert comparison.outcome is CIPComparisonOutcome.UNSUPPORTED
    assert comparison.deciding_rule is CIPSequenceRule.RULE_4_STEREOGENIC_UNIT
    assert comparison.left_evidence.stereogenic_units
    assert comparison.right_evidence.stereogenic_units
    assert "Sequence Rules 3-5" in comparison.reason


def test_virtual_hydrogen_and_lone_pair_have_rule_1a_priority() -> None:
    molecule = Chem.MolFromSmiles("N(F)Cl")
    assert molecule is not None
    comparison = CIPRanker(molecule).compare(0, "@H:0", "@LP:0")

    assert comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER
    assert comparison.deciding_rule is CIPSequenceRule.RULE_1A_ATOMIC_NUMBER


def test_identical_ligands_return_an_explicit_tie() -> None:
    molecule = Chem.MolFromSmiles("CC(C)F")
    assert molecule is not None
    comparison = CIPRanker(molecule).compare(1, 0, 2)
    ranking = CIPRanker(molecule).rank(1, (0, 2, 3, "@H:1"))

    assert comparison.outcome is CIPComparisonOutcome.TIE
    assert not comparison.decided
    assert not ranking.complete
    assert ranking.ordered_references == ()
