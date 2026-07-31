"""Derived CIP-label projection laws for stereo Sprint 32."""

from rdkit import Chem

from synkit.Chem.Molecule.cip_assignment import (
    CIPAssignmentStatus,
    assign_cip_label,
    assign_cip_labels,
)
from synkit.Graph.Stereo import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    descriptors_from_rdkit,
)


def test_extended_cis_trans_projects_terminal_priorities_to_e_or_z() -> None:
    molecule = Chem.MolFromSmiles("FC=C=C=CCl")
    assert molecule is not None
    descriptor = ExtendedCisTransStereo(
        (1, 2, 3, 4),
        ((0, "@H:1"), (5, "@H:4")),
        0,
    )

    assert assign_cip_label(molecule, descriptor).label == "Z"
    assert assign_cip_label(molecule, descriptor.invert()).label == "E"


def _descriptor(smiles: str, descriptor_class: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    registry = descriptors_from_rdkit(molecule, require_atom_maps=False)
    descriptor = next(
        value
        for value in registry.values()
        if value.descriptor_class == descriptor_class
    )
    mapping = {index + 1: index for index in range(molecule.GetNumAtoms())}
    return molecule, descriptor, mapping


def test_tetrahedral_projection_matches_known_r_and_s_centres() -> None:
    for smiles, expected in (
        ("F[C@](Cl)(Br)I", "S"),
        ("F[C@@](Cl)(Br)I", "R"),
    ):
        molecule, descriptor, mapping = _descriptor(smiles, "tetrahedral")
        assignment = assign_cip_label(
            molecule,
            descriptor,
            reference_to_index=mapping,
        )

        assert assignment.assigned
        assert assignment.label == expected
        assert assignment.status is CIPAssignmentStatus.ASSIGNED
        assert assignment.rankings[0].complete


def test_tetrahedral_inversion_changes_only_the_derived_label() -> None:
    molecule, descriptor, mapping = _descriptor("F[C@](Cl)(Br)I", "tetrahedral")
    before = descriptor.to_dict()

    assigned = assign_cip_label(molecule, descriptor, reference_to_index=mapping)
    inverted = assign_cip_label(
        molecule, descriptor.invert(), reference_to_index=mapping
    )

    assert (assigned.label, inverted.label) == ("S", "R")
    assert descriptor.to_dict() == before
    assert "label" not in descriptor.to_dict()


def test_isotope_priority_participates_in_tetrahedral_projection() -> None:
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles("[C]([H])([2H])(F)Cl", parameters)
    assert molecule is not None
    descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)

    assignment = assign_cip_label(molecule, descriptor)

    assert assignment.assigned
    assert any(rule.value == "2_isotope_mass" for rule in assignment.required_rules)


def test_planar_projection_assigns_e_and_z_from_ranked_substituents() -> None:
    for smiles, expected in (("F/C=C/F", "E"), ("F/C=C\\F", "Z")):
        molecule, descriptor, mapping = _descriptor(smiles, "planar_bond")

        assignment = assign_cip_label(
            molecule,
            descriptor,
            reference_to_index=mapping,
        )

        assert assignment.assigned
        assert assignment.label == expected
        assert len(assignment.rankings) == 2


def test_atom_renumbering_and_fragment_order_preserve_the_label() -> None:
    molecule, descriptor, mapping = _descriptor("F[C@](Cl)(Br)I.CC", "tetrahedral")
    original = assign_cip_label(molecule, descriptor, reference_to_index=mapping)
    order = (5, 6, 4, 2, 0, 1, 3)
    renumbered = Chem.RenumberAtoms(molecule, list(order))
    old_to_new = {old: new for new, old in enumerate(order)}
    transported_mapping = {
        old + 1: old_to_new[old] for old in range(molecule.GetNumAtoms())
    }

    transported = assign_cip_label(
        renumbered,
        descriptor,
        reference_to_index=transported_mapping,
    )

    assert original.label == transported.label == "S"
    assert original.status is transported.status


def test_planar_unspecified_configuration_never_emits_a_label() -> None:
    molecule, descriptor, mapping = _descriptor("F/C=C/F", "planar_bond")
    unspecified = PlanarBondStereo(
        descriptor.atoms,
        None,
        descriptor.provenance,
    )

    assignment = assign_cip_label(
        molecule,
        unspecified,
        reference_to_index=mapping,
    )

    assert assignment.label is None
    assert assignment.status is CIPAssignmentStatus.UNSPECIFIED_CONFIGURATION


def test_atrop_axis_reversal_is_stable_and_inversion_changes_m_p() -> None:
    molecule = Chem.MolFromSmiles("C(F)(Cl)C(Br)I")
    assert molecule is not None
    descriptor = AtropBondStereo((1, 2, 0, 3, 4, 5), 1)
    reversed_axis = AtropBondStereo((4, 5, 3, 0, 2, 1), 1)

    original = assign_cip_label(molecule, descriptor)
    reversed_result = assign_cip_label(molecule, reversed_axis)
    inverted = assign_cip_label(molecule, descriptor.invert())

    assert original.label == reversed_result.label == "M"
    assert inverted.label == "P"
    assert descriptor.same_configuration(reversed_axis)


def test_cumulene_reversal_is_stable_and_inversion_changes_m_p() -> None:
    molecule = Chem.MolFromSmiles("FC(Cl)=C=C(Br)I")
    assert molecule is not None
    descriptor = CumuleneAxisStereo(
        (1, 3, 4),
        ((0, 2), (5, 6)),
        1,
    )

    original = assign_cip_label(molecule, descriptor)
    reversed_result = assign_cip_label(molecule, descriptor.reversed())
    inverted = assign_cip_label(molecule, descriptor.invert())

    assert original.label == reversed_result.label == "M"
    assert inverted.label == "P"
    assert descriptor.same_configuration(descriptor.reversed())


def test_axis_projection_survives_external_reference_relabeling() -> None:
    molecule = Chem.MolFromSmiles("FC(Cl)=C=C(Br)I")
    assert molecule is not None
    descriptor = CumuleneAxisStereo(
        (1, 3, 4),
        ((0, 2), (5, 6)),
        1,
    )
    external_ids = {index: 20 + index * 3 for index in range(7)}
    external_descriptor = descriptor.relabel(external_ids)
    reference_to_index = {external: index for index, external in external_ids.items()}

    direct = assign_cip_label(molecule, descriptor)
    relabelled = assign_cip_label(
        molecule,
        external_descriptor,
        reference_to_index=reference_to_index,
    )

    assert direct.label == relabelled.label == "M"


def test_symmetric_axis_direction_fails_closed_without_index_bias() -> None:
    molecule = Chem.MolFromSmiles("FC(Cl)=C=C(F)Cl")
    assert molecule is not None
    descriptor = CumuleneAxisStereo(
        (1, 3, 4),
        ((0, 2), (5, 6)),
        1,
    )

    assignment = assign_cip_label(molecule, descriptor)

    assert assignment.label is None
    assert assignment.status is CIPAssignmentStatus.UNRESOLVED_PRIORITY
    assert "axis direction" in assignment.reason


def test_helical_projection_binds_parity_and_preserves_path_reversal() -> None:
    molecule = Chem.MolFromSmiles("CCCC")
    assert molecule is not None
    descriptor = HelicalStereo((0, 1, 2, 3), 1)

    assigned = assign_cip_label(molecule, descriptor)
    reversed_result = assign_cip_label(molecule, descriptor.reversed())
    inverted = assign_cip_label(molecule, descriptor.invert())
    unspecified = assign_cip_label(
        molecule,
        HelicalStereo((0, 1, 2, 3), None),
    )

    assert assigned.label == reversed_result.label == "P"
    assert inverted.label == "M"
    assert unspecified.label is None
    assert unspecified.status is CIPAssignmentStatus.UNSPECIFIED_CONFIGURATION


def test_coordination_geometry_returns_structured_unsupported_result() -> None:
    molecule = Chem.MolFromSmiles("C(F)(Cl)(Br)I")
    assert molecule is not None
    descriptor = SquarePlanarStereo((0, 1, 2, 3, 4), 0)

    assignment = assign_cip_label(molecule, descriptor)

    assert assignment.label is None
    assert assignment.status is CIPAssignmentStatus.UNSUPPORTED_DESCRIPTOR
    assert "Coordination" in assignment.reason


def test_tied_ligands_fail_closed_but_single_rule_five_pair_is_assigned() -> None:
    tied_molecule = Chem.MolFromSmiles("CC(C)(F)Cl")
    assert tied_molecule is not None
    tied = TetrahedralStereo((1, 0, 2, 3, 4), 1)
    tied_result = assign_cip_label(tied_molecule, tied)

    stereo_molecule = Chem.MolFromSmiles("C(C(F)Cl)(C(F)Cl)(Br)I")
    assert stereo_molecule is not None
    target = TetrahedralStereo((0, 1, 4, 7, 8), 1)
    companions = (
        TetrahedralStereo((1, 0, 2, 3, "@H:1"), 1),
        TetrahedralStereo((4, 0, 5, 6, "@H:4"), -1),
    )
    stereo_result = assign_cip_label(
        stereo_molecule,
        target,
        configured_descriptors=companions,
    )

    assert tied_result.label is None
    assert tied_result.status is CIPAssignmentStatus.UNRESOLVED_PRIORITY
    assert stereo_result.label == "r"
    assert stereo_result.status is CIPAssignmentStatus.ASSIGNED
    assert stereo_result.required_rules[-1].value == "5_reflection_variant"


def test_toolkit_cip_cache_is_not_an_assignment_input() -> None:
    molecule, descriptor, mapping = _descriptor("F[C@](Cl)(Br)I", "tetrahedral")
    before = assign_cip_label(molecule, descriptor, reference_to_index=mapping)
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp("_CIPCode", "R" if index % 2 else "S")
        atom.SetIntProp("_CIPRank", 99_999 - index)

    after = assign_cip_label(molecule, descriptor, reference_to_index=mapping)

    assert after == before
    assert after.digest == before.digest


def test_batch_assignment_is_recomputable_without_payload_mutation() -> None:
    molecule = Chem.MolFromSmiles("F[C@](Cl)(Br)I.CCCC")
    assert molecule is not None
    tetra = next(
        iter(
            descriptors_from_rdkit(
                molecule,
                require_atom_maps=False,
            ).values()
        )
    )
    helix = HelicalStereo((6, 7, 8, 9), 1)
    payloads = tetra.to_dict(), helix.to_dict()
    mapping = {index + 1: index for index in range(molecule.GetNumAtoms())}
    mapping.update({6: 5, 7: 6, 8: 7, 9: 8})

    first = assign_cip_labels(
        molecule,
        (tetra, helix),
        reference_to_index=mapping,
    )
    second = assign_cip_labels(
        molecule,
        (tetra, helix),
        reference_to_index=mapping,
    )

    assert tuple(result.label for result in first) == ("S", "P")
    assert first == second
    assert tuple(result.digest for result in first) == tuple(
        result.digest for result in second
    )
    assert (tetra.to_dict(), helix.to_dict()) == payloads
    assert all("label" not in payload for payload in payloads)
