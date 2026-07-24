"""Sprint 34 stereo-element perception boundary."""

import pytest
from rdkit import Chem

from synkit.Chem.Molecule.stereo_perception import (
    LocalNeighborKey,
    PotentialStereoElement,
    StereoConfigurationState,
    StereoElementType,
    TetrahedralConstitutionStatus,
    TetrahedralFrameStatus,
    analyze_tetrahedral_carriers,
    canonicalize_tetrahedral_configuration,
    canonicalize_tetrahedral_constitution,
    detect_constitutionally_distinct_tetrahedral_centers,
    detect_potential_stereo_elements,
    detect_tetrahedral_carriers,
    perceive_tetrahedral_stereo,
    tetrahedral_local_neighbor_keys,
)
from synkit.Graph.Stereo import (
    AtomStereoSupport,
    AxisStereoSupport,
    TetrahedralStereo,
    virtual_reference,
)


def _molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def test_perception_partitions_atom_bond_and_axis_supports() -> None:
    molecule = _molecule("FC(Cl)(Br)C=CCl.ClC=C=CCl")

    elements = detect_potential_stereo_elements(molecule)

    assert {element.element_type for element in elements} == {
        StereoElementType.TETRAHEDRAL,
        StereoElementType.DOUBLE_BOND,
        StereoElementType.CUMULENE_AXIS,
    }
    assert (
        sum(
            element.element_type is StereoElementType.DOUBLE_BOND
            for element in elements
        )
        == 1
    )
    assert (
        sum(
            element.element_type is StereoElementType.CUMULENE_AXIS
            for element in elements
        )
        == 1
    )


def test_cumulene_bonds_are_not_independent_double_bond_elements() -> None:
    elements = detect_potential_stereo_elements(_molecule("ClC=C=CCl"))

    assert tuple(element.identifier for element in elements) == ("cumulene_axis:1-2-3",)
    assert isinstance(elements[0].support, AxisStereoSupport)
    assert elements[0].configuration_state is StereoConfigurationState.UNSPECIFIED


def test_odd_cumulene_is_one_extended_cis_trans_path() -> None:
    elements = detect_potential_stereo_elements(_molecule("FC=C=C=CCl"))

    assert tuple(element.identifier for element in elements) == (
        "extended_cis_trans:1-2-3-4",
    )
    assert elements[0].element_type is StereoElementType.EXTENDED_CIS_TRANS
    assert isinstance(elements[0].support, AxisStereoSupport)
    assert elements[0].support.path == (1, 2, 3, 4)
    assert elements[0].configuration_state is StereoConfigurationState.UNSPECIFIED


def test_input_configuration_state_is_evidence_not_a_derived_label() -> None:
    unspecified = detect_potential_stereo_elements(_molecule("FC(Cl)Br"))
    specified = detect_potential_stereo_elements(_molecule("F[C@H](Cl)Br"))

    assert unspecified[0].element_type is StereoElementType.TETRAHEDRAL
    assert unspecified[0].configuration_state is StereoConfigurationState.UNSPECIFIED
    assert specified[0].configuration_state is StereoConfigurationState.SPECIFIED
    assert specified[0].support == unspecified[0].support == AtomStereoSupport(1)
    assert unspecified[0].constitutional_evidence is not None
    assert unspecified[0].configuration is None
    assert specified[0].constitutional_evidence is not None
    assert specified[0].configuration is not None
    assert (
        specified[0].configuration.atoms
        == specified[0].constitutional_evidence.canonical_frame
    )
    assert not hasattr(specified[0], "cip_label")


def test_perception_is_deterministic_and_does_not_mutate_input() -> None:
    molecule = _molecule("F/C=C/Cl")
    before = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)

    first = detect_potential_stereo_elements(molecule)
    second = detect_potential_stereo_elements(molecule)

    assert first == second
    assert first[0].identifier == "double_bond:1-2"
    assert first[0].configuration_state is StereoConfigurationState.SPECIFIED
    assert Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True) == before


def test_element_type_rejects_an_incompatible_support_family() -> None:
    with pytest.raises(TypeError, match="AtomStereoSupport"):
        PotentialStereoElement(
            StereoElementType.TETRAHEDRAL,
            AxisStereoSupport((0, 1), ((2, 3), (4, 5))),
            StereoConfigurationState.UNSPECIFIED,
            "test",
            "test:0",
        )


def test_bare_helical_connectivity_does_not_invent_a_path_configuration() -> None:
    elements = detect_potential_stereo_elements(_molecule("c1ccc2cc3ccccc3cc2c1"))

    assert all(element.element_type.value != "helical" for element in elements)


def test_broad_carrier_detection_precedes_stereogenicity_detection() -> None:
    methane = _molecule("C")
    chiral_carbon = _molecule("FC(Cl)Br")

    assert detect_tetrahedral_carriers(methane) == (AtomStereoSupport(0),)
    assert detect_tetrahedral_carriers(chiral_carbon) == (AtomStereoSupport(1),)
    assert detect_potential_stereo_elements(methane) == ()


def test_exact_center_stabilizer_confirms_four_distinct_ligands() -> None:
    evidence = canonicalize_tetrahedral_constitution(_molecule("FC(Cl)Br"), 1)

    assert evidence.status is TetrahedralConstitutionStatus.CONSTITUTIONALLY_DISTINCT
    assert evidence.confirms_stereogenic_center
    assert evidence.ligand_count == 4
    assert tuple(sorted(item.multiplicity for item in evidence.ligand_classes)) == (
        1,
        1,
        1,
        1,
    )


def test_center_stabilizer_retains_symmetry_related_ligands_as_unresolved() -> None:
    evidence = canonicalize_tetrahedral_constitution(_molecule("CC(C)C"), 1)

    assert evidence.status is TetrahedralConstitutionStatus.SYMMETRY_RELATED
    assert not evidence.confirms_stereogenic_center
    assert tuple(sorted(item.multiplicity for item in evidence.ligand_classes)) == (
        1,
        3,
    )


def test_confirmed_center_detector_filters_symmetry_related_carriers() -> None:
    confirmed = detect_constitutionally_distinct_tetrahedral_centers(
        _molecule("FC(Cl)Br.CC(C)C")
    )

    assert len(confirmed) == 1
    assert confirmed[0].support == AtomStereoSupport(1)
    assert confirmed[0].confirms_stereogenic_center


def test_constitutional_partition_is_independent_of_supplied_chiral_tag() -> None:
    unspecified = canonicalize_tetrahedral_constitution(_molecule("FC(Cl)Br"), 1)
    specified = canonicalize_tetrahedral_constitution(_molecule("F[C@H](Cl)Br"), 1)

    assert unspecified.status is specified.status
    assert tuple(item.multiplicity for item in unspecified.ligand_classes) == tuple(
        item.multiplicity for item in specified.ligand_classes
    )


def test_constitutional_partition_survives_atom_renumbering() -> None:
    molecule = _molecule("FC(Cl)Br")
    renumbered = Chem.RenumberAtoms(molecule, [1, 3, 0, 2])

    original = canonicalize_tetrahedral_constitution(molecule, 1)
    transported = canonicalize_tetrahedral_constitution(renumbered, 0)

    assert original.status is transported.status
    assert sorted(item.multiplicity for item in original.ligand_classes) == sorted(
        item.multiplicity for item in transported.ligand_classes
    )
    assert (
        original.automorphism_witness_checks == transported.automorphism_witness_checks
    )


def test_constitutional_analysis_rejects_non_tetrahedral_carrier() -> None:
    with pytest.raises(ValueError, match="not a supported tetrahedral carrier"):
        canonicalize_tetrahedral_constitution(_molecule("C=C"), 0)


def test_recursive_local_keys_distinguish_same_element_ligands_by_environment() -> None:
    molecule = _molecule("CC(F)(Cl)CC")
    assert molecule.GetAtomWithIdx(0).GetAtomicNum() == 6
    assert molecule.GetAtomWithIdx(4).GetAtomicNum() == 6

    keys = tetrahedral_local_neighbor_keys(molecule, 1)

    assert isinstance(keys[0], LocalNeighborKey)
    assert keys[0] != keys[4]
    assert keys[0].depth == keys[4].depth == molecule.GetNumAtoms()


def test_canonical_local_frame_is_transport_equivariant() -> None:
    molecule = _molecule("FC(Cl)Br")
    new_order = [1, 3, 0, 2]
    mapping = {old: new for new, old in enumerate(new_order)}
    renumbered = Chem.RenumberAtoms(molecule, new_order)

    original = canonicalize_tetrahedral_constitution(molecule, 1)
    transported = canonicalize_tetrahedral_constitution(renumbered, mapping[1])

    assert original.frame_status is TetrahedralFrameStatus.CANONICAL
    assert original.has_canonical_frame
    assert original.canonical_frame is not None
    expected = TetrahedralStereo(original.canonical_frame, 1).relabel(mapping).atoms
    assert transported.canonical_frame == expected
    assert tuple(item.neighborhood_key for item in original.ligand_classes) == tuple(
        item.neighborhood_key for item in transported.ligand_classes
    )


def test_local_frame_ignores_supplied_configuration_and_cip_properties() -> None:
    unspecified = _molecule("FC(Cl)Br")
    configured = _molecule("F[C@H](Cl)Br")
    for index, atom in enumerate(configured.GetAtoms()):
        atom.SetProp("_CIPRank", str(100 - index))
        atom.SetProp("_CIPCode", "R" if index % 2 else "S")

    left = canonicalize_tetrahedral_constitution(unspecified, 1)
    right = canonicalize_tetrahedral_constitution(configured, 1)

    assert left.canonical_frame == right.canonical_frame
    assert tuple(item.neighborhood_key for item in left.ligand_classes) == tuple(
        item.neighborhood_key for item in right.ligand_classes
    )


def test_symmetry_related_carrier_does_not_receive_a_canonical_frame() -> None:
    evidence = canonicalize_tetrahedral_constitution(_molecule("CC(C)C"), 1)

    assert evidence.frame_status is TetrahedralFrameStatus.SYMMETRY_RELATED
    assert evidence.canonical_frame is None
    assert not evidence.has_canonical_frame


def test_exact_audit_fails_closed_on_distinct_local_key_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from synkit.Chem.Molecule import stereo_perception

    monkeypatch.setattr(
        stereo_perception,
        "_local_key_digest",
        lambda _value: "0" * 64,
    )

    evidence = canonicalize_tetrahedral_constitution(_molecule("FC(Cl)Br"), 1)

    assert evidence.confirms_stereogenic_center
    assert evidence.frame_status is TetrahedralFrameStatus.NEIGHBORHOOD_KEY_COLLISION
    assert evidence.canonical_frame is None


def test_configuration_is_attached_only_after_canonical_frame_construction() -> None:
    molecule = _molecule("FC(Cl)Br")
    hydrogen = virtual_reference("H", 1)
    first = TetrahedralStereo((1, 0, 2, 3, hydrogen), 1, "test")
    same_state = TetrahedralStereo((1, 2, 0, 3, hydrogen), -1, "test")

    canonical_first = canonicalize_tetrahedral_configuration(molecule, first)
    canonical_same = canonicalize_tetrahedral_configuration(molecule, same_state)

    assert canonical_first == canonical_same
    assert canonical_first.atoms == canonical_same.atoms
    assert canonical_first.provenance == "test"


def test_canonical_configuration_survives_atom_renumbering() -> None:
    molecule = _molecule("FC(Cl)Br")
    descriptor = TetrahedralStereo(
        (1, 0, 2, 3, virtual_reference("H", 1)),
        1,
    )
    new_order = [1, 3, 0, 2]
    mapping = {old: new for new, old in enumerate(new_order)}
    renumbered = Chem.RenumberAtoms(molecule, new_order)

    original = canonicalize_tetrahedral_configuration(molecule, descriptor)
    transported = canonicalize_tetrahedral_configuration(
        renumbered,
        descriptor.relabel(mapping),
    )

    assert transported == original.relabel(mapping)
    assert transported.parity == original.parity


def test_integrated_perception_configuration_survives_atom_renumbering() -> None:
    molecule = _molecule("F[C@H](Cl)Br")
    new_order = [1, 3, 0, 2]
    mapping = {old: new for new, old in enumerate(new_order)}
    renumbered = Chem.RenumberAtoms(molecule, new_order)

    original = next(
        item
        for item in detect_potential_stereo_elements(molecule)
        if item.element_type is StereoElementType.TETRAHEDRAL
    )
    transported = next(
        item
        for item in detect_potential_stereo_elements(renumbered)
        if item.element_type is StereoElementType.TETRAHEDRAL
    )

    assert original.configuration is not None
    assert transported.configuration is not None
    assert transported.configuration == original.configuration.relabel(mapping)
    assert transported.constitutional_evidence is not None
    assert transported.constitutional_evidence.canonical_frame == (
        transported.configuration.atoms
    )


def test_fixed_point_resolves_center_with_opposite_configured_ligands() -> None:
    molecule = _molecule("F[C@](Cl)([C@H](Br)I)[C@@H](Br)I")

    primary = {
        item.support.center: item for item in analyze_tetrahedral_carriers(molecule)
    }
    perception = perceive_tetrahedral_stereo(molecule)
    resolved = {item.support.center: item for item in perception.carrier_evidence}

    assert primary[1].status is TetrahedralConstitutionStatus.SYMMETRY_RELATED
    assert resolved[1].status is TetrahedralConstitutionStatus.STEREO_DEPENDENT_DISTINCT
    assert resolved[1].dependency_depth == 1
    assert resolved[1].has_canonical_frame
    assert perception.dependency_iterations == 1
    assert {item.support.center for item in perception.elements} == {1, 3, 6}


def test_equal_configured_ligands_remain_symmetry_related() -> None:
    molecule = _molecule("F[C](Cl)([C@H](Br)I)[C@H](Br)I")

    perception = perceive_tetrahedral_stereo(molecule)
    evidence = {item.support.center: item for item in perception.carrier_evidence}

    assert evidence[1].status is TetrahedralConstitutionStatus.SYMMETRY_RELATED
    assert evidence[1].canonical_frame is None
    assert {item.support.center for item in perception.elements} == {3, 6}


def test_focal_configuration_cannot_construct_its_own_frame() -> None:
    molecule = _molecule("F[C@](Cl)([C@H](Br)I)[C@H](Br)I")

    evidence = canonicalize_tetrahedral_constitution(
        molecule,
        1,
        stereo_markers={1: ("tetrahedral", 1)},
        dependency_depth=1,
    )

    assert evidence.status is TetrahedralConstitutionStatus.SYMMETRY_RELATED
    assert evidence.canonical_frame is None
    assert evidence.stereo_marker_count == 0


def test_dependent_frame_ignores_focal_inversion_and_cip_properties() -> None:
    first = _molecule("F[C@](Cl)([C@H](Br)I)[C@@H](Br)I")
    inverted = _molecule("F[C@@](Cl)([C@H](Br)I)[C@@H](Br)I")
    for index, atom in enumerate(inverted.GetAtoms()):
        atom.SetProp("_CIPRank", str(index + 500))
        atom.SetProp("_CIPCode", "R" if index % 2 else "S")

    first_result = perceive_tetrahedral_stereo(first)
    inverted_result = perceive_tetrahedral_stereo(inverted)
    first_center = next(
        item for item in first_result.elements if item.support.center == 1
    )
    inverted_center = next(
        item for item in inverted_result.elements if item.support.center == 1
    )

    assert first_center.constitutional_evidence is not None
    assert inverted_center.constitutional_evidence is not None
    assert (
        first_center.constitutional_evidence.canonical_frame
        == inverted_center.constitutional_evidence.canonical_frame
    )
    assert first_center.configuration is not None
    assert inverted_center.configuration is not None
    assert first_center.configuration.parity == -inverted_center.configuration.parity


def test_dependent_fixed_point_survives_atom_renumbering() -> None:
    molecule = _molecule("F[C@](Cl)([C@H](Br)I)[C@@H](Br)I")
    new_order = list(reversed(range(molecule.GetNumAtoms())))
    mapping = {old: new for new, old in enumerate(new_order)}
    renumbered = Chem.RenumberAtoms(molecule, new_order)

    original = perceive_tetrahedral_stereo(molecule)
    transported = perceive_tetrahedral_stereo(renumbered)
    transported_elements = {item.support.center: item for item in transported.elements}

    assert original.dependency_iterations == transported.dependency_iterations
    for element in original.elements:
        other = transported_elements[mapping[element.support.center]]
        assert element.configuration is not None
        assert other.configuration == element.configuration.relabel(mapping)
        assert other.constitutional_evidence is not None
        assert (
            other.constitutional_evidence.dependency_depth
            == element.constitutional_evidence.dependency_depth
        )
