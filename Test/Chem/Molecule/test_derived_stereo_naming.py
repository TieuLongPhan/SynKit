"""Exact-assignment boundary laws for derived stereo naming."""

from dataclasses import replace

import pytest
from rdkit import Chem

from synkit.Chem.Molecule.derived_stereo_naming import (
    DERIVED_STEREO_NAMING_SCHEMA,
    derive_rdkit_stereo_names,
)
from synkit.Graph.Stereo import (
    StereoAssignment,
    TetrahedralStereo,
    enumerate_rdkit_stereographs,
)


@pytest.fixture(scope="module")
def tetrahedral_case():
    molecule = Chem.MolFromSmiles("FC(Cl)Br")
    assert molecule is not None
    enumeration = enumerate_rdkit_stereographs(molecule)
    return molecule, enumeration


def test_exact_enantiomer_assignments_receive_opposite_derived_names(
    tetrahedral_case,
) -> None:
    molecule, enumeration = tetrahedral_case

    named = tuple(
        derive_rdkit_stereo_names(molecule, assignment)
        for assignment in enumeration.assignments
    )

    assert enumeration.exact_assignment_count == 2
    assert {result.assignments[0].label for result in named} == {"R", "S"}
    assert all(result.complete for result in named)
    assert all(result.schema == DERIVED_STEREO_NAMING_SCHEMA for result in named)


def test_naming_retains_the_exact_source_certificate(
    tetrahedral_case,
) -> None:
    molecule, enumeration = tetrahedral_case
    assignment = enumeration.assignments[0]
    before = assignment.registry_items

    result = derive_rdkit_stereo_names(molecule, assignment)

    assert result.source_canonical_digest == assignment.canonical_digest
    assert assignment.canonical_code.startswith("synkit.canonical-stereograph/2\n")
    assert assignment.registry_items == before
    assert result.to_dict()["source_canonical_digest"] == (assignment.canonical_digest)


def test_names_and_toolkit_caches_cannot_change_exact_identity(
    tetrahedral_case,
) -> None:
    molecule, enumeration = tetrahedral_case
    assignment = enumeration.assignments[0]
    before = assignment.canonical_code, assignment.canonical_digest
    first = derive_rdkit_stereo_names(molecule, assignment)
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp("_CIPCode", "R" if index % 2 else "S")
        atom.SetIntProp("_CIPRank", 1000 - index)

    second = derive_rdkit_stereo_names(molecule, assignment)

    assert first == second
    assert (assignment.canonical_code, assignment.canonical_digest) == before


def test_rdkit_mapped_namespace_is_consumed_without_identity_rewrite() -> None:
    molecule = Chem.MolFromSmiles("[F:10][CH:20]([Cl:30])[Br:40]")
    assert molecule is not None
    enumeration = enumerate_rdkit_stereographs(molecule)

    labels = {
        derive_rdkit_stereo_names(molecule, assignment).assignments[0].label
        for assignment in enumeration.assignments
    }

    assert labels == {"R", "S"}
    assert all(
        next(iter(assignment.registry().values())).center == 20
        for assignment in enumeration.assignments
    )


def test_planar_assignments_receive_e_and_z_without_becoming_chirality() -> None:
    molecule = Chem.MolFromSmiles("FC=CCl")
    assert molecule is not None
    enumeration = enumerate_rdkit_stereographs(molecule)

    named = tuple(
        derive_rdkit_stereo_names(molecule, assignment)
        for assignment in enumeration.assignments
    )

    assert {result.assignments[0].label for result in named} == {"E", "Z"}
    assert all(
        assignment.mirror_status.value == "achiral"
        for assignment in enumeration.assignments
    )


def test_unsupported_local_name_remains_a_typed_result() -> None:
    molecule = Chem.MolFromSmiles("[Pt](F)(Cl)(Br)I")
    assert molecule is not None
    molecule.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_SQUAREPLANAR)
    molecule.GetAtomWithIdx(0).SetIntProp("_chiralPermutation", 1)
    enumeration = enumerate_rdkit_stereographs(molecule)

    result = derive_rdkit_stereo_names(
        molecule,
        enumeration.assignments[0],
    )

    assert not result.complete
    assert result.labels == ()
    assert result.status_counts == (("unsupported_descriptor", 1),)


def test_corrupt_certificate_digest_fails_closed(tetrahedral_case) -> None:
    molecule, enumeration = tetrahedral_case
    corrupt = replace(
        enumeration.assignments[0],
        canonical_digest="0" * 64,
    )

    with pytest.raises(ValueError, match="digest"):
        derive_rdkit_stereo_names(molecule, corrupt)


def test_unknown_descriptor_cannot_be_named_as_an_exact_assignment(
    tetrahedral_case,
) -> None:
    molecule, enumeration = tetrahedral_case
    assignment = enumeration.assignments[0]
    key, descriptor = assignment.registry_items[0]
    assert isinstance(descriptor, TetrahedralStereo)
    unknown = TetrahedralStereo(
        descriptor.atoms,
        None,
        descriptor.provenance,
    )
    malformed = StereoAssignment(
        ((key, unknown),),
        assignment.canonical_code,
        assignment.canonical_digest,
        assignment.mirror_status,
    )

    with pytest.raises(ValueError, match="complete fixed"):
        derive_rdkit_stereo_names(molecule, malformed)
