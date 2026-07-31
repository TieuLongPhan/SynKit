"""Tests for exact whole-stereograph pairwise relations."""

from __future__ import annotations

from rdkit import Chem

from synkit.Graph.Stereo import (
    StereoisomerRelation,
    classify_rdkit_stereoisomer_relation,
)


def _mol(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def test_atom_renumbering_is_identical() -> None:
    molecule = _mol("F[C@](Cl)(Br)I")
    renumbered = Chem.RenumberAtoms(
        molecule,
        tuple(reversed(range(molecule.GetNumAtoms()))),
    )

    result = classify_rdkit_stereoisomer_relation(molecule, renumbered)

    assert result.relation is StereoisomerRelation.IDENTICAL
    assert result.is_definitive
    assert result.left is not None
    assert result.right is not None
    assert result.left.same_stereograph(result.right)


def test_single_tetrahedral_inversion_is_enantiomeric() -> None:
    result = classify_rdkit_stereoisomer_relation(
        _mol("F[C@](Cl)(Br)I"),
        _mol("F[C@@](Cl)(Br)I"),
    )

    assert result.relation is StereoisomerRelation.ENANTIOMERS
    assert result.mirror_left is not None
    assert result.right is not None
    assert result.mirror_left.same_stereograph(result.right)


def test_e_z_pair_is_diastereomeric_not_enantiomeric() -> None:
    result = classify_rdkit_stereoisomer_relation(
        _mol("F/C=C/F"),
        _mol("F/C=C\\F"),
    )

    assert result.relation is StereoisomerRelation.DIASTEREOMERS


def test_e_z_change_with_unchanged_centers_is_diastereomeric() -> None:
    result = classify_rdkit_stereoisomer_relation(
        _mol("F/C=C/[C@H](Cl)[C@@H](Br)I"),
        _mol("F/C=C\\[C@H](Cl)[C@@H](Br)I"),
    )

    assert result.relation is StereoisomerRelation.DIASTEREOMERS


def test_different_connectivity_is_not_a_stereoisomer_relation() -> None:
    result = classify_rdkit_stereoisomer_relation(_mol("CC"), _mol("CCC"))

    assert result.relation is StereoisomerRelation.CONSTITUTIONALLY_DIFFERENT


def test_unspecified_supported_locus_is_incomplete() -> None:
    result = classify_rdkit_stereoisomer_relation(
        _mol("FC(Cl)(Br)I"),
        _mol("F[C@](Cl)(Br)I"),
    )

    assert result.relation is StereoisomerRelation.INCOMPLETE
    assert result.incomplete_loci
    assert not result.is_definitive
