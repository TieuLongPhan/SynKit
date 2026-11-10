"""SynKit-generated reference frames and chemistry round-trip properties.

Inputs are generated from the declared geometry/adapter contract rather than
selected from another package's examples or fixture collections.
"""
from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    TetrahedralStereo, SquarePlanarStereo, TrigonalBipyramidalStereo,
    OctahedralStereo, PlanarBondStereo, AtropBondStereo,
    apply_stereo_to_rdkit, descriptors_from_rdkit, descriptor_query_matches,
)


@pytest.mark.parametrize("constructor,degree,parity", (
    (TetrahedralStereo, 5, 1), (SquarePlanarStereo, 5, 0),
    (TrigonalBipyramidalStereo, 6, 1), (OctahedralStereo, 7, 1),
    (PlanarBondStereo, 6, 0), (AtropBondStereo, 6, 1),
))
def test_generated_reference_identity_information_and_immutability(constructor, degree, parity):
    atoms = tuple(101 + 19 * i for i in range(degree))
    value = constructor(atoms, parity)
    unknown = constructor(atoms, None)
    assert value == constructor(atoms, parity)
    assert hash(value) == hash(constructor(atoms, parity))
    assert value != unknown
    assert not descriptor_query_matches(unknown, value, unknown_policy="exact")
    assert descriptor_query_matches(unknown, value, unknown_policy="wildcard")
    with pytest.raises(FrozenInstanceError):
        value.parity = None
    renaming = {v: 9001 - 17 * i for i, v in enumerate(atoms)}
    back = {v: k for k, v in renaming.items()}
    assert value.relabel(renaming).relabel(back) == value


@pytest.mark.parametrize("tag,center,ligands,positions,constructor", (
    ("SP", "Pt", ("F", "Cl", "Br", "I"), 3, SquarePlanarStereo),
    ("TB", "As", ("F", "Cl", "Br", "I", "N"), 20, TrigonalBipyramidalStereo),
    ("OH", "Co", ("F", "Cl", "Br", "I", "N", "P"), 30, OctahedralStereo),
))
def test_generated_coordination_positions_survive_projection_and_renumbering(tag, center, ligands, positions, constructor):
    seen = set()
    for configuration in range(1, positions + 1):
        branches = "".join(f"({ligand})" for ligand in ligands[:-1]) + ligands[-1]
        molecule = Chem.MolFromSmiles(f"[{center}@{tag}{configuration}]{branches}")
        assert molecule is not None
        for i, atom in enumerate(molecule.GetAtoms()):
            atom.SetAtomMapNum(1009 + 11 * i)
        registry = descriptors_from_rdkit(molecule)
        descriptor = next(iter(registry.values()))
        assert isinstance(descriptor, constructor)
        seen.add(descriptor.canonical_form())
        Chem.RemoveStereochemistry(molecule)
        apply_stereo_to_rdkit(molecule, registry.values())
        assert next(iter(descriptors_from_rdkit(molecule).values())) == descriptor
        renamed = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
        assert next(iter(descriptors_from_rdkit(renamed).values())) == descriptor
    assert len(seen) == positions


def test_unknown_tetrahedral_value_erases_all_ligand_orders():
    values = {TetrahedralStereo((997, *refs), None)
              for refs in permutations((17, 31, 73, 109))}
    assert len(values) == 1
