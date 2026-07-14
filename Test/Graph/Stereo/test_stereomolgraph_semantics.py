"""Cross-check SynKit's core stereo semantics against StereoMolGraph.

The cases are independently expressed for SynKit from the public tests in
``maxim-papusha/StereoMolGraph`` at commit
``2189f610f23eaaf992e2e01a12ea4d0532496601``.  In particular, they cover the
semantics exercised by ``tests/unit/test_stereodescriptors.py`` and
``tests/unit/graphs/test_smg.py``.  StereoMolGraph is MIT licensed; copyright
(c) 2025 Maxim Papusha.  No upstream graph implementation is vendored here.
"""

from dataclasses import FrozenInstanceError

import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    PlanarBondStereo,
    TetrahedralStereo,
    apply_stereo_to_rdkit,
    descriptors_from_rdkit,
)


def test_unknown_tetrahedral_parity_is_permutation_invariant():
    first = TetrahedralStereo((6, 0, 1, 2, 3), None)
    permuted = TetrahedralStereo((6, 1, 2, 3, 0), None)

    assert first == permuted
    assert hash(first) == hash(permuted)


def test_unknown_planar_bond_parity_is_permutation_invariant():
    first = PlanarBondStereo((1, 0, 2, 3, 5, 4), None)
    reversed_bond = PlanarBondStereo((5, 4, 3, 2, 1, 0), None)

    assert first == reversed_bond
    assert hash(first) == hash(reversed_bond)


def test_stereo_descriptors_are_immutable():
    descriptor = TetrahedralStereo((6, 0, 1, 2, 3), 1)

    with pytest.raises(FrozenInstanceError):
        descriptor.parity = -1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        descriptor.atoms = (6, 1, 0, 2, 3)  # type: ignore[misc]


def test_virtual_hydrogen_reference_relabels_with_its_center():
    descriptor = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1)

    assert descriptor.relabel({1: 10, 2: 20, 3: 30, 4: 40}).atoms == (
        20,
        10,
        30,
        40,
        "@H:20",
    )


def test_unknown_planar_orientation_clears_rdkit_ez_assignment():
    mol = Chem.MolFromSmiles("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]")
    specified = next(iter(descriptors_from_rdkit(mol).values()))
    unknown = PlanarBondStereo(specified.atoms, None, "upstream-semantics")

    apply_stereo_to_rdkit(mol, [unknown])

    bond = mol.GetBondBetweenAtoms(1, 2)
    assert bond is not None
    assert bond.GetStereo() == Chem.BondStereo.STEREONONE


def test_planar_bond_end_swaps_preserve_or_invert_as_expected():
    reference = PlanarBondStereo((5, 4, 3, 2, 1, 0), 0)
    both_ends_swapped = PlanarBondStereo((4, 5, 3, 2, 0, 1), 0)
    reversed_bond = PlanarBondStereo((1, 0, 2, 3, 5, 4), 0)
    one_end_swapped = PlanarBondStereo((4, 5, 3, 2, 1, 0), 0)

    assert reference == both_ends_swapped == reversed_bond
    assert reference != one_end_swapped
