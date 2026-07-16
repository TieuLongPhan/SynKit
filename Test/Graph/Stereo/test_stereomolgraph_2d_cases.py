"""2D real-molecule conformance against pinned StereoMolGraph fixtures.

The fixture values are adapted under MIT from StereoMolGraph commit
2189f610f23eaaf992e2e01a12ea4d0532496601; see
``LICENSES/StereoMolGraph-MIT.txt``. No StereoMolGraph runtime dependency is
required for this always-on SynKit suite.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    SquarePlanarStereo,
    apply_stereo_to_rdkit,
    descriptors_from_rdkit,
)
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph

from Test.Graph.Stereo.stereomolgraph_2d_fixtures import (
    INCHI_ROUND_TRIP_CASES,
    MAPPED_CONNECTIVITY_ATOMS,
    MAPPED_CONNECTIVITY_SMILES,
    NON_TETRAHEDRAL_RDKIT_CASES,
    PINNED_COMMIT,
    SQUARE_PLANAR_CASES,
    InchiCase,
    NonTetrahedralCase,
)

EXPECTED_RDKIT_TAGS = {
    "square_planar": Chem.ChiralType.CHI_SQUAREPLANAR,
    "trigonal_bipyramidal": Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    "octahedral": Chem.ChiralType.CHI_OCTAHEDRAL,
}

DEFERRED_NON_TETRAHEDRAL_CASES = tuple(
    case
    for case in NON_TETRAHEDRAL_RDKIT_CASES
    if case.stereo_class != "square_planar"
)


def _assign_index_atom_maps(molecule: Chem.Mol) -> None:
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)


def _clear_square_planar_tag(molecule: Chem.Mol) -> None:
    center = next(
        atom
        for atom in molecule.GetAtoms()
        if atom.GetChiralTag() == Chem.ChiralType.CHI_SQUAREPLANAR
    )
    center.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if center.HasProp("_chiralPermutation"):
        center.ClearProp("_chiralPermutation")


def test_mapped_connectivity_fixture_preserves_all_atom_maps() -> None:
    molecule = Chem.MolFromSmiles(MAPPED_CONNECTIVITY_SMILES, sanitize=False)
    assert molecule is not None
    molecule = Chem.AddHs(molecule)

    graph = MolToGraph().transform(molecule, use_index_as_atom_map=True)

    assert set(graph) == MAPPED_CONNECTIVITY_ATOMS
    assert graph.has_edge(1, 2)
    assert graph.has_edge(2, 33)
    assert graph.has_edge(33, 4)
    assert graph.has_edge(4, 5)
    assert graph.has_edge(5, 13)


@pytest.mark.parametrize(
    "case",
    INCHI_ROUND_TRIP_CASES,
    ids=lambda case: case.name,
)
def test_inchi_survives_synkit_2d_graph_round_trip(case: InchiCase) -> None:
    """Preserve constitution and supported stereo with explicit hydrogens."""
    molecule = Chem.MolFromInchi(case.inchi)
    assert molecule is not None
    molecule = Chem.AddHs(molecule)

    graph = MolToGraph().transform(molecule)
    rebuilt = GraphToMol().graph_to_mol(graph)

    assert len(graph.graph["stereo_descriptors"]) == (case.expected_stereo_descriptors)
    assert Chem.MolToInchi(rebuilt) == case.inchi


@pytest.mark.parametrize(
    "case",
    SQUARE_PLANAR_CASES,
    ids=lambda case: case.name,
)
def test_square_planar_fixture_survives_clearing_and_application(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None
    _assign_index_atom_maps(molecule)

    before = descriptors_from_rdkit(molecule)
    assert len(before) == 1
    assert isinstance(next(iter(before.values())), SquarePlanarStereo)

    _clear_square_planar_tag(molecule)
    assert descriptors_from_rdkit(molecule) == {}

    apply_stereo_to_rdkit(molecule, before.values())
    assert descriptors_from_rdkit(molecule) == before


@pytest.mark.parametrize(
    "case",
    SQUARE_PLANAR_CASES,
    ids=lambda case: case.name,
)
def test_square_planar_fixture_survives_synkit_graph_round_trip(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None

    graph = MolToGraph().transform(molecule)
    rebuilt = GraphToMol().graph_to_mol(graph)

    assert len(graph.graph["stereo_descriptors"]) == 1
    assert descriptors_from_rdkit(
        rebuilt,
        require_atom_maps=False,
    ) == graph.graph["stereo_descriptors"]


@pytest.mark.parametrize(
    "case",
    SQUARE_PLANAR_CASES,
    ids=lambda case: case.name,
)
def test_square_planar_fixture_is_atom_renumbering_invariant(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None
    _assign_index_atom_maps(molecule)
    expected = descriptors_from_rdkit(molecule)

    order = list(reversed(range(molecule.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(molecule, order)
    assert descriptors_from_rdkit(renumbered) == expected

    _clear_square_planar_tag(renumbered)
    apply_stereo_to_rdkit(renumbered, expected.values())
    assert descriptors_from_rdkit(renumbered) == expected


def test_square_planar_raw_permutations_share_one_cyclic_identity() -> None:
    descriptors = []
    element_maps = {"Pt": 1, "C": 2, "F": 3, "Cl": 4, "H": 5}
    for case in SQUARE_PLANAR_CASES[-3:]:
        molecule = Chem.MolFromSmiles(case.smiles)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(element_maps[atom.GetSymbol()])
        descriptors.append(next(iter(descriptors_from_rdkit(molecule).values())))

    assert descriptors[0] == descriptors[1] == descriptors[2]


def test_square_planar_cis_and_trans_fixture_identities_are_distinct() -> None:
    descriptors = []
    for case in SQUARE_PLANAR_CASES[1:3]:
        molecule = Chem.MolFromSmiles(case.smiles)
        assert molecule is not None
        _assign_index_atom_maps(molecule)
        descriptors.append(next(iter(descriptors_from_rdkit(molecule).values())))

    assert descriptors[0] != descriptors[1]


@pytest.mark.parametrize(
    "case",
    DEFERRED_NON_TETRAHEDRAL_CASES,
    ids=lambda case: case.name,
)
def test_non_tetrahedral_rdkit_fixture_is_explicitly_deferred(
    case: NonTetrahedralCase,
) -> None:
    """Account for every portable upstream fixture without silent loss.

    SynKit stores these descriptor classes in graph metadata, but its RDKit
    adapter has not yet promoted TBP or octahedral projection. This test will
    fail when that capability changes, forcing the fixture to move from the
    deferred population into a true round-trip assertion.
    """
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None

    expected_tag = EXPECTED_RDKIT_TAGS[case.stereo_class]
    assert any(atom.GetChiralTag() == expected_tag for atom in molecule.GetAtoms())
    with pytest.raises(NotImplementedError, match=case.stereo_class):
        descriptors_from_rdkit(molecule, require_atom_maps=False)


@pytest.mark.parametrize(
    "case",
    (DEFERRED_NON_TETRAHEDRAL_CASES[0], DEFERRED_NON_TETRAHEDRAL_CASES[-1]),
    ids=lambda case: case.stereo_class,
)
def test_graph_conversion_propagates_unsupported_stereo_loss(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None

    with pytest.raises(NotImplementedError, match=case.stereo_class):
        MolToGraph().transform(molecule)


def test_unsubstituted_ethene_difference_is_recorded() -> None:
    """StereoMolGraph creates rigid-bond parity where RDKit reports no E/Z."""
    molecule = Chem.MolFromSmiles("C=C")
    assert molecule is not None
    molecule = Chem.AddHs(molecule)

    # SynKit follows RDKit's 2D E/Z boundary: identical substituents do not
    # form a planar-bond stereodescriptor.
    assert descriptors_from_rdkit(molecule, require_atom_maps=False) == {}


def test_fixture_population_and_pin_are_stable() -> None:
    assert PINNED_COMMIT == "2189f610f23eaaf992e2e01a12ea4d0532496601"
    assert len(INCHI_ROUND_TRIP_CASES) == 7
    assert len(NON_TETRAHEDRAL_RDKIT_CASES) == 86
