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
    OctahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
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
    OCTAHEDRAL_CASES,
    PINNED_COMMIT,
    SQUARE_PLANAR_CASES,
    TRIGONAL_BIPYRAMIDAL_CASES,
    InchiCase,
    NonTetrahedralCase,
)


def _assign_index_atom_maps(molecule: Chem.Mol) -> None:
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)


def _clear_non_tetrahedral_tag(
    molecule: Chem.Mol,
    tag: Chem.ChiralType,
) -> None:
    center = next(atom for atom in molecule.GetAtoms() if atom.GetChiralTag() == tag)
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

    _clear_non_tetrahedral_tag(molecule, Chem.ChiralType.CHI_SQUAREPLANAR)
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
    assert (
        descriptors_from_rdkit(
            rebuilt,
            require_atom_maps=False,
        )
        == graph.graph["stereo_descriptors"]
    )


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

    _clear_non_tetrahedral_tag(renumbered, Chem.ChiralType.CHI_SQUAREPLANAR)
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
    TRIGONAL_BIPYRAMIDAL_CASES,
    ids=lambda case: case.name,
)
def test_trigonal_bipyramidal_fixture_survives_clearing_and_application(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None
    _assign_index_atom_maps(molecule)

    before = descriptors_from_rdkit(molecule)
    assert len(before) == 1
    assert isinstance(next(iter(before.values())), TrigonalBipyramidalStereo)

    _clear_non_tetrahedral_tag(
        molecule,
        Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    )
    assert descriptors_from_rdkit(molecule) == {}

    apply_stereo_to_rdkit(molecule, before.values())
    assert descriptors_from_rdkit(molecule) == before


@pytest.mark.parametrize(
    "case",
    TRIGONAL_BIPYRAMIDAL_CASES,
    ids=lambda case: case.name,
)
def test_trigonal_bipyramidal_fixture_survives_synkit_graph_round_trip(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None

    graph = MolToGraph().transform(molecule)
    rebuilt = GraphToMol().graph_to_mol(graph)

    assert len(graph.graph["stereo_descriptors"]) == 1
    assert (
        descriptors_from_rdkit(
            rebuilt,
            require_atom_maps=False,
        )
        == graph.graph["stereo_descriptors"]
    )


@pytest.mark.parametrize(
    "case",
    TRIGONAL_BIPYRAMIDAL_CASES,
    ids=lambda case: case.name,
)
def test_trigonal_bipyramidal_fixture_is_atom_renumbering_invariant(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None
    _assign_index_atom_maps(molecule)
    expected = descriptors_from_rdkit(molecule)

    order = list(reversed(range(molecule.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(molecule, order)
    assert descriptors_from_rdkit(renumbered) == expected

    _clear_non_tetrahedral_tag(
        renumbered,
        Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    )
    apply_stereo_to_rdkit(renumbered, expected.values())
    assert descriptors_from_rdkit(renumbered) == expected


def test_all_tbp_raw_permutations_share_one_positional_identity() -> None:
    descriptors = []
    element_maps = {"As": 1, "S": 2, "F": 3, "Cl": 4, "Br": 5, "N": 6}
    for case in TRIGONAL_BIPYRAMIDAL_CASES:
        molecule = Chem.MolFromSmiles(case.smiles)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(element_maps[atom.GetSymbol()])
        descriptors.append(next(iter(descriptors_from_rdkit(molecule).values())))

    assert all(descriptor == descriptors[0] for descriptor in descriptors)


def test_tbp_raw_permutations_on_one_local_order_encode_inverses() -> None:
    element_maps = {"As": 1, "S": 2, "F": 3, "Cl": 4, "Br": 5, "N": 6}
    descriptors = []
    for smiles in (
        "S[As@TB1](F)(Cl)(Br)N",
        "S[As@TB2](F)(Cl)(Br)N",
    ):
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(element_maps[atom.GetSymbol()])
        descriptors.append(next(iter(descriptors_from_rdkit(molecule).values())))

    assert descriptors[0] != descriptors[1]
    assert descriptors[0].invert() == descriptors[1]


@pytest.mark.parametrize(
    "case",
    OCTAHEDRAL_CASES,
    ids=lambda case: case.name,
)
def test_octahedral_fixture_survives_clearing_and_application(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None
    _assign_index_atom_maps(molecule)

    before = descriptors_from_rdkit(molecule)
    assert len(before) == 1
    assert isinstance(next(iter(before.values())), OctahedralStereo)

    _clear_non_tetrahedral_tag(molecule, Chem.ChiralType.CHI_OCTAHEDRAL)
    assert descriptors_from_rdkit(molecule) == {}

    apply_stereo_to_rdkit(molecule, before.values())
    assert descriptors_from_rdkit(molecule) == before


@pytest.mark.parametrize(
    "case",
    OCTAHEDRAL_CASES,
    ids=lambda case: case.name,
)
def test_octahedral_fixture_survives_synkit_graph_round_trip(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None

    graph = MolToGraph().transform(molecule)
    rebuilt = GraphToMol().graph_to_mol(graph)

    assert len(graph.graph["stereo_descriptors"]) == 1
    assert (
        descriptors_from_rdkit(
            rebuilt,
            require_atom_maps=False,
        )
        == graph.graph["stereo_descriptors"]
    )


@pytest.mark.parametrize(
    "case",
    OCTAHEDRAL_CASES,
    ids=lambda case: case.name,
)
def test_octahedral_fixture_is_atom_renumbering_invariant(
    case: NonTetrahedralCase,
) -> None:
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None
    _assign_index_atom_maps(molecule)
    expected = descriptors_from_rdkit(molecule)

    order = list(reversed(range(molecule.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(molecule, order)
    assert descriptors_from_rdkit(renumbered) == expected

    _clear_non_tetrahedral_tag(renumbered, Chem.ChiralType.CHI_OCTAHEDRAL)
    apply_stereo_to_rdkit(renumbered, expected.values())
    assert descriptors_from_rdkit(renumbered) == expected


def test_all_distinct_ligand_oh_permutations_share_one_identity() -> None:
    descriptors = []
    element_maps = {
        "Co": 1,
        "O": 2,
        "P": 3,
        "Cl": 4,
        "C": 5,
        "N": 6,
        "F": 7,
    }
    for case in OCTAHEDRAL_CASES[:30]:
        molecule = Chem.MolFromSmiles(case.smiles)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(element_maps[atom.GetSymbol()])
        descriptors.append(next(iter(descriptors_from_rdkit(molecule).values())))

    assert all(descriptor == descriptors[0] for descriptor in descriptors)


def test_oh_raw_permutations_on_one_local_order_encode_inverses() -> None:
    element_maps = {"Co": 1, "O": 2, "Cl": 3, "C": 4, "N": 5, "F": 6, "P": 7}
    descriptors = []
    for smiles in (
        "O[Co@OH1](Cl)(C)(N)(F)P",
        "O[Co@OH2](Cl)(C)(N)(F)P",
    ):
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(element_maps[atom.GetSymbol()])
        descriptors.append(next(iter(descriptors_from_rdkit(molecule).values())))

    assert descriptors[0] != descriptors[1]
    assert descriptors[0].invert() == descriptors[1]


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
