"""2D real-molecule conformance against pinned StereoMolGraph fixtures.

The fixture values are adapted under MIT from StereoMolGraph commit
2189f610f23eaaf992e2e01a12ea4d0532496601; see
``LICENSES/StereoMolGraph-MIT.txt``. No StereoMolGraph runtime dependency is
required for this always-on SynKit suite.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from synkit.Graph.Stereo import descriptors_from_rdkit
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph

from Test.Graph.Stereo.stereomolgraph_2d_fixtures import (
    INCHI_ROUND_TRIP_CASES,
    MAPPED_CONNECTIVITY_ATOMS,
    MAPPED_CONNECTIVITY_SMILES,
    NON_TETRAHEDRAL_RDKIT_CASES,
    PINNED_COMMIT,
    InchiCase,
    NonTetrahedralCase,
)

EXPECTED_RDKIT_TAGS = {
    "square_planar": Chem.ChiralType.CHI_SQUAREPLANAR,
    "trigonal_bipyramidal": Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    "octahedral": Chem.ChiralType.CHI_OCTAHEDRAL,
}


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
    NON_TETRAHEDRAL_RDKIT_CASES,
    ids=lambda case: case.name,
)
def test_non_tetrahedral_rdkit_fixture_is_explicitly_deferred(
    case: NonTetrahedralCase,
) -> None:
    """Account for every portable upstream fixture without silent loss.

    SynKit stores these descriptor classes in graph metadata, but its RDKit
    adapter currently supports only tetrahedral and planar-bond projection.
    This test will fail when that capability changes, forcing the fixture to
    move from the deferred population into a true round-trip assertion.
    """
    molecule = Chem.MolFromSmiles(case.smiles)
    assert molecule is not None

    expected_tag = EXPECTED_RDKIT_TAGS[case.stereo_class]
    assert any(atom.GetChiralTag() == expected_tag for atom in molecule.GetAtoms())
    assert descriptors_from_rdkit(molecule, require_atom_maps=False) == {}


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
