import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    DEFERRED_STEREO_DESCRIPTOR_CLASSES,
    NonInvertibleStereoEffectError,
    PlanarBondStereo,
    SquarePlanarStereo,
    SUPPORTED_STEREO_DESCRIPTOR_CLASSES,
    StereoChange,
    TetrahedralStereo,
    apply_stereo_to_rdkit,
    descriptors_from_rdkit,
    candidate_mapping_stereo_matches,
    compare_stereo_registries,
    descriptor_query_matches,
    stereo_complete_reaction_center_nodes,
)
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.rc_extractor import RCExtractor
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph


def test_tetrahedral_equality_tracks_permutation_parity():
    first = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    even = TetrahedralStereo((1, 3, 4, 2, 5), 1)
    odd_with_opposite_parity = TetrahedralStereo((1, 3, 2, 4, 5), -1)
    opposite = TetrahedralStereo((1, 3, 2, 4, 5), 1)

    assert first == even == odd_with_opposite_parity
    assert first != opposite
    assert first.invert() == opposite
    assert hash(first) == hash(even)


def test_tetrahedral_relabeling_preserves_identity():
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), -1)
    relabeled = descriptor.relabel({1: 10, 2: 20, 3: 30, 4: 40, 5: 50})

    assert relabeled.atoms == (10, 20, 30, 40, 50)
    assert relabeled.parity == -1


def test_planar_bond_allowed_end_permutations_and_inversion():
    descriptor = PlanarBondStereo((1, 2, 3, 4, 5, 6))
    equivalent = PlanarBondStereo((2, 1, 3, 4, 6, 5))

    assert descriptor == equivalent
    assert descriptor != descriptor.invert()


def test_rdkit_tetrahedral_round_trip_uses_relative_descriptor():
    mol = Chem.MolFromSmiles("[CH3:1][C@H:2]([OH:3])[F:4]")
    descriptors = descriptors_from_rdkit(mol)
    expected = next(iter(descriptors.values()))
    Chem.RemoveStereochemistry(mol)

    apply_stereo_to_rdkit(mol, descriptors.values())
    observed = next(iter(descriptors_from_rdkit(mol).values()))

    assert observed == expected


def test_rdkit_ez_descriptors_are_distinct_and_round_trip():
    e_mol = Chem.MolFromSmiles("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]")
    z_mol = Chem.MolFromSmiles("[CH3:1]/[CH:2]=[CH:3]\\[CH3:4]")
    e_descriptor = next(iter(descriptors_from_rdkit(e_mol).values()))
    z_descriptor = next(iter(descriptors_from_rdkit(z_mol).values()))

    assert isinstance(e_descriptor, PlanarBondStereo)
    assert isinstance(z_descriptor, PlanarBondStereo)
    assert e_descriptor != z_descriptor

    Chem.RemoveStereochemistry(e_mol)
    apply_stereo_to_rdkit(e_mol, [e_descriptor])
    assert next(iter(descriptors_from_rdkit(e_mol).values())) == e_descriptor


def test_molecular_graph_registry_round_trip_preserves_tetrahedral_stereo():
    mol = Chem.MolFromSmiles("[CH3:1][C@H:2]([OH:3])[F:4]")
    graph = MolToGraph().transform(mol)

    rebuilt = GraphToMol().graph_to_mol(graph)
    rebuilt_registry = descriptors_from_rdkit(rebuilt)

    assert graph.graph["stereo_descriptors"] == rebuilt_registry


def test_unmapped_molecular_graph_round_trip_preserves_supported_stereo():
    for smiles in ("CC[C@H](O)C", "C/C=C\\C"):
        mol = Chem.MolFromSmiles(smiles)
        expected = descriptors_from_rdkit(mol, require_atom_maps=False)
        graph = MolToGraph().transform(mol)

        rebuilt = GraphToMol().graph_to_mol(graph)
        observed = descriptors_from_rdkit(rebuilt, require_atom_maps=False)

        assert observed == expected
        assert all(atom.GetAtomMapNum() == 0 for atom in rebuilt.GetAtoms())


def test_symmetric_stereo_identity_does_not_depend_on_canonical_smiles_text():
    molecule = Chem.MolFromSmiles("C[C@H]1CC[C@@H]([C@H]2CC[C@@H](C)CC2)CC1")
    expected = descriptors_from_rdkit(molecule, require_atom_maps=False)
    rebuilt = GraphToMol().graph_to_mol(MolToGraph().transform(molecule))

    assert descriptors_from_rdkit(rebuilt, require_atom_maps=False) == expected

    expected_constitution = Chem.Mol(molecule)
    observed_constitution = Chem.Mol(rebuilt)
    Chem.RemoveStereochemistry(expected_constitution)
    Chem.RemoveStereochemistry(observed_constitution)
    assert Chem.MolToSmiles(
        observed_constitution,
        canonical=True,
        isomericSmiles=True,
    ) == Chem.MolToSmiles(
        expected_constitution,
        canonical=True,
        isomericSmiles=True,
    )


def test_stereo_only_its_change_expands_reaction_center_dependencies():
    before = MolToGraph().transform(Chem.MolFromSmiles("[CH3:1][C@H:2]([OH:3])[F:4]"))
    after = MolToGraph().transform(Chem.MolFromSmiles("[CH3:1][C@@H:2]([OH:3])[F:4]"))

    its = ITSConstruction.construct(before, after)
    changes = compare_stereo_registries(before, after)
    nodes = stereo_complete_reaction_center_nodes(its)
    rc = RCExtractor().extract(its)

    assert {change.change for change in changes.values()} == {"INVERTED"}
    assert nodes == set(before.nodes)
    assert set(rc.nodes) == set(before.nodes)


def test_candidate_mapping_rejects_opposite_enantiomer_in_strict_mode():
    pattern = MolToGraph().transform(Chem.MolFromSmiles("[CH3:1][C@H:2]([OH:3])[F:4]"))
    same = MolToGraph().transform(Chem.MolFromSmiles("[CH3:10][C@H:20]([OH:30])[F:40]"))
    opposite = MolToGraph().transform(
        Chem.MolFromSmiles("[CH3:10][C@@H:20]([OH:30])[F:40]")
    )
    mapping = {1: 1, 2: 2, 3: 3, 4: 4}

    assert candidate_mapping_stereo_matches(pattern, same, mapping, mode="strict")
    assert not candidate_mapping_stereo_matches(
        pattern, opposite, mapping, mode="strict"
    )
    assert not candidate_mapping_stereo_matches(
        pattern, opposite, mapping, mode="require"
    )
    assert candidate_mapping_stereo_matches(pattern, opposite, mapping, mode="ignore")
    assert candidate_mapping_stereo_matches(
        pattern, opposite, mapping, mode="propagate"
    )


def test_unknown_parity_is_exact_unless_query_policy_is_wildcard():
    unknown = TetrahedralStereo((2, 1, 3, 4, "@H:2"), None)
    specified = TetrahedralStereo((2, 1, 3, 4, "@H:2"), 1)

    assert descriptor_query_matches(unknown, unknown, unknown_policy="exact")
    assert not descriptor_query_matches(unknown, specified, unknown_policy="exact")
    assert descriptor_query_matches(unknown, specified, unknown_policy="wildcard")
    assert not descriptor_query_matches(
        specified, specified.invert(), unknown_policy="wildcard"
    )
    assert descriptor_query_matches(
        specified, specified.invert(), unknown_policy="either"
    )


def test_virtual_h_query_matches_bound_explicit_h_projection():
    pattern = nx.Graph()
    host = nx.Graph()
    for atom_map, element in ((1, "C"), (2, "C"), (3, "O"), (4, "F")):
        pattern.add_node(atom_map, atom_map=atom_map, element=element)
    for node, atom_map, element in (
        (10, 10, "C"),
        (20, 20, "C"),
        (30, 30, "O"),
        (40, 40, "F"),
        (50, 50, "H"),
    ):
        host.add_node(node, atom_map=atom_map, element=element)
    host.add_edge(20, 50, order=1.0)
    pattern.graph["stereo_descriptors"] = {
        "atom:2": TetrahedralStereo((2, 1, 3, 4, "@H:2"), 1)
    }
    host.graph["stereo_descriptors"] = {
        "atom:20": TetrahedralStereo((20, 10, 30, 40, 50), 1)
    }

    assert candidate_mapping_stereo_matches(
        pattern,
        host,
        {1: 10, 2: 20, 3: 30, 4: 40},
        mode="require",
    )


def test_fleeting_transition_stereo_is_first_class_and_reversible():
    before = nx.Graph()
    after = nx.Graph()
    transition = nx.Graph()
    descriptor = TetrahedralStereo((2, 1, 3, 4, 5), 1, "its-transition")
    transition.graph["stereo_descriptors"] = {"atom:2": descriptor}

    change = compare_stereo_registries(before, after, transition)["atom:2"]

    assert change == StereoChange("FLEETING", None, None, descriptor)
    assert change.fleeting == descriptor
    assert change.reverse() == change
    assert change.dependencies == frozenset({1, 2, 3, 4, 5})


def test_unspecified_stereo_change_has_no_unique_reverse():
    specified = TetrahedralStereo((2, 1, 3, 4, 5), 1)
    unknown = TetrahedralStereo((2, 1, 3, 4, 5), None)
    change = StereoChange("UNSPECIFIED", specified, unknown)

    with pytest.raises(NonInvertibleStereoEffectError) as excinfo:
        change.reverse()

    assert excinfo.value.reason == "non_reversible_unspecified_descriptor"
    assert excinfo.value.targets == ()


def test_specified_geometric_stereomutation_remains_reversible():
    before = SquarePlanarStereo((1, 2, 3, 4, 5), 0)
    after = SquarePlanarStereo((1, 2, 4, 3, 5), 0)
    change = StereoChange("UNSPECIFIED", before, after)

    assert change.non_invertible is False
    assert change.reverse() == StereoChange("UNSPECIFIED", after, before)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            StereoChange(
                "RETAINED",
                TetrahedralStereo((2, 1, 3, 4, 5), 1),
                TetrahedralStereo((2, 1, 3, 4, 5), 1),
            ),
            "RETAINED",
        ),
        (
            StereoChange(
                "INVERTED",
                TetrahedralStereo((2, 1, 3, 4, 5), 1),
                TetrahedralStereo((2, 1, 3, 4, 5), -1),
            ),
            "INVERTED",
        ),
        (
            StereoChange("FORMED", None, TetrahedralStereo((2, 1, 3, 4, 5), 1)),
            "BROKEN",
        ),
        (
            StereoChange("BROKEN", TetrahedralStereo((2, 1, 3, 4, 5), 1), None),
            "FORMED",
        ),
    ],
)
def test_invertible_stereo_changes_still_reverse(change, expected):
    assert change.reverse().change == expected


def test_descriptor_capability_boundary_is_explicit():
    assert {"tetrahedral", "planar_bond"} <= SUPPORTED_STEREO_DESCRIPTOR_CLASSES
    assert "rigid_bond_33" in DEFERRED_STEREO_DESCRIPTOR_CLASSES
