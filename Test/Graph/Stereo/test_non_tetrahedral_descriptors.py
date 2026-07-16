"""SynKit-native tests for the extended StereoMolGraph shared subset.

The permutation examples are adapted from StereoMolGraph commit
2189f610f23eaaf992e2e01a12ea4d0532496601 (MIT; copyright (c) 2025
Maxim Papusha). SynKit independently tests graph metadata, explicit query
policy, reaction changes, and serialization around those descriptor values.
"""

from itertools import permutations

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Graph.Stereo import (
    AtropBondStereo,
    DEFERRED_STEREO_DESCRIPTOR_CLASSES,
    OctahedralStereo,
    RDKIT_STEREO_DESCRIPTOR_CLASSES,
    SUPPORTED_STEREO_DESCRIPTOR_CLASSES,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoOutcome,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    apply_stereo_to_rdkit,
    candidate_mapping_stereo_matches,
    classify_stereo_change,
    descriptor_id,
    descriptor_query_matches,
    descriptors_from_rdkit,
    stereo_from_dict,
    stereo_isomorphic,
    stereo_isomorphism_mapping,
)
from synkit.Mechanism import (
    StereoDescriptor,
    stereo_graph_from_gml,
    stereo_graph_to_gml,
)
from synkit.IO.mol_to_graph import MolToGraph


@pytest.mark.parametrize(
    ("reference", "equivalent", "distinct"),
    [
        (
            SquarePlanarStereo((6, 0, 1, 2, 3), 0),
            SquarePlanarStereo((6, 1, 2, 3, 0), 0),
            SquarePlanarStereo((6, 1, 0, 2, 3), 0),
        ),
        (
            TrigonalBipyramidalStereo((6, 0, 1, 2, 3, 4), 1),
            TrigonalBipyramidalStereo((6, 1, 0, 3, 4, 2), -1),
            TrigonalBipyramidalStereo((6, 1, 0, 3, 4, 2), 1),
        ),
        (
            OctahedralStereo((6, 0, 1, 2, 3, 4, 5), 1),
            OctahedralStereo((6, 1, 0, 2, 3, 4, 5), -1),
            OctahedralStereo((6, 1, 0, 2, 3, 4, 5), 1),
        ),
        (
            AtropBondStereo((0, 1, 2, 3, 4, 5), 1),
            AtropBondStereo((1, 0, 2, 3, 4, 5), -1),
            AtropBondStereo((1, 0, 2, 3, 4, 5), 1),
        ),
    ],
)
def test_non_tetrahedral_relative_identity(reference, equivalent, distinct):
    assert reference == equivalent
    assert hash(reference) == hash(equivalent)
    assert reference != distinct
    if reference.parity == 0:
        assert reference.invert() == reference
    else:
        assert reference.invert() != reference


@pytest.mark.parametrize(
    "descriptor",
    [
        SquarePlanarStereo((1, 2, 3, 4, 5), 0, "manual"),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), -1, "manual"),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1, "manual"),
        AtropBondStereo((1, 2, 3, 4, 5, 6), -1, "manual"),
    ],
)
def test_non_tetrahedral_relabel_and_value_round_trip(descriptor):
    mapping = {value: value + 10 for value in descriptor.dependencies}
    relabeled = descriptor.relabel(mapping)

    assert relabeled.dependencies == frozenset(mapping.values())
    assert relabeled.parity == descriptor.parity
    assert stereo_from_dict(descriptor.to_dict()) == descriptor
    assert descriptor_id(relabeled).startswith(
        "bond:" if isinstance(relabeled, AtropBondStereo) else "atom:"
    )


@pytest.mark.parametrize(
    "specified",
    [
        SquarePlanarStereo((1, 2, 3, 4, 5), 0),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), -1),
        AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
    ],
)
def test_non_tetrahedral_unknown_is_exact_or_explicit_wildcard(specified):
    unknown = type(specified)(specified.atoms, None, "query")

    assert not descriptor_query_matches(unknown, specified, unknown_policy="exact")
    assert descriptor_query_matches(unknown, specified, unknown_policy="wildcard")


@pytest.mark.parametrize(
    "descriptor",
    [
        SquarePlanarStereo((1, 2, 3, 4, 5), 0),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), -1),
        AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
    ],
)
def test_non_tetrahedral_descriptors_guard_candidate_mappings(descriptor):
    pattern = nx.Graph()
    host = nx.Graph()
    translation = {value: value + 20 for value in descriptor.dependencies}
    for atom_map in descriptor.dependencies:
        pattern.add_node(atom_map, atom_map=atom_map, element="*")
        host.add_node(atom_map + 20, atom_map=atom_map + 20, element="*")
    key = descriptor_id(descriptor)
    pattern.graph["stereo_descriptors"] = {key: descriptor}
    mapped = descriptor.relabel(translation)
    host.graph["stereo_descriptors"] = {descriptor_id(mapped): mapped}

    assert candidate_mapping_stereo_matches(
        pattern,
        host,
        translation,
        mode="strict",
    )
    if descriptor.parity in (-1, 1):
        host.graph["stereo_descriptors"] = {descriptor_id(mapped): mapped.invert()}
        assert not candidate_mapping_stereo_matches(
            pattern,
            host,
            translation,
            mode="require",
        )


def test_square_planar_transition_records_tetrahedral_inversion():
    before = TetrahedralStereo((1, 2, 3, 4, 5), -1)
    transition = SquarePlanarStereo((1, 5, 3, 4, 2), 0, "transition")
    after = before.invert()

    assert classify_stereo_change(before, after, transition) == "INVERTED"
    assert classify_stereo_change(None, None, transition) == "FLEETING"


@pytest.mark.parametrize(
    "before",
    [
        SquarePlanarStereo((1, 2, 3, 4, 5), 0),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), -1),
        PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
        AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
    ],
)
def test_ligand_replacement_aligns_retention_and_inversion(before):
    atoms = tuple(99 if value == before.atoms[-1] else value for value in before.atoms)
    retained = type(before)(atoms, before.parity)
    inverted = retained.invert()

    assert classify_stereo_change(before, retained) == "RETAINED"
    if before.parity in (-1, 1) or isinstance(before, PlanarBondStereo):
        assert classify_stereo_change(before, inverted) == "INVERTED"


def test_racemic_outcome_supports_any_specified_chiral_descriptor():
    seed = AtropBondStereo((1, 2, 3, 4, 5, 6), 1)
    alternatives = StereoOutcome.racemic().alternatives(seed)

    assert alternatives == (seed, seed.invert())
    with pytest.raises(ValueError, match="specified chiral product"):
        StereoOutcome.racemic().alternatives(SquarePlanarStereo((1, 2, 3, 4, 5), 0))


def test_all_supported_descriptors_survive_gml_registry_round_trip():
    graph = nx.Graph()
    graph.add_nodes_from(
        (atom_map, {"atom_map": atom_map, "element": "*"}) for atom_map in range(1, 8)
    )
    descriptors = [
        TetrahedralStereo((1, 2, 3, 4, 5), 1),
        SquarePlanarStereo((1, 2, 3, 4, 5), 0),
        TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), -1),
        OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
        PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
        AtropBondStereo((1, 2, 3, 4, 5, 6), -1),
    ]
    graph.graph["stereo_descriptors"] = {
        f"case:{index}": descriptor for index, descriptor in enumerate(descriptors)
    }

    text, report = stereo_graph_to_gml(graph)
    restored = stereo_graph_from_gml(text)

    assert report.lossless
    assert restored.graph["stereo_descriptors"] == graph.graph["stereo_descriptors"]


def test_mechanism_envelope_accepts_extended_descriptor_classes():
    values = [
        StereoDescriptor("square_planar", (1, 2, 3, 4, 5), 0),
        StereoDescriptor("trigonal_bipyramidal", (1, 2, 3, 4, 5, 6), 1),
        StereoDescriptor("octahedral", (1, 2, 3, 4, 5, 6, 7), -1),
        StereoDescriptor("atrop_bond", (1, 2, 3, 4, 5, 6), 1),
    ]

    assert [StereoDescriptor.from_dict(value.to_dict()) for value in values] == values


def test_rdkit_boundary_rejects_unimplemented_non_tetrahedral_projection():
    mol = Chem.MolFromSmiles("[CH4:1]")
    descriptor = AtropBondStereo((1, 2, 3, 4, 5, 6), 1)

    with pytest.raises(NotImplementedError, match="atrop_bond"):
        apply_stereo_to_rdkit(mol, [descriptor])


def test_rdkit_boundary_rejects_unknown_square_planar_projection():
    mol = Chem.MolFromSmiles(
        "[CH3:1][Pt:2]([F:3])([Cl:4])[Br:5]",
    )
    descriptor = SquarePlanarStereo((2, 1, 3, 4, 5), None)

    with pytest.raises(NotImplementedError, match="unknown square-planar"):
        apply_stereo_to_rdkit(mol, [descriptor])


def test_rdkit_boundary_rejects_mismatched_square_planar_ligands():
    mol = Chem.MolFromSmiles(
        "[CH3:1][Pt:2]([F:3])([Cl:4])[Br:5]",
    )
    descriptor = SquarePlanarStereo((2, 1, 3, 4, 99), 0)

    with pytest.raises(ValueError, match="ligands do not match"):
        apply_stereo_to_rdkit(mol, [descriptor])


@pytest.mark.parametrize("local_order", permutations((2, 3, 4, 5)))
def test_square_planar_projection_is_invariant_to_every_local_order(local_order):
    editable = Chem.RWMol()
    center = Chem.Atom("Pt")
    center.SetAtomMapNum(1)
    center_idx = editable.AddAtom(center)
    elements = {2: "H", 3: "F", 4: "Cl", 5: "Br"}
    for atom_map in local_order:
        ligand = Chem.Atom(elements[atom_map])
        ligand.SetAtomMapNum(atom_map)
        ligand_idx = editable.AddAtom(ligand)
        editable.AddBond(center_idx, ligand_idx, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    mol.UpdatePropertyCache(strict=False)
    descriptor = SquarePlanarStereo((1, 2, 3, 4, 5), 0)

    apply_stereo_to_rdkit(mol, [descriptor])

    assert descriptors_from_rdkit(mol) == {"atom:1": descriptor}


def test_square_planar_hidden_hydrogen_survives_clearing_and_application():
    mol = Chem.MolFromSmiles("[CH3:1][Pt@SP1H:2]([Cl:3])[F:4]")
    expected = descriptors_from_rdkit(mol)
    descriptor = next(iter(expected.values()))
    center = mol.GetAtomWithIdx(1)

    assert "@H:2" in descriptor.atoms
    center.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    center.ClearProp("_chiralPermutation")
    apply_stereo_to_rdkit(mol, expected.values())

    assert descriptors_from_rdkit(mol) == expected


def test_rdkit_boundary_rejects_unknown_tbp_projection():
    mol = Chem.MolFromSmiles(
        "[S:2][As:1]([F:3])([Cl:4])([Br:5])[N:6]",
    )
    descriptor = TrigonalBipyramidalStereo((1, 2, 6, 3, 4, 5), None)

    with pytest.raises(NotImplementedError, match="unknown trigonal-bipyramidal"):
        apply_stereo_to_rdkit(mol, [descriptor])


def test_rdkit_boundary_rejects_mismatched_tbp_ligands():
    mol = Chem.MolFromSmiles(
        "[S:2][As:1]([F:3])([Cl:4])([Br:5])[N:6]",
    )
    descriptor = TrigonalBipyramidalStereo((1, 2, 6, 3, 4, 99), 1)

    with pytest.raises(ValueError, match="ligands do not match"):
        apply_stereo_to_rdkit(mol, [descriptor])


@pytest.mark.parametrize("local_order", permutations((2, 3, 4, 5, 6)))
def test_tbp_projection_is_invariant_to_every_local_order(local_order):
    editable = Chem.RWMol()
    center = Chem.Atom("As")
    center.SetAtomMapNum(1)
    center_idx = editable.AddAtom(center)
    elements = {2: "S", 3: "F", 4: "Cl", 5: "Br", 6: "N"}
    for atom_map in local_order:
        ligand = Chem.Atom(elements[atom_map])
        ligand.SetAtomMapNum(atom_map)
        ligand_idx = editable.AddAtom(ligand)
        editable.AddBond(center_idx, ligand_idx, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    mol.UpdatePropertyCache(strict=False)
    descriptor = TrigonalBipyramidalStereo((1, 2, 6, 3, 4, 5), 1)

    apply_stereo_to_rdkit(mol, [descriptor])

    assert descriptors_from_rdkit(mol) == {"atom:1": descriptor}


def test_tbp_hidden_hydrogen_survives_clearing_and_application():
    mol = Chem.MolFromSmiles("[S:1][As@TB1H:2]([F:3])([Cl:4])[Br:5]")
    expected = descriptors_from_rdkit(mol)
    descriptor = next(iter(expected.values()))
    center = mol.GetAtomWithIdx(1)

    assert "@H:2" in descriptor.atoms
    center.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    center.ClearProp("_chiralPermutation")
    apply_stereo_to_rdkit(mol, expected.values())

    assert descriptors_from_rdkit(mol) == expected


def test_rdkit_boundary_rejects_unknown_octahedral_projection():
    mol = Chem.MolFromSmiles(
        "[O:2][Co:1]([Cl:3])([C:4])([N:5])([F:6])[P:7]",
    )
    descriptor = OctahedralStereo((1, 2, 7, 3, 4, 5, 6), None)

    with pytest.raises(NotImplementedError, match="unknown octahedral"):
        apply_stereo_to_rdkit(mol, [descriptor])


def test_rdkit_boundary_rejects_mismatched_octahedral_ligands():
    mol = Chem.MolFromSmiles(
        "[O:2][Co:1]([Cl:3])([C:4])([N:5])([F:6])[P:7]",
    )
    descriptor = OctahedralStereo((1, 2, 7, 3, 4, 5, 99), 1)

    with pytest.raises(ValueError, match="ligands do not match"):
        apply_stereo_to_rdkit(mol, [descriptor])


@pytest.mark.parametrize("local_order", permutations((2, 3, 4, 5, 6, 7)))
def test_octahedral_projection_is_invariant_to_every_local_order(local_order):
    editable = Chem.RWMol()
    center = Chem.Atom("Co")
    center.SetAtomMapNum(1)
    center_idx = editable.AddAtom(center)
    elements = {2: "O", 3: "Cl", 4: "C", 5: "N", 6: "F", 7: "P"}
    for atom_map in local_order:
        ligand = Chem.Atom(elements[atom_map])
        ligand.SetAtomMapNum(atom_map)
        ligand_idx = editable.AddAtom(ligand)
        editable.AddBond(center_idx, ligand_idx, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    mol.UpdatePropertyCache(strict=False)
    descriptor = OctahedralStereo((1, 2, 7, 3, 4, 5, 6), 1)

    apply_stereo_to_rdkit(mol, [descriptor])

    assert descriptors_from_rdkit(mol) == {"atom:1": descriptor}


def test_octahedral_hidden_hydrogen_survives_clearing_and_application():
    mol = Chem.MolFromSmiles(
        "[O:1][Co@OH1H:2]([Cl:3])([C:4])([N:5])[F:6]",
    )
    expected = descriptors_from_rdkit(mol)
    descriptor = next(iter(expected.values()))
    center = mol.GetAtomWithIdx(1)

    assert "@H:2" in descriptor.atoms
    center.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    center.ClearProp("_chiralPermutation")
    apply_stereo_to_rdkit(mol, expected.values())

    assert descriptors_from_rdkit(mol) == expected


def test_octahedral_missing_site_fails_instead_of_becoming_hydrogen():
    mol = Chem.MolFromSmiles("O[Mn@OH28](Cl)(C)(N)F")

    with pytest.raises(ValueError, match="missing coordination sites"):
        descriptors_from_rdkit(mol, require_atom_maps=False)
    with pytest.raises(ValueError, match="missing coordination sites"):
        MolToGraph().transform(mol)


def test_extended_capability_boundary_is_explicit():
    assert SUPPORTED_STEREO_DESCRIPTOR_CLASSES == {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
        "planar_bond",
        "atrop_bond",
    }
    assert RDKIT_STEREO_DESCRIPTOR_CLASSES == {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
        "planar_bond",
    }
    assert "rigid_bond_33" in DEFERRED_STEREO_DESCRIPTOR_CLASSES


def test_graph_stereo_isomorphism_distinguishes_tetrahedral_enantiomers():
    left = MolToGraph().transform(
        Chem.MolFromSmiles("[F:1][C@:2]([Cl:3])([Br:4])[I:5]")
    )
    relabeled = MolToGraph().transform(
        Chem.MolFromSmiles("[F:11][C@:12]([Cl:13])([Br:14])[I:15]")
    )
    inverse = MolToGraph().transform(
        Chem.MolFromSmiles("[F:11][C@@:12]([Cl:13])([Br:14])[I:15]")
    )

    mapping = stereo_isomorphism_mapping(left, relabeled)

    assert mapping is not None
    assert stereo_isomorphic(left, relabeled)
    assert not stereo_isomorphic(left, inverse)


def test_graph_stereo_isomorphism_handles_atrop_bonds_and_unknown_policy():
    def graph(descriptor):
        value = nx.Graph()
        elements = {1: "F", 2: "Cl", 3: "C", 4: "C", 5: "Br", 6: "I"}
        value.add_nodes_from(
            (atom_map, {"atom_map": atom_map, "element": element})
            for atom_map, element in elements.items()
        )
        value.add_edges_from(((1, 3), (2, 3), (3, 4), (4, 5), (4, 6)))
        value.graph["stereo_descriptors"] = {"bond:3-4": descriptor}
        return value

    specified = AtropBondStereo((1, 2, 3, 4, 5, 6), 1)
    unknown = AtropBondStereo(specified.atoms, None)

    assert stereo_isomorphic(graph(specified), graph(specified))
    assert not stereo_isomorphic(graph(specified), graph(specified.invert()))
    assert not stereo_isomorphic(graph(unknown), graph(specified))
    assert stereo_isomorphic(
        graph(unknown),
        graph(specified),
        unknown_policy="wildcard",
    )
