"""Check native Lewis minima and class coverage against tiny independent oracles."""

from itertools import permutations, product

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Chem.Mapper import Mapper, map_reaction


def _endpoint_molecules(reaction):
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    return tuple(
        Chem.AddHs(Chem.MolFromSmiles(side, parser))
        for side in reaction.split(">>")
    )


def _bond(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return (
        (float(bond.GetBondTypeAsDouble()), str(bond.GetStereo()))
        if bond is not None
        else (0.0, "")
    )


def _nonbonding_electrons(atom):
    # These oracle examples use ordinary integer bond orders and explicit H.
    # Electron bookkeeping is independent of the mapper's lone-pair estimate
    # and its auxiliary anchor representation.
    electrons = (
        Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
        - atom.GetFormalCharge()
        - sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
    )
    assert electrons >= atom.GetNumRadicalElectrons()
    assert (electrons - atom.GetNumRadicalElectrons()) % 2 == 0
    return electrons


def _oracle(reaction):
    mols = _endpoint_molecules(reaction)
    groups = []
    for mol in mols:
        endpoint = {}
        for atom in mol.GetAtoms():
            identity = atom.GetAtomicNum(), atom.GetIsotope()
            endpoint.setdefault(identity, []).append(atom.GetIdx())
        groups.append(endpoint)
    assert {key: len(value) for key, value in groups[0].items()} == {
        key: len(value) for key, value in groups[1].items()
    }
    identities = sorted(groups[0])
    records = []
    electrons = [
        [_nonbonding_electrons(atom) for atom in mol.GetAtoms()] for mol in mols
    ]
    for images in product(*(permutations(groups[1][key]) for key in identities)):
        mapping = [None] * mols[0].GetNumAtoms()
        for key, targets in zip(identities, images):
            for source, target in zip(groups[0][key], targets):
                mapping[source] = target
        bonds = sum(
            abs(_bond(mols[0], i, j)[0] - _bond(mols[1], mapping[i], mapping[j])[0])
            for i in range(len(mapping))
            for j in range(i + 1, len(mapping))
        )
        unary = sum(
            abs(electrons[0][i] - electrons[1][j]) / 2
            for i, j in enumerate(mapping)
        )
        records.append((tuple(mapping), bonds + unary))
    return mols, records


def _class_graph(mols, mapping):
    graph = nx.Graph()
    for i, j in enumerate(mapping):
        atoms = mols[0].GetAtomWithIdx(i), mols[1].GetAtomWithIdx(j)
        graph.add_node(
            i,
            color=(
                atoms[0].GetAtomicNum(),
                atoms[0].GetIsotope(),
                tuple(atom.GetFormalCharge() for atom in atoms),
                tuple(atom.GetNumRadicalElectrons() for atom in atoms),
                tuple(atom.GetTotalNumHs() for atom in atoms),
                tuple(
                    atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
                    for atom in atoms
                ),
            ),
        )
    for i in range(len(mapping)):
        for j in range(i + 1, len(mapping)):
            reactant = _bond(mols[0], i, j)
            target = _bond(mols[1], mapping[i], mapping[j])
            if reactant[0] or target[0]:
                graph.add_edge(i, j, color=(reactant, target))
    return graph


def _equivalent(left, right):
    return nx.is_isomorphic(
        left,
        right,
        node_match=nx.algorithms.isomorphism.categorical_node_match("color", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("color", None),
    )


def _assert_all_classes(result, mols, expected):
    size = mols[0].GetNumAtoms()
    actual = [
        _class_graph(mols, item.mapping[:size]) for item in result.reactions
    ]
    assert all(
        not _equivalent(left, right)
        for index, left in enumerate(actual)
        for right in actual[index + 1 :]
    )
    expected_graphs = [_class_graph(mols, mapping) for mapping, _ in expected]
    assert all(
        any(_equivalent(graph, candidate) for candidate in actual)
        for graph in expected_graphs
    )
    assert all(
        any(_equivalent(graph, candidate) for candidate in expected_graphs)
        for graph in actual
    )
    costs = dict(expected)
    assert all(
        costs[item.mapping[:size]] == item.distance for item in result.reactions
    )


@pytest.mark.parametrize(
    "reaction,minimum,classes",
    [
        ("N.Cl>>[NH4+].[Cl-]", 4, 1),
        ("[OH].[H]>>O", 2, 1),
        ("O=CC=O>>OC#CO", 8, 2),
    ],
)
def test_native_default_lewis_minimum_and_all_minimum_classes(
    native_library, reaction, minimum, classes
):
    mols, records = _oracle(reaction)
    expected_minimum = min(cost for _, cost in records)
    assert expected_minimum == minimum
    result = map_reaction(
        reaction,
        backend="cpp",
        library_path=native_library,
        max_mappings=None,
    )
    assert result.complete and result.search_complete and result.classification_complete
    assert result.minimum_cost == expected_minimum
    assert result.hydrogen == "explicit"
    assert result.objective == "lewis_bond_plus_nonbonding_electron_distance"
    assert len(result.reactions) == classes
    _assert_all_classes(
        result, mols, [(mapping, cost) for mapping, cost in records if cost == minimum]
    )


@pytest.mark.parametrize("limit", [7, 8, 8.25, 9, 12])
def test_native_at_most_distance_covers_all_and_only_budget_classes(
    native_library, limit
):
    reaction = "O=CC=O>>OC#CO"
    mols, records = _oracle(reaction)
    result = Mapper(
        "native",
        library_path=native_library,
        CD=limit,
        CD_mode="at_most",
        max_mappings=None,
    ).map(reaction)
    assert result.complete and result.minimum_cost is None
    expected = [(mapping, cost) for mapping, cost in records if cost <= limit]
    _assert_all_classes(result.reaction_result, mols, expected)
    assert all(distance <= limit for distance in result.distances)
    assert len(result.mapped_reactions) == (0 if limit < 8 else 4 if limit >= 12 else 2)


def test_native_numeric_distance_defaults_to_exact_shell(native_library):
    reaction = "O=CC=O>>OC#CO"
    mols, records = _oracle(reaction)
    result = map_reaction(
        reaction,
        backend="cpp",
        library_path=native_library,
        CD=12,
        max_mappings=None,
    )
    assert result.complete and result.minimum_cost is None
    assert len(result.reactions) == 2
    assert {item.distance for item in result.reactions} == {12}
    _assert_all_classes(
        result, mols, [(mapping, cost) for mapping, cost in records if cost == 12]
    )


def test_native_symmetry_keeps_e_and_z_endpoint_classes_distinct(native_library):
    # Constitutional CD is zero for both matching and swapping these isomers.
    # Their endpoint stereo annotations still distinguish two reaction classes.
    endpoint = "F/N=N/F.F/N=N\\F"
    reaction = endpoint + ">>" + endpoint
    mols, records = _oracle(reaction)
    result = map_reaction(
        reaction, backend="cpp", library_path=native_library, max_mappings=None
    )
    assert result.complete and result.minimum_cost == 0
    assert len(result.reactions) == 2
    _assert_all_classes(
        result, mols, [(mapping, cost) for mapping, cost in records if cost == 0]
    )


@pytest.mark.parametrize("entrypoint", ["mapper", "reaction"])
def test_default_lewis_classes_are_not_rejected_by_labeled_factorial_size(
    native_library, entrypoint
):
    # Ten explicit H atoms give 4! * 10! = 87,091,200 labeled bijections,
    # although the zero-distance reaction has a single symmetry class.
    if entrypoint == "mapper":
        run = Mapper("native", library_path=native_library).map
    else:
        def run(reaction, **options):
            return map_reaction(
                reaction, backend="cpp", library_path=native_library, **options
            )

    result = run("CCCC>>CCCC")
    assert result.complete and result.minimum_cost == 0
    assert len(result.mapped_reactions) == 1
    with pytest.raises(ValueError, match="max_bijections"):
        run("CCCC>>CCCC", max_bijections=1_000_000)


def test_native_budget_count_scope_and_compressed_hydrogen_validation(native_library):
    result = map_reaction(
        "FCl>>FCl", backend="cpp", library_path=native_library,
        CD=0, CD_mode="at_most",
    )
    assert result.complete and result.labeled_mapping_count == 1
    assert result.labeled_mapping_count_scope == "mapping_ball"
    with pytest.raises(ValueError, match="compressed"):
        map_reaction(
            "CO>>CO", backend="cpp", library_path=native_library,
            objective="bond", hydrogen="compressed", CD=2, CD_mode="at_most",
        )
