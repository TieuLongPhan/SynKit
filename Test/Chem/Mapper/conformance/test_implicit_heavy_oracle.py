"""Independent heavy-class and explicit-H oracles for minimum-H completion.

The oracle constructs endpoint chemistry from RDKit atoms, enumerates every
typed heavy permutation and compares exact attributed graph isomorphism. It
does not use production preparation, distance, H transport or canonical keys
to decide the expected score or class membership.
"""

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations, product

import networkx as nx
import pytest
from rdkit import Chem


@dataclass
class Endpoint:
    mol: object
    heavy: list
    types: list
    h: list
    electrons: list
    attributes: list
    bonds: dict
    stereo: dict


def _electrons(atom):
    # Independent valence-shell bookkeeping for this deliberately small test
    # domain. No production lone-pair estimator or atom graph is consulted.
    valence = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 16: 6, 17: 7}[atom.GetAtomicNum()]
    bonded = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
    remaining = (
        valence
        - atom.GetFormalCharge()
        - bonded
        - atom.GetTotalNumHs(includeNeighbors=False)
        - atom.GetNumRadicalElectrons()
    )
    assert remaining == int(remaining)
    return 2 * max(0, int(remaining) // 2) + atom.GetNumRadicalElectrons()


def _parse(smiles, *, clear_maps=True):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    if clear_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def _endpoints(reaction):
    left, _, right = reaction.split(">")
    mols = [_parse(left), _parse(right)]
    atoms = [[a for a in mol.GetAtoms() if a.GetAtomicNum() != 1] for mol in mols]
    inventories = [
        Counter((a.GetAtomicNum(), a.GetIsotope()) for a in side) for side in atoms
    ]
    universe = inventories[0] | inventories[1]
    result = []
    for mol, heavy, inventory in zip(mols, atoms, inventories):
        types = [(a.GetAtomicNum(), a.GetIsotope()) for a in heavy]
        types += sorted((universe - inventory).elements())
        positions = {a.GetIdx(): i for i, a in enumerate(heavy)}
        # Count before removing anything. Explicit isotope-H remains part of
        # the class H-count, but only isotope-zero H enters the count score.
        ordinary = [
            a.GetTotalNumHs(includeNeighbors=False)
            + sum(
                b.GetAtomicNum() == 1 and not b.GetIsotope() for b in a.GetNeighbors()
            )
            for a in heavy
        ]
        electrons = [_electrons(a) for a in heavy]
        attrs = [
            (
                types[i],
                a.GetFormalCharge(),
                a.GetNumRadicalElectrons(),
                a.GetTotalNumHs(includeNeighbors=True),
                a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "",
                False,
                ordinary[i],
                electrons[i],
            )
            for i, a in enumerate(heavy)
        ]
        for kind in types[len(heavy) :]:
            attrs.append((kind, 0, 0, 0, "", True, 0, 0))
        bonds, stereo = {}, {}
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in positions and b in positions:
                pair = tuple(sorted((positions[a], positions[b])))
                bonds[pair] = round(2 * bond.GetBondTypeAsDouble())
                stereo[pair] = str(bond.GetStereo())
        padding = len(types) - len(heavy)
        result.append(
            Endpoint(
                mol,
                heavy,
                types,
                ordinary + [0] * padding,
                electrons + [0] * padding,
                attrs,
                bonds,
                stereo,
            )
        )
    return result


def _typed_maps(left_types, right_types):
    groups = [defaultdict(list), defaultdict(list)]
    for group, kinds in zip(groups, (left_types, right_types)):
        for i, kind in enumerate(kinds):
            group[kind].append(i)
    assert Counter(left_types) == Counter(right_types)
    kinds = sorted(groups[0])
    for images in product(*(permutations(groups[1][kind]) for kind in kinds)):
        mapping = [-1] * len(left_types)
        for kind, targets in zip(kinds, images):
            for source, target in zip(groups[0][kind], targets):
                mapping[source] = target
        yield tuple(mapping)


def _edit2(left, right, mapping):
    inverse = {image: i for i, image in enumerate(mapping)}
    source = Counter(left)
    target = Counter(
        {
            tuple(sorted((inverse[i], inverse[j]))): value
            for (i, j), value in right.items()
        }
    )
    return sum((source - target).values()) + sum((target - source).values())


def _source_offset2(left):
    total = 0
    for atom in left.mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            total += _electrons(atom)
            if atom.GetIsotope():
                total += sum(
                    round(2 * bond.GetBondTypeAsDouble())
                    for bond in atom.GetBonds()
                    if bond.GetOtherAtom(atom).GetAtomicNum() != 1
                )
    total += sum(
        round(2 * bond.GetBondTypeAsDouble())
        for bond in left.mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() == bond.GetEndAtom().GetAtomicNum() == 1
    )
    return total


def _scores(endpoints):
    left, right = endpoints
    return {
        mapping: _edit2(left.bonds, right.bonds, mapping)
        + sum(abs(e - right.electrons[j]) for e, j in zip(left.electrons, mapping))
        + 2 * sum(abs(h - right.h[j]) for h, j in zip(left.h, mapping))
        + _source_offset2(left)
        for mapping in _typed_maps(left.types, right.types)
    }


def _its(endpoints, mapping):
    left, right = endpoints
    graph = nx.Graph()
    for i, j in enumerate(mapping):
        graph.add_node(i, state=(left.attributes[i], right.attributes[j]))
    for i in range(len(mapping)):
        for k in range(i):
            a = tuple(sorted((i, k)))
            b = tuple(sorted((mapping[i], mapping[k])))
            state = (
                left.bonds.get(a, 0),
                right.bonds.get(b, 0),
                left.stereo.get(a, ""),
                right.stereo.get(b, ""),
            )
            if state[0] or state[1]:
                graph.add_edge(i, k, state=state)
    return graph


def _same(left, right):
    return nx.is_isomorphic(
        left,
        right,
        node_match=lambda a, b: a["state"] == b["state"],
        edge_match=lambda a, b: a["state"] == b["state"],
    )


def _classes(endpoints, mappings):
    representatives = []
    for mapping in mappings:
        graph = _its(endpoints, mapping)
        if not any(_same(graph, previous) for previous in representatives):
            representatives.append(graph)
    return representatives


def _full_h_costs_by_heavy_map(endpoints):
    """Explicit-H typed enumeration, built without any AddHs or production code."""
    n = len(endpoints[0].types)
    hydrogen_atoms, hydrogen_types = [], []
    for endpoint in endpoints:
        explicit = [a for a in endpoint.mol.GetAtoms() if a.GetAtomicNum() == 1]
        generated = [
            i
            for i, atom in enumerate(endpoint.heavy)
            for _ in range(atom.GetTotalNumHs(includeNeighbors=False))
        ]
        hydrogen_atoms.append((explicit, generated))
        hydrogen_types.append(
            [(1, a.GetIsotope()) for a in explicit] + [(1, 0)] * len(generated)
        )
    universe = Counter(hydrogen_types[0]) | Counter(hydrogen_types[1])
    types, edges = [], []
    for endpoint, (explicit, generated), htypes in zip(
        endpoints, hydrogen_atoms, hydrogen_types
    ):
        padded_h = htypes + sorted((universe - Counter(htypes)).elements())
        kinds = endpoint.types + padded_h + [(0, -1)]
        anchor = len(kinds) - 1
        positions = {a.GetIdx(): i for i, a in enumerate(endpoint.heavy)}
        positions.update({a.GetIdx(): n + i for i, a in enumerate(explicit)})
        bonds = dict(endpoint.bonds)
        for bond in endpoint.mol.GetBonds():
            if (
                bond.GetBeginAtom().GetAtomicNum() == 1
                or bond.GetEndAtom().GetAtomicNum() == 1
            ):
                pair = tuple(
                    sorted(
                        (
                            positions[bond.GetBeginAtomIdx()],
                            positions[bond.GetEndAtomIdx()],
                        )
                    )
                )
                bonds[pair] = round(2 * bond.GetBondTypeAsDouble())
        for j, heavy_index in enumerate(generated, n + len(explicit)):
            bonds[(heavy_index, j)] = 2
        for i, electron in enumerate(endpoint.electrons):
            if electron:
                bonds[(i, anchor)] = electron
        for i, atom in enumerate(explicit, n):
            if _electrons(atom):
                bonds[(i, anchor)] = _electrons(atom)
        types.append(kinds)
        edges.append(bonds)
    costs = defaultdict(set)
    for mapping in _typed_maps(*types):
        costs[mapping[:n]].add(_edit2(*edges, mapping))
    return costs


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    for name in (
        "SYNKIT_MAPPER_OPTIONAL_LIBRARY",
        "SYNKIT_WAVE2_LIBRARY",
        "SYNKIT_CORE_LIBRARY",
    ):
        if os.environ.get(name):
            return os.environ[name]
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("implicit-heavy-oracle"))


REACTIONS = [
    "CCO>>COC",
    "[CH3][CH2][C]>>[CH3][C][CH2]",
    "CO>>C",
    "[13CH3]C>>C[13CH3]",
    "F[C@H](Cl)C>>F[C@@H](Cl)C",
    "F[C@H](Cl)[C@@H](F)Cl>>F[C@H](Cl)[C@@H](F)Cl",
    "F/C=C/F>>F/C=C\\F",
    "F/C=C/C=C\\F>>F/C=C/C=C\\F",
    "C.[H+]>>C",
    "C>>C.[H+]",
    "C.[H][H]>>C",
    "[CH3+].[H-]>>C",
    "[CH3].[H]>>C",
    "[CH3]>>C",
    "C.[H]>>C",
    "[2H]C>>C",
]

# Sulfur's H/radical states and a thiol acceptor fragment. These fixtures use
# the independent heavy-permutation/NetworkX oracle only: enumerating all
# hydrogen identities of the larger molecules would add factorial work.
SULFUR_REACTIONS = [
    "C([C@H]([NH3+])C(=O)[O-])S>>C[C@H]([NH3+])C(=O)[O-].S",
    "C([C@H]([NH3+])C(=O)[O-])S>>C[C@H]([NH3+])C(=O)[O-].[S]",
    "C([C@H]([NH3+])C(=O)[O-])S.CS>>C[C@H]([NH3+])C(=O)[O-].CSS",
]


@pytest.mark.parametrize("reaction", REACTIONS)
def test_each_heavy_map_score_is_minimum_of_all_full_h_lifts(reaction):
    endpoints = _endpoints(reaction)
    expected = _scores(endpoints)
    explicit = _full_h_costs_by_heavy_map(endpoints)
    assert set(explicit) == set(expected)
    assert {mapping: min(values) for mapping, values in explicit.items()} == expected
    assert min(expected.values()) == min(min(values) for values in explicit.values())


@pytest.mark.parametrize("reaction", REACTIONS + SULFUR_REACTIONS)
@pytest.mark.parametrize("query", ["minimum", "exact", "at_most"])
def test_native_implicit_scores_and_exact_heavy_class_sets(library, reaction, query):
    from synkit.Chem.Mapper._native.implicit import map_reaction_implicit

    endpoints = _endpoints(reaction)
    costs = _scores(endpoints)
    optimum = min(costs.values())
    target2 = max(costs.values()) if query != "minimum" else optimum
    chosen = {
        p
        for p, value in costs.items()
        if (value <= target2 if query == "at_most" else value == target2)
    }
    options = {} if query == "minimum" else {"CD": target2 / 2, "CD_mode": query}
    result = map_reaction_implicit(
        reaction, library_path=library, backend="pabs", time_limit_seconds=5, **options
    )
    assert result.complete and result.search_complete and result.classification_complete
    if query == "minimum":
        assert result.minimum_cost == optimum / 2
    actual = []
    n = len(endpoints[0].types)
    for record in result.reactions:
        assert len(record.mapping) == n + 2
        assert record.mapping[n:] == (n, n + 1)
        mapping = tuple(record.mapping[:n])
        assert mapping in chosen
        assert record.distance * 2 == costs[mapping]
        actual.append(_its(endpoints, mapping))
    expected = _classes(endpoints, chosen)
    # Every expected attributed isomorphism class occurs, every returned class
    # is admissible, and no duplicate class is hidden behind distinct hashes.
    assert len(actual) == len(expected)
    assert all(any(_same(a, b) for b in actual) for a in expected)
    assert all(any(_same(a, b) for a in expected) for b in actual)
    assert all(not _same(a, b) for i, a in enumerate(actual) for b in actual[:i])


def test_exact_heavy_minimum_shell_is_not_projection_of_full_h_shell(library):
    from synkit.Chem.Mapper._native.implicit import map_reaction_implicit

    reaction = "C.O>>C.O"
    endpoints = _endpoints(reaction)
    # One unique heavy map; exchanging one C-H and one O-H breaks four bonds.
    explicit = _full_h_costs_by_heavy_map(endpoints)
    assert explicit == {(0, 1): {0, 8, 16}}
    assert _scores(endpoints) == {(0, 1): 0}
    old_shell_has_lift = any(8 in values for values in explicit.values())
    assert old_shell_has_lift
    result = map_reaction_implicit(
        reaction, library_path=library, CD=4, CD_mode="exact"
    )
    assert result.complete and not result.reactions
    # Dmin <= CD is also insufficient to decide the old existential shell:
    # CD=2 is above the heavy minimum but no full-H lift has that cost.
    assert all(4 not in values for values in explicit.values())


@pytest.mark.parametrize("query", ["exact", "at_most"])
@pytest.mark.parametrize("reaction,minimum2", [("C.[H][H]>>C", 2), ("C.[H]>>C", 1)])
def test_target_below_source_constant_is_complete_and_empty(
    library, query, reaction, minimum2
):
    from synkit.Chem.Mapper._native.implicit import map_reaction_implicit

    assert min(_scores(_endpoints(reaction)).values()) == minimum2
    result = map_reaction_implicit(reaction, library_path=library, CD=0, CD_mode=query)
    assert result.complete and not result.reactions


@pytest.mark.parametrize("reaction", REACTIONS)
def test_prepared_original_endpoint_class_attributes(library, reaction):
    from synkit.Chem.Mapper._native.implicit import prepare_implicit_reaction

    expected = _endpoints(reaction)
    prepared = prepare_implicit_reaction(reaction)
    for endpoint, graph in zip(expected, prepared.graphs):
        n = len(endpoint.types)
        assert graph.labels[:n] == endpoint.types
        for name, column in (
            ("charge", 1),
            ("radical", 2),
            ("hcount", 3),
            ("stereo", 4),
            ("dummy", 5),
        ):
            assert list(graph.props[name][:n]) == [
                state[column] for state in endpoint.attributes
            ]
        assert list(graph.props["hcounts"][:n]) == endpoint.h
        assert list(graph.props["electrons"][:n]) == endpoint.electrons


@pytest.mark.parametrize("reaction", ["CCO>>COC", "CCO.[H+]>>COC.[H+]"])
def test_production_implicit_track_never_expands_hydrogen(
    library, monkeypatch, reaction
):
    from synkit.Chem.Mapper._native.implicit import map_reaction_implicit

    endpoints = _endpoints(reaction)
    optimum = min(_scores(endpoints).values())

    def fail(*args, **kwargs):
        raise AssertionError("Implicit-heavy mapping must not call Chem.AddHs")

    monkeypatch.setattr(Chem, "AddHs", fail)
    result = map_reaction_implicit(reaction, library_path=library)
    assert result.complete and result.minimum_cost == optimum / 2
    assert all(
        not atom.GetAtomMapNum()
        for record in result.reactions
        for side in (
            record.mapped_reaction.split(">")[0],
            record.mapped_reaction.split(">")[2],
        )
        for atom in _parse(side, clear_maps=False).GetAtoms()
        if atom.GetAtomicNum() == 1
    )


@pytest.mark.parametrize(
    "reaction,attribute,expected_with,expected_without",
    [
        (
            "F[C@H](Cl)[C@@H](F)Cl>>F[C@H](Cl)[C@@H](F)Cl",
            "CIP",
            8,
            4,
        ),
        ("F/C=C/C=C\\F>>F/C=C/C=C\\F", "EZ", 48, 16),
    ],
)
def test_stereo_fixtures_actually_detect_missing_class_attributes(
    reaction, attribute, expected_with, expected_without
):
    endpoints = _endpoints(reaction)
    maps = _scores(endpoints)
    assert len(_classes(endpoints, maps)) == expected_with
    for endpoint in endpoints:
        if attribute == "CIP":
            endpoint.attributes = [(*a[:4], "", *a[5:]) for a in endpoint.attributes]
        else:
            endpoint.stereo = {}
    assert len(_classes(endpoints, maps)) == expected_without
