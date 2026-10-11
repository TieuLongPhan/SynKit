"""Input-only selection for the prospectively declared controlled cost study."""

from collections import Counter
from dataclasses import asdict, replace

import networkx as nx
from rdkit import Chem

from Experiment.Stereo.Canonicalization.independent_fixtures import fixtures
from Experiment.Stereo.Canonicalization.independent_model import State, transport
from Experiment.Stereo.Canonicalization.independent_orbits import ATOM_KEYS, BOND_KEYS
from Experiment.Stereo.Chirality.published import EXPECTED_SHA256, load_dataset


def _spec(name, graph, states, *, stress=False, expected="complete"):
    return {"id": name, "kind": "constructed", "stress": stress, "expected": expected,
            "nodes": [(node, dict(data)) for node, data in graph.nodes(data=True)],
            "edges": [(a, b, dict(data)) for a, b, data in graph.edges(data=True)],
            "states": [asdict(value) for value in states], "atom_color": "color", "bond_color": "color"}


def _public():
    unique = {}
    for row in load_dataset():
        molecule = Chem.MolFromSmiles(row["Input SMILES"])
        if molecule is None:
            raise ValueError(f"Pinned ACS parse failure:{row['ID']}")
        neutral = Chem.Mol(molecule)
        Chem.RemoveStereochemistry(neutral)
        for atom in neutral.GetAtoms():
            atom.SetAtomMapNum(0)
        key = Chem.MolToSmiles(neutral, canonical=True, isomericSmiles=True)
        unique.setdefault(key, (molecule.GetNumAtoms(), row["ID"], row["Input SMILES"]))
    candidates = sorted(unique.values())
    selected = [candidates[index * (len(candidates) - 1) // 23] for index in range(24)]
    if len({identifier for _, identifier, _ in selected}) != 24:
        raise ValueError("ACS selection does not contain 24 unique constitutions.")
    return [{"id": "acs:" + identifier, "kind": "public", "stress": False, "expected": "complete",
             "smiles": smiles, "input_atom_count": count, "source_record_id": identifier,
             "dataset_sha256": EXPECTED_SHA256, "atom_color": ATOM_KEYS, "bond_color": BOND_KEYS}
            for count, identifier, smiles in selected]


def _support_cases():
    result = []
    for family in ("helical_open", "helical_cyclic", "planar_chirality"):
        for size in (5, 9, 17, 33):
            for symmetric in (False, True):
                graph = nx.path_graph(size) if family == "helical_open" else nx.cycle_graph(size)
                nx.set_node_attributes(graph, {node: "support" if symmetric else f"atom:{node}" for node in graph},
                                       "color")
                nx.set_edge_attributes(graph, "bond", "color")
                if family == "planar_chirality":
                    graph.add_node(size, color="pilot")
                    graph.add_edge(0, size, color="pilot-link")
                    seed = State(family, parity=1, path=tuple(range(size)), pilot=size)
                else:
                    seed = State("helical", parity=1, path=tuple(range(size)), cyclic=family == "helical_cyclic")
                name = f"support:{family}:{size}:{'symmetric' if symmetric else 'unique'}"
                result.append(_spec(name, graph, (seed,), stress=True))
    return result


def _components():
    seed = next(value for value in fixtures() if value.name == "tetrahedral:symmetric")
    result = []
    for count in (2, 4, 8):
        graph, states = nx.Graph(), []
        for index in range(count):
            mapping = {node: index * 10 + node for node in seed.graph}
            graph = nx.compose(graph, nx.relabel_nodes(seed.graph, mapping))
            states.append(transport(seed.seeds[0], mapping))
        result.append(_spec(f"components:tetrahedral:{count}", graph, states, stress=True))
    return result


def cases():
    """Freeze all selections without consulting runtime or identity predictions."""
    catalog = fixtures()
    result = [_spec(value.name, value.graph, value.seeds) for value in catalog]
    result += _public() + _support_cases() + _components()
    distinct = next(value for value in catalog if value.name == "tetrahedral:distinct")
    symmetric = next(value for value in catalog if value.name == "octahedral:symmetric")
    result.append(_spec("boundary:unknown_tetrahedral", distinct.graph,
                        (replace(distinct.seeds[0], parity=None),), expected="incomplete_input"))
    result.append(_spec("boundary:unsupported_registry", distinct.graph, (), expected="unsupported_input"))
    bounded = _spec("boundary:search_node_budget", symmetric.graph, symmetric.seeds, expected="bounded_search")
    bounded["max_search_nodes"] = 1
    result.append(bounded)
    for smiles, name in (("COP(=O)(N)[O-]", "phosphate"), ("O=[N+]([O-])c1ccccc1", "nitrobenzene"),
                         ("c1ccccc1", "benzene")):
        result.append({"id": "resonance:" + name, "kind": "resonance", "smiles": smiles,
                       "stress": False, "expected": "complete", "atom_color": ATOM_KEYS, "bond_color": BOND_KEYS})
    if len({value["id"] for value in result}) != len(result):
        raise ValueError("Duplicate controlled-study case identifier.")
    return tuple(result)


def state_from_json(value):
    """Restore raw handles without invoking production semantic methods."""
    value = dict(value)
    for key in ("atoms", "path", "support"):
        value[key] = tuple(value[key])
    value["frames"] = tuple((center, tuple(references), relation) for center, references, relation in value["frames"])
    return State(**value)


def selection_metadata(selected):
    return {"case_count": len(selected), "kinds": dict(Counter(value["kind"] for value in selected)),
            "selection_boundary": "input-only development-corpus strata; not a chemical holdout",
            "expected_outcomes": dict(Counter(value["expected"] for value in selected))}
