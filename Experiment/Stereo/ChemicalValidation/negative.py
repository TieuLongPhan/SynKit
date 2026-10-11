"""Replay frozen negative truth before inspecting production perception."""

from rdkit import Chem
import networkx as nx

from Experiment.Stereo.ChemicalValidation.common import replay_typed_map, wall_limit


def negative_graph(molecule):
    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        graph.add_node(atom.GetIdx(), color=(atom.GetSymbol(), atom.GetIsotope(), atom.GetFormalCharge(),
                                             atom.GetTotalNumHs(), atom.GetIsAromatic(), atom.GetNumRadicalElectrons()))
    for bond in molecule.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                       color=(str(bond.GetBondType()), bond.GetIsAromatic()))
    return graph


def _check_phenyl_ring(molecule, graph, ring):
    if any(not graph.has_edge(ring[index], ring[(index + 1) % 6]) for index in range(6)):
        raise ValueError("negative_ring_not_a_cycle")
    for node in ring:
        atom = molecule.GetAtomWithIdx(node)
        if atom.GetSymbol() != "C" or not atom.GetIsAromatic():
            raise ValueError("negative_ring_not_phenyl")
        if node != ring[0] and (atom.GetDegree() != 2 or atom.GetTotalNumHs() != 1):
            raise ValueError("negative_terminal_ring_is_substituted")


def replay_negative(record):
    """Certify the branch-exchange proof, independently of any predicted locus."""
    molecule = Chem.MolFromSmiles(record["smiles"])
    if molecule is None:
        raise ValueError("negative_source_smiles_parse_failure")
    witness = record["independent_negative_witness"]
    pairs = witness["atom_map"]
    mapping = dict(pairs)
    if len(mapping) != len(pairs):
        raise ValueError("negative_witness_duplicate_domain")
    graph = negative_graph(molecule)
    replay_typed_map(graph, graph, mapping)
    axis = tuple(witness["axis_endpoints"])
    ring = tuple(witness["ring_order"])
    if (len(axis) != 2 or len(set(axis)) != 2 or not graph.has_edge(*axis) or
            len(ring) != 6 or len(set(ring)) != 6 or witness["fixed_terminal"] not in axis or
            ring[0] != witness["fixed_terminal"]):
        raise ValueError("negative_witness_invalid_axis_or_ring")
    bond = molecule.GetBondBetweenAtoms(*axis)
    if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("negative_axis_not_nonring_single_bond")
    if not all(molecule.GetAtomWithIdx(node).GetIsAromatic() for node in axis):
        raise ValueError("negative_axis_not_biaryl")
    if any(mapping[node] != node for node in axis):
        raise ValueError("negative_axis_endpoints_not_fixed")
    if any(mapping[node] != node for node in graph if node not in ring):
        raise ValueError("negative_outside_ring_not_fixed")
    if any(mapping[ring[index]] != ring[(-index) % 6] for index in range(6)):
        raise ValueError("negative_map_not_declared_ring_reflection")
    _check_phenyl_ring(molecule, graph, ring)
    branches = tuple(witness["swapped_terminal_references"])
    if set(branches) != {ring[1], ring[-1]} or mapping[branches[0]] != branches[1]:
        raise ValueError("negative_witness_does_not_exchange_terminal_branches")
    return molecule, {"full_typed_bijection_replayed": True, "axis_endpoints_fixed": True,
                      "outside_ring_fixed": True, "terminal_branches_exchanged": True,
                      "expected": "no_constitutional_bond_atropisomerism",
                      "experimental_persistence": "not_assessed"}


def evaluate_negative(record):
    """Observe only the declared bond; a broad candidate is not a false positive."""
    molecule, truth = replay_negative(record)
    from synkit.Chem.Molecule.stereo_perception import detect_potential_stereo_elements

    endpoints = set(record["independent_negative_witness"]["axis_endpoints"])
    order = list(reversed(range(molecule.GetNumAtoms())))
    renumbering = {old: new for new, old in enumerate(order)}
    results = []
    for working, expected in ((molecule, endpoints),
                              (Chem.RenumberAtoms(molecule, order), {renumbering[node] for node in endpoints})):
        with wall_limit(10):
            candidates = [element for element in detect_potential_stereo_elements(working)
                          if element.element_type.value == "atrop_axis" and
                          set(element.support.path) == expected]
        results.append([{"element_type": element.element_type.value, "path": list(element.support.path),
                         "carrier_status": element.carrier_status.value, "reason": element.carrier_reason}
                        for element in candidates])
    correct = all(len(items) == 1 and items[0]["carrier_status"] == "symmetry_related" for items in results)
    return {"id": record["id"], "cid": record["cid"], "source_url": record["source_url"], "truth": truth,
            "axis_endpoints": sorted(endpoints), "production_original": results[0],
            "production_renumbered": results[1], "renumbering": sorted(renumbering.items()), "passed": correct}
