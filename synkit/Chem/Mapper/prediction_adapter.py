"""Reference-free prediction inputs and strict coordinate alignment.

The connectivity contract matches :mod:`identifiability`. Alignment uses only
each endpoint graph; neither a deposited mapping nor an optimal candidate set
is accepted by this interface. Alternative endpoint isomorphisms differ by
endpoint automorphisms, so quotient bond scores are independent of that choice.
"""

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
from rdkit import Chem

from .evaluation import _graph
from .identifiability import extract_label, parse_reaction
from .slap.sequential import GraphMatcher


def unmapped_input(reaction: str) -> str:
    """Remove all map labels before serialization; retain every component.

    Coordinates for the prediction study are those of the returned string.
    Stereo is intentionally removed for the connectivity-only task, after the
    strict parser rejects unsupported inventories, isotopes and radicals.
    """
    parse_reaction(reaction)
    sides = []
    for side in reaction.split(">>"):
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(side, params)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = Chem.RemoveHs(mol)
        sides.append(Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False))
    result = ">>".join(sides)
    parse_reaction(result)
    return result


def _map_ids(side):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(side, params)
    if mol is None:
        raise ValueError("Invalid predicted SMILES")
    # RemoveHs retains heavy-atom relative order. H mapping is outside scope.
    ids = tuple(a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if not ids or any(i <= 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every predicted heavy atom needs one unique positive map ID")
    return ids


def _align(source, target):
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        _graph(source), _graph(target),
        node_match=nx.algorithms.isomorphism.categorical_node_match("color", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("order", None),
    )
    match = next(matcher.isomorphisms_iter(), None)
    if match is None:
        raise ValueError("Prediction changed an attributed endpoint graph")
    return tuple(match[i] for i in range(len(source.atomic_numbers)))


@dataclass(frozen=True)
class AlignedPrediction:
    mapping: tuple[int, ...]
    reactant_to_output: tuple[int, ...]
    product_to_output: tuple[int, ...]


def align_mapped_prediction(unmapped_reaction: str, mapped_output: str) -> AlignedPrediction:
    """Validate an untouched mapped output and return alignment witnesses."""
    r, p = parse_reaction(unmapped_reaction)
    rr, pp = parse_reaction(mapped_output)
    rid, pid = map(_map_ids, mapped_output.split(">>"))
    if set(rid) != set(pid):
        raise ValueError("Predicted map-ID inventories differ between endpoints")
    ra, pa = _align(r, rr), _align(p, pp)
    product_by_id = {pid[out]: original for original, out in enumerate(pa)}
    mapping = tuple(product_by_id[rid[out]] for out in ra)
    extract_label(r, p, mapping)  # complete element-compatible permutation
    return AlignedPrediction(mapping, ra, pa)


def predict_slap(unmapped_reaction: str):
    """SLAP heavy-atom symmetry splitting with deterministic residual ties.

    Requests splitting for every heavy atom, uses the first returned partition,
    then pairs sorted indices
    within each common final label. This is a declared single-map baseline,
    not an exact optimizer, a ground-truth repair, or a claim that all unresolved
    label blocks contain automorphism-equivalent atoms.
    """
    r, p = parse_reaction(unmapped_reaction)
    matcher = GraphMatcher(binary=False, max_lap_fingerprints=1000,
                           cache_label_blocks=True, deterministic_labels=True)
    matcher.get_maps([r.graph(), p.graph()], break_sym_targets=list(range(len(r.atomic_numbers))))
    if not matcher.results:
        raise ValueError("SLAP returned no partition")
    result = matcher.results[0]
    lr, lp = result["lgp"]
    left, right = defaultdict(list), defaultdict(list)
    for i, label in enumerate(lr.labels):
        left[label].append(i)
    for i, label in enumerate(lp.labels):
        right[label].append(i)
    if set(left) != set(right) or any(len(left[k]) != len(right[k]) for k in left):
        raise ValueError("SLAP partition multiplicities disagree")
    mapping = [-1] * len(r.atomic_numbers)
    for label in sorted(left):
        for i, j in zip(left[label], right[label]):
            mapping[i] = j
    extract_label(r, p, mapping)
    return {
        "method": "slap-heavy-split-v1",
        "mapping": mapping,
        "raw_reactant_labels": list(lr.labels),
        "raw_product_labels": list(lp.labels),
        "returned_partition_count": len(matcher.results),
        "reported_partition_cost": int(result["val"]),
        "heavy_atom_splitting": True,
        "tie_policy": "first-returned-partition_then_increasing-index-within-label",
    }
