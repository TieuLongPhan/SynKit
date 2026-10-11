"""Reference-blinded conversion of mapped reaction SMILES.

Mapped atom identifiers are needed to recover the held-out reference mapping,
but using the same identifier order on both endpoints leaks that mapping into
the search traversal.  This module applies independent deterministic endpoint
permutations and returns the reference correspondence separately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised only without RDKit
    Chem = None

from synkit.IO import rsmi_to_graph

from ..graph.labeled_graph import LabeledGraph


@dataclass(frozen=True)
class BlindedReactionProblem:
    """Graph pair and held-out mapping in blinded coordinates."""

    lgp: tuple[LabeledGraph, LabeledGraph]
    reference_mapping: tuple[int, ...]
    reactant_atom_maps: tuple[int, ...]
    product_atom_maps: tuple[int, ...]
    reaction_sha256: str
    heavy_only: bool
    blind_seed: str


def _blind_order(nodes, *, blind_seed: str, side: str) -> tuple[int, ...]:
    """Return an endpoint order independent of reference alignment."""

    def key(node):
        payload = f"{blind_seed}\0{side}\0{int(node)}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    return tuple(sorted((int(node) for node in nodes), key=key))


def _fold_explicit_hydrogens(graph) -> None:
    for node, attributes in graph.nodes(data=True):
        if attributes.get("element") == "H":
            continue
        explicit = sum(
            graph.nodes[neighbour].get("element") == "H"
            for neighbour in graph.neighbors(node)
        )
        attributes["hcount"] = int(attributes.get("hcount", 0) or 0) + int(explicit)


def _selected_graph(graph, *, heavy_only: bool):
    if not heavy_only:
        return graph.copy()
    _fold_explicit_hydrogens(graph)
    return graph.subgraph(
        node
        for node, attributes in graph.nodes(data=True)
        if attributes.get("element") != "H"
    ).copy()


def _as_labeled_graph(graph, ordered_nodes) -> LabeledGraph:
    positions = {node: index for index, node in enumerate(ordered_nodes)}
    adjacency = {index: {} for index in range(len(ordered_nodes))}
    for begin, end, attributes in graph.edges(data=True):
        weight = float(attributes.get("order", 1.0))
        adjacency[positions[begin]][positions[end]] = weight
        adjacency[positions[end]][positions[begin]] = weight
    periodic_table = Chem.GetPeriodicTable()
    elements = [
        int(periodic_table.GetAtomicNumber(graph.nodes[node]["element"]))
        for node in ordered_nodes
    ]
    labeled = LabeledGraph(adjacency, elements)
    labeled.set_prop("atomic numbers", elements)
    labeled.set_prop(
        "hcounts",
        [int(graph.nodes[node].get("hcount", 0) or 0) for node in ordered_nodes],
    )
    labeled.set_prop(
        "charges",
        [int(graph.nodes[node].get("charge", 0) or 0) for node in ordered_nodes],
    )
    return labeled


def blinded_mapped_reaction_problem(
    reaction: str,
    *,
    heavy_only: bool = True,
    blind_seed: str = "synister-global-v1",
) -> BlindedReactionProblem:
    """Convert mapped reaction SMILES without exposing reference alignment.

    The two endpoint orders are deterministic but use different salts.  The
    held-out mapping is returned only as a separate tuple, suitable for
    evaluation after a global search.
    """
    if not isinstance(reaction, str) or not reaction.strip():
        raise ValueError("reaction must be a non-empty mapped reaction SMILES")
    if Chem is None:
        raise ImportError("RDKit is required for blinded reaction conversion")
    if not isinstance(heavy_only, bool):
        raise TypeError("heavy_only must be boolean")
    if not isinstance(blind_seed, str) or not blind_seed:
        raise ValueError("blind_seed must be a non-empty string")

    reactant, product = rsmi_to_graph(
        reaction,
        drop_non_aam=True,
        use_index_as_atom_map=True,
        include_stereo_descriptors=False,
    )
    if reactant is None or product is None:
        raise ValueError("SynKit could not convert reaction SMILES")
    endpoints = tuple(
        _selected_graph(graph, heavy_only=heavy_only) for graph in (reactant, product)
    )
    for side, graph in zip(("reactant", "product"), endpoints):
        for node, attributes in graph.nodes(data=True):
            if int(attributes.get("atom_map", 0) or 0) != int(node):
                raise ValueError(f"{side} graph is not atom-map indexed")

    reactant_maps = set(endpoints[0])
    product_maps = set(endpoints[1])
    if reactant_maps != product_maps:
        raise ValueError("mapped endpoint atom inventories differ")
    if sorted(endpoints[0].nodes[node]["element"] for node in reactant_maps) != sorted(
        endpoints[1].nodes[node]["element"] for node in product_maps
    ):
        raise ValueError("mapped endpoint element inventories differ")

    reactant_order = _blind_order(reactant_maps, blind_seed=blind_seed, side="reactant")
    product_order = _blind_order(product_maps, blind_seed=blind_seed, side="product")
    product_positions = {
        atom_map: position for position, atom_map in enumerate(product_order)
    }
    reference = tuple(product_positions[atom_map] for atom_map in reactant_order)
    return BlindedReactionProblem(
        lgp=(
            _as_labeled_graph(endpoints[0], reactant_order),
            _as_labeled_graph(endpoints[1], product_order),
        ),
        reference_mapping=reference,
        reactant_atom_maps=reactant_order,
        product_atom_maps=product_order,
        reaction_sha256=hashlib.sha256(reaction.encode("utf-8")).hexdigest(),
        heavy_only=heavy_only,
        blind_seed=blind_seed,
    )


__all__ = ["BlindedReactionProblem", "blinded_mapped_reaction_problem"]
