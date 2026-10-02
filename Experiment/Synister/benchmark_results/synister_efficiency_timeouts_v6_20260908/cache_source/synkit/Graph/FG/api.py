from __future__ import annotations

import networkx as nx

from synkit.IO.chem_converter import smiles_to_graph

from .detector import FunctionalGroupDetector

FunctionalGroupLabels = list[tuple[str, tuple[int, ...]]]


def smiles_to_graph_and_functional_groups(
    smiles: str,
    *,
    sanitize: bool = True,
) -> tuple[nx.Graph, FunctionalGroupLabels]:
    """Convert SMILES to a molecular graph and detect functional groups.

    Atom-mapped SMILES keep their non-zero atom-map numbers as graph node IDs.
    Unmapped atoms use their 1-based atom order as node IDs, so both mapped and
    unmapped SMILES can be passed to the same API.

    :param smiles: Input SMILES, with or without atom-map labels.
    :type smiles: str
    :param sanitize: If ``True``, sanitize the RDKit molecule during conversion.
    :type sanitize: bool
    :return: Molecular graph and detected ``(name, node_ids)`` FG labels.
    :rtype: tuple[nx.Graph, list[tuple[str, tuple[int, ...]]]]
    :raises ValueError: If the SMILES cannot be converted to a molecular graph.
    """
    graph = smiles_to_graph(
        smiles,
        drop_non_aam=False,
        sanitize=sanitize,
        use_index_as_atom_map=True,
    )
    if graph is None:
        raise ValueError(f"Could not convert SMILES to molecular graph: {smiles!r}")
    return graph, FunctionalGroupDetector().detect(graph)
