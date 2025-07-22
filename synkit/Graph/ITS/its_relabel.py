from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
from rdkit import Chem

from synkit.Graph.syn_graph import SynGraph
from synkit.IO import setup_logging
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.chem_converter import smiles_to_graph
from synkit.Graph.Matcher.graph_morphism import find_graph_isomorphism

logger = setup_logging("INFO")


class ITSRelabel:
    """Extend reaction SMILES through atom-map alignment between reactant and
    product SynGraphs.

    :cvar logger: Logger instance for debug and info messages.
    :type logger: logging.Logger
    :ivar graph_to_mol: Converter from SynGraph to RDKit Mol.
    :type graph_to_mol: GraphToMol
    """

    def __init__(self) -> None:
        """Initialize ITSRelabel with default GraphToMol converter."""
        self.graph_to_mol = GraphToMol()

    @staticmethod
    def _get_nodes_with_atom_map(graph: SynGraph) -> List[Any]:
        """Extract node IDs with a non-zero atom_map attribute from a SynGraph.

        :param graph: Input SynGraph with 'atom_map' on nodes.
        :type graph: SynGraph
        :returns: List of node identifiers where 'atom_map' != 0.
        :rtype: List[Any]
        """
        return [
            n
            for n, attrs in graph.raw.nodes(data=True)
            if attrs.get("atom_map", 0) != 0
        ]

    @staticmethod
    def _remove_internal_edges(graph: SynGraph, nodes: List[Any]) -> SynGraph:
        """Remove edges connecting nodes in the given list from a SynGraph.

        :param graph: Input SynGraph to prune.
        :type graph: SynGraph
        :param nodes: Node IDs whose connecting edges will be removed.
        :type nodes: List[Any]
        :returns: A new SynGraph with specified edges removed.
        :rtype: SynGraph
        """
        G_copy = SynGraph(graph.raw.copy())
        node_set = set(nodes)
        edges_to_remove = [
            (u, v) for u, v in G_copy.raw.edges() if u in node_set and v in node_set
        ]
        G_copy.raw.remove_edges_from(edges_to_remove)
        return G_copy

    @staticmethod
    def _dict_to_tuple_list(
        mapping: Dict[Any, Any], sort_by_key: bool = False, sort_by_value: bool = False
    ) -> List[Tuple[Any, Any]]:
        """Convert a mapping dict into a sorted list of tuples.

        :param mapping: Dictionary to convert.
        :type mapping: Dict[Any, Any]
        :param sort_by_key: Sort tuples by key if True.
        :type sort_by_key: bool
        :param sort_by_value: Sort tuples by value if True.
        :type sort_by_value: bool
        :returns: List of (key, value) tuples.
        :rtype: List[Tuple[Any, Any]]
        """
        items = list(mapping.items())
        if sort_by_key:
            items.sort(key=lambda x: x[0])
        elif sort_by_value:
            items.sort(key=lambda x: x[1])
        return items

    def _update_mapping(
        self,
        G: SynGraph,
        H: SynGraph,
        mapping: Iterable[Tuple[Any, Any]],
        aam_key: str = "atom_map",
    ) -> Tuple[SynGraph, SynGraph]:
        """Update node attributes in two SynGraphs based on a sequential
        mapping.

        This method resets the specified atom-map attribute for all
        nodes in both graphs to 0, then assigns a new atom-map value
        (i+1) for each mapped pair:     G.nodes[g_node][aam_key] = i + 1
        H.nodes[h_node][aam_key] = i + 1

        :param G: First SynGraph to update (reactant).
        :type G: SynGraph
        :param H: Second SynGraph to update (product).
        :type H: SynGraph
        :param mapping: Iterable of (g_node, h_node) tuples defining
            node correspondence.
        :type mapping: Iterable[Tuple[Any, Any]]
        :param aam_key: Name of the atom-map attribute on each node.
        :type aam_key: str
        :returns: Tuple of updated SynGraphs (G_updated, H_updated).
        :rtype: Tuple[SynGraph, SynGraph]
        """
        # Create deep copies of raw graphs
        G_copy = SynGraph(G.raw.copy())
        H_copy = SynGraph(H.raw.copy())

        # Reset atom-map to zero for all nodes
        for node in G_copy.raw.nodes():
            G_copy.raw.nodes[node][aam_key] = 0
        for node in H_copy.raw.nodes():
            H_copy.raw.nodes[node][aam_key] = 0

        # Assign new sequential mapping values
        for i, (g_node, h_node) in enumerate(mapping):
            value = i + 1
            if g_node in G_copy.raw:
                G_copy.raw.nodes[g_node][aam_key] = value
            else:
                logger.warning(f"Node {g_node} not found in reactant graph")
            if h_node in H_copy.raw:
                H_copy.raw.nodes[h_node][aam_key] = value
            else:
                logger.warning(f"Node {h_node} not found in product graph")

        return G_copy, H_copy

    def fit(self, rsmi: str) -> str:
        """Generate an extended reaction SMILES by aligning atom maps of
        reactant and product.

        :param rsmi: Reaction SMILES string formatted as 'reactant>>product'.
        :type rsmi: str
        :returns: Extended reaction SMILES after remapping.
        :rtype: str
        :raises ValueError: If input format is invalid or graphs are not isomorphic.

        :example:
        >>> its = ITSRelabel()
        >>> its.fit('CCO:1>>CC=O:1')
        'CCO>>CC=O'
        """
        # Parse reaction SMILES
        try:
            react_smiles, prod_smiles = rsmi.split(">>")
        except ValueError as e:
            raise ValueError("Expected 'reactant>>product' format") from e

        # Convert to SynGraphs
        G = SynGraph(
            smiles_to_graph(
                react_smiles, drop_non_aam=False, use_index_as_atom_map=False
            )
        )
        H = SynGraph(
            smiles_to_graph(
                prod_smiles, drop_non_aam=False, use_index_as_atom_map=False
            )
        )

        # Identify and prune mapped nodes
        R_nodes = self._get_nodes_with_atom_map(G)
        P_nodes = self._get_nodes_with_atom_map(H)
        Gp = self._remove_internal_edges(G, R_nodes)
        Hp = self._remove_internal_edges(H, P_nodes)

        # Find isomorphism mapping
        mapping = find_graph_isomorphism(Gp.raw, Hp.raw, use_defaults=True)
        if not mapping:
            raise ValueError("No isomorphism found between pruned graphs")

        # Update atom_map attributes
        mapped_pairs = self._dict_to_tuple_list(mapping, sort_by_key=True)
        G_new, H_new = self._update_mapping(G, H, mapped_pairs, aam_key="atom_map")

        # Convert back to molecules and SMILES
        mol_G = self.graph_to_mol.graph_to_mol(G_new.raw, use_h_count=True)
        mol_H = self.graph_to_mol.graph_to_mol(H_new.raw, use_h_count=True)
        return f"{Chem.MolToSmiles(mol_G)}>>{Chem.MolToSmiles(mol_H)}"
