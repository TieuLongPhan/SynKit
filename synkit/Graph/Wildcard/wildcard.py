import networkx as nx
from typing import Dict, Any, Tuple
from synkit.IO import rsmi_to_graph, graph_to_smi


class WildCard:
    """
    Static utility class for generating reaction SMILES with wildcards by
    augmenting the product graph with subgraphs unique to the reactant and
    patching lost external connections with wildcard atoms ('*').

    All methods are static and do not store any internal state.

    Example
    -------
    >>> WildCard.rsmi_with_wildcards('CCO>>CC')
    'CCO>>CC*'
    """

    @staticmethod
    def rsmi_with_wildcards(
        rsmi: str, attributes_defaults: Dict[str, Any] = None
    ) -> str:
        """
        Given a reaction SMILES string, returns a new reaction SMILES where the product
        side contains any disconnected subgraphs unique to the reactant, with lost
        external bonds patched with wildcard atoms.

        :param rsmi: Reaction SMILES (e.g., 'CCO>>CC')
        :type rsmi: str
        :param attributes_defaults: Optional dictionary of default attributes for wildcards
        :type attributes_defaults: dict, optional
        :returns: Augmented reaction SMILES string
        :rtype: str
        :raises ValueError: If parsing or output generation fails.

        Example
        -------
        >>> WildCard.rsmi_with_wildcards('CCO>>CC')
        'CCO>>CC*'
        """
        r, p = WildCard.from_rsmi(rsmi)
        _, new_p = WildCard.add_unique_subgraph_with_wildcards(
            r, p, attributes_defaults
        )
        try:
            return f"{WildCard.to_smi(r)}>>{WildCard.to_smi(new_p)}"
        except Exception as e:
            raise ValueError(
                "Could not convert to RSMI after wildcard patching."
            ) from e

    @staticmethod
    def add_unique_subgraph_with_wildcards(
        G: nx.Graph, H: nx.Graph, attributes_defaults: Dict[str, Any] = None
    ) -> Tuple[nx.Graph, nx.Graph]:
        """
        Add the subgraph unique to G as a disconnected union to H,
        and patch lost external connections with plain wildcard bonds.

        :param G: Reactant graph
        :type G: nx.Graph
        :param H: Product graph
        :type H: nx.Graph
        :param attributes_defaults: Optional attribute defaults for wildcard nodes
        :type attributes_defaults: dict, optional
        :returns: Tuple (G, new_H) with new_H augmented by wildcards
        :rtype: Tuple[nx.Graph, nx.Graph]
        :raises ValueError: If G or H are not valid graphs.
        """
        if not isinstance(G, nx.Graph) or not isinstance(H, nx.Graph):
            raise ValueError("G and H must be networkx.Graph instances")
        if G.number_of_nodes() == 0 or H.number_of_nodes() == 0:
            raise ValueError("Both G and H must have at least one node.")
        if not all("atom_map" in d for n, d in G.nodes(data=True)):
            raise ValueError(
                "All reactant nodes must have 'atom_map' attributes for unique subgraph logic."
            )
        if not all("atom_map" in d for n, d in H.nodes(data=True)):
            raise ValueError(
                "All product nodes must have 'atom_map' attributes for unique subgraph logic."
            )
        if attributes_defaults is None:
            attributes_defaults = {
                "element": "*",
                "aromatic": False,
                "hcount": 0,
                "charge": 0,
                "neighbors": [],
            }

        H_new = H.copy()
        unique_nodes = set(G.nodes) - set(H.nodes)
        G_unique = G.subgraph(unique_nodes).copy()

        for n, d in G_unique.nodes(data=True):
            H_new.add_node(n, **d)
        for u, v, d in G_unique.edges(data=True):
            H_new.add_edge(u, v, **d)

        existing_ids = [n for n in H_new.nodes if isinstance(n, int)]
        next_id = max(existing_ids, default=0) + 1

        for n in unique_nodes:
            for nbr in G.neighbors(n):
                if nbr not in unique_nodes:
                    wc_id = next_id
                    next_id += 1
                    H_new.add_node(
                        wc_id,
                        element="*",
                        charge=0,
                        typesGH=(("*", False, 0, 0, []), ("*", False, 0, 0, [])),
                        atom_map=wc_id,
                    )
                    H_new.add_edge(n, wc_id)
        return G, H_new

    @staticmethod
    def from_rsmi(rsmi: str) -> Tuple[nx.Graph, nx.Graph]:
        """
        Convert a reaction SMILES string into reactant and product graphs.

        :param rsmi: Reaction SMILES string
        :type rsmi: str
        :returns: Tuple (reactant_graph, product_graph)
        :rtype: Tuple[nx.Graph, nx.Graph]
        :raises ValueError: If input cannot be parsed.
        """
        try:
            return rsmi_to_graph(rsmi)
        except Exception as e:
            raise ValueError(f"Could not parse RSMI: {rsmi}") from e

    @staticmethod
    def to_smi(G: nx.Graph) -> str:
        """
        Convert a networkx molecular graph to a canonical SMILES string.

        :param G: Molecular graph
        :type G: nx.Graph
        :returns: SMILES string
        :rtype: str
        :raises ValueError: If conversion fails.
        """
        try:
            return graph_to_smi(G)
        except Exception as e:
            raise ValueError("Could not convert graph to SMILES") from e

    @staticmethod
    def describe():
        """
        Print a description and usage example for this class.
        """
        print(WildCard.__doc__)

    def __repr__(self):
        return "<WildCard: static wildcard-augmentation for reaction SMILES>"
