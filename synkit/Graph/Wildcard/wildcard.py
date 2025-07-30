import networkx as nx
from typing import Dict, Any, Tuple, Optional
from synkit.IO import rsmi_to_graph, graph_to_smi


class WildCard:
    """
    Static utility class for generating reaction SMILES with wildcards by
    augmenting the product graph with subgraphs unique to the reactant and
    patching lost external connections with wildcard atoms ('*').

    Optionally, can rebalance the reactant side to ensure both sides have
    matching atom maps (by adding wildcard atoms if needed).

    All methods are static and do not store any internal state.

    Example
    -------
    >>> WildCard.rsmi_with_wildcards('CCO>>CC')
    'CCO>>CC*'

    >>> WildCard.rsmi_with_wildcards('CCO>>CC', rebalance=True)
    'CCO*>>CC*'
    """

    @staticmethod
    def rsmi_with_wildcards(
        rsmi: str,
        attributes_defaults: Optional[Dict[str, Any]] = None,
        rebalance: bool = False,
    ) -> str:
        """
        Given a reaction SMILES string, returns a new reaction SMILES where the product
        side contains any disconnected subgraphs unique to the reactant, with lost
        external bonds patched with wildcard atoms. Optionally, also adds wildcards to
        the reactant side to ensure matching atom maps (rebalance).

        :param rsmi: Reaction SMILES (e.g., 'CCO>>CC')
        :type rsmi: str
        :param attributes_defaults: Optional dictionary of default attributes for wildcards
        :type attributes_defaults: dict, optional
        :param rebalance: Whether to rebalance the reactant side by adding wildcards
        :type rebalance: bool
        :returns: Augmented reaction SMILES string
        :rtype: str
        :raises ValueError: If parsing or output generation fails.

        Example
        -------
        >>> WildCard.rsmi_with_wildcards('CCO>>CC')
        'CCO>>CC*'
        >>> WildCard.rsmi_with_wildcards('CCO>>CC', rebalance=True)
        'CCO*>>CC*'
        """
        r, p = WildCard.from_rsmi(rsmi)
        new_r, new_p = WildCard.add_unique_subgraph_with_wildcards(
            r, p, attributes_defaults, rebalance=rebalance
        )
        try:
            return f"{WildCard.to_smi(new_r)}>>{WildCard.to_smi(new_p)}"
        except Exception as e:
            raise ValueError(
                "Could not convert to RSMI after wildcard patching."
            ) from e

    @staticmethod
    def add_unique_subgraph_with_wildcards(
        G: nx.Graph,
        H: nx.Graph,
        attributes_defaults: Optional[Dict[str, Any]] = None,
        rebalance: bool = False,
    ) -> Tuple[nx.Graph, nx.Graph]:
        """
        Add the subgraph unique to G as a disconnected union to H,
        and patch lost external connections with plain wildcard bonds.
        Optionally, rebalance the reactant side to ensure both sides have
        matching atom maps by adding wildcards.

        :param G: Reactant graph
        :type G: nx.Graph
        :param H: Product graph
        :type H: nx.Graph
        :param attributes_defaults: Optional attribute defaults for wildcard nodes
        :type attributes_defaults: dict, optional
        :param rebalance: Whether to rebalance the reactant side with wildcards
        :type rebalance: bool
        :returns: Tuple (new_G, new_H) with both graphs possibly augmented by wildcards
        :rtype: Tuple[nx.Graph, nx.Graph]
        :raises ValueError: If G or H are not valid graphs.

        Example
        -------
        >>> r, p = WildCard.from_rsmi('CCO>>CC')
        >>> r2, p2 = WildCard.add_unique_subgraph_with_wildcards(r, p, rebalance=True)
        """
        if not isinstance(G, nx.Graph) or not isinstance(H, nx.Graph):
            raise ValueError("G and H must be networkx.Graph instances")
        if G.number_of_nodes() == 0 or H.number_of_nodes() == 0:
            raise ValueError("Both G and H must have at least one node.")
        if not all("atom_map" in d for _, d in G.nodes(data=True)):
            raise ValueError(
                "All reactant nodes must have 'atom_map' attributes for unique subgraph logic."
            )
        if not all("atom_map" in d for _, d in H.nodes(data=True)):
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

        # Make working copies
        G_new = G.copy()
        H_new = H.copy()

        # ---------------------------
        # 1. PATCH PRODUCT SIDE (add unique reactant subgraphs and wildcards)
        # ---------------------------
        react_atom_maps = {d["atom_map"] for _, d in G.nodes(data=True)}
        prod_atom_maps = {d["atom_map"] for _, d in H.nodes(data=True)}

        # Identify nodes and subgraphs unique to reactant
        unique_atom_maps = react_atom_maps - prod_atom_maps
        node_map = {d["atom_map"]: n for n, d in G.nodes(data=True)}
        unique_nodes = [node_map[a] for a in unique_atom_maps if a in node_map]

        G_unique = G.subgraph(unique_nodes).copy()
        # Add unique reactant fragments to product
        for n, d in G_unique.nodes(data=True):
            H_new.add_node(n, **d)
        for u, v, d in G_unique.edges(data=True):
            H_new.add_edge(u, v, **d)

        # Add wildcards to patch lost external bonds (reactant → outside)
        existing_ids = set(H_new.nodes)
        next_id = (
            max([n for n in existing_ids if isinstance(n, int)], default=0) + 1
            if existing_ids
            else 1
        )

        for n in unique_nodes:
            for nbr in G.neighbors(n):
                nbr_map = G.nodes[nbr]["atom_map"]
                if nbr_map not in unique_atom_maps:
                    wc_id = next_id
                    next_id += 1
                    H_new.add_node(
                        wc_id,
                        **attributes_defaults,
                        atom_map=wc_id,
                        typesGH=(("*", False, 0, 0, []), ("*", False, 0, 0, [])),
                    )
                    H_new.add_edge(n, wc_id)

        # ---------------------------
        # 2. REBALANCE REACTANT SIDE (add wildcards to reactant if required)
        # ---------------------------
        if rebalance:
            prod_atom_maps = {d["atom_map"] for _, d in H_new.nodes(data=True)}
            missing_in_react = prod_atom_maps - react_atom_maps
            if missing_in_react:
                react_existing_ids = set(G_new.nodes)
                react_next_id = (
                    max(
                        [n for n in react_existing_ids if isinstance(n, int)], default=0
                    )
                    + 1
                    if react_existing_ids
                    else 1
                )
                for missing_map in missing_in_react:
                    wc_id = react_next_id
                    react_next_id += 1
                    G_new.add_node(
                        wc_id,
                        **attributes_defaults,
                        atom_map=missing_map,
                        typesGH=(("*", False, 0, 0, []), ("*", False, 0, 0, [])),
                    )

        return G_new, H_new

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
