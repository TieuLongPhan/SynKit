import networkx as nx
from typing import List, Dict, Any, Set, Optional
from rdkit import Chem
from rdkit.Chem import AllChem


class GraphToSMARTS:
    """
    Convert NetworkX graphs (with placeholder nodes/constraints) into SMARTS or reaction SMARTS strings.

    :param placeholder_labels: Set of labels recognized as placeholders (e.g., '_R', 'X', 'Y', 'Z').
    :type placeholder_labels: Optional[Set[str]]
    :param validate: If True, validate generated SMARTS or reaction SMARTS using RDKit.
    :type validate: bool
    :raises: None

    :Example:
    >>> G = nx.Graph()
    >>> G.add_node(1, label='C', constraint=None)
    >>> G.add_node(2, label='O', constraint=None)
    >>> G.add_edge(1, 2, order=1)
    >>> g2s = GraphToSMARTS()
    >>> smarts = g2s.graph_to_smarts(G)
    >>> isinstance(smarts, str)
    True
    """

    def __init__(
        self, placeholder_labels: Optional[Set[str]] = None, validate: bool = True
    ) -> None:
        """
        Initialize the GraphToSMARTS converter.

        :param placeholder_labels: Labels to treat as wildcard placeholders; defaults to {'_R','X','Y','Z'}.
        :type placeholder_labels: Optional[Set[str]]
        :param validate: Whether to validate SMARTS with RDKit if available.
        :type validate: bool
        :returns: None
        :rtype: None
        """
        if placeholder_labels is None:
            placeholder_labels = {"_R", "X", "Y", "Z"}
        self.placeholder_labels: Set[str] = placeholder_labels
        self.validate: bool = validate

    def graph_to_smarts(self, G: nx.Graph) -> str:
        """
        Convert a NetworkX graph into a SMARTS string representation.

        :param G: NetworkX Graph with node attributes:
                  - 'label': str atomic label or placeholder
                  - 'constraint': Optional[List[str]] allowed element list for placeholders
                  and edge attribute:
                  - 'order': float bond order (1,1.5,2,3)
        :type G: nx.Graph
        :returns: SMARTS string encoding the graph structure.
        :rtype: str
        :raises ValueError: If RDKit fails to parse the generated SMARTS when validate=True.

        :Example:
        >>> G = nx.Graph()
        >>> G.add_node(1, label='C', constraint=None)
        >>> G.add_node(2, label='O', constraint=None)
        >>> G.add_edge(1, 2, order=1)
        >>> smarts = GraphToSMARTS().graph_to_smarts(G)
        >>> smarts
        '[C:1](-[O:2])'
        """
        bond_sym: Dict[float, str] = {1: "-", 1.5: ":", 2: "=", 3: "#"}

        def bracket(node: Any, data: Dict[str, Any]) -> str:
            map_num = node
            if data.get("constraint"):
                core = ",".join(data["constraint"])
            else:
                core = data["label"]
            return f"[{core}:{map_num}]"

        def choose_root(sub: nx.Graph) -> Any:
            real_nodes = [
                n
                for n, d in sub.nodes(data=True)
                if d.get("label") not in self.placeholder_labels
            ]
            if real_nodes:
                return max(real_nodes, key=sub.degree)
            return min(sub.nodes)

        def rec(node: Any, parent: Any, sub: nx.Graph, visited: Set[Any]) -> str:
            visited.add(node)
            s = bracket(node, sub.nodes[node])
            for nbr in sub.neighbors(node):
                if nbr == parent:
                    continue
                order = sub[node][nbr].get("order", 1)
                bond = bond_sym.get(order, "-")
                s += f"({bond}{rec(nbr, node, sub, visited)})"
            return s

        frags: List[str] = []
        for comp in nx.connected_components(G):
            subgraph = G.subgraph(comp)
            root = choose_root(subgraph)
            frags.append(rec(root, None, subgraph, set()))

        smarts: str = ".".join(frags)

        if self.validate:
            try:
                if not Chem.MolFromSmarts(smarts):
                    raise ValueError("RDKit could not parse generated SMARTS.")
            except ImportError:
                pass

        return smarts

    def graphs_to_rxn_smarts(self, reactant: nx.Graph, product: nx.Graph) -> str:
        """
        Construct a reaction SMARTS string from reactant and product graphs.

        :param reactant: Reactant NetworkX graph.
        :type reactant: nx.Graph
        :param product: Product NetworkX graph.
        :type product: nx.Graph
        :returns: Reaction SMARTS in the form 'reactants>>products'.
        :rtype: str
        :raises ValueError: If RDKit fails to parse the reaction SMARTS when validate=True.

        :Example:
        >>> G1 = nx.Graph()
        >>> G1.add_node(1, label='C', constraint=None)
        >>> G1.add_node(2, label='O', constraint=None)
        >>> G1.add_edge(1, 2, order=1)
        >>> G2 = nx.Graph()
        >>> G2.add_node(1, label='C', constraint=None)
        >>> G2.add_node(2, label='O', constraint=None)
        >>> G2.add_edge(1, 2, order=2)
        >>> rxn = GraphToSMARTS().graphs_to_rxn_smarts(G1, G2)
        >>> rxn
        '[C:1](-[O:2])>>[C:1]([O:2])='
        """
        sm_r: str = self.graph_to_smarts(reactant)
        sm_p: str = self.graph_to_smarts(product)
        rxn: str = f"{sm_r}>>{sm_p}"

        if self.validate:
            try:
                if not AllChem.ReactionFromSmarts(rxn):
                    raise ValueError("RDKit could not parse generated reaction SMARTS.")
            except ImportError:
                pass

        return rxn

    def __repr__(self) -> str:
        """
        Return an unambiguous representation of the converter instance.

        :returns: String showing placeholder labels and validation setting.
        :rtype: str
        """
        return (
            f"<GraphToSMARTS placeholder_labels={self.placeholder_labels}, "
            f"validate={self.validate}>"
        )

    def help(self) -> str:
        """
        Provide usage information for GraphToSMARTS.

        :returns: Multi-line help string describing available methods.
        :rtype: str

        :Example:
        >>> print(GraphToSMARTS().help())  # doctest:+NORMALIZE_WHITESPACE
        GraphToSMARTS(placeholder_labels=None, validate=True)
         - Use .graph_to_smarts(G) for a single graph
         - Use .graphs_to_rxn_smarts(G_react, G_prod) for reaction SMARTS
        """
        return (
            "GraphToSMARTS(placeholder_labels=None, validate=True)\n"
            " - Use .graph_to_smarts(G) for a single graph\n"
            " - Use .graphs_to_rxn_smarts(G_react, G_prod) for reaction SMARTS\n"
            "Node attributes expected:\n"
            "  label: str (e.g., 'C', '_R', 'H+')\n"
            "  constraint: Optional[List[str]] for placeholders\n"
            "Edge attribute:\n"
            "  order: float (1, 1.5, 2, 3) mapped to '-', ':', '=', '#'\n"
        )
