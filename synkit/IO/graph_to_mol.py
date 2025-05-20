import networkx as nx
from rdkit import Chem
from typing import Dict


class GraphToMol:
    """
    Converts a NetworkX graph representation of a molecule into an RDKit molecule object.

    This class reconstructs RDKit molecules from node and edge attributes in a graph,
    correctly interpreting atom types, charges, mapping numbers, bond orders, and optionally
    explicit hydrogen counts.

    :param node_attributes: Mapping of expected attribute names to node keys in the graph. For example,
        {"element": "element", "charge": "charge", "atom_map": "atom_map"}.
    :type node_attributes: Dict[str, str]
    :param edge_attributes: Mapping of expected attribute names to edge keys in the graph.
        For example, {"order": "order"}.
    :type edge_attributes: Dict[str, str]
    """

    def __init__(
        self,
        node_attributes: Dict[str, str] = {
            "element": "element",
            "charge": "charge",
            "atom_map": "atom_map",
        },
        edge_attributes: Dict[str, str] = {"order": "order"},
    ):
        """
        Initializes the GraphToMol object with mappings for node and edge attributes.

        :param node_attributes: Mapping from desired atom attribute names to graph node keys.
            E.g. {"element": "element", "charge": "charge", "atom_map": "atom_map"}
        :type node_attributes: Dict[str, str]
        :param edge_attributes: Mapping from desired bond attribute names to graph edge keys.
            E.g. {"order": "order"}
        :type edge_attributes: Dict[str, str]
        """
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes

    def graph_to_mol(
        self,
        graph: nx.Graph,
        ignore_bond_order: bool = False,
        sanitize: bool = True,
        use_h_count: bool = False,
    ) -> Chem.Mol:
        """
        Converts a NetworkX graph into an RDKit molecule.

        :param graph: The NetworkX graph representing the molecule.
        :type graph: nx.Graph
        :param ignore_bond_order: If True, all bonds are created as single bonds regardless of edge attributes.
            Defaults to False.
        :type ignore_bond_order: bool
        :param sanitize: If True, the resulting RDKit molecule will be sanitized after construction.
            Defaults to True.
        :type sanitize: bool
        :param use_h_count: If True, the 'hcount' attribute (if present) will be used to set explicit hydrogen counts
            on atoms. Defaults to False.
        :type use_h_count: bool

        :returns: An RDKit molecule constructed from the graph's nodes and edges.
        :rtype: Chem.Mol
        """
        mol = Chem.RWMol()
        node_to_idx: Dict[int, int] = {}

        for node, data in graph.nodes(data=True):
            element = data.get(self.node_attributes["element"], "C")
            charge = data.get(self.node_attributes["charge"], 0)
            atom_map = (
                data.get(self.node_attributes["atom_map"], 0)
                if "atom_map" in data.keys()
                else None
            )
            hcount = (
                data.get("hcount", 0)
                if use_h_count and "hcount" in data.keys()
                else None
            )

            atom = Chem.Atom(element)
            atom.SetFormalCharge(charge)
            if atom_map is not None:
                atom.SetAtomMapNum(atom_map)
            if hcount is not None:
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(hcount)

            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx

        for u, v, data in graph.edges(data=True):
            bond_order = (
                1
                if ignore_bond_order
                else abs(data.get(self.edge_attributes["order"], 1))
            )
            bond_type = self.get_bond_type_from_order(bond_order)
            mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

        if sanitize:
            Chem.SanitizeMol(mol)

        return mol

    @staticmethod
    def get_bond_type_from_order(order: float) -> Chem.BondType:
        """
        Converts a numerical bond order into the corresponding RDKit BondType.

        :param order: The numerical bond order (typically 1, 2, or 3).
        :type order: float
        :returns: The corresponding RDKit bond type (single, double, triple, or aromatic).
        :rtype: Chem.BondType
        """
        if order == 1:
            return Chem.BondType.SINGLE
        elif order == 2:
            return Chem.BondType.DOUBLE
        elif order == 3:
            return Chem.BondType.TRIPLE
        return Chem.BondType.AROMATIC
