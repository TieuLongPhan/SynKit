import networkx as nx
from rdkit import Chem
from typing import Dict


class GraphToMol:
    """
    Converts a NetworkX graph representation of a molecule into an RDKit molecule object,
    considering specific node and edge attributes for the construction of the molecule.
    This includes handling different bond orders and optional hydrogen counts on nodes.
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

        Parameters:
        - node_attributes (Dict[str, str]): Mapping of attribute names to node keys in the graph.
        - edge_attributes (Dict[str, str]): Mapping of attribute names to edge keys in the graph.
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

        Parameters:
        - graph (nx.Graph): The molecule graph.
        - ignore_bond_order (bool): If True, all bonds are treated as single.
        - sanitize (bool): If True, attempts to sanitize the molecule.
        - use_h_count (bool): If True, adjusts hydrogen counts using the 'hcount' attribute.

        Returns:
        - Chem.Mol: An RDKit molecule object constructed from the graph.
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

        Parameters:
        - order (float): The bond order.

        Returns:
        - Chem.BondType: The corresponding RDKit bond type for the given order.
        """
        if order == 1:
            return Chem.BondType.SINGLE
        elif order == 2:
            return Chem.BondType.DOUBLE
        elif order == 3:
            return Chem.BondType.TRIPLE
        return Chem.BondType.AROMATIC
