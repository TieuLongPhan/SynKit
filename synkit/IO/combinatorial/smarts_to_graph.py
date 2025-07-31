import re
import networkx as nx
from typing import Optional, Set, Tuple, List, Dict

from rdkit import Chem


class SMARTSToGraph:
    """
    Convert SMARTS or reaction SMARTS strings into NetworkX graphs with full atom and constraint data.

    :param placeholder_labels: Optional set of labels to treat as placeholders (e.g., wildcard atoms).
    :type placeholder_labels: Optional[Set[str]]
    :raises: None
    """

    def __init__(self, placeholder_labels: Optional[Set[str]] = None) -> None:
        """
        Initialize a SMARTSToGraph converter.

        :param placeholder_labels: Set of placeholder labels used in SMARTS to identify wildcard positions.
                                   Defaults to {'_R', 'X', 'Y', 'Z'} if None.
        :type placeholder_labels: Optional[Set[str]]
        :returns: None
        :rtype: None
        """
        self.placeholder_labels: Set[str] = placeholder_labels or {"_R", "X", "Y", "Z"}

    @staticmethod
    def _safe_total_hs(atom: "Chem.Atom") -> int:
        """
        Compute the total number of hydrogens (explicit + implicit) for an RDKit Atom safely.

        :param atom: RDKit Atom instance whose hydrogen count is desired.
        :type atom: Chem.Atom
        :returns: Total hydrogen count for the atom.
        :rtype: int
        :raises: Exception if RDKit property cache update fails.

        :Example:
        >>> from rdkit import Chem
        >>> atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
        >>> SMARTSToGraph._safe_total_hs(atom)
        3
        """
        try:
            atom.UpdatePropertyCache(strict=False)
            return int(atom.GetTotalNumHs(includeExplicit=True))
        except Exception:
            return 0

    def smarts_to_graph(self, smarts: str) -> nx.Graph:
        """
        Parse a SMARTS string into a NetworkX graph representation, extracting wildcard constraints.

        :param smarts: SMARTS pattern to convert (e.g., '[C:1]C[O:2]').
        :type smarts: str
        :returns: NetworkX Graph with:
                  - node attributes: element (str), charge (int), hcount (int),
                    label (str), constraint (Optional[List[str]]), atom_map (int)
                  - edge attributes: order (float), standard_order (float)
        :rtype: nx.Graph
        :raises ImportError: If RDKit is not available.
        :raises ValueError: If the SMARTS string is invalid or atoms lack mapping numbers.

        :Example:
        >>> stg = SMARTSToGraph()
        >>> graph = stg.smarts_to_graph('[CH3:1]-[OH:2]')
        >>> graph.nodes[1]['element']
        'C'
        >>> graph.nodes[2]['element']
        'O'
        """
        if Chem is None:
            raise ImportError("RDKit is required for SMARTS parsing.")

        # Pre-scan SMARTS for wildcard constraint lists (e.g., [C,N:5])
        constraint_map: Dict[int, List[str]] = {}
        for match in re.finditer(r"\[([^:\]]+?):(\d+)\]", smarts):
            atom_expr, atom_idx = match.groups()
            idx = int(atom_idx)
            if "," in atom_expr:
                constraint_map[idx] = [s.strip() for s in atom_expr.split(",")]

        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            raise ValueError(f"Invalid SMARTS string: {smarts!r}")

        G = nx.Graph()
        idx_to_map: Dict[int, int] = {}

        # Add nodes with full atom data
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap == 0:
                raise ValueError(
                    "All atoms in SMARTS must have a mapping number (atom map)."
                )
            idx_to_map[atom.GetIdx()] = amap

            # Determine element, label, and constraints
            raw_label = atom.GetSymbol()
            if amap in constraint_map:
                constraint = constraint_map[amap]
                label = next(iter(self.placeholder_labels))
                element = "*"
            else:
                constraint = None
                label = raw_label
                element = "*" if raw_label in self.placeholder_labels else raw_label

            charge = atom.GetFormalCharge()
            hcount = self._safe_total_hs(atom)

            G.add_node(
                amap,
                element=element,
                charge=charge,
                hcount=hcount,
                label=label,
                constraint=constraint,
                atom_map=amap,
            )

        # Add edges with bond order data
        for bond in mol.GetBonds():
            u = idx_to_map[bond.GetBeginAtomIdx()]
            v = idx_to_map[bond.GetEndAtomIdx()]
            order = bond.GetBondTypeAsDouble()
            G.add_edge(u, v, order=order, standard_order=order)

        return G

    def rxn_smarts_to_graphs(self, rxn: str) -> Tuple[nx.Graph, nx.Graph]:
        """
        Split a reaction SMARTS into separate reactant and product graphs.

        :param rxn: Reaction SMARTS in the format 'reactants>>products'.
        :type rxn: str
        :returns: Tuple of (reactant_graph, product_graph).
        :rtype: Tuple[nx.Graph, nx.Graph]
        :raises ValueError: If the reaction SMARTS string does not contain '>>'.

        :Example:
        >>> stg = SMARTSToGraph()
        >>> react, prod = stg.rxn_smarts_to_graphs('[CH3:1]-[OH:2]>>[CH2:1]=[O:2]')
        >>> react.nodes
        [1, 2]
        >>> prod.nodes
        [1, 2]
        """
        if ">>" not in rxn:
            raise ValueError("Reaction SMARTS must contain '>>' separator.")
        lhs, rhs = rxn.split(">>", 1)
        return self.smarts_to_graph(lhs), self.smarts_to_graph(rhs)

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of this converter.

        :returns: Representation of the instance showing placeholder labels.
        :rtype: str
        """
        return f"<SMARTSToGraph placeholders={self.placeholder_labels}>"

    def describe(self) -> str:
        """
        Provide a usage summary for SMARTSToGraph.

        :returns: Multi-line string explaining available methods and usage.
        :rtype: str

        :Example:
        >>> print(SMARTSToGraph().describe())
        SMARTSToGraph(placeholder_labels=None)
         smarts_to_graph(smarts_str) -> Graph
         rxn_smarts_to_graphs(rxn_smarts) -> (Graph_react, Graph_prod)
        """
        return (
            "SMARTSToGraph(placeholder_labels=None)\n"
            " smarts_to_graph(smarts_str) -> Graph\n"
            " rxn_smarts_to_graphs(rxn_smarts) -> (Graph_react, Graph_prod)\n"
        )
