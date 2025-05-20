from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from typing import Any, Dict, List, Optional
import random
from synkit.IO.debug import setup_logging

logger = setup_logging()


class MolToGraph:
    """RDKit → NetworkX helper with **attribute selection**

    This class converts RDKit molecules into NetworkX graphs. The original
    conversion methods (`_create_light_weight_graph`, `_create_detailed_graph`,
    and `mol_to_graph`) are preserved for full-featured graph creation. The
    new `transform` method builds a NetworkX graph including only a specified
    subset of node and edge attributes.

    :param node_attrs: List of node attribute names to retain. If empty or None, all are included.
    :type node_attrs: List[str]
    :param edge_attrs: List of edge attribute names to retain. If empty or None, all are included.
    :type edge_attrs: List[str]
    """

    def __init__(
        self,
        node_attrs: Optional[List[str]] = [
            "element",
            "aromatic",
            "hcount",
            "charge",
            "neighbors",
            "atom_map",
        ],
        edge_attrs: Optional[List[str]] = ["order"],
    ) -> None:
        """
        Initialize the MolToGraph helper.

        :param node_attrs: Names of node attributes to keep when transforming.
        :type node_attrs: List[str]
        :param edge_attrs: Names of edge attributes to keep when transforming.
        :type edge_attrs: List[str]
        """
        self.node_attrs: List[str] = node_attrs or []
        self.edge_attrs: List[str] = edge_attrs or []

    def transform(
        self,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """
        Build a graph directly from a molecule, including only selected attributes.

        :param mol: The RDKit molecule to convert.
        :type mol: Chem.Mol
        :param drop_non_aam: If True, skips atoms without atom mapping numbers
        (requires use_index_as_atom_map=True). Defaults to False.
        :type drop_non_aam: bool
        :param use_index_as_atom_map: If True, uses atom mapping numbers as node IDs when present;
        otherwise uses atom index+1. Defaults to False.
        :type use_index_as_atom_map: bool
        :returns: A NetworkX graph containing only the specified node and edge attributes.
        :rtype: nx.Graph
        """
        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must both be True to drop unmapped atoms."
            )
        # Precompute partial charges for detailed properties
        AllChem.ComputeGasteigerCharges(mol)

        graph = nx.Graph()
        index_to_id: Dict[int, int] = {}

        # Nodes
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            atom_id = (
                atom_map
                if use_index_as_atom_map and atom_map != 0
                else atom.GetIdx() + 1
            )
            if drop_non_aam and atom_map == 0:
                continue
            props = self._gather_atom_properties(atom)
            if self.node_attrs:
                props = {k: v for k, v in props.items() if k in self.node_attrs}
            graph.add_node(atom_id, **props)
            index_to_id[atom.GetIdx()] = atom_id

        # Edges
        for bond in mol.GetBonds():
            begin = index_to_id.get(bond.GetBeginAtomIdx())
            end = index_to_id.get(bond.GetEndAtomIdx())
            if begin is None or end is None:
                continue
            bprops = self._gather_bond_properties(bond)
            if self.edge_attrs:
                bprops = {k: v for k, v in bprops.items() if k in self.edge_attrs}
            graph.add_edge(begin, end, **bprops)

        return graph

    @staticmethod
    def _gather_atom_properties(atom: Chem.Atom) -> Dict[str, Any]:
        """
        Collect the full set of atom attributes for graph nodes.

        :param atom: The RDKit Atom object.
        :type atom: Chem.Atom
        :returns: Dictionary of atom attribute names and values.
        :rtype: Dict[str, Any]
        """
        gcharge = (
            round(float(atom.GetProp("_GasteigerCharge")), 3)
            if atom.HasProp("_GasteigerCharge")
            else 0.0
        )
        return {
            "element": atom.GetSymbol(),
            "aromatic": atom.GetIsAromatic(),
            "hcount": atom.GetTotalNumHs(),
            "charge": atom.GetFormalCharge(),
            "radical": atom.GetNumRadicalElectrons(),
            "isomer": MolToGraph.get_stereochemistry(atom),
            "partial_charge": gcharge,
            "hybridization": str(atom.GetHybridization()),
            "in_ring": atom.IsInRing(),
            "implicit_hcount": atom.GetNumImplicitHs(),
            "neighbors": sorted(nb.GetSymbol() for nb in atom.GetNeighbors()),
            "atom_map": atom.GetAtomMapNum(),
        }

    @staticmethod
    def _gather_bond_properties(bond: Chem.Bond) -> Dict[str, Any]:
        """
        Collect the full set of bond attributes for graph edges.

        :param bond: The RDKit Bond object.
        :type bond: Chem.Bond
        :returns: Dictionary of bond attribute names and values.
        :rtype: Dict[str, Any]
        """
        return {
            "order": bond.GetBondTypeAsDouble(),
            "bond_type": str(bond.GetBondType()),
            "ez_isomer": MolToGraph.get_bond_stereochemistry(bond),
            "conjugated": bond.GetIsConjugated(),
            "in_ring": bond.IsInRing(),
        }

    @staticmethod
    def get_stereochemistry(atom: Chem.Atom) -> str:
        """
        Determine the stereochemistry (R/S) of a chiral atom.

        :param atom: The RDKit Atom object.
        :type atom: Chem.Atom
        :returns: 'R', 'S', or 'N' for non-chiral.
        :rtype: str
        """
        ch = atom.GetChiralTag()
        if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            return "S"
        if ch == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            return "R"
        return "N"

    @staticmethod
    def get_bond_stereochemistry(bond: Chem.Bond) -> str:
        """
        Determine the stereochemistry (E/Z) of a double bond.

        :param bond: The RDKit Bond object.
        :type bond: Chem.Bond
        :returns: 'E', 'Z', or 'N' for non-stereospecific or non-double bond.
        :rtype: str
        """
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            return "N"
        st = bond.GetStereo()
        if st == Chem.BondStereo.STEREOE:
            return "E"
        if st == Chem.BondStereo.STEREOZ:
            return "Z"
        return "N"

    @staticmethod
    def has_atom_mapping(mol: Chem.Mol) -> bool:
        """
        Check if any atom in the molecule has an atom mapping number.

        :param mol: The RDKit molecule.
        :type mol: Chem.Mol
        :returns: True if at least one atom has a mapping number.
        :rtype: bool
        """
        return any(atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms())

    @staticmethod
    def random_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """
        Assign random atom mapping numbers to all atoms in the molecule.

        :param mol: The RDKit molecule.
        :type mol: Chem.Mol
        :returns: The molecule with new random atom mapping numbers.
        :rtype: Chem.Mol
        """
        indices = list(range(1, mol.GetNumAtoms() + 1))
        random.shuffle(indices)
        for atom, idx in zip(mol.GetAtoms(), indices):
            atom.SetProp("molAtomMapNumber", str(idx))
        return mol

    @classmethod
    def mol_to_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        light_weight: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """
        Convert a molecule to a full-featured NetworkX graph.

        :param mol: The RDKit molecule to convert.
        :type mol: Chem.Mol
        :param drop_non_aam: If True, drop atoms without mapping numbers
        (requires use_index_as_atom_map=True). Defaults to False.
        :type drop_non_aam: bool
        :param light_weight: If True, create a lightweight graph with minimal attributes. Defaults to False.
        :type light_weight: bool
        :param use_index_as_atom_map: If True, prefer atom maps as node IDs. Defaults to False.
        :type use_index_as_atom_map: bool
        :returns: A NetworkX graph of the molecule with all attributes.
        :rtype: nx.Graph
        """
        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must be both False or both True."
            )
        if light_weight:
            return cls._create_light_weight_graph(
                mol, drop_non_aam, use_index_as_atom_map
            )
        return cls._create_detailed_graph(mol, drop_non_aam, use_index_as_atom_map)

    @classmethod
    def _create_light_weight_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        """
        Create a lightweight graph with basic atom and bond info.

        :param mol: The RDKit molecule.
        :type mol: Chem.Mol
        :param drop_non_aam: If True, skip atoms without mapping numbers. Defaults to False.
        :type drop_non_aam: bool
        :param use_index_as_atom_map: If True, use atom maps as node IDs when present. Defaults to False.
        :type use_index_as_atom_map: bool
        :returns: A NetworkX graph with minimal node/edge attributes.
        :rtype: nx.Graph
        """
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            atom_id = (
                atom_map
                if use_index_as_atom_map and atom_map != 0
                else atom.GetIdx() + 1
            )
            if drop_non_aam and atom_map == 0:
                continue
            graph.add_node(
                atom_id,
                element=atom.GetSymbol(),
                aromatic=atom.GetIsAromatic(),
                hcount=atom.GetTotalNumHs(),
                charge=atom.GetFormalCharge(),
                neighbors=sorted(nb.GetSymbol() for nb in atom.GetNeighbors()),
                atom_map=atom_map,
            )
            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtom(atom)
                nbr_id = (
                    nbr.GetAtomMapNum()
                    if use_index_as_atom_map and nbr.GetAtomMapNum() != 0
                    else nbr.GetIdx() + 1
                )
                if not drop_non_aam or nbr.GetAtomMapNum() != 0:
                    graph.add_edge(atom_id, nbr_id, order=bond.GetBondTypeAsDouble())
        return graph

    @classmethod
    def _create_detailed_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = True,
        use_index_as_atom_map: bool = True,
    ) -> nx.Graph:
        """
        Create a detailed graph with full atom and bond attributes.

        :param mol: The RDKit molecule.
        :type mol: Chem.Mol
        :param drop_non_aam: If True, skip atoms without mapping numbers. Defaults to True.
        :type drop_non_aam: bool
        :param use_index_as_atom_map: If True, use atom maps as node IDs when present. Defaults to True.
        :type use_index_as_atom_map: bool
        :returns: A NetworkX graph with full node/edge attributes.
        :rtype: nx.Graph
        """
        cls.add_partial_charges(mol)
        graph = nx.Graph()
        idx_map: Dict[int, int] = {}
        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            atom_id = (
                atom_map
                if use_index_as_atom_map and atom_map != 0
                else atom.GetIdx() + 1
            )
            if drop_non_aam and atom_map == 0:
                continue
            graph.add_node(atom_id, **cls._gather_atom_properties(atom))
            idx_map[atom.GetIdx()] = atom_id
        for bond in mol.GetBonds():
            b = idx_map.get(bond.GetBeginAtomIdx())
            e = idx_map.get(bond.GetEndAtomIdx())
            if b and e:
                graph.add_edge(b, e, **cls._gather_bond_properties(bond))
        return graph

    @staticmethod
    def add_partial_charges(mol: Chem.Mol) -> None:
        """
        Compute and assign Gasteiger charges to all atoms in the molecule.

        :param mol: The RDKit molecule.
        :type mol: Chem.Mol
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            logger.error(f"Error computing Gasteiger charges: {e}")
