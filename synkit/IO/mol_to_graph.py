from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from typing import Any, Dict, Optional
import random
from synutility.SynIO.debug import setup_logging

logger = setup_logging()


class MolToGraph:
    """
    A class for converting molecules from SMILES strings to graph representations using
    RDKit and NetworkX. It supports creating both lightweight and detailed
    graph representations with customizable atom and bond attributes,
    allowing for exclusion of atoms without atom mapping numbers.
    """

    def __init__(self) -> None:
        """
        Initialize the MolToGraph class.
        """
        pass

    @staticmethod
    def add_partial_charges(mol: Chem.Mol) -> None:
        """
        Computes and assigns Gasteiger partial charges to each atom in the given molecule.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            logger.error(f"Error computing Gasteiger charges: {e}")

    @staticmethod
    def get_stereochemistry(atom: Chem.Atom) -> str:
        """
        Determines the stereochemistry (R/S configuration) of a given atom.

        Parameters:
        - atom (Chem.Atom): An RDKit atom object.

        Returns:
        - str: The stereochemistry ('R', 'S', or 'N' for non-chiral).
        """
        chiral_tag = atom.GetChiralTag()
        return (
            "S"
            if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            else "R" if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else "N"
        )

    @staticmethod
    def get_bond_stereochemistry(bond: Chem.Bond) -> str:
        """
        Determines the stereochemistry (E/Z configuration) of a given bond.

        Parameters:
        - bond (Chem.Bond): An RDKit bond object.

        Returns:
        - str: The stereochemistry ('E', 'Z', or 'N' for non-stereospecific
        or non-double bonds).
        """
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            return "N"
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREOE:
            return "E"
        elif stereo == Chem.BondStereo.STEREOZ:
            return "Z"
        return "N"

    @staticmethod
    def has_atom_mapping(mol: Chem.Mol) -> bool:
        """
        Check if the given molecule has any atom mapping numbers.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.

        Returns:
        - bool: True if any atom in the molecule has a mapping number, False otherwise.
        """
        return any(atom.HasProp("molAtomMapNumber") for atom in mol.GetAtoms())

    @staticmethod
    def random_atom_mapping(mol: Chem.Mol) -> Chem.Mol:
        """
        Assigns a random atom mapping number to each atom in the given molecule.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.

        Returns:
        - Chem.Mol: The RDKit molecule object with random atom mapping numbers assigned.
        """
        atom_indices = list(range(1, mol.GetNumAtoms() + 1))
        random.shuffle(atom_indices)
        for atom, idx in zip(mol.GetAtoms(), atom_indices):
            atom.SetProp("molAtomMapNumber", str(idx))
        return mol

    @classmethod
    def mol_to_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: Optional[bool] = False,
        light_weight: Optional[bool] = False,
        use_index_as_atom_map: Optional[bool] = False,
    ) -> nx.Graph:
        """
        Converts an RDKit molecule object to a NetworkX graph with specified atom and bond
        attributes. Optionally excludes atoms without atom mapping numbers
        if drop_non_aam is True.

        Parameters:
        - mol (Chem.Mol): An RDKit molecule object.
        - drop_non_aam (bool, optional): If True, nodes without atom mapping numbers will
          be dropped. This option is useful for focusing on labeled parts of a molecule.
        - light_weight (bool, optional): If True, creates a graph with minimal attributes.
          This option is useful for reducing memory footprint or simplifying the graph.
        - use_index_as_atom_map (bool, optional): If True, uses the index of atoms as
        atom map numbers, otherwise uses existing atom map numbers or indices if not set.

        Raises:
        - ValueError: If `drop_non_aam` and `use_index_as_atom_map` are not both True or
        both False.

        Returns:
        - nx.Graph: A NetworkX graph representing the molecule.
        """

        if drop_non_aam and not use_index_as_atom_map:
            raise ValueError(
                "drop_non_aam and use_index_as_atom_map must be both False or both True."
            )

        if light_weight:
            return cls._create_light_weight_graph(
                mol, drop_non_aam, use_index_as_atom_map
            )
        else:
            return cls._create_detailed_graph(mol, drop_non_aam, use_index_as_atom_map)

    @classmethod
    def _create_light_weight_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> nx.Graph:
        graph = nx.Graph()

        for atom in mol.GetAtoms():
            if use_index_as_atom_map:
                # Use the atom map number if present; otherwise, use index + 1
                atom_id = (
                    atom.GetAtomMapNum()
                    if atom.GetAtomMapNum() != 0
                    else atom.GetIdx() + 1
                )
            else:
                # Always use index + 1
                atom_id = atom.GetIdx() + 1

            if drop_non_aam and atom.GetAtomMapNum() == 0:
                continue  # Skip atoms without atom map numbers if drop_non_aam is True

            graph.add_node(
                atom_id,
                element=atom.GetSymbol(),  # Store atom's element symbol
                aromatic=atom.GetIsAromatic(),
                hcount=atom.GetTotalNumHs(),
                charge=atom.GetFormalCharge(),
                neighbors=sorted(
                    neighbor.GetSymbol() for neighbor in atom.GetNeighbors()
                ),
                atom_map=atom.GetAtomMapNum(),
            )

            # Handle edges based on atom IDs and consistency checks
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if use_index_as_atom_map:
                    # Use the atom map number if present; otherwise, use index + 1
                    neighbor_id = (
                        neighbor.GetAtomMapNum()
                        if neighbor.GetAtomMapNum() != 0
                        else neighbor.GetIdx() + 1
                    )
                else:
                    # Always use index + 1 for the neighbor
                    neighbor_id = neighbor.GetIdx() + 1

                if not drop_non_aam or neighbor.GetAtomMapNum() != 0:
                    graph.add_edge(
                        atom_id, neighbor_id, order=bond.GetBondTypeAsDouble()
                    )

        return graph

    @classmethod
    def _create_detailed_graph(
        cls,
        mol: Chem.Mol,
        drop_non_aam: bool = True,
        use_index_as_atom_map: bool = True,
    ) -> nx.Graph:
        cls.add_partial_charges(mol)  # Compute charges if not already present
        graph = nx.Graph()
        index_to_id = {}

        for atom in mol.GetAtoms():
            if use_index_as_atom_map:
                # Use the atom map number if present; otherwise, use index + 1
                atom_id = (
                    atom.GetAtomMapNum()
                    if atom.GetAtomMapNum() != 0
                    else atom.GetIdx() + 1
                )
            else:
                # Always use index + 1
                atom_id = atom.GetIdx() + 1

            if drop_non_aam and atom.GetAtomMapNum() == 0:
                continue  # Skip atoms without atom map numbers if drop_non_aam is True

            props = cls._gather_atom_properties(atom)
            index_to_id[atom.GetIdx()] = atom_id
            graph.add_node(atom_id, **props)

        for bond in mol.GetBonds():
            begin_atom_id = index_to_id.get(bond.GetBeginAtomIdx())
            end_atom_id = index_to_id.get(bond.GetEndAtomIdx())

            if begin_atom_id and end_atom_id:
                # Apply consistent ID handling for edges
                graph.add_edge(
                    begin_atom_id, end_atom_id, **cls._gather_bond_properties(bond)
                )

        return graph

    @staticmethod
    def _gather_atom_properties(atom: Chem.Atom) -> Dict[str, Any]:
        """Collect all relevant properties from an atom to use
        as graph node attributes."""
        gasteiger_charge = (
            round(float(atom.GetProp("_GasteigerCharge")), 3)
            if atom.HasProp("_GasteigerCharge")
            else 0.0
        )
        return {
            "charge": atom.GetFormalCharge(),
            "hcount": atom.GetTotalNumHs(),
            "aromatic": atom.GetIsAromatic(),
            "element": atom.GetSymbol(),
            "atom_map": atom.GetAtomMapNum(),
            "isomer": MolToGraph.get_stereochemistry(atom),
            "partial_charge": gasteiger_charge,
            "hybridization": str(atom.GetHybridization()),
            "in_ring": atom.IsInRing(),
            "implicit_hcount": atom.GetNumImplicitHs(),
            "neighbors": sorted(
                neighbor.GetSymbol() for neighbor in atom.GetNeighbors()
            ),
        }

    @staticmethod
    def _gather_bond_properties(bond: Chem.Bond) -> Dict[str, Any]:
        """Collect all relevant properties from a bond to use as graph edge attributes."""
        return {
            "order": bond.GetBondTypeAsDouble(),
            "ez_isomer": MolToGraph.get_bond_stereochemistry(bond),
            "bond_type": str(bond.GetBondType()),
            "conjugated": bond.GetIsConjugated(),
            "in_ring": bond.IsInRing(),
        }
