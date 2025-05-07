import re
from rdkit import Chem
from typing import Optional
from synkit.IO.chem_converter import smart_to_gml
from synkit.Rule.Modify.strip_rule import strip_context


class MoleculeRule:
    """
    A class for generating molecule rules, atom-mapped SMILES, and GML representations from SMILES strings.
    """

    def __init__(self) -> None:
        """
        Initializes the MoleculeRule object.
        """
        pass

    @staticmethod
    def remove_edges_from_left_right(input_str: str) -> str:
        """
        Remove all contents from the 'left' and 'right' sections of a chemical rule description.

        Parameters:
        - input_str (str): The string representation of the rule.

        Returns:
        - str: The modified string with cleared 'left' and 'right' sections.
        """
        # Pattern to match 'left [' to the matching ']'
        left_pattern = r"(left \[)(.*?)(^\s*\])"
        # Pattern to match 'right [' to the matching ']'
        right_pattern = r"(right \[)(.*?)(^\s*\])"

        # Replace contents within 'left [' and 'right [' sections using non-greedy matching
        # Multiline mode to handle newlines and match start of lines with '^'
        input_str = re.sub(
            left_pattern, r"\1\n    \3", input_str, flags=re.DOTALL | re.MULTILINE
        )
        input_str = re.sub(
            right_pattern, r"\1\n    \3", input_str, flags=re.DOTALL | re.MULTILINE
        )

        return input_str

    @staticmethod
    def generate_atom_map(smiles: str) -> Optional[str]:
        """
        Generate atom-mapped SMILES by assigning unique map numbers to each atom in the molecule.

        Parameters:
        - smiles (str): The SMILES string representing the molecule.

        Returns:
        - Optional[str]: The atom-mapped SMILES string, or None if the SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None  # Invalid SMILES

        # Assign atom map numbers (1-based index)
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(idx + 1)

        # Return the SMILES with atom map
        return Chem.MolToSmiles(mol, canonical=True)

    @staticmethod
    def generate_molecule_smart(smiles: str) -> Optional[str]:
        """
        Generate a SMARTS-like string from atom-mapped SMILES.

        Parameters:
        - smiles (str): The SMILES string representing the molecule.

        Returns:
        - Optional[str]: The SMARTS-like string derived from atom-mapped SMILES, or None if the SMILES is invalid.
        """
        atom_map_smiles = MoleculeRule.generate_atom_map(smiles)
        if atom_map_smiles is None:
            return None  # Invalid SMILES

        # Return the SMARTS-like string
        return f"{atom_map_smiles}>>{atom_map_smiles}"

    def generate_molecule_rule(
        self,
        smiles: str,
        name: str = "molecule",
        explicit_hydrogen: bool = True,
        sanitize: bool = True,
    ) -> Optional[str]:
        """
        Generate a GML representation of the molecule rule from SMILES.

        Parameters:
        - smiles (str): The SMILES string representing the molecule.
        - name (str, optional): The rule name used in GML generation. Defaults to 'molecule'.
        - explicit_hydrogen (bool, optional): Whether to include explicit hydrogen atoms in GML. Defaults to True.
        - sanitize (bool, optional): Whether to sanitize the molecule before conversion. Defaults to True.

        Returns:
        - Optional[str]: The GML representation of the molecule rule, or None if invalid.
        """
        rsmi = self.generate_molecule_smart(smiles)
        if rsmi is None:
            return None  # Invalid SMARTS string
        # Return the GML representation
        gml = smart_to_gml(
            rsmi,
            core=False,
            sanitize=sanitize,
            explicit_hydrogen=explicit_hydrogen,
            rule_name=name,
        )
        gml = strip_context(gml, False)
        return gml
