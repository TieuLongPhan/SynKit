from rdkit import Chem
from typing import List, Optional, Tuple


class Standardize:
    def __init__(self) -> None:
        pass

    @staticmethod
    def remove_atom_mapping(reaction_smiles: str, symbol: str = ">>") -> str:
        """
        Removes atom mappings from both reactants and products in a reaction SMILES.

        Parameters:
        - reaction_smiles (str): A reaction SMILES string with atom mappings.
        - symbol (str): The symbol that separates reactants and products.
        Default is '>>'.

        Returns:
        - str: The reaction SMILES with atom mappings removed.
        """
        # Split the reaction SMILES into reactants and products
        parts = reaction_smiles.split(symbol)
        if len(parts) != 2:
            raise ValueError(
                "Invalid reaction SMILES format."
                + " Expected format: 'reactants>>products'."
            )

        def clean_smiles(smiles: str) -> str:
            mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to an RDKit mol object
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)  # Remove atom mapping
            return Chem.MolToSmiles(mol, True)  # Convert mol back to SMILES

        # Apply the cleaning function to both reactants and products
        reactants_clean = clean_smiles(parts[0])
        products_clean = clean_smiles(parts[1])

        # Combine the cleaned reactants and products back into a reaction SMILES
        return f"{reactants_clean}{symbol}{products_clean}"

    @staticmethod
    def filter_valid_molecules(smiles_list: List[str]) -> List[Chem.Mol]:
        """
        Filters and returns valid RDKit molecule objects from a list of SMILES strings.

        Parameters:
        - smiles_list (List[str]): A list of SMILES strings.

        Returns:
        - List[Chem.Mol]: A list of valid RDKit molecule objects.
        """
        valid_molecules = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_molecules.append(mol)
        return valid_molecules

    @staticmethod
    def standardize_rsmi(rsmi: str, stereo: bool = False) -> Optional[str]:
        """
        Standardizes a reaction SMILES (rSMI) by:
        - Ensuring all reactants and products are valid molecules.
        - Sorting the SMILES strings of reactants and products in ascending order.
        - Optionally considering stereochemistry.

        Parameters:
        - rsmi (str): The reaction SMILES string to be standardized.
        - stereo (bool): If True, stereochemical information is included in the SMILES.

        Returns:
        - Optional[str]: The standardized reaction SMILES,
        or None if no valid molecules are found.
        """
        try:
            reactants, products = rsmi.split(">>")
        except ValueError:
            raise ValueError(
                "Invalid reaction SMILES format."
                + " Expected format: 'reactants>>products'."
            )

        reactant_molecules = Standardize.filter_valid_molecules(reactants.split("."))
        product_molecules = Standardize.filter_valid_molecules(products.split("."))

        if not reactant_molecules or not product_molecules:
            return None

        standardized_reactants = ".".join(
            sorted(
                Chem.MolToSmiles(mol, isomericSmiles=stereo)
                for mol in reactant_molecules
            )
        )
        standardized_products = ".".join(
            sorted(
                Chem.MolToSmiles(mol, isomericSmiles=stereo)
                for mol in product_molecules
            )
        )

        return f"{standardized_reactants}>>{standardized_products}"

    def fit(
        self, rsmi: str, remove_aam: bool = True, ignore_stereo: bool = True
    ) -> Optional[str]:
        """
        Fits the reaction SMILES by removing atom mappings and standardizing the reaction.

        Parameters:
        - rsmi (str): The reaction SMILES string to be processed.
        - remove_aam (bool): If True, atom mappings are removed from the reaction SMILES.
        - ignore_stereo (bool): If True, stereochemistry is ignored in the SMILES.

        Returns:
        - Optional[str]: The processed reaction SMILES, or None if the
        standardization fails.
        """
        if remove_aam:
            rsmi = self.remove_atom_mapping(rsmi)

        rsmi = self.standardize_rsmi(rsmi, not ignore_stereo)
        rsmi = rsmi.replace("[HH]", "[H][H]")
        return rsmi

    @staticmethod
    def categorize_reactions(
        reactions: List[str], target_reaction: str
    ) -> Tuple[List[str], List[str]]:
        """
        Sorts a list of reaction SMILES strings into two groups based on
        their match with a specified target reaction. The categorization process
        distinguishes between reactions that align with the target reaction
        and those that do not.

        Parameters:
        - reactions (List[str]): The array of reaction SMILES strings to be categorized.
        - target_reaction (str): The SMILES string of the target reaction
                                    used as the benchmark for categorization.

        Returns:
        - Tuple[List[str], List[str]]: A pair of lists, where the first contains
                                    reactions matching the target and the second
                                    comprises non-matching reactions.
        """
        match, not_match = [], []
        target_reaction = Standardize.standardize_rsmi(target_reaction, stereo=False)
        for reaction_smiles in reactions:
            if reaction_smiles == target_reaction:
                match.append(reaction_smiles)
            else:
                not_match.append(reaction_smiles)
        return match, not_match
