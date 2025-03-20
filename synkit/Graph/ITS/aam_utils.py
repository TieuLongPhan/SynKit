import re
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from typing import Optional, List


def enumerate_tautomers(reaction_smiles: str) -> Optional[List[str]]:
    """
    Enumerates possible tautomers for reactants while canonicalizing the products in a
    reaction SMILES string. This function first splits the reaction SMILES string into
    reactants and products. It then generates all possible tautomers for the reactants and
    canonicalizes the product molecule. The function returns a list of reaction SMILES
    strings for each tautomer of the reactants combined with the canonical product.

    Parameters:
    - reaction_smiles (str): A SMILES string of the reaction formatted as
    'reactants>>products'.

    Returns:
    - List[str] | None: A list of SMILES strings for the reaction, with each string
    representing a different
    - tautomer of the reactants combined with the canonicalized products. Returns None if
    an error occurs or if invalid SMILES strings are provided.

    Raises:
    - ValueError: If the provided SMILES strings cannot be converted to molecule objects,
    indicating invalid input.
    """
    try:
        # Split the input reaction SMILES string into reactants and products
        reactants_smiles, products_smiles = reaction_smiles.split(">>")

        # Convert SMILES strings to molecule objects
        reactants_mol = Chem.MolFromSmiles(reactants_smiles)
        products_mol = Chem.MolFromSmiles(products_smiles)

        if reactants_mol is None or products_mol is None:
            raise ValueError(
                "Invalid SMILES string provided for reactants or products."
            )

        # Initialize tautomer enumerator

        enumerator = rdMolStandardize.TautomerEnumerator()

        # Enumerate tautomers for the reactants and canonicalize the products
        try:
            reactants_can = enumerator.Enumerate(reactants_mol)
        except Exception as e:
            print(f"An error occurred: {e}")
            reactants_can = [reactants_mol]
        products_can = products_mol

        # Convert molecule objects back to SMILES strings
        reactants_can_smiles = [Chem.MolToSmiles(i) for i in reactants_can]
        products_can_smiles = Chem.MolToSmiles(products_can)

        # Combine each reactant tautomer with the canonical product in SMILES format
        rsmi_list = [i + ">>" + products_can_smiles for i in reactants_can_smiles]
        if len(rsmi_list) == 0:
            return [reaction_smiles]
        else:
            # rsmi_list.remove(reaction_smiles)
            rsmi_list.insert(0, reaction_smiles)
            return rsmi_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return [reaction_smiles]


def mapping_success_rate(list_mapping_data):
    """
    Calculate the success rate of entries containing atom mappings in a list of data
    strings.

    Parameters:
    - list_mapping_in_data (list of str): List containing strings to be searched for atom
    mappings.

    Returns:
    - float: The success rate of finding atom mappings in the list as a percentage.

    Raises:
    - ValueError: If the input list is empty.
    """
    atom_map_pattern = re.compile(r":\d+")
    if not list_mapping_data:
        raise ValueError("The input list is empty, cannot calculate success rate.")

    success = sum(
        1 for entry in list_mapping_data if re.search(atom_map_pattern, entry)
    )
    rate = 100 * (success / len(list_mapping_data))

    return round(rate, 2)
