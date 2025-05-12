from rdkit import Chem
from rdkit.Chem import rdChemReactions
from typing import List, Tuple, Optional


def remove_explicit_H_from_rsmi(rsmi: str) -> str:
    """
    Remove explicit [H:...] atoms from a reaction SMILES with atom-atom mapping.
    Keeps atom mapping intact for non-hydrogen atoms and returns a simplified reaction SMILES.

    Args:
        rsmi (str): Atom-mapped reaction SMILES with explicit hydrogens.

    Returns:
        str: Reaction SMILES with implicit hydrogens and AAM preserved.
    """
    rxn = rdChemReactions.ReactionFromSmarts(rsmi, useSmiles=True)

    def cleaned_smiles(mols):
        return ".".join(
            Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True) for mol in mols
        )

    reactant_smiles = cleaned_smiles(rxn.GetReactants())
    product_smiles = cleaned_smiles(rxn.GetProducts())

    return f"{reactant_smiles}>>{product_smiles}"


def remove_common_reagents(reaction_smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Removes reagents that appear on both sides of a reaction SMILES string.

    Parameters:
    - reaction_smiles (str): The reaction in SMILES format.

    Returns:
    - Tuple[Optional[str], Optional[str]]: A tuple containing the cleaned reaction SMILES
    and a list of common reagents removed. If no common reagents are found, the reaction
    is returned unchanged and the second element of the tuple is `None`.
    """
    reactants, products = reaction_smiles.split(">>")
    reactant_list = reactants.split(".")
    product_list = products.split(".")
    common_reagents = set(reactant_list) & set(product_list)

    filtered_reactants = [r for r in reactant_list if r not in common_reagents]
    filtered_products = [p for p in product_list if p not in common_reagents]
    cleaned_reaction_smiles = (
        ".".join(filtered_reactants) + ">>" + ".".join(filtered_products)
    )

    return cleaned_reaction_smiles


def remove_duplicates(input_list: List[str]) -> List[str]:
    """
    Removes duplicate strings from a list, retaining only the first occurrence of each string.

    Parameters:
    - input_list (List[str]): A list of strings potentially containing duplicates.

    Returns:
    - List[str]: A list of strings with duplicates removed.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def reverse_reaction(rsmi: str) -> str:
    """
    Reverses the direction of a reaction SMILES string.

    Parameters:
    - rsmi (str): The reaction SMILES string to reverse.

    Returns:
    - str: The reversed reaction SMILES string.
    """
    reactants, products = rsmi.split(">>")
    return f"{products}>>{reactants}"


def merge_reaction(rsmi_1: str, rsmi_2: str) -> str:
    """
    Merges two reaction SMILES strings into a single reaction.

    Parameters:
    - rsmi_1 (str): The first reaction SMILES string.
    - rsmi_2 (str): The second reaction SMILES string.

    Returns:
    - str: A new reaction SMILES string combining both input reactions.
    """
    try:
        r1, p1 = rsmi_1.split(">>")
        r2, p2 = rsmi_2.split(">>")
    except ValueError:
        return None  # Returns None if there's a problem with splitting (e.g., no '>>')

    # Check if any part of the reaction is empty, which could be problematic for a meaningful merge.
    if any(len(part.strip()) == 0 for part in (r1, p1, r2, p2)):
        return None

    return f"{r1}.{r2}>>{p1}.{p2}"


def find_longest_fragment(input_list: List[str]) -> str:
    """
    Finds the longest string in a list of strings.

    Parameters:
    - input_list (List[str]): A list of strings from which the longest string is to be found.

    Returns:
    - str: The longest string found in the input list.
    """
    if len(input_list) == 0:
        return None
    longest_fragment = max(input_list, key=len)
    return longest_fragment
