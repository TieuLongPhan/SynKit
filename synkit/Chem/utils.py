from rdkit import Chem
from typing import List, Union
from synkit.IO.debug import setup_logging

logger = setup_logging()


def count_carbons(smiles: str) -> int:
    """ "
    Counts the number of carbon atoms in a molecule given a SMILES string.

    Parameters:
    - smiles (str): SMILES representation of the molecule.

    Returns:
    - int: Number of carbon atoms in the molecule if the SMILES string is valid.
    - str: Error message indicating an invalid SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
        return carbon_count
    else:
        return "Invalid SMILES string"


def get_max_fragment(smiles: Union[str, List[str]]) -> str:
    """
    Extracts and returns the SMILES string of the largest fragment from a SMILES string or a list of SMILES strings
    of a compound that may contain multiple fragments. This function determines the largest fragment based on the
    number of atoms.

    Parameters:
    - smiles (Union[str, List[str]]): A single SMILES string or a list of SMILES strings containing potentially
    multiple fragments.

    Returns:
    - str: SMILES string of the largest fragment based on the number of atoms. Returns an empty string if no valid
    fragments can be processed.

    Examples:
    - get_max_fragment("C.CC.CCC") returns "CCC"
    - get_max_fragment(["C.CC", "CCC.C"]) returns "CCC"
    """
    if isinstance(smiles, str):
        fragments = smiles.split(".")
    elif isinstance(smiles, list):
        fragments = [frag for s in smiles for frag in s.split(".")]
    else:
        return ""

    molecules = [Chem.MolFromSmiles(fragment) for fragment in fragments if fragment]
    if not molecules:
        return ""  # Return empty string if no valid molecules are found

    max_mol = max(
        molecules, key=lambda mol: mol.GetNumAtoms() if mol else 0, default=None
    )
    return Chem.MolToSmiles(max_mol) if max_mol else ""


def filter_smiles(smiles_list: List[str], target_smiles: str) -> List[str]:
    """
    Filters a list of SMILES strings to include only those that contain carbon atoms and are not identical
    to a given target SMILES string.

    Parameters:
    - smiles_list (List[str]): A list of SMILES strings to be filtered.
    - target_smiles (str): The target SMILES string to exclude from the output.

    Returns:
    - List[str]: A list of SMILES strings that contain carbon and are not the same as the target SMILES.
    """
    filtered_smiles = []
    # Convert target SMILES to a molecule and standardize it for comparison
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_canonical = Chem.MolToSmiles(target_mol) if target_mol else None

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Check if the molecule contains carbon
            if any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
                # Standardize the SMILES for comparison
                canonical_smiles = Chem.MolToSmiles(mol)
                # Check that the SMILES is not the same as the target SMILES
                if canonical_smiles != target_canonical:
                    filtered_smiles.append(smiles)

    return filtered_smiles


def remove_atom_mappings(mol: Chem.Mol) -> Chem.Mol:
    """
    Removes atom mapping numbers from a molecule by setting each atom's mapping number to zero.

    Parameters:
    - mol (Chem.Mol): The RDKit molecule object from which to remove atom mappings.

    Returns:
    - Chem.Mol: The same RDKit molecule object with atom mappings removed.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def get_sanitized_smiles(smiles_list: List[str]) -> List[str]:
    """
    Filters and returns a list of sanitizable SMILES strings from the provided list, with atom mappings removed
    and excluding any SMILES containing reaction indicators ('->').

    Parameters:
    - smiles_list (List[str]): A list of SMILES strings to be sanitized.

    Returns:
    - List[str]: A list of SMILES strings that can be successfully sanitized.
    """
    sanitized_smiles = []
    for smiles in smiles_list:
        if "->" in smiles:  # Skip SMILES with reaction indicators
            continue
        try:
            # Attempt to create a molecule from the SMILES string
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Remove atom mappings before sanitization
                mol = remove_atom_mappings(mol)

                # Attempt to sanitize the molecule
                Chem.SanitizeMol(mol)

                # If sanitization is successful, append the sanitized SMILES to the result list
                sanitized_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                sanitized_smiles = [get_max_fragment(sanitized_smiles)]
        except (Chem.rdchem.ChemicalReactionException, ValueError) as e:
            logger.error(e)
            continue

    return sanitized_smiles


def remove_duplicates(smiles_list: List[str]) -> List[str]:
    """
    Removes duplicate SMILES strings from a list, maintaining the order of
    first occurrences. Uses a set to track seen SMILES for efficiency.

    Parameters:
    - smiles_list (List[str]): A list of SMILES strings representing
    chemical reactions.

    Returns:
    - List[str]: A list with unique SMILES strings, preserving the original order.
    """
    seen = set()
    unique_smiles = [
        smiles for smiles in smiles_list if not (smiles in seen or seen.add(smiles))
    ]
    return unique_smiles


def process_smiles_list(smiles_list: List[str]) -> List[str]:
    """
    Processes a list of SMILES (Simplified Molecular Input Line Entry System) strings,
    splitting any entries that contain disconnected molecular components
    (indicated by a '.'), and returns a new list with each component as a separate entry.

    Parameters:
    - smiles_list (List[str]): A list of SMILES strings, where some entries may contain
                             disconnected components separated by dots.

    Returns:
    - List[str]: A new list of SMILES strings with all components separated. This list
               does not include any original strings that contained dots; instead, it
               includes their split components.
    """
    new_smiles_list = []  # Create a new list to store processed SMILES strings
    for smiles in smiles_list:
        if "." in smiles:
            # Split the SMILES string into components and extend the new list
            components = smiles.split(".")
            new_smiles_list.extend(components)
        else:
            # Add the unchanged SMILES string to the new list
            new_smiles_list.append(smiles)
    return new_smiles_list
