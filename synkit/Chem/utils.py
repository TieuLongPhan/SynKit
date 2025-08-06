from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdChemReactions
import re
from typing import List, Optional, Tuple, Union


def clean_radical_rsmi(rsmi: str) -> str:
    """
    Load each side of a reaction SMILES (rSMI) into RDKit, split into disconnected fragments,
    remove any fragment that contains an atom with nonzero radical electrons,
    then reassemble back into a cleaned reaction SMILES.

    :param rsmi: Reaction SMILES string, e.g.
                 'A>>B.C'
    :type rsmi: str
    :returns: Cleaned reaction SMILES with radical-containing fragments removed.
    :rtype: str

    Example:
    >>> clean_radical_rsmi(
    ...   'COC(=O)C(CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O'
    ...   '>>COC(=O)C(CCCCNC(=O)OCc1ccccc1)NC(N)=O.COc1c[c]c(O)c(C(C)(C)C)c1'
    ... )
    'COC(=O)C(CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O'
    '>>COC(=O)C(CCCCNC(=O)OCc1ccccc1)NC(N)=O'
    """
    if ">>" not in rsmi:
        return rsmi

    def _clean_side(side: str) -> str:
        mol = Chem.MolFromSmiles(side)
        if mol is None:
            return ""
        frags = Chem.GetMolFrags(mol, asMols=True)
        kept = []
        for frag in frags:
            if any(atom.GetNumRadicalElectrons() > 0 for atom in frag.GetAtoms()):
                continue
            kept.append(Chem.MolToSmiles(frag, isomericSmiles=True))
        return ".".join(kept)

    reac, prod = rsmi.split(">>", 1)
    return f"{_clean_side(reac)}>>{_clean_side(prod)}"


def enumerate_tautomers(reaction_smiles: str) -> Optional[List[str]]:
    """Enumerate possible tautomers of reactants while canonicalizing products.

    :param reaction_smiles: Reaction SMILES in 'reactants>>products'
        format.
    :type reaction_smiles: str
    :returns: List of reaction SMILES for each reactant tautomer
        (including the original), or None on error.
    :rtype: Optional[List[str]]
    :raises ValueError: If reactant or product SMILES are invalid.
    """
    try:
        reactants_smiles, products_smiles = reaction_smiles.split(">>")
        reactants_mol = Chem.MolFromSmiles(reactants_smiles)
        products_mol = Chem.MolFromSmiles(products_smiles)
        if reactants_mol is None or products_mol is None:
            raise ValueError("Invalid reactant or product SMILES.")
        enumerator = rdMolStandardize.TautomerEnumerator()
        reactants_tautos = enumerator.Enumerate(reactants_mol) or [reactants_mol]
        prod_can = Chem.MolToSmiles(products_mol, canonical=True)
        rsmi_list = [Chem.MolToSmiles(m) + ">>" + prod_can for m in reactants_tautos]
        rsmi_list.insert(0, reaction_smiles)
        return rsmi_list
    except ValueError:
        raise
    except Exception:
        return None


def mapping_success_rate(list_mapping_data: List[str]) -> float:
    """Calculate percentage of entries containing atom‑mapping annotations.

    :param list_mapping_data: List of strings to search for mappings.
    :type list_mapping_data: List[str]
    :returns: Percentage of entries containing `:<digits>` patterns,
        rounded to two decimals.
    :rtype: float
    :raises ValueError: If input list is empty.
    """
    if not list_mapping_data:
        raise ValueError("The input list is empty, cannot calculate success rate.")
    pattern = re.compile(r":\d+")
    success = sum(1 for entry in list_mapping_data if pattern.search(entry))
    return round(100 * success / len(list_mapping_data), 2)


def count_carbons(smiles: str) -> int:
    """Count the number of carbon atoms in a molecule.

    :param smiles: SMILES string of the molecule.
    :type smiles: str
    :returns: Number of carbon atoms, or raises ValueError if SMILES
        invalid.
    :rtype: int
    :raises ValueError: If the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")


def get_max_fragment(smiles: Union[str, List[str]]) -> str:
    """Return the largest fragment by atom count from SMILES.

    :param smiles: SMILES string(s), possibly with '.' separators.
    :type smiles: str or List[str]
    :returns: SMILES of the fragment with the most atoms, or empty
        string if none valid.
    :rtype: str
    """
    if isinstance(smiles, str):
        fragments = smiles.split(".")
    else:
        fragments = [frag for s in smiles for frag in s.split(".")]
    mols = [Chem.MolFromSmiles(f) for f in fragments if f]
    mols = [m for m in mols if m]
    if not mols:
        return ""
    max_mol = max(mols, key=lambda m: m.GetNumAtoms())
    return Chem.MolToSmiles(max_mol)


def filter_smiles(smiles_list: List[str], target_smiles: str) -> List[str]:
    """Filter SMILES list to those containing carbon and not equal to a target.

    :param smiles_list: List of SMILES strings to filter.
    :type smiles_list: List[str]
    :param target_smiles: SMILES string to exclude.
    :type target_smiles: str
    :returns: Filtered list containing SMILES with at least one carbon atom
              and not matching `target_smiles`.
    :rtype: List[str]
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_can = Chem.MolToSmiles(target_mol) if target_mol else ""
    result: List[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol and any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
            can = Chem.MolToSmiles(mol)
            if can != target_can:
                result.append(smi)
    return result


def remove_atom_mappings(mol: Chem.Mol) -> Chem.Mol:
    """Strip atom‑mapping numbers from a molecule.

    :param mol: RDKit Mol object.
    :type mol: Chem.Mol
    :returns: The same Mol with all atom‑map numbers set to zero.
    :rtype: Chem.Mol
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def get_sanitized_smiles(smiles_list: List[str]) -> List[str]:
    """Sanitize SMILES list by removing mappings and invalid entries.

    :param smiles_list: List of SMILES strings to sanitize.
    :type smiles_list: List[str]
    :returns: List of sanitized, isomeric SMILES of the largest
        fragments only.
    :rtype: List[str]
    """
    sanitized: List[str] = []
    for smiles in smiles_list:
        if "->" in smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        mol = remove_atom_mappings(mol)
        try:
            Chem.SanitizeMol(mol)
            sanitized.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        except Exception:
            continue
    # keep only the largest fragment across all
    if sanitized:
        sanitized = [get_max_fragment(sanitized)]
    return sanitized


def remove_duplicates(smiles_list: List[str]) -> List[str]:
    """Remove duplicate strings from a list, preserving first occurrence.

    :param smiles_list: List of strings (e.g., SMILES) possibly with
        duplicates.
    :type smiles_list: List[str]
    :returns: List with duplicates removed in original order.
    :rtype: List[str]
    """
    seen = set()
    unique: List[str] = []
    for s in smiles_list:
        if s not in seen:
            unique.append(s)
            seen.add(s)
    return unique


def process_smiles_list(smiles_list: List[str]) -> List[str]:
    """Split dot‑connected SMILES into individual components.

    :param smiles_list: List of SMILES strings, some containing '.'
        separators.
    :type smiles_list: List[str]
    :returns: Flattened list of component SMILES strings.
    :rtype: List[str]
    """
    new_list: List[str] = []
    for smiles in smiles_list:
        if "." in smiles:
            new_list.extend(smiles.split("."))
        else:
            new_list.append(smiles)
    return new_list


def remove_explicit_H_from_rsmi(rsmi: str) -> str:
    """Remove explicit H atoms from a reaction SMILES, preserving AAM.

    :param rsmi: Atom‑mapped reaction SMILES with explicit hydrogens.
    :type rsmi: str
    :returns: Simplified reaction SMILES with implicit hydrogens.
    :rtype: str
    """
    rxn = rdChemReactions.ReactionFromSmarts(rsmi, useSmiles=True)

    def cleaned(mols):
        return ".".join(
            Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True) for m in mols
        )

    react = cleaned(rxn.GetReactants())
    prod = cleaned(rxn.GetProducts())
    return f"{react}>>{prod}"


def remove_common_reagents(reaction_smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """Remove reagents present on both sides of a reaction SMILES.

    :param reaction_smiles: Reaction SMILES 'reactants>>products'.
    :type reaction_smiles: str
    :returns: Tuple(cleaned_reaction, list_of_removed_reagents or None
        if none found).
    :rtype: Tuple[str, Optional[List[str]]]
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


def reverse_reaction(rsmi: str) -> str:
    """Reverse a reaction SMILES.

    :param rsmi: Reaction SMILES 'reactants>>products'.
    :type rsmi: str
    :returns: Reaction SMILES 'products>>reactants'.
    :rtype: str
    """
    parts = rsmi.split(">>")
    return f"{parts[1]}>>{parts[0]}" if len(parts) == 2 else rsmi


def merge_reaction(rsmi_1: str, rsmi_2: str) -> Optional[str]:
    """Merge two reaction SMILES into a single combined reaction.

    :param rsmi_1: First reaction SMILES.
    :type rsmi_1: str
    :param rsmi_2: Second reaction SMILES.
    :type rsmi_2: str
    :returns: Merged reaction SMILES or None if inputs invalid.
    :rtype: Optional[str]
    """
    try:
        r1, p1 = rsmi_1.split(">>")
        r2, p2 = rsmi_2.split(">>")
    except ValueError:
        return None
    if not all([r1, p1, r2, p2]):
        return None
    return f"{r1}.{r2}>>{p1}.{p2}"


def find_longest_fragment(input_list: List[str]) -> Optional[str]:
    """Find the longest string in a list.

    :param input_list: List of strings to search.
    :type input_list: List[str]
    :returns: Longest string or None if list empty.
    :rtype: Optional[str]
    """
    if not input_list:
        return None
    return max(input_list, key=len)
