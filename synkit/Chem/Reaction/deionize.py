import random
from itertools import permutations
from itertools import combinations
from joblib import Parallel, delayed
from typing import List, Tuple, Callable, Dict

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from synkit.Chem.Reaction.balance_check import BalanceReactionCheck


class Deionize:
    """
    A class to deionize reactions.
    """

    @staticmethod
    def random_pair_ions(
        charges: List[int], smiles: List[str]
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """
        Generates non-overlapping groups of ions (2, 3, or 4) based on
        their charges and corresponding SMILES representations,
        aiming to maximize the total number of ions used by preferring
        multiple smaller groups over fewer larger groups.

        Parameters:
        - charges (List[int]): A list of integer charges of the ions.
        - smiles (List[str]): A list of SMILES strings representing the ions.

        Returns:
        - Tuple[List[List[str]], List[List[int]]]: A tuple containing two lists:
            - The first list contains the groups of SMILES strings.
            - The second list contains the groups of charges.
        """

        def find_groups(indices, size):
            """Finds and removes groups of a specific size that sum to zero charge."""
            for group in combinations(indices, size):
                if sum(charges[i] for i in group) == 0:
                    return group
            return []

        # Prepare initial variables
        indices = list(range(len(charges)))
        random.shuffle(indices)  # Shuffle indices to ensure variety
        used_indices = set()
        grouped_smiles = []
        grouped_charges = []

        for group_size in range(
            2, 5
        ):  # Start with pairs, then triples, and finally quads
            while True:
                group = find_groups(
                    [i for i in indices if i not in used_indices], group_size
                )
                if not group:
                    break  # No more groups of this size can be formed
                grouped_smiles.append([smiles[i] for i in group])
                grouped_charges.append([charges[i] for i in group])
                used_indices.update(group)

        return grouped_smiles, grouped_charges

    @staticmethod
    def uncharge_anion(smiles: str, charges: int = -1) -> str:
        """
        Removes charge from an anionic species represented by a SMILES string.

        This function uses RDKit's standardization tools to neutralize
        the charges in the molecule. It returns
        the SMILES representation of the uncharged molecule.

        Parameters::
        - smiles (str): A SMILES string representing the anionic species.

        Returns:
        - str: The SMILES string of the uncharged molecule.

        Note:
        - The function assumes valid SMILES input.
        """
        if smiles == "[N-]=[N+]=[N-]":
            return "[N-]=[N+]=[N]"
        if charges == -1:
            # Convert the SMILES string to an RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)

            # Initialize the uncharger
            uncharger = rdMolStandardize.Uncharger()

            # Apply the uncharger to the molecule
            uncharged_mol = uncharger.uncharge(mol)

            # Convert the uncharged molecule back to a SMILES string
            return Chem.MolToSmiles(uncharged_mol)

        elif charges < -1:
            new_smiles = (
                smiles.replace(f"{charges}", "").replace("[", "").replace("]", "")
            )
            return new_smiles

    @staticmethod
    def uncharge_cation(smiles: str, charges: int = 1) -> str:
        """
        Removes charge from a cationic species represented by a SMILES string.

        This function uses RDKit's standardization tools to neutralize
        the charges in the molecule. It returns the
        SMILES representation of the uncharged molecule.

        Parameters::
        - smiles (str): A SMILES string representing the cationic species.

        Returns:
        - str: The SMILES string of the uncharged molecule.

        Note:
        - The function assumes valid SMILES input.
        """

        if charges == 1:
            new_smiles = smiles.replace("+", "")
        elif charges > 1:
            # For multiple positive charges, directly modify the SMILES string
            new_smiles = smiles.replace(f"+{charges}", "")
        return new_smiles

    @staticmethod
    def uncharge_smiles(charge_smiles: str) -> str:
        """
        Processes a SMILES string containing ionic and non-ionic parts,
        neutralizes the charges, and returns a modified SMILES string.

        The function splits the input SMILES string into individual components,
        identifies ionic and non-ionic parts,
        and attempts to neutralize charged ions.
        It then creates permutations of the modified ions and combines them into
        a single SMILES string, ensuring the molecular structure is valid.

        Parameters::
        - charge_smiles (str): A SMILES string that may contain ionic and non-ionic parts.

        Returns:
        - str: A modified SMILES string with neutralized charges.

        Note:
        - This function depends on RDKit for molecular operations.
        - The function assumes a valid SMILES input.
        - The 'uncharge_anion' and 'random_pair_ions' functions
                    must be defined and accessible.
        """

        smiles = charge_smiles.split(".")
        charges = [Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(i)) for i in smiles]

        if all(charge == 0 for charge in charges):
            return charge_smiles

        valid_smiles, non_ionic_smiles = [], []
        original_ionic_parts, original_ion_charges = [], []

        # Splitting the SMILES into ionic and non-ionic parts
        for smile, charge in zip(smiles, charges):
            if charge == 0:
                non_ionic_smiles.append(smile)
            else:
                original_ionic_parts.append(smile)
                original_ion_charges.append(charge)

        valid_smiles.extend(non_ionic_smiles)
        paired_smiles, paired_charges = Deionize.random_pair_ions(
            original_ion_charges, original_ionic_parts
        )
        # Processing each pair of ionic parts
        for i_smile, i_charge in zip(paired_smiles, paired_charges):
            modified_ions = []
            for ion, charge in zip(i_smile, i_charge):
                if int(charge) > 0:
                    new_ion = Deionize.uncharge_cation(ion, charge)
                    modified_ions.append(new_ion)
                elif int(charge) < 0:
                    new_ion = Deionize.uncharge_anion(ion, charge)
                    modified_ions.append(new_ion)
            # Creating permutations of the modified ions
            check_merge = False
            for perm in permutations(modified_ions):
                combined_ionic = "".join(perm)
                if Chem.MolFromSmiles(combined_ionic):
                    coordinate_pattern = ["->", "<-"]
                    if all(
                        pattern not in Chem.CanonSmiles(combined_ionic)
                        for pattern in coordinate_pattern
                    ):
                        valid_smiles.append(Chem.CanonSmiles(combined_ionic))
                        check_merge = True
                        break
            if check_merge is False:
                valid_smiles.extend(i_smile)
        return ".".join(valid_smiles)

    @staticmethod
    def ammonia_hydroxide_standardize(reaction_smiles: str) -> str:
        """
        Replaces occurrences of ammonium hydroxide (NH4+ and OH-) in a
        reaction SMILES string with a simplified representation (N.O or O.N).

        Parameters::
            reaction_smiles (str): The reaction SMILES string to be standardized.

        Returns:
            str: The standardized reaction SMILES string with
                ammonium hydroxide represented as 'N.O' or 'O.N'.
        """
        # Simplify the representation of ammonium hydroxide in the reaction SMILES
        new_smiles = reaction_smiles.replace("[NH4+].[OH-]", "N.O").replace(
            "[OH-].[NH4+]", "O.N"
        )
        return new_smiles

    @classmethod
    def apply_uncharge_smiles_to_reactions(
        cls,
        reactions: List[Dict[str, str]],
        uncharge_smiles_func: Callable[[str], str],
        n_jobs: int = 4,
    ) -> List[Dict[str, str]]:
        """
        Applies a given uncharge SMILES function to the reactants
        and products of a list of chemical reactions,
        parallelizing the process for improved performance.
        Each reaction is expected to be a dictionary
        with at least 'reactants' and 'products' keys.
        The function adds three new keys to each reaction
        dictionary: 'uncharged_reactants', 'uncharged_products',
        and 'uncharged_reactions', containing
        the uncharged SMILES strings of reactants, products,
        and the overall reaction, respectively.

        Parameters::
        - reactions (List[Dict[str, str]]): A list of dictionaries, where each dictionary
        represents a chemical reaction with 'reactants' and 'products' keys.
        - uncharge_smiles_func (Callable[[str], str]): A function that takes a SMILES
        string as input and returns a modified SMILES string with neutralized charges.

        Returns:
        - List[Dict[str, str]]: The input list of reaction dictionaries, modified in-place
        to include 'uncharged_reactants', 'uncharged_products', and 'uncharged_reactions'
        keys.
        """

        # Define a helper function for processing a single reaction
        def process_reaction(reaction):
            fix_reactants = cls.ammonia_hydroxide_standardize(reaction["reactants"])
            fix_products = cls.ammonia_hydroxide_standardize(reaction["products"])

            uncharged_reactants = uncharge_smiles_func(fix_reactants)
            uncharged_products = uncharge_smiles_func(fix_products)
            uncharged_reactants_formula = (
                BalanceReactionCheck().get_combined_molecular_formula(
                    uncharged_reactants
                )
            )
            uncharged_products_formula = (
                BalanceReactionCheck().get_combined_molecular_formula(
                    uncharged_products
                )
            )
            if uncharged_reactants_formula != uncharged_products_formula:
                reaction["success"] = False
                reaction["new_reactants"] = fix_reactants
                reaction["new_products"] = fix_products
            else:
                reaction["success"] = True
                reaction["new_reactants"] = uncharged_reactants
                reaction["new_products"] = uncharged_products
            reaction["standardized_reactions"] = (
                f"{reaction['new_reactants']}>>{reaction['new_products']}"
            )
            return reaction

        # Use joblib to parallelize the processing of reactions
        reactions = Parallel(n_jobs=n_jobs)(
            delayed(process_reaction)(reaction) for reaction in reactions
        )
        return reactions
