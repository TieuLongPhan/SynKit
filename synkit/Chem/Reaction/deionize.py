import random
from itertools import combinations, permutations
from joblib import Parallel, delayed
from typing import List, Tuple, Callable, Dict, Any

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from synkit.Chem.Reaction.balance_check import BalanceReactionCheck


class Deionize:
    """Neutralize ionic species and mixtures of ions in reactions.

    Provides methods to group ions into neutral combinations, uncharge
    individual anions/cations, and apply these corrections to SMILES
    strings or entire reaction dictionaries.
    """

    @staticmethod
    def random_pair_ions(
        charges: List[int], smiles: List[str]
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """Identify non‑overlapping groups of ions whose charges sum to zero.

        :param charges: List of integer formal charges for each ion.
        :type charges: List[int]
        :param smiles: Corresponding SMILES strings for each ion.
        :type smiles: List[str]
        :returns: A tuple of two lists:
                  - groups of SMILES strings forming neutral sets,
                  - groups of their corresponding charges.
        :rtype: Tuple[List[List[str]], List[List[int]]]
        """

        def find_groups(indices: List[int], size: int) -> Tuple[int, ...]:
            for group in combinations(indices, size):
                if sum(charges[i] for i in group) == 0:
                    return group
            return ()

        indices = list(range(len(charges)))
        random.shuffle(indices)
        used = set()
        grouped_smiles: List[List[str]] = []
        grouped_charges: List[List[int]] = []

        for group_size in (2, 3, 4):
            while True:
                available = [i for i in indices if i not in used]
                group = find_groups(available, group_size)
                if not group:
                    break
                grouped_smiles.append([smiles[i] for i in group])
                grouped_charges.append([charges[i] for i in group])
                used.update(group)

        return grouped_smiles, grouped_charges

    @staticmethod
    def uncharge_anion(smiles: str, charges: int = -1) -> str:
        """Neutralize an anionic SMILES string.

        :param smiles: SMILES of the anion to neutralize.
        :type smiles: str
        :param charges: Formal charge of the ion (negative integer).
            Defaults to -1.
        :type charges: int
        :returns: SMILES of the uncharged molecule.
        :rtype: str
        """
        if smiles == "[N-]=[N+]=[N-]":
            return "[N-]=[N+]=[N]"
        if charges == -1:
            mol = Chem.MolFromSmiles(smiles)
            uncharger = rdMolStandardize.Uncharger()
            uncharged = uncharger.uncharge(mol)
            return Chem.MolToSmiles(uncharged)
        # for multi‐charged anions
        return smiles.replace(f"{charges}", "").replace("[", "").replace("]", "")

    @staticmethod
    def uncharge_cation(smiles: str, charges: int = 1) -> str:
        """Neutralize a cationic SMILES string.

        :param smiles: SMILES of the cation to neutralize.
        :type smiles: str
        :param charges: Formal charge of the ion (positive integer).
            Defaults to 1.
        :type charges: int
        :returns: SMILES of the uncharged molecule.
        :rtype: str
        """
        if charges == 1:
            return smiles.replace("+", "")
        return smiles.replace(f"+{charges}", "")

    @staticmethod
    def uncharge_smiles(charge_smiles: str) -> str:
        """Neutralize all ionic components in a dot‑separated SMILES string.

        Splits into components, identifies ionic species, groups
        them into neutral sets via `random_pair_ions`, then
        applies `uncharge_anion` or `uncharge_cation` and recombines.

        :param charge_smiles: SMILES string with ionic and non‑ionic parts.
        :type charge_smiles: str
        :returns: SMILES string with charges neutralized.
        :rtype: str
        """
        parts = charge_smiles.split(".")
        charges = [Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(p)) for p in parts]
        if all(c == 0 for c in charges):
            return charge_smiles

        non_ionic, ionic_parts, ionic_charges = [], [], []
        for p, c in zip(parts, charges):
            if c == 0:
                non_ionic.append(p)
            else:
                ionic_parts.append(p)
                ionic_charges.append(c)

        valid = non_ionic.copy()
        groups, group_chs = Deionize.random_pair_ions(ionic_charges, ionic_parts)
        for smiles_group, charge_group in zip(groups, group_chs):
            candidates = []
            for smi, ch in zip(smiles_group, charge_group):
                if ch > 0:
                    candidates.append(Deionize.uncharge_cation(smi, ch))
                else:
                    candidates.append(Deionize.uncharge_anion(smi, ch))
            # try permutations for valid SMILES
            for perm in permutations(candidates):
                combo = "".join(perm)
                if Chem.MolFromSmiles(combo):
                    valid.append(Chem.CanonSmiles(combo))
                    break
            else:
                valid.extend(smiles_group)
        return ".".join(valid)

    @staticmethod
    def ammonia_hydroxide_standardize(reaction_smiles: str) -> str:
        """Simplify ammonium hydroxide pairs in a reaction SMILES.

        :param reaction_smiles: Reaction SMILES string.
        :type reaction_smiles: str
        :returns: Reaction SMILES with '[NH4+].[OH-]' replaced by 'N.O'
            or 'O.N'.
        :rtype: str
        """
        return reaction_smiles.replace("[NH4+].[OH-]", "N.O").replace(
            "[OH-].[NH4+]", "O.N"
        )

    @classmethod
    def apply_uncharge_smiles_to_reactions(
        cls,
        reactions: List[Dict[str, Any]],
        uncharge_smiles_func: Callable[[str], str],
        n_jobs: int = 4,
    ) -> List[Dict[str, Any]]:
        """Apply a neutralization function to each reaction’s
        reactants/products in parallel.

        Adds keys 'new_reactants', 'new_products', and 'standardized_reactions'
        based on uncharged SMILES and verifies formula balance.

        :param reactions: List of reaction dicts with 'reactants' and 'products' keys.
        :type reactions: List[Dict[str, Any]]
        :param uncharge_smiles_func: Function to neutralize a SMILES string.
        :type uncharge_smiles_func: Callable[[str], str]
        :param n_jobs: Number of parallel jobs to run. Defaults to 4.
        :type n_jobs: int
        :returns: List of updated reaction dicts with:
                  - 'success': bool indicating formula match
                  - 'new_reactants' / 'new_products'
                  - 'standardized_reactions'
        :rtype: List[Dict[str, Any]]
        """

        def process(reaction: Dict[str, Any]) -> Dict[str, Any]:
            # pre‐standardize ammonia hydroxide
            r_fix = cls.ammonia_hydroxide_standardize(reaction["reactants"])
            p_fix = cls.ammonia_hydroxide_standardize(reaction["products"])
            ur = uncharge_smiles_func(r_fix)
            up = uncharge_smiles_func(p_fix)
            r_formula = BalanceReactionCheck().get_combined_molecular_formula(ur)
            p_formula = BalanceReactionCheck().get_combined_molecular_formula(up)
            reaction["success"] = r_formula == p_formula
            reaction["new_reactants"] = ur if reaction["success"] else r_fix
            reaction["new_products"] = up if reaction["success"] else p_fix
            reaction["standardized_reactions"] = (
                f"{reaction['new_reactants']}>>{reaction['new_products']}"
            )
            return reaction

        return Parallel(n_jobs=n_jobs)(delayed(process)(rxn) for rxn in reactions)
