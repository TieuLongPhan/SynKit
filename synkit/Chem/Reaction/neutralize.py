from rdkit import Chem
from joblib import Parallel, delayed
from typing import Dict, Any, List, Union, Tuple, Optional


class Neutralize:
    """Neutralize unbalanced charges in chemical reactions by adding
    counter‑ions.

    Provides utilities to calculate formal charges, parse reaction
    SMILES, and adjust reactants/products with [Na+] or [Cl‑] to restore
    neutrality.
    """

    @staticmethod
    def calculate_charge(smiles: str) -> int:
        """Calculate the formal charge of a molecule.

        :param smiles: SMILES string of the molecule.
        :type smiles: str
        :returns: Formal charge of the molecule (0 if invalid SMILES).
        :rtype: int
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Chem.rdmolops.GetFormalCharge(mol)

    @staticmethod
    def parse_reaction(reaction_smiles: str) -> Tuple[Optional[str], Optional[str]]:
        """Split a reaction SMILES into reactants and products.

        :param reaction_smiles: Reaction SMILES in 'reactants>>products'
            format.
        :type reaction_smiles: str
        :returns: Tuple of (reactants, products) SMILES, or (None, None)
            if parse fails.
        :rtype: Tuple[Optional[str], Optional[str]]
        """
        try:
            reactants, products = reaction_smiles.split(">>")
            return reactants, products
        except ValueError:
            return None, None

    @staticmethod
    def calculate_charge_dict(
        reaction: Dict[str, Any], reaction_column: str
    ) -> Dict[str, Union[str, int]]:
        """Compute and store the total formal charge of the products in a
        reaction dict.

        :param reaction: Dictionary containing at least `reaction_column` with a reaction SMILES.
        :type reaction: Dict[str, Any]
        :param reaction_column: Key under which the reaction SMILES is stored.
        :type reaction_column: str
        :returns: The same dictionary updated with:
                  - 'reactants': reactant SMILES or None
                  - 'products': product SMILES or None
                  - 'total_charge_in_products': integer sum of product charges or None
        :rtype: Dict[str, Union[str, int]]
        """
        reactants, products = Neutralize.parse_reaction(
            reaction.get(reaction_column, "")
        )
        if reactants is None or products is None:
            reaction.update(
                {"reactants": None, "products": None, "total_charge_in_products": None}
            )
        else:
            reaction["reactants"] = reactants
            reaction["products"] = products
            total = sum(Neutralize.calculate_charge(p) for p in products.split("."))
            reaction["total_charge_in_products"] = total
        return reaction

    @staticmethod
    def fix_negative_charge(
        reaction_dict: Dict[str, Any],
        charges_column: str = "total_charge_in_products",
        id_column: str = "R-id",
        reaction_column: str = "reactions",
    ) -> Dict[str, Any]:
        """Add [Na+] ions to neutralize negative product charge.

        :param reaction_dict: Dictionary with 'reactants', 'products', and charge info.
        :type reaction_dict: Dict[str, Any]
        :param charges_column: Key for product total charge. Defaults to 'total_charge_in_products'.
        :type charges_column: str
        :param id_column: Key for reaction identifier. Defaults to 'R-id'.
        :type id_column: str
        :param reaction_column: Key for reaction SMILES to update. Defaults to 'reactions'.
        :type reaction_column: str
        :returns: New dictionary with:
                  - updated `reaction_column` including added [Na+] ions
                  - 'reactants' and 'products' with ions appended
                  - charge column set to 0
        :rtype: Dict[str, Any]
        """
        num_to_add = abs(reaction_dict.get(charges_column, 0))
        sodium = "[Na+]"
        addition = ("." + ".".join([sodium] * num_to_add)) if num_to_add else ""
        new_react = reaction_dict["reactants"] + addition
        new_prod = reaction_dict["products"] + addition
        new_reaction = f"{new_react}>>{new_prod}"

        return {
            id_column: reaction_dict.get("R-id"),
            reaction_column: new_reaction,
            "reactants": new_react,
            "products": new_prod,
            charges_column: 0,
        }

    @staticmethod
    def fix_positive_charge(
        reaction_dict: Dict[str, Any],
        charges_column: str = "total_charge_in_products",
        id_column: str = "R-id",
        reaction_column: str = "reactions",
    ) -> Dict[str, Any]:
        """Add [Cl‑] ions to neutralize positive product charge.

        :param reaction_dict: Dictionary with 'reactants', 'products', and charge info.
        :type reaction_dict: Dict[str, Any]
        :param charges_column: Key for product total charge. Defaults to 'total_charge_in_products'.
        :type charges_column: str
        :param id_column: Key for reaction identifier. Defaults to 'R-id'.
        :type id_column: str
        :param reaction_column: Key for reaction SMILES to update. Defaults to 'reactions'.
        :type reaction_column: str
        :returns: New dictionary with:
                  - updated `reaction_column` including added [Cl‑] ions
                  - 'reactants' and 'products' with ions appended
                  - charge column set to 0
        :rtype: Dict[str, Any]
        """
        num_to_add = abs(reaction_dict.get(charges_column, 0))
        chloride = "[Cl-]"
        addition = ("." + ".".join([chloride] * num_to_add)) if num_to_add else ""
        new_react = reaction_dict["reactants"] + addition
        new_prod = reaction_dict["products"] + addition
        new_reaction = f"{new_react}>>{new_prod}"

        return {
            id_column: reaction_dict.get("R-id"),
            reaction_column: new_reaction,
            "reactants": new_react,
            "products": new_prod,
            charges_column: 0,
        }

    @staticmethod
    def fix_unbalanced_charged(
        reaction_dict: Dict[str, Any], reaction_column: str
    ) -> Dict[str, Any]:
        """Detect and neutralize unbalanced product charge by adding
        counter‑ions.

        :param reaction_dict: Dictionary with raw reaction SMILES under `reaction_column`.
        :type reaction_dict: Dict[str, Any]
        :param reaction_column: Key for reaction SMILES in the input dict.
        :type reaction_column: str
        :returns: Dictionary with balanced charges and updated SMILES.
        :rtype: Dict[str, Any]
        """
        rd = Neutralize.calculate_charge_dict(reaction_dict, reaction_column)
        total = rd.get("total_charge_in_products", 0)
        if total > 0:
            return Neutralize.fix_positive_charge(rd)
        if total < 0:
            return Neutralize.fix_negative_charge(rd)
        return rd

    @classmethod
    def parallel_fix_unbalanced_charge(
        cls, reaction_dicts: List[Dict[str, Any]], reaction_column: str, n_jobs: int = 4
    ) -> List[Dict[str, Any]]:
        """Neutralize charges in multiple reaction dictionaries in parallel.

        :param reaction_dicts: List of reaction dictionaries to process.
        :type reaction_dicts: List[Dict[str, Any]]
        :param reaction_column: Key for reaction SMILES in each dict.
        :type reaction_column: str
        :param n_jobs: Number of parallel jobs (use -1 for all cores).
            Defaults to 4.
        :type n_jobs: int
        :returns: List of dictionaries with balanced charges and updated
            SMILES.
        :rtype: List[Dict[str, Any]]
        """
        return Parallel(n_jobs=n_jobs)(
            delayed(cls.fix_unbalanced_charged)(d, reaction_column)
            for d in reaction_dicts
        )
