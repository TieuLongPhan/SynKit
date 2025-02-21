from rdkit import Chem
from joblib import Parallel, delayed
from typing import Dict, Any, List, Union, Tuple


class Neutralize:
    """
    A class for neutralizing unbalanced charges in a reaction.
    """

    @staticmethod
    def calculate_charge(smiles: str) -> int:
        """
        Calculates the formal charge of a given molecule represented by a SMILES string.

        Parameters:
        - smiles (str): A SMILES string representing a molecule.

        Returns:
        - int: The formal charge of the molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return Chem.rdmolops.GetFormalCharge(mol)

    @staticmethod
    def parse_reaction(reaction_smiles: str) -> Tuple[str, str]:
        """
        Parses a reaction SMILES string into reactants and products.

        Parameters:
        - reaction_smiles (str): A reaction SMILES string of the form
                            "reactants>>products".

        Returns:
        - Tuple[str, str]: A tuple containing the reactants and
                            products SMILES strings, respectively.

        This function uses a while loop and exception handling to
        manage parsing errors and ensure the input is correctly formatted.
        """
        try:
            reactants, products = reaction_smiles.split(">>")
            return reactants, products
        except ValueError:
            return None, None

    @staticmethod
    def calculate_charge_dict(
        reaction: Dict[str, str], reaction_column: str
    ) -> Dict[str, Union[str, int]]:
        """
        Calculates and adds the total charge of products in a single reaction.

        Parameters:
        - reaction (Dict[str, str]): A dictionary representing a reaction with keys
          'R-id' and 'new_reaction'.

        Returns:
        - Dict[str, Union[str, int]]: The same reaction dictionary, with an added key
          'total_charge_in_products' indicating the sum of formal charges in its products.
        """
        reactants, products = Neutralize.parse_reaction(reaction[reaction_column])
        if reactants is None or products is None:
            reaction.update(
                {"reactants": None, "products": None, "total_charge_in_products": None}
            )
        else:
            reaction["reactants"] = reactants
            reaction["products"] = products
            products = products.split(".")
            total_charge = sum(
                Neutralize.calculate_charge(product) for product in products
            )
            reaction["total_charge_in_products"] = total_charge
        return reaction

    @staticmethod
    def fix_negative_charge(
        reaction_dict: Dict[str, any],
        charges_column: str = "total_charge_in_products",
        id_column: str = "R-id",
        reaction_column: str = "reactions",
    ) -> Dict[str, any]:
        """
        Adjusts a reaction dictionary to compensate for a negative charge
        in the products by adding [Na+] ions.

        This function calculates the number of sodium ions ([Na+]) needed to neutralize
        negative charges in the reaction products. It then adds the appropriate number of
        sodium ions to both the reactants and products.

        Parameters::
        - reaction_dict (Dict[str, any]): A dictionary representing a chemical reaction.
        Must include keys for 'total_charge_in_products', 'reactants', 'products', 'R-id',
        and 'label'.

        Returns:
        - Dict[str, any]: A new reaction dictionary with adjusted reactants and products
        to neutralize the negative charge. The 'total_charge_in_products' is set to 0,
        assuming the charge has been neutralized.
        """

        num_na_to_add = abs(reaction_dict[charges_column])
        sodium_ion = "[Na+]"

        # Generate the string to add, with the correct number of sodium ions
        sodium_addition = (
            "." + ".".join([sodium_ion] * num_na_to_add) if num_na_to_add > 0 else ""
        )

        # Add the sodium ions to reactants and products
        new_reactants = reaction_dict["reactants"] + sodium_addition
        new_products = reaction_dict["products"] + sodium_addition

        # Generate the new reaction string
        new_reactions = new_reactants + ">>" + new_products

        # Create the new reaction dictionary
        new_reaction_dict = {
            id_column: reaction_dict["R-id"],
            reaction_column: new_reactions,
            "reactants": new_reactants,
            "products": new_products,
            charges_column: 0,  # Assuming the charge is neutralized
        }

        return new_reaction_dict

    @staticmethod
    def fix_positive_charge(
        reaction_dict: Dict[str, any],
        charges_column: str = "total_charge_in_products",
        id_column: str = "R-id",
        reaction_column: str = "reactions",
    ) -> Dict[str, any]:
        """
        Adjusts a reaction dictionary to compensate for a positive charge
        in the products by adding [Cl-] ions. The function
        takes into account the total positive charge indicated
        in the reaction dictionary and adds an equivalent number of
        chloride ions ([Cl-]) to both reactants and products to neutralize the charge.

        Parameters::
        - reaction_dict (Dict[str, any]): A dictionary representing a chemical reaction.
        This dictionary must include keys for reactants, products, and a specified charge
        column (default is 'total_charge_in_products') which contains the total charge of
        the products.
        - charges_column (str, optional): The key in `reaction_dict` that contains the
        total charge of the products. Defaults to 'total_charge_in_products'.

        Returns:
        - Dict[str, any]: A modified reaction dictionary with added [Cl-] ions to
        neutralize the positive charge. The 'total_charge_in_products' is updated to 0,
        indicating that the reaction's charge has been neutralized. The dictionary
        includes updated 'reactants', 'products', and a new reaction string.
        """

        num_cl_to_add = abs(reaction_dict[charges_column])
        chloride_ion = "[Cl-]"

        # Generate the string to add, with the correct number of chloride ions
        chloride_addition = (
            "." + ".".join([chloride_ion] * num_cl_to_add) if num_cl_to_add > 0 else ""
        )

        # Add the chloride ions to reactants and products
        new_reactants = reaction_dict["reactants"] + chloride_addition
        new_products = reaction_dict["products"] + chloride_addition

        # Generate the new reaction string
        new_reactions = new_reactants + ">>" + new_products

        # Create and return the new reaction dictionary with the neutralized charge
        new_reaction_dict = {
            "R-id": reaction_dict[id_column],
            reaction_column: new_reactions,
            "reactants": new_reactants,
            "products": new_products,
            charges_column: 0,
        }

        return new_reaction_dict

    @staticmethod
    def fix_unbalanced_charged(
        reaction_dict: Dict[str, any],
        reaction_column: str,
    ) -> Dict[str, any]:
        """
        Adjusts a reaction dictionary to compensate for an unbalanced charge in the
        products by adding either [Cl-] ions for a positive charge or [Na+] ions for a
        negative charge. The function determines the direction of the charge imbalance
        using the specified charges column and applies the appropriate correction.

        Parameters::
        - reaction_dict (Dict[str, any]): A dictionary representing a chemical reaction.
        This dictionary must include keys for reactants, products, and a specified charge
        column which contains the total charge of the products.
        - charges_column (str, optional): The key in `reaction_dict` that contains the
        total charge of the products. Defaults to 'total_charge_in_products'.

        Returns:
        - Dict[str, any]: A modified reaction dictionary with added ions to neutralize the
        charge imbalance. The returned dictionary will have its charge neutralized and
        include updated 'reactants', 'products', and a new reaction string. The specific
        ions added ([Cl-] for positive charges or [Na+] for negative charges) depend on
        the initial charge imbalance.
        """
        reaction_dict = Neutralize.calculate_charge_dict(reaction_dict, reaction_column)
        if reaction_dict["total_charge_in_products"] > 0:
            return Neutralize.fix_positive_charge(
                reaction_dict, "total_charge_in_products"
            )
        elif reaction_dict["total_charge_in_products"] < 0:
            return Neutralize.fix_negative_charge(
                reaction_dict, "total_charge_in_products"
            )
        else:
            return reaction_dict

    @classmethod
    def parallel_fix_unbalanced_charge(
        cls,
        reaction_dicts: List[Dict[str, Any]],
        reaction_column: str,
        n_jobs: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Processes a list of reaction dictionaries in parallel to compensate
        for unbalanced charges in the products, adding either [Cl-] ions
        for positive charges or [Na+] ions for negative charges.

        Parameters::
        - reaction_dicts (List[Dict[str, Any]]): A list of dictionaries, each representing
        a chemical reaction that may have an unbalanced charge.
        - charges_column (str): The key in each reaction dictionary that contains the
        total charge of the products. Defaults to 'total_charge_in_products'.
        - n_jobs (int): The number of CPU cores to use for parallel processing.
        -1 means using all available cores.

        Returns:
        - List[Dict[str, Any]]: A list of modified reaction dictionaries
        with charges neutralized, reflecting the addition of necessary ions.

        Note:
        - This function requires the joblib library for parallel execution.
        Ensure joblib is installed and available for import.
        """
        # Use joblib.Parallel and joblib.delayed to parallelize the charge fixing
        fixed_reactions = Parallel(n_jobs=n_jobs)(
            delayed(cls.fix_unbalanced_charged)(reaction_dict, reaction_column)
            for reaction_dict in reaction_dicts
        )
        return fixed_reactions
