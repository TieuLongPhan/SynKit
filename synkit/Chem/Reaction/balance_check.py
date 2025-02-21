from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from joblib import Parallel, delayed
from typing import List, Dict, Union, Tuple


class BalanceReactionCheck:
    """
    A class to check the balance of chemical reactions given in SMILES format.
    It supports parallel execution and maintains the input format in the output.
    """

    def __init__(
        self,
        n_jobs: int = 4,
        verbose: int = 0,
    ):
        """
        Initializes the class with given input data, the column name
        for reactions in the input, number of jobs for
        parallel processing, and verbosity level.

        Parameters:
        - input_data (Union[str, List[Union[str, Dict[str, str]]]]): A single SMILES
        string, a list of SMILES strings, or a list of dictionaries with 'reactions' keys.
        - rsmi_column (str): The key/column name for reaction SMILES strings
        in the input data.
        - n_jobs (int): The number of parallel jobs to run for balance checking.
        - verbose (int): The verbosity level of joblib parallel execution.
        """

        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def get_combined_molecular_formula(smiles: str) -> str:
        """
        Computes the molecular formula for a molecule represented by a SMILES string.

        Parameters:
        - smiles (str): The SMILES string of the molecule.

        Returns:
        - str: The molecular formula, or an empty string if the molecule is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return ""
        return CalcMolFormula(mol)

    @staticmethod
    def parse_input(
        input_data: Union[str, List[Union[str, Dict[str, str]]]],
        rsmi_column: str = "reactions",
    ) -> List[Dict[str, str]]:
        """
        Parses the input data into a standardized list containing
        dictionaries for each reaction.

        Parameters:
        - input_data (Union[str, List[Union[str, Dict[str, str]]]]):
        The input data to be processed.

        Returns:
        - List[Dict[str, str]]: A list of dictionaries with reaction SMILES strings.
        """
        standardized_input = []
        if isinstance(input_data, str):
            standardized_input.append({rsmi_column: input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, str):
                    standardized_input.append({rsmi_column: item})
                elif isinstance(item, dict) and rsmi_column in item:
                    standardized_input.append(item)
        else:
            raise ValueError("Unsupported input type")
        return standardized_input

    @staticmethod
    def parse_reaction(reaction_smiles: str) -> Tuple[List[str], List[str]]:
        """
        Splits a reaction SMILES string into reactants and products.

        Parameters:
        - reaction_smiles (str): A SMILES string representing a chemical reaction.

        Returns:
        - Tuple[List[str], List[str]]: Lists of SMILES strings for reactants and products.
        """
        reactants_smiles, products_smiles = reaction_smiles.split(">>")
        return reactants_smiles, products_smiles

    @staticmethod
    def rsmi_balance_check(reaction_smiles: str) -> bool:
        """
        Checks if a reaction SMILES string is balanced.

        Parameters:
        - reaction_smiles (str): A SMILES string representing a chemical reaction.

        Returns:
        - bool: True if the reaction is balanced, False otherwise.
        """
        reactants_smiles, products_smiles = BalanceReactionCheck.parse_reaction(
            reaction_smiles
        )
        reactants_forumula = BalanceReactionCheck.get_combined_molecular_formula(
            reactants_smiles
        )
        products_forumula = BalanceReactionCheck.get_combined_molecular_formula(
            products_smiles
        )
        return reactants_forumula == products_forumula

    @staticmethod
    def dict_balance_check(
        reaction_dict: Dict[str, str], rsmi_column: str
    ) -> Dict[str, Union[bool, str]]:
        """
        Checks if a single reaction (in SMILES format) is balanced, maintaining
        the input format.

        Parameters:
        - reaction_dict (Dict[str, str]): A dictionary containing the
        reaction SMILES string.

        Returns:
        - Dict[str, Union[bool, str]]: A dictionary indicating if the reaction is
        balanced, along with the original reaction data.
        """
        reaction_smiles = reaction_dict[rsmi_column]
        balance = BalanceReactionCheck.rsmi_balance_check(reaction_smiles)
        return {"balanced": balance, **reaction_dict}

    def dicts_balance_check(
        self,
        input_data: Union[str, List[Union[str, Dict[str, str]]]],
        rsmi_column: str = "reactions",
    ) -> Tuple[List[Dict[str, Union[bool, str]]], List[Dict[str, Union[bool, str]]]]:
        """
        Checks the balance of all reactions in the input data.

        Returns:
        - Tuple[List[Dict[str, Union[bool, str]]], List[Dict[str, Union[bool, str]]]]:
        Two lists containing dictionaries of balanced and unbalanced reactions,
        respectively.
        """

        reactions = self.parse_input(input_data, rsmi_column)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.dict_balance_check)(reaction, rsmi_column)
            for reaction in reactions
        )

        balanced_reactions = [reaction for reaction in results if reaction["balanced"]]
        unbalanced_reactions = [
            reaction for reaction in results if not reaction["balanced"]
        ]

        return balanced_reactions, unbalanced_reactions
