from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from joblib import Parallel, delayed
from typing import List, Dict, Union, Tuple, Any


class BalanceReactionCheck:
    """Check elemental balance of chemical reactions in SMILES format.

    Supports checking single reactions, reaction dictionaries, or lists
    in parallel.

    :ivar n_jobs: Number of parallel jobs for batch checking.
    :ivar verbose: Verbosity level for joblib.
    """

    def __init__(self, n_jobs: int = 4, verbose: int = 0) -> None:
        """
        :param n_jobs: Number of parallel jobs for batch balance checks. Defaults to 4.
        :type n_jobs: int
        :param verbose: Verbosity level passed to joblib. Defaults to 0.
        :type verbose: int
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def get_combined_molecular_formula(smiles: str) -> str:
        """Compute the molecular formula of a SMILES.

        :param smiles: SMILES string of the molecule.
        :type smiles: str
        :returns: Elemental formula (e.g., "C6H6") or empty string if
            invalid.
        :rtype: str
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
        """Normalize input into a list of reaction‐dicts.

        :param input_data: A single SMILES, list of SMILES, or list of dicts containing `rsmi_column`.
        :type input_data: str or List[Union[str, Dict[str, str]]]
        :param rsmi_column: Key in dicts for the reaction SMILES. Defaults to "reactions".
        :type rsmi_column: str
        :returns: List of dicts with a single key `rsmi_column` mapping to each reaction.
        :rtype: List[Dict[str, str]]
        :raises ValueError: If `input_data` is neither str nor list.
        """
        standardized: List[Dict[str, str]] = []
        if isinstance(input_data, str):
            standardized.append({rsmi_column: input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, str):
                    standardized.append({rsmi_column: item})
                elif isinstance(item, dict) and rsmi_column in item:
                    standardized.append(item)
        else:
            raise ValueError("Unsupported input type for balance checking")
        return standardized

    @staticmethod
    def parse_reaction(reaction_smiles: str) -> Tuple[str, str]:
        """Split a reaction SMILES into reactant and product SMILES strings.

        :param reaction_smiles: Reaction SMILES in 'reactants>>products'
            format.
        :type reaction_smiles: str
        :returns: Tuple of (reactants, products) SMILES.
        :rtype: Tuple[str, str]
        """
        return tuple(reaction_smiles.split(">>"))

    @staticmethod
    def rsmi_balance_check(reaction_smiles: str) -> bool:
        """Determine if a reaction SMILES is elementally balanced.

        :param reaction_smiles: Reaction SMILES in 'reactants>>products'
            format.
        :type reaction_smiles: str
        :returns: True if reactant and product formulas match, else
            False.
        :rtype: bool
        """
        react, prod = BalanceReactionCheck.parse_reaction(reaction_smiles)
        react_formula = BalanceReactionCheck.get_combined_molecular_formula(react)
        prod_formula = BalanceReactionCheck.get_combined_molecular_formula(prod)
        return react_formula == prod_formula

    @staticmethod
    def dict_balance_check(
        reaction_dict: Dict[str, str], rsmi_column: str
    ) -> Dict[str, Any]:
        """Check balance for a single reaction dict, preserving original keys.

        :param reaction_dict: Dict containing at least a `rsmi_column` key.
        :type reaction_dict: Dict[str, str]
        :param rsmi_column: Key for reaction SMILES in `reaction_dict`.
        :type rsmi_column: str
        :returns: Original dict augmented with `"balanced": bool`.
        :rtype: Dict[str, Any]
        """
        rsmi = reaction_dict[rsmi_column]
        balanced = BalanceReactionCheck.rsmi_balance_check(rsmi)
        return {"balanced": balanced, **reaction_dict}

    def dicts_balance_check(
        self,
        input_data: Union[str, List[Union[str, Dict[str, str]]]],
        rsmi_column: str = "reactions",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Batch‐check balance for multiple reactions, in parallel.

        :param input_data: Single reaction SMILES, list of SMILES, or
            list of dicts.
        :type input_data: Union[str, List[Union[str, Dict[str, str]]]]
        :param rsmi_column: Key for reaction SMILES in each dict.
            Defaults to "reactions".
        :type rsmi_column: str
        :returns: Tuple (balanced_list, unbalanced_list) of dicts each
            including `"balanced"`.
        :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        """
        reactions = self.parse_input(input_data, rsmi_column)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.dict_balance_check)(rd, rsmi_column) for rd in reactions
        )
        balanced = [r for r in results if r["balanced"]]
        unbalanced = [r for r in results if not r["balanced"]]
        return balanced, unbalanced
