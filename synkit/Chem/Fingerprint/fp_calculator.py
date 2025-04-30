from joblib import Parallel, delayed
from typing import Dict, List
from synkit.IO.debug import configure_warnings_and_logs
from synkit.Chem.Fingerprint.transformation_fp import TransformationFP

# Configure warnings and logging
configure_warnings_and_logs(True, True)


class FPCalculator:
    """
    Class to calculate fingerprint vectors for chemical compounds represented by
    SMILES strings. This class provides methods to process SMILES strings
    into various types of fingerprint vectors,
    either individually or in batches, and supports parallel processing.

    Attributes:
    - smiles_key (str): Key in the dictionary corresponding to the SMILES string.
    - fp_type (str): Type of fingerprint to calculate; supports various cheminformatics fingerprint types.
    - n_jobs (int): Number of parallel jobs to run for performance enhancement.
    - verbose (int): Verbosity level of parallel computation.
    """

    # Class-level instance to be used in static methods.
    fps = TransformationFP()

    def __init__(
        self,
        smiles_key: str,
        fp_type: str,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        """
        Initialize the FPCalculator with specific settings for SMILES string processing and fingerprint generation.

        Parameters:
        - smiles_key (str): The key in a dictionary corresponding to the SMILES string.
        - fp_type (str): The type of fingerprint to generate.
        - n_jobs (int): Number of parallel jobs.
        Default is 1.
        - verbose (int): Verbosity level for parallel processing.
        Default is 0.
        """
        self.smiles_key = smiles_key
        self.fp_type = fp_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._validate_fp_type(fp_type)

    def _validate_fp_type(self, fp_type: str) -> None:
        """
        Validate if the provided fingerprint type is supported.

        Parameters:
        - fp_type (str): The type of fingerprint to be validated.

        Raises:
            ValueError: If the fingerprint type is not supported.
        """
        valid_fps = [
            "drfp",
            "avalon",
            "maccs",
            "torsion",
            "pharm2D",
            "ecfp2",
            "ecfp4",
            "ecfp6",
            "fcfp2",
            "fcfp4",
            "fcfp6",
            "rdk5",
            "rdk6",
            "rdk7",
        ]
        if fp_type not in valid_fps:
            raise ValueError(
                f"Unsupported fingerprint type '{fp_type}'. Currently supported: {', '.join(valid_fps)}."
            )

    @staticmethod
    def dict_process(
        data_dict: Dict,
        rsmi_key: str,
        symbol: str = ">>",
        fp_type: str = "ap",
        absolute: bool = True,
    ) -> Dict:
        """
        Convert a reaction SMILES string to a fingerprint vector based on the
        specified fingerprint type.

        Parameters:
        - data_dict (Dict): A dictionary containing reaction SMILES.
        - rsmi_key (str): The key in the dictionary for the reaction SMILES.
        - symbol (str): The symbol used to separate reactants and products.
        Default is '>>'.
        - fp_type (str): The type of fingerprint to generate.
        Default is 'ap'.
        - absolute (bool): Whether to use absolute values.
        Default is True.

        Returns:
        - Dict: The updated dictionary with the fingerprint added
        under the key `fp_type`.

        Raises:
        - ValueError: If an unsupported fingerprint type is specified or
        the reaction SMILES key does not exist.
        """
        if rsmi_key not in data_dict:
            raise ValueError(f"Key '{rsmi_key}' does not exist in the dictionary.")
        data_dict[fp_type] = FPCalculator.fps.fit(
            data_dict[rsmi_key], symbols=symbol, fp_type=fp_type, abs=absolute
        )
        return data_dict

    def parallel_process(
        self,
        data_dicts: List[Dict],
        rsmi_key: str,
        symbol: str = ">>",
        fp_type: str = "ap",
        absolute: bool = True,
    ) -> List[Dict]:
        """
        Convert a list of SMILES strings to fingerprint vectors in parallel
        based on the specified fingerprint type. This method processes
        multiple dictionaries containing SMILES strings simultaneously
        using multiple workers.

        Parameters:
        - data_dicts (List[Dict]): A list of dictionaries, each containing reaction data.
        - rsmi_key (str): The key to access the reaction SMILES in each dictionary.
        - symbol (str): The symbol used to separate reactants and products.
        Default is '>>'.
        - fp_type (str): The type of fingerprint to generate.
        Default is 'ap'.
        - absolute (bool): Whether to use absolute values.
        Default is True.

        Returns:
        - List[Dict]: A list of dictionaries with updated fingerprint data,
        where each dictionary includes a fingerprint vector.

        Raises:
        - ValueError: If an unsupported fingerprint type is specified or the
        reaction SMILES key does not exist in any dictionary.
        """

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(FPCalculator.safe_dict_process)(
                data_dict, rsmi_key, symbol, fp_type, absolute
            )
            for data_dict in data_dicts
        )
        return results
