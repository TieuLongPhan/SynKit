import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Optional
from synkit.IO.debug import setup_logging
from synkit.Chem.Fingerprint.transformation_fp import TransformationFP


class FPCalculator:
    """
    Class to calculate fingerprint vectors for chemical compounds represented by SMILES strings.

    Attributes:
    - data (pd.DataFrame): DataFrame containing SMILES strings and potentially other data.
    - smiles_column (str): Name of the column containing the SMILES strings.
    - fp_type (str): Type of fingerprint to calculate; supports 'drfp'.
    - n_jobs (int): Number of parallel jobs to run for performance enhancement.
    - verbose (int): Verbosity level of parallel computation.
    - save_path (Optional[str]): Path to save the resulting DataFrame. If None, the DataFrame is not saved.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles",
        fp_type: str = "drfp",
        n_jobs: int = 2,
        verbose: int = 0,
        save_path: Optional[str] = None,
    ):
        self.data = data
        self.smiles_column = smiles_column
        self.fp_type = fp_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.save_path = save_path
        self.logger = setup_logging()
        self._validate_fp_type(fp_type)

    def _validate_fp_type(self, fp_type: str) -> None:
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
    def smiles_to_vec(reaction_smiles: str, fp_type: str) -> np.ndarray:
        """
        Convert a SMILES string to a fingerprint vector based on the specified fingerprint type.

        Parameters:
        - reaction_smiles (str): A SMILES string representing a chemical compound.
        - fp_type (str): Type of fingerprint to calculate.

        Returns:
        - np.ndarray: A numpy array representing the fingerprint vector.

        Raises:
            ValueError: If an unsupported fingerprint type is specified.
        """
        return TransformationFP.fit(reaction_smiles, ">>", fp_type, True)

    def fit(self) -> pd.DataFrame:
        """
        Calculates the fingerprints for all SMILES strings in the dataset according to the specified fingerprint type.

        Returns:
        - pd.DataFrame: The original DataFrame with an added column for the calculated fingerprints.

        Raises:
            ValueError: If the SMILES column specified does not exist in the DataFrame.
        """
        if self.smiles_column not in self.data.columns:
            raise ValueError(
                f"Column '{self.smiles_column}' does not exist in the DataFrame."
            )

        fps = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(FPCalculator.smiles_to_vec)(smiles, self.fp_type)
            for smiles in self.data[self.smiles_column]
        )

        fps_df = pd.DataFrame(
            fps, columns=[f"{self.fp_type}_{i}" for i in range(len(fps[0]))]
        )
        result_data = pd.concat([self.data, fps_df], axis=1)

        if self.save_path:
            result_data.to_csv(self.save_path, index=False)
            self.logger.info(f"Data saved to {self.save_path}")

        return result_data
