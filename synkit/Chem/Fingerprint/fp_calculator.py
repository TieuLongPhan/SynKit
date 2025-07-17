from __future__ import annotations
from typing import Any, Dict, List
from joblib import Parallel, delayed

from synkit.IO.debug import configure_warnings_and_logs
from synkit.Chem.Fingerprint.transformation_fp import TransformationFP

configure_warnings_and_logs(True, True)


class FPCalculator:
    """Calculate fingerprint vectors for chemical reactions represented by
    SMILES strings.

    :cvar fps: Shared fingerprint engine instance.
    :vartype fps: TransformationFP
    :cvar VALID_FP_TYPES: Supported fingerprint type identifiers.
    :vartype VALID_FP_TYPES: List[str]
    :param n_jobs: Number of parallel jobs to use for batch processing.
    :type n_jobs: int
    :param verbose: Verbosity level for parallel execution.
    :type verbose: int
    """

    fps: TransformationFP = TransformationFP()
    VALID_FP_TYPES: List[str] = [
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
        "ap",
    ]

    def __init__(self, n_jobs: int = 1, verbose: int = 0) -> None:
        """Initialize the FPCalculator.

        :param n_jobs: Number of parallel jobs to use for fingerprint
            computation.
        :type n_jobs: int
        :param verbose: Verbosity level for the parallel processing.
        :type verbose: int
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _validate_fp_type(self, fp_type: str) -> None:
        """Ensure the requested fingerprint type is supported.

        :param fp_type: Fingerprint type identifier to validate.
        :type fp_type: str
        :raises ValueError: If `fp_type` is not in VALID_FP_TYPES.
        """
        if fp_type not in self.VALID_FP_TYPES:
            valid = ", ".join(self.VALID_FP_TYPES)
            raise ValueError(
                f"Unsupported fingerprint type '{fp_type}'. Supported types: {valid}."
            )

    @staticmethod
    def dict_process(
        data_dict: Dict[str, Any],
        rsmi_key: str,
        symbol: str = ">>",
        fp_type: str = "ecfp4",
        absolute: bool = True,
    ) -> Dict[str, Any]:
        """Compute a fingerprint for a single reaction SMILES entry and add it
        to the dict.

        :param data_dict: Dictionary containing reaction data.
        :type data_dict: dict
        :param rsmi_key: Key in `data_dict` for the reaction SMILES string.
        :type rsmi_key: str
        :param symbol: Delimiter between reactant and product in the SMILES.
        :type symbol: str
        :param fp_type: Fingerprint type to compute.
        :type fp_type: str
        :param absolute: Whether to take absolute values of the fingerprint difference.
        :type absolute: bool
        :returns: The input dictionary with a new key `fp_{fp_type}` holding the fingerprint vector.
        :rtype: dict
        :raises ValueError: If `rsmi_key` is missing in `data_dict`.
        """
        if rsmi_key not in data_dict:
            raise ValueError(f"Key '{rsmi_key}' not found in data dictionary.")
        # compute and insert fingerprint
        vec = FPCalculator.fps.fit(
            data_dict[rsmi_key], symbols=symbol, fp_type=fp_type, abs=absolute
        )
        data_dict[f"{fp_type}"] = vec
        return data_dict

    def parallel_process(
        self,
        data_dicts: List[Dict[str, Any]],
        rsmi_key: str,
        symbol: str = ">>",
        fp_type: str = "ecfp4",
        absolute: bool = True,
    ) -> List[Dict[str, Any]]:
        """Compute fingerprints for a batch of reaction dictionaries in
        parallel.

        :param data_dicts: List of dictionaries, each containing a reaction SMILES.
        :type data_dicts: list of dict
        :param rsmi_key: Key in each dict for the reaction SMILES string.
        :type rsmi_key: str
        :param symbol: Delimiter between reactant and product in the SMILES.
        :type symbol: str
        :param fp_type: Fingerprint type to compute.
        :type fp_type: str
        :param absolute: Whether to take absolute values of the fingerprint difference.
        :type absolute: bool
        :returns: A list of dictionaries augmented with `fp_{fp_type}` entries.
        :rtype: list of dict
        :raises ValueError: If `fp_type` is unsupported or any dict is missing `rsmi_key`.
        """
        # Validate fingerprint type once
        self._validate_fp_type(fp_type)

        # Process in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.dict_process)(dd, rsmi_key, symbol, fp_type, absolute)
            for dd in data_dicts
        )
        return results

    def __str__(self) -> str:
        """Short string summarizing the calculator configuration.

        :returns: A summary of n_jobs and verbosity.
        :rtype: str
        """
        return f"<FPCalculator n_jobs={self.n_jobs} verbose={self.verbose}>"

    def help(self) -> None:
        """Print details about supported fingerprint types and usage.

        :returns: None
        :rtype: NoneType
        """
        print("FPCalculator supports the following fingerprint types:")
        for t in self.VALID_FP_TYPES:
            print("  -", t)
        print(f"Configured for {self.n_jobs} parallel jobs, verbose={self.verbose}")
