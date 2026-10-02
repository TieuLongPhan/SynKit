from typing import Dict, List, Optional, Union, Any
import re
from copy import deepcopy

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class Formula:
    """
    Decompose SMILES into element counts and generate Hill-order formulas / molecular weights using RDKit.

    Main APIs:
      - :meth:`decompose`: element counts as a dict (e.g., {'C': 6, 'H': 6})
      - :meth:`hill_formula`: Hill-order formula string
      - :meth:`mol_weight`: RDKit molecular weight (sums '.' fragments)
      - :meth:`process_list`: batch over a list of SMILES
      - :meth:`process_list_dict`: batch over a list of dicts or a pandas DataFrame

    Hill notation rules implemented:
      - If carbon ('C') present: list 'C', then 'H', then other elements alphabetically.
      - If no carbon: list all elements alphabetically.
      - Counts of 1 are omitted (CH3, not C1H3).

    :param n_jobs: Number of parallel jobs for batch processing via joblib. Use 1 to disable parallelism.
    :param verbose: Verbosity level passed to joblib.Parallel.
    """

    _element_pattern = re.compile(r"([A-Z][a-z]*)(\d*)")

    def __init__(self, n_jobs: int = 1, verbose: int = 0) -> None:
        self.n_jobs = int(n_jobs)
        self.verbose = int(verbose)

    def __repr__(self) -> str:
        return f"Formula(n_jobs={self.n_jobs}, verbose={self.verbose})"

    # ---------------------------- Core utilities ---------------------------- #

    @staticmethod
    def _parse_formula_string(formula: str) -> Dict[str, int]:
        """
        Parse a molecular formula string into a dict of element counts.

        :param formula: Molecular formula string (e.g., "C6H12O6", "CH4", "H2O").
        :return: Dict mapping element symbol to integer count.
        """
        parts = Formula._element_pattern.findall(formula)
        return {elem: (int(n) if n else 1) for elem, n in parts}

    @staticmethod
    def _sum_counts(counts_list: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Sum a list of element-count dictionaries.

        :param counts_list: List of dicts of element counts.
        :return: Single dict with summed counts.
        """
        out: Dict[str, int] = {}
        for d in counts_list:
            for k, v in d.items():
                out[k] = out.get(k, 0) + int(v)
        return out

    # ----------------------------- Single item ------------------------------ #

    def decompose(self, smiles: str) -> Dict[str, int]:
        """
        Decompose a SMILES string into element counts using RDKit's CalcMolFormula.
        Disconnected fragments separated by '.' are handled by summing counts.

        :param smiles: SMILES string (may contain '.' for multiple fragments).
        :return: Dict of element counts (empty dict if invalid/empty).
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return {}

        per_frag_counts: List[Dict[str, int]] = []
        for frag in smiles.split("."):
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                # Ignore invalid fragments
                continue
            fstr = rdMolDescriptors.CalcMolFormula(mol)
            per_frag_counts.append(self._parse_formula_string(fstr))

        return self._sum_counts(per_frag_counts)

    def hill_formula(self, smiles: str) -> str:
        """
        Convert a SMILES to a Hill-order formula string.

        Rules:
          - If 'C' present: C then H then other elements alphabetical.
          - If no 'C': all elements alphabetical.
          - Counts of 1 are omitted.

        :param smiles: SMILES string.
        :return: Hill-order formula string; empty string for invalid/empty SMILES.
        """
        counts = self.decompose(smiles)
        if not counts:
            return ""

        def fmt(elem: str, n: int) -> str:
            return f"{elem}{n if n != 1 else ''}"

        if "C" in counts:
            parts: List[str] = []
            parts.append(fmt("C", counts.get("C", 0)))
            if counts.get("H", 0) > 0:
                parts.append(fmt("H", counts["H"]))
            others = sorted(k for k in counts.keys() if k not in ("C", "H"))
            parts.extend(fmt(k, counts[k]) for k in others if counts[k] > 0)
            # Defensive: drop any accidental zeros
            parts = [p for p in parts if not p.endswith("0")]
            return "".join(parts)

        # No carbon present
        parts = [fmt(k, counts[k]) for k in sorted(counts.keys()) if counts[k] > 0]
        return "".join(parts)

    def mol_weight(self, smiles: str) -> Optional[float]:
        """
        Compute molecular weight using RDKit (sum over '.' fragments).

        :param smiles: SMILES string (may contain '.' fragments).
        :return: Molecular weight as float, or None if invalid/empty.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return None

        total = 0.0
        valid = False
        for frag in smiles.split("."):
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                continue
            total += Descriptors.MolWt(mol)
            valid = True

        return total if valid else None

    # ----------------------------- Batch APIs -------------------------------- #

    def process_list(
        self,
        smiles_list: List[str],
        what: str = "hill",
    ) -> List[Union[str, Dict[str, int], float, None]]:
        """
        Batch process a list of SMILES.

        :param smiles_list: List of SMILES strings.
        :param what: One of {'hill', 'decompose', 'molwt'}.
        :return: List of results corresponding to 'what'.
                 - 'hill' -> List[str]
                 - 'decompose' -> List[Dict[str,int]]
                 - 'molwt' -> List[Optional[float]]
        :raises ValueError: If 'what' is unsupported.
        """
        dispatch = {
            "hill": self.hill_formula,
            "decompose": self.decompose,
            "molwt": self.mol_weight,
        }
        if what not in dispatch:
            raise ValueError(
                "Unsupported 'what'. Choose from {'hill', 'decompose', 'molwt'}."
            )

        fn = dispatch[what]
        if self.n_jobs == 1:
            return [fn(s) for s in smiles_list]
        return Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fn)(s) for s in smiles_list
        )

    def process_list_dict(
        self,
        records: Union[List[Dict[str, Any]], pd.DataFrame],
        smiles_key: str = "smiles",
        out_key: str = "hill",
        what: str = "hill",
        copy: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch process a list of dictionaries (or a pandas DataFrame) containing SMILES,
        and return a list of dictionaries with the computed output appended.

        - If a pandas DataFrame is provided, it is converted to a list of dicts (records).
        - Input dicts are deep-copied by default to avoid in-place mutation.

        :param records: List[dict] or pandas DataFrame. Each record must contain `smiles_key`.
        :param smiles_key: Key in each record holding the SMILES string.
        :param out_key: Output key to store the computed value, e.g. 'hill', 'decompose', 'molwt'.
        :param what: One of {'hill', 'decompose', 'molwt'} specifying the computation.
        :param copy: If True, deep-copy each record before adding output.
        :return: List of dicts with an added `out_key`.
        :raises KeyError: If a record is missing `smiles_key`.
        :raises ValueError: If 'what' is unsupported.
        """
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient="records")

        if not isinstance(records, list):
            raise ValueError("`records` must be a list of dicts or a pandas DataFrame.")

        dispatch = {
            "hill": self.hill_formula,
            "decompose": self.decompose,
            "molwt": self.mol_weight,
        }
        if what not in dispatch:
            raise ValueError(
                "Unsupported 'what'. Choose from {'hill', 'decompose', 'molwt'}."
            )

        fn = dispatch[what]

        # Extract SMILES list with index mapping
        smiles_seq: List[str] = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ValueError(f"Record at index {i} is not a dict.")
            if smiles_key not in rec:
                raise KeyError(
                    f"Record at index {i} missing required key '{smiles_key}'."
                )
            smiles_seq.append(rec[smiles_key])

        # Compute results
        if self.n_jobs == 1:
            outputs = [fn(s) for s in smiles_seq]
        else:
            outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(fn)(s) for s in smiles_seq
            )

        # Attach outputs
        out_records: List[Dict[str, Any]] = []
        for rec, val in zip(records, outputs):
            r = deepcopy(rec) if copy else rec
            r[out_key] = val
            out_records.append(r)

        return out_records
