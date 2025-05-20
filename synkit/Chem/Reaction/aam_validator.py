import pandas as pd
import networkx as nx
from operator import eq
from itertools import combinations
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union, Optional
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.IO.chem_converter import rsmi_to_graph
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.ITS.its_construction import ITSConstruction
from .aam_utils import enumerate_tautomers, mapping_success_rate


class AAMValidator:
    """
    A utility class for validating atom‐atom mappings (AAM) in reaction SMILES.

    Provides methods to compare mapped SMILES against ground truth by
    using reaction‐center (RC) or ITS‐graph isomorphism checks, including
    tautomer enumeration support and batch validation over tabular data.
    """

    def __init__(self) -> None:
        """
        Initialize the AAMValidator.
        """
        pass

    @staticmethod
    def check_equivariant_graph(
        its_graphs: List[nx.Graph],
    ) -> Tuple[List[Tuple[int, int]], int]:
        """
        Identify all pairs of isomorphic ITS graphs.

        :param its_graphs: A list of ITS graphs to compare.
        :type its_graphs: list of networkx.Graph
        :returns:
            - A list of index‐pairs `(i, j)` where `its_graphs[i]` is isomorphic to `its_graphs[j]`.
            - The total count of such isomorphic pairs.
        :rtype: tuple (list of tuple of int, int, int)
        """
        node_labels = ["typesGH"]
        default = ["*", False, 0, 0, ()]
        ops = [eq, eq, eq, eq, eq]
        node_match = generic_node_match(node_labels, default, ops)
        edge_match = generic_edge_match("order", 1, eq)

        classified = []
        for i, j in combinations(range(len(its_graphs)), 2):
            if nx.is_isomorphic(
                its_graphs[i],
                its_graphs[j],
                node_match=node_match,
                edge_match=edge_match,
            ):
                classified.append((i, j))

        return classified, len(classified)

    @staticmethod
    def smiles_check(
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
    ) -> bool:
        """
        Validate a single mapped SMILES string against ground truth.

        :param mapped_smile: The mapped SMILES to validate.
        :type mapped_smile: str
        :param ground_truth: The reference SMILES string.
        :type ground_truth: str
        :param check_method: Which method to use:
                             `"RC"` for reaction‐center graph or
                             `"ITS"` for full ITS‐graph isomorphism.
        :type check_method: str
        :param ignore_aromaticity: If True, ignore aromaticity differences in ITS construction.
        :type ignore_aromaticity: bool
        :returns: True if exactly one isomorphic match is found; False otherwise.
        :rtype: bool
        """
        its_graphs, rc_graphs = [], []
        try:
            for rsmi in (mapped_smile, ground_truth):
                G, H = rsmi_to_graph(rsmi=rsmi, sanitize=True, drop_non_aam=True)
                its = ITSConstruction().ITSGraph(
                    G, H, ignore_aromaticity=ignore_aromaticity
                )
                its_graphs.append(its)
                rc_graphs.append(get_rc(its))

            graphs = rc_graphs if check_method.upper() == "RC" else its_graphs
            _, count = AAMValidator.check_equivariant_graph(graphs)
            return count == 1
        except Exception:
            return False

    @staticmethod
    def smiles_check_tautomer(
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
    ) -> Optional[bool]:
        """
        Validate against all tautomers of a ground truth SMILES.

        :param mapped_smile: The mapped SMILES to test.
        :type mapped_smile: str
        :param ground_truth: The reference SMILES for generating tautomers.
        :type ground_truth: str
        :param check_method: `"RC"` or `"ITS"` as in `smiles_check`.
        :type check_method: str
        :param ignore_aromaticity: If True, ignore aromaticity in ITS construction.
        :type ignore_aromaticity: bool
        :returns:
            - `True` if any tautomer matches.
            - `False` if none match.
            - `None` if an error occurs.
        :rtype: bool or None
        """
        try:
            tautomers = enumerate_tautomers(ground_truth)
            return any(
                AAMValidator.smiles_check(
                    mapped_smile, taut, check_method, ignore_aromaticity
                )
                for taut in tautomers
            )
        except Exception:
            return None

    @staticmethod
    def check_pair(
        mapping: Dict[str, str],
        mapped_col: str,
        ground_truth_col: str,
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        ignore_tautomers: bool = True,
    ) -> bool:
        """
        Validate a single record (dict) entry for equivalence.

        :param mapping: A record containing both mapped and ground‐truth SMILES.
        :type mapping: dict of str→str
        :param mapped_col: Key for the mapped SMILES in `mapping`.
        :type mapped_col: str
        :param ground_truth_col: Key for the ground-truth SMILES in `mapping`.
        :type ground_truth_col: str
        :param check_method: `"RC"` or `"ITS"`.
        :type check_method: str
        :param ignore_aromaticity: If True, ignore aromaticity in ITS construction.
        :type ignore_aromaticity: bool
        :param ignore_tautomers: If True, skip tautomer enumeration.
        :type ignore_tautomers: bool
        :returns: Validation result for this single pair.
        :rtype: bool
        """
        if ignore_tautomers:
            return AAMValidator.smiles_check(
                mapping[mapped_col],
                mapping[ground_truth_col],
                check_method,
                ignore_aromaticity,
            )
        else:
            return AAMValidator.smiles_check_tautomer(
                mapping[mapped_col],
                mapping[ground_truth_col],
                check_method,
                ignore_aromaticity,
            )

    @staticmethod
    def validate_smiles(
        data: Union[pd.DataFrame, List[Dict[str, str]]],
        ground_truth_col: str = "ground_truth",
        mapped_cols: List[str] = ["rxn_mapper", "graphormer", "local_mapper"],
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
        ignore_tautomers: bool = True,
    ) -> List[Dict[str, Union[str, float, List[bool]]]]:
        """
        Batch-validate mapped SMILES in tabular or list-of-dicts form.

        :param data: A pandas DataFrame or list of dicts, each row containing at least
                     `ground_truth_col` and each entry in `mapped_cols`.
        :type data: pandas.DataFrame or list of dict
        :param ground_truth_col: Column/key name for the ground-truth SMILES.
        :type ground_truth_col: str
        :param mapped_cols: List of column/key names for mapped SMILES to validate.
        :type mapped_cols: list of str
        :param check_method: `"RC"` or `"ITS"` validation method.
        :type check_method: str
        :param ignore_aromaticity: If True, ignore aromaticity in ITS construction.
        :type ignore_aromaticity: bool
        :param n_jobs: Number of parallel jobs to use (joblib).
        :type n_jobs: int
        :param verbose: Verbosity level for parallel execution.
        :type verbose: int
        :param ignore_tautomers: If True, use simple pairwise check; otherwise enumerate tautomers.
        :type ignore_tautomers: bool
        :returns: A list of dicts, one per mapper, with keys:
                  - `"mapper"`: the mapper name
                  - `"accuracy"`: percentage correct (float)
                  - `"results"`: list of individual bool results
                  - `"success_rate"`: mapping success rate metric
        :rtype: list of dict
        :raises ValueError: If `data` is not a DataFrame or list of dicts.
        """
        validation_results = []

        # Normalize to list-of-dicts
        if isinstance(data, pd.DataFrame):
            mappings = data.to_dict("records")
        elif isinstance(data, list):
            mappings = data
        else:
            raise ValueError(
                "Data must be either a pandas DataFrame or a list of dictionaries."
            )

        for mapped_col in mapped_cols:
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(AAMValidator.check_pair)(
                    mapping,
                    mapped_col,
                    ground_truth_col,
                    check_method,
                    ignore_aromaticity,
                    ignore_tautomers,
                )
                for mapping in mappings
            )
            accuracy = sum(results) / len(mappings) if mappings else 0.0
            validation_results.append(
                {
                    "mapper": mapped_col,
                    "accuracy": round(100 * accuracy, 2),
                    "results": results,
                    "success_rate": mapping_success_rate(
                        [m[mapped_col] for m in mappings]
                    ),
                }
            )

        return validation_results
