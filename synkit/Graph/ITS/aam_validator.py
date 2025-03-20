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
from synkit.Graph.ITS.aam_utils import enumerate_tautomers, mapping_success_rate


class AAMValidator:
    def __init__(self):
        """Initializes the AAMValidator class."""
        pass

    @staticmethod
    def check_equivariant_graph(
        its_graphs: List[nx.Graph],
    ) -> Tuple[List[Tuple[int, int]], int]:
        """
        Checks for isomorphism among a list of ITS graphs and
        identifies all pairs of isomorphic graphs.

        Parameters:
        - its_graphs (List[nx.Graph]): A list of ITS graphs.

        Returns:
        - List[Tuple[int, int]]: A list of tuples representing
                pairs of indices of isomorphic graphs.
        - int: The count of unique isomorphic graph pairs found.
        """
        nodeLabelNames = ["typesGH"]
        nodeLabelDefault = ["*", False, 0, 0, ()]
        nodeLabelOperator = [eq, eq, eq, eq, eq]
        nodeMatch = generic_node_match(
            nodeLabelNames, nodeLabelDefault, nodeLabelOperator
        )
        edgeMatch = generic_edge_match("order", 1, eq)

        classified = []
        for i, j in combinations(range(len(its_graphs)), 2):
            if nx.is_isomorphic(
                its_graphs[i], its_graphs[j], node_match=nodeMatch, edge_match=edgeMatch
            ):
                classified.append((i, j))

        return classified, len(classified)

    @staticmethod
    def smiles_check(
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",  # or 'ITS'
        ignore_aromaticity: bool = False,
    ) -> bool:
        """
        Checks the equivalence of mapped SMILES against ground truth
        using reaction center (RC) or ITS graph method.

        Parameters:
        - mapped_smile (str): The mapped SMILES string.
        - ground_truth (str): The ground truth SMILES string.
        - check_method (str): The method used for validation ('RC' or 'ITS').
        - ignore_aromaticity (bool): Flag to ignore aromaticity in ITS graph construction.

        Returns:
        - bool: True if the mapped SMILES is equivalent to the ground truth,
        False otherwise.
        """
        its_graphs = []
        rc_graphs = []
        try:
            for rsmi in [mapped_smile, ground_truth]:
                G, H = rsmi_to_graph(
                    rsmi=rsmi, sanitize=True, drop_non_aam=True, light_weight=True
                )

                ITS = ITSConstruction.ITSGraph(G, H, ignore_aromaticity)
                its_graphs.append(ITS)
                rc = get_rc(ITS)
                rc_graphs.append(rc)

            _, equivariant = AAMValidator.check_equivariant_graph(
                rc_graphs if check_method == "RC" else its_graphs
            )
            return equivariant == 1

        except Exception as e:
            print("An error occurred:", str(e))
            return False

    @staticmethod
    def smiles_check_tautomer(
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",  # or 'ITS'
        ignore_aromaticity: bool = False,
    ) -> Optional[bool]:
        """
        Determines if a given mapped SMILE string is equivalent to any tautomer of
        a ground truth SMILES string using a specified comparison method.

        Parameters:
        - mapped_smile (str): The mapped SMILES string to check against the tautomers of
        the ground truth.
        - ground_truth (str): The reference SMILES string for generating possible
        tautomers.
        - check_method (str): The method used for checking equivalence. Default is 'RC'.
        Possible values are 'RC' for reaction center or 'ITS'.
        - ignore_aromaticity (bool): Flag to ignore differences in aromaticity between
        the mapped SMILE and the tautomers.Default is False.

        Returns:
        - Optional[bool]: True if the mapped SMILE matches any of the enumerated tautomers
        of the ground truth according to the specified check method.
        Returns False if no match is found.
        Returns None if an error occurs during processing.

        Raises:
        - Exception: If an error occurs during the tautomer enumeration
        or the comparison process.
        """
        try:
            ground_truth_tautomers = enumerate_tautomers(ground_truth)
            return any(
                AAMValidator.smiles_check(
                    mapped_smile, t, check_method, ignore_aromaticity
                )
                for t in ground_truth_tautomers
            )
        except Exception as e:
            print(f"An error occurred: {e}")
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
        Checks the equivalence between the mapped and ground truth
        values within a given mapping dictionary, using a specified check method.
        The check can optionally ignore aromaticity.

        Parameters:
        - mapping (Dict[str, str]): A dictionary containing the data entries to check.
        - mapped_col (str): The key in the mapping dictionary corresponding
        to the mapped value.
        - ground_truth_col (str): The key in the mapping dictionary corresponding
        to the ground truth value.
        - check_method (str, optional): The method used for checking the equivalence.
        Defaults to 'RC'.
        - ignore_aromaticity (bool, optional): Flag to indicate whether aromaticity
        should be ignored during the check. Defaults to False.
        - ignore_tautomers (bool, optional): Flag to indicate whether tautomers
        should be ignored during the check. Defaults to False.

        Returns:
        - bool: The result of the check, indicating whether the mapped value is
        equivalent to the ground truth according to the specified method
        and considerations regarding aromaticity.
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
        ignore_tautomers=True,
    ) -> List[Dict[str, Union[str, float, List[bool]]]]:
        """
        Validates collections of mapped SMILES against their ground truths for
        multiple mappers and calculates the accuracy.

        Parameters:
        - data (Union[pd.DataFrame, List[Dict[str, str]]]):
        The input data containing mapped and ground truth SMILES.
        - id_col (str): The name of the column or key containing the reaction ID.
        - ground_truth_col (str): The name of the column or key containing
        the ground truth SMILES.
        - mapped_cols (List[str]): The list of columns or keys containing
        the mapped SMILES for different mappers.
        - check_method (str): The method used for validation ('RC' or 'ITS').
        - ignore_aromaticity (bool): Flag to ignore aromaticity in ITS graph construction.
        - n_jobs (int): The number of parallel jobs to run.
        - verbose (int): The verbosity level for joblib's parallel execution.

        Returns:
        - List[Dict[str, Union[str, float, List[bool]]]]: A list of dictionaries, each
        containing the mapper name, accuracy, and individual results for each SMILES pair.
        """

        validation_results = []

        for mapped_col in mapped_cols:

            if isinstance(data, pd.DataFrame):
                mappings = data.to_dict("records")
            elif isinstance(data, list):
                mappings = data
            else:
                raise ValueError(
                    "Data must be either a pandas DataFrame or a list of dictionaries."
                )

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
            accuracy = sum(results) / len(mappings) if mappings else 0
            mapped_data = [value[mapped_col] for value in mappings]

            validation_results.append(
                {
                    "mapper": mapped_col,
                    "accuracy": round(100 * accuracy, 2),
                    "results": results,
                    "success_rate": mapping_success_rate(mapped_data),
                }
            )

        return validation_results
