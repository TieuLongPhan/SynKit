import pandas as pd
import networkx as nx
from operator import eq
from itertools import combinations
from functools import wraps
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union, Optional
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.IO.chem_converter import rsmi_to_graph
from synkit.Graph.ITS.its_decompose import get_rc
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Chem.utils import enumerate_tautomers, mapping_success_rate


class _DualMethod:
    """Bind a method to either an instance or the class."""

    def __init__(self, func):
        self.func = func
        wraps(func)(self)

    def __get__(self, obj, cls=None):
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.func(obj, cls, *args, **kwargs)

        return wrapper


class AAMValidator:
    """A utility class for validating atom‐atom mappings (AAM) in reaction
    SMILES.

    Provides methods to compare mapped SMILES against ground truth by
    using reaction‐center (RC) or ITS‐graph isomorphism checks, including
    tautomer enumeration support and batch validation over tabular data.

    Quick start
    -----------
    >>> from synkit.Chem.Reaction.Mapper import AAMValidator
    >>> validator = AAMValidator()
    >>> rsmi_1 = (
       '[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][OH:6]'
       '>>'
       '[CH3:1][C:2](=[O:3])[O:6][CH3:5].[OH2:4]')
    >>> rsmi_2 = (
       '[CH3:5][C:1](=[O:2])[OH:3].[CH3:6][OH:4]'
       '>>'
       '[CH3:5][C:1](=[O:2])[O:4][CH3:6].[OH2:3]')
    >>> is_eq = validator.smiles_check(rsmi_1, rsmi_2, check_method='ITS')
    >>> print(is_eq)
    >>> True
    """

    def __init__(self, strip_unbalanced_maps: bool = True) -> None:
        """Initialize the AAMValidator.

        :param strip_unbalanced_maps: If True, remove atoms whose atom-map
            labels are present on only one side before building ITS/RC graphs.
            This reproduces the more permissive SynRXN benchmark behavior.
        :type strip_unbalanced_maps: bool
        """
        self.strip_unbalanced_maps = strip_unbalanced_maps

    @staticmethod
    def _resolve_strip_unbalanced_maps(
        instance: Optional["AAMValidator"],
        strip_unbalanced_maps: Optional[bool],
    ) -> bool:
        """Resolve call-level override against the instance default."""
        if strip_unbalanced_maps is not None:
            return strip_unbalanced_maps
        if instance is not None:
            return instance.strip_unbalanced_maps
        return True

    @staticmethod
    def _node_atom_map(graph: nx.Graph, node: object) -> Optional[int]:
        """Return a node's atom-map label when available."""
        value = graph.nodes[node].get("atom_map", node)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _strip_unbalanced_map_nodes(
        reactant_graph: nx.Graph,
        product_graph: nx.Graph,
    ) -> Tuple[nx.Graph, nx.Graph]:
        """Remove nodes whose atom maps are not present on both sides."""
        reactant_maps = {
            atom_map
            for node in reactant_graph.nodes
            if (atom_map := AAMValidator._node_atom_map(reactant_graph, node))
            is not None
        }
        product_maps = {
            atom_map
            for node in product_graph.nodes
            if (atom_map := AAMValidator._node_atom_map(product_graph, node))
            is not None
        }
        common_maps = reactant_maps & product_maps

        stripped_reactant = reactant_graph.copy()
        stripped_product = product_graph.copy()
        stripped_reactant.remove_nodes_from(
            [
                node
                for node in stripped_reactant.nodes
                if AAMValidator._node_atom_map(stripped_reactant, node)
                not in common_maps
            ]
        )
        stripped_product.remove_nodes_from(
            [
                node
                for node in stripped_product.nodes
                if AAMValidator._node_atom_map(stripped_product, node)
                not in common_maps
            ]
        )
        return stripped_reactant, stripped_product

    @staticmethod
    def check_equivariant_graph(
        its_graphs: List[nx.Graph],
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Identify all pairs of isomorphic ITS graphs.

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

    @_DualMethod
    def smiles_check(
        self,
        cls,
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        strip_unbalanced_maps: Optional[bool] = None,
    ) -> bool:
        """Validate a single mapped SMILES string against ground truth.

        :param mapped_smile: The mapped SMILES to validate.
        :type mapped_smile: str
        :param ground_truth: The reference SMILES string.
        :type ground_truth: str
        :param check_method: Which method to use: `"RC"` for
            reaction‐center graph or `"ITS"` for full ITS‐graph
            isomorphism.
        :type check_method: str
        :param ignore_aromaticity: If True, ignore aromaticity
            differences in ITS construction.
        :type ignore_aromaticity: bool
        :param strip_unbalanced_maps: If True, remove atom-map labels present
            on only one side before graph comparison. If None, use the
            instance default or False for class-level calls.
        :type strip_unbalanced_maps: bool or None
        :returns: True if exactly one isomorphic match is found; False
            otherwise.
        :rtype: bool
        """
        its_graphs, rc_graphs = [], []
        strip_unbalanced_maps = cls._resolve_strip_unbalanced_maps(
            self, strip_unbalanced_maps
        )
        try:
            for rsmi in (mapped_smile, ground_truth):
                G, H = rsmi_to_graph(rsmi=rsmi, sanitize=True, drop_non_aam=True)
                if strip_unbalanced_maps:
                    G, H = cls._strip_unbalanced_map_nodes(G, H)
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

    @_DualMethod
    def smiles_check_tautomer(
        self,
        cls,
        mapped_smile: str,
        ground_truth: str,
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        strip_unbalanced_maps: Optional[bool] = None,
    ) -> Optional[bool]:
        """Validate against all tautomers of a ground truth SMILES.

        :param mapped_smile: The mapped SMILES to test.
        :type mapped_smile: str
        :param ground_truth: The reference SMILES for generating tautomers.
        :type ground_truth: str
        :param check_method: `"RC"` or `"ITS"` as in `smiles_check`.
        :type check_method: str
        :param ignore_aromaticity: If True, ignore aromaticity in ITS construction.
        :type ignore_aromaticity: bool
        :param strip_unbalanced_maps: If True, remove atom-map labels present
            on only one side before graph comparison. If None, use the
            instance default or False for class-level calls.
        :type strip_unbalanced_maps: bool or None
        :returns:
            - `True` if any tautomer matches.
            - `False` if none match.
            - `None` if an error occurs.
        :rtype: bool or None
        """
        strip_unbalanced_maps = cls._resolve_strip_unbalanced_maps(
            self, strip_unbalanced_maps
        )
        try:
            tautomers = enumerate_tautomers(ground_truth)
            return any(
                cls.smiles_check(
                    mapped_smile,
                    taut,
                    check_method,
                    ignore_aromaticity,
                    strip_unbalanced_maps,
                )
                for taut in tautomers
            )
        except Exception:
            return None

    @_DualMethod
    def check_pair(
        self,
        cls,
        mapping: Dict[str, str],
        mapped_col: str,
        ground_truth_col: str,
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        ignore_tautomers: bool = True,
        strip_unbalanced_maps: Optional[bool] = None,
    ) -> bool:
        """Validate a single record (dict) entry for equivalence.

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
        :param strip_unbalanced_maps: If True, remove atom-map labels present
            on only one side before graph comparison. If None, use the
            instance default or False for class-level calls.
        :type strip_unbalanced_maps: bool or None
        :returns: Validation result for this single pair.
        :rtype: bool
        """
        strip_unbalanced_maps = cls._resolve_strip_unbalanced_maps(
            self, strip_unbalanced_maps
        )
        if ignore_tautomers:
            return cls.smiles_check(
                mapping[mapped_col],
                mapping[ground_truth_col],
                check_method,
                ignore_aromaticity,
                strip_unbalanced_maps,
            )
        else:
            return cls.smiles_check_tautomer(
                mapping[mapped_col],
                mapping[ground_truth_col],
                check_method,
                ignore_aromaticity,
                strip_unbalanced_maps,
            )

    @_DualMethod
    def validate_smiles(
        self,
        cls,
        data: Union[pd.DataFrame, List[Dict[str, str]]],
        ground_truth_col: str = "ground_truth",
        mapped_cols: List[str] = ["rxn_mapper", "graphormer", "local_mapper"],
        check_method: str = "RC",
        ignore_aromaticity: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
        ignore_tautomers: bool = True,
        strip_unbalanced_maps: Optional[bool] = None,
    ) -> List[Dict[str, Union[str, float, List[bool]]]]:
        """Batch-validate mapped SMILES in tabular or list-of-dicts form.

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
        :param strip_unbalanced_maps: If True, remove atom-map labels present
            on only one side before graph comparison. If None, use the
            instance default or False for class-level calls.
        :type strip_unbalanced_maps: bool or None
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

        strip_unbalanced_maps = cls._resolve_strip_unbalanced_maps(
            self, strip_unbalanced_maps
        )
        for mapped_col in mapped_cols:
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(cls.check_pair)(
                    mapping,
                    mapped_col,
                    ground_truth_col,
                    check_method,
                    ignore_aromaticity,
                    ignore_tautomers,
                    strip_unbalanced_maps,
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
