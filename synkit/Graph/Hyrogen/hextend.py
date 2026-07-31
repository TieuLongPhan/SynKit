import networkx as nx
from joblib import Parallel, delayed
from typing import Dict, Iterator, List, Optional, Tuple

from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Graph.Hyrogen.hcomplete import HComplete, ITSFormatInput

from synkit.Graph.Hyrogen._misc import check_hcount_change

cluster = GraphCluster()


class HExtend(HComplete):

    @staticmethod
    def get_unique_graphs_for_clusters(
        graphs: List[nx.Graph], cluster_indices: List[set]
    ) -> List[nx.Graph]:
        """Retrieve a unique graph for each cluster from a list of graphs based
        on cluster indices.

        This method selects one graph per cluster using the smallest index in
        each cluster set. Clusters are expected to be represented as sets of
        indices, each corresponding to a graph in the `graphs` list.

        Parameters:
        - graphs (List[nx.Graph]): List of networkx graphs.
        - cluster_indices (List[set]): List of sets, each containing indices representing graphs
        that belong to the same cluster.

        Returns:
        - List[nx.Graph]: A list containing one unique graph from each cluster.

        Raises:
        - ValueError: If any index in `cluster_indices` is out of the range of `graphs`.
        - TypeError: If `cluster_indices` is not a list of sets.
        """
        if not all(isinstance(cluster, set) for cluster in cluster_indices):
            raise TypeError("Each cluster index must be a set of integers.")
        if any(
            min(cluster) < 0 or max(cluster) >= len(graphs)
            for cluster in cluster_indices
            if cluster
        ):
            raise ValueError("Cluster indices are out of the range of the graphs list.")

        unique_graphs = [graphs[min(cluster)] for cluster in cluster_indices if cluster]
        return unique_graphs

    @staticmethod
    def extend_its(
        its: nx.Graph,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        """Extend a bare ITS graph into all valid hydrogen-completed variants.

        Parameters:
        - its (nx.Graph): The initial transition state graph to be processed.
        - ignore_aromaticity (bool): Flag to ignore aromaticity in graph construction.
        - balance_its (bool): Flag to balance the ITS graph during processing.
        - format (str): ITS representation: "auto", "typesGH", or "tuple".
        - max_candidates (Optional[int]): Optional cap for enumerated hydrogen assignments.

        Returns:
        - Tuple[List[nx.Graph], List[nx.Graph], List[str]]: Tuple containing lists of
        processed reaction graphs, ITS graphs, and their signatures.
        """
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            return [], [], []

        resolved_format = HComplete._resolve_format(its, format)
        react_graph, prod_graph = HComplete._decompose_its(its, resolved_format)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        if hcount_change == 0:
            rc = HComplete._extract_rc(its, resolved_format)
            if not HComplete._valid_rc(rc):
                return [], [], []
            its_list = [its]
            rc_list = [rc]
            sigs = [HComplete._rc_signature(rc, resolved_format)]
            return rc_list, its_list, sigs

        combinations_solution = HComplete._iter_hydrogen_node_completions(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            format=resolved_format,
            max_candidates=max_candidates,
        )

        rc_list, its_list, rc_sig = [], [], []
        for _, _, its, rc, sig in combinations_solution:
            if HComplete._valid_rc(rc):
                rc_list.append(rc)
                its_list.append(its)
                rc_sig.append(sig)
        return rc_list, its_list, rc_sig

    @staticmethod
    def _extend(
        its: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        """Compatibility wrapper for :meth:`extend_its`."""
        return HExtend.extend_its(
            its,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            format=format,
            max_candidates=max_candidates,
        )

    @staticmethod
    def _extend_unique(
        its: nx.Graph,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        """Extend an ITS graph and retain exact RC-class representatives."""
        representatives = list(
            HExtend.iter_unique_completions(
                its,
                ignore_aromaticity=ignore_aromaticity,
                balance_its=balance_its,
                format=format,
                max_candidates=max_candidates,
            )
        )
        return (
            [item[0] for item in representatives],
            [item[1] for item in representatives],
            [item[2] for item in representatives],
        )

    @staticmethod
    def iter_unique_completions(
        its: nx.Graph,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Iterator[Tuple[nx.Graph, nx.Graph, str]]:
        """Yield one completed ``(RC, ITS, signature)`` per exact RC class."""
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            return

        resolved_format = HComplete._resolve_format(its, format)
        react_graph, prod_graph = HComplete._decompose_its(its, resolved_format)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        if hcount_change == 0:
            rc = HComplete._extract_rc(its, resolved_format)
            if not HComplete._valid_rc(rc):
                return
            yield rc, its, HComplete._rc_signature(rc, resolved_format)
            return

        representatives = []
        for (
            _,
            _,
            completed_its,
            rc,
            signature,
        ) in HComplete._iter_hydrogen_node_completions(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            format=resolved_format,
            max_candidates=max_candidates,
        ):
            if not HComplete._valid_rc(rc):
                continue

            duplicate = False
            for representative_rc, representative_signature in representatives:
                if signature != representative_signature:
                    continue
                if (
                    HComplete._equivariant_count(
                        [representative_rc, rc], resolved_format
                    )
                    == 1
                ):
                    duplicate = True
                    break
            if not duplicate:
                representatives.append((rc, signature))
                yield rc, completed_its, signature

    @staticmethod
    def _process(
        data_dict: Dict,
        its_key: str,
        rc_key: str,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Dict:
        """Processes a dictionary of graphs using specific graph processing
        functions and updates the dictionary with new graph data.

        Parameters:
        - data_dict (Dict): Dictionary containing the graphs and their keys.
        - its_key (str): Key in the dictionary for the ITS graph.
        - rc_key (str): Key in the dictionary for the reaction graph.
        - ignore_aromaticity (bool): Whether to ignore aromaticity
        during graph processing.
        - balance_its (bool): Whether to balance the ITS graph.
        - format (str): ITS representation: "auto", "typesGH", or "tuple".
        - max_candidates (Optional[int]): Optional cap for enumerated hydrogen assignments.

        Returns:
        - Dict: The updated dictionary containing new ITS and reaction graphs.
        """
        its = data_dict[its_key]
        resolved_format = HComplete._resolve_format(its, format)
        rc_list, its_list, _ = HExtend._extend_unique(
            its, ignore_aromaticity, balance_its, resolved_format, max_candidates
        )
        if not rc_list:
            data_dict[rc_key] = []
            data_dict[its_key] = []
            return data_dict
        data_dict[rc_key] = rc_list
        data_dict[its_key] = its_list
        return data_dict

    @staticmethod
    def fit(
        data,
        its_key: str,
        rc_key: str,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        n_jobs: int = 1,
        verbose: int = 0,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
        backend: Optional[str] = None,
    ) -> List:
        """Fit the model to the data in parallel, processing each entry to
        generate new graph data based on the ITS and reaction graph keys.

        Parameters:
        - data (iterable): Data to be processed.
        - its_key (str): Key for the ITS graphs in the data.
        - rc_key (str): Key for the reaction graphs in the data.
        - ignore_aromaticity (bool): Whether to ignore aromaticity during processing.
        Default to False.
        - balance_its (bool): Whether to balance the ITS during processing.
        Default to True.
        - n_jobs (int): Number of jobs to run in parallel. Default to 1.
        - verbose (int): Verbosity level for parallel processing. Default to 0.
        - format (str): ITS representation: "auto", "typesGH", or "tuple".
        - max_candidates (Optional[int]): Optional cap for enumerated hydrogen assignments.
        - backend (Optional[str]): Optional joblib backend.

        Returns:
        - List: A list containing the results of the processed data.
        """
        if n_jobs == 1 and backend is None:
            return [
                HExtend._process(
                    item,
                    its_key,
                    rc_key,
                    ignore_aromaticity,
                    balance_its,
                    format,
                    max_candidates,
                )
                for item in data
            ]

        parallel_kwargs = {"n_jobs": n_jobs, "verbose": verbose}
        if backend is not None:
            parallel_kwargs["backend"] = backend

        results = Parallel(**parallel_kwargs)(
            delayed(HExtend._process)(
                item,
                its_key,
                rc_key,
                ignore_aromaticity,
                balance_its,
                format,
                max_candidates,
            )
            for item in data
        )
        return results
