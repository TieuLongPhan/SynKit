import networkx as nx
from joblib import Parallel, delayed
from typing import List, Tuple, Dict

from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Graph.Hyrogen.hcomplete import HComplete

from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.Hyrogen._misc import check_hcount_change
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose


cluster = GraphCluster()


class HExtend(HComplete):

    @staticmethod
    def get_unique_graphs_for_clusters(
        graphs: List[nx.Graph], cluster_indices: List[set]
    ) -> List[nx.Graph]:
        """
        Retrieve a unique graph for each cluster from a list of graphs based on cluster indices.

        This method selects one graph per cluster based on the first index found
        in each cluster set. Note: Clusters are expected to be represented
        as sets of indices, each corresponding to a graph in the `graphs` list.

        Parameters:
        - graphs (List[nx.Graph]): List of networkx graphs.
        - cluster_indices (List[set]): List of sets, each containing indices representing graphs
        that belong to the same cluster.

        Returns:
        - List[nx.Graph]: A list containing one unique graph from each cluster. The graph chosen
        is the one corresponding to the first index in each cluster set, which is arbitrary
        due to the unordered nature of sets.

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

        unique_graphs = [
            graphs[next(iter(cluster))] for cluster in cluster_indices if cluster
        ]
        return unique_graphs

    @staticmethod
    def _extend(
        its: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        """
        Process equivalent maps by adding hydrogen nodes and constructing
        ITS graphs based on the balance and aromaticity settings.

        Parameters:
        - its (nx.Graph): The initial transition state graph to be processed.
        - ignore_aromaticity (bool): Flag to ignore aromaticity in graph construction.
        - balance_its (bool): Flag to balance the ITS graph during processing.

        Returns:
        - Tuple[List[nx.Graph], List[nx.Graph], List[str]]: Tuple containing lists of
        processed reaction graphs, ITS graphs, and their signatures.
        """
        react_graph, prod_graph = its_decompose(its)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        if hcount_change == 0:
            its_list = [its]
            rc_list = [get_rc(its)]
            sigs = [
                WLHash(iterations=3).weisfeiler_lehman_graph_hash(i) for i in rc_list
            ]
            return rc_list, its_list, sigs

        combinations_solution = HComplete.add_hydrogen_nodes_multiple(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            get_priority_graph=True,
        )

        rc_list, its_list, rc_sig = [], [], []
        for _, _, its, rc, sig in combinations_solution:
            if rc and isinstance(rc, nx.Graph) and rc.number_of_nodes() > 0:
                rc_list.append(rc)
                its_list.append(its)
                rc_sig.append(sig)
        return rc_list, its_list, rc_sig

    @staticmethod
    def _process(
        data_dict: Dict,
        its_key: str,
        rc_key: str,
        ignore_aromaticity: bool,
        balance_its: bool,
    ) -> Dict:
        """
        Processes a dictionary of graphs using specific graph processing functions
        and updates the dictionary with new graph data.

        Parameters:
        - data_dict (Dict): Dictionary containing the graphs and their keys.
        - its_key (str): Key in the dictionary for the ITS graph.
        - rc_key (str): Key in the dictionary for the reaction graph.
        - ignore_aromaticity (bool): Whether to ignore aromaticity
        during graph processing.
        - balance_its (bool): Whether to balance the ITS graph.

        Returns:
        - Dict: The updated dictionary containing new ITS and reaction graphs.
        """
        its = data_dict[its_key]
        rc_list, its_list, rc_sig = HExtend._extend(
            its, ignore_aromaticity, balance_its
        )
        cls, _ = cluster.iterative_cluster(rc_list, rc_sig)
        new_rc = HExtend.get_unique_graphs_for_clusters(rc_list, cls)
        new_its = HExtend.get_unique_graphs_for_clusters(its_list, cls)
        data_dict[rc_key] = new_rc
        data_dict[its_key] = new_its
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
    ) -> List:
        """
        Fit the model to the data in parallel, processing each entry to generate
        new graph data based on the ITS and reaction graph keys.

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

        Returns:
        - List: A list containing the results of the processed data.
        """
        results = Parallel(n_jobs=n_jobs, verbose=verbose, backend="multiprocessing")(
            delayed(HExtend._process)(
                item, its_key, rc_key, ignore_aromaticity, balance_its
            )
            for item in data
        )
        return results
