import itertools
import networkx as nx
from copy import deepcopy, copy
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Iterable, Optional

from synkit.IO.debug import setup_logging
from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.Hyrogen._misc import (
    check_hcount_change,
    check_explicit_hydrogen,
    get_priority,
    check_equivariant_graph,
)


logger = setup_logging()


class HComplete:
    """
    A class for infering hydrogen to complete reaction center or ITS graph.
    """

    @staticmethod
    def process_single_graph_data(
        graph_data: Dict[str, nx.Graph],
        its_key: str = "ITS",
        rc_key: str = "RC",
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        get_priority_graph: bool = False,
        max_hydrogen: int = 7,
    ) -> Dict[str, Optional[nx.Graph]]:
        """
        Processes a single graph data dictionary by modifying hydrogen counts
        and other features based on configuration settings.

        Parameters:
        - graph_data (Dict[str, nx.Graph]): Dictionary containing the graph data.
        - its_key (str): Key where the ITS graph is stored.
        - rc_key (str): Key where the RC graph is stored.
        - ignore_aromaticity (bool): If True, aromaticity is ignored during processing. Default is False.
        - balance_its (bool): If True, the ITS is balanced. Default is True.
        - get_priority_graph (bool): If True, priority is given to graph data during processing. Default is False.
        - max_hydrogen (int): Maximum number of hydrogens that can be handled in the inference step.

        Returns:
        - Dict[str, Optional[nx.Graph]]: Dictionary with updated ITS and RC graph data, or None if processing fails.
        """
        graphs = copy(graph_data)
        its = graphs.get(its_key, None)
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            graphs[its_key], graphs[rc_key] = None, None
            return graphs
        react_graph, prod_graph = its_decompose(its)
        hcount_change = check_hcount_change(react_graph, prod_graph)
        if hcount_change == 0:
            graphs = graphs
        elif hcount_change <= max_hydrogen:
            graphs = HComplete.process_multiple_hydrogens(
                graphs,
                its_key,
                rc_key,
                react_graph,
                prod_graph,
                ignore_aromaticity,
                balance_its,
                get_priority_graph,
            )
        else:
            graphs[its_key], graphs[rc_key] = None, None
        if graphs[rc_key] is not None:
            is_empty_rc_present = (
                not isinstance(graphs[rc_key], nx.Graph)
                or graphs[rc_key].number_of_nodes() == 0
            )

            if is_empty_rc_present:
                graphs[its_key] = None
                graphs[rc_key] = None
        return graphs

    def process_graph_data_parallel(
        self,
        graph_data_list: List[Dict[str, nx.Graph]],
        its_key: str = "ITS",
        rc_key: str = "RC",
        n_jobs: int = 1,
        verbose: int = 0,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        get_priority_graph: bool = False,
        max_hydrogen: int = 7,
    ) -> List[Dict[str, Optional[nx.Graph]]]:
        """
        Processes a list of graph data dictionaries in parallel to optimize
        the hydrogen completion and other graph modifications.

        Parameters:
        - graph_data_list (List[Dict[str, nx.Graph]]): List of dictionaries containing the graph data.
        - its_key (str): Key where the ITS graph is stored.
        - rc_key (str): Key where the RC graph is stored.
        - n_jobs (int): Number of parallel jobs to run.
        - verbose (int): Verbosity level for the parallel process.
        - ignore_aromaticity (bool): If True, aromaticity is ignored during processing. Default is False.
        - balance_its (bool): If True, the ITS is balanced. Default is True.
        - get_priority_graph (bool): If True, priority is given to graph data during processing. Default is False.
        - max_hydrogen (int): Maximum number of hydrogens that can be handled in the inference step.

        Returns:
        - List[Dict[str, Optional[nx.Graph]]]: List of dictionaries with
        updated ITS and RC graph data, or None if processing fails.
        """
        processed_data = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.process_single_graph_data)(
                graph_data,
                its_key,
                rc_key,
                ignore_aromaticity,
                balance_its,
                get_priority_graph,
                max_hydrogen,
            )
            for graph_data in graph_data_list
        )

        return processed_data

    @staticmethod
    def process_multiple_hydrogens(
        graph_data: Dict[str, nx.Graph],
        its_key: str,
        rc_key: str,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        get_priority_graph: bool = False,
    ) -> Dict[str, Optional[nx.Graph]]:
        """
        Handles significant hydrogen count changes between reactant and product graphs,
        adjusting hydrogen nodes accordingly and assessing graph equivalence.

        Parameters:
        - graph_data (Dict[str, nx.Graph]): Dictionary containing the graph data.
        - its_key (str): Key for the ITS graph in the dictionary.
        - rc_key (str): Key for the RC graph in the dictionary.
        - react_graph (nx.Graph): Graph representing the reactants.
        - prod_graph (nx.Graph): Graph representing the products.
        - ignore_aromaticity (bool): If True, aromaticity will not be considered in processing.
        - balance_its (bool): If True, balances the ITS graph.
        - get_priority_graph (bool): If True, processes graphs with priority considerations.

        Returns:
        - Dict[str, Optional[nx.Graph]]: Updated graph dictionary with potentially modified ITS and RC graphs.
        """
        combinations_solution = HComplete.add_hydrogen_nodes_multiple(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            get_priority_graph,
        )
        if len(combinations_solution) == 0:
            graph_data[its_key], graph_data[rc_key] = None, None
            return graph_data

        filtered_combinations_solution = []
        react_list = []
        prod_list = []
        rc_list = []
        its_list = []
        rc_sig = []

        for react, prod, its, rc, sig in combinations_solution:
            if rc is not None and isinstance(rc, nx.Graph) and rc.number_of_nodes() > 0:
                filtered_combinations_solution.append((react, prod, rc, its, sig))
                react_list.append(react)
                prod_list.append(prod)
                rc_list.append(rc)
                its_list.append(its)
                rc_sig.append(sig)

        if len(set(rc_sig)) != 1:
            equivariant = 0
        else:
            _, equivariant = check_equivariant_graph(rc_list)

        pairwise_combinations = len(rc_list) - 1
        if equivariant == pairwise_combinations:
            graph_data[its_key] = its_list[0]
            graph_data[rc_key] = rc_list[0]
        else:
            graph_data[its_key], graph_data[rc_key] = None, None
            if get_priority_graph:
                priority_indices = get_priority(rc_list)
                rc_list = [rc_list[i] for i in priority_indices]
                rc_sig = [rc_sig[i] for i in priority_indices]
                its_list = [its_list[i] for i in priority_indices]
                react_list = [react_list[i] for i in priority_indices]
                prod_list = [prod_list[i] for i in priority_indices]
                if len(set(rc_sig)) == 1:
                    _, equivariant = check_equivariant_graph(rc_list)
                pairwise_combinations = len(rc_list) - 1
                if equivariant == pairwise_combinations:
                    graph_data[its_key] = its_list[0]
                    graph_data[rc_key] = rc_list[0]
        return graph_data

    @staticmethod
    def add_hydrogen_nodes_multiple(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        get_priority_graph: bool = False,
    ) -> List[Tuple[nx.Graph, nx.Graph]]:
        """
        Generates multiple permutations of reactant and product graphs by adjusting hydrogen counts,
        exploring all possible configurations of hydrogen node additions or removals.

        Parameters:
        - react_graph (nx.Graph): The reactant graph.
        - prod_graph (nx.Graph): The product graph.
        - ignore_aromaticity (bool): If True, aromaticity is ignored.
        - balance_its (bool): If True, attempts to balance the ITS by adjusting hydrogen nodes.
        - get_priority_graph (bool): If True, additional priority-based processing
        is applied to select optimal graph configurations.

        Returns:
        - List[Tuple[nx.Graph, nx.Graph]]: A list of graph tuples, each representing
        a possible configuration of reactant and product graphs with adjusted hydrogen nodes.
        """
        react_graph_copy = react_graph.copy()
        prod_graph_copy = prod_graph.copy()
        react_explicit_h, hydrogen_nodes = check_explicit_hydrogen(react_graph_copy)
        prod_explicit_h, _ = check_explicit_hydrogen(prod_graph_copy)
        hydrogen_nodes_form, hydrogen_nodes_break = [], []

        primary_graph = (
            react_graph_copy if react_explicit_h <= prod_explicit_h else prod_graph_copy
        )
        for node_id in primary_graph.nodes:
            try:
                # Calculate the difference in hydrogen counts
                hcount_diff = react_graph_copy.nodes[node_id].get(
                    "hcount", 0
                ) - prod_graph_copy.nodes[node_id].get("hcount", 0)
            except KeyError:
                # Handle cases where node_id does not exist in opposite_graph
                continue

            # Decide action based on hcount_diff
            if hcount_diff > 0:
                hydrogen_nodes_break.extend([node_id] * hcount_diff)
            elif hcount_diff < 0:
                hydrogen_nodes_form.extend([node_id] * -hcount_diff)

        max_index = max(
            max(react_graph_copy.nodes, default=0),
            max(prod_graph_copy.nodes, default=0),
        )
        range_implicit_h = range(
            max_index + 1,
            max_index + 1 + len(hydrogen_nodes_form) - react_explicit_h,
        )
        combined_indices = list(range_implicit_h) + hydrogen_nodes
        permutations = list(itertools.permutations(combined_indices))
        permutations_seed = permutations[0]

        updated_graphs = []
        for permutation in permutations:
            current_react_graph, current_prod_graph = react_graph_copy, prod_graph_copy

            new_hydrogen_node_ids = [i for i in permutations_seed]

            # Use `zip` to pair `hydrogen_nodes_break` with the new IDs
            node_id_pairs = zip(hydrogen_nodes_break, new_hydrogen_node_ids)
            # Call the method with the formed pairs and specify atom_map_update as False
            current_react_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                current_react_graph, node_id_pairs, atom_map_update=False
            )
            # Varied hydrogen nodes in the product graph based on permutation
            current_prod_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                current_prod_graph, zip(hydrogen_nodes_form, permutation)
            )
            its = ITSConstruction().ITSGraph(
                current_react_graph,
                current_prod_graph,
                ignore_aromaticity=ignore_aromaticity,
                balance_its=balance_its,
            )
            rc = get_rc(its)
            sig = WLHash(iterations=3).weisfeiler_lehman_graph_hash(rc)
            if get_priority_graph is False:
                if len(updated_graphs) > 0:
                    if sig != updated_graphs[-1][-1]:
                        return []
            updated_graphs.append(
                (current_react_graph, current_prod_graph, its, rc, sig)
            )
        return updated_graphs

    @staticmethod
    def add_hydrogen_nodes_multiple_utils(
        graph: nx.Graph,
        node_id_pairs: Iterable[Tuple[int, int]],
        atom_map_update: bool = True,
    ) -> nx.Graph:
        """
        Creates and returns a new graph with added hydrogen nodes based on the input graph
        and node ID pairs.

        Parameters:
        - graph (nx.Graph): The base graph to which the nodes will be added.
        - node_id_pairs (Iterable[Tuple[int, int]]): Pairs of node IDs (original node, new
        hydrogen node) to link with hydrogen.
        - atom_map_update (bool): If True, update the 'atom_map' attribute with the new
        hydrogen node ID; otherwise, retain the original node's 'atom_map'.

        Returns:
        - nx.Graph: A new graph instance with the added hydrogen nodes.
        """
        new_graph = deepcopy(graph)
        for node_id, new_hydrogen_node_id in node_id_pairs:
            atom_map_val = (
                new_hydrogen_node_id
                if atom_map_update
                else new_graph.nodes[node_id].get("atom_map", 0)
            )
            new_graph.add_node(
                new_hydrogen_node_id,
                charge=0,
                hcount=0,
                aromatic=False,
                element="H",
                atom_map=atom_map_val,
                # isomer="N",
                # partial_charge=0,
                # hybridization=0,
                # in_ring=False,
                # explicit_valence=0,
                # implicit_hcount=0,
            )
            new_graph.add_edge(
                node_id,
                new_hydrogen_node_id,
                order=1.0,
                # ez_isomer="N",
                bond_type="SINGLE",
                # conjugated=False,
                # in_ring=False,
            )
        return new_graph
