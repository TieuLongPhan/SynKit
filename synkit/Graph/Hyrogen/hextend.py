from collections import Counter
from operator import eq
from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx
from joblib import Parallel, delayed
from networkx.algorithms.isomorphism import generic_node_match

from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Graph.Hyrogen.hcomplete import HComplete, ITSFormatInput
from synkit.Graph.Stereo.matching import stereo_isomorphic

from synkit.Graph.Hyrogen._misc import check_hcount_change

cluster = GraphCluster()
_ANCHOR_ATTRIBUTE = "_synkit_hextend_anchor"
_anchored_node_match = generic_node_match(
    ["element", "charge", _ANCHOR_ATTRIBUTE],
    ["*", 0, None],
    [eq, eq, eq],
)

ColourDistance = Tuple[int, int]
RootedDistanceProfile = Tuple[Tuple[ColourDistance, int], ...]
HydrogenDistanceInvariant = Tuple[RootedDistanceProfile, ...]


def _invariant_order_key(value: Any) -> Tuple[Any, ...]:
    """Totally order supported frozen comparison values without coercion."""
    if isinstance(value, tuple):
        return ("tuple", tuple(_invariant_order_key(item) for item in value))
    return (
        "atom",
        type(value).__module__,
        type(value).__qualname__,
        repr(value),
    )


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
    def cluster_full_its(
        its_list: List[nx.Graph],
        signatures: List[str],
    ) -> Tuple[List[set], Dict[int, int]]:
        """Cluster complete ITS graphs with necessary-condition prefilters.

        Full-ITS isomorphism implies reaction-centre isomorphism, so unequal
        RC-invariant signatures conclusively rule out equivalence. Candidates
        that collide on that signature receive a second invariant describing
        distances from hydrogen atoms to coloured atoms. Equal signatures
        remain only prefilters: the final decision is made by chemistry- and
        stereo-aware full-graph isomorphism.
        """
        if len(its_list) != len(signatures):
            raise ValueError("ITS graphs and signatures must have equal lengths.")
        if not its_list:
            return [], {}

        signature_counts = Counter(signatures)
        invariant_blocks: Dict[
            Tuple[str, HydrogenDistanceInvariant],
            List[int],
        ] = {}
        for index, (its, signature) in enumerate(zip(its_list, signatures)):
            # A unique RC invariant already forms a singleton block. Compute
            # the more expensive rooted-distance invariant only on collisions.
            distance_invariant = (
                ()
                if signature_counts[signature] == 1
                else HExtend.hydrogen_distance_invariant(its)
            )
            invariant_blocks.setdefault(
                (signature, distance_invariant),
                [],
            ).append(index)

        exact_clusters = []
        for indices in invariant_blocks.values():
            exact_clusters.extend(
                HExtend._cluster_invariant_block(
                    indices,
                    its_list,
                )
            )

        exact_clusters.sort(key=min)
        rule_to_cluster = {
            index: cluster_index
            for cluster_index, indices in enumerate(exact_clusters)
            for index in indices
        }
        return exact_clusters, rule_to_cluster

    @staticmethod
    def _cluster_invariant_block(
        indices: List[int],
        its_list: List[nx.Graph],
    ) -> List[set]:
        """Exactly quotient one necessary-invariant block."""
        exact_clusters: List[set] = []
        for index in indices:
            for exact_cluster in exact_clusters:
                representative = min(exact_cluster)
                if HExtend._anchored_full_its_isomorphic(
                    its_list[representative],
                    its_list[index],
                ):
                    exact_cluster.add(index)
                    break
            else:
                exact_clusters.append({index})
        return exact_clusters

    @staticmethod
    def _anchored_full_its_isomorphic(
        left: nx.Graph,
        right: nx.Graph,
    ) -> bool:
        """Decide full ITS isomorphism by changed-core anchor extension.

        Every full ITS isomorphism restricts to an isomorphism of the subgraph
        induced by changed bonds, because the full edge matcher preserves the
        paired bond-order attribute. Conversely, a full isomorphism extending
        any enumerated core isomorphism is a valid witness. Exhaustive anchor
        extension is therefore equivalent to unconstrained full isomorphism.
        """
        left_rc = HExtend._changed_bond_core(left)
        right_rc = HExtend._changed_bond_core(right)
        if (
            left_rc.is_directed() != right_rc.is_directed()
            or left_rc.is_multigraph() != right_rc.is_multigraph()
        ):
            return False
        if (
            left_rc.number_of_nodes() != right_rc.number_of_nodes()
            or left_rc.number_of_edges() != right_rc.number_of_edges()
        ):
            return False
        if left_rc.number_of_nodes() == 0:
            return stereo_isomorphic(
                left,
                right,
                node_match=cluster.nodeMatch,
                edge_match=cluster.edgeMatch,
            )
        if left_rc.is_directed():
            matcher_type = (
                nx.algorithms.isomorphism.MultiDiGraphMatcher
                if left_rc.is_multigraph()
                else nx.algorithms.isomorphism.DiGraphMatcher
            )
        else:
            matcher_type = (
                nx.algorithms.isomorphism.MultiGraphMatcher
                if left_rc.is_multigraph()
                else nx.algorithms.isomorphism.GraphMatcher
            )
        matcher = matcher_type(
            left_rc,
            right_rc,
            node_match=cluster.nodeMatch,
            edge_match=cluster.edgeMatch,
        )
        anchored_left = left.copy()
        anchored_right = right.copy()
        nx.set_node_attributes(anchored_left, None, _ANCHOR_ATTRIBUTE)
        nx.set_node_attributes(anchored_right, None, _ANCHOR_ATTRIBUTE)
        for rc_mapping in matcher.isomorphisms_iter():
            for anchor, (left_node, right_node) in enumerate(rc_mapping.items()):
                anchored_left.nodes[left_node][_ANCHOR_ATTRIBUTE] = anchor
                anchored_right.nodes[right_node][_ANCHOR_ATTRIBUTE] = anchor
            if stereo_isomorphic(
                anchored_left,
                anchored_right,
                node_match=_anchored_node_match,
                edge_match=cluster.edgeMatch,
            ):
                return True
        return False

    @staticmethod
    def _changed_bond_core(its: nx.Graph) -> nx.Graph:
        """Return the isomorphism-invariant subgraph of changed ITS bonds."""
        core = its.__class__()
        for left, right, attributes in its.edges(data=True):
            order = attributes.get("order")
            if (
                not isinstance(order, (tuple, list))
                or len(order) != 2
                or order[0] == order[1]
            ):
                continue
            if left not in core:
                core.add_node(left, **its.nodes[left])
            if right not in core:
                core.add_node(right, **its.nodes[right])
            core.add_edge(left, right, **attributes)
        return core

    @staticmethod
    def hydrogen_distance_invariant(its: nx.Graph) -> HydrogenDistanceInvariant:
        r"""Return the exact hydrogen-rooted distance multiset ``I_H(G)``.

        For every hydrogen vertex ``h``, its profile is
        ``D_G(h) = multiset((colour(v), distance(h, v)) for v in C_G(h))``,
        where ``C_G(h)`` is the connected component containing ``h``.
        ``I_H(G)`` is the multiset of all ``D_G(h)``. Any colour-preserving
        graph isomorphism preserves connected components, vertex colours,
        shortest-path distances, and the hydrogen vertex set, hence preserves
        this value exactly.
        """
        resolved_format = HComplete._resolve_format(its, "auto")
        comparison = HComplete._comparison_graph(its, resolved_format)
        ordered_colours = sorted(
            {
                attributes["cmp_node"]
                for _, attributes in comparison.nodes(data=True)
            },
            key=_invariant_order_key,
        )
        colour_rank = {
            colour: rank for rank, colour in enumerate(ordered_colours)
        }
        profiles: List[RootedDistanceProfile] = []
        for hydrogen, attributes in comparison.nodes(data=True):
            if attributes["cmp_element"] != "H":
                continue
            distances = nx.single_source_shortest_path_length(
                comparison,
                hydrogen,
            )
            histogram = Counter(
                (colour_rank[comparison.nodes[node]["cmp_node"]], distance)
                for node, distance in distances.items()
            )
            profiles.append(tuple(sorted(histogram.items())))
        return tuple(sorted(profiles))

    @staticmethod
    def hydrogen_distance_signature(its: nx.Graph) -> HydrogenDistanceInvariant:
        """Compatibility alias for :meth:`hydrogen_distance_invariant`."""
        return HExtend.hydrogen_distance_invariant(its)

    @staticmethod
    def extend_unique_full_its(
        its: nx.Graph,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        """Return one representative per exact complete-ITS class."""
        rc_list, its_list, signatures = HExtend.extend_its(
            its,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            format=format,
            max_candidates=max_candidates,
        )
        clusters, _ = HExtend.cluster_full_its(its_list, signatures)
        indices = [min(items) for items in clusters if items]
        return (
            [rc_list[index] for index in indices],
            [its_list[index] for index in indices],
            [signatures[index] for index in indices],
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
