import networkx as nx
from dataclasses import dataclass
from copy import copy
from joblib import Parallel, delayed
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Iterable,
    Optional,
)

from synkit.IO.debug import setup_logging
from synkit.IO.chem_converter import detect_its_format
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.rc_extractor import RCExtractor
from synkit.Graph.Hyrogen._misc import (
    check_hcount_change,
    get_priority,
)
from synkit.Graph.Hyrogen.hcompletion_algorithms import (
    HydrogenCompletionAlgorithms,
    ITSFormat,
    ITSFormatInput,
    _HydrogenTransferPlan,
)

logger = setup_logging()


@dataclass
class HCompletionResult:
    """Container returned by graph-first hydrogen completion."""

    its: Optional[nx.Graph]
    rc: Optional[nx.Graph]
    reactant: Optional[nx.Graph] = None
    product: Optional[nx.Graph] = None
    signature: Optional[str] = None
    candidates: int = 0
    reason: str = ""
    format: str = "typesGH"
    exhaustive: bool = True

    @property
    def ok(self) -> bool:
        return (
            isinstance(self.its, nx.Graph)
            and isinstance(self.rc, nx.Graph)
            and self.rc.number_of_nodes() > 0
        )


class HComplete(HydrogenCompletionAlgorithms):
    """Complete reaction-centre hydrogens in ITS graphs."""

    TUPLE_NODE_ATTRS = [
        "element",
        "aromatic",
        "hcount",
        "charge",
        "atom_map",
        "lone_pairs",
        "radical",
        "valence_electrons",
    ]
    TUPLE_EDGE_ATTRS = [
        "order",
        "kekule_order",
        "sigma_order",
        "pi_order",
    ]

    @staticmethod
    def process_single_graph_data(
        graph_data: Dict[str, nx.Graph],
        its_key: str = "ITS",
        rc_key: str = "RC",
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        get_priority_graph: bool = False,
        max_hydrogen: int = 7,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Optional[nx.Graph]]:
        """Processes a single graph data dictionary by modifying hydrogen
        counts and other features based on configuration settings.

        :param graph_data: Dictionary containing the graph data.
        :type graph_data: Dict[str, nx.Graph]
        :param its_key: Key where the ITS graph is stored.
        :type its_key: str
        :param rc_key: Key where the RC graph is stored.
        :type rc_key: str
        :param ignore_aromaticity: If True, aromaticity is ignored during processing. Default is False.
        :type ignore_aromaticity: bool
        :param balance_its: If True, the ITS is balanced. Default is True.
        :type balance_its: bool
        :param get_priority_graph: If True, priority is given to graph data during processing. Default is False.
        :type get_priority_graph: bool
        :param max_hydrogen: Maximum number of hydrogens that can be handled in the inference step.
        :type max_hydrogen: int
        :param format: ITS representation: "auto", "typesGH", or "tuple".
        :type format: str
        :param max_candidates: Optional cap for enumerated hydrogen assignments.
        :type max_candidates: Optional[int]

        :return: Dictionary with updated ITS and RC graph data, or None if processing fails.
        :rtype: Dict[str, Optional[nx.Graph]]
        """
        graphs = copy(graph_data)
        its = graphs.get(its_key, None)
        result = HComplete.complete_its(
            its,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            get_priority_graph=get_priority_graph,
            max_hydrogen=max_hydrogen,
            format=format,
            max_candidates=max_candidates,
        )
        graphs[its_key] = result.its if result.ok else None
        graphs[rc_key] = result.rc if result.ok else None
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
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> List[Dict[str, Optional[nx.Graph]]]:
        """Processes a list of graph data dictionaries in parallel to optimize
        the hydrogen completion and other graph modifications.

        :param graph_data_list: List of dictionaries containing the graph data.
        :type graph_data_list: List[Dict[str, nx.Graph]]
        :param its_key: Key where the ITS graph is stored.
        :type its_key: str
        :param rc_key: Key where the RC graph is stored.
        :type rc_key: str
        :param n_jobs: Number of parallel jobs to run.
        :type n_jobs: int
        :param verbose: Verbosity level for the parallel process.
        :type verbose: int
        :param ignore_aromaticity: If True, aromaticity is ignored during processing. Default is False.
        :type ignore_aromaticity: bool
        :param balance_its: If True, the ITS is balanced. Default is True.
        :type balance_its: bool
        :param get_priority_graph: If True, priority is given to graph data during processing. Default is False.
        :type get_priority_graph: bool
        :param max_hydrogen: Maximum number of hydrogens that can be handled in the inference step.
        :type max_hydrogen: int
        :param format: ITS representation: "auto", "typesGH", or "tuple".
        :type format: str
        :param max_candidates: Optional cap for enumerated hydrogen assignments.
        :type max_candidates: Optional[int]

        :return: List of dictionaries with updated ITS and RC graph data, or None if processing fails.
        :rtype: List[Dict[str, Optional[nx.Graph]]]
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
                format,
                max_candidates,
            )
            for graph_data in graph_data_list
        )

        return processed_data

    @staticmethod
    def complete_its(
        its: nx.Graph,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        get_priority_graph: bool = False,
        max_hydrogen: int = 7,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> HCompletionResult:
        """Complete hydrogens for a bare ITS graph.

        This is the graph-first API. It accepts either legacy ``typesGH`` ITS
        graphs or tuple ITS graphs and returns completed ITS/RC graphs without
        requiring a dictionary entry wrapper.
        """
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            return HCompletionResult(None, None, reason="invalid_its")

        resolved_format = HComplete._resolve_format(its, format)
        react_graph, prod_graph = HComplete._decompose_its(its, resolved_format)
        hcount_change = check_hcount_change(react_graph, prod_graph)

        if hcount_change == 0:
            rc = HComplete._extract_rc(its, resolved_format)
            if not HComplete._valid_rc(rc):
                return HCompletionResult(
                    None,
                    None,
                    react_graph,
                    prod_graph,
                    candidates=1,
                    reason="empty_rc",
                    format=resolved_format,
                )
            return HCompletionResult(
                its,
                rc,
                react_graph,
                prod_graph,
                signature=HComplete._rc_signature(rc, resolved_format),
                candidates=1,
                format=resolved_format,
            )

        if hcount_change > max_hydrogen:
            return HCompletionResult(
                None,
                None,
                react_graph,
                prod_graph,
                reason="max_hydrogen_exceeded",
                format=resolved_format,
            )

        return HComplete._complete_from_side_graphs(
            react_graph,
            prod_graph,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            get_priority_graph=get_priority_graph,
            format=resolved_format,
            max_candidates=max_candidates,
        )

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
        format: ITSFormatInput = "typesGH",
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Optional[nx.Graph]]:
        """Handles significant hydrogen count changes between reactant and
        product graphs, adjusting hydrogen nodes accordingly and assessing
        graph equivalence.

        :param graph_data: Dictionary containing the graph data.
        :type graph_data: Dict[str, nx.Graph]
        :param its_key: Key for the ITS graph in the dictionary.
        :type its_key: str
        :param rc_key: Key for the RC graph in the dictionary.
        :type rc_key: str
        :param react_graph: Graph representing the reactants.
        :type react_graph: nx.Graph
        :param prod_graph: Graph representing the products.
        :type prod_graph: nx.Graph
        :param ignore_aromaticity: If True, aromaticity will not be considered in processing.
        :type ignore_aromaticity: bool
        :param balance_its: If True, balances the ITS graph.
        :type balance_its: bool
        :param get_priority_graph: If True, processes graphs with priority considerations.
        :type get_priority_graph: bool
        :param format: ITS representation: "auto", "typesGH", or "tuple".
        :type format: str
        :param max_candidates: Optional cap for enumerated hydrogen assignments.
        :type max_candidates: Optional[int]

        :return: Updated graph dictionary with potentially modified ITS and RC graphs.
        :rtype: Dict[str, Optional[nx.Graph]]
        """
        result = HComplete._complete_from_side_graphs(
            react_graph,
            prod_graph,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            get_priority_graph=get_priority_graph,
            format=HComplete._resolve_graph_pair_format(
                react_graph, prod_graph, format
            ),
            max_candidates=max_candidates,
        )
        graph_data[its_key] = result.its if result.ok else None
        graph_data[rc_key] = result.rc if result.ok else None
        return graph_data

    @staticmethod
    def _complete_from_side_graphs(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        get_priority_graph: bool,
        format: ITSFormat,
        max_candidates: Optional[int] = None,
    ) -> HCompletionResult:
        if get_priority_graph:
            return HComplete._complete_priority_from_transfer_plans(
                react_graph,
                prod_graph,
                ignore_aromaticity,
                balance_its,
                format,
                max_candidates,
            )
        if format == "typesGH":
            return HComplete._complete_typesgh_from_transfer_plans(
                react_graph,
                prod_graph,
                ignore_aromaticity,
                balance_its,
                max_candidates,
            )
        return HComplete._complete_exact_from_transfer_plans(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            format,
            max_candidates,
        )

    @staticmethod
    def _complete_typesgh_from_transfer_plans(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        max_candidates: Optional[int],
    ) -> HCompletionResult:
        """Decide typesGH completion using an exact necessary RC projection."""
        first_plan = None
        first_projection = None
        first_candidate = None
        valid_seen = 0

        for candidate_index, plan in enumerate(
            HComplete._iter_hydrogen_transfer_plans(react_graph, prod_graph)
        ):
            if max_candidates is not None and candidate_index >= max_candidates:
                return HComplete._candidate_limit_result(
                    react_graph, prod_graph, valid_seen, "typesGH"
                )

            projection = HComplete._typesgh_plan_comparison_graph(
                react_graph,
                prod_graph,
                plan,
                ignore_aromaticity,
            )
            if projection.number_of_nodes() == 0:
                continue

            valid_seen += 1
            if first_plan is None:
                first_plan = plan
                first_projection = projection
                continue

            # Full typesGH RC isomorphism preserves this labeled projection.
            # Its non-isomorphism is therefore a conclusive ambiguity witness.
            if not HComplete._comparison_graphs_isomorphic(
                first_projection, projection
            ):
                return HCompletionResult(
                    None,
                    None,
                    react_graph,
                    prod_graph,
                    candidates=valid_seen,
                    reason="non_equivariant_rc",
                    format="typesGH",
                )

            if first_candidate is None:
                first_candidate = HComplete._materialize_transfer_candidate(
                    react_graph,
                    prod_graph,
                    first_plan,
                    ignore_aromaticity,
                    balance_its,
                    "typesGH",
                )
                if not HComplete._valid_rc(first_candidate[3]):
                    first_candidate = None
                    continue

            candidate = HComplete._materialize_transfer_candidate(
                react_graph,
                prod_graph,
                plan,
                ignore_aromaticity,
                balance_its,
                "typesGH",
            )
            if not HComplete._valid_rc(candidate[3]):
                continue
            if (
                HComplete._equivariant_count(
                    [first_candidate[3], candidate[3]], "typesGH"
                )
                != 1
            ):
                return HCompletionResult(
                    None,
                    None,
                    react_graph,
                    prod_graph,
                    candidates=valid_seen,
                    reason="non_equivariant_rc",
                    format="typesGH",
                )

        if first_plan is None:
            return HCompletionResult(
                None,
                None,
                react_graph,
                prod_graph,
                candidates=valid_seen,
                reason="no_valid_candidate",
                format="typesGH",
            )
        if first_candidate is None:
            first_candidate = HComplete._materialize_transfer_candidate(
                react_graph,
                prod_graph,
                first_plan,
                ignore_aromaticity,
                balance_its,
                "typesGH",
            )
        return HComplete._completion_result(
            first_candidate,
            react_graph,
            prod_graph,
            valid_seen,
            "typesGH",
        )

    @staticmethod
    def _complete_exact_from_transfer_plans(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormat,
        max_candidates: Optional[int],
    ) -> HCompletionResult:
        """Decide completion solely through materialized exact RC isomorphism."""
        first = None
        valid_seen = 0
        for candidate_index, plan in enumerate(
            HComplete._iter_hydrogen_transfer_plans(react_graph, prod_graph)
        ):
            if max_candidates is not None and candidate_index >= max_candidates:
                return HComplete._candidate_limit_result(
                    react_graph, prod_graph, valid_seen, format
                )
            candidate = HComplete._materialize_transfer_candidate(
                react_graph,
                prod_graph,
                plan,
                ignore_aromaticity,
                balance_its,
                format,
            )
            if not HComplete._valid_rc(candidate[3]):
                continue
            valid_seen += 1
            if first is None:
                first = candidate
                continue
            if HComplete._equivariant_count([first[3], candidate[3]], format) != 1:
                return HCompletionResult(
                    None,
                    None,
                    react_graph,
                    prod_graph,
                    candidates=valid_seen,
                    reason="non_equivariant_rc",
                    format=format,
                )
        return HComplete._completion_result(
            first,
            react_graph,
            prod_graph,
            valid_seen,
            format,
        )

    @staticmethod
    def _complete_priority_from_transfer_plans(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormat,
        max_candidates: Optional[int],
    ) -> HCompletionResult:
        candidates = []
        valid_seen = 0
        for candidate_index, plan in enumerate(
            HComplete._iter_hydrogen_transfer_plans(react_graph, prod_graph)
        ):
            if max_candidates is not None and candidate_index >= max_candidates:
                return HComplete._candidate_limit_result(
                    react_graph, prod_graph, valid_seen, format
                )
            candidate = HComplete._materialize_transfer_candidate(
                react_graph,
                prod_graph,
                plan,
                ignore_aromaticity,
                balance_its,
                format,
            )
            if not HComplete._valid_rc(candidate[3]):
                continue
            valid_seen += 1
            candidates.append(candidate)
        selected = HComplete._select_priority_candidate(candidates, format)
        return HComplete._completion_result(
            selected,
            react_graph,
            prod_graph,
            valid_seen,
            format,
        )

    @staticmethod
    def _materialize_transfer_candidate(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        plan: _HydrogenTransferPlan,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormat,
    ) -> Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]:
        react, prod = HComplete._realize_hydrogen_transfer_plan(
            react_graph, prod_graph, plan
        )
        its = HComplete._construct_its(
            react,
            prod,
            ignore_aromaticity,
            balance_its,
            format,
        )
        rc = HComplete._extract_rc(its, format)
        signature = HComplete._rc_signature(rc, format)
        return react, prod, its, rc, signature

    @staticmethod
    def _candidate_limit_result(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        candidates: int,
        format: ITSFormat,
    ) -> HCompletionResult:
        return HCompletionResult(
            None,
            None,
            react_graph,
            prod_graph,
            candidates=candidates,
            reason="max_candidates_reached",
            format=format,
            exhaustive=False,
        )

    @staticmethod
    def _completion_result(
        selected: Optional[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]],
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        candidates: int,
        format: ITSFormat,
    ) -> HCompletionResult:
        if selected is None or not HComplete._valid_rc(selected[3]):
            return HCompletionResult(
                None,
                None,
                react_graph,
                prod_graph,
                candidates=candidates,
                reason="no_valid_candidate",
                format=format,
            )
        react, prod, its, rc, sig = selected
        return HCompletionResult(
            its,
            rc,
            react,
            prod,
            signature=sig,
            candidates=candidates,
            format=format,
        )

    @staticmethod
    def _select_priority_candidate(
        candidates: List[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]],
        format: ITSFormat,
    ) -> Optional[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]]:
        if not candidates:
            return None

        rc_list = [candidate[3] for candidate in candidates]
        rc_sig = [candidate[4] for candidate in candidates]

        if len(set(rc_sig)) == 1:
            equivariant = HComplete._equivariant_count(rc_list, format)
        else:
            equivariant = 0

        if equivariant == len(rc_list) - 1:
            return candidates[0]

        priority_indices = get_priority(rc_list)
        priority_candidates = [candidates[i] for i in priority_indices]
        priority_rc = [candidate[3] for candidate in priority_candidates]
        priority_sig = [candidate[4] for candidate in priority_candidates]

        if len(set(priority_sig)) == 1:
            equivariant = HComplete._equivariant_count(priority_rc, format)
            if equivariant == len(priority_rc) - 1:
                return priority_candidates[0]
        return None

    @staticmethod
    def add_hydrogen_nodes_multiple(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        get_priority_graph: bool = False,
        format: ITSFormatInput = "typesGH",
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]]:
        """Generates multiple permutations of reactant and product graphs by
        adjusting hydrogen counts, exploring all possible configurations of
        hydrogen node additions or removals.

        :param react_graph: The reactant graph.
        :type react_graph: nx.Graph
        :param prod_graph: The product graph.
        :type prod_graph: nx.Graph
        :param ignore_aromaticity: If True, aromaticity is ignored.
        :type ignore_aromaticity: bool
        :param balance_its: If True, attempts to balance the ITS by adjusting hydrogen nodes.
        :type balance_its: bool
        :param get_priority_graph: If True, additional priority-based processing
                                   is applied to select optimal graph configurations.
        :type get_priority_graph: bool
        :param format: ITS representation: "auto", "typesGH", or "tuple".
        :type format: str
        :param max_candidates: Optional cap for enumerated hydrogen assignments.
        :type max_candidates: Optional[int]

        :return: Candidate reactant, product, ITS, RC, and RC-signature tuples.
        :rtype: List[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]]
        """
        resolved_format = HComplete._resolve_graph_pair_format(
            react_graph, prod_graph, format
        )
        updated_graphs = []

        for candidate in HComplete._iter_hydrogen_node_completions(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            format=resolved_format,
            max_candidates=max_candidates,
        ):
            if get_priority_graph is False and updated_graphs:
                previous = updated_graphs[-1]
                if (
                    candidate[-1] != previous[-1]
                    or HComplete._equivariant_count(
                        [previous[3], candidate[3]], resolved_format
                    )
                    != 1
                ):
                    return []
            updated_graphs.append(candidate)
        return updated_graphs

    @staticmethod
    def _hcount_change_atoms(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
    ) -> Tuple[List[int], List[int]]:
        hydrogen_nodes_break: List[int] = []
        hydrogen_nodes_form: List[int] = []

        for node_id in sorted(set(react_graph.nodes) & set(prod_graph.nodes)):
            react_hcount = react_graph.nodes[node_id].get("hcount", 0)
            prod_hcount = prod_graph.nodes[node_id].get("hcount", 0)
            hcount_diff = react_hcount - prod_hcount

            if hcount_diff > 0:
                hydrogen_nodes_break.extend([node_id] * hcount_diff)
            elif hcount_diff < 0:
                hydrogen_nodes_form.extend([node_id] * -hcount_diff)

        return hydrogen_nodes_break, hydrogen_nodes_form

    @staticmethod
    def _resolve_format(its: nx.Graph, format: ITSFormatInput) -> ITSFormat:
        if format == "auto":
            return detect_its_format(its)
        if format in ("typesGH", "tuple"):
            return format
        raise ValueError("format must be one of 'auto', 'typesGH', or 'tuple'.")

    @staticmethod
    def _resolve_graph_pair_format(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        format: ITSFormatInput,
    ) -> ITSFormat:
        if format == "auto":
            return "typesGH"
        if format in ("typesGH", "tuple"):
            return format
        raise ValueError("format must be one of 'auto', 'typesGH', or 'tuple'.")

    @staticmethod
    def _decompose_its(its: nx.Graph, format: ITSFormat) -> Tuple[nx.Graph, nx.Graph]:
        if format == "typesGH":
            return its_decompose(its)

        reverter = ITSReverter(its)
        return reverter.to_reactant_graph(), reverter.to_product_graph()

    @staticmethod
    def _construct_its(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormat,
    ) -> nx.Graph:
        if format == "typesGH":
            return ITSConstruction().ITSGraph(
                react_graph,
                prod_graph,
                ignore_aromaticity=ignore_aromaticity,
                balance_its=balance_its,
            )

        return ITSConstruction.construct(
            react_graph,
            prod_graph,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            store=True,
            node_attrs=HComplete.TUPLE_NODE_ATTRS,
            edge_attrs=HComplete.TUPLE_EDGE_ATTRS,
        )

    @staticmethod
    def _extract_rc(its: nx.Graph, format: ITSFormat) -> nx.Graph:
        if format == "typesGH":
            return get_rc(its)
        return RCExtractor(preserve_full_attrs=True).extract(its)

    @staticmethod
    def _valid_rc(rc: Optional[nx.Graph]) -> bool:
        return isinstance(rc, nx.Graph) and rc.number_of_nodes() > 0

    @staticmethod
    def _candidate_edge_keys(
        react_graph: nx.Graph, prod_graph: nx.Graph
    ) -> List[Tuple[int, int]]:
        edge_keys = {tuple(sorted(edge)) for edge in react_graph.edges}
        edge_keys.update(tuple(sorted(edge)) for edge in prod_graph.edges)
        return sorted(edge_keys)

    @staticmethod
    def _candidate_edge_order_pair(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        u: int,
        v: int,
    ) -> Tuple[Any, Any]:
        return (
            HComplete._candidate_edge_attr(react_graph, u, v, "order", 0.0),
            HComplete._candidate_edge_attr(prod_graph, u, v, "order", 0.0),
        )

    @staticmethod
    def _candidate_standard_order(
        react_order: Any,
        prod_order: Any,
        ignore_aromaticity: bool,
    ) -> Any:
        try:
            standard_order = react_order - prod_order
        except TypeError:
            return 0
        if ignore_aromaticity and abs(standard_order) < 1:
            return 0
        return standard_order

    @staticmethod
    def _candidate_typesgh_hh_pair(react_graph: nx.Graph, u: int, v: int) -> bool:
        return (
            HComplete._candidate_node_attr(react_graph, u, "element") == "H"
            and HComplete._candidate_node_attr(react_graph, v, "element") == "H"
        )

    @staticmethod
    def _candidate_tuple_node_changed(
        react_graph: nx.Graph, prod_graph: nx.Graph, node_id: int
    ) -> bool:
        for attr in ("element", "charge", "radical", "valence_electrons"):
            pair = (
                HComplete._candidate_node_attr(react_graph, node_id, attr),
                HComplete._candidate_node_attr(prod_graph, node_id, attr),
            )
            if RCExtractor._pair_diff(pair):
                return True

        hcount_pair = (
            HComplete._candidate_node_attr(react_graph, node_id, "hcount"),
            HComplete._candidate_node_attr(prod_graph, node_id, "hcount"),
        )
        if RCExtractor._hcount_diff(hcount_pair):
            return True

        lp_pair = (
            HComplete._candidate_lone_pair_attr(react_graph, node_id),
            HComplete._candidate_lone_pair_attr(prod_graph, node_id),
        )
        return RCExtractor._pair_diff(lp_pair)

    @staticmethod
    def _add_candidate_comparison_node(
        graph: nx.Graph,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        node_id: int,
        format: ITSFormat,
    ) -> None:
        if graph.has_node(node_id):
            return

        if format == "typesGH":
            element = HComplete._candidate_node_attr(react_graph, node_id, "element")
            charge = HComplete._candidate_node_attr(react_graph, node_id, "charge")
        else:
            element = (
                HComplete._candidate_node_attr(react_graph, node_id, "element"),
                HComplete._candidate_node_attr(prod_graph, node_id, "element"),
            )
            charge = (
                HComplete._candidate_node_attr(react_graph, node_id, "charge"),
                HComplete._candidate_node_attr(prod_graph, node_id, "charge"),
            )

        cmp_element = HComplete._comparison_node_value(element)
        cmp_charge = HComplete._comparison_pair_value(charge)
        graph.add_node(
            node_id,
            cmp_element=cmp_element,
            cmp_charge=cmp_charge,
            cmp_node=f"{cmp_element}|{cmp_charge}",
        )

    @staticmethod
    def _add_candidate_comparison_edge(
        graph: nx.Graph,
        u: int,
        v: int,
        react_order: Any,
        prod_order: Any,
    ) -> None:
        graph.add_edge(
            u,
            v,
            cmp_order=HComplete._comparison_pair_value((react_order, prod_order)),
        )

    @staticmethod
    def _candidate_node_attr(
        graph: nx.Graph, node_id: int, attr: str, default: Any = None
    ) -> Any:
        if default is None:
            default = HComplete._candidate_default(
                ITSConstruction.CORE_NODE_DEFAULTS, attr
            )
        if node_id not in graph:
            return default
        return graph.nodes[node_id].get(attr, default)

    @staticmethod
    def _candidate_edge_attr(
        graph: nx.Graph,
        u: int,
        v: int,
        attr: str,
        default: Any = None,
    ) -> Any:
        if default is None:
            default = HComplete._candidate_default(
                ITSConstruction.CORE_EDGE_DEFAULTS, attr
            )
        if not graph.has_edge(u, v):
            return default
        return graph.edges[u, v].get(attr, default)

    @staticmethod
    def _candidate_lone_pair_attr(graph: nx.Graph, node_id: int) -> Any:
        if node_id not in graph:
            return HComplete._candidate_default(
                ITSConstruction.CORE_NODE_DEFAULTS, "lone_pairs"
            )
        attrs = graph.nodes[node_id]
        return attrs.get("lone_pairs", attrs.get("lp", 0))

    @staticmethod
    def _candidate_default(defaults: Dict[str, Any], attr: str) -> Any:
        value = defaults.get(attr)
        return value() if callable(value) else value

    @staticmethod
    def add_hydrogen_nodes_multiple_utils(
        graph: nx.Graph,
        node_id_pairs: Iterable[Tuple[int, int]],
        atom_map_update: bool = True,
    ) -> nx.Graph:
        """Creates and returns a new graph with added hydrogen nodes based on
        the input graph and node ID pairs.

        :param graph: The base graph to which the nodes will be added.
        :type graph: nx.Graph
        :param node_id_pairs: Pairs of node IDs (original node, new
                              hydrogen node) to link with hydrogen.
        :type node_id_pairs: Iterable[Tuple[int, int]]
        :param atom_map_update: If True, update the 'atom_map' attribute with the new
                                hydrogen node ID; otherwise, retain the original node's 'atom_map'.
        :type atom_map_update: bool

        :return: A new graph instance with the added hydrogen nodes.
        :rtype: nx.Graph
        """
        new_graph = graph.copy()
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
            )
            new_graph.add_edge(
                node_id,
                new_hydrogen_node_id,
                order=1.0,
                bond_type="SINGLE",
            )
            new_graph.nodes[node_id]["hcount"] -= 1
        return new_graph
