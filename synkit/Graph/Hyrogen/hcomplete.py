import itertools
import networkx as nx
from dataclasses import dataclass
from copy import copy
from joblib import Parallel, delayed
from typing import Any, Dict, Iterator, List, Literal, Tuple, Iterable, Optional
from operator import eq
from networkx.algorithms.isomorphism import generic_edge_match, generic_node_match

from synkit.IO.debug import setup_logging
from synkit.IO.chem_converter import detect_its_format
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_decompose import get_rc, its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.rc_extractor import RCExtractor
from synkit.Graph.Hyrogen._misc import (
    check_hcount_change,
    check_explicit_hydrogen,
    get_priority,
    check_equivariant_graph,
)

logger = setup_logging()

ITSFormat = Literal["typesGH", "tuple"]
ITSFormatInput = Literal["auto", "typesGH", "tuple"]


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

    @property
    def ok(self) -> bool:
        return (
            isinstance(self.its, nx.Graph)
            and isinstance(self.rc, nx.Graph)
            and self.rc.number_of_nodes() > 0
        )


class HComplete:
    """A class for infering hydrogen to complete reaction center or ITS
    graph."""

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

        Parameters:
        - graph_data (Dict[str, nx.Graph]): Dictionary containing the graph data.
        - its_key (str): Key where the ITS graph is stored.
        - rc_key (str): Key where the RC graph is stored.
        - ignore_aromaticity (bool): If True, aromaticity is ignored during processing. Default is False.
        - balance_its (bool): If True, the ITS is balanced. Default is True.
        - get_priority_graph (bool): If True, priority is given to graph data during processing. Default is False.
        - max_hydrogen (int): Maximum number of hydrogens that can be handled in the inference step.
        - format (str): ITS representation: "auto", "typesGH", or "tuple".
        - max_candidates (Optional[int]): Optional cap for enumerated hydrogen assignments.

        Returns:
        - Dict[str, Optional[nx.Graph]]: Dictionary with updated ITS and RC graph data, or None if processing fails.
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
        - format (str): ITS representation: "auto", "typesGH", or "tuple".
        - max_candidates (Optional[int]): Optional cap for enumerated hydrogen assignments.

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

        Parameters:
        - graph_data (Dict[str, nx.Graph]): Dictionary containing the graph data.
        - its_key (str): Key for the ITS graph in the dictionary.
        - rc_key (str): Key for the RC graph in the dictionary.
        - react_graph (nx.Graph): Graph representing the reactants.
        - prod_graph (nx.Graph): Graph representing the products.
        - ignore_aromaticity (bool): If True, aromaticity will not be considered in processing.
        - balance_its (bool): If True, balances the ITS graph.
        - get_priority_graph (bool): If True, processes graphs with priority considerations.
        - format (str): ITS representation: "auto", "typesGH", or "tuple".
        - max_candidates (Optional[int]): Optional cap for enumerated hydrogen assignments.

        Returns:
        - Dict[str, Optional[nx.Graph]]: Updated graph dictionary with potentially modified ITS and RC graphs.
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
        candidates = []
        first = None
        valid_seen = 0

        for candidate in HComplete._iter_hydrogen_node_completions(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            format=format,
            max_candidates=max_candidates,
        ):
            react, prod, its, rc, sig = candidate
            if not HComplete._valid_rc(rc):
                continue

            valid_seen += 1
            if get_priority_graph:
                candidates.append(candidate)
                continue

            if first is None:
                first = candidate
                continue

            first_equiv = HComplete._equivariant_count([first[3], rc], format)
            if sig != first[4] or first_equiv != 1:
                return HCompletionResult(
                    None,
                    None,
                    react_graph,
                    prod_graph,
                    candidates=valid_seen,
                    reason="non_equivariant_rc",
                    format=format,
                )

        if get_priority_graph:
            selected = HComplete._select_priority_candidate(candidates, format)
        else:
            selected = first

        if selected is None:
            return HCompletionResult(
                None,
                None,
                react_graph,
                prod_graph,
                candidates=valid_seen,
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
            candidates=valid_seen,
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

        Parameters:
        - react_graph (nx.Graph): The reactant graph.
        - prod_graph (nx.Graph): The product graph.
        - ignore_aromaticity (bool): If True, aromaticity is ignored.
        - balance_its (bool): If True, attempts to balance the ITS by adjusting hydrogen nodes.
        - get_priority_graph (bool): If True, additional priority-based processing
        is applied to select optimal graph configurations.
        - format (str): ITS representation: "auto", "typesGH", or "tuple".
        - max_candidates (Optional[int]): Optional cap for enumerated hydrogen assignments.

        Returns:
        - List[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]]: Candidate
        reactant, product, ITS, RC, and RC-signature tuples.
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
                if candidate[-1] != updated_graphs[-1][-1]:
                    return []
            updated_graphs.append(candidate)
        return updated_graphs

    @staticmethod
    def _iter_hydrogen_node_completions(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormat,
        max_candidates: Optional[int] = None,
    ) -> Iterator[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, str]]:
        for (
            current_react_graph,
            current_prod_graph,
        ) in HComplete._iter_hydrogen_side_graph_completions(
            react_graph,
            prod_graph,
            max_candidates=max_candidates,
        ):
            its = HComplete._construct_its(
                current_react_graph,
                current_prod_graph,
                ignore_aromaticity,
                balance_its,
                format,
            )
            rc = HComplete._extract_rc(its, format)
            sig = HComplete._rc_signature(rc, format)
            yield ((current_react_graph, current_prod_graph, its, rc, sig))

    @staticmethod
    def _iter_hydrogen_side_graph_completions(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        max_candidates: Optional[int] = None,
    ) -> Iterator[Tuple[nx.Graph, nx.Graph]]:
        react_graph_copy = react_graph.copy()
        prod_graph_copy = prod_graph.copy()
        _, react_hydrogen_nodes = check_explicit_hydrogen(react_graph_copy)
        _, prod_hydrogen_nodes = check_explicit_hydrogen(prod_graph_copy)
        hydrogen_nodes_break, hydrogen_nodes_form = HComplete._hcount_change_atoms(
            react_graph_copy, prod_graph_copy
        )

        n_break = len(hydrogen_nodes_break)
        n_form = len(hydrogen_nodes_form)
        n_hydrogen_needed = max(n_break, n_form)
        if n_hydrogen_needed == 0:
            return

        hydrogen_nodes = HComplete._hydrogen_node_ids(
            react_graph_copy,
            prod_graph_copy,
            sorted(set(react_hydrogen_nodes) | set(prod_hydrogen_nodes)),
            n_hydrogen_needed,
        )

        for candidate_index, permutation in enumerate(
            itertools.permutations(hydrogen_nodes, n_hydrogen_needed)
        ):
            if max_candidates is not None and candidate_index >= max_candidates:
                break

            current_react_graph, current_prod_graph = react_graph_copy, prod_graph_copy

            current_react_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                current_react_graph,
                zip(hydrogen_nodes_break, permutation[:n_break]),
            )
            current_prod_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                current_prod_graph,
                zip(hydrogen_nodes_form, permutation[:n_form]),
            )
            yield current_react_graph, current_prod_graph

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
    def _hydrogen_node_ids(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        existing_hydrogen_nodes: List[int],
        n_hydrogen_needed: int,
    ) -> List[int]:
        used_nodes = set(react_graph.nodes) | set(prod_graph.nodes)
        hydrogen_nodes = [
            node_id for node_id in existing_hydrogen_nodes if node_id in used_nodes
        ]
        n_new_needed = max(0, n_hydrogen_needed - len(hydrogen_nodes))
        max_index = max(used_nodes, default=0)
        hydrogen_nodes.extend(range(max_index + 1, max_index + 1 + n_new_needed))
        return hydrogen_nodes

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
    def _rc_signature(rc: nx.Graph, format: ITSFormat = "typesGH") -> str:
        return HComplete._comparison_graph_signature(
            HComplete._comparison_graph(rc, format)
        )

    @staticmethod
    def _comparison_graph_signature(graph: nx.Graph) -> str:
        return nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="cmp_node",
            edge_attr="cmp_order",
            iterations=3,
        )

    @staticmethod
    def _equivariant_count(rc_list: List[nx.Graph], format: ITSFormat) -> int:
        if format == "typesGH":
            _, equivariant = check_equivariant_graph(rc_list)
            return equivariant

        graphs = [HComplete._comparison_graph(rc, format) for rc in rc_list]
        node_match = generic_node_match(
            ["cmp_element", "cmp_charge"],
            ["*", 0],
            [eq, eq],
        )
        edge_match = generic_edge_match("cmp_order", 1, eq)
        equivariant = 0
        for graph in graphs[1:]:
            if nx.is_isomorphic(
                graphs[0],
                graph,
                node_match=node_match,
                edge_match=edge_match,
            ):
                equivariant += 1
        return equivariant

    @staticmethod
    def _comparison_graph(rc: nx.Graph, format: ITSFormat) -> nx.Graph:
        graph = nx.Graph()

        for node, attrs in rc.nodes(data=True):
            cmp_element = HComplete._comparison_node_value(attrs.get("element"))
            cmp_charge = HComplete._comparison_pair_value(attrs.get("charge", 0))
            graph.add_node(
                node,
                cmp_element=cmp_element,
                cmp_charge=cmp_charge,
                cmp_node=f"{cmp_element}|{cmp_charge}",
            )

        for u, v, attrs in rc.edges(data=True):
            graph.add_edge(
                u,
                v,
                cmp_order=HComplete._comparison_pair_value(attrs.get("order", 1)),
            )

        return graph

    @staticmethod
    def _comparison_node_value(value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if "H" in value:
                return "H"
            if value[0] == value[1]:
                return value[0]
            return tuple(value)
        return value

    @staticmethod
    def _comparison_pair_value(value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if value[0] == value[1]:
                return value[0]
            return tuple(value)
        return value

    @staticmethod
    def _candidate_comparison_graph(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        format: ITSFormat,
        ignore_aromaticity: bool = False,
        static_tuple_nodes: Optional[set] = None,
    ) -> nx.Graph:
        """Build the RC comparison graph directly from candidate side graphs."""
        if format == "tuple":
            return HComplete._tuple_candidate_comparison_graph(
                react_graph,
                prod_graph,
                ignore_aromaticity,
                static_tuple_nodes=static_tuple_nodes,
            )
        return HComplete._typesgh_candidate_comparison_graph(
            react_graph, prod_graph, ignore_aromaticity
        )

    @staticmethod
    def _typesgh_candidate_comparison_graph(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
    ) -> nx.Graph:
        graph = nx.Graph()

        for u, v in HComplete._candidate_edge_keys(react_graph, prod_graph):
            react_order, prod_order = HComplete._candidate_edge_order_pair(
                react_graph, prod_graph, u, v
            )
            standard_order = HComplete._candidate_standard_order(
                react_order, prod_order, ignore_aromaticity
            )
            if standard_order == 0 and not HComplete._candidate_typesgh_hh_pair(
                react_graph, u, v
            ):
                continue

            HComplete._add_candidate_comparison_node(
                graph, react_graph, prod_graph, u, "typesGH"
            )
            HComplete._add_candidate_comparison_node(
                graph, react_graph, prod_graph, v, "typesGH"
            )
            HComplete._add_candidate_comparison_edge(
                graph, u, v, react_order, prod_order
            )

        return graph

    @staticmethod
    def _tuple_candidate_comparison_graph(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        static_tuple_nodes: Optional[set] = None,
    ) -> nx.Graph:
        graph = nx.Graph()
        rc_nodes = set(static_tuple_nodes or ())
        edge_keys = HComplete._candidate_edge_keys(react_graph, prod_graph)

        if static_tuple_nodes is None:
            for node_id in set(react_graph.nodes) | set(prod_graph.nodes):
                if HComplete._candidate_tuple_node_changed(
                    react_graph, prod_graph, node_id
                ):
                    rc_nodes.add(node_id)

        for u, v in edge_keys:
            react_order, prod_order = HComplete._candidate_edge_order_pair(
                react_graph, prod_graph, u, v
            )
            standard_order = HComplete._candidate_standard_order(
                react_order, prod_order, ignore_aromaticity
            )
            if standard_order != 0:
                rc_nodes.add(u)
                rc_nodes.add(v)

        for node_id in rc_nodes:
            HComplete._add_candidate_comparison_node(
                graph, react_graph, prod_graph, node_id, "tuple"
            )

        for u, v in edge_keys:
            if u not in rc_nodes or v not in rc_nodes:
                continue
            react_order, prod_order = HComplete._candidate_edge_order_pair(
                react_graph, prod_graph, u, v
            )
            HComplete._add_candidate_comparison_edge(
                graph, u, v, react_order, prod_order
            )

        return graph

    @staticmethod
    def _tuple_static_node_changes(
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
    ) -> set:
        """Return tuple RC node changes that are independent of H assignment."""
        rc_nodes = set()
        for node_id in set(react_graph.nodes) | set(prod_graph.nodes):
            for attr in ("element", "charge", "radical", "valence_electrons"):
                pair = (
                    HComplete._candidate_node_attr(react_graph, node_id, attr),
                    HComplete._candidate_node_attr(prod_graph, node_id, attr),
                )
                if RCExtractor._pair_diff(pair):
                    rc_nodes.add(node_id)
                    break
            if node_id in rc_nodes:
                continue

            lp_pair = (
                HComplete._candidate_lone_pair_attr(react_graph, node_id),
                HComplete._candidate_lone_pair_attr(prod_graph, node_id),
            )
            if RCExtractor._pair_diff(lp_pair):
                rc_nodes.add(node_id)

        return rc_nodes

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

        Parameters:
        - graph (nx.Graph): The base graph to which the nodes will be added.
        - node_id_pairs (Iterable[Tuple[int, int]]): Pairs of node IDs (original node, new
        hydrogen node) to link with hydrogen.
        - atom_map_update (bool): If True, update the 'atom_map' attribute with the new
        hydrogen node ID; otherwise, retain the original node's 'atom_map'.

        Returns:
        - nx.Graph: A new graph instance with the added hydrogen nodes.
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
            new_graph.nodes[node_id]["hcount"] -= 1
        return new_graph
