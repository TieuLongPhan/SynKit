"""Legacy ambiguous-hydrogen completion.

This module preserves the historical same-permutation algorithm for
reproducibility and direct benchmarking. New code should use
``synkit.Graph.Hyrogen.hcomplete.HComplete``.
"""

import itertools
from copy import copy
from typing import Dict, Iterator, List, Optional, Tuple

import networkx as nx
from joblib import Parallel, delayed

from synkit.Graph.Hyrogen._misc import (
    check_explicit_hydrogen,
    check_hcount_change,
)
from synkit.Graph.Hyrogen.hcomplete import (
    HComplete,
    HCompletionResult,
    ITSFormat,
    ITSFormatInput,
)


class LegacyHComplete:
    """Historical completion based on shared H-ID permutations.

    The same permutation is assigned to reactant loss slots and product gain
    slots. This is intentionally retained unchanged even though it can collapse
    distinct donor-to-acceptor hydrogen transfers.
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
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Dict[str, Optional[nx.Graph]]:
        graphs = copy(graph_data)
        result = LegacyHComplete.complete_its(
            graphs.get(its_key),
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

    @staticmethod
    def process_graph_data_parallel(
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
        return Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(LegacyHComplete.process_single_graph_data)(
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
        """Complete an ITS with the historical same-permutation algorithm."""
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

        return LegacyHComplete._complete_from_side_graphs(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            get_priority_graph,
            resolved_format,
            max_candidates,
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
        resolved_format = HComplete._resolve_graph_pair_format(
            react_graph, prod_graph, format
        )
        result = LegacyHComplete._complete_from_side_graphs(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            get_priority_graph,
            resolved_format,
            max_candidates,
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

        for candidate in LegacyHComplete._iter_hydrogen_node_completions(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            format,
            max_candidates,
        ):
            react, prod, its, rc, signature = candidate
            if not HComplete._valid_rc(rc):
                continue

            valid_seen += 1
            if get_priority_graph:
                candidates.append(candidate)
                continue
            if first is None:
                first = candidate
                continue

            if (
                signature != first[4]
                or HComplete._equivariant_count([first[3], rc], format) != 1
            ):
                return HCompletionResult(
                    None,
                    None,
                    react_graph,
                    prod_graph,
                    candidates=valid_seen,
                    reason="non_equivariant_rc",
                    format=format,
                )

        selected = (
            HComplete._select_priority_candidate(candidates, format)
            if get_priority_graph
            else first
        )
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

        react, prod, its, rc, signature = selected
        return HCompletionResult(
            its,
            rc,
            react,
            prod,
            signature=signature,
            candidates=valid_seen,
            format=format,
        )

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
        resolved_format = HComplete._resolve_graph_pair_format(
            react_graph, prod_graph, format
        )
        updated_graphs = []
        for candidate in LegacyHComplete._iter_hydrogen_node_completions(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            resolved_format,
            max_candidates,
        ):
            if not get_priority_graph and updated_graphs:
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
        ) in LegacyHComplete._iter_hydrogen_side_graph_completions(
            react_graph,
            prod_graph,
            max_candidates,
        ):
            its = HComplete._construct_its(
                current_react_graph,
                current_prod_graph,
                ignore_aromaticity,
                balance_its,
                format,
            )
            rc = HComplete._extract_rc(its, format)
            signature = HComplete._rc_signature(rc, format)
            yield current_react_graph, current_prod_graph, its, rc, signature

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

        hydrogen_nodes = LegacyHComplete._hydrogen_node_ids(
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
            current_react_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                react_graph_copy,
                zip(hydrogen_nodes_break, permutation[:n_break]),
            )
            current_prod_graph = HComplete.add_hydrogen_nodes_multiple_utils(
                prod_graph_copy,
                zip(hydrogen_nodes_form, permutation[:n_form]),
            )
            yield current_react_graph, current_prod_graph

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


__all__ = ["LegacyHComplete"]
