"""Legacy all-variant hydrogen extension.

Use :mod:`synkit.Graph.Hyrogen.hextend` for provenance-aware transfer
enumeration. This module exists for reproducibility and regression comparison.
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
from joblib import Parallel, delayed

from synkit.Graph.Hyrogen._misc import check_hcount_change
from synkit.Graph.Hyrogen.hcomplete import HComplete, ITSFormatInput
from synkit.Graph.Hyrogen.hcomplete_legacy import LegacyHComplete
from synkit.Graph.Matcher.graph_cluster import GraphCluster

legacy_cluster = GraphCluster()


class LegacyHExtend(LegacyHComplete):
    """Historical extension driven by shared hydrogen-ID permutations."""

    @staticmethod
    def extend_its(
        its: nx.Graph,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            return [], [], []

        resolved_format = HComplete._resolve_format(its, format)
        react_graph, prod_graph = HComplete._decompose_its(its, resolved_format)
        if check_hcount_change(react_graph, prod_graph) == 0:
            rc = HComplete._extract_rc(its, resolved_format)
            if not HComplete._valid_rc(rc):
                return [], [], []
            return [rc], [its], [HComplete._rc_signature(rc, resolved_format)]

        rc_list, its_list, signatures = [], [], []
        for (
            _,
            _,
            completed_its,
            rc,
            signature,
        ) in LegacyHComplete._iter_hydrogen_node_completions(
            react_graph,
            prod_graph,
            ignore_aromaticity,
            balance_its,
            resolved_format,
            max_candidates,
        ):
            if HComplete._valid_rc(rc):
                rc_list.append(rc)
                its_list.append(completed_its)
                signatures.append(signature)
        return rc_list, its_list, signatures

    @staticmethod
    def _extend(
        its: nx.Graph,
        ignore_aromaticity: bool,
        balance_its: bool,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        return LegacyHExtend.extend_its(
            its,
            ignore_aromaticity,
            balance_its,
            format,
            max_candidates,
        )

    @staticmethod
    def _extend_unique(
        its: nx.Graph,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        format: ITSFormatInput = "auto",
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[nx.Graph], List[nx.Graph], List[str]]:
        if not isinstance(its, nx.Graph) or its.number_of_nodes() == 0:
            return [], [], []

        resolved_format = HComplete._resolve_format(its, format)
        react_graph, prod_graph = HComplete._decompose_its(its, resolved_format)
        if check_hcount_change(react_graph, prod_graph) == 0:
            rc = HComplete._extract_rc(its, resolved_format)
            if not HComplete._valid_rc(rc):
                return [], [], []
            return [rc], [its], [HComplete._rc_signature(rc, resolved_format)]

        side_candidates = []
        comparison_graphs = []
        signatures = []
        static_tuple_nodes = (
            HComplete._tuple_static_node_changes(react_graph, prod_graph)
            if resolved_format == "tuple"
            else None
        )
        for (
            current_react_graph,
            current_prod_graph,
        ) in LegacyHComplete._iter_hydrogen_side_graph_completions(
            react_graph,
            prod_graph,
            max_candidates,
        ):
            comparison = HComplete._candidate_comparison_graph(
                current_react_graph,
                current_prod_graph,
                resolved_format,
                ignore_aromaticity,
                static_tuple_nodes,
            )
            if comparison.number_of_nodes() == 0:
                continue
            side_candidates.append((current_react_graph, current_prod_graph))
            comparison_graphs.append(comparison)
            signatures.append(HComplete._comparison_graph_signature(comparison))

        if not comparison_graphs:
            return [], [], []

        clusters, _ = legacy_cluster.iterative_cluster(comparison_graphs, signatures)
        rc_list, its_list, rc_signatures = [], [], []
        for indices in clusters:
            if not indices:
                continue
            current_react_graph, current_prod_graph = side_candidates[min(indices)]
            completed_its = HComplete._construct_its(
                current_react_graph,
                current_prod_graph,
                ignore_aromaticity,
                balance_its,
                resolved_format,
            )
            rc = HComplete._extract_rc(completed_its, resolved_format)
            if HComplete._valid_rc(rc):
                rc_list.append(rc)
                its_list.append(completed_its)
                rc_signatures.append(HComplete._rc_signature(rc, resolved_format))
        return rc_list, its_list, rc_signatures

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
        rc_list, its_list, _ = LegacyHExtend._extend_unique(
            data_dict[its_key],
            ignore_aromaticity,
            balance_its,
            format,
            max_candidates,
        )
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
        if n_jobs == 1 and backend is None:
            return [
                LegacyHExtend._process(
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
        return Parallel(**parallel_kwargs)(
            delayed(LegacyHExtend._process)(
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


__all__ = ["LegacyHExtend", "legacy_cluster"]
