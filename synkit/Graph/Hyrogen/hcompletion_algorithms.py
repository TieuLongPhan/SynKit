"""Hydrogen transport and reaction-centre comparison algorithms."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from operator import eq
from typing import Any, Iterator, List, Literal, Optional, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import generic_edge_match, generic_node_match

from synkit.Graph.Hyrogen._misc import (
    check_equivariant_graph,
    check_explicit_hydrogen,
)
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.rc_extractor import RCExtractor

ITSFormat = Literal["typesGH", "tuple"]
ITSFormatInput = Literal["auto", "typesGH", "tuple"]
HydrogenState = Tuple[str, Optional[int]]

TUPLE_COMPARISON_NODE_ATTRS = (
    "element",
    "aromatic",
    "hcount",
    "charge",
    "lone_pairs",
    "radical",
    "valence_electrons",
    "present",
)
TUPLE_COMPARISON_EDGE_ATTRS = (
    "order",
    "kekule_order",
    "sigma_order",
    "pi_order",
)

_FREE_H_SOURCE: HydrogenState = ("free_source", None)
_FREE_H_TARGET: HydrogenState = ("free_target", None)


def _freeze_comparison_value(value: Any) -> Any:
    """Convert nested Lewis-state values to stable hashable values."""
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    (
                        _freeze_comparison_value(key),
                        _freeze_comparison_value(item),
                    )
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_comparison_value(item) for item in value)
    if isinstance(value, set):
        return tuple(
            sorted(
                (_freeze_comparison_value(item) for item in value),
                key=repr,
            )
        )
    return value


@dataclass(frozen=True)
class _HydrogenTransferPlan:
    """Transport counts between hydrogen state classes."""

    sources: Tuple[Tuple[HydrogenState, int], ...]
    targets: Tuple[Tuple[HydrogenState, int], ...]
    matrix: Tuple[Tuple[int, ...], ...]


class HydrogenCompletionAlgorithms:
    """Private algorithms shared by :class:`HComplete`."""

    @classmethod
    def _iter_hydrogen_node_completions(
        cls,
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
        ) in cls._iter_hydrogen_side_graph_completions(
            react_graph,
            prod_graph,
            max_candidates=max_candidates,
        ):
            its = cls._construct_its(
                current_react_graph,
                current_prod_graph,
                ignore_aromaticity,
                balance_its,
                format,
            )
            rc = cls._extract_rc(its, format)
            signature = cls._rc_signature(rc, format)
            yield current_react_graph, current_prod_graph, its, rc, signature

    @classmethod
    def _iter_hydrogen_side_graph_completions(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        max_candidates: Optional[int] = None,
    ) -> Iterator[Tuple[nx.Graph, nx.Graph]]:
        plans = cls._iter_hydrogen_transfer_plans(react_graph, prod_graph)
        for candidate_index, plan in enumerate(plans):
            if max_candidates is not None and candidate_index >= max_candidates:
                break
            yield cls._realize_hydrogen_transfer_plan(
                react_graph,
                prod_graph,
                plan,
            )

    @classmethod
    def _iter_hydrogen_transfer_plans(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
    ) -> Iterator[_HydrogenTransferPlan]:
        """Yield transport tables between hydrogen state classes."""
        sources, targets = cls._hydrogen_state_classes(react_graph, prod_graph)
        target_capacities = tuple(count for _, count in targets)

        def recurse(
            row_index: int,
            capacities: Tuple[int, ...],
            matrix: Tuple[Tuple[int, ...], ...],
        ) -> Iterator[_HydrogenTransferPlan]:
            if row_index == len(sources):
                if not any(capacities):
                    yield _HydrogenTransferPlan(sources, targets, matrix)
                return

            source, row_total = sources[row_index]
            allowed_capacities = tuple(
                capacity if cls._hydrogen_states_compatible(source, target) else 0
                for capacity, (target, _) in zip(capacities, targets)
            )
            for allocation in cls._bounded_compositions(
                row_total,
                allowed_capacities,
            ):
                remaining = tuple(
                    capacity - used for capacity, used in zip(capacities, allocation)
                )
                yield from recurse(
                    row_index + 1,
                    remaining,
                    matrix + (allocation,),
                )

        yield from recurse(0, target_capacities, ())

    @classmethod
    def _hydrogen_state_classes(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
    ) -> Tuple[
        Tuple[Tuple[HydrogenState, int], ...],
        Tuple[Tuple[HydrogenState, int], ...],
    ]:
        """Return provenance-aware hydrogen state multiplicities."""
        hydrogen_nodes_break, hydrogen_nodes_form = cls._hcount_change_atoms(
            react_graph,
            prod_graph,
        )
        _, react_hydrogen_nodes = check_explicit_hydrogen(react_graph)
        _, prod_hydrogen_nodes = check_explicit_hydrogen(prod_graph)
        shared_explicit = set(react_hydrogen_nodes) & set(prod_hydrogen_nodes)

        source_counts: Counter[HydrogenState] = Counter(
            ("loss", node_id) for node_id in hydrogen_nodes_break
        )
        target_counts: Counter[HydrogenState] = Counter(
            ("gain", node_id) for node_id in hydrogen_nodes_form
        )

        for hydrogen_id in sorted(react_hydrogen_nodes):
            if hydrogen_id not in shared_explicit:
                source_counts[("react_explicit", hydrogen_id)] += 1
        for hydrogen_id in sorted(prod_hydrogen_nodes):
            if hydrogen_id not in shared_explicit:
                target_counts[("prod_explicit", hydrogen_id)] += 1

        source_total = sum(source_counts.values())
        target_total = sum(target_counts.values())
        react_explicit_total = sum(
            count
            for state, count in source_counts.items()
            if state[0] == "react_explicit"
        )
        prod_explicit_total = sum(
            count
            for state, count in target_counts.items()
            if state[0] == "prod_explicit"
        )
        track_total = max(
            source_total,
            target_total,
            react_explicit_total + prod_explicit_total,
        )
        if track_total > source_total:
            source_counts[_FREE_H_SOURCE] = track_total - source_total
        if track_total > target_total:
            target_counts[_FREE_H_TARGET] = track_total - target_total

        return tuple(source_counts.items()), tuple(target_counts.items())

    @classmethod
    def _hydrogen_states_compatible(
        cls,
        source: HydrogenState,
        target: HydrogenState,
    ) -> bool:
        if source[0] == "react_explicit" and target[0] == "prod_explicit":
            return source[1] == target[1]
        return True

    @classmethod
    def _bounded_compositions(
        cls,
        total: int,
        capacities: Tuple[int, ...],
    ) -> Iterator[Tuple[int, ...]]:
        """Yield bounded weak compositions in deterministic order."""
        if not capacities:
            if total == 0:
                yield ()
            return
        if len(capacities) == 1:
            if total <= capacities[0]:
                yield (total,)
            return

        lower = max(0, total - sum(capacities[1:]))
        upper = min(total, capacities[0])
        for head in range(lower, upper + 1):
            for tail in cls._bounded_compositions(
                total - head,
                capacities[1:],
            ):
                yield (head,) + tail

    @classmethod
    def _realize_hydrogen_transfer_plan(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        plan: _HydrogenTransferPlan,
    ) -> Tuple[nx.Graph, nx.Graph]:
        reactant_pairs, product_pairs = cls._hydrogen_transfer_plan_pairs(
            react_graph,
            prod_graph,
            plan,
        )
        return (
            cls.add_hydrogen_nodes_multiple_utils(react_graph, reactant_pairs),
            cls.add_hydrogen_nodes_multiple_utils(prod_graph, product_pairs),
        )

    @classmethod
    def _hydrogen_transfer_plan_pairs(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        plan: _HydrogenTransferPlan,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        reactant_pairs: List[Tuple[int, int]] = []
        product_pairs: List[Tuple[int, int]] = []
        used_nodes = set(react_graph.nodes) | set(prod_graph.nodes)
        next_hydrogen_id = max(used_nodes, default=0) + 1

        for row_index, (source, _) in enumerate(plan.sources):
            for column_index, (target, _) in enumerate(plan.targets):
                for _ in range(plan.matrix[row_index][column_index]):
                    if source[0] == "react_explicit":
                        hydrogen_id = source[1]
                    elif target[0] == "prod_explicit":
                        hydrogen_id = target[1]
                    else:
                        hydrogen_id = next_hydrogen_id
                        next_hydrogen_id += 1

                    if source[0] == "loss":
                        reactant_pairs.append((source[1], hydrogen_id))
                    if target[0] == "gain":
                        product_pairs.append((target[1], hydrogen_id))

        return reactant_pairs, product_pairs

    @classmethod
    def _typesgh_plan_comparison_graph(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        plan: _HydrogenTransferPlan,
        ignore_aromaticity: bool,
    ) -> nx.Graph:
        """Build the typesGH RC projection for a transfer plan."""
        reactant_pairs, product_pairs = cls._hydrogen_transfer_plan_pairs(
            react_graph,
            prod_graph,
            plan,
        )
        reactant_added_edges = {
            cls._candidate_edge_key(node_id, hydrogen_id)
            for node_id, hydrogen_id in reactant_pairs
        }
        product_added_edges = {
            cls._candidate_edge_key(node_id, hydrogen_id)
            for node_id, hydrogen_id in product_pairs
        }
        reactant_hydrogens = {hydrogen_id for _, hydrogen_id in reactant_pairs}

        edge_keys = set(cls._candidate_edge_keys(react_graph, prod_graph))
        edge_keys.update(reactant_added_edges)
        edge_keys.update(product_added_edges)

        graph = nx.Graph()
        for u, v in sorted(edge_keys):
            edge_key = cls._candidate_edge_key(u, v)
            react_order = (
                1.0
                if edge_key in reactant_added_edges
                else cls._candidate_edge_attr(react_graph, u, v, "order", 0.0)
            )
            prod_order = (
                1.0
                if edge_key in product_added_edges
                else cls._candidate_edge_attr(prod_graph, u, v, "order", 0.0)
            )
            standard_order = cls._candidate_standard_order(
                react_order,
                prod_order,
                ignore_aromaticity,
            )
            is_hh = (
                cls._typesgh_plan_node_attr(
                    react_graph,
                    u,
                    "element",
                    reactant_hydrogens,
                )
                == "H"
                and cls._typesgh_plan_node_attr(
                    react_graph,
                    v,
                    "element",
                    reactant_hydrogens,
                )
                == "H"
            )
            if standard_order == 0 and not is_hh:
                continue

            for node_id in (u, v):
                if graph.has_node(node_id):
                    continue
                element = cls._typesgh_plan_node_attr(
                    react_graph,
                    node_id,
                    "element",
                    reactant_hydrogens,
                )
                charge = cls._typesgh_plan_node_attr(
                    react_graph,
                    node_id,
                    "charge",
                    reactant_hydrogens,
                )
                cmp_element = cls._comparison_node_value(element)
                cmp_charge = cls._comparison_pair_value(charge)
                graph.add_node(
                    node_id,
                    cmp_element=cmp_element,
                    cmp_charge=cmp_charge,
                    cmp_node=f"{cmp_element}|{cmp_charge}",
                )
            cls._add_candidate_comparison_edge(
                graph,
                u,
                v,
                react_order,
                prod_order,
            )
        return graph

    @classmethod
    def _candidate_edge_key(cls, u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    @classmethod
    def _typesgh_plan_node_attr(
        cls,
        graph: nx.Graph,
        node_id: int,
        attr: str,
        added_hydrogens: set,
    ) -> Any:
        if node_id in added_hydrogens:
            if attr == "element":
                return "H"
            if attr == "charge":
                return 0
        return cls._candidate_node_attr(graph, node_id, attr)

    @classmethod
    def _rc_signature(
        cls,
        rc: nx.Graph,
        format: ITSFormat = "typesGH",
    ) -> str:
        return cls._comparison_graph_signature(cls._comparison_graph(rc, format))

    @classmethod
    def _comparison_graph_signature(cls, graph: nx.Graph) -> str:
        return nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="cmp_node",
            edge_attr="cmp_order",
            iterations=3,
        )

    @classmethod
    def _comparison_graphs_isomorphic(
        cls,
        left: nx.Graph,
        right: nx.Graph,
    ) -> bool:
        node_match = generic_node_match("cmp_node", "*", eq)
        edge_match = generic_edge_match("cmp_order", 1, eq)
        return nx.is_isomorphic(
            left,
            right,
            node_match=node_match,
            edge_match=edge_match,
        )

    @classmethod
    def _equivariant_count(
        cls,
        rc_list: List[nx.Graph],
        format: ITSFormat,
    ) -> int:
        if format == "typesGH":
            _, equivariant = check_equivariant_graph(rc_list)
            return equivariant

        graphs = [cls._comparison_graph(rc, format) for rc in rc_list]
        return sum(
            cls._comparison_graphs_isomorphic(graphs[0], graph) for graph in graphs[1:]
        )

    @classmethod
    def _comparison_graph(
        cls,
        rc: nx.Graph,
        format: ITSFormat,
    ) -> nx.Graph:
        graph = nx.Graph()

        for node, attrs in rc.nodes(data=True):
            cmp_element = cls._comparison_node_value(attrs.get("element"))
            cmp_charge = cls._comparison_pair_value(attrs.get("charge", 0))
            if format == "tuple":
                cmp_node = tuple(
                    _freeze_comparison_value(
                        attrs.get(
                            name,
                            (
                                (True, True)
                                if name == "present"
                                else cls._candidate_default(
                                    ITSConstruction.CORE_NODE_DEFAULTS,
                                    name,
                                )
                            ),
                        )
                    )
                    for name in TUPLE_COMPARISON_NODE_ATTRS
                )
            else:
                cmp_node = f"{cmp_element}|{cmp_charge}"
            graph.add_node(
                node,
                cmp_element=cmp_element,
                cmp_charge=cmp_charge,
                cmp_node=cmp_node,
            )

        for u, v, attrs in rc.edges(data=True):
            if format == "tuple":
                cmp_order = tuple(
                    _freeze_comparison_value(
                        attrs.get(
                            name,
                            cls._candidate_default(
                                ITSConstruction.CORE_EDGE_DEFAULTS,
                                name,
                            ),
                        )
                    )
                    for name in TUPLE_COMPARISON_EDGE_ATTRS
                )
            else:
                cmp_order = cls._comparison_pair_value(attrs.get("order", 1))
            graph.add_edge(u, v, cmp_order=cmp_order)

        return graph

    @classmethod
    def _comparison_node_value(cls, value: Any) -> Any:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if "H" in value:
                return "H"
            if value[0] == value[1]:
                return value[0]
            return tuple(value)
        return value

    @classmethod
    def _comparison_pair_value(cls, value: Any) -> Any:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if value[0] == value[1]:
                return value[0]
            return tuple(value)
        return value

    @classmethod
    def _candidate_comparison_graph(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        format: ITSFormat,
        ignore_aromaticity: bool = False,
        static_tuple_nodes: Optional[set] = None,
    ) -> nx.Graph:
        if format == "tuple":
            return cls._tuple_candidate_comparison_graph(
                react_graph,
                prod_graph,
                ignore_aromaticity,
                static_tuple_nodes=static_tuple_nodes,
            )
        return cls._typesgh_candidate_comparison_graph(
            react_graph,
            prod_graph,
            ignore_aromaticity,
        )

    @classmethod
    def _typesgh_candidate_comparison_graph(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
    ) -> nx.Graph:
        graph = nx.Graph()

        for u, v in cls._candidate_edge_keys(react_graph, prod_graph):
            react_order, prod_order = cls._candidate_edge_order_pair(
                react_graph,
                prod_graph,
                u,
                v,
            )
            standard_order = cls._candidate_standard_order(
                react_order,
                prod_order,
                ignore_aromaticity,
            )
            if standard_order == 0 and not cls._candidate_typesgh_hh_pair(
                react_graph,
                u,
                v,
            ):
                continue

            cls._add_candidate_comparison_node(
                graph,
                react_graph,
                prod_graph,
                u,
                "typesGH",
            )
            cls._add_candidate_comparison_node(
                graph,
                react_graph,
                prod_graph,
                v,
                "typesGH",
            )
            cls._add_candidate_comparison_edge(
                graph,
                u,
                v,
                react_order,
                prod_order,
            )

        return graph

    @classmethod
    def _tuple_candidate_comparison_graph(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
        ignore_aromaticity: bool,
        static_tuple_nodes: Optional[set] = None,
    ) -> nx.Graph:
        graph = nx.Graph()
        rc_nodes = set(static_tuple_nodes or ())
        edge_keys = cls._candidate_edge_keys(react_graph, prod_graph)

        if static_tuple_nodes is None:
            for node_id in set(react_graph.nodes) | set(prod_graph.nodes):
                if cls._candidate_tuple_node_changed(
                    react_graph,
                    prod_graph,
                    node_id,
                ):
                    rc_nodes.add(node_id)

        for u, v in edge_keys:
            react_order, prod_order = cls._candidate_edge_order_pair(
                react_graph,
                prod_graph,
                u,
                v,
            )
            if (
                cls._candidate_standard_order(
                    react_order,
                    prod_order,
                    ignore_aromaticity,
                )
                != 0
            ):
                rc_nodes.add(u)
                rc_nodes.add(v)

        for node_id in rc_nodes:
            cls._add_candidate_comparison_node(
                graph,
                react_graph,
                prod_graph,
                node_id,
                "tuple",
            )

        for u, v in edge_keys:
            if u not in rc_nodes or v not in rc_nodes:
                continue
            react_order, prod_order = cls._candidate_edge_order_pair(
                react_graph,
                prod_graph,
                u,
                v,
            )
            cls._add_candidate_comparison_edge(
                graph,
                u,
                v,
                react_order,
                prod_order,
            )

        return graph

    @classmethod
    def _tuple_static_node_changes(
        cls,
        react_graph: nx.Graph,
        prod_graph: nx.Graph,
    ) -> set:
        rc_nodes = set()
        for node_id in set(react_graph.nodes) | set(prod_graph.nodes):
            for attr in ("element", "charge", "radical", "valence_electrons"):
                pair = (
                    cls._candidate_node_attr(react_graph, node_id, attr),
                    cls._candidate_node_attr(prod_graph, node_id, attr),
                )
                if RCExtractor._pair_diff(pair):
                    rc_nodes.add(node_id)
                    break
            if node_id in rc_nodes:
                continue

            lp_pair = (
                cls._candidate_lone_pair_attr(react_graph, node_id),
                cls._candidate_lone_pair_attr(prod_graph, node_id),
            )
            if RCExtractor._pair_diff(lp_pair):
                rc_nodes.add(node_id)

        return rc_nodes
