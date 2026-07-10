from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
from rdkit import Chem

from synkit.Graph.Mech.electron_accounting import (
    ChargeEdit,
    change_atom_charge,
)
from synkit.Graph.Mech.lwg_ops import (
    EdgeChange,
    LonePairChange,
    change_edge_order,
    change_lone_pairs,
    normalize_lwg_graph,
)
from synkit.IO.chem_converter import smiles_to_graph
from synkit.IO.graph_to_mol import GraphToMol


@dataclass(frozen=True)
class LWGStepReport:
    """Report for one applied typed electron-pushing action.

    :param action_index: Zero-based action index in the input ``epd_typed``.
    :type action_index: int
    :param action: Typed action label, e.g. ``"LP-/Sigma+"``.
    :type action: str
    :param source: Source atom-map list from the action.
    :type source: tuple[int, ...]
    :param target: Target atom-map list from the action.
    :type target: tuple[int, ...]
    :param edge_changes: Sigma/pi edge-order edits applied by this step.
    :type edge_changes: tuple[EdgeChange, ...]
    :param lone_pair_changes: Lone-pair edits applied by this step.
    :type lone_pair_changes: tuple[LonePairChange, ...]
    :param charge_changes: Local formal-charge edits applied by this step.
    :type charge_changes: tuple[ChargeEdit, ...]
    :param changed_atom_maps: Atom maps touched by this step.
    :type changed_atom_maps: tuple[int, ...]
    """

    action_index: int
    action: str
    source: tuple[int, ...]
    target: tuple[int, ...]
    edge_changes: tuple[EdgeChange, ...]
    lone_pair_changes: tuple[LonePairChange, ...]
    charge_changes: tuple[ChargeEdit, ...]
    changed_atom_maps: tuple[int, ...]


@dataclass(frozen=True)
class LWGEditResult:
    """Result returned by :class:`LWGEditor`.

    :param intermediates: Edited reactant graph after each action.
    :type intermediates: list[nx.Graph]
    :param final_graph: Final edited reactant Lewis graph.
    :type final_graph: nx.Graph
    :param product_graph: Product Lewis graph parsed from input RSMI.
    :type product_graph: nx.Graph
    :param final_smiles: Canonical SMILES exported from ``final_graph``.
    :type final_smiles: str | None
    :param product_smiles: Canonical SMILES exported from ``product_graph``.
    :type product_smiles: str | None
    :param structural_match: Whether mapped atom identities and Kekule edge
        orders match exactly.
    :type structural_match: bool
    :param charge_match: Whether mapped atom formal charges match.
    :type charge_match: bool
    :param smiles_match: Whether exported canonical SMILES match.
    :type smiles_match: bool
    :param matches_product: Product-match flag used by the editor. This requires
        ``charge_match`` and ``smiles_match``; strict Kekule alternation
        differences are diagnostic only.
    :type matches_product: bool
    :param step_reports: Per-action edit reports.
    :type step_reports: list[LWGStepReport]
    """

    intermediates: list[nx.Graph]
    final_graph: nx.Graph
    product_graph: nx.Graph
    final_smiles: str | None
    product_smiles: str | None
    structural_match: bool
    charge_match: bool
    smiles_match: bool
    matches_product: bool
    step_reports: list[LWGStepReport]


class LWGEditor:
    """Apply typed EPD actions to a Lewis graph.

    The editor mutates lone-pair counts plus sigma/pi edge orders. Formal
    charges are updated from local grammar deltas rather than recomputed from
    valence-electron bookkeeping:

    * ``LP-``: +2 on the source atom.
    * ``LP+``: -2 on the target atom.
    * ``Sigma/Pi-``: +1 on each source bond endpoint.
    * ``Sigma/Pi+``: -1 on each target bond endpoint.

    Example
    -------
    .. code-block:: python

        from synkit.Graph.Mech import LWGEditor

        result = LWGEditor().apply(
            "[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]",
            [
                ["LP-/Sigma+", [1], [1, 2]],
                ["Sigma-/LP+", [2, 3], [3]],
            ],
        )

        assert result.matches_product
        assert result.intermediates[0].nodes[1]["charge"] == 1
        assert result.intermediates[0].nodes[2]["charge"] == -1
        assert result.final_graph.nodes[3]["charge"] == -1
    """

    def apply(
        self,
        rsmi: str,
        epd_typed: Sequence[Sequence[Any]],
    ) -> LWGEditResult:
        """Apply typed EPD actions from reactant to product graph.

        :param rsmi: Atom-mapped reaction SMILES in ``reactants>>products``
            form.
        :type rsmi: str
        :param epd_typed: Typed electron-pushing descriptors. Each item must be
            ``[action, source, target]`` where ``action`` is a label such as
            ``"LP-/Sigma+"`` and source/target are atom-map lists.
        :type epd_typed: Sequence[Sequence[Any]]
        :returns: Edited graph sequence, product graph, match flags, and
            per-step edit reports.
        :rtype: LWGEditResult
        :raises ValueError: If the RSMI is malformed, a typed action is
            unsupported, an atom map is missing, or an edge edit is invalid.
        """
        reactant_smiles, product_smiles = self._split_rsmi(rsmi)
        r_graph = self._smiles_to_lwg(reactant_smiles)
        p_graph = self._smiles_to_lwg(product_smiles)

        intermediates: list[nx.Graph] = []
        reports: list[LWGStepReport] = []

        for action_index, step in enumerate(epd_typed):
            action, source, target = self._parse_step(step)
            (
                edge_changes,
                lone_pair_changes,
                charge_changes,
                changed_atom_maps,
            ) = self._apply_action(
                r_graph,
                action=action,
                source=source,
                target=target,
            )
            normalize_lwg_graph(r_graph, in_place=True)

            reports.append(
                LWGStepReport(
                    action_index=action_index,
                    action=action,
                    source=tuple(source),
                    target=tuple(target),
                    edge_changes=tuple(edge_changes),
                    lone_pair_changes=tuple(lone_pair_changes),
                    charge_changes=tuple(charge_changes),
                    changed_atom_maps=tuple(sorted(set(changed_atom_maps))),
                )
            )
            intermediates.append(r_graph.copy())

        final_smiles = self.graph_to_smiles(r_graph)
        expected_smiles = self.graph_to_smiles(p_graph)
        structural_match = self.structural_match(r_graph, p_graph)
        charge_match = self.charge_match(r_graph, p_graph)
        smiles_match = final_smiles is not None and final_smiles == expected_smiles

        return LWGEditResult(
            intermediates=intermediates,
            final_graph=r_graph,
            product_graph=p_graph,
            final_smiles=final_smiles,
            product_smiles=expected_smiles,
            structural_match=structural_match,
            charge_match=charge_match,
            smiles_match=smiles_match,
            matches_product=charge_match and smiles_match,
            step_reports=reports,
        )

    @staticmethod
    def graph_to_smiles(graph: nx.Graph) -> str | None:
        """Export a Lewis graph through Kekule order using RDKit.

        :param graph: Lewis graph with ``kekule_order`` or sigma/pi edge fields.
        :type graph: nx.Graph
        :returns: Canonical SMILES, or ``None`` when RDKit reconstruction fails.
        :rtype: str | None
        """
        try:
            export_graph = normalize_lwg_graph(graph)
            mol = GraphToMol(edge_attributes={"order": "kekule_order"}).graph_to_mol(
                export_graph,
                sanitize=True,
                use_h_count=True,
            )
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def structural_match(final_graph: nx.Graph, product_graph: nx.Graph) -> bool:
        """Compare mapped atom identities and edge Kekule orders.

        :param final_graph: Final edited reactant graph.
        :type final_graph: nx.Graph
        :param product_graph: Product-side graph.
        :type product_graph: nx.Graph
        :returns: Whether mapped atoms and mapped Kekule edge orders match.
        :rtype: bool
        """
        return _mapped_node_signature(final_graph) == _mapped_node_signature(
            product_graph
        ) and _mapped_edge_signature(final_graph) == _mapped_edge_signature(
            product_graph
        )

    @staticmethod
    def charge_match(final_graph: nx.Graph, product_graph: nx.Graph) -> bool:
        """Compare charges for all mapped atoms.

        :param final_graph: Final edited reactant graph.
        :type final_graph: nx.Graph
        :param product_graph: Product-side graph.
        :type product_graph: nx.Graph
        :returns: Whether mapped atom formal charges match exactly.
        :rtype: bool
        """
        return _mapped_charge_signature(final_graph) == _mapped_charge_signature(
            product_graph
        )

    @staticmethod
    def _split_rsmi(rsmi: str) -> tuple[str, str]:
        if rsmi.count(">>") != 1:
            raise ValueError("RSMI must contain exactly one '>>' separator.")
        return tuple(rsmi.split(">>", 1))  # type: ignore[return-value]

    @staticmethod
    def _smiles_to_lwg(smiles: str) -> nx.Graph:
        graph = smiles_to_graph(
            smiles,
            drop_non_aam=False,
            sanitize=True,
            use_index_as_atom_map=False,
        )
        if graph is None:
            raise ValueError(f"Could not convert SMILES to graph: {smiles}")
        return normalize_lwg_graph(graph, in_place=True)

    @staticmethod
    def _parse_step(step: Sequence[Any]) -> tuple[str, list[int], list[int]]:
        if len(step) != 3:
            raise ValueError(f"Typed EPD step must have 3 items: {step!r}")

        action = str(step[0])
        source = [int(value) for value in step[1]]
        target = [int(value) for value in step[2]]
        return action, source, target

    def _apply_action(
        self,
        graph: nx.Graph,
        *,
        action: str,
        source: list[int],
        target: list[int],
    ) -> tuple[list[EdgeChange], list[LonePairChange], list[ChargeEdit], list[int]]:
        source_type, target_type = self._split_action(action)

        edge_changes: list[EdgeChange] = []
        lone_pair_changes: list[LonePairChange] = []
        charge_changes: list[ChargeEdit] = []
        changed_atom_maps: set[int] = set(source) | set(target)

        if source_type == "Sigma":
            edge_changes.append(
                change_edge_order(
                    graph,
                    source,
                    field="sigma_order",
                    delta=-1.0,
                    create=False,
                )
            )
            charge_changes.extend(change_atom_charge(graph, source, delta=1))
        elif source_type == "Pi":
            edge_changes.append(
                change_edge_order(
                    graph,
                    source,
                    field="pi_order",
                    delta=-1.0,
                    create=False,
                )
            )
            charge_changes.extend(change_atom_charge(graph, source, delta=1))
        elif source_type == "LP":
            lone_pair_changes.extend(change_lone_pairs(graph, source, delta=-1.0))
            charge_changes.extend(change_atom_charge(graph, source, delta=2))
        else:
            raise ValueError(f"Unsupported source action type: {source_type}")

        if target_type == "Sigma":
            edge_changes.append(
                change_edge_order(
                    graph,
                    target,
                    field="sigma_order",
                    delta=1.0,
                    create=True,
                )
            )
            charge_changes.extend(change_atom_charge(graph, target, delta=-1))
        elif target_type == "Pi":
            edge_changes.append(
                change_edge_order(
                    graph,
                    target,
                    field="pi_order",
                    delta=1.0,
                    create=False,
                )
            )
            charge_changes.extend(change_atom_charge(graph, target, delta=-1))
        elif target_type == "LP":
            lone_pair_changes.extend(change_lone_pairs(graph, target, delta=1.0))
            charge_changes.extend(change_atom_charge(graph, target, delta=-2))
        else:
            raise ValueError(f"Unsupported target action type: {target_type}")

        return (
            edge_changes,
            lone_pair_changes,
            charge_changes,
            sorted(changed_atom_maps),
        )

    @staticmethod
    def _split_action(action: str) -> tuple[str, str]:
        if "-/" not in action or not action.endswith("+"):
            raise ValueError(f"Unsupported typed action label: {action}")

        source_type, target_plus = action.split("-/", 1)
        return source_type, target_plus.removesuffix("+")


def _mapped_node_signature(graph: nx.Graph) -> dict[int, str]:
    return {
        int(attrs["atom_map"]): str(attrs.get("element", "*"))
        for _, attrs in graph.nodes(data=True)
        if int(attrs.get("atom_map", 0) or 0) != 0
    }


def _mapped_edge_signature(graph: nx.Graph) -> dict[tuple[int, int], float]:
    lookup = {
        node: int(attrs["atom_map"])
        for node, attrs in graph.nodes(data=True)
        if int(attrs.get("atom_map", 0) or 0) != 0
    }
    signature: dict[tuple[int, int], float] = {}

    for node_a, node_b, data in normalize_lwg_graph(graph).edges(data=True):
        map_a = lookup.get(node_a)
        map_b = lookup.get(node_b)
        if map_a is None or map_b is None:
            continue
        edge = tuple(sorted((map_a, map_b)))
        signature[edge] = float(data.get("kekule_order", data.get("order", 1.0)))

    return signature


def _mapped_charge_signature(graph: nx.Graph) -> dict[int, int | float]:
    return {
        int(attrs["atom_map"]): attrs.get("charge", 0)
        for _, attrs in graph.nodes(data=True)
        if int(attrs.get("atom_map", 0) or 0) != 0
    }
