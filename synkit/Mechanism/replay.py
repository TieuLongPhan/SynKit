"""Atomic replay and formal verification of supplied electron-flow groups."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import networkx as nx

from synkit.Graph.Mech.electron_accounting import (
    recompute_charge,
    refresh_electron_fields,
)
from synkit.Graph.Mech.lwg_ops import normalize_lwg_graph
from synkit.IO.chem_converter import smiles_to_graph

from .audit import audit_local_electron_state
from .model import (
    ElectronLocus,
    ElectronMoveGroup,
    MechanismRecord,
    VerificationCertificate,
    VerificationIssue,
    StereoDescriptor,
)
from .stereo_state import apply_stereo_effects, stereo_timeline
from .symbols import LONE_PAIR, PI, RADICAL, SIGMA


@dataclass(frozen=True)
class GroupReplayReport:
    step_id: str
    group_id: str
    status: str
    changed_atom_maps: tuple[int, ...]
    issues: tuple[VerificationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "group_id": self.group_id,
            "status": self.status,
            "changed_atom_maps": list(self.changed_atom_maps),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class MechanismReplayResult:
    final_graph: nx.Graph
    intermediates: tuple[nx.Graph, ...]
    certificate: VerificationCertificate
    mtg: nx.DiGraph


class MechanismReplayer:
    """Replay simultaneous groups against one common pre-state per group."""

    def __init__(
        self,
        *,
        validation: str = "strict",
        aromatic_policy: str = "kekule",
        repair: bool = False,
        verify_stereo: str = "off",
    ) -> None:
        if validation not in {"strict", "diagnostic"}:
            raise ValueError("validation must be 'strict' or 'diagnostic'.")
        if aromatic_policy not in {"kekule", "presentation"}:
            raise ValueError("Unsupported aromatic policy.")
        self.validation = validation
        self.aromatic_policy = aromatic_policy
        self.repair = repair
        if verify_stereo not in {"off", "endpoint", "stepwise"}:
            raise ValueError("verify_stereo must be 'off', 'endpoint', or 'stepwise'.")
        self.verify_stereo = verify_stereo

    def replay(self, record: MechanismRecord) -> MechanismReplayResult:
        reactants, products = record.mapped_reaction.split(">>", 1)
        current = self._parse_side(reactants)
        expected = self._parse_side(products)
        self._seed_mechanism_stereo(current)
        self._seed_mechanism_stereo(expected)
        intermediates: list[nx.Graph] = []
        reports: list[GroupReplayReport] = []
        issues: list[VerificationIssue] = list(record.grammar_issues())
        mtg = nx.DiGraph(schema="synkit-mtg-2.0-draft1")
        mtg.add_node(0, graph=deepcopy(current), status="INITIAL")

        if not issues:
            state_index = 0
            for step in record.steps:
                step_state_index = state_index
                for group in step.groups:
                    next_graph, group_report = self._apply_group(
                        current, group, step_id=step.step_id
                    )
                    reports.append(group_report)
                    issues.extend(group_report.issues)
                    if group_report.status != "VALID":
                        if self.validation == "strict":
                            break
                        continue
                    current = next_graph
                    intermediates.append(deepcopy(current))
                    state_index += 1
                    mtg.add_node(state_index, graph=deepcopy(current), status="VALID")
                    mtg.add_edge(
                        state_index - 1,
                        state_index,
                        step_id=step.step_id,
                        group_id=group.group_id,
                        macro=group.macro,
                        event_signature=group.canonical_signature(),
                    )
                if not (issues and self.validation == "strict"):
                    before_stereo = dict(
                        current.graph.get("mechanism_stereo_descriptors", {})
                    )
                    current, stereo_issues = apply_stereo_effects(
                        current,
                        step.stereo_effects,
                        step_id=step.step_id,
                    )
                    if self.verify_stereo == "stepwise":
                        issues.extend(stereo_issues)
                    after_stereo = current.graph.get("mechanism_stereo_descriptors", {})
                    if state_index > step_state_index:
                        intermediates[-1] = deepcopy(current)
                        mtg.nodes[state_index]["graph"] = deepcopy(current)
                        if step.stereo_effects:
                            mtg.edges[state_index - 1, state_index][
                                "stereo_effects"
                            ] = [effect.to_dict() for effect in step.stereo_effects]
                    elif step.stereo_effects or before_stereo != after_stereo:
                        intermediates.append(deepcopy(current))
                        state_index += 1
                        mtg.add_node(
                            state_index, graph=deepcopy(current), status="VALID"
                        )
                        mtg.add_edge(
                            state_index - 1,
                            state_index,
                            step_id=step.step_id,
                            group_id=None,
                            macro=None,
                            event_signature=(),
                            stereo_effects=[
                                effect.to_dict() for effect in step.stereo_effects
                            ],
                        )
                if issues and self.validation == "strict":
                    break

        final_match = self._compare_graphs(
            current,
            expected,
            include_stereo=self.verify_stereo in {"endpoint", "stepwise"},
        )
        if not issues and not final_match["matches"]:
            issues.append(
                VerificationIssue(
                    "FINAL_PRODUCT_MISMATCH",
                    "Replayed Lewis state does not match the supplied mapped product.",
                    expected=final_match["expected"],
                    observed=final_match["observed"],
                )
            )
        status = (
            "VALID"
            if not any(issue.severity == "error" for issue in issues)
            else "INVALID"
        )
        certificate = VerificationCertificate(
            status=status,
            step_reports=tuple(report.to_dict() for report in reports),
            issues=tuple(issues),
            final_match=final_match,
            repaired=self.repair
            and any(issue.code.endswith("_REPAIRED") for issue in issues),
        )
        ordered_states = [mtg.nodes[node]["graph"] for node in sorted(mtg.nodes)]
        mtg.graph["stereo_timeline"] = stereo_timeline(ordered_states)
        mtg.graph["verify_stereo"] = self.verify_stereo
        return MechanismReplayResult(current, tuple(intermediates), certificate, mtg)

    def _apply_group(
        self, pre_state: nx.Graph, group: ElectronMoveGroup, *, step_id: str
    ) -> tuple[nx.Graph, GroupReplayReport]:
        issues = [
            VerificationIssue(
                issue.code,
                issue.message,
                issue.severity,
                step_id=step_id,
                group_id=group.group_id,
                atom_maps=issue.atom_maps,
                expected=issue.expected,
                observed=issue.observed,
            )
            for issue in group.validate_pre_state(pre_state)
        ]
        changed = tuple(
            sorted(
                {
                    atom_map
                    for move in group.moves
                    for locus in (move.source, move.target)
                    for atom_map in locus.atom_maps
                }
            )
        )
        if issues:
            return pre_state, GroupReplayReport(
                step_id, group.group_id, "INVALID", changed, tuple(issues)
            )

        graph = deepcopy(pre_state)
        lookup = self._atom_map_lookup(graph)
        deltas: Counter[ElectronLocus] = Counter()
        for move in group.moves:
            deltas[move.source] -= move.electron_count
            deltas[move.target] += move.electron_count

        try:
            self._commit_bond_deltas(graph, lookup, deltas)
            self._commit_atom_deltas(graph, lookup, deltas)
            refresh_electron_fields(graph, in_place=True)
            for atom_map in changed:
                node = lookup[atom_map]
                if "valence_electrons" in graph.nodes[node]:
                    graph.nodes[node]["charge"] = recompute_charge(graph, node)
            electron_audit = audit_local_electron_state(graph, repair=self.repair)
            issues.extend(electron_audit.issues)
        except (KeyError, ValueError) as exc:
            issues.append(
                VerificationIssue(
                    "UNBALANCED_EVENT_GROUP",
                    str(exc),
                    step_id=step_id,
                    group_id=group.group_id,
                    atom_maps=changed,
                )
            )

        status = (
            "INVALID" if any(issue.severity == "error" for issue in issues) else "VALID"
        )
        return graph if status == "VALID" else pre_state, GroupReplayReport(
            step_id, group.group_id, status, changed, tuple(issues)
        )

    @staticmethod
    def _atom_map_lookup(graph: nx.Graph) -> dict[int, Any]:
        lookup: dict[int, Any] = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = int(attrs.get("atom_map", 0) or 0)
            if atom_map <= 0:
                continue
            if atom_map in lookup:
                raise ValueError(f"Duplicate atom map {atom_map}.")
            lookup[atom_map] = node
        return lookup

    @staticmethod
    def _commit_bond_deltas(
        graph: nx.Graph,
        lookup: dict[int, Any],
        deltas: Counter[ElectronLocus],
    ) -> None:
        for locus, electron_delta in deltas.items():
            if locus.kind not in {SIGMA, PI} or electron_delta == 0:
                continue
            left, right = (lookup[atom_map] for atom_map in locus.atom_maps)
            if not graph.has_edge(left, right):
                if electron_delta < 0:
                    raise ValueError(f"Cannot consume absent {locus.kind} bond.")
                graph.add_edge(
                    left,
                    right,
                    order=0.0,
                    kekule_order=0.0,
                    sigma_order=0.0,
                    pi_order=0.0,
                )
            field = "sigma_order" if locus.kind == SIGMA else "pi_order"
            current_electrons = 2 * float(graph.edges[left, right].get(field, 0.0))
            new_electrons = current_electrons + electron_delta
            if new_electrons < 0 or abs(new_electrons % 2) > 1e-9:
                raise ValueError(
                    f"Committed {locus.kind} resource is not a whole electron pair."
                )
            graph.edges[left, right][field] = new_electrons / 2

        for left, right in list(graph.edges):
            edge = graph.edges[left, right]
            sigma = float(edge.get("sigma_order", 0.0))
            pi = float(edge.get("pi_order", 0.0))
            if sigma == 0 and pi == 0:
                graph.remove_edge(left, right)
            else:
                edge["kekule_order"] = sigma + pi
                edge["order"] = sigma + pi

    @staticmethod
    def _commit_atom_deltas(
        graph: nx.Graph,
        lookup: dict[int, Any],
        deltas: Counter[ElectronLocus],
    ) -> None:
        per_atom: dict[int, Counter[str]] = {}
        for locus, electron_delta in deltas.items():
            if locus.kind not in {LONE_PAIR, RADICAL} or electron_delta == 0:
                continue
            per_atom.setdefault(locus.atom_maps[0], Counter())[
                locus.kind
            ] += electron_delta
        for atom_map, atom_deltas in per_atom.items():
            attrs = graph.nodes[lookup[atom_map]]
            lp_electrons = 2 * int(attrs.get("lone_pairs", 0)) + atom_deltas[LONE_PAIR]
            radicals = int(attrs.get("radical", 0)) + atom_deltas[RADICAL]
            if lp_electrons < 0 or lp_electrons % 2:
                raise ValueError("Lone-pair resource must commit as complete pairs.")
            if radicals < 0:
                raise ValueError("Committed radical resource cannot be negative.")
            attrs["lone_pairs"] = lp_electrons // 2
            attrs["radical"] = radicals

    def _compare_graphs(
        self, observed: nx.Graph, expected: nx.Graph, *, include_stereo: bool = False
    ) -> dict[str, Any]:
        observed_sig = self._graph_signature(observed, include_stereo=include_stereo)
        expected_sig = self._graph_signature(expected, include_stereo=include_stereo)
        return {
            "matches": observed_sig == expected_sig,
            "observed": observed_sig,
            "expected": expected_sig,
            "aromatic_policy": self.aromatic_policy,
        }

    def _graph_signature(
        self, graph: nx.Graph, *, include_stereo: bool = False
    ) -> dict[str, Any]:
        graph = normalize_lwg_graph(graph)
        lookup = self._atom_map_lookup(graph)
        nodes = {
            atom_map: (
                str(graph.nodes[node].get("element", "*")),
                int(graph.nodes[node].get("charge", 0)),
                int(graph.nodes[node].get("radical", 0)),
            )
            for atom_map, node in lookup.items()
        }
        reverse = {node: atom_map for atom_map, node in lookup.items()}
        edges = {}
        for left, right, attrs in graph.edges(data=True):
            if left not in reverse or right not in reverse:
                continue
            key = tuple(sorted((reverse[left], reverse[right])))
            if self.aromatic_policy == "presentation" and attrs.get("order") == 1.5:
                edges[key] = ("aromatic",)
            else:
                edges[key] = (
                    float(attrs.get("sigma_order", 0.0)),
                    float(attrs.get("pi_order", 0.0)),
                )
        signature = {"nodes": nodes, "edges": edges}
        if include_stereo:
            signature["stereo"] = {
                key: descriptor.to_dict()
                for key, descriptor in sorted(
                    graph.graph.get("mechanism_stereo_descriptors", {}).items()
                )
            }
        return signature

    @staticmethod
    def _seed_mechanism_stereo(graph: nx.Graph) -> None:
        registry = {}
        for key, descriptor in graph.graph.get("stereo_descriptors", {}).items():
            value = descriptor.to_dict()
            value["state"] = (
                "specified" if value.get("parity") is not None else "unknown"
            )
            registry[key] = StereoDescriptor.from_dict(value)
        graph.graph["mechanism_stereo_descriptors"] = registry

    @staticmethod
    def _parse_side(smiles: str) -> nx.Graph:
        graph = smiles_to_graph(
            smiles,
            drop_non_aam=False,
            sanitize=True,
            use_index_as_atom_map=False,
        )
        if graph is None:
            raise ValueError(f"Could not parse mapped mechanism side: {smiles!r}")
        return normalize_lwg_graph(graph, in_place=True)
