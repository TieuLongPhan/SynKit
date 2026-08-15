"""Atomic replay and formal verification of supplied electron-flow groups."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import networkx as nx

from synkit.Graph.Stereo import stereo_from_dict
from synkit.Graph.Mech.electron_accounting import (
    recompute_charge,
    refresh_electron_fields,
)
from synkit.Graph.Mech.lwg_ops import normalize_lwg_graph
from synkit.IO.chem_converter import DEFAULT_NODE_ATTRS, smiles_to_graph

from .audit import audit_local_electron_state
from .model import (
    ElectronLocus,
    ElectronMoveGroup,
    MechanismRecord,
    VerificationCertificate,
    VerificationIssue,
    StereoDescriptor,
)
from .stereo_state import (
    StereoStateTimeline,
    apply_stereo_effects,
    descriptor_support_errors,
    stereo_timeline,
)
from .symbols import LONE_PAIR, PI, RADICAL, SIGMA


@dataclass(frozen=True)
class GroupReplayReport:
    step_id: str
    group_id: str
    status: str
    changed_atom_maps: tuple[int, ...]
    issues: tuple[VerificationIssue, ...] = ()
    charge_updates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "group_id": self.group_id,
            "status": self.status,
            "changed_atom_maps": list(self.changed_atom_maps),
            "issues": [issue.to_dict() for issue in self.issues],
            "charge_updates": [dict(update) for update in self.charge_updates],
        }


@dataclass(frozen=True)
class MechanismReplayResult:
    final_graph: nx.Graph
    intermediates: tuple[nx.Graph, ...]
    certificate: VerificationCertificate
    mtg: nx.DiGraph

    @property
    def stereo_timeline(self) -> StereoStateTimeline:
        """Return the typed proof carried by the replay MTG."""
        return self.mtg.graph["stereo_timeline_proof"]


class MechanismReplayer:
    """Replay simultaneous groups against one common pre-state per group.

    Formal charges are propagated from committed sigma, pi, lone-pair, and
    radical deltas. Absolute Lewis-state residuals are retained as an
    independent certificate claim. Endpoint resource comparison is strict by
    default; ``closed_shell_pair`` is an explicit, atom-selected comparison
    policy for a two-radical/one-lone-pair serialization boundary.
    """

    def __init__(
        self,
        *,
        validation: str = "strict",
        aromatic_policy: str = "presentation",
        repair: bool = False,
        verify_stereo: str = "off",
        endpoint_resource_policy: str | None = None,
        closed_shell_atom_maps: set[int] | tuple[int, ...] | list[int] | None = None,
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
        if endpoint_resource_policy not in {None, "strict", "closed_shell_pair"}:
            raise ValueError("Unsupported endpoint resource policy.")
        self.endpoint_resource_policy = endpoint_resource_policy
        self.closed_shell_atom_maps = frozenset(
            int(atom_map) for atom_map in (closed_shell_atom_maps or ())
        )
        if any(atom_map <= 0 for atom_map in self.closed_shell_atom_maps):
            raise ValueError("Closed-shell atom maps must be positive integers.")

    def replay(self, record: MechanismRecord) -> MechanismReplayResult:  # noqa: C901
        reactants, products = record.mapped_reaction.split(">>", 1)
        current = self._parse_side(reactants)
        expected = self._parse_side(products)
        self._seed_mechanism_stereo(current)
        self._seed_mechanism_stereo(expected)
        self._select_endpoint_stereo(current, record, side="reactant")
        self._select_endpoint_stereo(expected, record, side="product")
        endpoint_policy, closed_shell_atom_maps = self._resolve_endpoint_policy(record)
        intermediates: list[nx.Graph] = []
        reports: list[GroupReplayReport] = []
        canonical_neighbor_changes: list[dict[str, Any]] = []
        issues: list[VerificationIssue] = list(record.grammar_issues())
        issues.extend(self._graph_mapping_issues(current, side="reactant"))
        issues.extend(self._graph_mapping_issues(expected, side="product"))
        initial_absolute_audit = audit_local_electron_state(current)
        product_absolute_audit = audit_local_electron_state(expected)
        issues.extend(self._blocking_audit_issues(initial_absolute_audit.issues))
        issues.extend(self._blocking_audit_issues(product_absolute_audit.issues))
        issues.extend(
            self._endpoint_policy_issues(
                current,
                expected,
                policy=endpoint_policy,
                atom_maps=closed_shell_atom_maps,
            )
        )
        issues.extend(self._endpoint_stereo_issues(current, side="reactant"))
        issues.extend(self._endpoint_stereo_issues(expected, side="product"))
        initial_absolute = self._absolute_state(current)
        product_absolute = self._absolute_state(expected)
        committed_absolute_states = [initial_absolute]
        mtg = nx.DiGraph(schema="synkit-mtg-2.0-draft1")
        mtg.add_node(0, graph=deepcopy(current), status="INITIAL")

        if not issues:
            state_index = 0
            for step in record.steps:
                step_start = deepcopy(current)
                step_state_index = state_index
                step_intermediate_count = len(intermediates)
                step_failed = False
                for group in step.groups:
                    next_graph, group_report = self._apply_group(
                        current, group, step_id=step.step_id
                    )
                    reports.append(group_report)
                    issues.extend(group_report.issues)
                    if group_report.status != "VALID":
                        if self.validation == "strict":
                            step_failed = True
                            break
                        continue
                    current = next_graph
                    committed_absolute_states.append(self._absolute_state(current))
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
                if not step_failed:
                    motion_evidence = tuple(
                        motion.verification_evidence(
                            step_id=step.step_id,
                            before_graph=step_start,
                            graph=current,
                        )
                        for motion in step.stereo_motions
                    )
                    step_neighbor_changes = [
                        change.to_dict()
                        for changes, _motion_issues in motion_evidence
                        for change in changes
                    ]
                    motion_issues = tuple(
                        issue
                        for _changes, result_issues in motion_evidence
                        for issue in result_issues
                    )
                    if self.verify_stereo == "stepwise":
                        issues.extend(motion_issues)
                        if motion_issues and self.validation == "strict":
                            step_failed = True
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
                        if stereo_issues and self.validation == "strict":
                            step_failed = True
                    after_stereo = current.graph.get("mechanism_stereo_descriptors", {})
                    if not step_failed:
                        canonical_neighbor_changes.extend(step_neighbor_changes)
                    if not step_failed and state_index > step_state_index:
                        intermediates[-1] = deepcopy(current)
                        mtg.nodes[state_index]["graph"] = deepcopy(current)
                        if step.stereo_effects:
                            mtg.edges[state_index - 1, state_index][
                                "stereo_effects"
                            ] = [effect.to_dict() for effect in step.stereo_effects]
                        if step.stereo_motions:
                            mtg.edges[state_index - 1, state_index][
                                "stereo_motions"
                            ] = [motion.to_dict() for motion in step.stereo_motions]
                        if step_neighbor_changes:
                            mtg.edges[state_index - 1, state_index][
                                "canonical_neighbor_changes"
                            ] = step_neighbor_changes
                    elif not step_failed and (
                        step.stereo_effects
                        or step.stereo_motions
                        or before_stereo != after_stereo
                    ):
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
                            stereo_motions=[
                                motion.to_dict() for motion in step.stereo_motions
                            ],
                            canonical_neighbor_changes=step_neighbor_changes,
                        )
                if step_failed and self.validation == "strict":
                    current = step_start
                    del intermediates[step_intermediate_count:]
                    mtg.remove_nodes_from(
                        node for node in tuple(mtg.nodes) if node > step_state_index
                    )
                    state_index = step_state_index
                    break

        final_match = self._compare_graphs(
            current,
            expected,
            include_stereo=self.verify_stereo in {"endpoint", "stepwise"},
            endpoint_resource_policy=endpoint_policy,
            closed_shell_atom_maps=closed_shell_atom_maps,
        )
        final_match["stereo_verification"] = self.verify_stereo
        final_match["stereo_verification_performed"] = self.verify_stereo != "off"
        final_match["canonical_neighbor_changes"] = canonical_neighbor_changes
        transition_valid = not any(issue.severity == "error" for issue in issues)
        observed_charges = self._mapped_charges(current)
        expected_charges = self._mapped_charges(expected)
        delta_charge_match = observed_charges == expected_charges
        residuals_invariant = all(
            state == initial_absolute
            for state in (*committed_absolute_states[1:], product_absolute)
        )
        if transition_valid and not delta_charge_match:
            issues.append(
                VerificationIssue(
                    "DELTA_CHARGE_MISMATCH",
                    "Delta-derived final charges do not match the mapped product.",
                    atom_maps=tuple(
                        sorted(
                            atom_map
                            for atom_map in set(observed_charges) | set(expected_charges)
                            if observed_charges.get(atom_map)
                            != expected_charges.get(atom_map)
                        )
                    ),
                    expected=expected_charges,
                    observed=observed_charges,
                )
            )
        if transition_valid and not residuals_invariant:
            issues.append(
                VerificationIssue(
                    "ABSOLUTE_RESIDUAL_CHANGED",
                    "Absolute Lewis residuals changed across delta replay.",
                    expected=initial_absolute,
                    observed={
                        "committed_states": committed_absolute_states[1:],
                        "product": product_absolute,
                    },
                )
            )
        if transition_valid and not final_match["matches"]:
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
        absolute_lwg_valid = initial_absolute_audit.valid and product_absolute_audit.valid
        normalization_required = bool(
            final_match.get("matches")
            and not final_match.get("raw_matches", final_match["matches"])
            and final_match.get("normalization_evidence")
        )
        if status == "INVALID":
            verification_level = "INVALID"
        elif normalization_required:
            verification_level = "NORMALIZED"
        elif absolute_lwg_valid:
            verification_level = "EXACT"
        else:
            verification_level = "DELTA_CONSISTENT"
        diagnostics = []
        if not absolute_lwg_valid:
            diagnostics.append("ABSOLUTE_LEWIS_STATE_INCOMPLETE")
        if normalization_required:
            diagnostics.append("ENDPOINT_RESOURCE_NORMALIZED")
        certificate = VerificationCertificate(
            status=status,
            verification_level=verification_level,
            transition_valid=transition_valid,
            delta_charge_match=delta_charge_match,
            endpoint_match=bool(final_match["matches"]),
            absolute_lwg_valid=absolute_lwg_valid,
            initial_absolute_residuals=initial_absolute["atom_residuals"],
            product_absolute_residuals=product_absolute["atom_residuals"],
            initial_global_electron_residual=initial_absolute["global_residual"],
            product_global_electron_residual=product_absolute["global_residual"],
            absolute_residuals_invariant=residuals_invariant,
            endpoint_resource_policy=endpoint_policy,
            normalization_evidence=tuple(final_match.get("normalization_evidence", ())),
            diagnostics=tuple(diagnostics),
            step_reports=tuple(report.to_dict() for report in reports),
            issues=tuple(issues),
            final_match=final_match,
            repaired=self.repair
            and any(issue.code.endswith("_REPAIRED") for issue in issues),
        )
        ordered_states = [mtg.nodes[node]["graph"] for node in sorted(mtg.nodes)]
        mtg.graph["stereo_timeline"] = stereo_timeline(ordered_states)
        mtg.graph["verify_stereo"] = self.verify_stereo
        mtg.graph["stereo_timeline_proof"] = StereoStateTimeline.create(
            ordered_states,
            verification_mode=self.verify_stereo,
        )
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
        deltas: Counter[ElectronLocus] = Counter()
        for move in group.moves:
            if (
                group.macro == "LONE_PAIR_RADICAL_RELOCATION"
                and move.electron_count == 1
                and move.source.kind == LONE_PAIR
                and move.target.kind == LONE_PAIR
            ):
                # A fishhook can split one donor lone pair and pair with one
                # acceptor radical. Pair/ radical resources must remain whole
                # at the committed boundary even though only one electron
                # moves between atoms.
                deltas[move.source] -= 2
                deltas[ElectronLocus(RADICAL, move.source.atom_maps)] += 1
                deltas[ElectronLocus(RADICAL, move.target.atom_maps)] -= 1
                deltas[move.target] += 2
            else:
                deltas[move.source] -= move.electron_count
                deltas[move.target] += move.electron_count

        try:
            lookup = self._atom_map_lookup(graph)
            charge_deltas = self._charge_deltas(deltas)
            charge_updates = self._commit_charge_deltas(
                graph,
                lookup,
                charge_deltas,
            )
            self._commit_bond_deltas(graph, lookup, deltas)
            self._commit_atom_deltas(graph, lookup, deltas)
            refresh_electron_fields(graph, in_place=True)
            electron_audit = audit_local_electron_state(graph)
            issues.extend(self._blocking_audit_issues(electron_audit.issues))
        except (KeyError, ValueError) as exc:
            charge_updates = ()
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
            step_id,
            group.group_id,
            status,
            changed,
            tuple(issues),
            tuple(charge_updates),
        )

    @staticmethod
    def _charge_deltas(
        deltas: Counter[ElectronLocus],
    ) -> Counter[int]:
        """Derive formal-charge deltas from committed electron resources."""
        result: Counter[int] = Counter()
        for locus, electron_delta in deltas.items():
            if electron_delta == 0:
                continue
            if locus.kind in {SIGMA, PI}:
                charge_delta = -electron_delta / 2
                for atom_map in locus.atom_maps:
                    result[atom_map] += charge_delta
            elif locus.kind in {LONE_PAIR, RADICAL}:
                result[locus.atom_maps[0]] -= electron_delta
        return result

    @staticmethod
    def _commit_charge_deltas(
        graph: nx.Graph,
        lookup: dict[int, Any],
        charge_deltas: Counter[int],
    ) -> tuple[dict[str, Any], ...]:
        """Apply delta-authoritative charges without an absolute Lewis rewrite."""
        updates = []
        for atom_map in sorted(charge_deltas):
            delta = charge_deltas[atom_map]
            if isinstance(delta, float) and delta.is_integer():
                delta = int(delta)
            node = lookup[atom_map]
            before = graph.nodes[node].get("charge", 0)
            after = before + delta
            if isinstance(after, float) and after.is_integer():
                after = int(after)
            graph.nodes[node]["charge"] = after
            updates.append(
                {
                    "atom_map": atom_map,
                    "before": before,
                    "delta": delta,
                    "after": after,
                }
            )
        return tuple(updates)

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
    def _graph_mapping_issues(
        graph: nx.Graph,
        *,
        side: str,
    ) -> tuple[VerificationIssue, ...]:
        """Require complete, unique AAM for formal mapped endpoint replay."""
        issues: list[VerificationIssue] = []
        seen: dict[int, Any] = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map", 0)
            if type(atom_map) is not int or atom_map <= 0:
                issues.append(
                    VerificationIssue(
                        "MISSING_ATOM_MAP",
                        f"Every {side} atom requires a positive atom map.",
                        observed={"node": node, "atom_map": atom_map},
                    )
                )
                continue
            if atom_map in seen:
                issues.append(
                    VerificationIssue(
                        "DUPLICATE_ATOM_MAP",
                        f"The {side} graph contains duplicate atom map {atom_map}.",
                        atom_maps=(atom_map,),
                        observed=(seen[atom_map], node),
                    )
                )
            else:
                seen[atom_map] = node
        return tuple(issues)

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
        self,
        observed: nx.Graph,
        expected: nx.Graph,
        *,
        include_stereo: bool = False,
        endpoint_resource_policy: str | None = None,
        closed_shell_atom_maps: frozenset[int] | None = None,
    ) -> dict[str, Any]:
        presentation_edges: set[frozenset[int]] | None = None
        if self.aromatic_policy == "presentation":
            presentation_edges = self._mapped_aromatic_edges(
                observed
            ) | self._mapped_aromatic_edges(expected)
        raw_observed_sig = self._graph_signature(
            observed,
            include_stereo=include_stereo,
            presentation_edges=presentation_edges,
        )
        raw_expected_sig = self._graph_signature(
            expected,
            include_stereo=include_stereo,
            presentation_edges=presentation_edges,
        )
        policy = endpoint_resource_policy or self.endpoint_resource_policy or "strict"
        selected_maps = (
            self.closed_shell_atom_maps
            if closed_shell_atom_maps is None
            else closed_shell_atom_maps
        )
        normalization_evidence: tuple[dict[str, Any], ...] = ()
        observed_sig = raw_observed_sig
        expected_sig = raw_expected_sig
        if policy == "closed_shell_pair":
            observed_working, observed_evidence = self._closed_shell_pair_copy(
                observed,
                selected_maps,
                side="observed",
            )
            expected_working, expected_evidence = self._closed_shell_pair_copy(
                expected,
                selected_maps,
                side="expected",
            )
            normalization_evidence = (*observed_evidence, *expected_evidence)
            observed_sig = self._graph_signature(
                observed_working,
                include_stereo=include_stereo,
                presentation_edges=presentation_edges,
            )
            expected_sig = self._graph_signature(
                expected_working,
                include_stereo=include_stereo,
                presentation_edges=presentation_edges,
            )
        return {
            "matches": observed_sig == expected_sig,
            "raw_matches": raw_observed_sig == raw_expected_sig,
            "observed": observed_sig,
            "expected": expected_sig,
            "raw_observed": raw_observed_sig,
            "raw_expected": raw_expected_sig,
            "aromatic_policy": self.aromatic_policy,
            "endpoint_resource_policy": policy,
            "normalization_evidence": list(normalization_evidence),
        }

    @classmethod
    def _closed_shell_pair_copy(
        cls,
        graph: nx.Graph,
        atom_maps: frozenset[int],
        *,
        side: str,
    ) -> tuple[nx.Graph, tuple[dict[str, Any], ...]]:
        """Normalize explicitly selected diradical scalars on a graph copy."""
        working = deepcopy(graph)
        lookup = cls._atom_map_lookup(working)
        evidence = []
        for atom_map in sorted(atom_maps):
            node = lookup.get(atom_map)
            if node is None:
                continue
            attrs = working.nodes[node]
            if int(attrs.get("radical", 0)) != 2:
                continue
            before = {
                "charge": attrs.get("charge", 0),
                "lone_pairs": int(attrs.get("lone_pairs", 0)),
                "radical": 2,
            }
            attrs["lone_pairs"] = before["lone_pairs"] + 1
            attrs["radical"] = 0
            after = {
                "charge": attrs.get("charge", 0),
                "lone_pairs": attrs["lone_pairs"],
                "radical": 0,
            }
            evidence.append(
                {
                    "atom_map": atom_map,
                    "element": str(attrs.get("element", "*")),
                    "side": side,
                    "before": before,
                    "after": after,
                    "policy_name": "closed_shell_pair",
                    "policy_version": "1.0",
                }
            )
        return working, tuple(evidence)

    def _resolve_endpoint_policy(
        self,
        record: MechanismRecord,
    ) -> tuple[str, frozenset[int]]:
        policy = self.endpoint_resource_policy
        atom_maps = self.closed_shell_atom_maps
        if policy is None:
            policy = str(record.metadata.get("endpoint_resource_policy", "strict"))
            atom_maps = frozenset(
                int(atom_map)
                for atom_map in record.metadata.get("closed_shell_atom_maps", atom_maps)
            )
        if policy not in {"strict", "closed_shell_pair"}:
            raise ValueError(f"Unsupported endpoint resource policy: {policy!r}.")
        return policy, atom_maps

    @classmethod
    def _endpoint_policy_issues(
        cls,
        reactant: nx.Graph,
        product: nx.Graph,
        *,
        policy: str,
        atom_maps: frozenset[int],
    ) -> tuple[VerificationIssue, ...]:
        if policy != "closed_shell_pair":
            return ()
        if not atom_maps:
            return (
                VerificationIssue(
                    "CLOSED_SHELL_ATOM_MAPS_REQUIRED",
                    "The closed-shell-pair policy requires explicit mapped atoms.",
                ),
            )
        reactant_maps = set(cls._atom_map_lookup(reactant))
        product_maps = set(cls._atom_map_lookup(product))
        missing = atom_maps - (reactant_maps & product_maps)
        if not missing:
            return ()
        return (
            VerificationIssue(
                "CLOSED_SHELL_ATOM_MAP_MISSING",
                "Every selected closed-shell atom must occur on both endpoints.",
                atom_maps=tuple(sorted(missing)),
            ),
        )

    @staticmethod
    def _blocking_audit_issues(
        issues: tuple[VerificationIssue, ...],
    ) -> tuple[VerificationIssue, ...]:
        """Keep malformed resources blocking while absolute mismatch is diagnostic."""
        absolute_codes = {
            "LOCAL_ELECTRON_MISMATCH",
            "GLOBAL_ELECTRON_MISMATCH",
        }
        return tuple(issue for issue in issues if issue.code not in absolute_codes)

    @staticmethod
    def _number(value: float) -> int | float:
        return int(value) if float(value).is_integer() else value

    @classmethod
    def _absolute_state(cls, graph: nx.Graph) -> dict[str, Any]:
        atom_residuals = {}
        expected_total = 0.0
        represented_total = 0.0
        for node, attrs in graph.nodes(data=True):
            atom_map = int(attrs.get("atom_map", 0) or 0)
            if "valence_electrons" in attrs:
                residual = float(attrs.get("charge", 0)) - float(
                    recompute_charge(graph, node)
                )
                if residual:
                    atom_residuals[atom_map] = cls._number(residual)
                expected_total += (
                    float(attrs["valence_electrons"])
                    + float(attrs.get("hcount", 0))
                    - float(attrs.get("charge", 0))
                )
            represented_total += (
                2 * float(attrs.get("lone_pairs", 0))
                + float(attrs.get("radical", 0))
                + 2 * float(attrs.get("hcount", 0))
            )
        represented_total += 2 * sum(
            float(attrs.get("sigma_order", 0.0))
            + float(attrs.get("pi_order", 0.0))
            for _, _, attrs in graph.edges(data=True)
        )
        return {
            "atom_residuals": atom_residuals,
            "global_residual": cls._number(represented_total - expected_total),
        }

    @staticmethod
    def _mapped_charges(graph: nx.Graph) -> dict[int, int | float]:
        return {
            int(attrs["atom_map"]): attrs.get("charge", 0)
            for _, attrs in graph.nodes(data=True)
            if int(attrs.get("atom_map", 0) or 0) > 0
        }

    def _graph_signature(
        self,
        graph: nx.Graph,
        *,
        include_stereo: bool = False,
        presentation_edges: set[frozenset[int]] | None = None,
    ) -> dict[str, Any]:
        # ``normalize_lwg_graph`` deliberately replaces the stable aromatic
        # presentation order (1.5) with the editable Kekule sigma/pi order.
        # Remember which input edges were aromatic before normalization so
        # presentation comparison remains independent of the arbitrary
        # alternating Kekule phase selected by RDKit.
        aromatic_edges = {
            frozenset((left, right))
            for left, right, attrs in graph.edges(data=True)
            if attrs.get("aromatic") is True or attrs.get("order") == 1.5
        }
        graph = normalize_lwg_graph(graph)
        lookup: dict[int, Any] = {}
        node_keys: dict[Any, Any] = {}
        duplicate_counts: Counter[int] = Counter()
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map", 0)
            if type(atom_map) is int and atom_map > 0 and atom_map not in lookup:
                lookup[atom_map] = node
                node_keys[node] = atom_map
            elif type(atom_map) is int and atom_map > 0:
                duplicate_counts[atom_map] += 1
                node_keys[node] = f"duplicate:{atom_map}:{duplicate_counts[atom_map]}"
            else:
                node_keys[node] = f"unmapped:{node}"
        nodes = {
            node_keys[node]: (
                str(graph.nodes[node].get("element", "*")),
                int(graph.nodes[node].get("isotope", 0) or 0),
                int(graph.nodes[node].get("charge", 0)),
                int(graph.nodes[node].get("radical", 0)),
                int(graph.nodes[node].get("hcount", 0)),
                int(graph.nodes[node].get("lone_pairs", 0)),
                int(graph.nodes[node].get("valence_electrons", 0)),
            )
            for node in graph.nodes
        }
        edges = {}
        for left, right, attrs in graph.edges(data=True):
            endpoints = tuple(sorted((node_keys[left], node_keys[right]), key=repr))
            key = f"{endpoints[0]!r}|{endpoints[1]!r}"
            mapped_key = (
                frozenset((node_keys[left], node_keys[right]))
                if all(type(node_keys[node]) is int for node in (left, right))
                else None
            )
            if self.aromatic_policy == "presentation" and (
                frozenset((left, right)) in aromatic_edges
                or (
                    mapped_key is not None
                    and presentation_edges is not None
                    and mapped_key in presentation_edges
                )
            ):
                edges[key] = ("aromatic",)
            else:
                edges[key] = (
                    float(attrs.get("sigma_order", 0.0)),
                    float(attrs.get("pi_order", 0.0)),
                )
        signature = {"nodes": nodes, "edges": edges}
        if include_stereo:
            signature["stereo"] = {
                key: self._stereo_descriptor_signature(descriptor)
                for key, descriptor in sorted(
                    graph.graph.get("mechanism_stereo_descriptors", {}).items()
                )
            }
        return signature

    @staticmethod
    def _mapped_aromatic_edges(graph: nx.Graph) -> set[frozenset[int]]:
        """Return mapped bonds rendered aromatically by the input toolkit."""
        edges: set[frozenset[int]] = set()
        for left, right, attrs in graph.edges(data=True):
            if not (attrs.get("aromatic") is True or attrs.get("order") == 1.5):
                continue
            atom_maps = (
                graph.nodes[left].get("atom_map"),
                graph.nodes[right].get("atom_map"),
            )
            if all(type(atom_map) is int and atom_map > 0 for atom_map in atom_maps):
                edges.add(frozenset(atom_maps))
        return edges

    @staticmethod
    def _stereo_descriptor_signature(descriptor: StereoDescriptor) -> tuple[Any, ...]:
        if descriptor.descriptor_class == "unknown":
            return ("unknown", descriptor.state)
        native = stereo_from_dict(descriptor.to_dict())
        return (descriptor.state, *native.canonical_form())

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
    def _select_endpoint_stereo(
        graph: nx.Graph,
        record: MechanismRecord,
        *,
        side: str,
    ) -> None:
        """Use an explicit endpoint sidecar when SMILES cannot encode state."""
        if side in record.endpoint_stereo:
            graph.graph["mechanism_stereo_descriptors"] = dict(
                record.endpoint_stereo[side]
            )

    @staticmethod
    def _endpoint_stereo_issues(
        graph: nx.Graph,
        *,
        side: str,
    ) -> tuple[VerificationIssue, ...]:
        issues = []
        for key, descriptor in graph.graph.get(
            "mechanism_stereo_descriptors", {}
        ).items():
            for reason in descriptor_support_errors(
                graph,
                descriptor,
                registry_key=key,
            ):
                issues.append(
                    VerificationIssue(
                        "INVALID_ENDPOINT_STEREO",
                        f"Invalid {side} descriptor {key}: {reason}.",
                    )
                )
        return tuple(issues)

    @staticmethod
    def _parse_side(smiles: str) -> nx.Graph:
        graph = smiles_to_graph(
            smiles,
            drop_non_aam=False,
            sanitize=True,
            use_index_as_atom_map=False,
            node_attrs=(*DEFAULT_NODE_ATTRS, "isotope"),
        )
        if graph is None:
            raise ValueError(f"Could not parse mapped mechanism side: {smiles!r}")
        return normalize_lwg_graph(graph, in_place=True)
