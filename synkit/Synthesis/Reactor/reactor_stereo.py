"""Stereo propagation and application behavior for SynReactor."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple

import networkx as nx
from rdkit import Chem

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.IO.graph_to_mol import GraphToMol

MappingDict = Dict[Any, Any]


class ReactorStereoMixin:
    """Internal stereo behavior shared by the SynReactor facade."""

    def _matching_host_graph(self) -> nx.Graph:
        """Return the host graph normalized to the active rule representation."""
        if self._host_for_matching is None:
            host = self.graph.raw
            if getattr(self.rule, "_format", None) == "tuple":
                host = self._implicit_heavy_hydrogens(
                    host,
                    preserve_mapped_hydrogens=self.preserve_mapped_hydrogens,
                )
            self._host_for_matching = host
        return self._host_for_matching

    def _noncovariant_propagation_targets(self) -> set[str]:
        """Return guards whose coupled product stereo cannot be inferred.

        One same-locus endpoint change can be mirrored directly. A destroyed
        descriptor with no other stereo output is also safe. If other product
        descriptors are formed, however, their correlation to an inverse
        input is not encoded by independent ``StereoChange`` objects; such a
        rule must match its reference geometry or provide a companion rule.
        """
        from synkit.Graph.Stereo import descriptor_id

        exact_targets = set()
        for target, guard in self.rule.stereo_guards.items():
            effect = self.rule.stereo_effects.get(target)
            other_outputs = any(
                key != target and change.after is not None
                for key, change in self.rule.stereo_effects.items()
            )
            same_locus_output = (
                effect is not None
                and effect.after is not None
                and type(guard) is type(effect.after)
                and descriptor_id(effect.after) == target
            )
            safe_destruction = (
                effect is not None and effect.after is None and not other_outputs
            )
            if not same_locus_output and not safe_destruction:
                exact_targets.add(target)
            elif other_outputs:
                exact_targets.add(target)
        return exact_targets

    @staticmethod
    def _unknown_stereo_orientation(descriptor: Any) -> Any:
        """Keep a descriptor locus/reference frame but remove its orientation."""
        return type(descriptor)(
            descriptor.atoms,
            None,
            descriptor.provenance,
        )

    def _propagated_stereo_effect(
        self,
        change: Any,
        reactant_registry: Mapping[str, Any],
        host: nx.Graph,
    ) -> tuple[Any, Any]:
        """Apply an aligned reaction relation to the matched substrate frame.

        The rule endpoint is only a reference frame. In chemical ``propagate``
        mode the stored permutation witness acts on the concrete substrate
        configuration after ligand replacement. Missing, unknown, or refused
        alignment evidence never acquires arbitrary template orientation.
        """
        before, after = change.before, change.after
        if self.stereo_mode != "propagate" or before is None:
            return before, after

        from synkit.Graph.Stereo import (
            StereoAlignmentError,
            StereoRelationKind,
            descriptor_id,
            normalize_hydrogen_references,
        )

        target = descriptor_id(before)
        candidate = reactant_registry.get(target)
        linked_output = (
            after is not None
            and type(before) is type(after)
            and descriptor_id(after) == target
        )

        if candidate is None:
            unknown_before = self._unknown_stereo_orientation(before)
            unknown_after = (
                self._unknown_stereo_orientation(after) if linked_output else after
            )
            return unknown_before, unknown_after

        if candidate.parity is None:
            try:
                propagated = (
                    change.apply_to(
                        candidate,
                        semantics=self.stereo_semantics,
                        diagnostics=self._stereo_semantic_diagnostics,
                    )
                    if linked_output
                    else after
                )
            except StereoAlignmentError:
                propagated = (
                    self._unknown_stereo_orientation(after) if linked_output else after
                )
            return candidate, propagated

        expected = normalize_hydrogen_references(before, host)
        actual = normalize_hydrogen_references(candidate, host)
        relation = expected.relation_to(actual)
        if relation.kind not in {
            StereoRelationKind.UNRELATED,
            StereoRelationKind.UNSPECIFIED,
        }:
            try:
                return candidate, (
                    change.apply_to(
                        actual,
                        semantics=self.stereo_semantics,
                        diagnostics=self._stereo_semantic_diagnostics,
                    )
                    if linked_output
                    else after
                )
            except StereoAlignmentError:
                pass

        unknown_after = (
            self._unknown_stereo_orientation(after) if linked_output else after
        )
        return candidate, unknown_after

    @staticmethod
    def _node_by_atom_map(graph: nx.Graph) -> Dict[int, Any]:
        """Return product nodes keyed by the application-local atom map."""
        result = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map", node)
            if isinstance(atom_map, int) and atom_map > 0:
                result[atom_map] = node
        return result

    @staticmethod
    def _node_reference(graph: nx.Graph, node: Any) -> int | None:
        """Return the stable mapped reference for one product node."""
        atom_map = graph.nodes[node].get("atom_map", node)
        return atom_map if isinstance(atom_map, int) and atom_map > 0 else None

    def _coupled_planar_product(self, coupling: Any, product: nx.Graph) -> Any:
        """Construct a formed alkene descriptor from a syn/anti coupling."""
        by_map = self._node_by_atom_map(product)
        left_center, right_center = coupling.centers
        left_ligand, right_ligand = coupling.ligands
        required = {left_center, right_center, left_ligand, right_ligand}
        if not required <= set(by_map):
            return None
        left_node, right_node = by_map[left_center], by_map[right_center]
        if not product.has_edge(left_node, right_node):
            return None
        edge = product.edges[left_node, right_node]
        if float(edge.get("pi_order", 0.0)) < 1.0:
            return None

        references = []
        for center, other, ligand in (
            (left_center, right_center, left_ligand),
            (right_center, left_center, right_ligand),
        ):
            center_node = by_map[center]
            excluded = {by_map[other], by_map[ligand]}
            peripheral = [
                self._node_reference(product, neighbor)
                for neighbor in product.neighbors(center_node)
                if neighbor not in excluded
            ]
            peripheral = [value for value in peripheral if value is not None]
            if len(peripheral) == 1:
                references.append(peripheral[0])
            elif not peripheral and product.nodes[center_node].get("hcount", 0) == 1:
                references.append(f"@H:{center}")
            else:
                return None
        return coupling.planar_product_descriptor(*references)

    @staticmethod
    def _potential_tetrahedral_atom_maps(product: nx.Graph) -> set[int] | None:
        """Return constitutionally stereogenic tetrahedral product centers.

        Coupling geometry must not create a descriptor when two ligands are
        constitutionally identical, as in ordinary H2 addition to a CH=CH
        alkene. ``None`` is a conservative reconstruction-failure sentinel:
        callers then retain the rule-derived descriptors rather than losing a
        potentially valid stereocenter.
        """
        probe = product.copy()
        probe.graph["stereo_descriptors"] = {}
        try:
            molecule = GraphToMol().graph_to_mol(probe)
        except Exception:
            return None
        atom_maps = {
            atom.GetIdx(): atom.GetAtomMapNum() for atom in molecule.GetAtoms()
        }
        return {
            atom_maps[stereo.centeredOn]
            for stereo in Chem.FindPotentialStereo(molecule)
            if stereo.type == Chem.StereoType.Atom_Tetrahedral
            and atom_maps.get(stereo.centeredOn, 0) > 0
        }

    def _apply_stereo_couplings(
        self,
        states: List[Tuple[Dict, Dict, Dict, Dict, Dict]],
        couplings: List[Any],
        reactant_registry: Mapping[str, Any],
        product_graph: nx.Graph,
    ) -> List[Tuple[Dict, Dict, Dict, Dict, Dict]]:
        """Derive coupled endpoint descriptors from chemical rule semantics."""
        if not couplings:
            return states

        from synkit.Graph.Stereo import PlanarBondStereo, descriptor_id
        from synkit.Graph.Stereo.changes import StereoChange

        potential_tetrahedral = self._potential_tetrahedral_atom_maps(product_graph)
        for coupling in couplings:
            if coupling.kind != "VICINAL_ADDITION":
                continue
            planar_product = self._coupled_planar_product(coupling, product_graph)
            planar_reactant = reactant_registry.get(coupling.target)
            next_states = []
            for (
                product_registry,
                relabeled_changes,
                branch_metadata,
                outcome_metadata,
                coupling_branch_metadata,
            ) in states:
                if planar_product is not None:
                    registry = dict(product_registry)
                    changes = dict(relabeled_changes)
                    registry[coupling.target] = planar_product
                    changes[coupling.target] = StereoChange(
                        "FORMED", None, planar_product
                    )
                    next_states.append(
                        (
                            registry,
                            changes,
                            dict(branch_metadata),
                            dict(outcome_metadata),
                            dict(coupling_branch_metadata),
                        )
                    )
                    continue

                if not isinstance(planar_reactant, PlanarBondStereo):
                    next_states.append(
                        (
                            product_registry,
                            relabeled_changes,
                            branch_metadata,
                            outcome_metadata,
                            coupling_branch_metadata,
                        )
                    )
                    continue

                pairs = coupling.tetrahedral_product_pairs(planar_reactant)
                if potential_tetrahedral is not None:
                    pairs = tuple(
                        tuple(
                            descriptor
                            for descriptor in pair
                            if descriptor.center in potential_tetrahedral
                        )
                        for pair in pairs
                    )
                if not any(pairs):
                    registry = dict(product_registry)
                    changes = dict(relabeled_changes)
                    registry.pop(coupling.target, None)
                    changes[coupling.target] = StereoChange(
                        "BROKEN", planar_reactant, None
                    )
                    next_states.append(
                        (
                            registry,
                            changes,
                            dict(branch_metadata),
                            dict(outcome_metadata),
                            dict(coupling_branch_metadata),
                        )
                    )
                    continue

                for face_index, pair in enumerate(pairs):
                    registry = dict(product_registry)
                    changes = dict(relabeled_changes)
                    coupling_branches = dict(coupling_branch_metadata)
                    registry.pop(coupling.target, None)
                    changes[coupling.target] = StereoChange(
                        "BROKEN", planar_reactant, None
                    )
                    for descriptor in pair:
                        target = descriptor_id(descriptor)
                        registry[target] = descriptor
                        changes[target] = StereoChange("FORMED", None, descriptor)
                    coupling_branches[coupling.target] = {
                        "kind": coupling.kind,
                        "relation": coupling.relation,
                        "face_branch": face_index,
                    }
                    next_states.append(
                        (
                            registry,
                            changes,
                            dict(branch_metadata),
                            dict(outcome_metadata),
                            coupling_branches,
                        )
                    )
            states = next_states
        return states

    def _apply_stereo_rule_metadata(
        self,
        its: nx.Graph,
        host: nx.Graph,
        mapping: MappingDict,
    ) -> List[nx.Graph]:
        """Relabel rule stereo effects and expand declared product branches."""
        if not self.rule.stereo_effects and not self.rule.stereo_couplings:
            reactant_registry = dict(host.graph.get("stereo_descriptors", {}))
            its.graph["stereo_descriptors"] = {
                "reactant": dict(reactant_registry),
                "product": dict(reactant_registry),
            }
            its.graph["stereo_changes"] = {}
            its.graph["stereo_outcomes"] = {}
            its.graph["stereo_couplings"] = {}
            its.graph["stereo_coupling_branch"] = {}
            its.graph["stereo_branch"] = {}
            its.graph["stereo_branch_weight"] = 1.0
            return [its]

        from synkit.Graph.Stereo import descriptor_id
        from synkit.Graph.Stereo.changes import StereoChange

        translation: Dict[int, int] = {}
        pattern = self.rule.left.raw
        for pattern_node, host_node in mapping.items():
            pattern_map = pattern.nodes[pattern_node].get("atom_map", pattern_node)
            if isinstance(pattern_map, tuple) and len(pattern_map) == 2:
                pattern_map = pattern_map[0]
            host_map = host.nodes[host_node].get("atom_map") or host_node
            if isinstance(pattern_map, int) and isinstance(host_map, int):
                translation[pattern_map] = host_map

        reactant_registry = dict(host.graph.get("stereo_descriptors", {}))
        transition_registry = {
            descriptor_id(
                change.transition.relabel(translation)
            ): change.transition.relabel(translation)
            for change in self.rule.stereo_effects.values()
            if change.transition is not None
        }
        relabeled_coupling_values = []
        relabeled_couplings = {}
        for coupling in self.rule.stereo_couplings.values():
            relabeled = coupling.relabel(translation)
            relabeled_coupling_values.append(relabeled)
            relabeled_couplings[relabeled.target] = relabeled.to_dict()
        states: List[
            Tuple[
                Dict[str, Any],
                Dict[str, StereoChange],
                Dict[str, Dict[str, Any]],
                Dict[str, Dict[str, Any]],
                Dict[str, Dict[str, Any]],
            ]
        ] = [(dict(reactant_registry), {}, {}, {}, {})]

        for rule_key, change in self.rule.stereo_effects.items():
            applied_effect = change.relabel(translation)
            before = applied_effect.before
            after = applied_effect.after
            transition = applied_effect.transition
            before, after = self._propagated_stereo_effect(
                applied_effect,
                reactant_registry,
                host,
            )
            outcome = self.rule.stereo_outcomes.get(rule_key)
            alternatives = (
                outcome.alternatives(after)
                if outcome is not None and after is not None
                else (after,)
            )
            branch_weights = (
                tuple(outcome.weights or ()) if outcome is not None else (1.0,)
            )
            next_states = []
            for (
                product_registry,
                relabeled_changes,
                branch_metadata,
                outcome_metadata,
                coupling_branch_metadata,
            ) in states:
                for branch_index, (alternative, weight) in enumerate(
                    zip(alternatives, branch_weights)
                ):
                    branch_registry = dict(product_registry)
                    branch_changes = dict(relabeled_changes)
                    branch_info = dict(branch_metadata)
                    branch_outcomes = dict(outcome_metadata)
                    if before is not None:
                        branch_registry.pop(descriptor_id(before), None)
                    if alternative is not None:
                        branch_registry[descriptor_id(alternative)] = alternative
                    key_descriptor = alternative or before or transition
                    if key_descriptor is not None:
                        target = descriptor_id(key_descriptor)
                        reference_mapping = (
                            dict(applied_effect.reference_mapping)
                            if applied_effect.alignment.status == "explicit"
                            else None
                        )
                        branch_changes[target] = StereoChange.from_endpoints(
                            before,
                            alternative,
                            transition,
                            reference_mapping=reference_mapping,
                        )
                        if outcome is not None:
                            branch_outcomes[target] = outcome.to_dict()
                            branch_info[target] = {
                                "kind": outcome.kind,
                                "branch_index": branch_index,
                                "weight": weight,
                            }
                    next_states.append(
                        (
                            branch_registry,
                            branch_changes,
                            branch_info,
                            branch_outcomes,
                            dict(coupling_branch_metadata),
                        )
                    )
            states = next_states

        if relabeled_coupling_values:
            product_graph = ITSReverter(its).to_product_graph()
            states = self._apply_stereo_couplings(
                states,
                relabeled_coupling_values,
                reactant_registry,
                product_graph,
            )

        results = []
        for state_index, (
            product_registry,
            relabeled_changes,
            branch_metadata,
            outcome_metadata,
            coupling_branch_metadata,
        ) in enumerate(states):
            # The candidate is no longer needed after metadata application.
            # Reuse it for the final branch and copy only genuine alternatives.
            branch = its if state_index == len(states) - 1 else deepcopy(its)
            branch.graph["stereo_descriptors"] = {
                "reactant": dict(reactant_registry),
                "product": product_registry,
            }
            if transition_registry:
                branch.graph["stereo_descriptors"]["transition"] = dict(
                    transition_registry
                )
            branch.graph["stereo_changes"] = relabeled_changes
            branch.graph["stereo_outcomes"] = outcome_metadata
            branch.graph["stereo_couplings"] = relabeled_couplings
            branch.graph["stereo_coupling_branch"] = coupling_branch_metadata
            branch.graph["stereo_branch"] = branch_metadata
            total_weight = 1.0
            for metadata in branch_metadata.values():
                total_weight *= float(metadata["weight"])
            branch.graph["stereo_branch_weight"] = total_weight
            results.append(branch)
        return results

    @staticmethod
    def _validate_product_stereo_registry(its: nx.Graph) -> bool:
        """Validate product descriptors after the requested H representation."""
        from synkit.Graph.Stereo import descriptor_graph_support_errors

        layers = its.graph.get("stereo_descriptors", {})
        product_registry = dict(layers.get("product", {}))
        if not product_registry:
            return True

        changes = its.graph.get("stereo_changes", {})
        product = ITSReverter(its).to_product_graph()
        invalid = {
            key: descriptor_graph_support_errors(
                product,
                descriptor,
                registry_key=key,
            )
            for key, descriptor in product_registry.items()
        }
        invalid = {key: errors for key, errors in invalid.items() if errors}
        if any(key in changes for key in invalid):
            return False
        if invalid:
            layers = dict(layers)
            layers["product"] = {
                key: descriptor
                for key, descriptor in product_registry.items()
                if key not in invalid
            }
            its.graph["stereo_descriptors"] = layers
        return True
