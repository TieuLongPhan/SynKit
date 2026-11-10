"""Conservative extraction of reaction centers with persistent wildcard ports.

Only unchanged context with one single-bond attachment is generalized. Shared
or multiply bonded context is retained, so extraction never splits a ring or
duplicates a shared fragment. Domains describe the observed attachment root;
they do not identify the rest of its substituent or an omitted reagent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import networkx as nx
from rdkit import Chem

from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.rc_extractor import RCExtractor
from synkit.Graph.Morphism import LewisLabelledGraph, WildcardConstraint, WildcardRole
from synkit.IO import rsmi_to_its

from .syn_rule import SynRule


class WildcardRuleExtractionError(ValueError):
    """A source reaction is outside the declared extraction profile."""

    def __init__(self, code: str, message: str, **context: Any):
        self.code = code
        self.context = context
        super().__init__(message)


@dataclass(frozen=True)
class ExtractedWildcardPort:
    """One persistent root and the source context abstracted behind it."""

    atom: int
    owner: int
    context_atoms: tuple[int, ...]
    constraint: WildcardConstraint

    def to_dict(self):
        constraint = {
            item.name: getattr(self.constraint, item.name)
            for item in fields(self.constraint)
            if getattr(self.constraint, item.name) is not None
        }
        for key, value in constraint.items():
            if isinstance(value, frozenset):
                constraint[key] = sorted(value)
            elif hasattr(value, "value"):
                constraint[key] = value.value
        return {
            "atom": self.atom,
            "owner": self.owner,
            "context_atoms": list(self.context_atoms),
            "constraint": constraint,
        }


@dataclass(frozen=True)
class WildcardRuleExtraction:
    """An extracted rule and its context partition, without a replay claim."""

    rule: SynRule
    core_atoms: tuple[int, ...]
    ports: tuple[ExtractedWildcardPort, ...]
    retained_context_atoms: tuple[int, ...]
    spectator_atoms: tuple[int, ...]
    endpoint_policy: str = "closed"

    def to_dict(self):
        return {
            "schema": "synkit.wildcard-rule-extraction/1",
            "endpoint_policy": self.endpoint_policy,
            "domain_source": "observed_root",
            "core_atoms": list(self.core_atoms),
            "ports": [port.to_dict() for port in self.ports],
            "retained_context_atoms": list(self.retained_context_atoms),
            "spectator_atoms": list(self.spectator_atoms),
        }


def _side(value, index):
    return value[index] if isinstance(value, (tuple, list)) and len(value) == 2 else value


def _mapped_graph(graph, *, observation=False):
    schema = None
    if isinstance(graph, LewisLabelledGraph):
        if graph.partial:
            raise WildcardRuleExtractionError("PARTIAL_SOURCE", "Extraction requires complete source endpoint states.")
        schema, graph = graph.schema, graph.to_networkx()
    if not isinstance(graph, nx.Graph):
        raise TypeError("Expected a LewisLabelledGraph or NetworkX endpoint graph.")
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise WildcardRuleExtractionError(
            "INVALID_GRAPH_SHAPE", "Extraction requires simple undirected graphs without loops."
        )
    prepared = graph.copy()
    display_orders = {}
    # IO uses 1.5 as aromatic display order and stores the integral Kekule
    # occupancies separately. Validate the latter as Lewis coordinates, then
    # retain the display convention for the existing ITS matcher/serializer.
    for u, v, attrs in prepared.edges(data=True):
        if (
            attrs.get("order") == 1.5
            and all(prepared.nodes[n].get("aromatic", False) for n in (u, v))
            and attrs.get("kekule_order") == attrs.get("sigma_order", 0) + attrs.get("pi_order", 0)
        ):
            display_orders[frozenset((u, v))] = attrs.pop("order")
    if observation:
        # Patent observations may omit electronic context (e.g. metal bonds).
        # Preserve the supplied labels without asserting a complete Lewis state.
        value = prepared
    else:
        checked = LewisLabelledGraph.from_electron_graph(prepared, schema=schema)
        value = checked.to_networkx()
    for edge, order in display_orders.items():
        u, v = tuple(edge)
        value.edges[u, v]["order"] = order
    mapping = {node: attrs.get("atom_map", node) for node, attrs in value.nodes(data=True)}
    if any(not isinstance(m, int) or isinstance(m, bool) or m <= 0 for m in mapping.values()) or len(
        set(mapping.values())
    ) != len(mapping):
        raise WildcardRuleExtractionError("INVALID_ATOM_MAPS", "Endpoint atoms require unique positive atom maps.")
    if any(attrs.get("isotope", 0) for _, attrs in value.nodes(data=True)):
        raise WildcardRuleExtractionError(
            "UNSUPPORTED_ISOTOPE", "Isotope-aware wildcard extraction is not in this profile."
        )
    return nx.relabel_nodes(value, mapping, copy=True)


class WildcardRuleExtractor:
    """Extract an atom-preserving, non-stereo rule with observed-root domains.

    Absolute hydrogen populations are retained. The same wildcard identity,
    owner and predicate persist across both endpoints, allowing inversion.
    Atom-map annotations align extraction only; bindings do not require host
    map equality. Unbalanced observations require a separate observation model.
    """

    def __init__(self, *, endpoint_policy="closed"):
        if endpoint_policy not in {"closed", "observed"}:
            raise ValueError("endpoint_policy must be 'closed' or 'observed'.")
        self.endpoint_policy = endpoint_policy

    def extract_from_lewis_graphs(self, left, right) -> WildcardRuleExtraction:
        """Extract from two complete Lewis graphs aligned by atom maps."""
        observed = self.endpoint_policy == "observed"
        left, right = (_mapped_graph(g, observation=observed) for g in (left, right))
        if set(left) != set(right) and not observed:
            raise WildcardRuleExtractionError(
                "UNBALANCED_ATOM_CARRIER",
                "Wildcard extraction requires the same observed atoms on both sides.",
                left_only=sorted(set(left) - set(right)),
                right_only=sorted(set(right) - set(left)),
            )
        its = ITSConstruction.construct(left, right)
        if observed:
            from synkit.IO.chem_converter import _local_charge_model_consistent

            for node, attrs in its.nodes(data=True):
                attrs["charge_model_consistent"] = (
                    node in left and _local_charge_model_consistent(left, node),
                    node in right and _local_charge_model_consistent(right, node),
                )
        return self._extract(its)

    def extract(self, reaction: str | nx.Graph) -> WildcardRuleExtraction:
        """Extract from a fully mapped reaction SMILES or absolute tuple ITS."""
        if isinstance(reaction, str):
            parts = reaction.split(">>")
            if len(parts) != 2:
                raise WildcardRuleExtractionError("INVALID_REACTION", "Expected reactants>>products.")
            map_sets, normalized = [], []
            for part in parts:
                parameters = Chem.SmilesParserParams()
                parameters.removeHs = self.endpoint_policy == "observed"
                molecule = Chem.MolFromSmiles(part, parameters)
                if molecule is None:
                    raise WildcardRuleExtractionError("INVALID_REACTION", "An endpoint cannot be parsed.")
                maps = [a.GetAtomMapNum() for a in molecule.GetAtoms()]
                if not maps or any(m <= 0 for m in maps) or len(set(maps)) != len(maps):
                    raise WildcardRuleExtractionError(
                        "INVALID_ATOM_MAPS", "Every endpoint atom needs a unique positive atom map."
                    )
                if any(a.GetIsotope() for a in molecule.GetAtoms()):
                    raise WildcardRuleExtractionError(
                        "UNSUPPORTED_ISOTOPE", "Isotope-aware extraction is not in this profile."
                    )
                if self.endpoint_policy == "closed" and any(
                    a.GetSymbol() == "H" and a.GetDegree() for a in molecule.GetAtoms()
                ):
                    raise WildcardRuleExtractionError(
                        "UNSUPPORTED_EXPLICIT_HYDROGEN",
                        "Attached explicit hydrogen carriers are not in this extraction profile.",
                    )
                map_sets.append(set(maps))
                if self.endpoint_policy == "observed":
                    Chem.RemoveStereochemistry(molecule)
                normalized.append(Chem.MolToSmiles(molecule))
            if map_sets[0] != map_sets[1] and self.endpoint_policy == "closed":
                raise WildcardRuleExtractionError(
                    "UNBALANCED_ATOM_CARRIER",
                    "Wildcard extraction requires the same observed atoms on both sides.",
                    left_only=sorted(map_sets[0] - map_sets[1]),
                    right_only=sorted(map_sets[1] - map_sets[0]),
                )
            reaction = rsmi_to_its(
                ">>".join(normalized), format="tuple", drop_non_aam=False, use_index_as_atom_map=True
            )
        if not isinstance(reaction, nx.Graph):
            raise TypeError("Expected mapped reaction SMILES or tuple ITS.")
        if reaction.is_directed() or reaction.is_multigraph() or nx.number_of_selfloops(reaction):
            raise WildcardRuleExtractionError(
                "INVALID_GRAPH_SHAPE", "Extraction requires simple undirected graphs without loops."
            )
        if any(reaction.graph.get(key) for key in ("stereo_changes", "stereo_outcomes", "stereo_couplings")) or any(
            reaction.graph.get("stereo_descriptors", {}).values()
        ):
            raise WildcardRuleExtractionError(
                "UNSUPPORTED_STEREO", "Use the stereo extractor for stereo-bearing reactions."
            )
        reverter = ITSReverter(deepcopy(reaction))
        return self.extract_from_lewis_graphs(reverter.to_reactant_graph(), reverter.to_product_graph())

    def _reaction_center(self, its: nx.Graph) -> set[Any]:
        core = set(RCExtractor().extract(its))
        # Total-order differences alone cannot detect a sigma/pi redistribution.
        for node, attrs in its.nodes(data=True):
            persistent = all(_side(attrs.get("present", True), side) for side in (0, 1))
            if not persistent:
                core.add(node)
                if self.endpoint_policy == "closed":
                    raise WildcardRuleExtractionError(
                        "UNBALANCED_ATOM_CARRIER", "Closed extraction requires persistent atoms."
                    )
                continue
            if any(
                _side(attrs.get(key), 0) != _side(attrs.get(key), 1)
                for key in ("element", "aromatic", "charge", "hcount", "radical", "lone_pairs", "valence_electrons")
            ):
                core.add(node)
            if _side(attrs["element"], 0) != _side(attrs["element"], 1):
                raise WildcardRuleExtractionError("CHANGED_ELEMENT", "Material elements must remain fixed.")
            if _side(attrs["valence_electrons"], 0) != _side(attrs["valence_electrons"], 1):
                raise WildcardRuleExtractionError(
                    "CHANGED_ATOMIC_VALENCE", "Atomic valence-electron labels must remain fixed."
                )
        for u, v, attrs in its.edges(data=True):
            if any(
                _side(attrs.get(key, 0), 0) != _side(attrs.get(key, 0), 1)
                for key in ("order", "sigma_order", "pi_order", "kekule_order")
            ):
                core.update((u, v))
        if not core and self.endpoint_policy == "closed":
            raise WildcardRuleExtractionError("EMPTY_CENTER", "The source has no recorded reaction-center edit.")
        return core

    def _extract(self, its: nx.Graph) -> WildcardRuleExtraction:
        core = self._reaction_center(its)
        retained, spectators, ports = set(), set(), []
        selected = set(core)
        for component in nx.connected_components(its.subgraph(set(its) - core)):
            attachments = [(owner, root) for root in component for owner in its.neighbors(root) if owner in core]
            if not attachments:
                spectators.update(component)
                continue
            single = len(attachments) == 1
            owner, root = attachments[0]
            bond = its.edges[owner, root]
            single = single and all(
                _side(bond.get(key), side) == expected
                for key, expected in (("order", 1), ("sigma_order", 1), ("pi_order", 0))
                for side in (0, 1)
            )
            single = single and _side(its.nodes[root]["element"], 0) != "H"
            if not single:
                retained.update(component)
                selected.update(component)
                continue
            attrs = its.nodes[root]
            constraint = WildcardConstraint(
                WildcardRole.ATTACHMENT_PORT,
                elements={_side(attrs["element"], 0)},
                charges={_side(attrs["charge"], 0)},
                radicals={_side(attrs["radical"], 0)},
                bond_orders={1},
                owner=owner,
                capacity=1,
                resource_budget=1,
            )
            ports.append(ExtractedWildcardPort(root, owner, tuple(sorted(component)), constraint))
            selected.add(root)
        extracted = its.subgraph(selected).copy()
        for port in ports:
            # A query root asserts a predicate, not a material VE/LP population.
            attrs = extracted.nodes[port.atom]
            atom_map = attrs.get("atom_map", (port.atom, port.atom))
            attrs.clear()
            attrs.update(
                atom_map=atom_map,
                element=("*", "*"),
                aromatic=(False, False),
                present=(True, True),
                charge=(0, 0),
                hcount=(0, 0),
                radical=(0, 0),
                lone_pairs=(0, 0),
                valence_electrons=(0, 0),
                typesGH=(("*", False, 0, 0, []), ("*", False, 0, 0, [])),
                wildcard_role=port.constraint.role.value,
            )
            attrs.update(
                {
                    item.name: getattr(port.constraint, item.name)
                    for item in fields(port.constraint)
                    if item.name != "role" and getattr(port.constraint, item.name) is not None
                }
            )
        extracted.graph["endpoint_policy"] = self.endpoint_policy
        rule = SynRule(extracted, format="tuple", implicit_h=False, canon=False)
        from synkit.Synthesis.Reactor.matching.wildcard import validate_wildcard_rule

        validate_wildcard_rule(rule.left.raw, rule.right.raw, observation=self.endpoint_policy == "observed")
        result = WildcardRuleExtraction(
            rule,
            tuple(sorted(core)),
            tuple(sorted(ports, key=lambda p: p.atom)),
            tuple(sorted(retained)),
            tuple(sorted(spectators)),
            self.endpoint_policy,
        )
        rule.rc.raw.graph["wildcard_extraction"] = result.to_dict()
        return result


__all__ = ["ExtractedWildcardPort", "WildcardRuleExtraction", "WildcardRuleExtractionError", "WildcardRuleExtractor"]
