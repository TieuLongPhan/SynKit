"""Proof-bearing extraction of conservative generic stereo rules.

The extractor in this module is additive: ordinary :class:`SynRule`
construction remains exact.  Generalization replaces only graph-backed,
unchanged peripheral stereo references under an explicit policy and records
enough evidence to replay the source reaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import networkx as nx

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Morphism import WildcardConstraint, WildcardRole
from synkit.Graph.syn_graph import SynGraph
from synkit.IO.chem_converter import rsmi_to_graph, rsmi_to_its

from .syn_rule import NonInvertibleStereoEffectError, SynRule

EXTRACTION_SCHEMA = "synkit.stereo-rule-extraction/1"


class GenericStereoDomainSource(str, Enum):
    """Authority used to define one extracted wildcard domain."""

    EXACT = "exact"
    OBSERVED = "observed"
    CLASS = "class"
    CORPUS = "corpus"


class GenericStereoExtractionIssueCode(str, Enum):
    """Stable refusal codes for generic stereo extraction."""

    INPUT_NOT_MAPPED = "GENERIC_STEREO_INPUT_NOT_MAPPED"
    NO_STEREO = "GENERIC_STEREO_NO_STEREO"
    REFERENCE_NOT_FOUND = "GENERIC_STEREO_REFERENCE_NOT_FOUND"
    REFERENCE_NOT_ELIGIBLE = "GENERIC_STEREO_REFERENCE_NOT_ELIGIBLE"
    AMBIGUOUS_BINDING = "GENERIC_STEREO_AMBIGUOUS_BINDING"
    DOMAIN_REQUIRED = "GENERIC_STEREO_DOMAIN_REQUIRED"
    SOURCE_REPLAY_FAILED = "GENERIC_STEREO_SOURCE_REPLAY_FAILED"
    REVERSE_REPLAY_FAILED = "GENERIC_STEREO_REVERSE_REPLAY_FAILED"


@dataclass(frozen=True)
class GenericStereoExtractionIssue:
    """One structured extraction refusal."""

    code: GenericStereoExtractionIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible issue record."""
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class GenericStereoExtractionError(ValueError):
    """Raised when conservative generic extraction cannot be certified."""

    def __init__(self, *issues: GenericStereoExtractionIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


def _freeze_domain_mapping(
    values: Mapping[int, Mapping[str, Any]],
) -> Mapping[int, Mapping[str, Any]]:
    frozen = {}
    for reference, domains in values.items():
        normalized = {}
        for name, value in domains.items():
            if name in {"elements", "charges", "radicals", "bond_orders"}:
                if isinstance(value, (set, frozenset, tuple, list)):
                    value = frozenset(value)
                else:
                    value = frozenset({value})
            normalized[str(name)] = value
        frozen[int(reference)] = MappingProxyType(normalized)
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class GenericStereoRulePolicy:
    """Explicit policy controlling which spectator references generalize."""

    domain_source: GenericStereoDomainSource | str = GenericStereoDomainSource.OBSERVED
    selected_references: frozenset[int] | Iterable[int] | None = None
    explicit_domains: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    domain_evidence: Mapping[int, Iterable[str]] = field(default_factory=dict)
    verify_source: bool = True
    verify_reverse: bool = True

    def __post_init__(self) -> None:
        source = GenericStereoDomainSource(self.domain_source)
        selected = self.selected_references
        if selected is not None and not isinstance(selected, frozenset):
            selected = frozenset(int(value) for value in selected)
        object.__setattr__(self, "domain_source", source)
        object.__setattr__(self, "selected_references", selected)
        object.__setattr__(
            self,
            "explicit_domains",
            _freeze_domain_mapping(self.explicit_domains),
        )
        evidence = {}
        for reference, records in self.domain_evidence.items():
            values = (records,) if isinstance(records, str) else records
            evidence[int(reference)] = tuple(sorted(map(str, values)))
        object.__setattr__(self, "domain_evidence", MappingProxyType(evidence))
        if (
            source
            in {
                GenericStereoDomainSource.EXACT,
                GenericStereoDomainSource.OBSERVED,
            }
            and self.explicit_domains
        ):
            raise ValueError(
                "Explicit domains require domain_source='class' or 'corpus'."
            )


@dataclass(frozen=True)
class ExtractedStereoPort:
    """One generalized peripheral ligand and its authoritative binding."""

    reference: int
    target: str
    owner: int
    stereo_slot: int
    constraint: WildcardConstraint
    domain_source: GenericStereoDomainSource
    domain_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible port record."""
        normalized = self.constraint.normalized()
        return {
            "reference": self.reference,
            "target": self.target,
            "owner": self.owner,
            "stereo_slot": self.stereo_slot,
            "domain_source": self.domain_source.value,
            "domain_evidence": list(self.domain_evidence),
            "constraint": {
                "role": normalized[0],
                "elements": normalized[1],
                "charges": normalized[2],
                "radicals": normalized[3],
                "bond_orders": normalized[4],
                "side": normalized[5],
                "owner": self.owner,
                "capacity": normalized[7],
                "resource_budget": normalized[8],
                "stereo_slot": normalized[9],
                "virtual_kind": normalized[10],
                "mapped_identity": normalized[11],
                "materialization": normalized[12],
            },
        }


@dataclass(frozen=True)
class RuleExtractionCertificate:
    """Immutable source-to-generic extraction and replay evidence."""

    source_rule_digest: str
    generic_rule_digest: str
    ports: tuple[ExtractedStereoPort, ...]
    source_mapping_count: int
    source_unique_products: int
    source_replay_exact: bool
    reverse_status: str
    reverse_mapping_count: int = 0
    schema: str = EXTRACTION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Return stable certificate payload."""
        return {
            "schema": self.schema,
            "source_rule_digest": self.source_rule_digest,
            "generic_rule_digest": self.generic_rule_digest,
            "ports": [port.to_dict() for port in self.ports],
            "source_replay": {
                "mapping_count": self.source_mapping_count,
                "unique_products": self.source_unique_products,
                "exact": self.source_replay_exact,
            },
            "reverse_replay": {
                "status": self.reverse_status,
                "mapping_count": self.reverse_mapping_count,
            },
        }

    @property
    def digest(self) -> str:
        """Return a deterministic atom-map-invariant certificate digest."""
        port_contracts = []
        for port in self.ports:
            normalized = list(port.constraint.normalized())
            normalized[6] = None
            port_contracts.append(
                (
                    port.stereo_slot,
                    port.domain_source.value,
                    port.domain_evidence,
                    tuple(normalized),
                )
            )
        payload = json.dumps(
            {
                "schema": self.schema,
                "source_rule_digest": self.source_rule_digest,
                "generic_rule_digest": self.generic_rule_digest,
                "ports": sorted(port_contracts, key=repr),
                "source_replay": (
                    self.source_mapping_count,
                    self.source_unique_products,
                    self.source_replay_exact,
                ),
                "reverse_replay": (self.reverse_status, self.reverse_mapping_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GenericStereoRuleResult:
    """Extracted rule paired with its replay certificate."""

    rule: SynRule
    certificate: RuleExtractionCertificate


@dataclass(frozen=True)
class _PortBinding:
    reference: int
    target: str
    owner: int
    slot: int


_ATOM_CENTERED = {
    "tetrahedral",
    "square_planar",
    "trigonal_bipyramidal",
    "octahedral",
}


def _descriptor_bindings(target: str, descriptor: Any) -> tuple[_PortBinding, ...]:
    # ``stereo_slot`` is defined on the authoritative orbit configuration,
    # not on the parser's raw atom order.  For example, tetrahedral parity can
    # exchange two peripheral positions while preserving the same descriptor.
    frame = descriptor.configuration.frame
    bindings = []
    if descriptor.descriptor_class in _ATOM_CENTERED:
        owner = frame[0]
        if not isinstance(owner, int):
            return ()
        for slot, reference in enumerate(frame[1:]):
            if isinstance(reference, int):
                bindings.append(_PortBinding(reference, target, owner, slot))
        return tuple(bindings)
    left_owner, right_owner = frame[2:4]
    if isinstance(left_owner, int):
        for slot, reference in enumerate(frame[:2]):
            if isinstance(reference, int):
                bindings.append(_PortBinding(reference, target, left_owner, slot))
    if isinstance(right_owner, int):
        for slot, reference in enumerate(frame[4:]):
            if isinstance(reference, int):
                bindings.append(_PortBinding(reference, target, right_owner, slot))
    return tuple(bindings)


def _side_value(value: Any, side: int) -> Any:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[side]
    return value


def _node_by_atom_map(graph: nx.Graph) -> dict[int, Any]:
    result = {}
    for node, attrs in graph.nodes(data=True):
        atom_map = attrs.get("atom_map")
        values = atom_map if isinstance(atom_map, (tuple, list)) else (atom_map,)
        for value in values:
            if not isinstance(value, int) or value <= 0:
                continue
            if value in result and result[value] != node:
                raise GenericStereoExtractionError(
                    GenericStereoExtractionIssue(
                        GenericStereoExtractionIssueCode.INPUT_NOT_MAPPED,
                        "A generic stereo source requires unique positive atom maps.",
                        {"atom_map": value},
                    )
                )
            result[value] = node
    return result


def _validate_mapped_reaction(reaction: str) -> None:
    reactant, product = rsmi_to_graph(
        reaction,
        drop_non_aam=False,
        use_index_as_atom_map=False,
    )
    layers = {"reactant": reactant, "product": product}
    layer_maps = {}
    for layer, graph in layers.items():
        if graph is None:
            raise GenericStereoExtractionError(
                GenericStereoExtractionIssue(
                    GenericStereoExtractionIssueCode.INPUT_NOT_MAPPED,
                    "Generic stereo extraction requires a valid mapped reaction.",
                    {"layer": layer},
                )
            )
        values = [attrs.get("atom_map") for _, attrs in graph.nodes(data=True)]
        if any(type(value) is not int or value <= 0 for value in values) or (
            len(set(values)) != len(values)
        ):
            raise GenericStereoExtractionError(
                GenericStereoExtractionIssue(
                    GenericStereoExtractionIssueCode.INPUT_NOT_MAPPED,
                    "Every source atom requires one unique positive atom map.",
                    {"layer": layer},
                )
            )
        layer_maps[layer] = set(values)
    if layer_maps["reactant"] != layer_maps["product"]:
        raise GenericStereoExtractionError(
            GenericStereoExtractionIssue(
                GenericStereoExtractionIssueCode.INPUT_NOT_MAPPED,
                "Reactant and product atom-map domains must be identical.",
            )
        )


def _rule_digest(rule: SynRule) -> str:
    payload = (
        rule.canonical_smiles,
        rule._stereo_signature(),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _unique_graphs(graphs: Iterable[nx.Graph]) -> list[SynGraph]:
    unique: list[SynGraph] = []
    for graph in graphs:
        wrapped = SynGraph(graph)
        if any(wrapped == existing for existing in unique):
            continue
        unique.append(wrapped)
    return unique


class GenericStereoRuleExtractor:
    """Extract conservative typed-wildcard stereo rules from mapped reactions."""

    def __init__(self, policy: GenericStereoRulePolicy | None = None):
        self.policy = policy or GenericStereoRulePolicy()

    @staticmethod
    def _eligible_bindings(
        its: nx.Graph,
        by_map: Mapping[int, Any],
    ) -> tuple[dict[int, _PortBinding], frozenset[int]]:
        stereo = its.graph.get("stereo_descriptors", {})
        reactant = stereo.get("reactant", {})
        product = stereo.get("product", {})
        changes = its.graph.get("stereo_changes", {})
        candidates: dict[int, list[_PortBinding]] = {}
        for target, descriptor in reactant.items():
            after = product.get(target)
            if after is None or descriptor.descriptor_class != after.descriptor_class:
                continue
            after_references = set(after.atoms)
            alignment = changes.get(target)
            replaced = (
                set(alignment.alignment.removed) | set(alignment.alignment.added)
                if alignment is not None
                else set()
            )
            for binding in _descriptor_bindings(target, descriptor):
                if binding.reference not in after_references or (
                    binding.reference in replaced
                ):
                    continue
                node = by_map.get(binding.reference)
                owner_node = by_map.get(binding.owner)
                if (
                    node is None
                    or owner_node is None
                    or not its.has_edge(node, owner_node)
                ):
                    continue
                attrs = its.nodes[node]
                stable_node = all(
                    _side_value(attrs.get(name), 0) == _side_value(attrs.get(name), 1)
                    for name in ("element", "charge", "radical", "present")
                )
                order = its.edges[node, owner_node].get("order")
                stable_bond = _side_value(order, 0) == _side_value(order, 1)
                if stable_node and stable_bond:
                    candidates.setdefault(binding.reference, []).append(binding)
        result = {}
        ambiguous = set()
        for reference, bindings in candidates.items():
            distinct = set(bindings)
            if len(distinct) == 1:
                result[reference] = next(iter(distinct))
            else:
                ambiguous.add(reference)
        return result, frozenset(ambiguous)

    def _selected_bindings(
        self,
        its: nx.Graph,
        by_map: Mapping[int, Any],
    ) -> tuple[_PortBinding, ...]:
        eligible, ambiguous = self._eligible_bindings(its, by_map)
        selected = self.policy.selected_references
        if selected is None:
            if ambiguous:
                raise GenericStereoExtractionError(
                    GenericStereoExtractionIssue(
                        GenericStereoExtractionIssueCode.AMBIGUOUS_BINDING,
                        "A peripheral reference binds multiple stereo frames.",
                        {"references": tuple(sorted(ambiguous))},
                    )
                )
            references = set(eligible)
        else:
            references = set(selected)
        issues = []
        for reference in sorted(references):
            if reference not in by_map:
                issues.append(
                    GenericStereoExtractionIssue(
                        GenericStereoExtractionIssueCode.REFERENCE_NOT_FOUND,
                        "A selected stereo reference is absent from the source rule.",
                        {"reference": reference},
                    )
                )
            elif reference in ambiguous:
                issues.append(
                    GenericStereoExtractionIssue(
                        GenericStereoExtractionIssueCode.AMBIGUOUS_BINDING,
                        "A selected reference binds multiple stereo frames.",
                        {"reference": reference},
                    )
                )
            elif reference not in eligible:
                issues.append(
                    GenericStereoExtractionIssue(
                        GenericStereoExtractionIssueCode.REFERENCE_NOT_ELIGIBLE,
                        "A selected reference is not one unchanged peripheral ligand.",
                        {"reference": reference},
                    )
                )
        if issues:
            raise GenericStereoExtractionError(*issues)
        return tuple(eligible[reference] for reference in sorted(references))

    def _constraint(
        self,
        its: nx.Graph,
        by_map: Mapping[int, Any],
        binding: _PortBinding,
    ) -> WildcardConstraint:
        node = by_map[binding.reference]
        owner = by_map[binding.owner]
        attrs = its.nodes[node]
        observed = {
            "elements": {_side_value(attrs.get("element"), 0)},
            "charges": {int(_side_value(attrs.get("charge", 0), 0))},
            "radicals": {int(_side_value(attrs.get("radical", 0), 0))},
            "bond_orders": {float(_side_value(its.edges[node, owner].get("order"), 0))},
            "materialization": "concrete",
        }
        source = self.policy.domain_source
        supplied = self.policy.explicit_domains.get(binding.reference)
        if (
            source
            in {
                GenericStereoDomainSource.CLASS,
                GenericStereoDomainSource.CORPUS,
            }
            and supplied is None
        ):
            raise GenericStereoExtractionError(
                GenericStereoExtractionIssue(
                    GenericStereoExtractionIssueCode.DOMAIN_REQUIRED,
                    "Class/corpus generalization requires an explicit domain.",
                    {"reference": binding.reference, "source": source.value},
                )
            )
        if source is GenericStereoDomainSource.CORPUS and not (
            self.policy.domain_evidence.get(binding.reference)
        ):
            raise GenericStereoExtractionError(
                GenericStereoExtractionIssue(
                    GenericStereoExtractionIssueCode.DOMAIN_REQUIRED,
                    "Corpus generalization requires aligned record evidence.",
                    {"reference": binding.reference, "source": source.value},
                )
            )
        values = observed if supplied is None else {**observed, **dict(supplied)}
        return WildcardConstraint(
            WildcardRole.STEREO_LIGAND_PORT,
            elements=values.get("elements"),
            charges=values.get("charges"),
            radicals=values.get("radicals"),
            bond_orders=values.get("bond_orders"),
            owner=binding.owner,
            stereo_slot=binding.slot,
            virtual_kind=values.get("virtual_kind"),
            materialization=values.get("materialization", "concrete"),
            capacity=int(values.get("capacity", 1)),
            resource_budget=values.get("resource_budget"),
        )

    @staticmethod
    def _make_wildcard(attrs: dict[str, Any], constraint: WildcardConstraint) -> None:
        atom_map = attrs.get("atom_map")
        maps = atom_map if isinstance(atom_map, tuple) else (atom_map, atom_map)
        neighbors = attrs.get("neighbors", ([], []))
        attrs.update(
            element=("*", "*"),
            aromatic=(False, False),
            hcount=(0, 0),
            charge=(0, 0),
            radical=(0, 0),
            lone_pairs=(0, 0),
            valence_electrons=(0, 0),
            typesGH=(
                ("*", False, 0, 0, 0, 0, 0, neighbors[0], maps[0]),
                ("*", False, 0, 0, 0, 0, 0, neighbors[1], maps[1]),
            ),
            wildcard_role=constraint.role.value,
            owner=constraint.owner,
            stereo_slot=constraint.stereo_slot,
            elements=set(constraint.elements) if constraint.elements else None,
            charges=set(constraint.charges) if constraint.charges else None,
            radicals=set(constraint.radicals) if constraint.radicals else None,
            bond_orders=(
                set(constraint.bond_orders) if constraint.bond_orders else None
            ),
            side=constraint.side.value,
            capacity=constraint.capacity,
            resource_budget=constraint.resource_budget,
            virtual_kind=constraint.virtual_kind,
            mapped_identity=constraint.mapped_identity,
            materialization=constraint.materialization,
        )
        for key in (
            "elements",
            "charges",
            "radicals",
            "bond_orders",
            "resource_budget",
            "virtual_kind",
            "mapped_identity",
        ):
            if attrs[key] is None:
                attrs.pop(key)

    @staticmethod
    def _source_replay(
        rule: SynRule,
        concrete_its: nx.Graph,
    ) -> tuple[int, int, bool]:
        from synkit.Synthesis.Reactor.syn_reactor import SynReactor

        reverter = ITSReverter(concrete_its)
        reactant = reverter.to_reactant_graph()
        expected = SynGraph(reverter.to_product_graph())
        reactor = SynReactor(
            reactant,
            rule,
            template_format="tuple",
            explicit_h=False,
            stereo_mode="strict",
            dedup_its=False,
        )
        products = [ITSReverter(its).to_product_graph() for its in reactor.its_list]
        unique = _unique_graphs(products)
        exact = bool(products) and all(
            SynGraph(product) == expected for product in products
        )
        return reactor.mapping_count, len(unique), exact

    @staticmethod
    def _reverse_replay(
        rule: SynRule,
        concrete_its: nx.Graph,
    ) -> tuple[str, int, bool]:
        from synkit.Synthesis.Reactor.syn_reactor import SynReactor

        try:
            reverse = rule.reversed()
        except NonInvertibleStereoEffectError:
            return "non_invertible", 0, True
        reverter = ITSReverter(concrete_its)
        product = reverter.to_product_graph()
        expected = SynGraph(reverter.to_reactant_graph())
        reactor = SynReactor(
            product,
            reverse,
            template_format="tuple",
            explicit_h=False,
            stereo_mode="strict",
            dedup_its=False,
        )
        reactants = [ITSReverter(its).to_product_graph() for its in reactor.its_list]
        exact = bool(reactants) and all(
            SynGraph(candidate) == expected for candidate in reactants
        )
        return "exact" if exact else "failed", reactor.mapping_count, exact

    def extract(self, reaction: str | nx.Graph) -> GenericStereoRuleResult:
        """Return a certified generic rule for one mapped reaction or ITS."""
        if isinstance(reaction, str):
            _validate_mapped_reaction(reaction)
        concrete_its = (
            rsmi_to_its(
                reaction,
                format="tuple",
                drop_non_aam=False,
                use_index_as_atom_map=True,
            )
            if isinstance(reaction, str)
            else reaction.copy()
        )
        concrete_rule = SynRule(concrete_its, format="tuple", implicit_h=False)
        if not concrete_rule.stereo_guards and not concrete_rule.stereo_effects:
            raise GenericStereoExtractionError(
                GenericStereoExtractionIssue(
                    GenericStereoExtractionIssueCode.NO_STEREO,
                    "Generic stereo extraction requires a stereo-bearing reaction.",
                )
            )
        by_map = _node_by_atom_map(concrete_its)
        if len(by_map) != concrete_its.number_of_nodes():
            raise GenericStereoExtractionError(
                GenericStereoExtractionIssue(
                    GenericStereoExtractionIssueCode.INPUT_NOT_MAPPED,
                    "Every concrete ITS node requires one unique positive atom map.",
                )
            )
        bindings = (
            ()
            if self.policy.domain_source is GenericStereoDomainSource.EXACT
            else self._selected_bindings(concrete_its, by_map)
        )
        if (
            self.policy.domain_source is not GenericStereoDomainSource.EXACT
            and not bindings
        ):
            raise GenericStereoExtractionError(
                GenericStereoExtractionIssue(
                    GenericStereoExtractionIssueCode.REFERENCE_NOT_ELIGIBLE,
                    "No unchanged peripheral stereo reference is eligible.",
                )
            )
        generalized = concrete_its.copy()
        generalized_by_map = _node_by_atom_map(generalized)
        ports = []
        for binding in bindings:
            constraint = self._constraint(concrete_its, by_map, binding)
            self._make_wildcard(
                generalized.nodes[generalized_by_map[binding.reference]],
                constraint,
            )
            ports.append(
                ExtractedStereoPort(
                    binding.reference,
                    binding.target,
                    binding.owner,
                    binding.slot,
                    constraint,
                    self.policy.domain_source,
                    self.policy.domain_evidence.get(binding.reference, ()),
                )
            )
        generalized.graph["generic_stereo_extraction"] = {
            "schema": EXTRACTION_SCHEMA,
            "domain_source": self.policy.domain_source.value,
            "generalized_references": [port.reference for port in ports],
        }
        rule = SynRule(generalized, format="tuple", implicit_h=False)

        mapping_count = unique_count = 0
        source_exact = True
        if self.policy.verify_source:
            mapping_count, unique_count, source_exact = self._source_replay(
                rule,
                concrete_its,
            )
            if not source_exact:
                raise GenericStereoExtractionError(
                    GenericStereoExtractionIssue(
                        GenericStereoExtractionIssueCode.SOURCE_REPLAY_FAILED,
                        "The extracted generic rule did not exactly replay its source.",
                        {
                            "mapping_count": mapping_count,
                            "unique_products": unique_count,
                        },
                    )
                )

        reverse_status = "not_checked"
        reverse_count = 0
        if self.policy.verify_reverse:
            reverse_status, reverse_count, reverse_exact = self._reverse_replay(
                rule,
                concrete_its,
            )
            if not reverse_exact:
                raise GenericStereoExtractionError(
                    GenericStereoExtractionIssue(
                        GenericStereoExtractionIssueCode.REVERSE_REPLAY_FAILED,
                        "The extracted generic rule failed exact reverse replay.",
                        {"mapping_count": reverse_count},
                    )
                )

        certificate = RuleExtractionCertificate(
            source_rule_digest=_rule_digest(concrete_rule),
            generic_rule_digest=_rule_digest(rule),
            ports=tuple(ports),
            source_mapping_count=mapping_count,
            source_unique_products=unique_count,
            source_replay_exact=source_exact,
            reverse_status=reverse_status,
            reverse_mapping_count=reverse_count,
        )
        rule.rc.raw.graph["generic_stereo_extraction_certificate"] = (
            certificate.to_dict()
        )
        return GenericStereoRuleResult(rule, certificate)


__all__ = [
    "EXTRACTION_SCHEMA",
    "ExtractedStereoPort",
    "GenericStereoDomainSource",
    "GenericStereoExtractionError",
    "GenericStereoExtractionIssue",
    "GenericStereoExtractionIssueCode",
    "GenericStereoRuleExtractor",
    "GenericStereoRulePolicy",
    "GenericStereoRuleResult",
    "RuleExtractionCertificate",
]
