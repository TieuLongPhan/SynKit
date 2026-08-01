"""Finite Lewis-labelled graphs and their strict embedding category.

This module defines the generic, non-stereo graph layer used by formal rule
composition.  Objects are finite simple undirected attributed graphs.  Arrows
are total injective node maps whose induced edge maps preserve incidence and
all declared identity and state labels.  Derived labels are checked when an
object is built but are not independent matching data; annotations are never
matching data.

The category deliberately makes no blanket adhesivity claim.  Pullbacks of
strict embeddings are available as finite fibre products.  A pushout along
strict embeddings is admitted only when quotienting creates no identity/state
label conflict.  Pushout complements are partial and belong to the DPO layer,
where dangling and identification premises can be reported explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

FrozenValue = Hashable
LabelBundle = tuple[tuple[str, FrozenValue], ...]
EdgeKey = frozenset[Hashable]


class LLGIssueCode(str, Enum):
    """Stable failures for LLG objects, adapters, and morphisms."""

    SCHEMA_OVERLAP = "LLG_SCHEMA_OVERLAP"
    DIRECTED_GRAPH = "LLG_DIRECTED_GRAPH"
    MULTIGRAPH = "LLG_MULTIGRAPH"
    SELF_LOOP = "LLG_SELF_LOOP"
    DUPLICATE_NODE = "LLG_DUPLICATE_NODE"
    DUPLICATE_EDGE = "LLG_DUPLICATE_EDGE"
    UNKNOWN_ENDPOINT = "LLG_UNKNOWN_ENDPOINT"
    MISSING_NODE_LABEL = "LLG_MISSING_NODE_LABEL"
    MISSING_EDGE_LABEL = "LLG_MISSING_EDGE_LABEL"
    DERIVED_LABEL_MISMATCH = "LLG_DERIVED_LABEL_MISMATCH"
    OUTSIDE_SOURCE = "LLG_MORPHISM_OUTSIDE_SOURCE"
    OUTSIDE_TARGET = "LLG_MORPHISM_OUTSIDE_TARGET"
    PARTIAL_MAPPING = "LLG_MORPHISM_PARTIAL_MAPPING"
    NON_INJECTIVE = "LLG_MORPHISM_NON_INJECTIVE"
    MISSING_EDGE = "LLG_MORPHISM_MISSING_EDGE"
    NODE_LABEL_MISMATCH = "LLG_MORPHISM_NODE_LABEL_MISMATCH"
    EDGE_LABEL_MISMATCH = "LLG_MORPHISM_EDGE_LABEL_MISMATCH"
    ENDPOINT_MISMATCH = "LLG_MORPHISM_ENDPOINT_MISMATCH"
    INVALID_RELABELING = "LLG_INVALID_RELABELING"
    UNSUPPORTED_ITS_FORMAT = "LLG_UNSUPPORTED_ITS_FORMAT"
    LOSSY_ELECTRON_ADAPTER = "LLG_LOSSY_ELECTRON_ADAPTER"


@dataclass(frozen=True)
class LLGIssue:
    """One typed LLG refusal."""

    code: LLGIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class LLGError(ValueError):
    """Raised when an invalid object or arrow would enter the category."""

    def __init__(self, *issues: LLGIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


def _freeze(value: Any) -> FrozenValue:
    """Recursively freeze common mutable attribute values."""
    if isinstance(value, Mapping):
        return tuple(
            sorted(((str(key), _freeze(item)) for key, item in value.items()), key=repr)
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(
            f"LLG attribute values must be recursively hashable: {value!r}"
        ) from error
    return value


def _bundle(values: Mapping[str, Any]) -> LabelBundle:
    return tuple(sorted(((str(key), _freeze(value)) for key, value in values.items())))


def _bundle_dict(values: LabelBundle) -> dict[str, FrozenValue]:
    return dict(values)


def _carrier_key(value: Hashable) -> tuple[str, str, str]:
    kind = type(value)
    return (kind.__module__, kind.__qualname__, repr(value))


def _edge_sort_key(edge: EdgeKey) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((_carrier_key(node) for node in edge)))


@dataclass(frozen=True)
class LabelSchema:
    """Classify labels by mathematical role.

    Identity and state labels are preserved by every morphism.  Derived labels
    must be coherent with authoritative state but are not matched separately.
    Annotation labels, including atom maps, names, and display fields, are
    retained without affecting admissibility.
    """

    node_identity: tuple[str, ...] = ()
    node_state: tuple[str, ...] = ()
    edge_identity: tuple[str, ...] = ()
    edge_state: tuple[str, ...] = ()
    node_derived: tuple[str, ...] = ()
    edge_derived: tuple[str, ...] = ()
    node_annotations: tuple[str, ...] = ()
    edge_annotations: tuple[str, ...] = ()
    name: str = "llg"

    def __post_init__(self) -> None:
        groups = {
            "node_identity": self.node_identity,
            "node_state": self.node_state,
            "node_derived": self.node_derived,
            "node_annotations": self.node_annotations,
            "edge_identity": self.edge_identity,
            "edge_state": self.edge_state,
            "edge_derived": self.edge_derived,
            "edge_annotations": self.edge_annotations,
        }
        for key, values in groups.items():
            normalized = tuple(dict.fromkeys(str(value) for value in values))
            object.__setattr__(self, key, normalized)
        issues: list[LLGIssue] = []
        for prefix in ("node", "edge"):
            owners: dict[str, list[str]] = {}
            for key, values in groups.items():
                if not key.startswith(prefix):
                    continue
                for value in values:
                    owners.setdefault(str(value), []).append(key)
            overlap = {key: value for key, value in owners.items() if len(value) > 1}
            if overlap:
                issues.append(
                    LLGIssue(
                        LLGIssueCode.SCHEMA_OVERLAP,
                        f"{prefix.title()} labels must have exactly one class.",
                        {"labels": overlap},
                    )
                )
        if issues:
            raise LLGError(*issues)

    @property
    def semantic_node_keys(self) -> tuple[str, ...]:
        return self.node_identity + self.node_state

    @property
    def semantic_edge_keys(self) -> tuple[str, ...]:
        return self.edge_identity + self.edge_state

    @property
    def declared_node_keys(self) -> tuple[str, ...]:
        return self.semantic_node_keys + self.node_derived + self.node_annotations

    @property
    def declared_edge_keys(self) -> tuple[str, ...]:
        return self.semantic_edge_keys + self.edge_derived + self.edge_annotations


COMMON_CHEMICAL_SCHEMA = LabelSchema(
    node_identity=("element",),
    node_state=("aromatic", "hcount", "charge"),
    edge_state=("order",),
    node_annotations=("atom_map", "source_format"),
    name="chemical-common/1",
)

ELECTRON_LLG_SCHEMA = LabelSchema(
    node_identity=("element",),
    node_state=(
        "aromatic",
        "hcount",
        "radical",
        "lone_pairs",
        "valence_electrons",
    ),
    edge_state=("sigma_order", "pi_order"),
    node_derived=("charge", "bond_order_sum"),
    edge_derived=("order", "kekule_order"),
    node_annotations=("atom_map", "source_format"),
    name="lewis-electron/1",
)


def _select_labels(
    values: Mapping[str, Any],
    declared: Iterable[str],
    *,
    required: Iterable[str],
    issue_code: LLGIssueCode,
    owner: Any,
) -> dict[str, Any]:
    required_set = set(required)
    missing = required_set - set(values)
    if missing:
        raise LLGError(
            LLGIssue(
                issue_code,
                "A declared semantic label is missing.",
                {"owner": repr(owner), "labels": tuple(sorted(missing))},
            )
        )
    selected = {key: values[key] for key in declared if key in values}
    declared_set = set(declared)
    for key, value in values.items():
        if key not in declared_set:
            selected[f"annotation:{key}"] = value
    return selected


@dataclass(frozen=True)
class LewisLabelledGraph:
    """Immutable object of the finite LLG category."""

    schema: LabelSchema
    nodes: tuple[tuple[Hashable, LabelBundle], ...]
    edges: tuple[tuple[EdgeKey, LabelBundle], ...]
    name: Hashable | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        normalized_nodes = tuple(
            sorted(
                ((node, tuple(labels)) for node, labels in self.nodes),
                key=lambda item: _carrier_key(item[0]),
            )
        )
        normalized_edges = tuple(
            sorted(
                ((frozenset(edge), tuple(labels)) for edge, labels in self.edges),
                key=lambda item: _edge_sort_key(item[0]),
            )
        )
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "edges", normalized_edges)

        issues: list[LLGIssue] = []
        node_ids = tuple(node for node, _ in normalized_nodes)
        if len(set(node_ids)) != len(node_ids):
            issues.append(
                LLGIssue(LLGIssueCode.DUPLICATE_NODE, "Duplicate carrier node.")
            )
        edge_keys = tuple(edge for edge, _ in normalized_edges)
        if len(set(edge_keys)) != len(edge_keys):
            issues.append(
                LLGIssue(LLGIssueCode.DUPLICATE_EDGE, "Duplicate simple edge.")
            )
        for edge in edge_keys:
            if len(edge) != 2:
                issues.append(
                    LLGIssue(
                        LLGIssueCode.SELF_LOOP,
                        "LLG objects are loop-free simple graphs.",
                        {"edge": tuple(map(repr, edge))},
                    )
                )
            unknown = set(edge) - set(node_ids)
            if unknown:
                issues.append(
                    LLGIssue(
                        LLGIssueCode.UNKNOWN_ENDPOINT,
                        "An edge endpoint is outside the node carrier.",
                        {"nodes": tuple(sorted(map(repr, unknown)))},
                    )
                )
        if issues:
            raise LLGError(*issues)

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        schema: LabelSchema,
        *,
        name: Hashable | None = None,
    ) -> "LewisLabelledGraph":
        """Freeze a NetworkX graph under an explicit label schema."""
        if graph.is_directed():
            raise LLGError(
                LLGIssue(LLGIssueCode.DIRECTED_GRAPH, "LLG objects are undirected.")
            )
        if graph.is_multigraph():
            raise LLGError(
                LLGIssue(LLGIssueCode.MULTIGRAPH, "LLG objects are simple graphs.")
            )
        if nx.number_of_selfloops(graph):
            raise LLGError(
                LLGIssue(LLGIssueCode.SELF_LOOP, "LLG objects are loop-free.")
            )

        nodes = []
        for node, attrs in graph.nodes(data=True):
            labels = _select_labels(
                attrs,
                schema.declared_node_keys,
                required=schema.semantic_node_keys,
                issue_code=LLGIssueCode.MISSING_NODE_LABEL,
                owner=node,
            )
            nodes.append((node, _bundle(labels)))
        edges = []
        for left, right, attrs in graph.edges(data=True):
            labels = _select_labels(
                attrs,
                schema.declared_edge_keys,
                required=schema.semantic_edge_keys,
                issue_code=LLGIssueCode.MISSING_EDGE_LABEL,
                owner=(left, right),
            )
            edges.append((frozenset((left, right)), _bundle(labels)))
        return cls(schema, tuple(nodes), tuple(edges), name)

    @property
    def node_ids(self) -> frozenset[Hashable]:
        return frozenset(node for node, _ in self.nodes)

    @property
    def edge_keys(self) -> frozenset[EdgeKey]:
        return frozenset(edge for edge, _ in self.edges)

    def node_labels(self, node: Hashable, *, semantic: bool = False) -> dict[str, Any]:
        values = _bundle_dict(dict(self.nodes)[node])
        if semantic:
            return {key: values[key] for key in self.schema.semantic_node_keys}
        return values

    def edge_labels(self, edge: EdgeKey, *, semantic: bool = False) -> dict[str, Any]:
        values = _bundle_dict(dict(self.edges)[frozenset(edge)])
        if semantic:
            return {key: values[key] for key in self.schema.semantic_edge_keys}
        return values

    def to_networkx(self) -> nx.Graph:
        """Return a mutable copy. Frozen container values remain immutable."""
        graph = nx.Graph()
        graph.add_nodes_from(
            (node, _bundle_dict(labels)) for node, labels in self.nodes
        )
        for edge, labels in self.edges:
            left, right = tuple(edge)
            graph.add_edge(left, right, **_bundle_dict(labels))
        return graph

    def relabel(self, mapping: Mapping[Hashable, Hashable]) -> "LewisLabelledGraph":
        """Transport an object along a total carrier bijection."""
        if set(mapping) != set(self.node_ids) or len(set(mapping.values())) != len(
            mapping
        ):
            raise LLGError(
                LLGIssue(
                    LLGIssueCode.INVALID_RELABELING,
                    "An LLG relabeling must be a total carrier bijection.",
                )
            )
        return LewisLabelledGraph(
            self.schema,
            tuple((mapping[node], labels) for node, labels in self.nodes),
            tuple(
                (frozenset(mapping[node] for node in edge), labels)
                for edge, labels in self.edges
            ),
            self.name,
        )

    def is_isomorphic(self, other: "LewisLabelledGraph") -> bool:
        """Decide exact labeled-graph isomorphism; no digest is a proof."""
        if self.schema != other.schema:
            return False
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            self.to_networkx(),
            other.to_networkx(),
            node_match=lambda left, right: all(
                left.get(key) == right.get(key)
                for key in self.schema.semantic_node_keys
            ),
            edge_match=lambda left, right: all(
                left.get(key) == right.get(key)
                for key in self.schema.semantic_edge_keys
            ),
        )
        return matcher.is_isomorphic()


@dataclass(frozen=True)
class LLGMorphism:
    """A strict injective incidence- and label-preserving LLG morphism."""

    source: LewisLabelledGraph
    target: LewisLabelledGraph
    f: tuple[tuple[Hashable, Hashable], ...]

    def __post_init__(self) -> None:
        pairs = tuple(self.f.items()) if isinstance(self.f, Mapping) else tuple(self.f)
        object.__setattr__(
            self,
            "f",
            tuple(
                sorted(
                    pairs,
                    key=lambda item: (_carrier_key(item[0]), _carrier_key(item[1])),
                )
            ),
        )
        issues: list[LLGIssue] = []
        if self.source.schema != self.target.schema:
            issues.append(
                LLGIssue(
                    LLGIssueCode.NODE_LABEL_MISMATCH,
                    "Morphism endpoints must use the same label schema.",
                )
            )
        source_keys = tuple(left for left, _ in pairs)
        target_values = tuple(right for _, right in pairs)
        outside_source = set(source_keys) - set(self.source.node_ids)
        outside_target = set(target_values) - set(self.target.node_ids)
        missing = set(self.source.node_ids) - set(source_keys)
        if outside_source:
            issues.append(
                LLGIssue(
                    LLGIssueCode.OUTSIDE_SOURCE,
                    "The map contains a node outside its source.",
                    {"nodes": tuple(sorted(map(repr, outside_source)))},
                )
            )
        if outside_target:
            issues.append(
                LLGIssue(
                    LLGIssueCode.OUTSIDE_TARGET,
                    "The map contains an image outside its target.",
                    {"nodes": tuple(sorted(map(repr, outside_target)))},
                )
            )
        if len(set(source_keys)) != len(source_keys) or missing:
            issues.append(
                LLGIssue(
                    LLGIssueCode.PARTIAL_MAPPING,
                    "An LLG morphism maps each source node exactly once.",
                    {"missing": tuple(sorted(map(repr, missing)))},
                )
            )
        if len(set(target_values)) != len(target_values):
            issues.append(
                LLGIssue(
                    LLGIssueCode.NON_INJECTIVE,
                    "An LLG morphism is injective on material nodes.",
                )
            )
        if issues:
            raise LLGError(*issues)

        mapping = dict(pairs)
        for node, image in pairs:
            if self.source.node_labels(node, semantic=True) != self.target.node_labels(
                image, semantic=True
            ):
                issues.append(
                    LLGIssue(
                        LLGIssueCode.NODE_LABEL_MISMATCH,
                        "A node identity or state label is not preserved.",
                        {"source": repr(node), "target": repr(image)},
                    )
                )
        target_edges = self.target.edge_keys
        for edge in self.source.edge_keys:
            image_edge = frozenset(mapping[node] for node in edge)
            if image_edge not in target_edges:
                issues.append(
                    LLGIssue(
                        LLGIssueCode.MISSING_EDGE,
                        "The induced edge image is absent from the target.",
                        {"edge": tuple(sorted(map(repr, edge)))},
                    )
                )
                continue
            if self.source.edge_labels(edge, semantic=True) != self.target.edge_labels(
                image_edge, semantic=True
            ):
                issues.append(
                    LLGIssue(
                        LLGIssueCode.EDGE_LABEL_MISMATCH,
                        "An edge identity or state label is not preserved.",
                        {"edge": tuple(sorted(map(repr, edge)))},
                    )
                )
        if issues:
            raise LLGError(*issues)

    @property
    def mapping(self) -> dict[Hashable, Hashable]:
        return dict(self.f)

    @property
    def edge_mapping(self) -> dict[EdgeKey, EdgeKey]:
        mapping = self.mapping
        return {
            edge: frozenset(mapping[node] for node in edge)
            for edge in self.source.edge_keys
        }

    @property
    def is_isomorphism(self) -> bool:
        return len(self.source.node_ids) == len(self.target.node_ids) and len(
            self.source.edge_keys
        ) == len(self.target.edge_keys)

    @classmethod
    def identity(cls, graph: LewisLabelledGraph) -> "LLGMorphism":
        return cls(graph, graph, tuple((node, node) for node in graph.node_ids))

    def then(self, after: "LLGMorphism") -> "LLGMorphism":
        """Return ``after ∘ self``."""
        if self.target != after.source:
            raise LLGError(
                LLGIssue(
                    LLGIssueCode.ENDPOINT_MISMATCH,
                    "Composable LLG morphisms must share the same object.",
                )
            )
        right = after.mapping
        return LLGMorphism(
            self.source,
            after.target,
            tuple((node, right[image]) for node, image in self.f),
        )

    def compose(self, after: "LLGMorphism") -> "LLGMorphism":
        return self.then(after)

    def relabel(
        self,
        source_labels: Mapping[Hashable, Hashable],
        target_labels: Mapping[Hashable, Hashable],
    ) -> "LLGMorphism":
        source = self.source.relabel(source_labels)
        target = self.target.relabel(target_labels)
        return LLGMorphism(
            source,
            target,
            tuple(
                (source_labels[left], target_labels[right]) for left, right in self.f
            ),
        )


def derive_electron_labeled_graph(graph: nx.Graph) -> nx.Graph:
    """Return a copy with deterministic Lewis-derived charge and bond orders.

    Authoritative fields are ``valence_electrons``, ``hcount``, ``lone_pairs``,
    ``radical``, ``sigma_order``, and ``pi_order``.  A stored derived value may
    be absent; if present and inconsistent, construction fails rather than
    silently rewriting it.
    """
    result = graph.copy()
    for left, right, attrs in result.edges(data=True):
        try:
            total = float(attrs["sigma_order"]) + float(attrs["pi_order"])
        except KeyError as error:
            raise LLGError(
                LLGIssue(
                    LLGIssueCode.MISSING_EDGE_LABEL,
                    "Electron LLG edges require sigma_order and pi_order.",
                    {"edge": (repr(left), repr(right)), "label": str(error)},
                )
            ) from error
        for key in ("order", "kekule_order"):
            if key in attrs and not _numbers_equal(attrs[key], total):
                raise LLGError(
                    LLGIssue(
                        LLGIssueCode.DERIVED_LABEL_MISMATCH,
                        f"Stored {key} disagrees with sigma+pi order.",
                        {
                            "edge": (repr(left), repr(right)),
                            "stored": attrs[key],
                            "derived": total,
                        },
                    )
                )
            attrs[key] = total

    for node, attrs in result.nodes(data=True):
        required = {"valence_electrons", "hcount", "lone_pairs", "radical"}
        missing = required - set(attrs)
        if missing:
            raise LLGError(
                LLGIssue(
                    LLGIssueCode.MISSING_NODE_LABEL,
                    "Electron LLG nodes lack authoritative electron state.",
                    {"node": repr(node), "labels": tuple(sorted(missing))},
                )
            )
        bond_sum = sum(
            float(edge_attrs["sigma_order"]) + float(edge_attrs["pi_order"])
            for _, _, edge_attrs in result.edges(node, data=True)
        )
        charge = (
            float(attrs["valence_electrons"])
            - 2.0 * float(attrs["lone_pairs"])
            - float(attrs["radical"])
            - float(attrs["hcount"])
            - bond_sum
        )
        charge = int(charge) if charge.is_integer() else charge
        for key, derived in (("bond_order_sum", bond_sum), ("charge", charge)):
            if key in attrs and not _numbers_equal(attrs[key], derived):
                raise LLGError(
                    LLGIssue(
                        LLGIssueCode.DERIVED_LABEL_MISMATCH,
                        f"Stored {key} disagrees with authoritative electron state.",
                        {"node": repr(node), "stored": attrs[key], "derived": derived},
                    )
                )
            attrs[key] = derived
    return result


def _numbers_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Real) and isinstance(right, Real):
        return float(left) == float(right)
    return left == right


def llg_from_its(
    graph: nx.Graph,
    side: str | int,
    *,
    electron_complete: bool = False,
) -> LewisLabelledGraph:
    """Adapt tuple or legacy ``typesGH`` ITS data to one LLG value contract.

    The common schema is lossless for element, aromaticity, H count, charge,
    and total bond order in both formats.  Electron-complete adaptation is
    available only for tuple ITS because legacy ``typesGH`` does not retain
    side-specific lone pairs, radicals, or sigma/pi decomposition.
    """
    from synkit.Graph.ITS.its_decompose import its_decompose
    from synkit.Graph.ITS.its_reverter import ITSReverter
    from synkit.IO.chem_converter import detect_its_format

    side_index = {
        "reactant": 0,
        "r": 0,
        "left": 0,
        0: 0,
        "product": 1,
        "p": 1,
        "right": 1,
        1: 1,
    }.get(side)
    if side_index is None:
        raise ValueError(f"Unsupported ITS side: {side!r}")
    representation = detect_its_format(graph)
    if representation == "tuple":
        molecular = ITSReverter(graph).to_graph(side_index)
    elif representation == "typesGH":
        if electron_complete:
            raise LLGError(
                LLGIssue(
                    LLGIssueCode.LOSSY_ELECTRON_ADAPTER,
                    "Legacy typesGH lacks side-specific electron fields.",
                )
            )
        molecular = its_decompose(graph)[side_index]
    else:  # pragma: no cover - detect_its_format is a closed public contract
        raise LLGError(
            LLGIssue(
                LLGIssueCode.UNSUPPORTED_ITS_FORMAT,
                "Unsupported ITS representation.",
                {"format": repr(representation)},
            )
        )

    for node in molecular:
        molecular.nodes[node]["source_format"] = representation
        molecular.nodes[node].setdefault("atom_map", node)
    if electron_complete:
        molecular = derive_electron_labeled_graph(molecular)
        return LewisLabelledGraph.from_networkx(
            molecular, ELECTRON_LLG_SCHEMA, name=(representation, side_index)
        )

    for _, _, attrs in molecular.edges(data=True):
        if "order" not in attrs and "kekule_order" in attrs:
            attrs["order"] = attrs["kekule_order"]
    return LewisLabelledGraph.from_networkx(
        molecular, COMMON_CHEMICAL_SCHEMA, name=(representation, side_index)
    )


__all__ = [
    "COMMON_CHEMICAL_SCHEMA",
    "ELECTRON_LLG_SCHEMA",
    "LLGError",
    "LLGIssue",
    "LLGIssueCode",
    "LLGMorphism",
    "LabelSchema",
    "LewisLabelledGraph",
    "derive_electron_labeled_graph",
    "llg_from_its",
]
