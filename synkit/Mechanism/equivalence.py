"""Deterministic mechanism comparison at net, event, and trajectory levels."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Hashable

import networkx as nx
from rdkit import Chem

from synkit.Graph.Stereo import (
    descriptor_relative_form,
    mapped_stereo_registries_match,
    parse_virtual_reference,
    stereo_from_dict,
)
from synkit.Graph.Stereo.identity import (
    stereo_identity_edge_match,
    stereo_identity_node_match,
)
from synkit.IO.mol_to_graph import MolToGraph

from .model import MechanismRecord

ReferenceMap = Mapping[int, Hashable]


def _side_graph(
    text: str,
    endpoint_registry: Mapping[str, Any] | None = None,
) -> nx.Graph:
    """Parse one reaction side into SynKit's relative-stereo graph model."""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Cannot parse reaction side {text!r}.")
    graph = MolToGraph().transform(mol)
    if endpoint_registry is not None:
        graph.graph["stereo_descriptors"] = {
            key: stereo_from_dict(descriptor.to_dict())
            for key, descriptor in endpoint_registry.items()
        }
    return graph


def _reaction_graphs(record: MechanismRecord) -> tuple[nx.Graph, nx.Graph]:
    sides = record.mapped_reaction.split(">>", 1)
    return tuple(
        _side_graph(
            text,
            record.endpoint_stereo.get(side)
            if side in record.endpoint_stereo
            else None,
        )
        for side, text in zip(("reactant", "product"), sides)
    )


def _matcher_type(graph: nx.Graph) -> type:
    if graph.is_directed():
        return (
            nx.algorithms.isomorphism.MultiDiGraphMatcher
            if graph.is_multigraph()
            else nx.algorithms.isomorphism.DiGraphMatcher
        )
    return (
        nx.algorithms.isomorphism.MultiGraphMatcher
        if graph.is_multigraph()
        else nx.algorithms.isomorphism.GraphMatcher
    )


def _stereo_structural_mappings(
    left: nx.Graph,
    right: nx.Graph,
    *,
    required_atom_maps: Mapping[int, int] | None = None,
) -> Iterator[dict[Any, Any]]:
    """Yield exact structural mappings that preserve relative stereo."""
    if (
        left.is_directed() != right.is_directed()
        or left.is_multigraph() != right.is_multigraph()
    ):
        return

    def node_match(
        left_attributes: Mapping[str, Any],
        right_attributes: Mapping[str, Any],
    ) -> bool:
        if not _mechanism_node_match(left_attributes, right_attributes):
            return False
        if required_atom_maps is None:
            return True
        left_map = _positive_atom_map(left_attributes)
        if left_map not in required_atom_maps:
            return True
        return _positive_atom_map(right_attributes) == required_atom_maps[left_map]

    matcher = _matcher_type(left)(
        left,
        right,
        node_match=node_match,
        edge_match=stereo_identity_edge_match,
    )
    for mapping in matcher.isomorphisms_iter():
        if mapped_stereo_registries_match(left, right, mapping):
            yield dict(mapping)


def _positive_atom_map(attributes: Mapping[str, Any]) -> int | None:
    value = attributes.get("atom_map", 0)
    return value if type(value) is int and value > 0 else None


def _mechanism_node_match(
    left_attributes: Mapping[str, Any],
    right_attributes: Mapping[str, Any],
) -> bool:
    """Compare molecular identity, including isotope but excluding AAM."""
    return stereo_identity_node_match(left_attributes, right_attributes) and (
        left_attributes.get("isotope", 0) == right_attributes.get("isotope", 0)
    )


def _atom_map_correspondence(
    left: nx.Graph,
    right: nx.Graph,
    node_mapping: Mapping[Any, Any],
) -> dict[int, int] | None:
    """Project one node isomorphism onto a bijection of positive atom maps."""
    correspondence: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for left_node, right_node in node_mapping.items():
        left_map = _positive_atom_map(left.nodes[left_node])
        right_map = _positive_atom_map(right.nodes[right_node])
        if (left_map is None) != (right_map is None):
            return None
        if left_map is None or right_map is None:
            continue
        if left_map in correspondence or right_map in reverse:
            # Mapped mechanism sides require unique positive AAM values.
            return None
        correspondence[left_map] = right_map
        reverse[right_map] = left_map
    return correspondence


def _merge_correspondences(
    reactant: Mapping[int, int],
    product: Mapping[int, int],
) -> dict[int, int] | None:
    """Require one atom identity translation across both reaction sides."""
    merged = dict(reactant)
    reverse = {right: left for left, right in merged.items()}
    for left, right in product.items():
        if left in merged and merged[left] != right:
            return None
        if right in reverse and reverse[right] != left:
            return None
        merged[left] = right
        reverse[right] = left
    return merged


def _reaction_correspondences(
    left_sides: tuple[nx.Graph, nx.Graph],
    right_sides: tuple[nx.Graph, nx.Graph],
) -> Iterator[dict[int, int]]:
    """Yield map bijections induced by exact mappings of both endpoints."""
    for reactant_mapping in _stereo_structural_mappings(
        left_sides[0], right_sides[0]
    ):
        reactant = _atom_map_correspondence(
            left_sides[0], right_sides[0], reactant_mapping
        )
        if reactant is None:
            continue
        for product_mapping in _stereo_structural_mappings(
            left_sides[1],
            right_sides[1],
            required_atom_maps=reactant,
        ):
            product = _atom_map_correspondence(
                left_sides[1], right_sides[1], product_mapping
            )
            if product is None:
                continue
            merged = _merge_correspondences(reactant, product)
            if merged is not None:
                yield merged


def _reference_map(correspondence: Mapping[int, int]) -> dict[int, Hashable]:
    """Give mapped atoms comparable typed tokens independent of AAM values."""
    return {left: ("atom", right) for left, right in correspondence.items()}


def _right_reference_map(correspondence: Mapping[int, int]) -> dict[int, Hashable]:
    return {right: ("atom", right) for right in correspondence.values()}


def _resolve_atom(atom_map: int, references: ReferenceMap) -> Hashable:
    return references.get(atom_map, ("missing", atom_map))


def _relative_descriptor_signature(
    descriptor: Any,
    state: str,
    references: ReferenceMap,
) -> tuple[Any, ...]:
    """Resolve one native relative descriptor through an atom correspondence."""

    def resolve(reference: Any) -> Hashable:
        if isinstance(reference, int):
            return _resolve_atom(reference, references)
        virtual = parse_virtual_reference(reference)
        if virtual is None:
            return ("invalid", repr(reference))
        return (
            "virtual",
            virtual.kind,
            _resolve_atom(virtual.center, references),
        )

    return state, descriptor_relative_form(descriptor, resolve)


def _descriptor_signature(
    descriptor: Any, references: ReferenceMap
) -> tuple[Any, ...] | None:
    if descriptor is None:
        return None
    if descriptor.descriptor_class == "unknown":
        return ("unknown", descriptor.state)
    native = stereo_from_dict(descriptor.to_dict())
    return _relative_descriptor_signature(native, descriptor.state, references)


def _endpoint_stereo_signature(
    record: MechanismRecord,
    side_graphs: tuple[nx.Graph, nx.Graph],
    references: ReferenceMap,
) -> tuple[Any, ...]:
    """Return effective endpoint stereo, preserving sidecar state semantics."""
    result = []
    for index, side in enumerate(("reactant", "product")):
        if side in record.endpoint_stereo:
            signatures = (
                _descriptor_signature(descriptor, references)
                for descriptor in record.endpoint_stereo[side].values()
            )
        else:
            signatures = (
                _relative_descriptor_signature(
                    descriptor,
                    "specified" if descriptor.parity is not None else "unknown",
                    references,
                )
                for descriptor in side_graphs[index].graph.get(
                    "stereo_descriptors", {}
                ).values()
            )
        result.append((side, tuple(sorted(signatures, key=repr))))
    return tuple(result)


def _stereo_effect_signature(
    effect: Any, references: ReferenceMap
) -> tuple[Any, ...]:
    target_kind, target_reference = effect.descriptor_target
    if isinstance(target_reference, tuple):
        ranked_reference: Any = tuple(
            sorted(_resolve_atom(value, references) for value in target_reference)
        )
    else:
        ranked_reference = _resolve_atom(target_reference, references)
    target = (
        target_kind,
        ranked_reference,
    )
    return (
        effect.effect,
        target,
        _descriptor_signature(effect.before, references),
        _descriptor_signature(effect.after, references),
    )


def _stereo_effect_dependencies(
    effect: Any, references: ReferenceMap
) -> frozenset[Hashable]:
    dependencies = {
        value
        for descriptor in (effect.before, effect.after)
        if descriptor is not None
        for value in descriptor.atoms
        if isinstance(value, int)
    }
    target = effect.descriptor_target[1]
    if isinstance(target, int):
        dependencies.add(target)
    else:
        dependencies.update(target)
    return frozenset(_resolve_atom(value, references) for value in dependencies)


def _group_signature(group: Any, references: ReferenceMap) -> tuple[Any, ...]:
    moves = []
    for move in group.moves:
        source = (
            move.source.kind,
            tuple(
                sorted(
                    _resolve_atom(value, references)
                    for value in move.source.atom_maps
                )
            ),
        )
        target = (
            move.target.kind,
            tuple(
                sorted(
                    _resolve_atom(value, references)
                    for value in move.target.atom_maps
                )
            ),
        )
        moves.append(
            (source, target, move.electron_count, move.arrow_type, move.coupling_id)
        )
    return (group.macro, tuple(sorted(moves, key=repr)))


def _group_dependencies(
    group: Any, references: ReferenceMap
) -> frozenset[Hashable]:
    return frozenset(
        _resolve_atom(atom_map, references)
        for move in group.moves
        for locus in (move.source, move.target)
        for atom_map in locus.atom_maps
    )


def _event_signature(
    record: MechanismRecord,
    references: ReferenceMap,
    *,
    trajectory: bool,
) -> tuple[Any, ...]:
    steps = []
    event_entries: list[tuple[tuple[Any, ...], frozenset[Hashable]]] = []
    for step in record.steps:
        groups = tuple(
            sorted(
                (_group_signature(group, references) for group in step.groups),
                key=repr,
            )
        )
        stereo = tuple(
            sorted(
                (
                    _stereo_effect_signature(effect, references)
                    for effect in step.stereo_effects
                ),
                key=repr,
            )
        )
        entry = (groups, stereo)
        if trajectory:
            steps.append(entry)
        else:
            for group, original in sorted(
                zip(
                    (
                        _group_signature(value, references)
                        for value in step.groups
                    ),
                    step.groups,
                ),
                key=lambda item: repr(item[0]),
            ):
                event_entries.append(
                    (
                        ("group", group),
                        _group_dependencies(original, references),
                    )
                )
            for effect in step.stereo_effects:
                event_entries.append(
                    (
                        ("stereo", _stereo_effect_signature(effect, references)),
                        _stereo_effect_dependencies(effect, references),
                    )
                )
    if trajectory:
        return tuple(steps)
    # Canonicalize only adjacent independent entries. Dependent entries retain
    # their supplied order, so endpoint-equivalent noncommuting paths differ.
    changed = True
    while changed:
        changed = False
        for index in range(len(event_entries) - 1):
            left, right = event_entries[index : index + 2]
            if left[1].isdisjoint(right[1]) and repr(left[0]) > repr(right[0]):
                event_entries[index : index + 2] = [right, left]
                changed = True
    return tuple(value for value, _ in event_entries)


def mechanism_equivalent(
    left: MechanismRecord,
    right: MechanismRecord,
    *,
    level: str = "net",
) -> bool:
    """Compare net products, supplied events, or ordered trajectory states.

    Equality is witnessed by one exact, relative-stereo-preserving graph
    correspondence that is consistent across reactant and product sides.
    Backend canonical ranks and canonical SMILES are not identity inputs.
    """
    if level not in {"net", "events", "trajectory"}:
        raise ValueError("level must be 'net', 'events', or 'trajectory'.")
    trajectory = level == "trajectory"
    left_sides = _reaction_graphs(left)
    right_sides = _reaction_graphs(right)
    for correspondence in _reaction_correspondences(left_sides, right_sides):
        left_references = _reference_map(correspondence)
        right_references = _right_reference_map(correspondence)
        if _endpoint_stereo_signature(
            left,
            left_sides,
            left_references,
        ) != _endpoint_stereo_signature(
            right,
            right_sides,
            right_references,
        ):
            continue
        if level == "net":
            return True
        if _event_signature(
            left,
            left_references,
            trajectory=trajectory,
        ) == _event_signature(
            right,
            right_references,
            trajectory=trajectory,
        ):
            return True
    return False
