"""Canonical reaction changes derived from relative local neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from synkit.Graph.Stereo import parse_virtual_reference

from .electron_models import VerificationIssue

Reference = int | str


@dataclass(frozen=True)
class RelativeNeighborChange:
    """One canonicalized terminal change over two relative local neighbors."""

    terminus: int
    neighbors: tuple[Reference, Reference]
    rotation: int
    frame_parity: int
    before_bonds: tuple[tuple[Any, ...], tuple[Any, ...]]
    after_bonds: tuple[tuple[Any, ...], tuple[Any, ...]]

    @property
    def physical_rotation(self) -> int:
        """Recover the supplied rotation in its derived local frame."""
        return self.rotation * self.frame_parity

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "synkit.relative-neighbor-change/1",
            "terminus": self.terminus,
            "frame": {
                "neighbors": list(self.neighbors),
                "permutation_parity": self.frame_parity,
            },
            "rotation": self.rotation,
            "changes": [
                {
                    "neighbor": neighbor,
                    "before_bond": list(before),
                    "after_bond": list(after),
                }
                for neighbor, before, after in zip(
                    self.neighbors,
                    self.before_bonds,
                    self.after_bonds,
                )
            ],
        }


def _map_lookup(graph: nx.Graph) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for node, attrs in graph.nodes(data=True):
        atom_map = attrs.get("atom_map")
        if type(atom_map) is not int or atom_map <= 0:
            continue
        if atom_map in result:
            raise ValueError(f"duplicate atom map {atom_map}")
        result[atom_map] = node
    return result


def _bond_state(
    graph: nx.Graph,
    lookup: dict[int, Any],
    owner: int,
    reference: Reference,
) -> tuple[Any, ...]:
    virtual = parse_virtual_reference(reference)
    if virtual is not None:
        return ("virtual", virtual.kind)
    edge = graph.edges[lookup[owner], lookup[reference]]
    return (
        "bond",
        float(edge.get("sigma_order", 0.0)),
        float(edge.get("pi_order", 0.0)),
    )


def _neighbor_token(
    before: nx.Graph,
    after: nx.Graph,
    before_lookup: dict[int, Any],
    after_lookup: dict[int, Any],
    owner: int,
    reference: Reference,
) -> tuple[Any, ...]:
    virtual = parse_virtual_reference(reference)
    if virtual is not None:
        return ("virtual", virtual.kind)

    def atom_identity(graph: nx.Graph, node: Any) -> tuple[Any, ...]:
        attrs = graph.nodes[node]
        return (
            attrs.get("element", ""),
            int(attrs.get("isotope", 0) or 0),
            int(attrs.get("charge", 0) or 0),
            int(attrs.get("radical", 0) or 0),
            bool(attrs.get("aromatic", False)),
            int(attrs.get("hcount", 0) or 0),
        )

    def local_star(
        graph: nx.Graph,
        lookup: dict[int, Any],
    ) -> tuple[Any, ...]:
        node = lookup[reference]
        owner_node = lookup[owner]
        shell = []
        for adjacent in graph.neighbors(node):
            if adjacent == owner_node:
                continue
            edge = graph.edges[node, adjacent]
            shell.append(
                (
                    float(edge.get("sigma_order", 0.0)),
                    float(edge.get("pi_order", 0.0)),
                    atom_identity(graph, adjacent),
                )
            )
        return atom_identity(graph, node), tuple(sorted(shell))

    return (
        "material",
        local_star(before, before_lookup),
        local_star(after, after_lookup),
        _bond_state(before, before_lookup, owner, reference),
        _bond_state(after, after_lookup, owner, reference),
    )


def _validate_substituent(
    graph: nx.Graph,
    lookup: dict[int, Any],
    terminus: int,
    substituent: Reference,
) -> str | None:
    virtual = parse_virtual_reference(substituent)
    if virtual is not None:
        if virtual.center != terminus:
            return f"virtual substituent {substituent!r} has the wrong owner"
        field = "hcount" if virtual.kind == "H" else "lone_pairs"
        if float(graph.nodes[lookup[terminus]].get(field, 0) or 0) < 1:
            return f"virtual substituent {substituent!r} " f"has no {field} resource"
        return None
    if type(substituent) is not int or substituent not in lookup:
        return f"mapped substituent {substituent!r} is absent"
    if not graph.has_edge(lookup[terminus], lookup[substituent]):
        return (
            f"mapped substituent {substituent} is not adjacent "
            f"to terminus {terminus}"
        )
    return None


def derive_relative_neighbor_changes(
    motion: Any,
    before: nx.Graph,
    after: nx.Graph,
    *,
    step_id: str,
) -> tuple[tuple[RelativeNeighborChange, ...], tuple[VerificationIssue, ...]]:
    """Derive and canonicalize terminal changes from local neighbors."""
    issues: list[VerificationIssue] = []
    try:
        before_lookup = _map_lookup(before)
        after_lookup = _map_lookup(after)
    except ValueError as exc:
        return (), (
            VerificationIssue(
                "ELECTROCYCLIC_NEIGHBOR_INVALID",
                str(exc),
                step_id=step_id,
            ),
        )

    required = set(motion.termini)
    if not required <= set(before_lookup) or not required <= set(after_lookup):
        return (), (
            VerificationIssue(
                "ELECTROCYCLIC_NEIGHBOR_INVALID",
                "A terminal map is absent from an endpoint neighborhood.",
                step_id=step_id,
            ),
        )

    changes: list[RelativeNeighborChange] = []
    for index, (terminus, substituent, rotation) in enumerate(
        zip(motion.termini, motion.substituents, motion.terminal_motion)
    ):
        partner = motion.termini[1 - index]
        endpoint_error = next(
            (
                error
                for graph, lookup in (
                    (before, before_lookup),
                    (after, after_lookup),
                )
                if (
                    error := _validate_substituent(
                        graph,
                        lookup,
                        terminus,
                        substituent,
                    )
                )
            ),
            None,
        )
        if endpoint_error is not None:
            issues.append(
                VerificationIssue(
                    "ELECTROCYCLIC_NEIGHBOR_INVALID",
                    endpoint_error,
                    step_id=step_id,
                    atom_maps=(terminus,),
                )
            )
            continue

        excluded = {partner}
        if type(substituent) is int:
            excluded.add(substituent)

        def candidates(
            graph: nx.Graph,
            lookup: dict[int, Any],
        ) -> set[int]:
            reverse = {node: atom_map for atom_map, node in lookup.items()}
            return {
                reverse[node]
                for node in graph.neighbors(lookup[terminus])
                if node in reverse and reverse[node] not in excluded
            }

        inward = candidates(
            before,
            before_lookup,
        ) & candidates(after, after_lookup)
        if len(inward) != 1:
            issues.append(
                VerificationIssue(
                    "ELECTROCYCLIC_NEIGHBOR_AMBIGUOUS",
                    "A terminus requires one shared inward " "local neighbor.",
                    step_id=step_id,
                    atom_maps=(terminus, *sorted(inward)),
                )
            )
            continue
        inward_reference = next(iter(inward))
        raw_neighbors: tuple[Reference, Reference] = (
            inward_reference,
            substituent,
        )
        tokens = tuple(
            _neighbor_token(
                before,
                after,
                before_lookup,
                after_lookup,
                terminus,
                reference,
            )
            for reference in raw_neighbors
        )
        if tokens[0] == tokens[1]:
            issues.append(
                VerificationIssue(
                    "ELECTROCYCLIC_NEIGHBOR_AMBIGUOUS",
                    "The relative local-neighbor frame has no unique "
                    "canonical order.",
                    step_id=step_id,
                    atom_maps=(terminus,),
                )
            )
            continue
        order = (0, 1) if tokens[0] < tokens[1] else (1, 0)
        frame_parity = 1 if order == (0, 1) else -1
        changes.append(
            RelativeNeighborChange(
                terminus,
                tuple(raw_neighbors[position] for position in order),
                rotation * frame_parity,
                frame_parity,
                tuple(
                    _bond_state(
                        before,
                        before_lookup,
                        terminus,
                        raw_neighbors[position],
                    )
                    for position in order
                ),
                tuple(
                    _bond_state(
                        after,
                        after_lookup,
                        terminus,
                        raw_neighbors[position],
                    )
                    for position in order
                ),
            )
        )
    changes.sort(key=lambda change: change.terminus)
    return tuple(changes), tuple(issues)
