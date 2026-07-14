"""Deterministic mechanism comparison at net, event, and trajectory levels."""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from .model import MechanismRecord


def _canonical_side(text: str) -> str:
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Cannot parse reaction side {text!r}.")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _map_ranks(record: MechanismRecord) -> dict[int, int]:
    reactants = record.mapped_reaction.split(">>", 1)[0]
    mol = Chem.MolFromSmiles(reactants)
    if mol is None:
        raise ValueError("Cannot canonicalize mapped reactants.")
    atom_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=True)
    return {
        atom_map: int(ranks[index]) + 1
        for index, atom_map in enumerate(atom_maps)
        if atom_map > 0
    }


def _descriptor_signature(
    descriptor: Any, ranks: dict[int, int]
) -> tuple[Any, ...] | None:
    if descriptor is None:
        return None
    atoms = tuple(
        ranks.get(value, value) if isinstance(value, int) else value
        for value in descriptor.atoms
    )
    return (descriptor.descriptor_class, atoms, descriptor.parity, descriptor.state)


def _group_signature(group: Any, ranks: dict[int, int]) -> tuple[Any, ...]:
    moves = []
    for move in group.moves:
        source = (
            move.source.kind,
            tuple(sorted(ranks.get(x, x) for x in move.source.atom_maps)),
        )
        target = (
            move.target.kind,
            tuple(sorted(ranks.get(x, x) for x in move.target.atom_maps)),
        )
        moves.append(
            (source, target, move.electron_count, move.arrow_type, move.coupling_id)
        )
    return (group.macro, tuple(sorted(moves, key=repr)))


def _group_dependencies(group: Any, ranks: dict[int, int]) -> frozenset[int]:
    return frozenset(
        ranks.get(atom_map, atom_map)
        for move in group.moves
        for locus in (move.source, move.target)
        for atom_map in locus.atom_maps
    )


def _event_signature(record: MechanismRecord, *, trajectory: bool) -> tuple[Any, ...]:
    ranks = _map_ranks(record)
    steps = []
    event_entries: list[tuple[tuple[Any, ...], frozenset[int]]] = []
    for step in record.steps:
        groups = tuple(
            sorted((_group_signature(group, ranks) for group in step.groups), key=repr)
        )
        stereo = tuple(
            sorted(
                (
                    (
                        effect.effect,
                        (
                            effect.descriptor_target[0],
                            ranks.get(
                                effect.descriptor_target[1], effect.descriptor_target[1]
                            ),
                        ),
                        _descriptor_signature(effect.before, ranks),
                        _descriptor_signature(effect.after, ranks),
                    )
                    for effect in step.stereo_effects
                ),
                key=repr,
            )
        )
        entry = (groups, stereo)
        if trajectory:
            steps.append(entry)
        else:
            stereo_dependencies = frozenset(target[1] for _, target, _, _ in stereo)
            for group, original in sorted(
                zip(
                    (_group_signature(value, ranks) for value in step.groups),
                    step.groups,
                ),
                key=lambda item: repr(item[0]),
            ):
                event_entries.append(
                    (
                        (group, stereo),
                        _group_dependencies(original, ranks) | stereo_dependencies,
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
    """Compare net products, supplied events, or ordered trajectory states."""
    if level not in {"net", "events", "trajectory"}:
        raise ValueError("level must be 'net', 'events', or 'trajectory'.")
    left_sides = left.mapped_reaction.split(">>", 1)
    right_sides = right.mapped_reaction.split(">>", 1)
    if tuple(map(_canonical_side, left_sides)) != tuple(
        map(_canonical_side, right_sides)
    ):
        return False
    if level == "net":
        return True
    return _event_signature(left, trajectory=level == "trajectory") == _event_signature(
        right, trajectory=level == "trajectory"
    )
