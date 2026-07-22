"""Compatibility adapters between v1 EPD/EF-SMIRKS and v2 models."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .model import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismModelError,
    MechanismRecord,
    MechanisticStep,
)
from .symbols import legacy_action_label, normalize_locus_symbol

_ACTION_LOCI = {
    "LP": "lp",
    "RAD": "∙",
    "SIGMA": "σ",
    "PI": "π",
}


def _locus(token: str, atom_maps: Sequence[int]) -> ElectronLocus:
    kind = _ACTION_LOCI.get(token.strip().upper())
    if kind is None:
        if token.strip().upper() == "B":
            raise MechanismModelError(
                "Generic legacy 'B' does not identify σ versus π; use typed "
                "EPD or an ITS-aware adapter."
            )
        try:
            kind = normalize_locus_symbol(token)
        except (TypeError, ValueError) as exc:
            raise MechanismModelError(
                f"Unsupported legacy EPD locus: {token!r}"
            ) from exc
    return ElectronLocus(kind=kind, atom_maps=tuple(int(value) for value in atom_maps))


def electron_move_from_legacy_epd(
    row: Sequence[Any], *, group_id: str = "g1", event_id: str | None = None
) -> ElectronMove:
    """Losslessly adapt one polar v1 ``[action, source, target]`` row."""
    if len(row) < 3:
        raise MechanismModelError(f"Invalid legacy EPD row: {row!r}")
    action, source, target = row[:3]
    try:
        source_token, target_token = str(action).split("-/", 1)
        target_token = target_token.removesuffix("+")
    except ValueError as exc:
        raise MechanismModelError(f"Invalid legacy EPD action: {action!r}") from exc
    return ElectronMove(
        event_id=event_id,
        source=_locus(source_token, source),
        target=_locus(target_token, target),
        electron_count=2,
        arrow_type="curved",
        group_id=group_id,
        metadata={"legacy_action": str(action)},
    )


def group_from_legacy_epd(
    epd: Iterable[Sequence[Any]], *, group_id: str = "g1"
) -> ElectronMoveGroup:
    moves = tuple(
        electron_move_from_legacy_epd(row, group_id=group_id, event_id=f"e{index}")
        for index, row in enumerate(epd, start=1)
    )
    return ElectronMoveGroup(group_id=group_id, moves=moves)


def mechanism_from_legacy_epd(
    mapped_reaction: str,
    epd: Iterable[Sequence[Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> MechanismRecord:
    # Legacy EPD is an ordered list of local edits.  Preserve that ordering by
    # giving every row its own step/group; callers can explicitly coalesce
    # events when they have simultaneous-event provenance.
    steps = tuple(
        MechanisticStep(
            step_id=f"s{index}",
            groups=(
                ElectronMoveGroup(
                    group_id=f"g{index}",
                    moves=(
                        electron_move_from_legacy_epd(
                            row, group_id=f"g{index}", event_id=f"e{index}"
                        ),
                    ),
                ),
            ),
        )
        for index, row in enumerate(epd, start=1)
    )
    return MechanismRecord(
        mapped_reaction=mapped_reaction,
        steps=steps,
        provenance={"format": "legacy_epd", **(provenance or {})},
    )


def mechanism_from_ef_smirks(text: str, **kwargs: Any) -> MechanismRecord:
    """Create a v2 record from existing EF-SMIRKS conversion support.

    EF-SMIRKS currently encodes legacy two-electron arrow geometry only;
    fishhook metadata is represented directly by :class:`ElectronMove`.
    """
    from synkit.Graph.Mech.conversion import ef_smirks_to_epd

    converted = ef_smirks_to_epd(text, **kwargs)
    return mechanism_from_legacy_epd(
        converted["complete_aam"],
        converted["epd_lw"],
        provenance={"format": "ef_smirks", "ef_smirks": text},
    )


def legacy_epd_from_group(group: ElectronMoveGroup) -> list[list[Any]]:
    """Return v1 rows when all moves are representable polar curved arrows."""
    rows: list[list[Any]] = []
    for move in group.moves:
        if move.electron_count != 2:
            raise MechanismModelError("Fishhooks cannot be represented by legacy EPD.")
        rows.append(
            [
                legacy_action_label(move.source.kind, move.target.kind),
                list(move.source.atom_maps),
                list(move.target.atom_maps),
            ]
        )
    return rows
