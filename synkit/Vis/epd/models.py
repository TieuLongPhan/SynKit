from __future__ import annotations

"""Simple data models for mechanism visualization."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from synkit.Mechanism.symbols import internal_action_label, legacy_action_label

from .constants import transition_family


@dataclass(frozen=True)
class Transition:
    """Minimal transition record used by the visualizer.

    :param kind: Transition kind, e.g. ``'LP-/B+'`` or ``'B-/B+'``.
    :type kind: str
    :param src: Source atom/bond tuple.
    :type src: Tuple[int, ...]
    :param dst: Destination atom/bond tuple.
    :type dst: Tuple[int, ...]
    :param data: Auxiliary metadata such as ``shared_atom``.
    :type data: Dict[str, Any]
    """

    kind: str
    src: Tuple[int, ...]
    dst: Tuple[int, ...]
    data: Dict[str, Any]


def transition_from_epd_step(step: Any) -> Transition:
    """Normalize one SynKit EPD step into a drawable transition.

    :param step: Transition-like object. Supported forms are ``Transition``,
        a mapping with ``kind``, ``src`` and ``dst`` keys, or an ``epd_lw``
        row of ``[typed_action, source, target]``.
    :type step: Any
    :return: Drawable transition with typed action preserved in ``data``.
    :rtype: Transition
    :raises ValueError: If the step cannot be interpreted.

    Example
    -------
    .. code-block:: python

        from synkit.Vis.epd import transition_from_epd_step

        step = ["LP-/Sigma+", [1], [1, 2]]
        transition = transition_from_epd_step(step)
        assert transition.kind == "LP-/B+"
        assert transition.data["typed_kind"] == "LP-/Sigma+"
    """
    if isinstance(step, Transition):
        return step

    # Avoid coupling the visual model to a concrete Mechanism class while
    # accepting ElectronMove-compatible objects directly.
    if all(hasattr(step, name) for name in ("source", "target", "electron_count")):
        source_locus = step.source
        target_locus = step.target
        raw_kind = legacy_action_label(source_locus.kind, target_locus.kind)
        src = source_locus.atom_maps
        dst = target_locus.atom_maps
        data = {
            "internal_kind": internal_action_label(
                source_locus.kind, target_locus.kind
            ),
            "typed_kind": raw_kind,
            "electron_count": int(step.electron_count),
            "arrow_type": getattr(step, "arrow_type", None),
            "group_id": getattr(step, "group_id", None),
            "coupling_id": getattr(step, "coupling_id", None),
        }

    elif isinstance(step, Mapping):
        raw_kind = step.get("kind")
        src = step.get("src")
        dst = step.get("dst")
        data = dict(step.get("data", {}) or {})
    elif (
        isinstance(step, Sequence)
        and not isinstance(step, (str, bytes))
        and len(step) >= 3
    ):
        raw_kind, src, dst = step[:3]
        data = {}
    else:
        raise ValueError(f"Unsupported EPD transition step: {step!r}")

    if raw_kind is None or src is None or dst is None:
        raise ValueError(f"Incomplete EPD transition step: {step!r}")

    raw_kind = str(raw_kind)
    family = transition_family(raw_kind)
    if family != raw_kind:
        data.setdefault("typed_kind", raw_kind)
    if "internal_kind" not in data:
        try:
            source_token, target_token = raw_kind.split("-/", 1)
            data["internal_kind"] = internal_action_label(
                source_token, target_token.removesuffix("+")
            )
        except (TypeError, ValueError):
            pass

    return Transition(
        kind=family,
        src=tuple(int(x) for x in src),
        dst=tuple(int(x) for x in dst),
        data=data,
    )


def transitions_from_epd(epd: Iterable[Any]) -> List[Transition]:
    """Normalize SynKit ``epd_lw`` data for the mechanism visualizer.

    :param epd: Iterable of EPD transition steps.
    :type epd: Iterable[Any]
    :return: List of normalized transitions.
    :rtype: List[Transition]

    Example
    -------
    .. code-block:: python

        from synkit.Vis.epd import transitions_from_epd

        epd_lw = [
            ["LP-/Sigma+", [1], [1, 2]],
            ["Sigma-/LP+", [2, 3], [3]],
        ]
        transitions = transitions_from_epd(epd_lw)
    """
    return [transition_from_epd_step(step) for step in epd]
