"""Derived CIP-style labels projected from configured stereo descriptors.

Assignments are immutable reporting values.  They never mutate a molecule or
descriptor, and their labels are deliberately absent from stereo identity,
hashing, rule matching, and descriptor serialization.  Ligand priority comes
only from :mod:`synkit.Chem.Molecule.cip_ranking`.

Axial ``M/P`` projection uses the IUPAC helicity convention: from a witnessed
viewing direction, the path from the nearer high-priority ligand to the farther
high-priority ligand is ``P`` for the positive sense and ``M`` for the negative
sense.  If constitution cannot witness a stable viewing direction, projection
fails closed instead of using atom indices as a tiebreak.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, Iterable, Mapping

from rdkit import Chem

from synkit.Chem.Molecule.cip_ranking import (
    CIPComparison,
    CIPComparisonOutcome,
    CIPRanker,
    CIPRanking,
    CIPSequenceRule,
)
from synkit.Graph.Stereo import (
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
)


class CIPAssignmentStatus(str, Enum):
    ASSIGNED = "assigned"
    UNSPECIFIED_CONFIGURATION = "unspecified_configuration"
    UNRESOLVED_PRIORITY = "unresolved_priority"
    UNSUPPORTED_SEQUENCE_RULE = "unsupported_sequence_rule"
    UNSUPPORTED_DESCRIPTOR = "unsupported_descriptor"


@dataclass(frozen=True)
class CIPAssignment:
    """One derived label result with all priority and projection witnesses."""

    descriptor_class: str
    descriptor_identifier: str
    label: str | None
    status: CIPAssignmentStatus
    rankings: tuple[CIPRanking, ...]
    supplemental_comparisons: tuple[CIPComparison, ...]
    projection_witness: tuple[Any, ...]
    required_rules: tuple[CIPSequenceRule, ...]
    reason: str

    @property
    def assigned(self) -> bool:
        return self.status is CIPAssignmentStatus.ASSIGNED

    @property
    def digest(self) -> str:
        return sha256(repr(self).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return an inspectable report with no label in stereo identity."""
        return {
            "descriptor_class": self.descriptor_class,
            "descriptor_identifier": self.descriptor_identifier,
            "label": self.label,
            "status": self.status.value,
            "ranking_digests": [ranking.digest for ranking in self.rankings],
            "ordered_references": [
                list(ranking.ordered_references) for ranking in self.rankings
            ],
            "supplemental_comparisons": [
                {
                    "outcome": comparison.outcome.value,
                    "deciding_rule": (
                        comparison.deciding_rule.value
                        if comparison.deciding_rule is not None
                        else None
                    ),
                    "depth": comparison.depth,
                    "left_witness": comparison.left_witness,
                    "right_witness": comparison.right_witness,
                }
                for comparison in self.supplemental_comparisons
            ],
            "projection_witness": list(self.projection_witness),
            "required_rules": [rule.value for rule in self.required_rules],
            "reason": self.reason,
            "digest": self.digest,
        }


_SUPPORTED_TYPES = (
    TetrahedralStereo,
    PlanarBondStereo,
    AtropBondStereo,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
)
_COORDINATION_TYPES = (
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
)


def _permutation_sign(source: tuple[Any, ...], target: tuple[Any, ...]) -> int:
    positions = {value: index for index, value in enumerate(source)}
    image = tuple(positions[value] for value in target)
    inversions = sum(
        image[left] > image[right]
        for left in range(len(image))
        for right in range(left + 1, len(image))
    )
    return -1 if inversions % 2 else 1


def _normalize_descriptor(
    molecule: Chem.Mol,
    descriptor: Any,
    reference_to_index: Mapping[int, int] | None,
) -> Any:
    dependencies = frozenset(getattr(descriptor, "dependencies", ()))
    if reference_to_index is None:
        normalized = descriptor
    else:
        missing = dependencies - set(reference_to_index)
        if missing:
            raise ValueError(
                "CIP reference mapping is missing descriptor references: "
                f"{sorted(missing)}."
            )
        values = tuple(reference_to_index[reference] for reference in dependencies)
        if any(type(value) is not int for value in values):
            raise TypeError("CIP reference mappings require integer atom indices.")
        if len(set(values)) != len(values):
            raise ValueError("CIP reference mappings must be injective.")
        normalized = descriptor.relabel(reference_to_index)
    count = molecule.GetNumAtoms()
    invalid = sorted(
        reference
        for reference in normalized.dependencies
        if reference < 0 or reference >= count
    )
    if invalid:
        raise ValueError(f"CIP descriptor references are absent: {invalid}.")
    return normalized


def _comparisons(
    rankings: Iterable[CIPRanking],
    supplemental: Iterable[CIPComparison] = (),
) -> tuple[CIPComparison, ...]:
    return tuple(
        comparison for ranking in rankings for comparison in ranking.comparisons
    ) + tuple(supplemental)


def _required_rules(
    comparisons: Iterable[CIPComparison],
) -> tuple[CIPSequenceRule, ...]:
    rules = {
        comparison.deciding_rule
        for comparison in comparisons
        if comparison.deciding_rule is not None
    }
    return tuple(sorted(rules, key=lambda rule: rule.value))


def _result(
    original: Any,
    status: CIPAssignmentStatus,
    reason: str,
    *,
    label: str | None = None,
    rankings: Iterable[CIPRanking] = (),
    supplemental: Iterable[CIPComparison] = (),
    witness: Iterable[Any] = (),
) -> CIPAssignment:
    ranking_values = tuple(rankings)
    supplemental_values = tuple(supplemental)
    comparisons = _comparisons(ranking_values, supplemental_values)
    return CIPAssignment(
        original.descriptor_class,
        descriptor_id(original),
        label,
        status,
        ranking_values,
        supplemental_values,
        tuple(witness),
        _required_rules(comparisons),
        reason,
    )


def _incomplete_result(
    original: Any,
    rankings: Iterable[CIPRanking],
    *,
    supplemental: Iterable[CIPComparison] = (),
) -> CIPAssignment:
    ranking_values = tuple(rankings)
    supplemental_values = tuple(supplemental)
    comparisons = _comparisons(ranking_values, supplemental_values)
    unsupported = tuple(
        comparison
        for comparison in comparisons
        if comparison.outcome is CIPComparisonOutcome.UNSUPPORTED
    )
    if unsupported:
        status = CIPAssignmentStatus.UNSUPPORTED_SEQUENCE_RULE
        reason = "; ".join(dict.fromkeys(item.reason for item in unsupported))
    else:
        status = CIPAssignmentStatus.UNRESOLVED_PRIORITY
        reason = "At least two descriptor ligands have unresolved equal priority."
    return _result(
        original,
        status,
        reason,
        rankings=ranking_values,
        supplemental=supplemental_values,
    )


def _assign_tetrahedral(
    original: TetrahedralStereo,
    descriptor: TetrahedralStereo,
    ranker: CIPRanker,
) -> CIPAssignment:
    if descriptor.parity is None:
        return _result(
            original,
            CIPAssignmentStatus.UNSPECIFIED_CONFIGURATION,
            "Tetrahedral orientation is unspecified.",
        )
    references = tuple(descriptor.atoms[1:])
    ranking = ranker.rank(descriptor.center, references)
    if not ranking.complete:
        return _incomplete_result(original, (ranking,))
    sign = descriptor.parity * _permutation_sign(references, ranking.ordered_references)
    label = "R" if sign == 1 else "S"
    if CIPSequenceRule.RULE_5_REFLECTION_VARIANT in _required_rules(
        ranking.comparisons
    ):
        reflected = ranker.reflected().rank(descriptor.center, references)
        if reflected.complete:
            reflected_sign = -descriptor.parity * _permutation_sign(
                references,
                reflected.ordered_references,
            )
            if reflected_sign == sign:
                label = label.lower()
    return _result(
        original,
        CIPAssignmentStatus.ASSIGNED,
        "Tetrahedral chirality projected from descending ligand priority.",
        label=label,
        rankings=(ranking,),
        witness=(descriptor.parity, sign),
    )


def _assign_planar(
    original: PlanarBondStereo | ExtendedCisTransStereo,
    descriptor: PlanarBondStereo | ExtendedCisTransStereo,
    ranker: CIPRanker,
) -> CIPAssignment:
    if descriptor.parity is None:
        return _result(
            original,
            CIPAssignmentStatus.UNSPECIFIED_CONFIGURATION,
            "Cis/trans orientation is unspecified.",
        )
    left = ranker.rank(descriptor.atoms[2], descriptor.atoms[:2])
    right = ranker.rank(descriptor.atoms[3], descriptor.atoms[4:])
    rankings = left, right
    if not all(ranking.complete for ranking in rankings):
        return _incomplete_result(original, rankings)
    left_position = descriptor.atoms[:2].index(left.ordered_references[0])
    right_position = descriptor.atoms[4:].index(right.ordered_references[0])
    label = "Z" if left_position == right_position else "E"
    if CIPSequenceRule.RULE_5_REFLECTION_VARIANT in _required_rules(
        _comparisons(rankings)
    ):
        reflected_ranker = ranker.reflected()
        reflected_left = reflected_ranker.rank(
            descriptor.atoms[2],
            descriptor.atoms[:2],
        )
        reflected_right = reflected_ranker.rank(
            descriptor.atoms[3],
            descriptor.atoms[4:],
        )
        if reflected_left.complete and reflected_right.complete:
            reflected_left_position = descriptor.atoms[:2].index(
                reflected_left.ordered_references[0]
            )
            reflected_right_position = descriptor.atoms[4:].index(
                reflected_right.ordered_references[0]
            )
            reflected_label = (
                "Z" if reflected_left_position == reflected_right_position else "E"
            )
            if reflected_label != label:
                label = label.lower()
    return _result(
        original,
        CIPAssignmentStatus.ASSIGNED,
        "Cis/trans label projected from the two higher-priority substituents.",
        label=label,
        rankings=rankings,
        witness=(left_position, right_position),
    )


def _axis_direction(
    ranker: CIPRanker,
    left_center: int,
    right_center: int,
    left: CIPRanking,
    right: CIPRanking,
) -> tuple[int | None, tuple[CIPComparison, ...]]:
    comparisons = []
    pairs = (
        (left.ordered_references[0], right.ordered_references[0]),
        (left.ordered_references[1], right.ordered_references[1]),
    )
    for left_reference, right_reference in pairs:
        left_evidence = ranker.build_evidence(left_center, left_reference)
        right_evidence = ranker.build_evidence(right_center, right_reference)
        comparison = ranker.compare(
            left_center,
            left_reference,
            right_reference,
            left_evidence=left_evidence,
            right_evidence=right_evidence,
        )
        comparisons.append(comparison)
        if comparison.outcome is CIPComparisonOutcome.LEFT_HIGHER:
            return 1, tuple(comparisons)
        if comparison.outcome is CIPComparisonOutcome.RIGHT_HIGHER:
            return -1, tuple(comparisons)
        if comparison.outcome is CIPComparisonOutcome.UNSUPPORTED:
            return None, tuple(comparisons)
    return None, tuple(comparisons)


def _assign_axis(
    original: AtropBondStereo | CumuleneAxisStereo,
    descriptor: AtropBondStereo | CumuleneAxisStereo,
    ranker: CIPRanker,
) -> CIPAssignment:
    if descriptor.parity is None:
        return _result(
            original,
            CIPAssignmentStatus.UNSPECIFIED_CONFIGURATION,
            "Axial orientation is unspecified.",
        )
    left_frame = tuple(descriptor.atoms[:2])
    right_frame = tuple(descriptor.atoms[4:])
    left_center, right_center = descriptor.atoms[2:4]
    left = ranker.rank(left_center, left_frame)
    right = ranker.rank(right_center, right_frame)
    rankings = left, right
    if not all(ranking.complete for ranking in rankings):
        return _incomplete_result(original, rankings)
    direction, direction_comparisons = _axis_direction(
        ranker, left_center, right_center, left, right
    )
    if direction is None:
        if any(
            comparison.outcome is CIPComparisonOutcome.UNSUPPORTED
            for comparison in direction_comparisons
        ):
            return _incomplete_result(
                original,
                rankings,
                supplemental=direction_comparisons,
            )
        return _result(
            original,
            CIPAssignmentStatus.UNRESOLVED_PRIORITY,
            "Constitution cannot witness a relabel-invariant axis direction.",
            rankings=rankings,
            supplemental=direction_comparisons,
        )
    left_factor = 1 if left_frame[0] == left.ordered_references[0] else -1
    right_factor = 1 if right_frame[0] == right.ordered_references[0] else -1
    sign = descriptor.parity * left_factor * right_factor * direction
    label = "P" if sign == 1 else "M"
    if CIPSequenceRule.RULE_5_REFLECTION_VARIANT in _required_rules(
        (*_comparisons(rankings), *direction_comparisons)
    ):
        reflected_ranker = ranker.reflected()
        reflected_left = reflected_ranker.rank(left_center, left_frame)
        reflected_right = reflected_ranker.rank(right_center, right_frame)
        if reflected_left.complete and reflected_right.complete:
            reflected_direction, _comparisons_reflected = _axis_direction(
                reflected_ranker,
                left_center,
                right_center,
                reflected_left,
                reflected_right,
            )
            if reflected_direction is not None:
                reflected_left_factor = (
                    1 if left_frame[0] == reflected_left.ordered_references[0] else -1
                )
                reflected_right_factor = (
                    1 if right_frame[0] == reflected_right.ordered_references[0] else -1
                )
                reflected_sign = (
                    -descriptor.parity
                    * reflected_left_factor
                    * reflected_right_factor
                    * reflected_direction
                )
                if reflected_sign == sign:
                    label = label.lower()
    return _result(
        original,
        CIPAssignmentStatus.ASSIGNED,
        "Axial helicity projected from ranked terminal frames.",
        label=label,
        rankings=rankings,
        supplemental=direction_comparisons,
        witness=(
            descriptor.parity,
            left_factor,
            right_factor,
            direction,
            sign,
        ),
    )


def _assign_helical(
    original: HelicalStereo,
    descriptor: HelicalStereo,
) -> CIPAssignment:
    if descriptor.parity is None:
        return _result(
            original,
            CIPAssignmentStatus.UNSPECIFIED_CONFIGURATION,
            "Helical orientation is unspecified.",
        )
    label = "P" if descriptor.parity == 1 else "M"
    return _result(
        original,
        CIPAssignmentStatus.ASSIGNED,
        "Helical label projected from the configured path handedness.",
        label=label,
        witness=(descriptor.parity,),
    )


def _assign_normalized(
    original: Any,
    descriptor: Any,
    ranker: CIPRanker,
) -> CIPAssignment:
    if isinstance(descriptor, TetrahedralStereo):
        return _assign_tetrahedral(original, descriptor, ranker)
    if isinstance(descriptor, (PlanarBondStereo, ExtendedCisTransStereo)):
        return _assign_planar(original, descriptor, ranker)
    if isinstance(descriptor, (AtropBondStereo, CumuleneAxisStereo)):
        return _assign_axis(original, descriptor, ranker)
    if isinstance(descriptor, HelicalStereo):
        return _assign_helical(original, descriptor)
    if isinstance(descriptor, _COORDINATION_TYPES):
        return _result(
            original,
            CIPAssignmentStatus.UNSUPPORTED_DESCRIPTOR,
            "Coordination-geometry configuration indices are not implemented.",
        )
    return _result(
        original,
        CIPAssignmentStatus.UNSUPPORTED_DESCRIPTOR,
        f"No CIP projection exists for {descriptor.descriptor_class!r}.",
    )


def assign_cip_labels(
    molecule: Chem.Mol,
    descriptors: Iterable[Any],
    *,
    reference_to_index: Mapping[int, int] | None = None,
) -> tuple[CIPAssignment, ...]:
    """Project labels for descriptors sharing one explicit reference space."""
    if molecule is None:
        raise ValueError("CIP assignment requires a molecule.")
    original = tuple(descriptors)
    normalized = tuple(
        _normalize_descriptor(molecule, descriptor, reference_to_index)
        for descriptor in original
    )

    def assign_all(labels: dict[str, str]) -> tuple[CIPAssignment, ...]:
        return tuple(
            _assign_normalized(
                source,
                target,
                CIPRanker(
                    molecule,
                    configured_descriptors=tuple(
                        candidate
                        for candidate_index, candidate in enumerate(normalized)
                        if candidate_index != index
                        and isinstance(candidate, _SUPPORTED_TYPES)
                    ),
                    configured_labels=labels,
                ),
            )
            for index, (source, target) in enumerate(zip(original, normalized))
        )

    labels: dict[str, str] = {}
    results = assign_all(labels)
    for _iteration in range(len(normalized)):
        updated = {
            descriptor_id(descriptor): assignment.label
            for descriptor, assignment in zip(normalized, results)
            if assignment.assigned and assignment.label is not None
        }
        if updated == labels:
            break
        labels = updated
        results = assign_all(labels)
    return results


def assign_cip_label(
    molecule: Chem.Mol,
    descriptor: Any,
    *,
    reference_to_index: Mapping[int, int] | None = None,
    configured_descriptors: Iterable[Any] = (),
) -> CIPAssignment:
    """Project one label with optional configured stereo witnesses."""
    companions = tuple(configured_descriptors)
    values = (descriptor,) + companions
    return assign_cip_labels(
        molecule,
        values,
        reference_to_index=reference_to_index,
    )[0]


__all__ = [
    "CIPAssignment",
    "CIPAssignmentStatus",
    "assign_cip_label",
    "assign_cip_labels",
]
