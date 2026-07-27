"""Exact pairwise relations between configured molecular stereographs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx

from .canonical import (
    CanonicalStereographResult,
    _rdkit_graph_and_registry,
    canonicalize_stereo_registry,
    mirror_stereo_descriptor,
)
from .configured import (
    CONFIGURED_DESCRIPTOR_TYPES,
)
from .descriptors import StereoValue
from .orbits import StereoSpecification

STEREOISOMER_RELATION_SCHEMA = "synkit.stereoisomer-relation/1"


class StereoisomerRelation(str, Enum):
    """Mutually exclusive outcomes of an exact pairwise comparison."""

    IDENTICAL = "identical"
    ENANTIOMERS = "enantiomers"
    DIASTEREOMERS = "diastereomers"
    CONSTITUTIONALLY_DIFFERENT = "constitutionally_different"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class StereoisomerRelationResult:
    """Certificates and diagnostics supporting one pairwise relation."""

    relation: StereoisomerRelation
    left_constitution: CanonicalStereographResult
    right_constitution: CanonicalStereographResult
    left: CanonicalStereographResult | None = None
    right: CanonicalStereographResult | None = None
    mirror_left: CanonicalStereographResult | None = None
    incomplete_loci: tuple[str, ...] = ()
    unsupported_loci: tuple[str, ...] = ()
    unsupported_families: tuple[str, ...] = ()
    schema: str = STEREOISOMER_RELATION_SCHEMA
    method: str = "exact_configured_stereograph_pair_comparison"

    @property
    def is_definitive(self) -> bool:
        return self.relation not in {
            StereoisomerRelation.INCOMPLETE,
            StereoisomerRelation.UNSUPPORTED,
        }


def _qualified(side: str, loci: Iterable[str]) -> tuple[str, ...]:
    return tuple(f"{side}:{locus}" for locus in loci)


def classify_stereoisomer_relation(
    left_graph: nx.Graph,
    left_registry: Mapping[str, StereoValue],
    right_graph: nx.Graph,
    right_registry: Mapping[str, StereoValue],
    *,
    left_incomplete_loci: Iterable[str] = (),
    right_incomplete_loci: Iterable[str] = (),
    left_unsupported_loci: Iterable[str] = (),
    right_unsupported_loci: Iterable[str] = (),
) -> StereoisomerRelationResult:
    """Classify two configured structures by exact certificate comparison.

    Constitution is compared first.  For equal constitutions the complete
    stereographs are compared directly and after mirroring every
    mirror-sensitive descriptor on the left.  A remaining difference is
    diastereomeric.  No CIP ranking or local R/S or E/Z text label enters the
    certificates.
    """
    left_constitution = canonicalize_stereo_registry(left_graph, {})
    right_constitution = canonicalize_stereo_registry(right_graph, {})
    if not left_constitution.same_stereograph(right_constitution):
        return StereoisomerRelationResult(
            StereoisomerRelation.CONSTITUTIONALLY_DIFFERENT,
            left_constitution,
            right_constitution,
        )

    entries = (
        ("left", left_registry),
        ("right", right_registry),
    )
    unsupported_entries = tuple(
        (side, key, value.descriptor_class)
        for side, registry in entries
        for key, value in registry.items()
        if not isinstance(value, CONFIGURED_DESCRIPTOR_TYPES)
    )
    declared_unsupported = (
        *_qualified("left", left_unsupported_loci),
        *_qualified("right", right_unsupported_loci),
    )
    if unsupported_entries or declared_unsupported:
        return StereoisomerRelationResult(
            StereoisomerRelation.UNSUPPORTED,
            left_constitution,
            right_constitution,
            unsupported_loci=tuple(
                sorted(
                    {
                        *declared_unsupported,
                        *(
                            f"{side}:{key}"
                            for side, key, _family in unsupported_entries
                        ),
                    }
                )
            ),
            unsupported_families=tuple(
                sorted({family for _side, _key, family in unsupported_entries})
            ),
        )

    unknown = tuple(
        f"{side}:{key}"
        for side, registry in entries
        for key, value in registry.items()
        if value.specification is StereoSpecification.UNSPECIFIED
    )
    declared_incomplete = (
        *_qualified("left", left_incomplete_loci),
        *_qualified("right", right_incomplete_loci),
    )
    if unknown or declared_incomplete:
        return StereoisomerRelationResult(
            StereoisomerRelation.INCOMPLETE,
            left_constitution,
            right_constitution,
            incomplete_loci=tuple(sorted({*unknown, *declared_incomplete})),
        )

    left = canonicalize_stereo_registry(left_graph, left_registry)
    right = canonicalize_stereo_registry(right_graph, right_registry)
    if left.same_stereograph(right):
        return StereoisomerRelationResult(
            StereoisomerRelation.IDENTICAL,
            left_constitution,
            right_constitution,
            left,
            right,
        )

    mirrored_registry = {
        key: mirror_stereo_descriptor(value)
        for key, value in left_registry.items()
    }
    mirror_left = canonicalize_stereo_registry(left_graph, mirrored_registry)
    relation = (
        StereoisomerRelation.ENANTIOMERS
        if mirror_left.same_stereograph(right)
        else StereoisomerRelation.DIASTEREOMERS
    )
    return StereoisomerRelationResult(
        relation,
        left_constitution,
        right_constitution,
        left,
        right,
        mirror_left,
    )


def _rdkit_information_state(
    molecule: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from synkit.Chem.Molecule.stereo_perception import (
        StereoConfigurationState,
        detect_potential_stereo_elements,
    )

    incomplete = tuple(
        element.identifier
        for element in detect_potential_stereo_elements(molecule)
        if element.configuration_state is StereoConfigurationState.UNSPECIFIED
    )
    unsupported = tuple(
        f"enhanced_stereo_group:{index}:{group.GetGroupType()}"
        for index, group in enumerate(molecule.GetStereoGroups())
    )
    return incomplete, unsupported


def classify_rdkit_stereoisomer_relation(
    left_molecule: Any,
    right_molecule: Any,
) -> StereoisomerRelationResult:
    """Classify two RDKit molecules without converting stereo to CIP labels."""
    left_graph, left_registry = _rdkit_graph_and_registry(left_molecule)
    right_graph, right_registry = _rdkit_graph_and_registry(right_molecule)
    left_incomplete, left_unsupported = _rdkit_information_state(left_molecule)
    right_incomplete, right_unsupported = _rdkit_information_state(right_molecule)
    return classify_stereoisomer_relation(
        left_graph,
        left_registry,
        right_graph,
        right_registry,
        left_incomplete_loci=left_incomplete,
        right_incomplete_loci=right_incomplete,
        left_unsupported_loci=left_unsupported,
        right_unsupported_loci=right_unsupported,
    )


__all__ = [
    "STEREOISOMER_RELATION_SCHEMA",
    "StereoisomerRelation",
    "StereoisomerRelationResult",
    "classify_rdkit_stereoisomer_relation",
    "classify_stereoisomer_relation",
]
