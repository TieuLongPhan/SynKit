"""Exact pairwise relations between configured molecular stereographs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import networkx as nx

from .canonical import (
    AttributeSelector,
    _DEFAULT_ATOM_COLOR_KEYS,
    _DEFAULT_BOND_COLOR_KEYS,
    _selected,
    _rdkit_graph_and_registry,
    canonicalize_stereo_registry,
    mirror_stereo_descriptor,
)
from .configured import (
    CONFIGURED_DESCRIPTOR_TYPES,
    MirrorIdentityProfile,
    _is_terminal_phosphate_resonance_locus,
    _mirror_profile_options,
    _resonance_graphs,
)
from .descriptors import StereoValue
from .global_stereo import FrameworkStereo
from .orbits import StereoSpecification
from .rdkit_identity import RDKitIdentityResult, canonicalize_rdkit_registry

STEREOISOMER_RELATION_SCHEMA = "synkit.stereoisomer-relation/2"


class StereoisomerRelation(str, Enum):
    """Mutually exclusive outcomes of an exact pairwise comparison."""

    IDENTICAL = "identical"
    ENANTIOMERS = "enantiomers"
    DIASTEREOMERS = "diastereomers"
    CONSTITUTIONALLY_DIFFERENT = "constitutionally_different"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    INFORMATION_MISMATCH = "information_mismatch"


@dataclass(frozen=True)
class StereoisomerRelationResult:
    """Certificates and diagnostics supporting one pairwise relation."""

    relation: StereoisomerRelation
    left_constitution: RDKitIdentityResult
    right_constitution: RDKitIdentityResult
    left: RDKitIdentityResult | None = None
    right: RDKitIdentityResult | None = None
    mirror_left: RDKitIdentityResult | None = None
    incomplete_loci: tuple[str, ...] = ()
    unsupported_loci: tuple[str, ...] = ()
    unsupported_families: tuple[str, ...] = ()
    schema: str = STEREOISOMER_RELATION_SCHEMA
    method: str = "exact_configured_stereograph_pair_comparison"
    identity_profile: str = "attributed_graph"

    @property
    def is_definitive(self) -> bool:
        return self.relation not in {
            StereoisomerRelation.INCOMPLETE,
            StereoisomerRelation.UNSUPPORTED,
            StereoisomerRelation.INFORMATION_MISMATCH,
        }


def _qualified(side: str, loci: Iterable[str]) -> tuple[str, ...]:
    return tuple(f"{side}:{locus}" for locus in loci)


def _inventory_key(descriptor):
    if isinstance(descriptor, FrameworkStereo):
        # Preserve relative frame coupling while omitting overall orientation.
        return min(
            (replace(descriptor, orientation=sign).canonical_form() for sign in (-1, 1)),
            key=repr,
        )
    return replace(descriptor, parity=None).canonical_form()


def _inventories_match(left_graph, left_registry, right_graph, right_registry, **options):
    """Require a constitutional map carrying the entire unconfigured inventory."""
    if Counter(value.descriptor_class for value in left_registry.values()) != Counter(
        value.descriptor_class for value in right_registry.values()
    ):
        return False
    target = Counter(_inventory_key(value) for value in right_registry.values())
    atom_color = options.get("atom_color", _DEFAULT_ATOM_COLOR_KEYS)
    bond_color = options.get("bond_color", _DEFAULT_BOND_COLOR_KEYS)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        left_graph, right_graph,
        node_match=lambda left, right: _selected(left, atom_color) == _selected(right, atom_color),
        edge_match=lambda left, right: _selected(left, bond_color) == _selected(right, bond_color),
    )
    return any(
        Counter(_inventory_key(value.relabel(mapping)) for value in left_registry.values()) == target
        for mapping in matcher.isomorphisms_iter()
    )


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
    atom_color: AttributeSelector = _DEFAULT_ATOM_COLOR_KEYS,
    bond_color: AttributeSelector = _DEFAULT_BOND_COLOR_KEYS,
) -> StereoisomerRelationResult:
    """Classify two configured structures by exact certificate comparison.

    Constitution is compared first.  For equal constitutions the complete
    stereographs are compared directly and after mirroring every
    mirror-sensitive descriptor on the left.  A remaining difference is
    diastereomeric only when the unconfigured carrier inventories correspond.
    A different inventory returns ``information_mismatch``. No CIP ranking or
    local R/S or E/Z text label enters the
    certificates.
    """
    options = {"atom_color": atom_color, "bond_color": bond_color}
    left_constitution = canonicalize_stereo_registry(left_graph, {}, **options)
    right_constitution = canonicalize_stereo_registry(right_graph, {}, **options)
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

    left = canonicalize_stereo_registry(left_graph, left_registry, **options)
    right = canonicalize_stereo_registry(right_graph, right_registry, **options)
    if left.same_stereograph(right):
        return StereoisomerRelationResult(
            StereoisomerRelation.IDENTICAL,
            left_constitution,
            right_constitution,
            left,
            right,
        )

    mirrored_registry = {
        key: mirror_stereo_descriptor(value) for key, value in left_registry.items()
    }
    mirror_left = canonicalize_stereo_registry(left_graph, mirrored_registry, **options)
    relation = (
        StereoisomerRelation.ENANTIOMERS
        if mirror_left.same_stereograph(right)
        else (
            StereoisomerRelation.DIASTEREOMERS
            if _inventories_match(left_graph, left_registry, right_graph, right_registry, **options)
            else StereoisomerRelation.INFORMATION_MISMATCH
        )
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
    *,
    identity_profile: MirrorIdentityProfile = "lewis_state",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from synkit.Chem.Molecule.stereo_perception import (
        StereoConfigurationState,
        detect_potential_stereo_elements,
    )

    incomplete = tuple(
        element.identifier
        for element in detect_potential_stereo_elements(molecule)
        if element.configuration_state is StereoConfigurationState.UNSPECIFIED
        and not _is_terminal_phosphate_resonance_locus(
            molecule, element, identity_profile=identity_profile,
        )
    )
    unsupported = tuple(
        f"enhanced_stereo_group:{index}:{group.GetGroupType()}"
        for index, group in enumerate(molecule.GetStereoGroups())
    )
    return incomplete, unsupported


def classify_rdkit_stereoisomer_relation(
    left_molecule: Any,
    right_molecule: Any,
    *,
    identity_profile: MirrorIdentityProfile = "lewis_state",
) -> StereoisomerRelationResult:
    """Classify strict RDKit inputs under the declared common identity profile.

    The compatibility default is the supplied Lewis drawing. Chemical
    comparison uses full resonance-family keys, with drawing-specific witnesses
    retained in the family members. Mirror classification defaults to chemical;
    select the same profile when comparing different public operations.
    """
    options = _mirror_profile_options(identity_profile)
    left_graph, left_registry = _rdkit_graph_and_registry(left_molecule)
    right_graph, right_registry = _rdkit_graph_and_registry(right_molecule)
    left_incomplete, left_unsupported = _rdkit_information_state(
        left_molecule, identity_profile=identity_profile,
    )
    right_incomplete, right_unsupported = _rdkit_information_state(
        right_molecule, identity_profile=identity_profile,
    )
    if identity_profile == "chemical":
        return _classify_rdkit_chemical_pair(
            left_molecule, left_graph, left_registry, left_incomplete, left_unsupported,
            right_molecule, right_graph, right_registry, right_incomplete, right_unsupported,
        )
    result = classify_stereoisomer_relation(
        left_graph,
        left_registry,
        right_graph,
        right_registry,
        left_incomplete_loci=left_incomplete,
        right_incomplete_loci=right_incomplete,
        left_unsupported_loci=left_unsupported,
        right_unsupported_loci=right_unsupported,
        **options,
    )
    from .rdkit_identity import profile_certificate

    return replace(
        result, identity_profile=identity_profile,
        **{
            name: profile_certificate(value, identity_profile)
            for name in ("left_constitution", "right_constitution", "left", "right", "mirror_left")
            if (value := getattr(result, name)) is not None
        },
    )


def _classify_rdkit_chemical_pair(lmol, lg, lr, li, lu, rmol, rg, rr, ri, ru):
    lc = canonicalize_rdkit_registry(lmol, {}, identity_profile="chemical")
    rc = canonicalize_rdkit_registry(rmol, {}, identity_profile="chemical")

    def result(relation, **evidence):
        return StereoisomerRelationResult(
            relation, lc, rc, identity_profile="chemical", **evidence,
        )

    if not lc.same_stereograph(rc):
        return result(StereoisomerRelation.CONSTITUTIONALLY_DIFFERENT)
    if lu or ru:
        return result(
            StereoisomerRelation.UNSUPPORTED,
            unsupported_loci=(*_qualified("left", lu), *_qualified("right", ru)),
        )
    if li or ri:
        return result(
            StereoisomerRelation.INCOMPLETE,
            incomplete_loci=(*_qualified("left", li), *_qualified("right", ri)),
        )
    left = canonicalize_rdkit_registry(lmol, lr, identity_profile="chemical")
    right = canonicalize_rdkit_registry(rmol, rr, identity_profile="chemical")
    if left.same_stereograph(right):
        return result(StereoisomerRelation.IDENTICAL, left=left, right=right)
    mirrored = {key: mirror_stereo_descriptor(value) for key, value in lr.items()}
    mirror = canonicalize_rdkit_registry(lmol, mirrored, identity_profile="chemical")
    if mirror.same_stereograph(right):
        relation = StereoisomerRelation.ENANTIOMERS
    elif any(
        _inventories_match(lgraph, lr, rgraph, rr)
        for lgraph in _resonance_graphs(lmol)
        for rgraph in _resonance_graphs(rmol)
    ):
        relation = StereoisomerRelation.DIASTEREOMERS
    else:
        relation = StereoisomerRelation.INFORMATION_MISMATCH
    return result(relation, left=left, right=right, mirror_left=mirror)


__all__ = [
    "STEREOISOMER_RELATION_SCHEMA",
    "StereoisomerRelation",
    "StereoisomerRelationResult",
    "classify_rdkit_stereoisomer_relation",
    "classify_stereoisomer_relation",
]
