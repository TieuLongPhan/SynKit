"""Descriptor registry comparison for molecules, ITS graphs, and rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx

from .descriptors import (
    OctahedralStereo,
    SquarePlanarStereo,
    StereoValue,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    descriptor_id,
    parse_virtual_reference,
    virtual_reference,
)
from .orbits import StereoRelation, StereoRelationKind

_ATOM_STEREO_TYPES = (
    TetrahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
)


class NonInvertibleStereoEffectError(ValueError):
    """Raised when reverse-rule construction would invent stereochemistry.

    A known-to-unknown ``UNSPECIFIED`` endpoint relation is not one-to-one: a
    specified reactant descriptor can map to an unknown product descriptor,
    but that unknown state cannot identify which descriptor to recreate.
    """

    reason = "non_reversible_unspecified_descriptor"

    def __init__(self, targets: Iterable[str] = ()) -> None:
        self.targets = tuple(sorted(set(targets)))
        detail = f"; targets={list(self.targets)!r}" if self.targets else ""
        super().__init__(f"{self.reason}{detail}")


class StereoAlignmentError(ValueError):
    """Raised when a reaction effect has no unambiguous reference alignment."""

    def __init__(self, issue_code: str, detail: str = "") -> None:
        self.issue_code = issue_code
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{issue_code}{suffix}")


def _peripheral_references(descriptor: StereoValue) -> tuple[Any, ...]:
    if isinstance(descriptor, _ATOM_STEREO_TYPES):
        return descriptor.atoms[1:]
    return (*descriptor.atoms[:2], *descriptor.atoms[4:])


def _mapping_sort_key(pair: tuple[Any, Any]) -> tuple[str, str, str, str]:
    return type(pair[0]).__name__, repr(pair[0]), type(pair[1]).__name__, repr(pair[1])


@dataclass(frozen=True)
class StereoReferenceAlignment:
    """Reference replacement evidence for one reaction-local stereo frame."""

    mapping: tuple[tuple[Any, Any], ...] = ()
    status: str = "identity"
    removed: tuple[Any, ...] = ()
    added: tuple[Any, ...] = ()
    issue_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        mapping = tuple(
            sorted(
                (tuple(pair) for pair in self.mapping),
                key=_mapping_sort_key,
            )
        )
        object.__setattr__(self, "mapping", mapping)
        if self.status not in {"identity", "inferred", "explicit", "refused"}:
            raise ValueError(f"Unsupported stereo alignment status: {self.status!r}.")
        if self.status == "refused" and self.issue_code is None:
            raise ValueError("A refused stereo alignment requires an issue code.")
        if self.status != "refused" and self.issue_code is not None:
            raise ValueError("Only a refused stereo alignment may carry an issue code.")
        sources = [source for source, _ in mapping]
        targets = [target for _, target in mapping]
        if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
            raise ValueError("A stereo reference mapping must be bijective.")

    @property
    def accepted(self) -> bool:
        return self.status != "refused"

    def reverse(self) -> "StereoReferenceAlignment":
        if not self.accepted:
            return StereoReferenceAlignment(
                (),
                "refused",
                self.added,
                self.removed,
                self.issue_code,
                self.detail,
            )
        status = "identity" if not self.mapping else "explicit"
        return StereoReferenceAlignment(
            tuple((target, source) for source, target in self.mapping),
            status,
            self.added,
            self.removed,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "mapping": [list(pair) for pair in self.mapping],
            "removed": list(self.removed),
            "added": list(self.added),
        }
        if self.issue_code is not None:
            result["issue_code"] = self.issue_code
        if self.detail is not None:
            result["detail"] = self.detail
        return result


def _reference_delta(
    old: StereoValue,
    new: StereoValue,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    old_refs = _peripheral_references(old)
    new_refs = _peripheral_references(new)
    removed = tuple(reference for reference in old_refs if reference not in new_refs)
    added = tuple(reference for reference in new_refs if reference not in old_refs)
    return removed, added


def align_stereo_references(
    old: StereoValue,
    new: StereoValue,
    reference_mapping: Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None = None,
) -> StereoReferenceAlignment:
    """Align endpoint frames, inferring only a unique one-for-one replacement."""
    if type(old) is not type(new) or descriptor_id(old) != descriptor_id(new):
        return StereoReferenceAlignment(
            (),
            "refused",
            issue_code="STEREO_ALIGNMENT_INCOMPATIBLE_LOCUS",
            detail="Endpoint descriptors do not share a shape and locus.",
        )

    removed, added = _reference_delta(old, new)
    if reference_mapping is not None:
        raw_items = (
            reference_mapping.items()
            if isinstance(reference_mapping, Mapping)
            else reference_mapping
        )
        mapping = tuple(raw_items)
        try:
            candidate = StereoReferenceAlignment(
                mapping,
                "explicit" if mapping else "identity",
                removed,
                added,
            )
        except (TypeError, ValueError) as exc:
            return StereoReferenceAlignment(
                (),
                "refused",
                removed,
                added,
                "STEREO_ALIGNMENT_INVALID_EXPLICIT_MAP",
                str(exc),
            )
        if set(source for source, _ in candidate.mapping) != set(removed) or set(
            target for _, target in candidate.mapping
        ) != set(added):
            return StereoReferenceAlignment(
                (),
                "refused",
                removed,
                added,
                "STEREO_ALIGNMENT_INVALID_EXPLICIT_MAP",
                "The explicit map must pair every removed reference with every added reference.",
            )
        return candidate

    if not removed and not added:
        return StereoReferenceAlignment((), "identity")
    if len(removed) == len(added) == 1:
        return StereoReferenceAlignment(
            ((removed[0], added[0]),),
            "inferred",
            removed,
            added,
        )
    issue = (
        "STEREO_ALIGNMENT_AMBIGUOUS"
        if len(removed) == len(added) and len(removed) > 1
        else "STEREO_ALIGNMENT_INCOMPATIBLE_REFERENCES"
    )
    return StereoReferenceAlignment(
        (),
        "refused",
        removed,
        added,
        issue,
        (
            "Multiple replacements require an explicit bijection."
            if issue == "STEREO_ALIGNMENT_AMBIGUOUS"
            else "Removed and added reference populations differ."
        ),
    )


def _relation_after_alignment(
    old: StereoValue,
    new: StereoValue,
    alignment: StereoReferenceAlignment,
) -> StereoRelation | None:
    if not alignment.accepted:
        return None
    aligned = old.replace_references(dict(alignment.mapping))
    return aligned.relation_to(new)


def _change_from_relation(
    old: StereoValue | None,
    new: StereoValue | None,
    transition: StereoValue | None,
    relation: StereoRelation | None,
) -> str:
    if old is None and new is None:
        return "FLEETING" if transition is not None else "UNSPECIFIED"
    if old is None:
        return "FORMED"
    if new is None:
        return "BROKEN"
    if relation is None:
        return "UNSPECIFIED"
    if relation.kind is StereoRelationKind.EQUIVALENT:
        return "RETAINED"
    if relation.kind is StereoRelationKind.OPPOSITE:
        return "INVERTED"
    if relation.kind is StereoRelationKind.RECONFIGURED:
        # Beta-2 projected every descriptor-level TBP/octahedral inversion onto
        # the binary INVERTED label. Keep that public string when applicable,
        # while ``relation`` remains the authoritative non-binary class.
        from .legacy import legacy_classify_stereo_change

        legacy_projection = legacy_classify_stereo_change(old, new, transition)
        if legacy_projection in {"RETAINED", "INVERTED"}:
            return legacy_projection
    return "UNSPECIFIED"


def _descriptor_from_configuration(
    prototype: StereoValue,
    frame: tuple[Any, ...],
    *,
    unspecified: bool,
) -> StereoValue:
    parity = (
        None
        if unspecified
        else (
            0 if prototype.descriptor_class in {"square_planar", "planar_bond"} else 1
        )
    )
    return type(prototype)(frame, parity, prototype.provenance)


@dataclass(frozen=True)
class StereoChange:
    """One rule-local stereochemical state transition.

    ``before`` and ``after`` are the stable molecular states. ``transition``
    is optional ITS/transition-state stereo and is deliberately separate from
    either endpoint.  The textual ``change`` remains a derived, readable
    classification for compatibility with existing SynKit rule consumers.
    """

    change: str
    before: StereoValue | None
    after: StereoValue | None
    transition: StereoValue | None = None
    relation: StereoRelation | None = None
    reference_mapping: tuple[tuple[Any, Any], ...] | Mapping[Any, Any] | None = None
    alignment: StereoReferenceAlignment | None = None

    def __post_init__(self) -> None:
        if self.before is None or self.after is None:
            alignment = self.alignment or StereoReferenceAlignment((), "identity")
            relation = self.relation
        else:
            alignment = self.alignment or align_stereo_references(
                self.before,
                self.after,
                self.reference_mapping,
            )
            relation = self.relation or _relation_after_alignment(
                self.before,
                self.after,
                alignment,
            )
        object.__setattr__(self, "alignment", alignment)
        object.__setattr__(self, "reference_mapping", alignment.mapping)
        object.__setattr__(self, "relation", relation)

    @classmethod
    def from_endpoints(
        cls,
        before: StereoValue | None,
        after: StereoValue | None,
        transition: StereoValue | None = None,
        *,
        reference_mapping: Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None = None,
    ) -> "StereoChange":
        alignment = (
            align_stereo_references(before, after, reference_mapping)
            if before is not None and after is not None
            else StereoReferenceAlignment((), "identity")
        )
        relation = (
            _relation_after_alignment(before, after, alignment)
            if before is not None and after is not None
            else None
        )
        return cls(
            _change_from_relation(before, after, transition, relation),
            before,
            after,
            transition,
            relation,
            alignment.mapping,
            alignment,
        )

    @property
    def dependencies(self) -> frozenset[int]:
        values = set()
        if self.before:
            values.update(self.before.dependencies)
        if self.after:
            values.update(self.after.dependencies)
        if self.transition:
            values.update(self.transition.dependencies)
        return frozenset(values)

    @property
    def fleeting(self) -> StereoValue | None:
        """Compatibility term for stereo present only at the transition state."""
        return self.transition

    @property
    def evidence_kind(self) -> str:
        """Return the rich relation label behind the compatibility projection."""
        if self.before is None and self.after is None:
            return "fleeting" if self.transition is not None else "unspecified"
        if self.before is None:
            return "formed"
        if self.after is None:
            return "broken"
        if self.relation is None:
            return "alignment_refused"
        if self.relation.kind is StereoRelationKind.RECONFIGURED:
            encoded = "-".join(map(str, self.relation.class_id or ()))
            return f"reconfigured:{encoded}"
        return self.relation.kind.value

    def relation_evidence(self) -> dict[str, Any]:
        """Return deterministic, machine-readable reaction relation evidence."""
        result: dict[str, Any] = {
            "kind": self.evidence_kind,
            "alignment": self.alignment.to_dict(),
        }
        if self.relation is not None:
            relation: dict[str, Any] = {
                "kind": self.relation.kind.value,
                "shape": self.relation.shape,
                "class_id": (
                    list(self.relation.class_id)
                    if self.relation.class_id is not None
                    else None
                ),
            }
            if self.relation.witness is not None:
                relation["witness"] = list(self.relation.witness.permutation.image)
            result["relation"] = relation
        return result

    @property
    def non_invertible(self) -> bool:
        """Whether reversing this change would fabricate endpoint parity."""
        return (
            self.change == "UNSPECIFIED"
            and self.before is not None
            and self.after is not None
            and self.before.parity is not None
            and self.after.parity is None
        )

    def reverse(
        self,
        *,
        semantics: str = "orbit",
        diagnostics: list[Any] | None = None,
    ) -> "StereoChange":
        """Return the same state transition in the opposite direction."""
        from .legacy import (
            StereoSemanticComparison,
            StereoSemanticsMode,
            legacy_classify_stereo_change,
        )

        if self.non_invertible:
            raise NonInvertibleStereoEffectError()
        mode = StereoSemanticsMode(semantics)
        reverse_mapping = (
            {target: source for source, target in self.reference_mapping}
            if self.alignment.status == "explicit"
            else None
        )
        orbit_change = StereoChange.from_endpoints(
            self.after,
            self.before,
            self.transition,
            reference_mapping=reverse_mapping,
        )
        if mode is StereoSemanticsMode.ORBIT:
            return orbit_change
        legacy_change = legacy_classify_stereo_change(
            self.after,
            self.before,
            self.transition,
        )
        if mode is StereoSemanticsMode.COMPARE and diagnostics is not None:
            diagnostics.append(
                StereoSemanticComparison.create(
                    "reaction_stereo_reverse",
                    orbit_change.change,
                    legacy_change,
                )
            )
        if mode is StereoSemanticsMode.LEGACY:
            return StereoChange(
                legacy_change,
                orbit_change.before,
                orbit_change.after,
                orbit_change.transition,
                orbit_change.relation,
                orbit_change.reference_mapping,
                orbit_change.alignment,
            )
        return orbit_change

    def _apply_orbit_to(self, descriptor: StereoValue) -> StereoValue | None:
        if self.before is None:
            return self.after
        if self.after is None:
            return None
        if type(descriptor) is not type(self.before) or descriptor_id(descriptor) != (
            descriptor_id(self.before)
        ):
            raise StereoAlignmentError(
                "STEREO_ALIGNMENT_INCOMPATIBLE_SUBSTRATE",
                "The substrate descriptor does not share the rule effect locus.",
            )
        if not self.alignment.accepted:
            raise StereoAlignmentError(
                self.alignment.issue_code or "STEREO_ALIGNMENT_REFUSED",
                self.alignment.detail or "",
            )
        if self.relation is None or self.relation.witness is None:
            raise StereoAlignmentError(
                "STEREO_RELATION_NOT_REPLAYABLE",
                "The rule effect has no concrete permutation witness.",
            )
        aligned = descriptor.replace_references(dict(self.reference_mapping))
        frame = self.relation.witness.apply(aligned.configuration.frame)
        unspecified = (
            descriptor.parity is None
            or self.after.parity is None
            or self.relation.kind is StereoRelationKind.UNSPECIFIED
        )
        return _descriptor_from_configuration(
            self.after,
            frame,
            unspecified=unspecified,
        )

    def apply_to(
        self,
        descriptor: StereoValue,
        *,
        semantics: str = "orbit",
        diagnostics: list[Any] | None = None,
    ) -> StereoValue | None:
        """Apply this relation, optionally auditing the frozen Beta-2 result."""
        from .legacy import (
            StereoSemanticComparison,
            StereoSemanticsMode,
            legacy_apply_stereo_change,
        )

        mode = StereoSemanticsMode(semantics)
        if mode is StereoSemanticsMode.LEGACY:
            return legacy_apply_stereo_change(self, descriptor)
        orbit_result = self._apply_orbit_to(descriptor)
        if mode is StereoSemanticsMode.COMPARE:
            legacy_result = legacy_apply_stereo_change(self, descriptor)
            expected_divergence = None
            if (
                self.relation is not None
                and self.relation.kind is StereoRelationKind.RECONFIGURED
                and self.relation.shape in {"trigonal_bipyramidal", "octahedral"}
            ):
                expected_divergence = "nonbinary_orbit_reconfiguration"
            if diagnostics is not None:
                diagnostics.append(
                    StereoSemanticComparison.create(
                        "reaction_stereo_application",
                        orbit_result,
                        legacy_result,
                        expected_divergence=expected_divergence,
                    )
                )
        return orbit_result

    def relabel(self, mapping: Mapping[int, int]) -> "StereoChange":
        """Relabel endpoints and the explicit reference alignment together."""

        def relabel_reference(reference: Any) -> Any:
            if type(reference) is int:
                return mapping.get(reference, reference)
            virtual = parse_virtual_reference(reference)
            if virtual is None:
                return reference
            return virtual_reference(
                virtual.kind,
                mapping.get(virtual.center, virtual.center),
            )

        translated_reference_mapping = (
            {
                relabel_reference(source): relabel_reference(target)
                for source, target in self.reference_mapping
            }
            if self.alignment.status == "explicit"
            else None
        )
        return StereoChange.from_endpoints(
            self.before.relabel(mapping) if self.before is not None else None,
            self.after.relabel(mapping) if self.after is not None else None,
            self.transition.relabel(mapping) if self.transition is not None else None,
            reference_mapping=translated_reference_mapping,
        )


def classify_stereo_change(
    old: StereoValue | None,
    new: StereoValue | None,
    transition: StereoValue | None = None,
    *,
    reference_mapping: Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None = None,
    semantics: str = "orbit",
    diagnostics: list[Any] | None = None,
) -> str:
    """Derive the compatibility label from an aligned orbit relation."""
    from .legacy import (
        StereoSemanticComparison,
        StereoSemanticsMode,
        legacy_classify_stereo_change,
    )

    mode = StereoSemanticsMode(semantics)
    legacy_result = None
    if mode is not StereoSemanticsMode.ORBIT:
        legacy_result = legacy_classify_stereo_change(old, new, transition)
        if mode is StereoSemanticsMode.LEGACY:
            return legacy_result
    change = StereoChange.from_endpoints(
        old,
        new,
        transition,
        reference_mapping=reference_mapping,
    )
    if mode is StereoSemanticsMode.COMPARE and diagnostics is not None:
        diagnostics.append(
            StereoSemanticComparison.create(
                "reaction_stereo_change",
                change.change,
                legacy_result,
            )
        )
        if (
            change.relation is not None
            and change.relation.kind is StereoRelationKind.RECONFIGURED
            and change.relation.shape in {"trigonal_bipyramidal", "octahedral"}
        ):
            diagnostics.append(
                StereoSemanticComparison.create(
                    "reaction_stereo_relation",
                    change.evidence_kind,
                    str(legacy_result).lower(),
                    expected_divergence="nonbinary_orbit_reconfiguration",
                )
            )
    return change.change


def stereo_registry(graph: nx.Graph) -> Mapping[str, StereoValue]:
    return graph.graph.get("stereo_descriptors", {})


def compare_stereo_registries(
    before: nx.Graph,
    after: nx.Graph,
    transition: nx.Graph | None = None,
    *,
    reference_mappings: (
        Mapping[
            str,
            Mapping[Any, Any] | Iterable[tuple[Any, Any]],
        ]
        | None
    ) = None,
    semantics: str = "orbit",
    diagnostics: list[Any] | None = None,
) -> dict[str, StereoChange]:
    left, right = stereo_registry(before), stereo_registry(after)
    middle = stereo_registry(transition) if transition is not None else {}
    changes: dict[str, StereoChange] = {}
    for key in sorted(set(left) | set(right) | set(middle)):
        old, new, transient = left.get(key), right.get(key), middle.get(key)
        reference_mapping = (
            None if reference_mappings is None else reference_mappings.get(key)
        )
        change = StereoChange.from_endpoints(
            old,
            new,
            transient,
            reference_mapping=reference_mapping,
        )
        if semantics != "orbit":
            classify_stereo_change(
                old,
                new,
                transient,
                reference_mapping=reference_mapping,
                semantics=semantics,
                diagnostics=diagnostics,
            )
        changes[key] = change
    return changes


def annotate_its_stereo(
    its: nx.Graph,
    before: nx.Graph,
    after: nx.Graph,
    transition: nx.Graph | None = None,
    *,
    reference_mappings: (
        Mapping[
            str,
            Mapping[Any, Any] | Iterable[tuple[Any, Any]],
        ]
        | None
    ) = None,
) -> None:
    changes = compare_stereo_registries(
        before,
        after,
        transition,
        reference_mappings=reference_mappings,
    )
    its.graph["stereo_changes"] = changes
    its.graph["stereo_descriptors"] = {
        "reactant": dict(stereo_registry(before)),
        "product": dict(stereo_registry(after)),
    }
    if transition is not None:
        its.graph["stereo_descriptors"]["transition"] = dict(
            stereo_registry(transition)
        )


def stereo_complete_reaction_center_nodes(its: nx.Graph) -> set[Any]:
    changes = its.graph.get("stereo_changes", {})
    if not changes:
        return set()
    atom_map_nodes = {}
    for node, attrs in its.nodes(data=True):
        atom_map = attrs.get("atom_map", node)
        if isinstance(atom_map, int):
            atom_map_nodes[atom_map] = node
        elif isinstance(atom_map, tuple) and len(atom_map) == 2:
            for side_map in atom_map:
                if isinstance(side_map, int) and side_map > 0:
                    atom_map_nodes[side_map] = node
    return {
        atom_map_nodes[atom_map]
        for change in changes.values()
        if change.change != "RETAINED"
        for atom_map in change.dependencies
        if atom_map in atom_map_nodes
    }
