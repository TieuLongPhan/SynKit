"""Versioned public models for supplied reaction mechanisms.

The models in this module deliberately describe *annotations supplied by a
user or dataset*.  They do not score, enumerate, or select mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .electron_models import (  # noqa: F401
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismModelError,
    VerificationIssue,
)
from .symbols import CanonicalLocus

SCHEMA_VERSION = "2.0.0-draft1"
LocusKind = CanonicalLocus
ArrowType = Literal["curved", "fishhook"]
StereoEffectKind = Literal[
    "PRESERVE", "INVERT", "BREAK", "FORM", "FLEETING", "UNSPECIFIED"
]


@dataclass(frozen=True)
class StereoDescriptor:
    """Serializable envelope for SynKit's permutation-aware descriptors."""

    descriptor_class: Literal[
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
        "planar_bond",
        "atrop_bond",
        "unknown",
    ]
    atoms: tuple[int | str, ...]
    parity: int | None = None
    state: Literal["specified", "unknown", "unspecified", "absent"] = "specified"
    provenance: str | None = None

    def __post_init__(self) -> None:
        expected = {
            "tetrahedral": 5,
            "square_planar": 5,
            "trigonal_bipyramidal": 6,
            "octahedral": 7,
            "planar_bond": 6,
            "atrop_bond": 6,
            "unknown": 0,
        }[self.descriptor_class]
        if expected and len(self.atoms) != expected:
            raise MechanismModelError(
                f"{self.descriptor_class} descriptors require {expected} atom maps."
            )
        allowed = (
            (0, None)
            if self.descriptor_class in {"square_planar", "planar_bond"}
            else (-1, 1, None)
        )
        if self.parity not in allowed:
            raise MechanismModelError(
                f"Invalid parity for {self.descriptor_class} stereo."
            )
        if self.state == "specified" and self.parity is None:
            raise MechanismModelError("Specified stereo descriptors require a parity.")
        if self.state != "specified" and self.parity is not None:
            raise MechanismModelError(
                "Unknown, unspecified, and absent stereo states cannot carry parity."
            )
        if self.descriptor_class == "unknown" and self.state == "specified":
            raise MechanismModelError(
                "Unknown descriptor classes cannot declare specified stereo."
            )
        if self.descriptor_class != "unknown":
            # Reuse the executable descriptor contract so mechanism envelopes
            # cannot reintroduce StereoMolGraph's untyped ``None`` or attach a
            # virtual ligand to the wrong stereochemical owner.
            from synkit.Graph.Stereo.descriptors import stereo_from_dict

            try:
                stereo_from_dict(
                    {
                        "descriptor_class": self.descriptor_class,
                        "atoms": self.atoms,
                        "parity": self.parity,
                        "provenance": self.provenance,
                    }
                )
            except (TypeError, ValueError) as error:
                raise MechanismModelError(str(error)) from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_class": self.descriptor_class,
            "atoms": list(self.atoms),
            "parity": self.parity,
            "state": self.state,
            "provenance": self.provenance,
        }

    def inverted(self) -> "StereoDescriptor":
        """Return the opposite relative orientation of specified stereo."""
        if self.descriptor_class == "unknown" or self.state != "specified":
            raise MechanismModelError(
                "Only specified executable stereo descriptors can be inverted."
            )
        from synkit.Graph.Stereo import stereo_from_dict

        value = stereo_from_dict(self.to_dict()).invert().to_dict()
        value["state"] = self.state
        return StereoDescriptor.from_dict(value)

    @property
    def target_key(self) -> str:
        """Return the canonical registry key owned by this descriptor."""
        if self.descriptor_class in {
            "tetrahedral",
            "square_planar",
            "trigonal_bipyramidal",
            "octahedral",
        }:
            return f"atom:{self.atoms[0]}"
        if self.descriptor_class in {"planar_bond", "atrop_bond"}:
            return f"bond:{min(self.atoms[2:4])}-{max(self.atoms[2:4])}"
        raise MechanismModelError(
            "Unknown descriptor classes do not own an executable endpoint locus."
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StereoDescriptor":
        return cls(
            descriptor_class=value["descriptor_class"],
            atoms=tuple(value.get("atoms", ())),
            parity=value.get("parity"),
            state=value.get("state", "specified"),
            provenance=value.get("provenance"),
        )


@dataclass(frozen=True)
class StereoEffect:
    descriptor_target: tuple[str, int | tuple[int, int]]
    effect: StereoEffectKind
    before: StereoDescriptor | None = None
    after: StereoDescriptor | None = None
    provenance: str = "annotated"

    def __post_init__(self) -> None:
        if len(self.descriptor_target) != 2:
            raise MechanismModelError(
                "Stereo descriptor targets require a kind and mapped locus."
            )
        kind, reference = self.descriptor_target
        if kind == "atom":
            if type(reference) is not int or reference <= 0:
                raise MechanismModelError(
                    "Atom stereo targets require one positive atom map."
                )
            normalized_reference: int | tuple[int, int] = reference
        elif kind == "bond":
            if (
                not isinstance(reference, (tuple, list))
                or len(reference) != 2
                or any(type(value) is not int or value <= 0 for value in reference)
                or reference[0] == reference[1]
            ):
                raise MechanismModelError(
                    "Bond stereo targets require two distinct positive atom maps."
                )
            normalized_reference = tuple(sorted(reference))
        else:
            raise MechanismModelError("Stereo target kind must be 'atom' or 'bond'.")
        if self.effect not in {
            "PRESERVE",
            "INVERT",
            "BREAK",
            "FORM",
            "FLEETING",
            "UNSPECIFIED",
        }:
            raise MechanismModelError(f"Unsupported stereo effect: {self.effect!r}.")
        object.__setattr__(self, "descriptor_target", (kind, normalized_reference))

        target_key = (
            f"atom:{normalized_reference}"
            if kind == "atom"
            else f"bond:{normalized_reference[0]}-{normalized_reference[1]}"
        )
        for descriptor in (self.before, self.after):
            if descriptor is None or descriptor.descriptor_class == "unknown":
                continue
            descriptor_key = (
                f"atom:{descriptor.atoms[0]}"
                if descriptor.descriptor_class
                in {
                    "tetrahedral",
                    "square_planar",
                    "trigonal_bipyramidal",
                    "octahedral",
                }
                else f"bond:{min(descriptor.atoms[2:4])}-{max(descriptor.atoms[2:4])}"
            )
            if descriptor_key != target_key:
                raise MechanismModelError(
                    f"Stereo descriptor belongs to {descriptor_key}, not {target_key}."
                )

    def to_dict(self) -> dict[str, Any]:
        target_kind, target_reference = self.descriptor_target
        return {
            "descriptor_target": [
                target_kind,
                (
                    list(target_reference)
                    if isinstance(target_reference, tuple)
                    else target_reference
                ),
            ],
            "effect": self.effect,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "provenance": self.provenance,
        }

    def reversed(self) -> "StereoEffect":
        """Return the inverse endpoint transition when it is well-defined.

        ``UNSPECIFIED`` is deliberately non-invertible: discarding an
        orientation does not contain enough information to reconstruct it.
        """
        if self.effect == "UNSPECIFIED":
            raise MechanismModelError(
                "NONREVERSIBLE_STEREO_EFFECT: UNSPECIFIED stereo cannot be "
                "reversed without a supplied recovery descriptor."
            )
        if self.effect == "FORM":
            return StereoEffect(
                self.descriptor_target,
                "BREAK",
                before=self.after,
                provenance=self.provenance,
            )
        if self.effect == "BREAK":
            return StereoEffect(
                self.descriptor_target,
                "FORM",
                after=self.before,
                provenance=self.provenance,
            )
        if self.effect == "INVERT":
            if self.before is None:
                raise MechanismModelError(
                    "NONREVERSIBLE_STEREO_EFFECT: INVERT requires a before "
                    "descriptor."
                )
            reverse_before = self.after or self.before.inverted()
            return StereoEffect(
                self.descriptor_target,
                "INVERT",
                before=reverse_before,
                after=self.before,
                provenance=self.provenance,
            )
        if self.effect == "PRESERVE" and self.after is None:
            return self
        return StereoEffect(
            self.descriptor_target,
            self.effect,
            before=self.after,
            after=self.before,
            provenance=self.provenance,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StereoEffect":
        before = value.get("before")
        after = value.get("after")
        return cls(
            descriptor_target=tuple(value["descriptor_target"]),
            effect=value["effect"],
            before=StereoDescriptor.from_dict(before) if before else None,
            after=StereoDescriptor.from_dict(after) if after else None,
            provenance=value.get("provenance", "annotated"),
        )


@dataclass(frozen=True)
class MechanisticStep:
    step_id: str
    groups: tuple[ElectronMoveGroup, ...]
    stereo_effects: tuple[StereoEffect, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            raise MechanismModelError("MechanisticStep requires a step_id.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "groups": [group.to_dict() for group in self.groups],
            "stereo_effects": [effect.to_dict() for effect in self.stereo_effects],
            "metadata": dict(self.metadata),
        }

    def reversed(self) -> "MechanisticStep":
        """Reverse group order, electron flow, and stereo transitions."""
        return MechanisticStep(
            step_id=self.step_id,
            groups=tuple(group.reversed() for group in reversed(self.groups)),
            stereo_effects=tuple(
                effect.reversed() for effect in reversed(self.stereo_effects)
            ),
            metadata=self.metadata,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MechanisticStep":
        return cls(
            step_id=value["step_id"],
            groups=tuple(
                ElectronMoveGroup.from_dict(group) for group in value["groups"]
            ),
            stereo_effects=tuple(
                StereoEffect.from_dict(effect)
                for effect in value.get("stereo_effects", ())
            ),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class VerificationCertificate:
    status: Literal["VALID", "INVALID", "INCOMPLETE", "UNSUPPORTED"]
    schema_version: str = SCHEMA_VERSION
    step_reports: tuple[Mapping[str, Any], ...] = ()
    issues: tuple[VerificationIssue, ...] = ()
    final_match: Mapping[str, Any] = field(default_factory=dict)
    repaired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "schema_version": self.schema_version,
            "step_reports": [dict(report) for report in self.step_reports],
            "issues": [issue.to_dict() for issue in self.issues],
            "final_match": dict(self.final_match),
            "repaired": self.repaired,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationCertificate":
        return cls(
            status=value["status"],
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            step_reports=tuple(value.get("step_reports", ())),
            issues=tuple(
                VerificationIssue.from_dict(issue) for issue in value.get("issues", ())
            ),
            final_match=value.get("final_match", {}),
            repaired=bool(value.get("repaired", False)),
        )


@dataclass(frozen=True)
class MechanismRecord:
    """Versioned record for a supplied, mapped mechanism annotation."""

    mapped_reaction: str
    steps: tuple[MechanisticStep, ...]
    schema_version: str = SCHEMA_VERSION
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    endpoint_stereo: Mapping[str, Mapping[str, StereoDescriptor]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.mapped_reaction.count(">>") != 1:
            raise MechanismModelError("mapped_reaction must contain exactly one '>>'.")
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise MechanismModelError("Mechanistic step IDs must be unique.")
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))
        endpoint_stereo: dict[str, dict[str, StereoDescriptor]] = {}
        for side, registry in self.endpoint_stereo.items():
            if side not in {"reactant", "product"}:
                raise MechanismModelError(
                    "Endpoint stereo side must be 'reactant' or 'product'."
                )
            endpoint_stereo[side] = {}
            for key, descriptor in registry.items():
                if not isinstance(descriptor, StereoDescriptor):
                    raise MechanismModelError(
                        "Endpoint stereo registries require StereoDescriptor values."
                    )
                if descriptor.target_key != key:
                    raise MechanismModelError(
                        f"Endpoint descriptor belongs to {descriptor.target_key}, "
                        f"not {key}."
                    )
                endpoint_stereo[side][key] = descriptor
        object.__setattr__(self, "endpoint_stereo", endpoint_stereo)

    @classmethod
    def from_ef_smirks(cls, text: str, **kwargs: Any) -> "MechanismRecord":
        from .adapters import mechanism_from_ef_smirks

        return mechanism_from_ef_smirks(text, **kwargs)

    def grammar_issues(self) -> tuple[VerificationIssue, ...]:
        return tuple(
            issue
            for step in self.steps
            for group in step.groups
            for issue in group.issues()
        )

    def reversed(self) -> "MechanismRecord":
        """Return the typed mechanism operating from product to reactant."""
        reactants, products = self.mapped_reaction.split(">>", 1)
        return MechanismRecord(
            mapped_reaction=f"{products}>>{reactants}",
            steps=tuple(step.reversed() for step in reversed(self.steps)),
            schema_version=self.schema_version,
            provenance=self.provenance,
            metadata=self.metadata,
            endpoint_stereo={
                side: self.endpoint_stereo[other]
                for side, other in (
                    ("reactant", "product"),
                    ("product", "reactant"),
                )
                if other in self.endpoint_stereo
            },
        )

    def verify(
        self,
        *,
        electron: Literal["strict", "diagnostic"] = "strict",
        stereo: Literal["off", "endpoint", "stepwise"] = "off",
        repair: bool = False,
    ) -> VerificationCertificate:
        """Replay this supplied mechanism and return its certificate."""
        from .replay import MechanismReplayer

        return (
            MechanismReplayer(
                validation=electron,
                verify_stereo=stereo,
                repair=repair,
            )
            .replay(self)
            .certificate
        )

    def to_mtg(
        self,
        *,
        electron: Literal["strict", "diagnostic"] = "strict",
        stereo: Literal["off", "endpoint", "stepwise"] = "stepwise",
    ) -> Any:
        """Replay and return the mechanism trajectory graph."""
        from .replay import MechanismReplayer

        return (
            MechanismReplayer(validation=electron, verify_stereo=stereo)
            .replay(self)
            .mtg
        )

    def to_rule(
        self,
        *,
        core: bool = False,
        implicit_h: bool = False,
        name: str | None = None,
    ) -> Any:
        """Build a tuple :class:`SynRule` from the same mapped endpoints.

        Explicit endpoint stereo sidecars replace the corresponding SMILES
        registry, which keeps unknown stereo and backend-independent relative
        descriptors available at the rule boundary.  ``core=False`` is the
        conservative default because it preserves the complete reviewed
        endpoint context used for rule/replay agreement checks.
        """
        from synkit.Graph.Stereo import (
            StereoChange,
            classify_stereo_change,
            stereo_from_dict,
        )
        from synkit.IO.chem_converter import rsmi_to_its
        from synkit.Rule import SynRule

        its = rsmi_to_its(
            self.mapped_reaction,
            core=core,
            format="tuple",
            drop_non_aam=False,
            use_index_as_atom_map=True,
        )
        registries = {
            side: dict(registry)
            for side, registry in its.graph.get("stereo_descriptors", {}).items()
        }
        for side, registry in self.endpoint_stereo.items():
            registries[side] = {
                key: stereo_from_dict(descriptor.to_dict())
                for key, descriptor in registry.items()
            }
        registries.setdefault("reactant", {})
        registries.setdefault("product", {})
        its.graph["stereo_descriptors"] = registries
        its.graph["stereo_changes"] = {
            key: StereoChange(
                classify_stereo_change(
                    registries["reactant"].get(key),
                    registries["product"].get(key),
                ),
                registries["reactant"].get(key),
                registries["product"].get(key),
            )
            for key in set(registries["reactant"]) | set(registries["product"])
        }
        return SynRule(
            its,
            name=name,
            format="tuple",
            implicit_h=implicit_h,
        )

    def draw(
        self,
        *,
        show_certificate: bool = False,
        certificate: VerificationCertificate | None = None,
        path: str | Path | None = None,
    ) -> Any:
        """Create a headless static mechanism overview figure."""
        from .drawing import draw_mechanism_record

        if show_certificate and certificate is None:
            certificate = self.verify(stereo="stepwise")
        return draw_mechanism_record(self, certificate=certificate, path=path)

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the versioned record and optionally write it to disk."""
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True
        )
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload

    @classmethod
    def from_json(cls, value: str | Path) -> "MechanismRecord":
        """Read a JSON string or an existing JSON path."""
        if isinstance(value, Path):
            text = value.read_text(encoding="utf-8")
        else:
            stripped = value.lstrip()
            if stripped.startswith(("{", "[")):
                text = value
            else:
                candidate = Path(value)
                text = (
                    candidate.read_text(encoding="utf-8")
                    if candidate.exists()
                    else value
                )
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mapped_reaction": self.mapped_reaction,
            "steps": [step.to_dict() for step in self.steps],
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
            "endpoint_stereo": {
                side: {
                    key: descriptor.to_dict() for key, descriptor in registry.items()
                }
                for side, registry in self.endpoint_stereo.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MechanismRecord":
        return cls(
            mapped_reaction=value["mapped_reaction"],
            steps=tuple(MechanisticStep.from_dict(step) for step in value["steps"]),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            provenance=value.get("provenance", {}),
            metadata=value.get("metadata", {}),
            endpoint_stereo={
                side: {
                    key: StereoDescriptor.from_dict(descriptor)
                    for key, descriptor in registry.items()
                }
                for side, registry in value.get("endpoint_stereo", {}).items()
            },
        )


def issues_are_errors(issues: Iterable[VerificationIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)
