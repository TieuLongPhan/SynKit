"""Versioned, deterministic wire value for reaction stereochemistry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .changes import StereoChange
from .couplings import StereoCoupling
from .descriptors import StereoValue, descriptor_id, stereo_from_dict
from .outcomes import StereoOutcome
from .semantics import (
    StereoReactionSemantics,
    StereoRefusal,
    StereoRefusalCode,
)

LEGACY_REACTION_STEREO_SCHEMA = "synkit.reaction-stereo/1"
REACTION_STEREO_SCHEMA = "synkit.reaction-stereo/2"
SUPPORTED_REACTION_STEREO_SCHEMAS = frozenset(
    {LEGACY_REACTION_STEREO_SCHEMA, REACTION_STEREO_SCHEMA}
)


class ReactionStereoSchemaError(ValueError):
    """Raised when a reaction-stereo payload is invalid or would lose data."""

    def __init__(self, refusal: StereoRefusal) -> None:
        self.refusal = refusal
        super().__init__(f"{refusal.code.value}: {refusal.detail}")


def _normalized_items(
    values: Mapping[str, Any] | Iterable[tuple[str, Any]] | None,
    *,
    field: str,
) -> tuple[tuple[str, Any], ...]:
    if values is None:
        return ()
    items = tuple(values.items() if isinstance(values, Mapping) else values)
    keys = [key for key, _value in items]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ReactionStereoSchemaError(
            StereoRefusal(
                StereoRefusalCode.CONTRADICTORY_ASSERTION,
                f"{field} targets must be nonempty strings.",
            )
        )
    if len(keys) != len(set(keys)):
        raise ReactionStereoSchemaError(
            StereoRefusal(
                StereoRefusalCode.CONTRADICTORY_ASSERTION,
                f"{field} contains duplicate targets.",
                tuple(keys),
            )
        )
    return tuple(sorted(items, key=lambda item: item[0]))


@dataclass(frozen=True, init=False)
class StereoReactionValue:
    """Immutable normalized reaction-stereo values and assertions."""

    schema: str
    _guards: tuple[tuple[str, StereoValue], ...]
    _effects: tuple[tuple[str, StereoChange], ...]
    _outcomes: tuple[tuple[str, StereoOutcome], ...]
    _couplings: tuple[tuple[str, StereoCoupling], ...]
    _assertions: tuple[tuple[str, StereoReactionSemantics], ...]
    refusals: tuple[StereoRefusal, ...]

    def __init__(
        self,
        *,
        guards: (
            Mapping[str, StereoValue] | Iterable[tuple[str, StereoValue]] | None
        ) = None,
        effects: (
            Mapping[str, StereoChange] | Iterable[tuple[str, StereoChange]] | None
        ) = None,
        outcomes: (
            Mapping[str, StereoOutcome | str | Mapping[str, Any]]
            | Iterable[tuple[str, StereoOutcome | str | Mapping[str, Any]]]
            | None
        ) = None,
        couplings: (
            Mapping[str, StereoCoupling | Mapping[str, Any]]
            | Iterable[tuple[str, StereoCoupling | Mapping[str, Any]]]
            | None
        ) = None,
        assertions: (
            Mapping[str, StereoReactionSemantics]
            | Iterable[tuple[str, StereoReactionSemantics]]
            | None
        ) = None,
        refusals: Iterable[StereoRefusal] = (),
        schema: str = REACTION_STEREO_SCHEMA,
    ) -> None:
        if schema not in SUPPORTED_REACTION_STEREO_SCHEMAS:
            raise ReactionStereoSchemaError(
                StereoRefusal(
                    StereoRefusalCode.CONTRADICTORY_ASSERTION,
                    f"Unsupported reaction-stereo schema: {schema!r}.",
                )
            )
        normalized_guards = _normalized_items(guards, field="guards")
        normalized_effects = _normalized_items(effects, field="effects")
        normalized_outcomes = tuple(
            (target, StereoOutcome.from_value(value))
            for target, value in _normalized_items(
                outcomes,
                field="outcomes",
            )
        )
        normalized_couplings = tuple(
            (target, StereoCoupling.from_value(value))
            for target, value in _normalized_items(
                couplings,
                field="couplings",
            )
        )
        normalized_assertions = _normalized_items(
            assertions,
            field="assertions",
        )
        normalized_refusals = tuple(
            sorted(
                refusals,
                key=lambda item: (
                    item.code.value,
                    item.targets,
                    item.detail,
                ),
            )
        )

        self._validate(
            normalized_guards,
            normalized_effects,
            normalized_outcomes,
            normalized_couplings,
            normalized_assertions,
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "_guards", normalized_guards)
        object.__setattr__(self, "_effects", normalized_effects)
        object.__setattr__(self, "_outcomes", normalized_outcomes)
        object.__setattr__(self, "_couplings", normalized_couplings)
        object.__setattr__(self, "_assertions", normalized_assertions)
        object.__setattr__(self, "refusals", normalized_refusals)

    @staticmethod
    def _validate(
        guards: tuple[tuple[str, StereoValue], ...],
        effects: tuple[tuple[str, StereoChange], ...],
        outcomes: tuple[tuple[str, StereoOutcome], ...],
        couplings: tuple[tuple[str, StereoCoupling], ...],
        assertions: tuple[tuple[str, StereoReactionSemantics], ...],
    ) -> None:
        del assertions
        for target, guard in guards:
            if descriptor_id(guard) != target:
                raise ReactionStereoSchemaError(
                    StereoRefusal(
                        StereoRefusalCode.INVALID_REFERENCE,
                        f"Guard belongs to {descriptor_id(guard)}, " f"not {target}.",
                        (target,),
                    )
                )
        effect_map = dict(effects)
        for target, effect in effects:
            for descriptor in (
                effect.before,
                effect.after,
                effect.transition,
            ):
                if descriptor is not None and descriptor_id(descriptor) != target:
                    raise ReactionStereoSchemaError(
                        StereoRefusal(
                            StereoRefusalCode.INVALID_REFERENCE,
                            f"Effect descriptor belongs to "
                            f"{descriptor_id(descriptor)}, not {target}.",
                            (target,),
                        )
                    )
        for target, _outcome in outcomes:
            if target not in effect_map:
                raise ReactionStereoSchemaError(
                    StereoRefusal(
                        StereoRefusalCode.CONTRADICTORY_ASSERTION,
                        "A product outcome requires an effect at the same " "target.",
                        (target,),
                    )
                )
        for target, coupling in couplings:
            if coupling.target != target:
                raise ReactionStereoSchemaError(
                    StereoRefusal(
                        StereoRefusalCode.INVALID_COUPLING,
                        f"Coupling belongs to {coupling.target}, " f"not {target}.",
                        (target,),
                    )
                )

    @property
    def guards(self) -> Mapping[str, StereoValue]:
        return MappingProxyType(dict(self._guards))

    @property
    def effects(self) -> Mapping[str, StereoChange]:
        return MappingProxyType(dict(self._effects))

    @property
    def outcomes(self) -> Mapping[str, StereoOutcome]:
        return MappingProxyType(dict(self._outcomes))

    @property
    def couplings(self) -> Mapping[str, StereoCoupling]:
        return MappingProxyType(dict(self._couplings))

    @property
    def assertions(self) -> Mapping[str, StereoReactionSemantics]:
        return MappingProxyType(dict(self._assertions))

    def to_dict(self) -> dict[str, Any]:
        """Return normalized schema-v2 content."""
        return {
            "schema": REACTION_STEREO_SCHEMA,
            "guards": {target: value.to_dict() for target, value in self._guards},
            "effects": {target: value.to_dict() for target, value in self._effects},
            "outcomes": {target: value.to_dict() for target, value in self._outcomes},
            "couplings": {target: value.to_dict() for target, value in self._couplings},
            "assertions": {
                target: value.to_dict() for target, value in self._assertions
            },
            "refusals": [value.to_dict() for value in self.refusals],
        }

    def normalized_json(self) -> str:
        """Return byte-stable normalized JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StereoReactionValue":
        """Read schema v1/v2 or the equivalent unversioned v1 value."""
        schema = str(value.get("schema", LEGACY_REACTION_STEREO_SCHEMA))
        if schema not in SUPPORTED_REACTION_STEREO_SCHEMAS:
            raise ReactionStereoSchemaError(
                StereoRefusal(
                    StereoRefusalCode.CONTRADICTORY_ASSERTION,
                    f"Unsupported reaction-stereo schema: {schema!r}.",
                )
            )
        return cls(
            guards={
                target: stereo_from_dict(item)
                for target, item in value.get("guards", {}).items()
            },
            effects={
                target: StereoChange.from_dict(item)
                for target, item in value.get("effects", {}).items()
            },
            outcomes=value.get("outcomes", {}),
            couplings=value.get("couplings", {}),
            assertions={
                target: StereoReactionSemantics.from_dict(item)
                for target, item in value.get("assertions", {}).items()
            },
            refusals=tuple(
                StereoRefusal.from_dict(item) for item in value.get("refusals", ())
            ),
            schema=REACTION_STEREO_SCHEMA,
        )

    @classmethod
    def from_json(cls, value: str) -> "StereoReactionValue":
        """Read one JSON value."""
        return cls.from_dict(json.loads(value))

    def to_legacy_dict(self) -> dict[str, Any]:
        """Downgrade only when schema v1 can preserve all assertions."""
        if self._assertions or self.refusals:
            raise ReactionStereoSchemaError(
                StereoRefusal(
                    StereoRefusalCode.LOSSY_PROJECTION,
                    "Schema v1 cannot carry normalized assertions " "or refusals.",
                    tuple(target for target, _value in self._assertions),
                )
            )
        payload = self.to_dict()
        payload["schema"] = LEGACY_REACTION_STEREO_SCHEMA
        payload.pop("assertions")
        payload.pop("refusals")
        return payload


def reaction_stereo_schema() -> dict[str, Any]:
    """Return the dependency-free JSON-Schema view of schema v2."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": REACTION_STEREO_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "guards",
            "effects",
            "outcomes",
            "couplings",
            "assertions",
            "refusals",
        ],
        "properties": {
            "schema": {"const": REACTION_STEREO_SCHEMA},
            "guards": {"type": "object"},
            "effects": {"type": "object"},
            "outcomes": {"type": "object"},
            "couplings": {"type": "object"},
            "assertions": {"type": "object"},
            "refusals": {"type": "array"},
        },
    }


__all__ = [
    "LEGACY_REACTION_STEREO_SCHEMA",
    "REACTION_STEREO_SCHEMA",
    "SUPPORTED_REACTION_STEREO_SCHEMAS",
    "ReactionStereoSchemaError",
    "StereoReactionValue",
    "reaction_stereo_schema",
]
