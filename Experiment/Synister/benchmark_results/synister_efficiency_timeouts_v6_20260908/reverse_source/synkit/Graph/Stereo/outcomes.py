"""Rule-level declarations for stereochemical product branching."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping

from .descriptors import StereoValue


@dataclass(frozen=True)
class StereoOutcome:
    """Describe how one stored rule expands a product stereo descriptor.

    ``SINGLE`` keeps the descriptor encoded by the rule product. ``RACEMIC``
    emits equal enantiomers. ``ENANTIOMERIC_MIXTURE`` uses the same two
    orientations with explicitly unequal weights. Unknown descriptor parity
    is a matching concern and never implies either product distribution.
    """

    kind: str = "SINGLE"
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        kind = self.kind.upper()
        object.__setattr__(self, "kind", kind)
        allowed = {"SINGLE", "RACEMIC", "ENANTIOMERIC_MIXTURE"}
        if kind not in allowed:
            raise ValueError(
                "Stereo outcome must be 'SINGLE', 'RACEMIC', or "
                "'ENANTIOMERIC_MIXTURE'."
            )
        expected_count = 1 if kind == "SINGLE" else 2
        weights = self.weights
        if weights is None:
            weights = (0.5, 0.5) if kind == "RACEMIC" else (1.0,)
            object.__setattr__(self, "weights", weights)
        if len(weights) != expected_count:
            raise ValueError(
                f"{kind} stereo outcome requires {expected_count} branch weight(s)."
            )
        if any(weight < 0 for weight in weights) or not isclose(sum(weights), 1.0):
            raise ValueError(
                "Stereo outcome branch weights must be nonnegative and sum to 1."
            )
        if kind == "RACEMIC" and not all(isclose(weight, 0.5) for weight in weights):
            raise ValueError("RACEMIC stereo outcome requires equal 0.5/0.5 weights.")
        if kind == "ENANTIOMERIC_MIXTURE" and isclose(weights[0], weights[1]):
            raise ValueError(
                "Equal enantiomer weights are RACEMIC, not ENANTIOMERIC_MIXTURE."
            )

    @classmethod
    def racemic(cls) -> "StereoOutcome":
        """Return an equal two-enantiomer product outcome."""
        return cls("RACEMIC", (0.5, 0.5))

    @classmethod
    def enantiomeric_mixture(
        cls,
        first: float,
        inverse: float,
    ) -> "StereoOutcome":
        """Return a non-racemic two-enantiomer product distribution."""
        return cls("ENANTIOMERIC_MIXTURE", (first, inverse))

    @classmethod
    def from_value(
        cls, value: "StereoOutcome | str | Mapping[str, Any]"
    ) -> "StereoOutcome":
        """Normalize public string/dictionary declarations."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        return cls(
            kind=str(value["kind"]),
            weights=tuple(float(weight) for weight in value.get("weights", ())) or None,
        )

    def alternatives(self, seed: StereoValue) -> tuple[StereoValue, ...]:
        """Expand a rule product descriptor into concrete branch descriptors."""
        if self.kind == "SINGLE":
            return (seed,)
        if seed.parity not in (-1, 1):
            raise ValueError(
                f"{self.kind} outcome requires a specified chiral product "
                "descriptor with parity -1 or 1."
            )
        return seed, seed.invert()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/GML-friendly rule metadata."""
        return {"kind": self.kind, "weights": list(self.weights or ())}

    def signature(self) -> tuple[str, tuple[float, ...]]:
        """Return a stable value-object signature."""
        return self.kind, tuple(self.weights or ())
