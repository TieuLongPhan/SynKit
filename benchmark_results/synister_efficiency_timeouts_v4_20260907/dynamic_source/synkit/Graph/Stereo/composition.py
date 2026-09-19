"""Proof-bearing composition of normalized reaction-stereo values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from itertools import product
import json
from typing import Any, Mapping

from .changes import StereoAlignmentError, StereoChange
from .wire import StereoReactionValue


class StereoCompositionIssueCode(str, Enum):
    """Stable reasons a sequential stereo assertion cannot be fused."""

    INTERMEDIATE_MISMATCH = "STEREO_COMPOSITION_INTERMEDIATE_MISMATCH"
    INFORMATION_LOSS = "STEREO_COMPOSITION_INFORMATION_LOSS"
    OUTCOME_CONFLICT = "STEREO_COMPOSITION_OUTCOME_CONFLICT"
    COUPLING_CONFLICT = "STEREO_COMPOSITION_COUPLING_CONFLICT"
    ASSERTION_CONFLICT = "STEREO_COMPOSITION_ASSERTION_CONFLICT"
    NON_INVERTIBLE = "STEREO_COMPOSITION_NON_INVERTIBLE"
    PROOF_TAMPERED = "STEREO_COMPOSITION_PROOF_TAMPERED"


class StereoCompositionError(ValueError):
    """Typed refusal for lossy, conflicting, or tampered composition."""

    def __init__(
        self,
        code: StereoCompositionIssueCode,
        detail: str,
        targets: tuple[str, ...] = (),
    ) -> None:
        self.code = code
        self.detail = detail
        self.targets = targets
        super().__init__(f"{code.value}: {detail}")


@dataclass(frozen=True)
class StereoBranchContribution:
    """One raw sequential path contributing to a final product branch."""

    path: tuple[tuple[int, str, int], ...]
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": [list(item) for item in self.path],
            "weight": self.weight,
        }


@dataclass(frozen=True)
class StereoComposedBranch:
    """One final branch with accumulated measure and raw path evidence."""

    final_choices: tuple[tuple[str, int], ...]
    weight: float
    contributions: tuple[StereoBranchContribution, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.contributions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_choices": [list(item) for item in self.final_choices],
            "weight": self.weight,
            "multiplicity": self.multiplicity,
            "contributions": [
                contribution.to_dict() for contribution in self.contributions
            ],
        }


def _composition_digest(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("proof_digest", None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StereoReactionComposition:
    """A fused semantic value and replayable branch-measure proof."""

    result: StereoReactionValue
    branches: tuple[StereoComposedBranch, ...]
    correlation_policy: str
    proof_digest: str

    @property
    def total_weight(self) -> float:
        return sum(branch.weight for branch in self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "synkit.reaction-stereo-composition/1",
            "result": self.result.to_dict(),
            "branches": [branch.to_dict() for branch in self.branches],
            "correlation_policy": self.correlation_policy,
            "total_weight": self.total_weight,
            "proof_digest": self.proof_digest,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "StereoReactionComposition":
        if value.get("schema") != "synkit.reaction-stereo-composition/1":
            raise StereoCompositionError(
                StereoCompositionIssueCode.PROOF_TAMPERED,
                "Unsupported composition proof schema.",
            )
        if value.get("proof_digest") != _composition_digest(value):
            raise StereoCompositionError(
                StereoCompositionIssueCode.PROOF_TAMPERED,
                "The composition proof digest does not match its content.",
            )
        branches = tuple(
            StereoComposedBranch(
                tuple(
                    (str(target), int(index))
                    for target, index in branch["final_choices"]
                ),
                float(branch["weight"]),
                tuple(
                    StereoBranchContribution(
                        tuple(
                            (int(step), str(target), int(index))
                            for step, target, index in contribution["path"]
                        ),
                        float(contribution["weight"]),
                    )
                    for contribution in branch["contributions"]
                ),
            )
            for branch in value["branches"]
        )
        result = cls(
            StereoReactionValue.from_dict(value["result"]),
            branches,
            str(value["correlation_policy"]),
            str(value["proof_digest"]),
        )
        if abs(result.total_weight - float(value["total_weight"])) > 1e-12:
            raise StereoCompositionError(
                StereoCompositionIssueCode.PROOF_TAMPERED,
                "The serialized total branch measure is inconsistent.",
            )
        return result


def _compose_effects(
    first: StereoReactionValue,
    second: StereoReactionValue,
) -> dict[str, StereoChange]:
    effects = {}
    for target in sorted(set(first.effects) | set(second.effects)):
        left = first.effects.get(target)
        right = second.effects.get(target)
        if left is None:
            effects[target] = right
            continue
        if right is None:
            effects[target] = left
            continue
        try:
            composed = left.then(right)
        except StereoAlignmentError as error:
            raise StereoCompositionError(
                StereoCompositionIssueCode.INTERMEDIATE_MISMATCH,
                str(error),
                (target,),
            ) from error
        if composed.non_invertible:
            raise StereoCompositionError(
                StereoCompositionIssueCode.INFORMATION_LOSS,
                "The intermediate configuration was destroyed and recreated.",
                (target,),
            )
        effects[target] = composed
    return effects


def _validate_intermediate_guards(
    first: StereoReactionValue,
    second: StereoReactionValue,
) -> None:
    for target, guard in second.guards.items():
        produced = first.effects.get(target)
        if produced is not None and produced.after != guard:
            raise StereoCompositionError(
                StereoCompositionIssueCode.INTERMEDIATE_MISMATCH,
                "The first product does not satisfy the second guard.",
                (target,),
            )
        retained = first.guards.get(target)
        if produced is None and retained is not None and retained != guard:
            raise StereoCompositionError(
                StereoCompositionIssueCode.INTERMEDIATE_MISMATCH,
                "The unchanged first-step state conflicts with the " "second guard.",
                (target,),
            )


def _compose_outcomes(
    first: StereoReactionValue,
    second: StereoReactionValue,
    effects: Mapping[str, StereoChange],
) -> dict[str, Any]:
    overlap = set(first.outcomes) & set(second.outcomes)
    if overlap:
        raise StereoCompositionError(
            StereoCompositionIssueCode.OUTCOME_CONFLICT,
            "Two population declarations at one locus require a richer "
            "explicit correlation model.",
            tuple(sorted(overlap)),
        )
    outcomes = dict(second.outcomes)
    for target, outcome in first.outcomes.items():
        effect = effects.get(target)
        if effect is not None and effect.after is not None:
            outcomes[target] = outcome
    return outcomes


def _compose_couplings(
    first: StereoReactionValue,
    second: StereoReactionValue,
) -> dict[str, Any]:
    overlap = set(first.couplings) & set(second.couplings)
    if overlap:
        raise StereoCompositionError(
            StereoCompositionIssueCode.COUPLING_CONFLICT,
            "Sequential couplings at one bond are not representable as one "
            "vicinal operation.",
            tuple(sorted(overlap)),
        )
    result = dict(first.couplings)
    result.update(second.couplings)
    return result


def _branch_proof(
    first: StereoReactionValue,
    second: StereoReactionValue,
    effects: Mapping[str, StereoChange],
) -> tuple[StereoComposedBranch, ...]:
    choices = []
    for step, outcomes in ((1, first.outcomes), (2, second.outcomes)):
        for target, outcome in sorted(outcomes.items()):
            choices.append(
                tuple(
                    ((step, target, index), float(weight))
                    for index, weight in enumerate(outcome.weights or ())
                    if float(weight) > 0.0
                )
            )
    raw_paths = product(*choices) if choices else ((),)
    grouped: dict[
        tuple[tuple[str, int], ...],
        list[StereoBranchContribution],
    ] = {}
    for raw_path in raw_paths:
        path = tuple(item for item, _weight in raw_path)
        weight = 1.0
        for _item, local_weight in raw_path:
            weight *= local_weight
        final = []
        for step, target, index in path:
            effect = effects.get(target)
            if effect is None or effect.after is None:
                continue
            if step == 1 and target in second.outcomes:
                continue
            final.append((target, index))
        final_choices = tuple(sorted(final))
        grouped.setdefault(final_choices, []).append(
            StereoBranchContribution(path, weight)
        )
    return tuple(
        StereoComposedBranch(
            final_choices,
            sum(item.weight for item in contributions),
            tuple(contributions),
        )
        for final_choices, contributions in sorted(grouped.items())
    )


def compose_reaction_stereo(
    first: StereoReactionValue,
    second: StereoReactionValue,
) -> StereoReactionComposition:
    """Fuse two semantic values or refuse without losing evidence."""
    _validate_intermediate_guards(first, second)
    effects = _compose_effects(first, second)
    outcomes = _compose_outcomes(first, second, effects)
    couplings = _compose_couplings(first, second)

    assertions = dict(first.assertions)
    for target, assertion in second.assertions.items():
        if target in assertions and assertions[target] != assertion:
            raise StereoCompositionError(
                StereoCompositionIssueCode.ASSERTION_CONFLICT,
                "Sequential semantic assertions disagree.",
                (target,),
            )
        assertions[target] = assertion

    guards = dict(first.guards)
    for target, guard in second.guards.items():
        if target not in first.effects:
            guards.setdefault(target, guard)
    result = StereoReactionValue(
        guards=guards,
        effects=effects,
        outcomes=outcomes,
        couplings=couplings,
        assertions=assertions,
        refusals=(*first.refusals, *second.refusals),
    )
    branches = _branch_proof(first, second, effects)
    payload = {
        "schema": "synkit.reaction-stereo-composition/1",
        "result": result.to_dict(),
        "branches": [branch.to_dict() for branch in branches],
        "correlation_policy": "independent_across_declared_outcomes",
        "total_weight": sum(branch.weight for branch in branches),
    }
    return StereoReactionComposition(
        result,
        branches,
        payload["correlation_policy"],
        _composition_digest(payload),
    )


def reverse_reaction_stereo(
    value: StereoReactionValue,
) -> StereoReactionValue:
    """Reverse only when no population or lost evidence would be invented."""
    non_single = [
        target for target, outcome in value.outcomes.items() if outcome.kind != "SINGLE"
    ]
    non_invertible = [
        target for target, effect in value.effects.items() if effect.non_invertible
    ]
    if non_single or non_invertible or value.refusals:
        raise StereoCompositionError(
            StereoCompositionIssueCode.NON_INVERTIBLE,
            "Reversal would invent population or lost configuration evidence.",
            tuple(sorted(set(non_single) | set(non_invertible))),
        )
    effects = {target: effect.reverse() for target, effect in value.effects.items()}
    guards = {
        target: effect.before
        for target, effect in effects.items()
        if effect.before is not None
    }
    return StereoReactionValue(
        guards=guards,
        effects=effects,
        outcomes=value.outcomes,
        couplings={
            target: coupling.reverse() for target, coupling in value.couplings.items()
        },
        assertions=value.assertions,
    )


__all__ = [
    "StereoBranchContribution",
    "StereoComposedBranch",
    "StereoCompositionError",
    "StereoCompositionIssueCode",
    "StereoReactionComposition",
    "compose_reaction_stereo",
    "reverse_reaction_stereo",
]
