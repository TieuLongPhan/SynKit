"""Versioned public models for supplied reaction mechanisms.

The models in this module deliberately describe *annotations supplied by a
user or dataset*.  They do not score, enumerate, or select mechanisms.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, Mapping, Sequence

from .symbols import (
    CanonicalLocus,
    LONE_PAIR,
    PI,
    RADICAL,
    SIGMA,
    normalize_locus_symbol,
)

SCHEMA_VERSION = "2.0.0-draft1"
LocusKind = CanonicalLocus
ArrowType = Literal["curved", "fishhook"]
StereoEffectKind = Literal[
    "PRESERVE", "INVERT", "BREAK", "FORM", "FLEETING", "UNSPECIFIED"
]


class MechanismModelError(ValueError):
    """Raised when a public mechanism record is structurally invalid."""


@dataclass(frozen=True)
class ElectronLocus:
    """A mapped electron resource or destination.

    Atom loci (``lp`` and ``∙``) reference one atom map; bond loci
    (``σ`` and ``π``) reference two distinct atom maps in canonical
    ascending order.
    """

    kind: LocusKind
    atom_maps: tuple[int, ...]

    _ATOM_KINDS: ClassVar[frozenset[str]] = frozenset({LONE_PAIR, RADICAL})
    _BOND_KINDS: ClassVar[frozenset[str]] = frozenset({SIGMA, PI})

    def __post_init__(self) -> None:
        try:
            kind = normalize_locus_symbol(self.kind)
        except (TypeError, ValueError) as exc:
            raise MechanismModelError(str(exc)) from exc
        maps = tuple(int(atom_map) for atom_map in self.atom_maps)
        if any(atom_map <= 0 for atom_map in maps):
            raise MechanismModelError("Electron-locus atom maps must be positive.")
        expected = 1 if kind in self._ATOM_KINDS else 2
        if len(maps) != expected or (expected == 2 and maps[0] == maps[1]):
            raise MechanismModelError(
                f"{kind!r} loci require exactly {expected} "
                f"{'distinct ' if expected == 2 else ''}atom map(s)."
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "atom_maps", tuple(sorted(maps)) if expected == 2 else maps
        )

    @classmethod
    def atom(cls, kind: str, *, atom_map: int) -> "ElectronLocus":
        return cls(kind=kind, atom_maps=(atom_map,))

    @classmethod
    def bond(cls, kind: str, *, atom_maps: Sequence[int]) -> "ElectronLocus":
        return cls(kind=kind, atom_maps=tuple(atom_maps))

    def to_dict(self) -> dict[str, Any]:
        return {"locus": self.kind, "atom_maps": list(self.atom_maps)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ElectronLocus":
        return cls(kind=value["locus"], atom_maps=tuple(value["atom_maps"]))


@dataclass(frozen=True)
class ElectronMove:
    """One curved-arrow or fishhook annotation.

    A move is syntactically valid on its own.  Coupling and resource checks are
    performed by :class:`ElectronMoveGroup`, because all moves in a group read
    the same pre-state and commit atomically.
    """

    source: ElectronLocus
    target: ElectronLocus
    electron_count: Literal[1, 2]
    arrow_type: ArrowType
    group_id: str
    event_id: str | None = None
    coupling_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.electron_count not in (1, 2):
            raise MechanismModelError("electron_count must be 1 or 2.")
        expected = "fishhook" if self.electron_count == 1 else "curved"
        if self.arrow_type != expected:
            raise MechanismModelError(
                f"ARROW_ELECTRON_COUNT_MISMATCH: {self.electron_count} electron "
                f"moves require arrow_type={expected!r}."
            )
        if not str(self.group_id).strip():
            raise MechanismModelError("Electron moves require a non-empty group_id.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "electron_count": self.electron_count,
            "arrow_type": self.arrow_type,
            "group_id": self.group_id,
            "coupling_id": self.coupling_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ElectronMove":
        return cls(
            event_id=value.get("event_id"),
            source=ElectronLocus.from_dict(value["source"]),
            target=ElectronLocus.from_dict(value["target"]),
            electron_count=int(value["electron_count"]),
            arrow_type=value["arrow_type"],
            group_id=value["group_id"],
            coupling_id=value.get("coupling_id"),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True)
class VerificationIssue:
    """A deterministic, machine-readable mechanism validation finding."""

    code: str
    message: str
    severity: Literal["error", "warning", "info"] = "error"
    step_id: str | None = None
    group_id: str | None = None
    atom_maps: tuple[int, ...] = ()
    expected: Any = None
    observed: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "step_id": self.step_id,
            "group_id": self.group_id,
            "atom_maps": list(self.atom_maps),
            "expected": self.expected,
            "observed": self.observed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationIssue":
        return cls(
            code=value["code"],
            message=value["message"],
            severity=value.get("severity", "error"),
            step_id=value.get("step_id"),
            group_id=value.get("group_id"),
            atom_maps=tuple(value.get("atom_maps", ())),
            expected=value.get("expected"),
            observed=value.get("observed"),
        )


@dataclass(frozen=True)
class ElectronMoveGroup:
    """Simultaneous electron moves with optional named radical macro."""

    group_id: str
    moves: tuple[ElectronMove, ...]
    macro: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SUPPORTED_MACROS: ClassVar[frozenset[str]] = frozenset(
        {
            "HOMOLYSIS",
            "RECOMBINATION",
            "RADICAL_ADDITION",
            "BETA_SCISSION",
            "H_ABSTRACTION",
            "RADICAL_RESONANCE",
        }
    )

    def __post_init__(self) -> None:
        if not self.moves:
            raise MechanismModelError("ElectronMoveGroup requires at least one move.")
        if not str(self.group_id).strip():
            raise MechanismModelError(
                "ElectronMoveGroup requires a non-empty group_id."
            )
        if any(move.group_id != self.group_id for move in self.moves):
            raise MechanismModelError("Every move must use its containing group_id.")
        macro = self.macro.upper() if self.macro else None
        if macro and macro not in self.SUPPORTED_MACROS:
            raise MechanismModelError(f"UNSUPPORTED_RADICAL_MACRO: {self.macro!r}")
        object.__setattr__(self, "macro", macro)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def issues(self) -> tuple[VerificationIssue, ...]:
        """Return grammar issues independent of a particular molecular state."""
        issues: list[VerificationIssue] = []
        seen: set[tuple[Any, ...]] = set()
        coupled: dict[str, list[ElectronMove]] = {}
        for move in self.moves:
            signature = (move.source, move.target, move.electron_count, move.arrow_type)
            if signature in seen:
                issues.append(
                    VerificationIssue(
                        code=(
                            "DUPLICATE_FISHHOOK"
                            if move.electron_count == 1
                            else "DUPLICATE_ELECTRON_MOVE"
                        ),
                        message="Duplicate electron move in one simultaneous group.",
                        group_id=self.group_id,
                    )
                )
            seen.add(signature)
            if move.electron_count == 1:
                if not move.coupling_id:
                    issues.append(
                        VerificationIssue(
                            code="MISSING_COUPLED_FISHHOOK",
                            message="Fishhook moves require a coupling_id before commit.",
                            group_id=self.group_id,
                        )
                    )
                else:
                    coupled.setdefault(move.coupling_id, []).append(move)
        for coupling_id, moves in coupled.items():
            if len(moves) < 2:
                issues.append(
                    VerificationIssue(
                        code="MISSING_COUPLED_FISHHOOK",
                        message=f"Coupling {coupling_id!r} has fewer than two fishhooks.",
                        group_id=self.group_id,
                    )
                )
        issues.extend(self._macro_issues())
        return tuple(issues)

    def _macro_issues(self) -> list[VerificationIssue]:
        if not self.macro:
            return []
        fishhooks = [move for move in self.moves if move.electron_count == 1]
        if len(fishhooks) < 2:
            return [
                VerificationIssue(
                    code="UNBALANCED_EVENT_GROUP",
                    message=f"{self.macro} requires coupled one-electron moves.",
                    group_id=self.group_id,
                )
            ]
        if self.macro == "HOMOLYSIS":
            sources = {move.source for move in fishhooks}
            targets = {move.target for move in fishhooks}
            valid = (
                len(sources) == 1
                and next(iter(sources)).kind == SIGMA
                and len(targets) == 2
                and all(target.kind == RADICAL for target in targets)
            )
            if not valid:
                return [
                    VerificationIssue(
                        "UNBALANCED_EVENT_GROUP",
                        "HOMOLYSIS requires one sigma source and two radical targets.",
                        group_id=self.group_id,
                    )
                ]
        if self.macro == "RECOMBINATION":
            sources = {move.source for move in fishhooks}
            targets = {move.target for move in fishhooks}
            valid = (
                len(sources) == 2
                and all(source.kind == RADICAL for source in sources)
                and len(targets) == 1
                and next(iter(targets)).kind == SIGMA
            )
            if not valid:
                return [
                    VerificationIssue(
                        "UNBALANCED_EVENT_GROUP",
                        "RECOMBINATION requires two radical sources and one sigma target.",
                        group_id=self.group_id,
                    )
                ]
        pairs = Counter((move.source.kind, move.target.kind) for move in fishhooks)
        exact_patterns = {
            "RADICAL_ADDITION": Counter(
                {(RADICAL, SIGMA): 1, (PI, SIGMA): 1, (PI, RADICAL): 1}
            ),
            "BETA_SCISSION": Counter(
                {(RADICAL, PI): 1, (SIGMA, PI): 1, (SIGMA, RADICAL): 1}
            ),
            "RADICAL_RESONANCE": Counter(
                {(RADICAL, PI): 1, (PI, PI): 1, (PI, RADICAL): 1}
            ),
        }
        expected = exact_patterns.get(self.macro)
        if expected is not None and pairs != expected:
            return [
                VerificationIssue(
                    "UNBALANCED_EVENT_GROUP",
                    f"{self.macro} fishhook locus pattern is invalid.",
                    group_id=self.group_id,
                    expected={f"{a}->{b}": count for (a, b), count in expected.items()},
                    observed={f"{a}->{b}": count for (a, b), count in pairs.items()},
                )
            ]
        if self.macro == "H_ABSTRACTION":
            bond_sources = Counter(
                move.source for move in fishhooks if move.source.kind in {SIGMA, PI}
            )
            bond_targets = Counter(
                move.target for move in fishhooks if move.target.kind in {SIGMA, PI}
            )
            has_radical_source = any(move.source.kind == RADICAL for move in fishhooks)
            valid = (
                len(fishhooks) >= 3
                and has_radical_source
                and any(count >= 2 for count in bond_sources.values())
                and any(count >= 2 for count in bond_targets.values())
            )
            if not valid:
                return [
                    VerificationIssue(
                        "UNBALANCED_EVENT_GROUP",
                        "H_ABSTRACTION requires a radical source plus coupled old/new bond electron pairs.",
                        group_id=self.group_id,
                    )
                ]
        return []

    def canonical_signature(self) -> tuple[Any, ...]:
        """Return an order-invariant signature for simultaneous events."""
        moves = tuple(
            sorted(
                (
                    move.source.kind,
                    move.source.atom_maps,
                    move.target.kind,
                    move.target.atom_maps,
                    move.electron_count,
                    move.arrow_type,
                    move.coupling_id or "",
                )
                for move in self.moves
            )
        )
        return (self.macro or "", moves)

    @property
    def read_loci(self) -> frozenset[ElectronLocus]:
        return frozenset(move.source for move in self.moves)

    @property
    def write_loci(self) -> frozenset[ElectronLocus]:
        return frozenset(move.target for move in self.moves)

    def validate_pre_state(self, graph: Any) -> tuple[VerificationIssue, ...]:
        """Check source resources and simultaneous overconsumption on a graph.

        The graph is expected to use SynKit's scalar molecular-graph fields.
        This method does not mutate it; replay is introduced in Sprint 4.
        """
        issues = list(self.issues())
        by_map = {
            int(attrs.get("atom_map", 0) or 0): node
            for node, attrs in graph.nodes(data=True)
            if int(attrs.get("atom_map", 0) or 0) > 0
        }
        requested: dict[ElectronLocus, int] = {}
        for move in self.moves:
            requested[move.source] = requested.get(move.source, 0) + move.electron_count
            for locus, is_source in ((move.source, True), (move.target, False)):
                if any(atom_map not in by_map for atom_map in locus.atom_maps):
                    issues.append(
                        VerificationIssue(
                            (
                                "SOURCE_LOCUS_ABSENT"
                                if is_source
                                else "TARGET_LOCUS_INVALID"
                            ),
                            f"{locus.kind} locus references an atom absent from the pre-state.",
                            group_id=self.group_id,
                            atom_maps=locus.atom_maps,
                        )
                    )
                    continue
                if is_source and self._available_electrons(graph, by_map, locus) <= 0:
                    issues.append(
                        VerificationIssue(
                            "SOURCE_LOCUS_ABSENT",
                            f"Source {locus.kind} resource is absent.",
                            group_id=self.group_id,
                            atom_maps=locus.atom_maps,
                        )
                    )
        for locus, count in requested.items():
            available = self._available_electrons(graph, by_map, locus)
            if count > available:
                issues.append(
                    VerificationIssue(
                        "LOCUS_OVERCONSUMED",
                        f"{count} electrons requested from {locus.kind}; only {available} available.",
                        group_id=self.group_id,
                        atom_maps=locus.atom_maps,
                        expected=available,
                        observed=count,
                    )
                )
        return tuple(issues)

    @staticmethod
    def _available_electrons(
        graph: Any, by_map: Mapping[int, Any], locus: ElectronLocus
    ) -> int:
        if locus.kind == LONE_PAIR:
            return 2 * int(graph.nodes[by_map[locus.atom_maps[0]]].get("lone_pairs", 0))
        if locus.kind == RADICAL:
            return int(graph.nodes[by_map[locus.atom_maps[0]]].get("radical", 0))
        left, right = (by_map[atom_map] for atom_map in locus.atom_maps)
        if not graph.has_edge(left, right):
            return 0
        field = "sigma_order" if locus.kind == SIGMA else "pi_order"
        return int(round(2 * float(graph.edges[left, right].get(field, 0))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "macro": self.macro,
            "moves": [move.to_dict() for move in self.moves],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ElectronMoveGroup":
        return cls(
            group_id=value["group_id"],
            macro=value.get("macro"),
            moves=tuple(ElectronMove.from_dict(move) for move in value["moves"]),
            metadata=value.get("metadata", {}),
        )


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
    atoms: tuple[int | str | None, ...]
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_class": self.descriptor_class,
            "atoms": list(self.atoms),
            "parity": self.parity,
            "state": self.state,
            "provenance": self.provenance,
        }

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
    descriptor_target: tuple[str, int]
    effect: StereoEffectKind
    before: StereoDescriptor | None = None
    after: StereoDescriptor | None = None
    provenance: str = "annotated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_target": list(self.descriptor_target),
            "effect": self.effect,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "provenance": self.provenance,
        }

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

    def __post_init__(self) -> None:
        if self.mapped_reaction.count(">>") != 1:
            raise MechanismModelError("mapped_reaction must contain exactly one '>>'.")
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise MechanismModelError("Mechanistic step IDs must be unique.")
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))

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
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MechanismRecord":
        return cls(
            mapped_reaction=value["mapped_reaction"],
            steps=tuple(MechanisticStep.from_dict(step) for step in value["steps"]),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            provenance=value.get("provenance", {}),
            metadata=value.get("metadata", {}),
        )


def issues_are_errors(issues: Iterable[VerificationIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)
