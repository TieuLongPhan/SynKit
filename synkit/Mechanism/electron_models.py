"""Electron-flow primitives and validated move groups."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Sequence

from .symbols import (
    CanonicalLocus,
    LONE_PAIR,
    PI,
    RADICAL,
    SIGMA,
    normalize_locus_symbol,
)

LocusKind = CanonicalLocus
ArrowType = Literal["curved", "fishhook"]


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
            raise MechanismModelError(
                "INVALID_ELECTRON_COUNT: electron_count must be 1 or 2."
            )
        expected = "fishhook" if self.electron_count == 1 else "curved"
        if self.arrow_type != expected:
            raise MechanismModelError(
                f"ARROW_ELECTRON_COUNT_MISMATCH: {self.electron_count} electron "
                f"moves require arrow_type={expected!r}."
            )
        if not str(self.group_id).strip():
            raise MechanismModelError("Electron moves require a non-empty group_id.")
        if self.source == self.target:
            raise MechanismModelError(
                "Electron moves must change locus; source and target are identical."
            )
        if self.electron_count == 2 and RADICAL in {
            self.source.kind,
            self.target.kind,
        }:
            raise MechanismModelError(
                "RADICAL_LOCUS_REQUIRES_FISHHOOK: radical loci carry "
                "one-electron moves."
            )
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

    def reversed(self) -> "ElectronMove":
        """Return the same electron transfer in the opposite direction."""
        return ElectronMove(
            source=self.target,
            target=self.source,
            electron_count=self.electron_count,
            arrow_type=self.arrow_type,
            group_id=self.group_id,
            event_id=self.event_id,
            coupling_id=self.coupling_id,
            metadata=self.metadata,
        )

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
            "LONE_PAIR_RADICAL_RELOCATION",
        }
    )
    REVERSE_MACRO: ClassVar[Mapping[str, str]] = {
        "HOMOLYSIS": "RECOMBINATION",
        "RECOMBINATION": "HOMOLYSIS",
        "RADICAL_ADDITION": "BETA_SCISSION",
        "BETA_SCISSION": "RADICAL_ADDITION",
        "H_ABSTRACTION": "H_ABSTRACTION",
        "RADICAL_RESONANCE": "RADICAL_RESONANCE",
        "LONE_PAIR_RADICAL_RELOCATION": "LONE_PAIR_RADICAL_RELOCATION",
    }

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
                    if self.macro != "LONE_PAIR_RADICAL_RELOCATION":
                        issues.append(
                            VerificationIssue(
                                code="MISSING_COUPLED_FISHHOOK",
                                message=(
                                    "Fishhook moves require a coupling_id before commit."
                                ),
                                group_id=self.group_id,
                            )
                        )
                else:
                    coupled.setdefault(move.coupling_id, []).append(move)
        for coupling_id, moves in coupled.items():
            if len(moves) < 2 and self.macro != "LONE_PAIR_RADICAL_RELOCATION":
                issues.append(
                    VerificationIssue(
                        code="MISSING_COUPLED_FISHHOOK",
                        message=f"Coupling {coupling_id!r} has fewer than two fishhooks.",
                        group_id=self.group_id,
                    )
                )
        issues.extend(self._macro_issues())
        return tuple(issues)

    def _macro_issues(self) -> list[VerificationIssue]:  # noqa: C901
        if not self.macro:
            return []
        fishhooks = [move for move in self.moves if move.electron_count == 1]
        pairs = Counter((move.source.kind, move.target.kind) for move in fishhooks)

        def pair_counts(value: Counter) -> dict[str, int]:
            return {
                f"{source}->{target}": count
                for (source, target), count in value.items()
            }

        if self.macro == "LONE_PAIR_RADICAL_RELOCATION":
            valid = (
                len(fishhooks) == 1
                and fishhooks[0].source.kind == LONE_PAIR
                and fishhooks[0].target.kind == LONE_PAIR
                and fishhooks[0].source.atom_maps != fishhooks[0].target.atom_maps
            )
            if valid:
                return []
            return [
                VerificationIssue(
                    code="UNBALANCED_EVENT_GROUP",
                    message=(
                        "LONE_PAIR_RADICAL_RELOCATION requires one fishhook "
                        "between distinct atom-local lone-pair loci."
                    ),
                    group_id=self.group_id,
                    expected={"lp->lp": 1},
                    observed={
                        "fishhook_count": len(fishhooks),
                        "locus_pairs": pair_counts(pairs),
                    },
                )
            ]

        if len(fishhooks) < 2:
            return [
                VerificationIssue(
                    code="UNBALANCED_EVENT_GROUP",
                    message=f"{self.macro} requires coupled one-electron moves.",
                    group_id=self.group_id,
                    expected={"minimum_fishhooks": 2},
                    observed={
                        "fishhook_count": len(fishhooks),
                        "locus_pairs": pair_counts(pairs),
                    },
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
                and {target.atom_maps[0] for target in targets}
                == set(next(iter(sources)).atom_maps)
            )
            if not valid:
                return [
                    VerificationIssue(
                        "UNBALANCED_EVENT_GROUP",
                        "HOMOLYSIS requires one sigma source and two radical targets.",
                        group_id=self.group_id,
                        expected={"σ->∙": 2},
                        observed=pair_counts(pairs),
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
                and {source.atom_maps[0] for source in sources}
                == set(next(iter(targets)).atom_maps)
            )
            if not valid:
                return [
                    VerificationIssue(
                        "UNBALANCED_EVENT_GROUP",
                        "RECOMBINATION requires two radical sources and one sigma target.",
                        group_id=self.group_id,
                        expected={"∙->σ": 2},
                        observed=pair_counts(pairs),
                    )
                ]
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
                    expected=pair_counts(expected),
                    observed=pair_counts(pairs),
                )
            ]
        if expected is not None and not self._macro_incidence_is_valid(fishhooks):
            return [
                VerificationIssue(
                    "MACRO_INCIDENCE_MISMATCH",
                    f"{self.macro} loci do not share the required reacting atoms.",
                    group_id=self.group_id,
                )
            ]
        if self.macro == "H_ABSTRACTION":
            simple_h_abstraction = Counter(
                {
                    (RADICAL, SIGMA): 1,
                    (SIGMA, SIGMA): 1,
                    (SIGMA, RADICAL): 1,
                }
            )
            expected_h_abstraction = Counter(
                {
                    (RADICAL, SIGMA): 1,
                    (SIGMA, SIGMA): 1,
                    (SIGMA, PI): 1,
                    (RADICAL, PI): 1,
                }
            )
            reverse_h_abstraction = Counter(
                {
                    (target, source): count
                    for (source, target), count in expected_h_abstraction.items()
                }
            )
            valid = pairs in (
                simple_h_abstraction,
                expected_h_abstraction,
                reverse_h_abstraction,
            ) and self._h_abstraction_incidence_is_valid(fishhooks)
            if not valid:
                return [
                    VerificationIssue(
                        "UNBALANCED_EVENT_GROUP",
                        "H_ABSTRACTION requires a radical source plus coupled old/new bond electron pairs.",
                        group_id=self.group_id,
                        expected={
                            "allowed": [
                                pair_counts(simple_h_abstraction),
                                pair_counts(expected_h_abstraction),
                                pair_counts(reverse_h_abstraction),
                            ]
                        },
                        observed=pair_counts(pairs),
                    )
                ]
        return []

    @staticmethod
    def _single_move(
        moves: Sequence[ElectronMove], source: str, target: str
    ) -> ElectronMove:
        return next(
            move
            for move in moves
            if move.source.kind == source and move.target.kind == target
        )

    def _macro_incidence_is_valid(self, moves: Sequence[ElectronMove]) -> bool:
        """Check atom incidence for the three allylic radical macros."""
        if self.macro == "RADICAL_ADDITION":
            radical_to_new = self._single_move(moves, RADICAL, SIGMA)
            old_to_new = self._single_move(moves, PI, SIGMA)
            old_to_radical = self._single_move(moves, PI, RADICAL)
        elif self.macro == "BETA_SCISSION":
            radical_to_new = self._single_move(moves, RADICAL, PI)
            old_to_new = self._single_move(moves, SIGMA, PI)
            old_to_radical = self._single_move(moves, SIGMA, RADICAL)
        elif self.macro == "RADICAL_RESONANCE":
            radical_to_new = self._single_move(moves, RADICAL, PI)
            old_to_new = self._single_move(moves, PI, PI)
            old_to_radical = self._single_move(moves, PI, RADICAL)
        else:
            return True

        if radical_to_new.target != old_to_new.target:
            return False
        if old_to_new.source != old_to_radical.source:
            return False
        old_atoms = set(old_to_new.source.atom_maps)
        new_atoms = set(old_to_new.target.atom_maps)
        shared = old_atoms & new_atoms
        return (
            len(shared) == 1
            and {radical_to_new.source.atom_maps[0]} == new_atoms - shared
            and {old_to_radical.target.atom_maps[0]} == old_atoms - shared
        )

    @classmethod
    def _h_abstraction_incidence_is_valid(cls, moves: Sequence[ElectronMove]) -> bool:
        pairs = Counter((move.source.kind, move.target.kind) for move in moves)
        reverse_conjugated = Counter(
            {
                (SIGMA, RADICAL): 1,
                (SIGMA, SIGMA): 1,
                (PI, SIGMA): 1,
                (PI, RADICAL): 1,
            }
        )
        if pairs == reverse_conjugated:
            return cls._h_abstraction_incidence_is_valid(
                tuple(move.reversed() for move in moves)
            )

        radical_to_new_sigma = cls._single_move(moves, RADICAL, SIGMA)
        old_to_new_sigma = cls._single_move(moves, SIGMA, SIGMA)

        if radical_to_new_sigma.target != old_to_new_sigma.target:
            return False

        old_sigma = set(old_to_new_sigma.source.atom_maps)
        new_sigma = set(old_to_new_sigma.target.atom_maps)
        hydrogen = old_sigma & new_sigma
        if len(hydrogen) != 1:
            return False
        donor = old_sigma - hydrogen
        acceptor = new_sigma - hydrogen
        if {radical_to_new_sigma.source.atom_maps[0]} != acceptor:
            return False

        if pairs[(SIGMA, RADICAL)] == 1:
            old_to_radical = cls._single_move(moves, SIGMA, RADICAL)
            return (
                old_to_radical.source == old_to_new_sigma.source
                and {old_to_radical.target.atom_maps[0]} == donor
            )

        old_to_new_pi = cls._single_move(moves, SIGMA, PI)
        radical_to_new_pi = cls._single_move(moves, RADICAL, PI)
        if (
            old_to_new_sigma.source != old_to_new_pi.source
            or old_to_new_pi.target != radical_to_new_pi.target
        ):
            return False
        new_pi = set(old_to_new_pi.target.atom_maps)
        return (
            donor <= new_pi
            and {radical_to_new_pi.source.atom_maps[0]} == new_pi - donor
        )

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

    def reversed(self) -> "ElectronMoveGroup":
        """Reverse every move and select the inverse radical macro."""
        return ElectronMoveGroup(
            group_id=self.group_id,
            moves=tuple(move.reversed() for move in self.moves),
            macro=self.REVERSE_MACRO[self.macro] if self.macro else None,
            metadata=self.metadata,
        )

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
            if any(atom_map not in by_map for atom_map in locus.atom_maps):
                continue
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
        if self.macro == "LONE_PAIR_RADICAL_RELOCATION" and len(self.moves) == 1:
            move = self.moves[0]
            target_radical = ElectronLocus(RADICAL, move.target.atom_maps)
            if all(atom_map in by_map for atom_map in target_radical.atom_maps) and (
                self._available_electrons(graph, by_map, target_radical) < 1
            ):
                issues.append(
                    VerificationIssue(
                        "TARGET_RADICAL_ABSENT",
                        "Lone-pair radical relocation requires a radical acceptor.",
                        group_id=self.group_id,
                        atom_maps=move.target.atom_maps,
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
