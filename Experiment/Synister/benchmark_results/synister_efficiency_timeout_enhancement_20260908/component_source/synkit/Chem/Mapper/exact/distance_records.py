"""Result and certificate records for exact chemical-distance enumeration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

DistanceTarget = Literal["minimal"] | float
EnumerationStatus = Literal["complete", "no_solutions", "timeout"]


class ExactEnumerationLimitError(ValueError):
    """Raised when a complete enumeration would exceed its explicit cap."""


class CertificateVerificationError(ValueError):
    """Raised when an exact-distance certificate fails independent replay."""


def normalize_distance_target(CD: str | Real) -> DistanceTarget:
    """Validate and normalize ``CD='minimal'`` or a non-negative number."""
    if isinstance(CD, str):
        if CD.lower() != "minimal":
            raise ValueError("CD must be 'minimal' or a non-negative number")
        return "minimal"
    if isinstance(CD, bool) or not isinstance(CD, Real):
        raise TypeError("CD must be 'minimal' or a non-negative number")
    target = float(CD)
    if not math.isfinite(target) or target < 0:
        raise ValueError("CD must be finite and non-negative")
    return target


@dataclass(frozen=True)
class DistanceEnumerationCertificate:
    """Replayable complete-cover certificate for one exact-CD query."""

    schema_version: int
    kind: str
    input_sha256: str
    target: DistanceTarget
    binary: bool
    tolerance: float
    reactant_order: tuple[int, ...]
    terminal_prefixes: tuple[tuple[int, ...], ...]
    frontier_prefixes: tuple[tuple[int, ...], ...]
    total_bijections: int
    maximum_cost_upper_bound: float
    selected_mapping_count: int
    selected_mappings_sha256: str
    cost: float | None
    pruning_limit: float | None
    status: EnumerationStatus
    certificate_sha256: str
    mapping_scope: str = "complete_atom_compatible_assignment_space"
    symmetry_prefixes: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    lower_bound_prefixes: tuple[tuple[int, ...], ...] = ()
    upper_bound_prefixes: tuple[tuple[int, ...], ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible certificate record."""
        record = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "input_sha256": self.input_sha256,
            "target": self.target,
            "binary": self.binary,
            "tolerance": self.tolerance,
            "reactant_order": list(self.reactant_order),
            "terminal_prefixes": [list(value) for value in self.terminal_prefixes],
            "frontier_prefixes": [list(value) for value in self.frontier_prefixes],
            "total_bijections": self.total_bijections,
            "maximum_cost_upper_bound": self.maximum_cost_upper_bound,
            "selected_mapping_count": self.selected_mapping_count,
            "selected_mappings_sha256": self.selected_mappings_sha256,
            "cost": self.cost,
            "pruning_limit": self.pruning_limit,
            "status": self.status,
            "certificate_sha256": self.certificate_sha256,
        }
        if self.schema_version >= 2:
            record["mapping_scope"] = self.mapping_scope
            record["symmetry_prefixes"] = [
                [list(prefix), list(witness)]
                for prefix, witness in self.symmetry_prefixes
            ]
        if self.schema_version >= 3:
            record["lower_bound_prefixes"] = [
                list(prefix) for prefix in self.lower_bound_prefixes
            ]
        if self.schema_version >= 4:
            record["upper_bound_prefixes"] = [
                list(prefix) for prefix in self.upper_bound_prefixes
            ]
        return record

    @classmethod
    def from_dict(cls, record: dict[str, object]):
        """Restore a certificate from its JSON-compatible representation."""
        return cls(
            schema_version=int(record["schema_version"]),
            kind=str(record["kind"]),
            input_sha256=str(record["input_sha256"]),
            target=record["target"],
            binary=bool(record["binary"]),
            tolerance=float(record["tolerance"]),
            reactant_order=tuple(int(value) for value in record["reactant_order"]),
            terminal_prefixes=tuple(
                tuple(int(value) for value in prefix)
                for prefix in record["terminal_prefixes"]
            ),
            frontier_prefixes=tuple(
                tuple(int(value) for value in prefix)
                for prefix in record.get("frontier_prefixes", [])
            ),
            total_bijections=int(record["total_bijections"]),
            maximum_cost_upper_bound=float(record["maximum_cost_upper_bound"]),
            selected_mapping_count=int(record["selected_mapping_count"]),
            selected_mappings_sha256=str(record["selected_mappings_sha256"]),
            cost=None if record["cost"] is None else float(record["cost"]),
            pruning_limit=(
                None
                if record.get("pruning_limit") is None
                else float(record["pruning_limit"])
            ),
            status=record["status"],
            certificate_sha256=str(record["certificate_sha256"]),
            mapping_scope=str(
                record.get(
                    "mapping_scope",
                    "complete_atom_compatible_assignment_space",
                )
            ),
            symmetry_prefixes=tuple(
                (
                    tuple(int(value) for value in prefix),
                    tuple(int(value) for value in witness),
                )
                for prefix, witness in record.get("symmetry_prefixes", [])
            ),
            lower_bound_prefixes=tuple(
                tuple(int(value) for value in prefix)
                for prefix in record.get("lower_bound_prefixes", [])
            ),
            upper_bound_prefixes=tuple(
                tuple(int(value) for value in prefix)
                for prefix in record.get("upper_bound_prefixes", [])
            ),
        )


@dataclass
class DistanceEnumerationResult:
    """Complete mappings and search metadata for one exact-CD query."""

    target: DistanceTarget
    cost: float | None
    minimum_cost: float | None
    mappings: list[list[int]]
    distances: list[float]
    total_bijections: int
    maximum_cost_upper_bound: float
    visited_leaves: int
    pruned_branches: int
    elapsed_seconds: float
    status: EnumerationStatus
    complete: bool
    truncation_reason: str | None = None
    scope: str = "complete_atom_compatible_assignment_space"
    certificate: DistanceEnumerationCertificate | None = None
    visited_nodes: int = 0
    symmetry_pruned_branches: int = 0
    symmetry_automorphism_count: int = 1
    symmetry_search_complete: bool = True
    lower_bound_pruned_branches: int = 0
    upper_bound_pruned_branches: int = 0
    selected_mapping_count: int = 0
    symmetry_group_order: int | None = 1
    symmetry_quotient_complete: bool = True
    selected_labeled_mapping_count: int | None = 0
    backend: str = "assignment_branch_and_bound"
    backend_statistics: dict[str, object] | None = None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _typed_element(value: object) -> tuple[str, str]:
    return (
        f"{type(value).__module__}.{type(value).__qualname__}",
        repr(value),
    )


def _input_sha256(reactant, product, reactant_elements, product_elements, binary):
    payload = {
        "binary": bool(binary),
        "reactant": reactant.tolist(),
        "product": product.tolist(),
        "reactant_elements": [_typed_element(value) for value in reactant_elements],
        "product_elements": [_typed_element(value) for value in product_elements],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _mappings_sha256(mappings) -> str:
    normalized = sorted(tuple(int(image) for image in mapping) for mapping in mappings)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _certificate_sha256(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("certificate_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _make_certificate(**kwargs) -> DistanceEnumerationCertificate:
    unsigned = DistanceEnumerationCertificate(certificate_sha256="", **kwargs)
    digest = _certificate_sha256(unsigned.as_dict())
    return DistanceEnumerationCertificate(certificate_sha256=digest, **kwargs)


__all__ = [
    "CertificateVerificationError",
    "DistanceEnumerationCertificate",
    "DistanceEnumerationResult",
    "DistanceTarget",
    "EnumerationStatus",
    "ExactEnumerationLimitError",
    "normalize_distance_target",
]
