"""Versioned RDKit identity profiles and complete resonance-family keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
import json
from typing import Any

from .canonical import CanonicalStereographResult, _rdkit_graph_and_registry
from .configured import (
    MirrorIdentityProfile,
    _mirror_profile_options,
    _resonance_graphs,
    canonicalize_configured_registry,
)
from .descriptors import StereoValue

RDKIT_IDENTITY_SCHEMA = "synkit.rdkit-stereograph/1"
RESONANCE_FAMILY_SCHEMA = "synkit.resonance-family-stereograph/1"


def profile_certificate(result: CanonicalStereographResult, profile: str):
    """Namespace a single drawing's full key without changing its witnesses."""
    code = f"{RDKIT_IDENTITY_SCHEMA}\n{profile}\n{result.canonical_code}"
    return replace(
        result, schema=RDKIT_IDENTITY_SCHEMA, canonical_code=code,
        canonical_digest=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        identity_profile=profile,
    )


@dataclass(frozen=True)
class ResonanceFamilyCanonicalResult:
    """Identity of an entire declared resonance family.

    ``members`` retain one witness per distinct canonical drawing.  An atom
    order or automorphism of one member must not be interpreted as a unique
    atom order or the full automorphism group of the family.  Equality uses
    the complete sorted family, never a finite digest or a selected member.
    """

    members: tuple[CanonicalStereographResult, ...]
    schema: str = RESONANCE_FAMILY_SCHEMA
    identity_profile: str = "chemical"

    def __post_init__(self):
        if not self.members or any(not member.complete or not member.exact for member in self.members):
            raise ValueError("A family certificate requires complete exact members.")
        codes = tuple(member.canonical_code for member in self.members)
        if codes != tuple(sorted(set(codes))):
            raise ValueError("Family members must have sorted distinct full certificates.")

    @cached_property
    def canonical_code(self) -> str:
        return f"{self.schema}\n" + json.dumps(
            [self.identity_profile, [member.canonical_code for member in self.members]],
            ensure_ascii=True, separators=(",", ":"),
        )

    @cached_property
    def canonical_digest(self) -> str:
        return hashlib.sha256(self.canonical_code.encode("utf-8")).hexdigest()

    @property
    def canonical_certificate(self) -> str:
        return self.canonical_code

    @property
    def complete(self) -> bool:
        return True

    @property
    def exact(self) -> bool:
        return True

    def same_stereograph(self, other: object) -> bool:
        return (
            isinstance(other, ResonanceFamilyCanonicalResult)
            and self.schema == other.schema
            and self.identity_profile == other.identity_profile
            and self.canonical_code == other.canonical_code
        )


RDKitIdentityResult = CanonicalStereographResult | ResonanceFamilyCanonicalResult


def canonicalize_rdkit_registry(
    molecule: Any,
    registry: Mapping[str, StereoValue],
    *,
    identity_profile: MirrorIdentityProfile,
) -> RDKitIdentityResult:
    """Canonize a registry in the molecule's shared atom-handle namespace."""
    options = _mirror_profile_options(identity_profile)
    if identity_profile != "chemical":
        graph, _source = _rdkit_graph_and_registry(molecule)
        return profile_certificate(
            canonicalize_configured_registry(graph, registry, **options), identity_profile,
        )
    members = {}
    for graph in _resonance_graphs(molecule):
        result = profile_certificate(canonicalize_configured_registry(graph, registry), "lewis_state")
        members.setdefault(result.canonical_code, result)
    return ResonanceFamilyCanonicalResult(tuple(members[code] for code in sorted(members)))


def canonicalize_rdkit_identity(
    molecule: Any,
    *,
    identity_profile: MirrorIdentityProfile = "lewis_state",
) -> RDKitIdentityResult:
    """Produce a full source-declared configured identity in one profile."""
    _graph, registry = _rdkit_graph_and_registry(molecule)
    return canonicalize_rdkit_registry(molecule, registry, identity_profile=identity_profile)
