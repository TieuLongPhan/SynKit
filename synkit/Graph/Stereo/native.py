"""Opt-in native certificates on the existing faithful configured expansion.

The experimental sparse convention is independent of the reference /2 JSON
format. Input is an already prepared attributed constitution and registry;
this module does not infer missing carriers or perform resonance normalization.
"""

from dataclasses import dataclass, field
import hashlib
import struct
from time import perf_counter

from synkit.Graph.Canon._native import canonicalize_native_graph
from synkit.Graph.Canon._native.search import _cap, _timeout
from synkit.Graph.Canon.exact import CanonicalSearchIncomplete

from .canonical import (
    AtomVertex, StereographMirrorStatus, _DEFAULT_ATOM_COLOR_KEYS, _DEFAULT_BOND_COLOR_KEYS,
)
from .configured import CONFIGURED_DESCRIPTOR_TYPES, expand_configured_stereograph, mirror_configured_descriptor
from .orbits import StereoSpecification

NATIVE_STEREOGRAPH_SCHEMA = "synkit.native-sparse-stereograph/1"


@dataclass(frozen=True)
class NativeStereographResult:
    """A new-convention configured key or an explicit nondefinitive outcome."""

    complete: bool
    reason: str | None
    canonical_code: bytes | None = None
    atom_order: tuple = ()
    statistics: dict = field(default_factory=dict)
    identity_profile: str = "attributed_graph"
    schema: str = NATIVE_STEREOGRAPH_SCHEMA

    @property
    def canonical_digest(self):
        return hashlib.sha256(self.canonical_code).hexdigest() if self.complete else None

    def require_complete(self):
        """Refuse to use an incomplete representation or stopped search as identity."""
        if not self.complete:
            raise CanonicalSearchIncomplete(f"Native stereograph is incomplete: {self.reason}")
        return self

    def same_stereograph(self, other):
        """Compare full native keys only within the same declared convention."""
        self.require_complete()
        if not isinstance(other, NativeStereographResult) or other.schema != self.schema:
            raise TypeError("Cannot compare different stereograph certificate conventions")
        other.require_complete()
        if self.identity_profile != other.identity_profile:
            raise ValueError("Identity profiles differ; reassess under one shared profile")
        return self.canonical_code == other.canonical_code


@dataclass(frozen=True)
class NativeMirrorResult:
    """Mirror classification, retaining separate completion for both legs."""

    status: StereographMirrorStatus
    reason: str | None = None
    original: NativeStereographResult | None = None
    mirror: NativeStereographResult | None = None
    atom_mirror_isomorphism: tuple | None = None

    @property
    def is_definitive(self):
        return self.status in {StereographMirrorStatus.ACHIRAL, StereographMirrorStatus.CHIRAL}


def _remaining(seconds, started):
    return None if seconds < 0 else max(0.0, seconds - (perf_counter() - started))


def canonicalize_native_stereograph(
    base_graph, descriptors, *, library_path=None,
    atom_color=_DEFAULT_ATOM_COLOR_KEYS, bond_color=_DEFAULT_BOND_COLOR_KEYS,
    identity_profile="attributed_graph", timeout_seconds=None,
    max_search_nodes=None, max_workspace_bytes=None, max_output_bytes=None,
    max_depth=1024, cancel_token=None,
):
    """Canonicalize the supplied registry using the unchanged Python expansion.

    Profile names label caller-prepared attributes; no chemical normalization
    is implied. All supplied descriptors must be configured. Latent unlisted
    carriers are outside this source-declared boundary. The old APIs are unchanged.
    """
    started = perf_counter()
    seconds = _timeout(timeout_seconds)
    _cap(max_search_nodes, "max_search_nodes")
    _cap(max_workspace_bytes, "max_workspace_bytes")
    _cap(max_output_bytes, "max_output_bytes")
    if not isinstance(identity_profile, str) or not identity_profile:
        raise ValueError("identity_profile must be a nonempty string identifying prepared attributes")
    descriptors = tuple(descriptors.values() if hasattr(descriptors, "values") else descriptors)
    if any(not isinstance(d, CONFIGURED_DESCRIPTOR_TYPES) for d in descriptors):
        return NativeStereographResult(False, "unsupported_descriptor", identity_profile=identity_profile)
    if any(d.specification is not StereoSpecification.FIXED for d in descriptors):
        return NativeStereographResult(False, "incomplete_configuration", identity_profile=identity_profile)
    if seconds == 0 or (seconds > 0 and perf_counter() - started >= seconds):
        return NativeStereographResult(False, "time_limit", identity_profile=identity_profile)
    if cancel_token is not None and cancel_token.cancelled:
        return NativeStereographResult(False, "cancelled", identity_profile=identity_profile)
    expanded = expand_configured_stereograph(base_graph, descriptors, atom_color=atom_color, bond_color=bond_color)
    expansion_seconds = perf_counter() - started
    profile = identity_profile.encode("utf-8")
    prefix = NATIVE_STEREOGRAPH_SCHEMA.encode("ascii") + b"\0" + struct.pack("<Q", len(profile)) + profile
    limit = max_output_bytes - len(prefix) if max_output_bytes is not None else None
    if limit is not None and limit <= 0:
        return NativeStereographResult(False, "output_limit", identity_profile=identity_profile)
    result = canonicalize_native_graph(
        expanded, library_path=library_path, node_color="color", timeout_seconds=_remaining(seconds, started),
        max_search_nodes=max_search_nodes, max_workspace_bytes=max_workspace_bytes,
        max_output_bytes=limit, max_depth=max_depth, cancel_token=cancel_token,
    )
    metrics = dict(result.statistics, expansion_seconds=expansion_seconds,
                   auxiliary_vertices=len(expanded), auxiliary_edges=expanded.number_of_edges())
    if not result.complete:
        return NativeStereographResult(False, result.reason, statistics=metrics, identity_profile=identity_profile)
    code = prefix + result.canonical_code
    order = tuple(v.reference for v in result.canonical_order if isinstance(v, AtomVertex))
    metrics["total_seconds"] = perf_counter() - started
    if seconds >= 0 and metrics["total_seconds"] >= seconds:
        return NativeStereographResult(False, "time_limit", statistics=metrics, identity_profile=identity_profile)
    return NativeStereographResult(True, None, code, order, metrics, identity_profile)


def classify_native_stereograph_mirror(base_graph, descriptors, *, timeout_seconds=None, **options):
    """Assess the supplied configured registry; both mirror legs share one deadline."""
    started = perf_counter()
    seconds = _timeout(timeout_seconds)
    descriptors = tuple(descriptors.values() if hasattr(descriptors, "values") else descriptors)
    original = canonicalize_native_stereograph(
        base_graph, descriptors, timeout_seconds=_remaining(seconds, started), **options,
    )
    if not original.complete:
        status = (StereographMirrorStatus.UNSUPPORTED if original.reason == "unsupported_descriptor"
                  else StereographMirrorStatus.INCOMPLETE)
        return NativeMirrorResult(status, original.reason, original)
    mirrored = tuple(mirror_configured_descriptor(d) for d in descriptors)
    mirror = canonicalize_native_stereograph(
        base_graph, mirrored, timeout_seconds=_remaining(seconds, started), **options,
    )
    if not mirror.complete:
        return NativeMirrorResult(StereographMirrorStatus.INCOMPLETE, mirror.reason, original, mirror)
    equal = original.same_stereograph(mirror)
    mapping = tuple(zip(original.atom_order, mirror.atom_order)) if equal else None
    if seconds >= 0 and perf_counter() - started >= seconds:
        return NativeMirrorResult(StereographMirrorStatus.INCOMPLETE, "time_limit", original, mirror)
    status = StereographMirrorStatus.ACHIRAL if equal else StereographMirrorStatus.CHIRAL
    return NativeMirrorResult(status, None, original, mirror, mapping)
