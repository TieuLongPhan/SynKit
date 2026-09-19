"""Reference-blinded exact-shell analysis for Synister experiments."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

from .exact.hybrid import enumerate_hybrid_distance_mappings
from .slap.lap import _adjacency_and_elements, chemical_distance
from .slap.lap import recover_mapping
from .slap.sequential import GraphMatcher
from .spectrum import (
    ExactStructureSpectrum,
    ExactStructureSpectrumAccumulator,
)

ShellTargetMode = Literal["reference_cd", "minimal"]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _validate_optional_positive_integer(value, name):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True)
class GlobalShellConfig:
    """Resource and exact-search policy for one blinded global shell."""

    binary: bool = False
    time_limit_seconds: float | None = 300.0
    max_bijections: int | None = None
    max_mappings: int | None = 100_000
    tolerance: float = 1e-9
    symmetry_pruning: bool = True
    max_symmetry_automorphisms: int = 256
    symmetry_timeout_seconds: float = 0.25
    symmetry_max_search_nodes: int = 10_000
    symmetry_node_properties: tuple[str, ...] = ("hcounts", "charges")
    reaction_center_properties: tuple[str, ...] = ("hcounts", "charges")
    backend: str = "auto"
    max_edit_support_pairs: int | None = 50_000
    use_slap_seed: bool = True
    structure_analysis: bool = True
    template_radius: int = 1
    structure_timeout_seconds: float = 0.25
    structure_max_search_nodes: int = 100_000

    def __post_init__(self):
        if not isinstance(self.binary, bool):
            raise TypeError("binary must be boolean")
        if self.time_limit_seconds is not None and (
            isinstance(self.time_limit_seconds, bool)
            or not isinstance(self.time_limit_seconds, Real)
            or not math.isfinite(float(self.time_limit_seconds))
            or self.time_limit_seconds < 0
        ):
            raise ValueError("time_limit_seconds must be finite and non-negative")
        _validate_optional_positive_integer(self.max_bijections, "max_bijections")
        _validate_optional_positive_integer(self.max_mappings, "max_mappings")
        if self.tolerance < 0 or not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and non-negative")
        if not isinstance(self.symmetry_pruning, bool):
            raise TypeError("symmetry_pruning must be boolean")
        if not isinstance(self.structure_analysis, bool):
            raise TypeError("structure_analysis must be boolean")
        if not isinstance(self.use_slap_seed, bool):
            raise TypeError("use_slap_seed must be boolean")
        if self.backend not in {"auto", "assignment", "edit_support"}:
            raise ValueError("backend must be 'auto', 'assignment', or 'edit_support'")
        _validate_optional_positive_integer(
            self.max_edit_support_pairs,
            "max_edit_support_pairs",
        )
        if (
            isinstance(self.template_radius, bool)
            or not isinstance(self.template_radius, Integral)
            or self.template_radius < 0
        ):
            raise ValueError("template_radius must be a non-negative integer")
        _validate_optional_positive_integer(
            self.max_symmetry_automorphisms, "max_symmetry_automorphisms"
        )
        _validate_optional_positive_integer(
            self.symmetry_max_search_nodes, "symmetry_max_search_nodes"
        )
        if (
            isinstance(self.symmetry_timeout_seconds, bool)
            or not isinstance(self.symmetry_timeout_seconds, Real)
            or not math.isfinite(float(self.symmetry_timeout_seconds))
            or self.symmetry_timeout_seconds < 0
        ):
            raise ValueError("symmetry_timeout_seconds must be finite and non-negative")
        if (
            isinstance(self.structure_timeout_seconds, bool)
            or not isinstance(self.structure_timeout_seconds, Real)
            or not math.isfinite(float(self.structure_timeout_seconds))
            or self.structure_timeout_seconds < 0
        ):
            raise ValueError(
                "structure_timeout_seconds must be finite and non-negative"
            )
        _validate_optional_positive_integer(
            self.structure_max_search_nodes,
            "structure_max_search_nodes",
        )


@dataclass(frozen=True)
class ReactionCenterSpectrum:
    """Exact reaction-centre statistics over observed shell representatives."""

    representative_count: int
    frequency_denominator: int
    frequency_scope: str
    bond_union: tuple[tuple[int, int], ...]
    bond_intersection: tuple[tuple[int, int], ...]
    bond_change_counts: tuple[tuple[int, int, int], ...]
    atom_union: tuple[int, ...]
    atom_intersection: tuple[int, ...]
    atom_change_counts: tuple[tuple[int, int], ...]
    unary_properties: tuple[str, ...]
    representative_stream_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "representative_count": self.representative_count,
            "frequency_denominator": self.frequency_denominator,
            "frequency_scope": self.frequency_scope,
            "bond_union": [list(pair) for pair in self.bond_union],
            "bond_intersection": [list(pair) for pair in self.bond_intersection],
            "bond_change_counts": [list(item) for item in self.bond_change_counts],
            "atom_union": list(self.atom_union),
            "atom_intersection": list(self.atom_intersection),
            "atom_change_counts": [list(item) for item in self.atom_change_counts],
            "unary_properties": list(self.unary_properties),
            "representative_stream_sha256": self.representative_stream_sha256,
        }


@dataclass(frozen=True)
class GlobalShellAnalysisResult:
    """JSON-ready result of blind search followed by reference reveal."""

    target_mode: ShellTargetMode
    target: str | float
    binary: bool
    status: str
    complete: bool
    truncation_reason: str | None
    scope: str
    minimum_cost: float | None
    reference_cd: float
    reference_gap_from_minimum: float | None
    reference_mapping_observed: bool
    reference_class_observed: bool
    shell_complete_and_reference_class_observed: bool
    reference_is_global_minimum_proven: bool
    total_bijections: int
    representative_solution_count: int
    mapping_hartley_entropy_nats: float | None
    labeled_solution_count: int | None
    symmetry_group_order: int | None
    symmetry_quotient_complete: bool
    visited_nodes: int
    visited_leaves: int
    distance_pruned_branches: int
    lower_bound_pruned_branches: int
    upper_bound_pruned_branches: int
    symmetry_pruned_branches: int
    elapsed_seconds: float
    backend: str
    backend_statistics: dict[str, object] | None
    reaction_center: ReactionCenterSpectrum
    structure: ExactStructureSpectrum
    schema_version: int = 4
    kind: str = "synister_reference_blinded_global_shell"

    def as_dict(self) -> dict[str, object]:
        payload = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"reaction_center", "structure"}
        }
        payload["total_bijections"] = str(self.total_bijections)
        payload["labeled_solution_count"] = (
            None
            if self.labeled_solution_count is None
            else str(self.labeled_solution_count)
        )
        payload["symmetry_group_order"] = (
            None
            if self.symmetry_group_order is None
            else str(self.symmetry_group_order)
        )
        payload["reaction_center"] = self.reaction_center.as_dict()
        payload["structure"] = self.structure.as_dict()
        return payload


def _property_vectors(lgp, names):
    vectors = {}
    atom_count = len(lgp[0].labels)
    for name in names:
        reactant = lgp[0].props.get(name)
        product = lgp[1].props.get(name)
        if (
            reactant is not None
            and product is not None
            and len(reactant) == atom_count
            and len(product) == atom_count
        ):
            vectors[str(name)] = (tuple(reactant), tuple(product))
    return vectors


def _mapping_sha256(mapping) -> bytes:
    return hashlib.sha256(_canonical_json([int(value) for value in mapping])).digest()


def _transport_sha256(product, properties, mapping) -> bytes:
    images = np.asarray(mapping, dtype=int)
    transported = product[images[:, None], images[None, :]]
    payload = {
        "adjacency": transported.tolist(),
        "properties": {
            name: [product_values[image] for image in mapping]
            for name, (_, product_values) in properties.items()
        },
    }
    return hashlib.sha256(_canonical_json(payload)).digest()


class _BlindShellObserver:
    def __init__(
        self,
        reactant,
        product,
        elements,
        properties,
        config,
    ):
        self.reactant = reactant
        self.product = product
        self.properties = properties
        self.tolerance = config.tolerance
        self.count = 0
        self.bond_counts = Counter()
        self.atom_counts = Counter()
        self.mapping_hashes = set()
        self.transport_hashes = set()
        self.stream_digest = hashlib.sha256()
        self.structure = ExactStructureSpectrumAccumulator(
            reactant,
            product,
            elements,
            properties,
            enabled=config.structure_analysis,
            template_radius=config.template_radius,
            tolerance=config.tolerance,
            timeout_seconds=config.structure_timeout_seconds,
            max_search_nodes=config.structure_max_search_nodes,
        )

    def observe(self, mapping, cost):
        normalized = tuple(int(image) for image in mapping)
        self.count += 1
        self.mapping_hashes.add(_mapping_sha256(normalized))
        self.transport_hashes.add(
            _transport_sha256(self.product, self.properties, normalized)
        )
        payload = _canonical_json([list(normalized), float(cost)])
        self.stream_digest.update(len(payload).to_bytes(8, "little"))
        self.stream_digest.update(payload)
        self.structure.observe(normalized)
        images = np.asarray(normalized, dtype=int)
        transported = self.product[images[:, None], images[None, :]]
        for left in range(self.reactant.shape[0]):
            atom_changed = any(
                reactant_values[left] != product_values[normalized[left]]
                for reactant_values, product_values in self.properties.values()
            )
            if atom_changed:
                self.atom_counts[left] += 1
            for right in range(left + 1, self.reactant.shape[0]):
                if not math.isclose(
                    float(self.reactant[left, right]),
                    float(transported[left, right]),
                    abs_tol=self.tolerance,
                    rel_tol=0.0,
                ):
                    self.bond_counts[(left, right)] += 1

    def spectrum(self, frequency_scope):
        bonds = tuple(sorted(self.bond_counts))
        atoms = tuple(sorted(self.atom_counts))
        return ReactionCenterSpectrum(
            representative_count=self.count,
            frequency_denominator=self.count,
            frequency_scope=frequency_scope,
            bond_union=bonds,
            bond_intersection=tuple(
                pair for pair in bonds if self.bond_counts[pair] == self.count
            ),
            bond_change_counts=tuple(
                (left, right, self.bond_counts[(left, right)]) for left, right in bonds
            ),
            atom_union=atoms,
            atom_intersection=tuple(
                atom for atom in atoms if self.atom_counts[atom] == self.count
            ),
            atom_change_counts=tuple((atom, self.atom_counts[atom]) for atom in atoms),
            unary_properties=tuple(self.properties),
            representative_stream_sha256=self.stream_digest.hexdigest(),
        )


def _reference_free_slap_seed(lgp, binary):
    started = time.perf_counter()
    try:
        matcher = GraphMatcher(
            binary=binary,
            max_lap_fingerprints=1_000,
            cache_label_blocks=True,
            deterministic_labels=True,
        )
        matcher.get_maps(lgp)
        if not matcher.results:
            raise RuntimeError("SLAP produced no complete candidate")
        from .exact.enumerate import complete_mapping

        mapping = complete_mapping(
            lgp,
            recover_mapping(matcher.results[0]["lgp"]),
            binary=binary,
        )
        if sorted(mapping) != list(range(len(mapping))):
            raise RuntimeError("SLAP candidate is not a complete permutation")
        return mapping, {
            "method": "reference_free_slap",
            "available": True,
            "elapsed_seconds": time.perf_counter() - started,
            "cost": chemical_distance(lgp, mapping, binary=binary),
        }
    except Exception as error:
        return None, {
            "method": "reference_free_slap",
            "available": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _run_blind_search(lgp, target, config, observer, symmetry_properties):
    """Run the global search without accepting any reference-map argument."""
    if config.use_slap_seed:
        initial_mapping, seed_statistics = _reference_free_slap_seed(
            lgp,
            config.binary,
        )
    else:
        initial_mapping = None
        seed_statistics = {
            "method": "disabled",
            "available": False,
            "elapsed_seconds": 0.0,
        }
    result = enumerate_hybrid_distance_mappings(
        lgp,
        CD=target,
        binary=config.binary,
        backend=config.backend,
        max_edit_support_pairs=config.max_edit_support_pairs,
        max_bijections=config.max_bijections,
        tolerance=config.tolerance,
        time_limit_seconds=config.time_limit_seconds,
        certify=False,
        symmetry_pruning=config.symmetry_pruning,
        max_symmetry_automorphisms=config.max_symmetry_automorphisms,
        symmetry_timeout_seconds=config.symmetry_timeout_seconds,
        symmetry_max_search_nodes=config.symmetry_max_search_nodes,
        expand_symmetry=False,
        initial_mapping=initial_mapping,
        max_mappings=config.max_mappings,
        assignment_lower_bound=True,
        assignment_upper_bound=True,
        atom_profile_pruning=True,
        seed_center_order=True,
        symmetry_node_properties=symmetry_properties,
        fixed_mapping=None,
        collect_mappings=False,
        mapping_callback=observer.observe,
        compute_minimum_cost=target == "minimal",
    )
    statistics = dict(result.backend_statistics or {})
    statistics["seed"] = seed_statistics
    result.backend_statistics = statistics
    return result


def analyze_reference_blinded_global_shell(
    lgp,
    reference_mapping,
    *,
    target_mode: ShellTargetMode,
    config: GlobalShellConfig | None = None,
) -> GlobalShellAnalysisResult:
    """Search globally, then reveal a held-out reference for evaluation.

    ``target_mode='reference_cd'`` permits the scalar distance of the held-out
    mapping to define the shell.  The mapping itself is never passed to the
    search.  ``target_mode='minimal'`` hides both the reference mapping and its
    distance until the global minimum and optimizer shell have been explored.
    """
    if target_mode not in {"reference_cd", "minimal"}:
        raise ValueError("target_mode must be 'reference_cd' or 'minimal'")
    config = GlobalShellConfig() if config is None else config
    if not isinstance(config, GlobalShellConfig):
        raise TypeError("config must be GlobalShellConfig or None")

    reactant, reactant_elements = _adjacency_and_elements(lgp[0], config.binary)
    product, product_elements = _adjacency_and_elements(lgp[1], config.binary)
    reference = tuple(int(image) for image in reference_mapping)
    if len(reference) != len(reactant_elements) or sorted(reference) != list(
        range(len(product_elements))
    ):
        raise ValueError("reference_mapping must be a complete permutation")
    if any(
        reactant_elements[atom] != product_elements[image]
        for atom, image in enumerate(reference)
    ):
        raise ValueError("reference_mapping must preserve atom types")

    reference_cd = chemical_distance(lgp, reference, binary=config.binary)
    target = reference_cd if target_mode == "reference_cd" else "minimal"
    properties = _property_vectors(lgp, config.reaction_center_properties)
    symmetry_properties = tuple(
        name
        for name in config.symmetry_node_properties
        if name in _property_vectors(lgp, (name,))
    )
    observer = _BlindShellObserver(
        reactant,
        product,
        reactant_elements,
        properties,
        config,
    )
    result = _run_blind_search(lgp, target, config, observer, symmetry_properties)
    if observer.count != result.selected_mapping_count:
        raise RuntimeError("streamed mapping count disagrees with search result")

    mapping_observed = _mapping_sha256(reference) in observer.mapping_hashes
    class_observed = (
        _transport_sha256(product, properties, reference) in observer.transport_hashes
    )
    minimum = result.minimum_cost
    gap = None if minimum is None else reference_cd - float(minimum)
    is_minimal = bool(
        result.complete
        and minimum is not None
        and math.isclose(
            reference_cd,
            float(minimum),
            abs_tol=config.tolerance,
            rel_tol=0.0,
        )
        and class_observed
    )
    frequency_scope = (
        "verified_product_subgroup_orbit_representatives"
        if config.symmetry_pruning
        else "labeled_mappings"
    )
    return GlobalShellAnalysisResult(
        target_mode=target_mode,
        target=result.target,
        binary=config.binary,
        status=result.status,
        complete=result.complete,
        truncation_reason=result.truncation_reason,
        scope=result.scope,
        minimum_cost=result.minimum_cost,
        reference_cd=reference_cd,
        reference_gap_from_minimum=gap,
        reference_mapping_observed=mapping_observed,
        reference_class_observed=class_observed,
        shell_complete_and_reference_class_observed=(
            result.complete and class_observed
        ),
        reference_is_global_minimum_proven=is_minimal,
        total_bijections=result.total_bijections,
        representative_solution_count=result.selected_mapping_count,
        mapping_hartley_entropy_nats=(
            math.log(result.selected_mapping_count)
            if result.complete
            and result.symmetry_quotient_complete
            and result.selected_mapping_count
            else None
        ),
        labeled_solution_count=result.selected_labeled_mapping_count,
        symmetry_group_order=result.symmetry_group_order,
        symmetry_quotient_complete=result.symmetry_quotient_complete,
        visited_nodes=result.visited_nodes,
        visited_leaves=result.visited_leaves,
        distance_pruned_branches=result.pruned_branches,
        lower_bound_pruned_branches=result.lower_bound_pruned_branches,
        upper_bound_pruned_branches=result.upper_bound_pruned_branches,
        symmetry_pruned_branches=result.symmetry_pruned_branches,
        elapsed_seconds=result.elapsed_seconds,
        backend=result.backend,
        backend_statistics=result.backend_statistics,
        reaction_center=observer.spectrum(frequency_scope),
        structure=observer.structure.finalize(
            shell_complete=result.complete,
            reference_mapping=reference,
            class_count_scope=frequency_scope,
        ),
    )


__all__ = [
    "GlobalShellAnalysisResult",
    "GlobalShellConfig",
    "ReactionCenterSpectrum",
    "ShellTargetMode",
    "analyze_reference_blinded_global_shell",
]
