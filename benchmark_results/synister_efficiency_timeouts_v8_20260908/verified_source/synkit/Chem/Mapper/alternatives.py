"""Candidate-complete alternative-ITS generation from exact AAM shells.

The term ``alternative`` is deliberate: a non-reference ITS is a controlled
structural alternative under the declared chemical-distance objective, not a
claim that the mapping is mechanistically false.  A trusted reference may be
used as an incumbent or traversal seed, but it never fixes assignments or
removes candidates from the global shell.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from .analysis import GlobalShellConfig, _property_vectors, _reference_free_slap_seed
from .chem.blind import blinded_mapped_reaction_problem
from .exact.distance_records import DistanceTarget, normalize_distance_target
from .exact.hybrid import enumerate_hybrid_distance_mappings
from .slap.lap import _adjacency_and_elements, chemical_distance
from .spectrum import exact_its_and_template_codes

AlternativeTarget = Literal["reference", "minimal"] | Real
SeedMode = Literal["reference", "slap", "none"]


def _code_identifier(code) -> str:
    return hashlib.sha256(repr(code).encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class ExactITSAlternative:
    """One deterministic AAM representative of a non-reference ITS class."""

    its_class_id: str
    template_class_id: str
    representative_mapping: tuple[int, ...]
    distance: float
    representative_mapping_count: int
    labeled_mapping_count: int | None

    def atom_map_correspondence(self, reactant_atom_maps, product_atom_maps):
        """Express the representative using endpoint atom-map identifiers."""
        if len(reactant_atom_maps) != len(self.representative_mapping):
            raise ValueError("reactant atom-map inventory has the wrong size")
        if len(product_atom_maps) != len(self.representative_mapping):
            raise ValueError("product atom-map inventory has the wrong size")
        return tuple(
            (
                int(reactant_atom_maps[atom]),
                int(product_atom_maps[image]),
            )
            for atom, image in enumerate(self.representative_mapping)
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible alternative record."""
        return {
            "its_class_id": self.its_class_id,
            "template_class_id": self.template_class_id,
            "representative_mapping": list(self.representative_mapping),
            "distance": self.distance,
            "representative_mapping_count": self.representative_mapping_count,
            "labeled_mapping_count": self.labeled_mapping_count,
        }


@dataclass(frozen=True)
class ExactITSAlternativeResult:
    """Complete or explicitly incomplete alternative-ITS shell analysis."""

    target: DistanceTarget
    reference_cd: float
    seed_mode: SeedMode
    seed_applied_to_backend: bool
    seed_statistics: dict[str, object]
    status: str
    complete: bool
    shell_complete: bool
    classification_complete: bool
    incomplete_reason: str | None
    scope: str
    shell_representative_mapping_count: int
    shell_labeled_mapping_count: int | None
    shell_its_class_count: int
    reference_its_class_id: str | None
    reference_its_class_observed: bool | None
    alternative_its_class_count: int | None
    alternatives: tuple[ExactITSAlternative, ...]
    minimum_cost: float | None
    elapsed_seconds: float
    visited_nodes: int
    backend: str
    backend_statistics: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible result record."""
        return {
            "target": self.target,
            "reference_cd": self.reference_cd,
            "seed_mode": self.seed_mode,
            "seed_applied_to_backend": self.seed_applied_to_backend,
            "seed_statistics": self.seed_statistics,
            "status": self.status,
            "complete": self.complete,
            "shell_complete": self.shell_complete,
            "classification_complete": self.classification_complete,
            "incomplete_reason": self.incomplete_reason,
            "scope": self.scope,
            "shell_representative_mapping_count": (
                self.shell_representative_mapping_count
            ),
            "shell_labeled_mapping_count": self.shell_labeled_mapping_count,
            "shell_its_class_count": self.shell_its_class_count,
            "reference_its_class_id": self.reference_its_class_id,
            "reference_its_class_observed": self.reference_its_class_observed,
            "alternative_its_class_count": self.alternative_its_class_count,
            "alternatives": [item.as_dict() for item in self.alternatives],
            "minimum_cost": self.minimum_cost,
            "elapsed_seconds": self.elapsed_seconds,
            "visited_nodes": self.visited_nodes,
            "backend": self.backend,
            "backend_statistics": self.backend_statistics,
        }


@dataclass(frozen=True)
class MappedReactionITSAlternativeResult:
    """Alternative ITS classes in the original reaction atom-map coordinates."""

    reaction_sha256: str
    heavy_only: bool
    blind_seed: str
    reactant_atom_maps: tuple[int, ...]
    product_atom_maps: tuple[int, ...]
    shell: ExactITSAlternativeResult

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record with AAM correspondences."""
        payload = self.shell.as_dict()
        for record, alternative in zip(
            payload["alternatives"], self.shell.alternatives
        ):
            record["atom_map_correspondence"] = [
                list(pair)
                for pair in alternative.atom_map_correspondence(
                    self.reactant_atom_maps,
                    self.product_atom_maps,
                )
            ]
        return {
            "reaction_sha256": self.reaction_sha256,
            "heavy_only": self.heavy_only,
            "blind_seed": self.blind_seed,
            "shell": payload,
        }


class _ITSClassCollector:
    def __init__(self, reactant, product, elements, properties, config):
        self.reactant = reactant
        self.product = product
        self.elements = elements
        self.properties = properties
        self.config = config
        self.classes = {}
        self.incomplete_reason = None

    def codes(self, mapping):
        return exact_its_and_template_codes(
            self.reactant,
            self.product,
            self.elements,
            self.properties,
            mapping,
            template_radius=self.config.template_radius,
            tolerance=self.config.tolerance,
            timeout_seconds=self.config.structure_timeout_seconds,
            max_search_nodes=self.config.structure_max_search_nodes,
        )

    def observe(self, mapping, cost):
        if self.incomplete_reason is not None:
            return
        its_code, template_code, reason = self.codes(mapping)
        if reason is not None:
            self.incomplete_reason = reason
            return
        normalized = tuple(int(image) for image in mapping)
        current = self.classes.get(its_code)
        if current is None:
            self.classes[its_code] = [1, normalized, float(cost), template_code]
            return
        current[0] += 1
        if template_code != current[3]:
            self.incomplete_reason = "one exact ITS produced multiple template codes"
            return
        if normalized < current[1]:
            current[1] = normalized


def _normalize_alternative_target(CD, reference_cd):
    if isinstance(CD, str) and CD.lower() == "reference":
        return float(reference_cd)
    return normalize_distance_target(CD)


def _validate_reference(lgp, reference_mapping, binary):
    reactant, reactant_elements = _adjacency_and_elements(lgp[0], binary)
    product, product_elements = _adjacency_and_elements(lgp[1], binary)
    reference = tuple(int(image) for image in reference_mapping)
    if reactant.shape != product.shape:
        raise ValueError("reactant and product must have the same number of atoms")
    if len(reference) != len(reactant_elements) or sorted(reference) != list(
        range(len(product_elements))
    ):
        raise ValueError("reference_mapping must be a complete permutation")
    if any(
        reactant_elements[atom] != product_elements[image]
        for atom, image in enumerate(reference)
    ):
        raise ValueError("reference_mapping must preserve atom types")
    return reactant, product, reactant_elements, reference


def _select_seed(lgp, reference, config, seed_mode):
    if seed_mode == "reference":
        return list(reference), {
            "method": "reference_ordering_and_incumbent",
            "available": True,
            "cost": chemical_distance(lgp, reference, binary=config.binary),
            "candidate_space_restricted": False,
        }
    if seed_mode == "slap":
        return _reference_free_slap_seed(lgp, config.binary)
    return None, {
        "method": "disabled",
        "available": False,
        "candidate_space_restricted": False,
    }


def enumerate_exact_its_alternatives(
    lgp,
    reference_mapping,
    *,
    CD: AlternativeTarget = "reference",
    seed_mode: SeedMode = "reference",
    config: GlobalShellConfig | None = None,
) -> ExactITSAlternativeResult:
    """Enumerate one AAM per exact non-reference ITS class in a global shell.

    ``CD='reference'`` enumerates the shell at the reference map's distance;
    a numeric value may be lower or higher, and ``CD='minimal'`` first proves
    the global minimum.  ``seed_mode='reference'`` may improve the incumbent
    and traversal order but never fixes atoms or prunes by reference identity.
    Consequently every complete result has the same alternative classes as a
    seed-free run under the same declared CD and atom-compatibility policy.
    """
    if seed_mode not in {"reference", "slap", "none"}:
        raise ValueError("seed_mode must be 'reference', 'slap', or 'none'")
    config = GlobalShellConfig() if config is None else config
    if not isinstance(config, GlobalShellConfig):
        raise TypeError("config must be GlobalShellConfig or None")
    if not config.structure_analysis:
        raise ValueError("structure_analysis must be enabled for ITS alternatives")
    reactant, product, elements, reference = _validate_reference(
        lgp,
        reference_mapping,
        config.binary,
    )
    reference_cd = chemical_distance(lgp, reference, binary=config.binary)
    target = _normalize_alternative_target(CD, reference_cd)
    property_names = tuple(
        dict.fromkeys(
            (*config.reaction_center_properties, *config.symmetry_node_properties)
        )
    )
    properties = _property_vectors(lgp, property_names)
    # Every unary attribute used to distinguish ITS classes must also refine
    # product automorphisms; otherwise symmetry pruning could merge classes.
    symmetry_properties = tuple(properties)
    collector = _ITSClassCollector(
        reactant,
        product,
        elements,
        properties,
        config,
    )
    initial_mapping, seed_statistics = _select_seed(
        lgp,
        reference,
        config,
        seed_mode,
    )
    result = enumerate_hybrid_distance_mappings(
        lgp,
        CD=target,
        binary=config.binary,
        backend=config.backend,
        max_edit_support_pairs=config.max_edit_support_pairs,
        max_bijections=config.max_bijections,
        tolerance=config.tolerance,
        time_limit_seconds=config.time_limit_seconds,
        max_mappings=config.max_mappings,
        collect_mappings=False,
        mapping_callback=collector.observe,
        compute_minimum_cost=target == "minimal",
        symmetry_pruning=config.symmetry_pruning,
        max_symmetry_automorphisms=config.max_symmetry_automorphisms,
        symmetry_timeout_seconds=config.symmetry_timeout_seconds,
        symmetry_max_search_nodes=config.symmetry_max_search_nodes,
        expand_symmetry=False,
        initial_mapping=initial_mapping,
        assignment_lower_bound=True,
        assignment_upper_bound=True,
        atom_profile_pruning=True,
        seed_center_order=True,
        symmetry_node_properties=symmetry_properties,
        fixed_mapping=None,
    )
    reference_its, _, reference_reason = collector.codes(reference)
    reason = collector.incomplete_reason or reference_reason
    classification_complete = reason is None
    complete = result.complete and classification_complete
    if not result.complete:
        incomplete_reason = result.truncation_reason or "shell_incomplete"
        status = result.status
    elif not classification_complete:
        incomplete_reason = reason
        status = "classification_incomplete"
    else:
        incomplete_reason = None
        status = result.status

    reference_id = None if reference_its is None else _code_identifier(reference_its)
    reference_observed = (
        None if reference_its is None else reference_its in collector.classes
    )
    group_order = (
        result.symmetry_group_order if result.symmetry_quotient_complete else None
    )
    alternatives = []
    if reference_its is not None:
        for its_code, values in collector.classes.items():
            if its_code == reference_its:
                continue
            quotient_count, mapping, distance, template_code = values
            alternatives.append(
                ExactITSAlternative(
                    its_class_id=_code_identifier(its_code),
                    template_class_id=_code_identifier(template_code),
                    representative_mapping=mapping,
                    distance=distance,
                    representative_mapping_count=quotient_count,
                    labeled_mapping_count=(
                        None
                        if group_order is None
                        else int(quotient_count) * int(group_order)
                    ),
                )
            )
    alternatives.sort(key=lambda item: item.its_class_id)
    statistics = dict(result.backend_statistics or {})
    statistics["seed"] = seed_statistics
    return ExactITSAlternativeResult(
        target=result.target,
        reference_cd=float(reference_cd),
        seed_mode=seed_mode,
        seed_applied_to_backend=(
            seed_mode != "none" and result.backend == "assignment_branch_and_bound"
        ),
        seed_statistics=seed_statistics,
        status=status,
        complete=complete,
        shell_complete=result.complete,
        classification_complete=classification_complete,
        incomplete_reason=incomplete_reason,
        scope=result.scope,
        shell_representative_mapping_count=result.selected_mapping_count,
        shell_labeled_mapping_count=result.selected_labeled_mapping_count,
        shell_its_class_count=len(collector.classes),
        reference_its_class_id=reference_id,
        reference_its_class_observed=reference_observed,
        alternative_its_class_count=(
            len(alternatives) if classification_complete else None
        ),
        alternatives=tuple(alternatives),
        minimum_cost=result.minimum_cost,
        elapsed_seconds=result.elapsed_seconds,
        visited_nodes=result.visited_nodes,
        backend=result.backend,
        backend_statistics=statistics,
    )


def enumerate_mapped_reaction_its_alternatives(
    reaction,
    *,
    CD: AlternativeTarget = "reference",
    seed_mode: SeedMode = "reference",
    config: GlobalShellConfig | None = None,
    heavy_only: bool = True,
    blind_seed: str = "synister-alternatives-v1",
) -> MappedReactionITSAlternativeResult:
    """Generate exact alternative AAM classes from mapped reaction SMILES."""
    problem = blinded_mapped_reaction_problem(
        reaction,
        heavy_only=heavy_only,
        blind_seed=blind_seed,
    )
    shell = enumerate_exact_its_alternatives(
        problem.lgp,
        problem.reference_mapping,
        CD=CD,
        seed_mode=seed_mode,
        config=config,
    )
    return MappedReactionITSAlternativeResult(
        reaction_sha256=problem.reaction_sha256,
        heavy_only=problem.heavy_only,
        blind_seed=problem.blind_seed,
        reactant_atom_maps=problem.reactant_atom_maps,
        product_atom_maps=problem.product_atom_maps,
        shell=shell,
    )


__all__ = [
    "AlternativeTarget",
    "ExactITSAlternative",
    "ExactITSAlternativeResult",
    "MappedReactionITSAlternativeResult",
    "SeedMode",
    "enumerate_exact_its_alternatives",
    "enumerate_mapped_reaction_its_alternatives",
]
