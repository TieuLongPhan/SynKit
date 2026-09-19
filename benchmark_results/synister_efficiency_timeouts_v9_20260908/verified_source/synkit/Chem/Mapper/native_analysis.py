"""Reference-blinded exact shells using the optional native orbit backend.

The retained-mapping budget counts worker-local double-orbit representatives.
Weighted reported counts retain the original product-orbit and labeled units.
"""

import hashlib
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .analysis import (
    GlobalShellAnalysisResult,
    GlobalShellConfig,
    _BlindShellObserver,
    _mapping_sha256,
    _property_vectors,
    _reference_free_slap_seed,
)
from .exact.native_frontier import frontier_orbit_search
from .exact.native_parallel import parallel_orbit_search
from .slap.lap import _adjacency_and_elements, chemical_distance


def analyze_reference_blinded_native_shell(
    lgp,
    reference_mapping,
    *,
    library_path,
    workers=8,
    scheduler="frontier",
    slice_nodes=8192,
    config: GlobalShellConfig | None = None,
):
    """Analyze one numeric reference-CD shell without passing its reference to search.

    Compilation and backend selection are explicit. The original Python API
    and its mapping-budget unit remain unchanged. This backend needs symmetric
    half-integer inputs and complete side-group proofs; unsupported inputs fail
    explicitly instead of silently changing the requested protocol.
    """
    config = GlobalShellConfig(time_limit_seconds=60) if config is None else config
    if not isinstance(config, GlobalShellConfig):
        raise TypeError("config must be GlobalShellConfig")
    if config.binary or config.tolerance >= 0.25 or not config.symmetry_pruning:
        raise ValueError(
            "native orbit analysis requires weighted exact symmetry analysis"
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if config.max_bijections is not None:
        raise ValueError("native orbit analysis does not support max_bijections")
    if (
        config.max_symmetry_automorphisms,
        config.symmetry_timeout_seconds,
        config.symmetry_max_search_nodes,
    ) != (256, 0.25, 10000):
        raise ValueError(
            "native orbit analysis currently requires default side-group proof budgets"
        )
    if scheduler not in {"static", "frontier"}:
        raise ValueError("scheduler must be static or frontier")
    path = Path(library_path).resolve()
    binary_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    reactant, elements = _adjacency_and_elements(lgp[0], False)
    product, product_elements = _adjacency_and_elements(lgp[1], False)
    reference = tuple(int(image) for image in reference_mapping)
    if sorted(reference) != list(range(len(elements))) or any(
        elements[i] != product_elements[j] for i, j in enumerate(reference)
    ):
        raise ValueError("reference_mapping must be an atom-compatible permutation")
    properties = _property_vectors(lgp, config.reaction_center_properties)
    symmetry_properties = tuple(
        name
        for name in config.symmetry_node_properties
        if name in _property_vectors(lgp, (name,))
    )
    if set(symmetry_properties) != set(properties):
        raise ValueError(
            "native aggregation requires symmetry to preserve every reported property"
        )
    effective = replace(
        config,
        symmetry_node_properties=symmetry_properties,
        reaction_center_properties=tuple(properties),
    )
    observer = _BlindShellObserver(reactant, product, elements, properties, effective)
    target = chemical_distance(lgp, reference, binary=False)
    seed, seed_stats = (
        _reference_free_slap_seed(lgp, False, repair=True)
        if config.use_slap_seed
        else (None, {"method": "disabled"})
    )
    # The reference itself is first queried after the complete/partial search.
    search = frontier_orbit_search if scheduler == "frontier" else parallel_orbit_search
    options = {"slice_nodes": slice_nodes} if scheduler == "frontier" else {}
    result, accumulator = search(
        lgp,
        target,
        effective,
        seed,
        observer,
        library_path=path,
        workers=workers,
        **options,
    )
    reference_observed = accumulator.contains_reference(reference)
    scope = "verified_product_subgroup_orbit_representatives"
    structure = observer.structure.finalize(
        shell_complete=result["complete"],
        reference_mapping=reference,
        class_count_scope=scope,
    )
    counts = observer.count
    complete = result["complete"]
    node_counts = [item.get("visited_nodes") for item in result["shards"]]
    leaf_counts = [item.get("visited_leaves") for item in result["shards"]]
    pruned = [item.get("pruned") for item in result["shards"]]
    statistics = {
        "native_orbits": {**result, "library_sha256": binary_sha256},
        "seed": seed_stats,
        "stream_digest_scope": "merged double-orbit records: length-prefixed weight and representative mapping hash",
        "counter_scope": "native combined distance/assignment pruning reported as lower_bound; upper_bound and symmetry counters not separately tracked",
        "mapping_limit_scope": "retained_worker_double_orbit_representatives",
    }
    return GlobalShellAnalysisResult(
        target_mode="reference_cd",
        target=target,
        binary=False,
        status="complete" if complete else "timeout",
        complete=complete,
        truncation_reason=None if complete else ",".join(result["reasons"]),
        scope="exact_weighted_product_automorphism_orbits",
        minimum_cost=None,
        reference_cd=target,
        reference_gap_from_minimum=None,
        reference_mapping_observed=_mapping_sha256(reference)
        in observer.mapping_hashes,
        reference_class_observed=reference_observed,
        shell_complete_and_reference_class_observed=complete and reference_observed,
        reference_is_global_minimum_proven=False,
        total_bijections=math.prod(
            math.factorial(n) for n in Counter(elements).values()
        ),
        representative_solution_count=counts,
        mapping_hartley_entropy_nats=math.log(counts) if complete and counts else None,
        labeled_solution_count=counts * accumulator.product_order,
        symmetry_group_order=accumulator.product_order,
        symmetry_quotient_complete=True,
        visited_nodes=sum(x for x in node_counts if x is not None),
        visited_leaves=sum(x for x in leaf_counts if x is not None),
        distance_pruned_branches=0,
        lower_bound_pruned_branches=sum(x for x in pruned if x is not None),
        upper_bound_pruned_branches=0,
        symmetry_pruned_branches=0,
        elapsed_seconds=result["elapsed_seconds"],
        backend="native_two_sided_orbit_aggregation",
        backend_statistics=statistics,
        reaction_center=observer.spectrum(scope),
        structure=structure,
    )
