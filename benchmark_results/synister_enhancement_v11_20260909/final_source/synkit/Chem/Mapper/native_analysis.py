"""Reference-blinded exact shells using the optional native orbit backend.

The retained-mapping budget counts worker-local double-orbit representatives.
Weighted reported counts retain the original product-orbit and labeled units.
"""

import hashlib
import math
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .analysis import (
    GlobalShellAnalysisResult,
    GlobalShellConfig,
    _BlindShellObserver,
    _mapping_key,
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
    target_mode="reference_cd",
    workers=8,
    scheduler="frontier",
    slice_nodes=8192,
    deadline=None,
    config: GlobalShellConfig | None = None,
):
    """Prove and enumerate a minimum or numeric shell with reference-blinded search.

    Compilation and backend selection are explicit. The original Python API
    and its mapping-budget unit remain unchanged. This backend needs symmetric
    half-integer inputs and complete side-group proofs; unsupported inputs fail
    explicitly instead of silently changing the requested protocol.
    """
    if target_mode not in {"minimal", "reference_cd"}:
        raise ValueError("target_mode must be minimal or reference_cd")
    analysis_started = time.perf_counter()
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
    reference_cd = chemical_distance(lgp, reference, binary=False)
    target = reference_cd
    seed, seed_stats = (
        _reference_free_slap_seed(lgp, False, repair=True)
        if config.use_slap_seed
        else (None, {"method": "disabled"})
    )
    if deadline is None and scheduler == "frontier" and config.time_limit_seconds is not None:
        deadline = analysis_started + config.time_limit_seconds
    minimum = None
    proof = None
    if target_mode == "minimal":
        if scheduler != "frontier":
            raise ValueError("minimum mode requires the shared frontier deadline")
        from .exact.distance import enumerate_distance_mappings

        proof = enumerate_distance_mappings(
            lgp, CD="minimal", binary=False, max_bijections=None,
            tolerance=config.tolerance,
            time_limit_seconds=None if deadline is None else max(0, deadline - time.perf_counter()),
            symmetry_pruning=True,
            symmetry_node_properties=symmetry_properties,
            initial_mapping=seed, max_mappings=None,
            collect_mappings=True, _optimization_only=True,
        )
        if not proof.complete or proof.minimum_cost is None:
            raise TimeoutError("minimum proof incomplete; no provisional shell enumerated")
        minimum = target = proof.minimum_cost
        if proof.mappings:
            seed = proof.mappings[0]
    # The reference itself is first queried after the complete/partial search.
    search = frontier_orbit_search if scheduler == "frontier" else parallel_orbit_search
    options = {"slice_nodes": slice_nodes} if scheduler == "frontier" else {}
    if deadline is not None:
        if scheduler != "frontier" or not math.isfinite(deadline):
            raise ValueError(
                "absolute deadlines require frontier and finite monotonic time"
            )
        options["absolute_deadline"] = deadline
    search_started = time.perf_counter()
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
    search_finished = time.perf_counter()
    reference_observed = accumulator.contains_reference(reference)
    reference_finished = time.perf_counter()
    scope = "verified_product_subgroup_orbit_representatives"
    structure = observer.structure.finalize(
        shell_complete=result["complete"],
        reference_mapping=reference,
        class_count_scope=scope,
    )
    formatting_finished = time.perf_counter()
    counts = observer.count
    complete = result["complete"]
    node_counts = [item.get("visited_nodes") for item in result["shards"]]
    leaf_counts = [item.get("visited_leaves") for item in result["shards"]]
    pruned = [item.get("pruned") for item in result["shards"]]
    statistics = {
        "phase_timings": {
            "preparation_seconds": search_started - analysis_started,
            "search_with_setup_seconds": search_finished - search_started,
            "reference_check_seconds": reference_finished - search_finished,
            "structure_finalize_seconds": formatting_finished - reference_finished,
        },
        "native_orbits": {**result, "library_sha256": binary_sha256},
        "seed": seed_stats,
        "minimum_proof": None if proof is None else {
            "complete": proof.complete, "minimum_cost": minimum,
            "elapsed_seconds": proof.elapsed_seconds, "visited_nodes": proof.visited_nodes,
        },
        "stream_digest_scope": "merged double-orbit records: length-prefixed weight and representative mapping hash",
        "counter_scope": "native combined distance/assignment pruning reported as lower_bound; upper_bound and symmetry counters not separately tracked",
        "mapping_limit_scope": "retained_worker_double_orbit_representatives",
    }
    return GlobalShellAnalysisResult(
        target_mode=target_mode,
        target="minimal" if target_mode == "minimal" else target,
        binary=False,
        status="complete" if complete else "timeout",
        complete=complete,
        truncation_reason=None if complete else ",".join(result["reasons"]),
        scope="exact_weighted_product_automorphism_orbits",
        minimum_cost=minimum,
        reference_cd=reference_cd,
        reference_gap_from_minimum=None if minimum is None else reference_cd - minimum,
        reference_mapping_observed=_mapping_key(reference)
        in observer.mapping_hashes,
        reference_class_observed=reference_observed,
        shell_complete_and_reference_class_observed=complete and reference_observed,
        reference_is_global_minimum_proven=minimum is not None and abs(reference_cd - minimum) <= config.tolerance,
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
